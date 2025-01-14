/*
 * Copyright (C) 2019 - 2020 by the emerging fields initiative 'Novel Biopolymer
 * Hydrogels for Understanding Complex Soft Tissue Biomechanics' of the FAU
 *
 * This file is part of the EFI library.
 *
 * The EFI library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * Author: Stefan Kaessmair
 */

// efi header
#include <efi/constitutive/maxwell_element.h>
#include <efi/constitutive/constitutive_factory.h>
#include <efi/factory/registry.h>
#include <efi/base/factory_tools.h>
#include <efi/base/extractor.h>
#include <efi/lac/tensor_shape.h>


namespace efi {


template <int dim>
MaxwellElement<dim>::
MaxwellElement(const std::string &subsection_name,
               const std::string &unprocessed_input)
:
    ConstitutiveBase<dim>(subsection_name,unprocessed_input),
    eta_d(0),
    eta_v(0)
{
    using namespace dealii;

    unsigned int count = 0;

    FactoryTools::action_type create_contsitutive
    = [&] (const FactoryTools::Specifications &specs,
           const std::string                  &unprocessed_input) -> void
    {
        ++count;
        this->spring.reset (
                ConstitutiveFactory<dim>::create (
                        this->get_section_path(), specs, unprocessed_input));
    };

    // Put all actions into a map.
    std::map<std::string,FactoryTools::action_type> actions;
    actions[ConstitutiveFactory<dim>::keyword()] = create_contsitutive;

    std::istringstream unprocessed_input_stream (unprocessed_input);
    // Apply the actions to the unprocessed input.
    // The only thing we expect to happen is that
    // spring is going to be initialized with a
    // constitutive model.
    FactoryTools::apply (actions, unprocessed_input_stream);

    Assert(count == 1, ExcMessage("Constitutive model is ambiguous."
                                  "In your parameter file, in the section "
                                  "constitutive@[type=maxwell_element,...], "
                                  "there should only exist one subsection "
                                  "constitutive@[type=...] which describes "
                                  "the spring of the Maxwell element."));

    efilog(Verbosity::verbose) << "New MaxwellElement constitutive model created ("
                               << subsection_name
                               << ")."<< std::endl;
}



template <int dim>
void
MaxwellElement<dim>::
evaluate (ScratchData<dim> &scratch_data) const
{
    using namespace dealii;
    using namespace dealii::Physics::Elasticity;

    using ad_type = scalar_type;

    const unsigned int n_q_points = ScratchDataTools::n_quadrature_points (scratch_data);

    const auto &global_vector_name = Extractor<dim>::global_vector_name();

    // Create some aliases.
    auto &F   = ScratchDataTools::get_or_add_deformation_grads (scratch_data,global_vector_name,ad_type(0));
    auto &tau = ScratchDataTools::get_or_add_kirchoff_stresses (scratch_data,global_vector_name,ad_type(0));
    auto &cc  = ScratchDataTools::get_or_add_kirchoff_stress_tangents (scratch_data,global_vector_name,ad_type(0));

    // Get the displacement gradients.
    auto &Grad_u = ScratchDataTools::get_gradients (scratch_data,global_vector_name,Extractor<dim>::displacement(),ad_type(0));

    // Get access to the cell_data_storage to get
    // the history variables.
    auto &history_data     = ScratchDataTools::get_history_data (scratch_data);
    auto &tmp_history_data = ScratchDataTools::get_tmp_history_data (scratch_data);

    std::string str_extension = "";
    if (dynamic_cast<const dealii::FEFaceValues<dim>*>(
                &ScratchDataTools::get_current_fe_values (scratch_data))
                != nullptr)
    {
        str_extension = "_face" + Utilities::int_to_string (
                static_cast<const dealii::FEFaceValues<dim>&> (
                        ScratchDataTools::get_current_fe_values (scratch_data)
                ).get_face_index());
    }

    // Add the inelastic right Cauchy-Green deformation
    // tensor to the cell data storage. Since we have
    // to store this for every cell, only add the vari-
    // ables which are absolutely necessary like history
    // variables or the like.
    auto& C_i_inv = history_data.template get_or_add_object_with_name<
            std::vector<SymmetricTensor<2,dim,ad_type>>>(
                    this->section_path_str + "inv_inelastic_right_cauchy_green"
                    + str_extension, int(n_q_points), StandardTensors<dim>::I);

    auto& C_i_inv_tmp = tmp_history_data.template get_or_add_object_with_name<
            std::vector<SymmetricTensor<2,dim,ad_type>>>(
                    this->section_path_str + "inv_inelastic_right_cauchy_green"
                    + str_extension, int(n_q_points), StandardTensors<dim>::I);

    // Auxiliary arrays
    std::array<ad_type,dim> lambda_e_trial;       // elastic principal stretches (predictor)
    std::array<ad_type,dim> inv_lambda_e_trial;   // elastic principal stretches (predictor) inverted (1./lambda_e_trial)
    std::array<ad_type,dim> lambda_e;             // elastic principal stretches
    std::array<ad_type,dim> principal_tau;        // principal Kirchoff stresses

    // Auxiliary tensors (internal equilibrium)
    Tensor<1,dim,ad_type> R;                      // residual
    Tensor<2,dim,ad_type> K;                      // tangent
    Tensor<2,dim,ad_type> K_inv;                  // inverse tangent
    Tensor<1,dim,ad_type> depsilon_e;             // elastic principal stretch updates

    Tensor<1,dim,ad_type> epsilon_e_trial;        // logarithmic elastic principal stretches (predictor)
    Tensor<1,dim,ad_type> epsilon_e;              // logarithmic elastic principal stretches

    Tensor<1,dim,ad_type> stress_e_iso;           // principal elastic isochoric stresses d(psi^dev)/d(lambda[a]), a = 1,2,3
    Tensor<1,dim,ad_type> stress_e_vol;           // principal elastic volumetric stresses d(psi^vol)/d(lambda[a]), a = 1,2,3

    Tensor<1,dim,ad_type> dtr_tau_vol_dlambda_e;  // d(tr(tau^vol))/d(lambda_e[b])

    SymmetricTensor<2,dim,ad_type> tangent_e_iso; // principal elastic isochoric stress tangent
    SymmetricTensor<2,dim,ad_type> tangent_e_vol; // principal elastic volumetric stresses tangent

    SymmetricTensor<2,dim,ad_type> b_e_trial;     // left Cauchy-Green deformation tensor (predictor)

    // Auxiliary tensors (tangent)
    SymmetricTensor<2,dim,ad_type> dprincipal_tau_depsilon_e;  // d(tau[a])/d(epsilon_e[b])
    SymmetricTensor<2,dim,ad_type> A_alg;            // Algorithmic tangent

    std::array<std::array<Tensor<2,dim,ad_type>,dim>,dim> n_dyad_n; // N_dyad_N[a][b] = outer_product(N[a],N[b])

    std::array<std::array<Tensor<2,dim,ad_type>,dim>,dim> ln_dyad_ln_trial; // N_dyad_N[a][b] = outer_product(lambda_e_trial[a] *N[a],lambda_e_trial[a] *N[b])
    std::array<Tensor<1,dim,ad_type>,dim> ln_trial;

    double dt = ScratchDataTools::get_time_step_size (scratch_data);

    // indices
    unsigned int a,b;

    ad_type tr_tau_vol = 0; // trace of the volumentric Kirchoff stress
    ad_type iso_coeff  = 1./(2.*this->eta_d);
    ad_type vol_coeff = 0;
    // if (this->eta_v > 0)
    // {
    //     vol_coeff  = 1./(double(dim)*double(dim)*this->eta_v);
    // }
    // FIXME is doulbe(dim)*double(dim) correct? In the paper, it was 1/9

    // Loop over all quadrature points.
    for (unsigned int q = 0; q < n_q_points; ++q)
    {

        F[q] = StandardTensors<dim>::I + Grad_u[q];

        b_e_trial = symmetrize (F[q]*C_i_inv[q]*transpose(F[q]));

        // Now, compute the eigenvalues of the elastic
        // left Cauchy-Green deformation gradient within
        // the elastic predictor step (trial).
        auto eigen_b_e_trial = eigenvectors (b_e_trial);

        for (a = 0; a < dim; ++a)
        {
            lambda_e_trial[a]     = std::sqrt(eigen_b_e_trial[a].first);
            inv_lambda_e_trial[a] = 1./lambda_e_trial[a];
            epsilon_e_trial[a]    = std::log(lambda_e_trial[a]);
        }

        // Initialize our desired solution
        epsilon_e = epsilon_e_trial;
        lambda_e  = lambda_e_trial;

        ad_type norm_initial = 1;
        ad_type norm = 1;

        unsigned int step = 0;

        if (this->eta_d > 0 || this->eta_v > 0)
        {
            // local newton iteration
            do {
                // If it takes us more than 50 steps to achieve
                // an internal equilibrium, throw an exception.
                AssertThrow (step < 50, ExcMessage("Local Newton iteration failed, "
                        "i.e. no internal equilibrium obtained"));

                // Compute the principal stresses and their tangents
                // i.e. the first and second derivatives of the free
                // energy with respect to the principal stretches.
                this->spring->compute_principal_stresses (
                        lambda_e, stress_e_iso, stress_e_vol);
                this->spring->compute_principal_stress_tangents (
                        lambda_e, tangent_e_iso, tangent_e_vol);

                // Do some preprocessing.
                tr_tau_vol = 0;
                dtr_tau_vol_dlambda_e = stress_e_vol;
                for (a = 0; a < dim; ++a)
                {
                    tr_tau_vol += lambda_e[a]*stress_e_vol[a];
                    for (b = 0; b < dim; ++b)
                        dtr_tau_vol_dlambda_e[b] += lambda_e[a]*tangent_e_vol[a][b];
                }//a

                // Assemble the residual vector and the tangent.
                K = 0;
                for (a = 0; a < dim; ++a)
                {
                    R[a] = epsilon_e[a] - epsilon_e_trial[a] +
                            dt * (iso_coeff * lambda_e[a] * stress_e_iso[a]
                                + vol_coeff * tr_tau_vol);

                    K[a][a] += 1. + dt * lambda_e[a] * iso_coeff * stress_e_iso[a];

                    for (b = 0; b < dim; ++b)
                    {
                        K[a][b] += dt * lambda_e[b] * (
                                  iso_coeff * lambda_e[a] * tangent_e_iso[a][b]
                                + vol_coeff * dtr_tau_vol_dlambda_e[b]);
                    }//b
                }//a

                // Solve the linear system.
                K_inv = invert(K);
                depsilon_e = -K_inv*R;

                // Update the logarithmic principal stretches and
                // compute the principal stretches.
                for (a = 0; a < dim; ++a)
                {
                    epsilon_e[a] += depsilon_e[a];
                    lambda_e[a]   = std::exp (epsilon_e[a]);
                }

                // Compute the norm of the residual for the
                // convergence check.
                norm = R.norm();

                // Check if the norm is finite. Throw an error
                // if the assertion fails.
                AssertThrow (std::isfinite(norm), ExcMessage(
                        "Local Newton iteration failed, "
                        "i.e. no internal equilibrium obtained"));

                if (step == 0)
                    norm_initial = norm;

                ++step;
            } while ((norm > 1e-10) && (norm/norm_initial > 1e-10));

            // After the inelastic corrector step, recompute the left
            // Cauchy-Green deformation tensor...
            Tensor<2,dim,ad_type> b_e;
            for (a = 0; a < dim; ++a)
                b_e += outer_product((lambda_e[a] * lambda_e[a])
                        * eigen_b_e_trial[a].second,
                          eigen_b_e_trial[a].second);

            // ... and update the inelastic deformation.
            auto F_inv = invert(F[q]);
            C_i_inv_tmp[q] = symmetrize(F_inv * b_e * transpose(F_inv));

            // Compute the principal stresses and their tangents
            // i.e. the first and second derivatives of the free
            // energy with respect to the principal stretches.
            this->spring->compute_principal_stresses (
                    lambda_e, stress_e_iso, stress_e_vol);
            this->spring->compute_principal_stress_tangents (
                    lambda_e, tangent_e_iso, tangent_e_vol);

            for (a = 0; a < dim; ++a)
                ln_trial[a] = lambda_e_trial[a]*eigen_b_e_trial[a].second;

            // Compute the algorithmic tangent, first compute
            // d(tau[a])/d(epsilon_e[b]) ...
            for (a = 0; a < dim; ++a)
            {
                principal_tau[a] = lambda_e[a]*(stress_e_iso[a]+stress_e_vol[a]);

                for (b = a; b < dim; ++b)
                {
                    dprincipal_tau_depsilon_e[a][b] = lambda_e[b]*lambda_e[a]
                            * (tangent_e_iso[a][b] + tangent_e_vol[a][b]);

                    ln_dyad_ln_trial[a][b] = outer_product(ln_trial[a],ln_trial[b]);

                    if (b > a)
                        ln_dyad_ln_trial[b][a] = transpose(ln_dyad_ln_trial[a][b]);
                }//b
                dprincipal_tau_depsilon_e[a][a] += principal_tau[a];
            }//a

            // ... the the tangent can be obtained via:
            A_alg = symmetrize(dprincipal_tau_depsilon_e * K_inv);

            // d(S_tile[a])/d(lambda_e_trial[b])
            Tensor<2,dim,ad_type> dprincipal_S_dlambda_e_trial;

            for (a = 0; a < dim; ++a)
            {
                dprincipal_S_dlambda_e_trial[a][a] -= 2.* principal_tau[a] * inv_lambda_e_trial[a];

                for (b = 0; b < dim; ++b)
                    dprincipal_S_dlambda_e_trial[a][b] += A_alg[a][b] * inv_lambda_e_trial[b];

                dprincipal_S_dlambda_e_trial[a] *=  inv_lambda_e_trial[a] * inv_lambda_e_trial[a];
            }//a

            // Don't forget to reset the Kirchoff stress.
            tau[q] = 0;

            // Non-symmetric tangent. Only when all contributions are
            // gathered, the tangent is symmetric, i.e. we have to store
            // intermediate results in a general tensor object.
            Tensor<4,dim,ad_type> A_aux;

            // Auxiliary variable
            ad_type coefficient;

            for (a = 0; a < dim; ++a)
            {
                // Compute the Kirchoff stress tensor.
                tau[q] += symmetrize (outer_product(
                        lambda_e[a] * (stress_e_iso[a]+stress_e_vol[a])
                                       * eigen_b_e_trial[a].second,
                                         eigen_b_e_trial[a].second));

                for (b = 0; b < dim; ++b)
                {
                    A_aux += outer_product(dprincipal_S_dlambda_e_trial[a][b] * inv_lambda_e_trial[b] * ln_dyad_ln_trial[a][a],ln_dyad_ln_trial[b][b]);

                    // Only non-zero contributions
                    // if a and b are different.
                    if (a != b)
                    {
                        // Check if the eigenvalues are equal, if so,
                        // l'Hospital's rule is used in the else-
                        // statement. Otherwise, we would divide by
                        // zero.
                        if (std::fabs(lambda_e_trial[b]-lambda_e_trial[a]) > 1e-8)
                        {
                            coefficient = (principal_tau[b]* inv_lambda_e_trial[b]* inv_lambda_e_trial[b]
                                          -principal_tau[a]* inv_lambda_e_trial[a]* inv_lambda_e_trial[a]) /
                                    (lambda_e_trial[b]*lambda_e_trial[b] - lambda_e_trial[a]* lambda_e_trial[a]);
                        }
                        else
                        {
                            coefficient = 0.5 * inv_lambda_e_trial[b]
                                    * (dprincipal_S_dlambda_e_trial[b][b]-dprincipal_S_dlambda_e_trial[a][b]);
                        }

                        const auto tmp = coefficient * ln_dyad_ln_trial[a][b];

                        A_aux += (dealii::outer_product(tmp,ln_dyad_ln_trial[a][b]+ln_dyad_ln_trial[b][a]));
                    }//if
                }//b
            }//a

            // Assemble both contributions into the SymmetricTensor cc[q].
            // When doing so, the minor symmetries can be exploited. Note that
            // the tensor A_aux is symmetric when fully assembled, therefore using
            // dealii::symmetrize would be more expensive.
            for(unsigned int i = 0; i < dim; ++i)
                for(unsigned int j = i; j < dim; ++j)
                    for(unsigned int k = 0; k < dim; ++k)
                        for(unsigned int l = k; l< dim; ++l)
                            cc[q][i][j][k][l] = A_aux[i][j][k][l];
        }//if
        else
        {
            std::cout << "zero viscosity" << std::endl;
        }
    }//q
}


// Instantiation
template class MaxwellElement<2>;
template class MaxwellElement<3>;

// Registration
EFI_REGISTER_OBJECT (EFI_TEMPLATE_CLASS (MaxwellElement,2));
EFI_REGISTER_OBJECT (EFI_TEMPLATE_CLASS (MaxwellElement,3));

}// namespace efi

