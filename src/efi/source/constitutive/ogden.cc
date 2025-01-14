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

// stl headers
#include <functional>

// deal.II headers
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

// efi headers
#include <efi/constitutive/ogden.h>
#include <efi/base/extractor.h>
#include <efi/lac/tensor_shape.h>
#include <efi/factory/registry.h>


namespace efi {



//ResidualLinearization version
template<int dim>
void
Ogden<dim>::
evaluate (ScratchData<dim> &scratch_data) const
{
   using namespace dealii;
   using namespace dealii::Physics::Elasticity;

   using ad_type = scalar_type;

   // Get the number of quadrature points.
   const unsigned int n_q_points = ScratchDataTools::n_quadrature_points (scratch_data);

   const auto &global_vector_name = Extractor<dim>::global_vector_name();

   // Create some aliases.
   auto &F   = ScratchDataTools::get_or_add_deformation_grads        (scratch_data,global_vector_name,ad_type(0));
   auto &tau = ScratchDataTools::get_or_add_kirchoff_stresses        (scratch_data,global_vector_name,ad_type(0));
   auto &cc  = ScratchDataTools::get_or_add_kirchoff_stress_tangents (scratch_data,global_vector_name,ad_type(0));

   // Get the displacement gradients.
   auto &Grad_u = ScratchDataTools::get_gradients (scratch_data,global_vector_name,Extractor<dim>::displacement(),ad_type(0));

   // Array of eigenvalues and -vectors.
   std::array<std::pair<ad_type,Tensor<1,dim,ad_type>>,dim> eigen_b;

   // Auxiliary arrays
   std::array<ad_type,dim> lambda;                                   // principal stretches
   std::array<std::array<Tensor<2,dim,ad_type>,dim>,dim> ln_dyad_ln; // ln_dyad_ln[a][b] = outer_product(lambda[a]*n[a],lambda[b]*n[b])


   Tensor<1,dim,ad_type> principal_stresses_vol;  // principal isochoric stresses
   Tensor<1,dim,ad_type> principal_stresses_iso;  // principal volumetric stresses

   Tensor<1,dim,ad_type> principal_S;             // principal Kirchoff stresses

   SymmetricTensor<2,dim,ad_type> principal_stress_tangents_vol;  // principal isochoric stresses
   SymmetricTensor<2,dim,ad_type> principal_stress_tangents_iso;  // principal volumetric stresses

   // Since the principal stresses are derived from
   // a potential their tangent is symmetric.
   // 1/lambda[b]*(d(principal_S[a])/d(lambda[b]))
   SymmetricTensor<2,dim,ad_type> lambda_inv_dprincipal_S_dlambda;

   // Loop over the quadrature points.
   for (unsigned int q = 0; q < n_q_points; ++q)
   {
       F [q] = StandardTensors<dim>::I + Grad_u[q];

       // Compute the eigenvalues and -vectors of b = F*F^T.
       // By default ql_implicit_shifts algorithm is used.
       eigen_b = eigenvectors(symmetrize (F[q] * transpose(F[q])));

       // Calculate the principal stretches (of F),
       // the isochoric principal stresses and the
       // their derivatives. Further, the dyadic
       // products of the eigenvectors are precomputed.
       for (unsigned int a = 0; a < dim; ++a)
           lambda[a] = std::sqrt(eigen_b[a].first);

       for (unsigned int a = 0; a < dim; ++a)
       {
           ln_dyad_ln[a][a] = outer_product (lambda[a]*eigen_b[a].second,lambda[a]*eigen_b[a].second);

           // Due to the symmetry only the right upper triangle
           // must be computed, the rest can be is given by a
           // simple transpose operation.
           for (unsigned int b = a+1; b < dim; ++b)
           {
               ln_dyad_ln[a][b] = outer_product (lambda[a]*eigen_b[a].second,lambda[b]*eigen_b[b].second);
               ln_dyad_ln[b][a] = transpose (ln_dyad_ln[a][b]);
           }
       }

       this->compute_principal_stresses (
               lambda,principal_stresses_iso,principal_stresses_vol);
       this->compute_principal_stress_tangents (
               lambda,principal_stress_tangents_iso, principal_stress_tangents_vol);

       lambda_inv_dprincipal_S_dlambda = 0;
       for (unsigned int a = 0; a < dim; ++a)
       {
           principal_S[a] = (principal_stresses_iso[a]+principal_stresses_vol[a])/lambda[a];
           lambda_inv_dprincipal_S_dlambda[a][a] -= principal_S[a]/(lambda[a]*lambda[a]);

           for (unsigned int b = a; b < dim; ++b)
               lambda_inv_dprincipal_S_dlambda[a][b] +=
                       (principal_stress_tangents_iso[a][b]+principal_stress_tangents_vol[a][b])/(lambda[a]*lambda[b]);
       }

       // Don't forget to reset the Kirchoff stress.
       tau[q] = 0;

       // Non-symmetric tangent. Only when all contributions are
       // gathered, the tangent is symmetric, i.e. we have to store
       // intermediate results in a general tensor object.
       Tensor<4,dim,ad_type> A_aux;

       // Auxiliary variable
       ad_type coefficient;

       for (unsigned int a = 0; a < dim; ++a)
       {
           // Compute the Kirchoff stress tensor.
           tau[q] += principal_S[a]*symmetrize(ln_dyad_ln[a][a]);

           for (unsigned int b = 0; b < dim; ++b)
           {
               A_aux += lambda_inv_dprincipal_S_dlambda[a][b] * outer_product(ln_dyad_ln[a][a],ln_dyad_ln[b][b]);

               // Only non-zero contributions
               // if a and b are different.
               if (a != b)
               {
                   // Check if the eigenvalues are equal, is so
                   // l'Hospital's rule is used in the else-
                   // statement. Otherwise, we would divide by
                   // zero.
                   if (std::fabs(lambda[b]-lambda[a]) > 1e-8)
                   {
                       coefficient = (principal_S[b] - principal_S[a])/
                               (eigen_b[b].first - eigen_b[a].first);
                   }
                   else
                   {
                       coefficient = 0.5*(lambda_inv_dprincipal_S_dlambda[b][b]-lambda_inv_dprincipal_S_dlambda[a][b]);
                   }

                   A_aux += (dealii::outer_product(coefficient * ln_dyad_ln[a][b],ln_dyad_ln[a][b])
                       + dealii::outer_product(coefficient * ln_dyad_ln[a][b],ln_dyad_ln[b][a]));
               }//if
           }//b
       }//a

       // Assemble both contributions into the SymmetricTensor dSdC[q].
       // When doing so, the minor symmetries can be exploited. Note that
       // the sum A_iso + A_vol is already symmetric, therefore using
       // dealii::symmetrize would be more expensive.
       for(unsigned int i = 0; i < dim; ++i)
           for(unsigned int j = i; j < dim; ++j)
               for(unsigned int k = 0; k < dim; ++k)
                   for(unsigned int l = k; l< dim; ++l)
                       cc[q][i][j][k][l] = A_aux[i][j][k][l];
   }//q
}



template<int dim>
std::vector<DataInterpretation>
Ogden<dim>::
get_data_interpretation () const
{
    using namespace dealii;

    // The base class returns the DataInterpretation of the
    // variables returned by evaluate_vector_field.
    std::vector<DataInterpretation> data_interpretation;

    unsigned int position = 0;

    data_interpretation.push_back (
            create_data_interpretation<Tensor<1,dim,scalar_type>>("displacement",position));
    position += data_interpretation.back().n_components();

    data_interpretation.push_back (
            create_data_interpretation<SymmetricTensor<2,dim,scalar_type>>("kirchoff_stress",position));
    position += data_interpretation.back().n_components();

    return data_interpretation;
}



template<int dim>
void
Ogden<dim>::
evaluate_vector_field (const dealii::DataPostprocessorInputs::Vector<dim> &input_data,
                       std::vector<dealii::Vector<double>> &computed_quantities,
                       const dealii::GeneralDataStorage *) const
{
    using namespace dealii;

    unsigned int position = 0;

    // deformation gradient
    Tensor<2,dim,double> F;

    std::array<double,dim> lambda; // principal stretches

    Tensor<1,dim,double> principal_stresses_vol;  // principal isochoric stresses
    Tensor<1,dim,double> principal_stresses_iso;  // principal volumetric stresses

    for (unsigned int q=0; q<input_data.solution_values.size(); ++q)
    {
        double *computed_quantities_ptr = std::addressof(computed_quantities[q][0])+position;

        // displacement
        TensorShape<1,dim,double> u (computed_quantities_ptr);
        computed_quantities_ptr += Utilities::pow (dim,1);

        // Piola stress
        TensorShape<2,dim,double> tau (computed_quantities_ptr);
        computed_quantities_ptr += Utilities::pow (dim,2);

        for(unsigned int i = 0; i < dim; ++i)
        {
            u [i] = input_data.solution_values[q][Extractor<dim>::first_displacement_component+i];
            F [i] = input_data.solution_gradients[q][Extractor<dim>::first_displacement_component+i];
            F [i][i] += 1.0;
        }

        auto eigen_b = eigenvectors(symmetrize(F*transpose(F)));

        // Calculate the principal stretches (of F),
        // the isochoric principal stresses and the
        // their derivatives. Further, the dyadic
        // products of the eigenvectors are precomputed.
        for (unsigned int a = 0; a < dim; ++a)
            lambda[a] = std::sqrt(eigen_b[a].first);

        this->compute_principal_stresses(lambda,principal_stresses_iso,principal_stresses_vol);

        tau = 0;
        for (unsigned int a = 0; a < dim; ++a)
            tau += lambda[a]*(principal_stresses_iso[a]+principal_stresses_vol[a])
                *outer_product(eigen_b[a].second,eigen_b[a].second);
    }
}



template <int dim>
inline
void
Ogden<dim>::
compute_principal_stresses (const std::array<double,dim> &lambda,
                            dealii::Tensor<1,dim,double> &principal_stress_iso,
                            dealii::Tensor<1,dim,double> &principal_stress_vol) const
{
    using namespace dealii;

    // Auxiliary arrays
    std::array<double,dim> lambda_bar;         // principal stretches / J^(1/dim)
    std::array<double,dim> dpsi_dlambda_bar;   // d(Psi[a])/d(lambda_bar[a])

    // Indices
    unsigned int a,b,p;

    // Compute the Jacobian. It is given as the product
    // of the three principal stretches.
    double J = 1.;

    for (unsigned int a = 0; a < dim; ++a)
        J *= lambda[a];

    double dimrt_J = std::pow (J,1./double(dim));

    AssertThrow (J>1e-18, ExcMessage("Degenerated element (Jacobian < 0)."));

    for (a = 0; a < dim; ++a)
    {
        lambda_bar [a] = lambda[a] / dimrt_J;

        dpsi_dlambda_bar[a] = 0;

        for (p = 0; p < this->n_terms; ++p)
            dpsi_dlambda_bar[a] += this->mu[p] * std::pow (lambda_bar[a], this->alpha[p]-1.);
    }

    // See G. Holzapfel, Nonlinear Solid Mechanics, wiley, (1999).
    for (a = 0; a < dim; ++a)
    {
        principal_stress_vol[a] = this->kappa / (this->beta * lambda[a]) * (1. - std::pow(J,-this->beta));
        principal_stress_iso[a] = double(dim) * lambda_bar[a] * dpsi_dlambda_bar[a];

        for (b = 0; b < dim; ++b)
            principal_stress_iso[a] -= lambda_bar[b] * dpsi_dlambda_bar[b];

        principal_stress_iso[a] *= 1./(double(dim)*lambda[a]);
    }
}



template <int dim>
inline
void
Ogden<dim>::
compute_principal_stress_tangents (const std::array<double,dim> &lambda,
                                   dealii::SymmetricTensor<2,dim,double> &principal_stress_tangent_iso,
                                   dealii::SymmetricTensor<2,dim,double> &principal_stress_tangent_vol) const
{
    using namespace dealii;

    // Auxiliary arrays
    std::array<double,dim> lambda_bar;             // principal stretches / J^(1/dim)
    std::array<double,dim> dpsi_iso_dlambda;       // d(Psi[a])/d(lambda[a])
    std::array<double,dim> dpsi_iso_dlambda_bar;   // d(Psi[a])/d(lambda_bar[a])
    std::array<double,dim> d2psi_iso_dlambda_bar2; // d²(Psi[a])/d(lambda_bar[a])²
    std::array<double,dim> dJ_dlambda;             // dJ/d(lambda[a])

    std::array<std::array<double,dim>,dim> dlamba_bar_dlambda;

    // Indices
    unsigned int a,b,c,p;

    // Compute the Jacobian. It is given as the product
    // of the three principal stretches.
    double J = 1.;

    for (a = 0; a < dim; ++a)
        J *= lambda[a];

    AssertThrow(J>1e-18, ExcMessage("Degenerated element (Jacobian < 0)."));

    double dimrt_J     = std::pow (J,1./double(dim));
    double inv_dimrt_J = 1./dimrt_J;
    double inv_dim     = 1./double(dim);

    double dpsi_vol_dJ   = this->kappa / (this->beta*J) * (1. - std::pow(J,-this->beta));
    double d2psi_vol_dJ2 = this->kappa / (this->beta*J*J) * (-1. + (this->beta+1.) * std::pow(J,-this->beta));

    // See G. Holzapfel, Nonlinear Solid Mechanics, wiley, (1999).
    for (a = 0; a < dim; ++a)
    {
        lambda_bar [a] = lambda[a] * inv_dimrt_J;

        dJ_dlambda[a] = J/lambda[a];

        dpsi_iso_dlambda_bar  [a] = 0;
        d2psi_iso_dlambda_bar2[a] = 0;

        for (p = 0; p < this->n_terms; ++p)
        {
            dpsi_iso_dlambda_bar  [a] += this->mu[p] * std::pow (lambda_bar[a], this->alpha[p]-1.);
            d2psi_iso_dlambda_bar2[a] += (this->alpha[p]-1.) * this->mu[p] * std::pow (lambda_bar[a], this->alpha[p]-2.);
        }
    }

    for (a = 0; a < dim; ++a)
    {
        dpsi_iso_dlambda[a] = 0;

        for (b = 0; b < dim; ++b)
        {
            dlamba_bar_dlambda[b][a] = -inv_dim * inv_dimrt_J * (lambda_bar[b]/lambda_bar[a]);
            if (a == b)
                dlamba_bar_dlambda[a][a] += inv_dimrt_J;

            dpsi_iso_dlambda[a] += dpsi_iso_dlambda_bar[b] * dlamba_bar_dlambda[b][a];
        }
    }

    principal_stress_tangent_iso = 0;
    principal_stress_tangent_vol = 0;

    for (a = 0; a < dim; ++a)
    {
        principal_stress_tangent_iso[a][a] -= dpsi_iso_dlambda[a]/lambda[a];
        principal_stress_tangent_vol[a][a] -= dpsi_vol_dJ*J/(lambda[a]*lambda[a]);

        for (b = a; b < dim; ++b)
        {
            principal_stress_tangent_vol[a][b] += d2psi_vol_dJ2*dJ_dlambda[a]*dJ_dlambda[b]
                    + dpsi_vol_dJ/lambda[a] *dJ_dlambda[b];

            double tmp1;
            double tmp2;

            for (c = 0; c < dim; ++c)
            {
                tmp1 = dpsi_iso_dlambda_bar[c]*dlamba_bar_dlambda[c][b];
                tmp2 = d2psi_iso_dlambda_bar2[c]*dlamba_bar_dlambda[c][b]*lambda_bar[c];

                if (c==a)
                    principal_stress_tangent_iso[a][b] += (1.-inv_dim) /lambda[a]* (tmp1+tmp2);
                else
                    principal_stress_tangent_iso[a][b] -= inv_dim /lambda[a]* (tmp1+tmp2);
            }
        }
    }
}



// Instantiations
template class Ogden<2>;
template class Ogden<3>;

// Registration
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Ogden,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Ogden,3));

}// namespace efi


