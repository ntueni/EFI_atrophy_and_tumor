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

// deal.II headers
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

// efi headers
#include <efi/worker/cell_worker.h>
#include <efi/factory/registry.h>
#include <efi/base/automatic_differentiation.h>


namespace efi
{


namespace ScratchDataTools
{

template <int dim, class Number = double>
bool
has_piola_data (ScratchData<dim>  &scratch_data,
                const std::string &global_vector_name,
                const Number       = Number(0))
{
    auto ad_helper_type = get_ad_helper_type(scratch_data, global_vector_name, Number());

    return (stores_piola_stresses (scratch_data, global_vector_name, Number())
          || ((ad_helper_type == energy_functional) && AD::is_ad_number<Number>::value))
         &&(stores_piola_stress_tangents (scratch_data, global_vector_name, Number())
          || ((ad_helper_type == energy_functional
             ||ad_helper_type == residual_linearization)
             && AD::is_ad_number<Number>::value));
}



template <int dim, class Number = double>
bool
has_piola_kirchoff_data (ScratchData<dim>  &scratch_data,
                         const std::string &global_vector_name,
                         const Number       = Number(0))
{
    auto ad_helper_type = get_ad_helper_type(scratch_data, global_vector_name, Number());

    return (stores_deformation_grads (scratch_data, global_vector_name, Number())
           && (stores_piola_kirchoff_stresses (scratch_data, global_vector_name, Number())
              || ((ad_helper_type == energy_functional) && AD::is_ad_number<Number>::value)))
        &&(stores_piola_kirchoff_stress_tangents (scratch_data, global_vector_name, Number())
           || ((ad_helper_type == energy_functional
              ||ad_helper_type == residual_linearization)
              && AD::is_ad_number<Number>::value));
}



template <int dim, class Number = double>
bool
has_kirchoff_data (ScratchData<dim>  &scratch_data,
                         const std::string &global_vector_name,
                         const Number       = Number(0))
{
    auto ad_helper_type = get_ad_helper_type(scratch_data, global_vector_name, Number());

    return (stores_deformation_grads (scratch_data, global_vector_name, Number())
           && (stores_kirchoff_stresses (scratch_data, global_vector_name, Number())
              || ((ad_helper_type == energy_functional) && AD::is_ad_number<Number>::value)))
        &&(stores_kirchoff_stress_tangents (scratch_data, global_vector_name, Number())
           || ((ad_helper_type == energy_functional
              ||ad_helper_type == residual_linearization)
              && AD::is_ad_number<Number>::value));
}

}// namespace ScratchDataTools




namespace efi_internal
{

/// Check if the finite elements contained in fe are primitive and equal.
/// This is particularly useful for <tt>dealii::FESystem</tt>s. However,
/// if the input is valid, this function won't fail for single FEs. Then, it
/// only checks if fe is primitive.
template <int dim>
bool
sub_fes_are_primitive_and_equal (const dealii::FiniteElement<dim> &fe,
                                 const unsigned int first_component,
                                 const unsigned int n_components)
{
    Assert(n_components > 0,dealii::ExcMessage("No FEs selected."));

    const dealii::FiniteElement<dim> &first_fe
                = fe.get_sub_fe(first_component,1);

    // Since we check if the following FEs are equal to the first one, we only
    // need to make sure that the first FE is primitive.
    if (!first_fe.is_primitive())
        return false;

    for (unsigned int i = 1; i < n_components; ++i)
    {
        const dealii::FiniteElement<dim> &next_fe
                = fe.get_sub_fe(first_component+i,1);
        if (first_fe != next_fe)
            return false;
    }

    return true;
}

/// Compute the result of the contraction u_i A_ijkl v_l. Here it is assumed
/// A is symmetric in the indices i and j.
template<int dim>
inline
dealii::Tensor<2,dim>
contractSym3 (const dealii::Tensor<1,dim> &u,
              const dealii::Tensor<4,dim> &A,
              const dealii::Tensor<1,dim> &v)
{

    dealii::Tensor<3,dim> tmp;
    unsigned int i,j,k;
    for (i=0; i<dim; ++i)
    {
        for (j=i+1; j<dim; ++j)
            for (k=0; k<dim; ++k)
            {
                tmp[i][j][k] = A[i][j][k]*v;
                tmp[j][i][k] = tmp[i][j][k];
            }
        for (k=0; k<dim; ++k)
            tmp[i][i][k] = A[i][i][k]*v;
    }
    return u*tmp;
}

}// namespace efi_internal




template <int dim>
void
CellWorker<dim>::
do_fill (ScratchData<dim> &scratch_data,
         CopyData         &copy_data) const
{
    using namespace dealii;

    auto global_vector_name = Extractor<dim>::global_vector_name();

    // Get the number of quadrature points and
    // the number of dofs per cell.
    auto n_q_points    = ScratchDataTools::n_quadrature_points (scratch_data);
    auto dofs_per_cell = ScratchDataTools::dofs_per_cell (scratch_data);

    // Get the current fe values object, which
    // is needed to access the shape functions.
    auto &fe  = ScratchDataTools::get_current_fe_values (scratch_data);
    auto &JxW = ScratchDataTools::get_JxW_values        (scratch_data);

    // Create references to the copy data objects
    // for easy accessibility.
    auto &local_rhs    = copy_data.vectors.back();
    auto &local_matrix = copy_data.matrices.back();

    local_rhs = 0;
    local_matrix = 0;

    // Depending on which stress measures and stress tangents are
    // provided different assembly routines are used.
    if (ScratchDataTools::has_kirchoff_data (scratch_data, global_vector_name, ad_type(0)))
    {

        // Get the quantities which are required for
        // the assembly of the cell contributions.
        auto &F       = ScratchDataTools::get_deformation_grads        (scratch_data, global_vector_name, ad_type());
        auto &tau     = ScratchDataTools::get_kirchoff_stresses        (scratch_data, global_vector_name, ad_type());
        auto &tangent = ScratchDataTools::get_kirchoff_stress_tangents (scratch_data, global_vector_name, ad_type());

        std::vector<Tensor<2,dim,ad_type>> gradN(dofs_per_cell);
        std::vector<SymmetricTensor<2,dim,ad_type>> sym_gradN(dofs_per_cell);

        // If the used finite elements for each displacement component are
        // primitive and equal, we can use a faster routine to evaluate the
        // cell contribution.
        if (efi_internal::sub_fes_are_primitive_and_equal(
                fe.get_fe(),Extractor<dim>::first_displacement_component,dim))
        {
            std::vector<unsigned int> system_to_component_index(dofs_per_cell);
            std::vector<unsigned int> system_to_shape_index(dofs_per_cell);
            std::vector<std::vector<unsigned int>> shape_to_component_indices(dofs_per_cell/dim);
            std::vector<std::vector<unsigned int>> shape_to_system_indices(dofs_per_cell/dim);

            Tensor<2,dim,ad_type> K;
            Tensor<1,dim,ad_type> R;

            unsigned int i,j,r,s;

            for (i = 0; i < dofs_per_cell; ++i)
            {
                system_to_component_index[i] = fe.get_fe().system_to_component_index(i).first;
                system_to_shape_index[i] = fe.get_fe().system_to_base_index(i).second;
                shape_to_component_indices[system_to_shape_index[i]].push_back(system_to_component_index[i]);
                shape_to_system_indices[system_to_shape_index[i]].push_back(i);
            }

            // loop over all quadrature points
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                Tensor<2,dim,ad_type> F_inv = invert(F[q]);
                const Tensor<2,dim,ad_type> my_tau = tau[q];
                const Tensor<4,dim,ad_type> my_cc  = tangent[q];

                for (i = 0; i < dofs_per_cell; ++i)
                    gradN[i] = fe[Extractor<dim>::displacement()].gradient(i,q)*F_inv;

                // loop over shape functions
                for (i = 0; i < dofs_per_cell/dim; ++i)
                {
                    const unsigned int si0 = shape_to_system_indices[i][0];
                    const unsigned int ci0 = shape_to_component_indices[i][0];

                    //FIXME segfault thrown?
                    Assert(si0 < gradN.size(),StandardExceptions::ExcMessage("index exceeds vector gradN"));
                    R = -JxW[q] * (my_tau * gradN[si0][ci0]);

                    for (r = 0; r < dim; ++r)
                        local_rhs(shape_to_system_indices[i][r]) += R[r];

                    for (j = i; j < dofs_per_cell/dim; ++j)
                    {
                        const unsigned int sj0 = shape_to_system_indices[j][0];
                        const unsigned int cj0 = shape_to_component_indices[j][0];

                        K = JxW[q] * efi_internal::contractSym3(gradN[si0][ci0],my_cc,gradN[sj0][cj0]);

                        const double tmp = JxW[q] * (gradN[sj0][cj0] * (my_tau * gradN[si0][ci0]));
                        for (r = 0; r < dim; ++r)
                        {
                            K[r][r] += tmp;
                            const unsigned int sir = shape_to_system_indices[i][r];

                            for (s = 0; s < dim; ++s)
                                local_matrix(sir,shape_to_system_indices[j][s]) += K[r][s];
                        }// r loop
                    }// j loop
                }// i loop
            }// q loop

            // Copy the symmetric part
            for (i = 0; i < dofs_per_cell/dim; ++i)
                for (j = i+1; j < dofs_per_cell/dim; ++j)
                {
                    for (r = 0; r < dim; ++r)
                    {
                        const unsigned int sir = shape_to_system_indices[i][r];
                        for (s = 0; s < dim; ++s)
                        {
                            const unsigned int sjs = shape_to_system_indices[j][s];
                            local_matrix(sjs,sir) = local_matrix(sir,sjs);
                        }// s loop
                    }// r loop
                }// j loop
        }
        else
        {
            // loop over all quadrature points
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                Tensor<2,dim,ad_type> F_inv = invert(F[q]);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    gradN[i]     = fe[Extractor<dim>::displacement()].gradient(i,q)*F_inv;
                    sym_gradN[i] = symmetrize(gradN[i]);
                }

                // loop over all degrees of freedom (rows)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    local_rhs(i) += -JxW[q] * scalar_product(gradN[i],tau[q]);

                    local_matrix(i,i) += JxW[q] * (scalar_product(gradN[i], gradN[i] * tau[q])
                                + (sym_gradN[i] * (tangent[q] * sym_gradN[i])));

                    // Compute the upper triangle of the tangent.
                    for (unsigned int j = i+1; j < dofs_per_cell; ++j)
                    {
                        local_matrix(i,j) += JxW[q] * (scalar_product(gradN[i], gradN[j] * tau[q])
                                + (sym_gradN[i] * (tangent[q] * sym_gradN[j])));
                    }// j loop
                }// i loop
            }// q loop

            // Make use of symmetries
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int j = i+1; j < dofs_per_cell; ++j)
                    local_matrix(j,i) = local_matrix(i,j);
        }
    }
    else if (ScratchDataTools::has_piola_data (scratch_data, global_vector_name, ad_type(0)))
    {
        // get the Piola stress vector
        auto &P    = ScratchDataTools::get_piola_stresses        (scratch_data, global_vector_name, ad_type(0));
        auto &dPdF = ScratchDataTools::get_piola_stress_tangents (scratch_data, global_vector_name, ad_type(0));

        // loop over all quadrature points
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            // loop over all degrees of freedom (rows)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                auto GradNi = fe[Extractor<dim>::displacement()].gradient(i,q);

                local_rhs(i) += -JxW[q] * dealii::scalar_product(GradNi,P[q]);

                local_matrix(i,i) += JxW[q] * dealii::contract3(GradNi,dPdF[q],GradNi);

                // loop over all degrees of freedom (columns)
                for (unsigned int j = i+1; j < dofs_per_cell; ++j)
                {
                    auto GradNj = fe[Extractor<dim>::displacement()].gradient(j,q);

                    local_matrix(i,j) += JxW[q] * dealii::contract3(GradNi,dPdF[q],GradNj);
                }// j loop
            }// i loop
        }// q loop

        // Make use of symmetries
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = i+1; j < dofs_per_cell; ++j)
                local_matrix(j,i) = local_matrix(i,j);
    }
    else if (ScratchDataTools::has_piola_kirchoff_data (scratch_data, global_vector_name, ad_type(0)))
    {
        // Get the quantities which are required for
        // the assembly of the cell contributions.
        auto &F    = ScratchDataTools::get_deformation_grads              (scratch_data, global_vector_name, ad_type());
        auto &S    = ScratchDataTools::get_piola_kirchoff_stresses        (scratch_data, global_vector_name, ad_type());
        auto &dSdC = ScratchDataTools::get_piola_kirchoff_stress_tangents (scratch_data, global_vector_name, ad_type());

        // loop over all quadrature points
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            // loop over all degrees of freedom (rows)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                auto GradNi = fe[Extractor<dim>::displacement()].gradient(i,q);

                local_rhs(i) += -JxW[q] * scalar_product(GradNi,F[q]*S[q]);

                // Compute the diagonal contribution of the tangent.
                auto tmp  = dSdC[q] * symmetrize(transpose(GradNi)*F[q]);

                local_matrix(i,i) += JxW[q] *(scalar_product(GradNi*S[q],GradNi)
                                            + scalar_product(F[q]*tmp,GradNi));

                // Compute the upper triangle of the tangent.
                for (unsigned int j = i+1; j < dofs_per_cell; ++j)
                {
                    auto GradNj = fe[Extractor<dim>::displacement()].gradient(j,q);

                    tmp  = dSdC[q] * symmetrize(transpose(GradNj)*F[q]);

                    local_matrix(i,j) += JxW[q] *(scalar_product(GradNj*S[q],GradNi)
                                                + scalar_product(F[q]*tmp,GradNi));
                }// j loop
            }// i loop
        }// q loop

        // Make use of symmetries
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = i+1; j < dofs_per_cell; ++j)
                local_matrix(j,i) = local_matrix(i,j);
    }
}



template <int dim>
void
CellWorkerAD<dim>::
do_fill (ScratchData<dim> &scratch_data,
         CopyData         &copy_data) const
{
    using namespace dealii;

    auto global_vector_name = Extractor<dim>::global_vector_name();

    // Get the number of quadrature points and
    // the number of dofs per cell.
    auto n_q_points    = ScratchDataTools::n_quadrature_points (scratch_data);
    auto dofs_per_cell = ScratchDataTools::dofs_per_cell (scratch_data);

    // Get the current fe values object, which
    // is needed to access the shape functions.
    auto &fe  = ScratchDataTools::get_current_fe_values (scratch_data);
    auto &JxW = ScratchDataTools::get_JxW_values        (scratch_data);

    // Create references to the copy data objects
    // for easy accessibility.
    auto &local_rhs         = copy_data.vectors.back();
    auto &local_matrix      = copy_data.matrices.back();

    local_rhs = 0;
    local_matrix = 0;

    if (ScratchDataTools::stores_strain_energy_density(scratch_data, global_vector_name, ad_type(0))
        && (ScratchDataTools::get_ad_helper_type(scratch_data, global_vector_name, ad_type()))
            == ScratchDataTools::energy_functional)
    {
        // Here, we use the EnergsFunctional AD-helper
        // get the strain energy function
        auto &Psi = ScratchDataTools::get_strain_energy_density (scratch_data, global_vector_name, ad_type(0));

        ad_type local_energy_ad(0.);

        // loop over all quadrature points
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            local_energy_ad += JxW[q] * Psi[q];
        }// q loop

        ScratchDataTools::register_energy_functional (scratch_data,global_vector_name,ad_type(),local_energy_ad);
        ScratchDataTools::compute_residual           (scratch_data,global_vector_name,ad_type(),local_rhs);
        ScratchDataTools::compute_linearization      (scratch_data,global_vector_name,ad_type(),local_matrix);
        local_rhs *= -1.;
    }
    else
    {
        // Here, we use the ResidualLinearization AD-helper.
        // Therefore initialize the auto-differentiable local
        // rhs-vector.
        std::vector<ad_type> local_rhs_ad (dofs_per_cell, ad_type(0.));

        // Depending on which stress measures and stress tangents are
        // provided different assembly routines are used.
        if (ScratchDataTools::has_piola_data(scratch_data, global_vector_name, ad_type(0)))
        {
            auto &P    = ScratchDataTools::get_piola_stresses        (scratch_data, global_vector_name, ad_type(0));

            // loop over all quadrature points
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                // loop over all degrees of freedom (rows)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    auto GradNi = fe[Extractor<dim>::displacement()].gradient(i,q);

                    local_rhs_ad[i] += JxW[q] * dealii::scalar_product(GradNi,P[q]);
                }// i loop
            }// q loop
        }
        else if (ScratchDataTools::has_piola_kirchoff_data(scratch_data, global_vector_name, ad_type(0)))
        {
            auto &F = ScratchDataTools::get_deformation_grads              (scratch_data, global_vector_name, ad_type());
            auto &S = ScratchDataTools::get_piola_kirchoff_stresses        (scratch_data, global_vector_name, ad_type());

            // loop over all quadrature points
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                // loop over all degrees of freedom (rows)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    auto GradNi = fe[Extractor<dim>::displacement()].gradient(i,q);

                    local_rhs_ad[i] += JxW[q] * dealii::scalar_product(GradNi,F[q]*S[q]);
                }// i loop
            }// q loop
        }

        ScratchDataTools::register_residual     (scratch_data,global_vector_name,ad_type(),local_rhs_ad);
        ScratchDataTools::compute_residual      (scratch_data,global_vector_name,ad_type(),local_rhs);
        ScratchDataTools::compute_linearization (scratch_data,global_vector_name,ad_type(),local_matrix);
        local_rhs *= -1.;
    }
}



// Instantiation
template class CellWorker<2>;
template class CellWorker<3>;

template class CellWorkerAD<2>;
template class CellWorkerAD<3>;


// Registration
EFI_REGISTER_OBJECT (EFI_TEMPLATE_CLASS (CellWorker,2));
EFI_REGISTER_OBJECT (EFI_TEMPLATE_CLASS (CellWorker,3));

EFI_REGISTER_OBJECT (EFI_TEMPLATE_CLASS (CellWorkerAD,2));
EFI_REGISTER_OBJECT (EFI_TEMPLATE_CLASS (CellWorkerAD,3));

}//namespace efi

