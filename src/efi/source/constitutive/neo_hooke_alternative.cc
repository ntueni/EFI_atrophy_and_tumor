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

// efi headers
#include <efi/constitutive/neo_hooke_alternative.h>
#include <efi/base/extractor.h>
#include <efi/base/automatic_differentiation.h>
#include <efi/lac/tensor_shape.h>
#include <efi/factory/registry.h>


namespace efi {


template<int dim>
inline
void
NeoHookeAlternative<dim>::
evaluate (ScratchData<dim> &scratch_data) const
{
    using namespace dealii;

    using ad_type = double;

    auto global_vector_name = Extractor<dim>::global_vector_name();

    // get the number of quadrature points.
    const unsigned int n_q_points = ScratchDataTools::n_quadrature_points (scratch_data);

    // Create some aliases.
    auto &F      = ScratchDataTools::get_or_add_deformation_grads         (scratch_data,global_vector_name,ad_type());
    auto &J      = ScratchDataTools::get_or_add_jacobians                 (scratch_data,global_vector_name,ad_type());
    auto &Finv   = ScratchDataTools::get_or_add_inverse_deformation_grads (scratch_data,global_vector_name,ad_type());
    auto &P      = ScratchDataTools::get_or_add_piola_stresses            (scratch_data,global_vector_name,ad_type());
    auto &dPdF   = ScratchDataTools::get_or_add_piola_stress_tangents     (scratch_data,global_vector_name,ad_type());
    auto &tau    = ScratchDataTools::get_or_add_kirchoff_stresses         (scratch_data,global_vector_name,ad_type());

    // Get the displacement gradients.
    auto &Grad_u = ScratchDataTools::get_gradients (scratch_data,global_vector_name,Extractor<dim>::displacement(),ad_type());

    // Store a copy of the identity matrix.
    auto Id = unit_symmetric_tensor<dim>();

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
        F    [q] = Id + Grad_u[q];
        J    [q] = determinant (F[q]);
        Finv [q] = invert (F[q]);

        double kappa = this->lambda +2./3.*this->mu;
        double beta = -2.;
        double dimrtJ = std::pow(J[q],1./double(dim));
        Tensor<2,dim,ad_type> Pvol = kappa/beta *(1.-std::pow(J[q],-beta))*transpose(Finv[q]);
        Tensor<2,dim,ad_type> Pdev = this->mu*(1./(dimrtJ*dimrtJ)*F[q]
                   - 1./(double(dim)*dimrtJ*dimrtJ) * scalar_product(F[q],F[q]) * transpose(Finv[q]));

        P   [q] = Pdev + Pvol;
        tau [q] = symmetrize (P[q] * transpose (F[q]));

        // Compute the tangent dP/dF.
        unsigned int i,j,k,l;

        for (i = 0; i < dim; ++i)
            for (j = 0; j < dim; ++j)
                for (k = 0; k < dim; ++k)
                    for (l = 0; l < dim; ++l)
                        dPdF[q][i][j][k][l] = this->mu * (1./(dimrtJ*dimrtJ) * Id[i][k]*Id[j][l]
                                            - 2./(double(dim)*dimrtJ*dimrtJ)*F[q][i][j]*Finv[q][l][k]
                                            + 1./(double(dim)*dimrtJ*dimrtJ) * scalar_product(F[q],F[q]) *Finv[q][l][i]*Finv[q][j][k])
                                            - 2./double(dim)*Finv[q][j][i]*Pdev[k][l]
                                            + kappa*std::pow(J[q],-beta)*Finv[q][j][i]*Finv[q][l][k]
                                            -kappa/beta *(1.-std::pow(J[q],-beta))*Finv[q][l][i]*Finv[q][j][k];
    }
}



template<int dim>
inline
dealii::UpdateFlags
NeoHookeAlternative<dim>::
get_needed_update_flags () const
{
    return dealii::update_gradients;
}



template<int dim>
inline
std::vector<DataInterpretation>
NeoHookeAlternative<dim>::
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
inline
void
NeoHookeAlternative<dim>::
evaluate_vector_field (const dealii::DataPostprocessorInputs::Vector<dim> &input_data,
                       std::vector<dealii::Vector<double>> &computed_quantities,
                       const dealii::GeneralDataStorage*) const
{
    using namespace dealii;

    // tensors
    Tensor<2,dim> F;

    unsigned int position = 0;

    for (unsigned int q=0; q<input_data.solution_values.size(); ++q)
    {
        double *computed_quantities_ptr = std::addressof(computed_quantities[q][0])+position;

        // displacement
        TensorShape<1,dim> u (computed_quantities_ptr);
        computed_quantities_ptr += Utilities::pow (dim,1);

        // Piola stress
        TensorShape<2,dim> tau (computed_quantities_ptr);
        computed_quantities_ptr += Utilities::pow (dim,2);

        for(unsigned int i = 0; i < dim; ++i)
        {
            u [i] = input_data.solution_values[q][Extractor<dim>::first_displacement_component+i];
            F [i] = input_data.solution_gradients[q][Extractor<dim>::first_displacement_component+i];
            F [i][i] += 1.0;
        }

        double kappa = this->lambda +2./3.*this->mu;
        double beta = -2.;
        double J = determinant(F);
        double dimrtJ = std::pow(J,1./double(dim));
        Tensor<2,dim> Pvol = kappa/beta *(1.-std::pow(J,-beta))*transpose(invert(F));
        Tensor<2,dim> Pdev = this->mu*(1./(dimrtJ*dimrtJ)*F
                   - 1./(double(dim)*dimrtJ*dimrtJ) * scalar_product(F,F) * transpose(invert(F)));

        tau = (Pdev+Pvol)*transpose(F);
    }
}



// Instantiations
template class NeoHookeAlternative<2>;
template class NeoHookeAlternative<3>;

// Registration
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(NeoHookeAlternative,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(NeoHookeAlternative,3));

}// namespace efi
