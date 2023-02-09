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
#include <efi/constitutive/maxwell_wiechert.h>
#include <efi/constitutive/constitutive_factory.h>
#include <efi/factory/registry.h>
#include <efi/base/factory_tools.h>
#include <efi/base/cloneable_function.h>
#include <efi/base/extractor.h>
#include <efi/lac/tensor_shape.h>

namespace efi {


template<int dim>
inline
MaxwellWiechert<dim>::
MaxwellWiechert(const std::string &subsection_name,
                const std::string &unprocessed_input)
:
    ConstitutiveBase<dim>(subsection_name,unprocessed_input)
{
    using namespace dealii;

    FactoryTools::action_type add_component
    = [&] (const FactoryTools::Specifications &specs,
           const std::string                  &unprocessed_input) -> void
    {
        this->components.emplace_back (
                ConstitutiveFactory<dim>::create (
                        this->get_section_path(), specs, unprocessed_input));
    };

    // Put all actions into a map.
    std::map<std::string,FactoryTools::action_type> actions;
    actions[ConstitutiveFactory<dim>::keyword()] = add_component;

    std::istringstream unprocessed_input_stream (unprocessed_input);
    // Apply the actions to the unprocessed input.
    // The only thing we expect to happen is that
    // spring is going to be initialized with a
    // constitutive model.
    FactoryTools::apply (actions, unprocessed_input_stream);

    Assert(!this->components.empty(), ExcMessage("No components created."));

    efilog(Verbosity::verbose) << "New MaxwellWiechert constitutive model created ("
                               << subsection_name
                               << ")."<< std::endl;
}



template <int dim>
void
MaxwellWiechert<dim>::
evaluate (ScratchData<dim> &scratch_data) const
{
    using namespace dealii;

    using ad_type = scalar_type;

    auto global_vector_name = Extractor<dim>::global_vector_name();

    // Get the number of quadrature points.
    const unsigned int n_q_points = ScratchDataTools::n_quadrature_points (scratch_data);

    std::vector<SymmetricTensor<2,dim,ad_type>> sum_tau(n_q_points);
    std::vector<SymmetricTensor<4,dim,ad_type>> sum_cc(n_q_points);

    auto &F   = ScratchDataTools::get_or_add_deformation_grads        (scratch_data,global_vector_name,ad_type(0));
    auto &tau = ScratchDataTools::get_or_add_kirchoff_stresses        (scratch_data,global_vector_name,ad_type(0));
    auto &cc  = ScratchDataTools::get_or_add_kirchoff_stress_tangents (scratch_data,global_vector_name,ad_type(0));

    // Get the displacement gradients.
    auto &Grad_u = ScratchDataTools::get_gradients (scratch_data,global_vector_name,Extractor<dim>::displacement(),ad_type(0));

    for (auto &c : this->components)
    {
        c->evaluate (scratch_data);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            sum_tau[q] += tau[q];
            sum_cc [q] += cc[q];
        }
    }

    // It would be surprising if F has not been calculated by one of
    // the components. However, we cannot be sure, therefore do it here
    // as well.
    for (unsigned int q = 0; q < n_q_points; ++q)
        F[q] = Physics::Elasticity::StandardTensors<dim>::I + Grad_u[q];

    tau = sum_tau;
    cc  = sum_cc;

    // Get access to the cell_data_storage to get the updated
    // history variables.
    dealii::GeneralDataStorage &tmp_history_data
        = ScratchDataTools::get_tmp_history_data (scratch_data);

    // TODO would be nice if we would have a better solution to deduce
    // whether we work a cell or on a face.Also the subface case, which might
    // occur on interface is  is not covered not vovered yet.
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

    // Add the kirchoff stress to the updated_history_data
    tmp_history_data.template add_or_overwrite_copy (
            this->section_path_str + "kirchoff_stresses" + str_extension, tau);

    // Add the quadrature points to the updated_history_data
    tmp_history_data.template add_or_overwrite_copy (
            this->section_path_str + "quadrature_points"  + str_extension,
                ScratchDataTools::get_quadrature_points(scratch_data));
}



template<int dim>
std::vector<DataInterpretation>
MaxwellWiechert<dim>::
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

    data_interpretation.push_back (
            create_data_interpretation<SymmetricTensor<2,dim,scalar_type>>("lagrangian_strain",position));
    position += data_interpretation.back().n_components();

    data_interpretation.push_back (
            create_data_interpretation<Tensor<0,dim,scalar_type>>("max_principal_strain",position));
    position += data_interpretation.back().n_components();

    data_interpretation.push_back (
            create_data_interpretation<Tensor<0,dim,scalar_type>>("min_principal_strain",position));
    position += data_interpretation.back().n_components();

    data_interpretation.push_back (
            create_data_interpretation<Tensor<0,dim,scalar_type>>("max_principal_stress",position));
    position += data_interpretation.back().n_components();

    // data_interpretation.push_back (
    //         create_data_interpretation<Tensor<0,dim,scalar_type>>("max_shear_strain",position));
    // position += data_interpretation.back().n_components();
    
    // data_interpretation.push_back (
    //         create_data_interpretation<Tensor<0,dim,scalar_type>>("von_mises_stress",position));
    // position += data_interpretation.back().n_components();

    return data_interpretation;
}



template<int dim>
void
MaxwellWiechert<dim>::
evaluate_vector_field (const dealii::DataPostprocessorInputs::Vector<dim> &input_data,
                       std::vector<dealii::Vector<double>> &computed_quantities,
                       const dealii::GeneralDataStorage* additional_intput_data) const
{
    using namespace dealii;

    // deformation gradient
    Tensor<2,dim,double> F;
    Tensor<2,dim> identity;
    identity = 0.;
    identity[0][0] = 1.0;
    identity[1][1] = 1.0;
    identity[2][2] = 1.0;

    if (additional_intput_data != nullptr)
    {
        // Add the kirchoff stress to the updated_history_data
        auto &tau_stored =
        additional_intput_data->get_object_with_name<
            std::vector<dealii::SymmetricTensor<2,dim,scalar_type>>> (
                this->section_path_str + "kirchoff_stresses");

        // Add the quadrature points to the updated_history_data
        auto &qp_stored =
        additional_intput_data->get_object_with_name<
            std::vector<dealii::Point<dim>>> (
                this->section_path_str + "quadrature_points");

        FittedFunction<dim> tau_fitted(tau_stored, qp_stored);

        for (unsigned int q=0; q<input_data.solution_values.size(); ++q)
        {
            double *computed_quantities_ptr = std::addressof(computed_quantities[q][0]);

            // displacement
            TensorShape<1,dim,double> u (computed_quantities_ptr);
            computed_quantities_ptr += Tensor<1,dim>::n_independent_components;

            // Piola stress
            double *tau_begin = computed_quantities_ptr;
            computed_quantities_ptr += Tensor<2,dim>::n_independent_components;

            // Lagranigan strain
            TensorShape<2,dim,double> E (computed_quantities_ptr);
            computed_quantities_ptr += Utilities::pow (dim,2);

            // Lagranigan strain
            TensorShape<0,dim,double> max_principal_strain (computed_quantities_ptr);
            computed_quantities_ptr += Utilities::pow (dim,0);
            TensorShape<0,dim,double> min_principal_strain (computed_quantities_ptr);
            computed_quantities_ptr += Utilities::pow (dim,0);
            TensorShape<0,dim,double> max_principal_stress (computed_quantities_ptr);
            computed_quantities_ptr += Utilities::pow (dim,0);
            // TensorShape<0,dim,double> von_mises_stress (computed_quantities_ptr);
            // computed_quantities_ptr += Utilities::pow (dim,0);
            
            // AssertDimension ((Tensor<1,dim>::n_independent_components+Tensor<2,dim>::n_independent_components),
            //                      computed_quantities[q].size())

            for(unsigned int i = 0; i < dim; ++i)           
                {
                    u [i] = input_data.solution_values[q][Extractor<dim>::first_displacement_component+i];
                    F [i] = input_data.solution_gradients[q][Extractor<dim>::first_displacement_component+i];
                    F [i][i] += 1.0;
                }

            E = 0.5*(transpose(F)*F-identity);     

            auto eigen_E = eigenvectors(symmetrize(0.5*(transpose(F)*F-identity)));
            max_principal_strain = eigen_E[0].first;
            min_principal_strain = eigen_E[2].first;

            for (unsigned int i = 0; i < (Tensor<2,dim>::n_independent_components); ++i)
                tau_begin[i] = tau_fitted.value (qp_stored[q],i);
            
            auto eigen_S = eigenvectors(tau_stored[q]);
            max_principal_stress = eigen_S[0].first;
        }
    }
    else
    {
        for (unsigned int q=0; q<input_data.solution_values.size(); ++q)
        {
            double *computed_quantities_ptr = std::addressof(computed_quantities[q][0]);

            // displacement
            TensorShape<1,dim,double> u (computed_quantities_ptr);
            computed_quantities_ptr += Tensor<1,dim>::n_independent_components;

            AssertDimension ((Tensor<1,dim>::n_independent_components),
                             computed_quantities[q].size());

            for(unsigned int i = 0; i < dim; ++i)
                u [i] = input_data.solution_values[q][Extractor<dim>::first_displacement_component+i];
        }
    }
}



// Instantiation
template class MaxwellWiechert<2>;
template class MaxwellWiechert<3>;

// Registration
EFI_REGISTER_OBJECT (EFI_TEMPLATE_CLASS (MaxwellWiechert,2));
EFI_REGISTER_OBJECT (EFI_TEMPLATE_CLASS (MaxwellWiechert,3));

}// namespace efi

