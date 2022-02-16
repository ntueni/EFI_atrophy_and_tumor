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


// efi headers
#include <efi/base/postprocessor.h>
#include <efi/worker/worker_base.h>
#include <efi/constitutive/constitutive_base.h>


namespace efi
{


DataInterpretation::
DataInterpretation (const std::string &name,
                    const unsigned int rank,
                    const unsigned int dimension,
                    const unsigned int first_global_component)
: name (name),
  rank (rank),
  dimension (dimension),
  number_of_components (dealii::Utilities::pow(dimension, rank)),
  first_component (first_global_component),
  data_component_names (number_of_components, name),
  data_component_interpretation (number_of_components,
          (rank==1)? dealii::DataComponentInterpretation::component_is_part_of_vector
                    :dealii::DataComponentInterpretation::component_is_scalar)
{
    if (rank > 1)
    {
        static const char suffixes[] = { 'x', 'y', 'z' };
        for (unsigned int c = 0; c < number_of_components; ++c)
        {
            this->data_component_names[c] += "_";
            for (auto i : unrolled_to_component_index (c, rank, dimension))
                this->data_component_names[c] += suffixes[i];
        }
    }
}



// template <int dim>
// CellDataPostProcessor<dim>::
// CellDataPostProcessor (const ConstitutiveBase<dim>* model,
//                        const GeneralCellDataStorage* cell_data_storage)
// : constitutive_model (model),
//   cell_data_storage (cell_data_storage)
// {
    
//     using namespace dealii;

//     this->update_flags = update_values | update_quadrature_points;

//     auto data_interpretation = constitutive_model->get_data_interpretation ();
//     this->update_flags = this->update_flags | constitutive_model->get_needed_update_flags ();

//     unsigned int first_component = 0;

//     for (auto &info :  data_interpretation)
//     {
//         first_component += info.n_components ();
//         this->names.insert (
//             std::end(this->names),
//             std::begin(info.get_names ()),
//             std::end(info.get_names ()));

//         this->data_component_interpretation.insert (
//             std::end(this->data_component_interpretation),
//             std::begin(info.get_data_component_interpretation ()),
//             std::end(info.get_data_component_interpretation ()));
//     }
// }

template <int dim>
CellDataPostProcessor<dim>::
CellDataPostProcessor (const std::map<int,std::unique_ptr<ConstitutiveBase<dim>>>& model_map,
                       const GeneralCellDataStorage* cell_data_storage)
: constitutive_model_map(model_map),
cell_data_storage (cell_data_storage)
{
    using namespace dealii;

    this->update_flags = update_values | update_quadrature_points;

    for (const auto &cm_pair : constitutive_model_map)
    {
        // ConstitutiveBase<dim> *constitutive_m = &(cm_pair.second);
        auto data_interpretation = cm_pair.second->get_data_interpretation ();
        this->update_flags = this->update_flags | cm_pair.second->get_needed_update_flags ();

        unsigned int first_component = 0;

        for (auto &info :  data_interpretation)
        {
            first_component += info.n_components ();

            for (auto name: info.get_names ())
            {
                if (!std::count(this->names.begin(), this->names.end(), name))
                {
                this->names.insert (
                    std::end(this->names),
                    std::begin(info.get_names ()),
                    std::end(info.get_names ()));

                this->data_component_interpretation.insert (
                    std::end(this->data_component_interpretation),
                    std::begin(info.get_data_component_interpretation ()),
                    std::end(info.get_data_component_interpretation ()));
                }
            }
            
        }
    }
}


template <int dim>
void
CellDataPostProcessor<dim>::
evaluate_vector_field (const dealii::DataPostprocessorInputs::Vector<dim> & input_data,
                       std::vector<dealii::Vector<double>>  &computed_quantities) const
{
    using namespace dealii;

    // Reset everything.
    for (auto &vec : computed_quantities)
        vec = 0;

    const GeneralDataStorage *additional_input_data = nullptr;

    auto cell = input_data.template get_cell<DoFHandler<dim>>();
    int material_id = cell->material_id();
    if (this->cell_data_storage != nullptr)
    {
        additional_input_data = &(this->cell_data_storage->get_data (cell));
    }
    

    this->constitutive_model_map.at(material_id)->evaluate_vector_field (
            input_data, computed_quantities, additional_input_data);
}



template class CellDataPostProcessor<2>;
template class CellDataPostProcessor<3>;

}//namespace efi
