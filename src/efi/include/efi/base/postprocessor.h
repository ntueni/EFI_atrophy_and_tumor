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

#ifndef SRC_MYLIB_INCLUDE_EFI_BASE_POSTPROCESSOR_H_
#define SRC_MYLIB_INCLUDE_EFI_BASE_POSTPROCESSOR_H_

// stl headers
#include <string>
#include <vector>
#include <type_traits>

// deal.II headers
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>

// efi headers
#include <efi/base/utility.h>
#include <efi/base/type_traits.h>
#include <efi/worker/general_cell_data_storage.h>


namespace efi
{

// Forward declaration
template <int dim> class ConstitutiveBase;


class DataInterpretation
{
public:

    // Constructor.
    DataInterpretation (const std::string  &name,
                        const unsigned int  rank = 0,
                        const unsigned int  dimension  = 1,
                        const unsigned int  first_global_component = 0);

    // Copy constructor.
    DataInterpretation (const DataInterpretation &) = default;

    // Assignment operator.
    DataInterpretation&
    operator= (const DataInterpretation &) = default;

    // Get the list of the data component names.
    const std::vector<std::string> &
    get_names () const;

    // Get the interpretation of the data components.
    const std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> &
    get_data_component_interpretation () const;

    // Number of components.
    unsigned int
    n_components () const;

    // The first component of the field
    // in the vector of all components.
    unsigned int
    get_first_component () const;

private:

    // Data name, e.g. Piola_stress.
    // Note that
    std::string name;

    // Tensorial rank of the represented
    // data object.
    unsigned int rank;

    // Tensor dimension
    unsigned int dimension;

    // Number of components, i.e. dim^rank
    unsigned int number_of_components;

    // This is for the assembly and denotes
    // the position of the first component,
    // where this object goes.
    unsigned int first_component;

    // Names of the single data components,
    // e.g. Piola_stress_xx, Piola_stress_xy, ...
    std::vector<std::string> data_component_names;

    // Interpretation of the data components,
    // component_is_scalar and
    // component_is_part_of_vector. Unfortunately,
    // there is no interpretation for tensors, why
    // the single components of tensors are interpreted
    // as scalar fields.
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation;
};



/// Return the data interpretation for tensor-valued data objects.
/// To work with this function, a class needs to have the static
/// members rank and dimension.
template <class T>
std::enable_if_t<(
     has_static_member_data_rank<std::decay_t<T>,const unsigned int>::value
  && has_static_member_data_dimension<std::decay_t<T>,const unsigned int>::value),
DataInterpretation>
create_data_interpretation (const std::string & name,
                            const unsigned int position = 0);



/// Return the data interpretation for scalar-valued data objects.
template <class T>
std::enable_if_t<!(
     has_static_member_data_rank<std::decay_t<T>,const unsigned int>::value
  && has_static_member_data_dimension<std::decay_t<T>,const unsigned int>::value),
DataInterpretation>
create_data_interpretation (const std::string & name,
                            const unsigned int position = 0);


// Base class of other *PostProcessor classes.
// It already implements some common functionalities.
template <int dim>
class PostProcessorBase : public dealii::DataPostprocessor<dim>
{
public:

    // Constructor.
    PostProcessorBase ();

    // Destructor.
    virtual
    ~PostProcessorBase ();

    // Return a copy of the list of the data component names.
    virtual std::vector<std::string>
    get_names () const override;

    // Return the UpdateFlags specifying which data is needed
    // in order to post-process the data.
    virtual
    dealii::UpdateFlags
    get_needed_update_flags () const override;

    // Return a vector specifying how the given data has to
    // be interpreted.
    virtual
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation () const override;

protected:

    // Required update flags.
    dealii::UpdateFlags update_flags;

    // Names of the data components.
    std::vector<std::string> names;

    // Interpretation of the data components.
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation;
};



// This is a post processor for a material model.
template <int dim>
class CellDataPostProcessor : public PostProcessorBase<dim>
{
public:

    // Constructor.
    // CellDataPostProcessor (const ConstitutiveBase<dim> &model,
    //                        const GeneralCellDataStorage* cell_data_storage = nullptr);

    // Overloaded Constructor.
    CellDataPostProcessor (const std::map<int,std::unique_ptr<ConstitutiveBase<dim>>> &model_map,
                           const GeneralCellDataStorage* cell_data_storage = nullptr);

    // Destructor.
    virtual
    ~CellDataPostProcessor () = default;

    // Evaluate function overriding the dealii::DataPostprocessor
    // evaluate_vector_field function.
    virtual
    void
    evaluate_vector_field (const dealii::DataPostprocessorInputs::Vector<dim> &input_data,
                           std::vector<dealii::Vector<double>> &computed_quantities) const override;

protected:

    // // Reference to the constitutive model
    // const ConstitutiveBase<dim>* constitutive_model;

    // Reference to the constitutive model map
    const std::map<int,std::unique_ptr<ConstitutiveBase<dim>>> & constitutive_model_map;

    // Pointer to a cell_data_storage, which provides additional
    // information when postprocessing the single cells.
    const GeneralCellDataStorage* cell_data_storage;
};



//------------------- INLINE AND TEMPLATE FUNCTIONS -------------------//



inline
const std::vector<std::string> &
DataInterpretation::
get_names () const
{
    return this->data_component_names;
}



inline
const std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> &
DataInterpretation::
get_data_component_interpretation () const
{
    return this->data_component_interpretation;
}



inline
unsigned int
DataInterpretation::
n_components () const
{
    return this->number_of_components;
}



inline
unsigned int
DataInterpretation::
get_first_component () const
{
    return this->first_component;
}



template <class T>
inline
std::enable_if_t<(
     has_static_member_data_rank<std::decay_t<T>,const unsigned int>::value
  && has_static_member_data_dimension<std::decay_t<T>,const unsigned int>::value),
DataInterpretation>
create_data_interpretation (const std::string & name,
                            const unsigned int position)
{
    return DataInterpretation (name,T::rank,T::dimension,position);
}



template <class T>
inline
std::enable_if_t<!(
     has_static_member_data_rank<std::decay_t<T>,const unsigned int>::value
  && has_static_member_data_dimension<std::decay_t<T>,const unsigned int>::value),
DataInterpretation>
create_data_interpretation (const std::string & name,
                            const unsigned int position)
{
    static_assert (std::is_arithmetic<T>::value," ");

    return DataInterpretation (name,0,1,position);
}


template <int dim>
PostProcessorBase<dim>::
PostProcessorBase ()
: update_flags (dealii::UpdateFlags::update_default)
{ }



template <int dim>
PostProcessorBase<dim>::
~PostProcessorBase ()
{ }



template <int dim>
std::vector<std::string>
PostProcessorBase<dim>::
get_names () const
{
    return this->names;
}



template <int dim>
dealii::UpdateFlags
PostProcessorBase<dim>::
get_needed_update_flags () const
{
    return this->update_flags;
}



template <int dim>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
PostProcessorBase<dim>::
get_data_component_interpretation () const
{
    return this->data_component_interpretation;
}


}//namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_BASE_POSTPROCESSOR_H_ */
