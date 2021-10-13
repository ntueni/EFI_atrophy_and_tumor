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

#ifndef SRC_MYLIB_INCLUDE_EFI_WORKER_SCRATCH_DATA_H_
#define SRC_MYLIB_INCLUDE_EFI_WORKER_SCRATCH_DATA_H_

// dealii headers
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/meshworker/scratch_data.h>

// boost headers
#include <boost/any.hpp>

// efi headers
#include <efi/base/automatic_differentiation.h>
#include <efi/base/utility.h>
#include <efi/base/type_traits.h>
#include <efi/worker/general_cell_data_storage.h>


// This macro is used to define the function
// get_or_add_dataname, and get_dataname where
// "dataname" is the first macro argument.
// The function signatures are as follows.
// This macro had to be used in ScratchDataTools
// namespace.
//
// Return a reference to a vector with objects of type datatype.
// If it does not exist, then the it is constructed
// and a reference to this new object then be returned.
// The size of the vector is set to the number of
// quadrature points of the current initialized fe values object.
// Note that one of the ScratchData::reinit (...) functions
// must have been called before.
// template <int dim, class Number = double>
// std::vector<datatype> &
// get_or_add_dataname (
//     ScratchData<dim>  &scratch_data,
//     const std::string &global_vector_name,
//     const Number dummy = Number(0))
//
// Return a reference to a vector of objects of type datatype.
// Note that one of the ScratchData::reinit (...) functions
// must have been called before. The vector elements are the
// values of the field 'dataname' at the quadrature points of the
// current finite element stored in scratch_data.
// template <int dim, class Number = double>
// std::vector<datatype> &
// get_dataname (
//     ScratchData<dim>  &scratch_data,
//     const std::string &global_vector_name,
//     const Number dummy = Number(0))
//
// Return true if the field 'dataname' is stored in scratch_data,
// false otherwise.
// template <int dim, class Number = double>
// bool
// stores_dataname (
//     const ScratchData<dim>  &scratch_data,
//     const std::string &global_vector_name,
//     const Number dummy = Number(0))
#define EFI_SCRATCH_DATA_ACCESS_TO(dataname,datatype) template <int dim, class Number = double>         \
inline                                                                                                  \
std::vector<typename efi::efi_internal::get_type_from_macro_input<void datatype>::type> & \
get_or_add_##dataname (                                                                                 \
    ScratchData<dim>  &scratch_data,                                                                    \
    const std::string &global_vector_name,                                                              \
    const Number dummy = Number(0))                                                                     \
{                                                                                                       \
    const std::string name = efi_internal::get_unique_name (global_vector_name,                             \
                                              EFI_STRINGIFY(EFI_COMBINE_NAMES(dataname,_q)),            \
                                              n_quadrature_points (scratch_data),                       \
                                              dummy);                                                   \
                                                                                                        \
    return scratch_data.get_general_data_storage().                                                     \
            template get_or_add_object_with_name<                                                       \
                std::vector<typename efi::efi_internal::get_type_from_macro_input<void datatype>::type>>    \
            (name, n_quadrature_points (scratch_data));                                                 \
}                                                                                                       \
                                                                                                        \
                                                                                                        \
                                                                                                        \
template <int dim, class Number = double>                                                               \
inline                                                                                                  \
std::vector<typename efi::efi_internal::get_type_from_macro_input<void datatype>::type> &                   \
get_##dataname (                                                                                        \
    ScratchData<dim>  &scratch_data,                                                                    \
    const std::string &global_vector_name,                                                              \
    const Number dummy = Number(0))                                                                     \
{                                                                                                       \
    const std::string name = efi_internal::get_unique_name (global_vector_name,                             \
                                              EFI_STRINGIFY(EFI_COMBINE_NAMES(dataname,_q)),            \
                                              n_quadrature_points (scratch_data),                       \
                                              dummy);                                                   \
                                                                                                        \
    Assert (scratch_data.get_general_data_storage().stores_object_with_name(name),                      \
            dealii::ExcMessage ("No object with name " + name +                                            \
                                " not stored in ScratchData::user_data_storage."));                     \
                                                                                                        \
    return scratch_data.get_general_data_storage().                                                     \
            template get_object_with_name<                                                              \
                std::vector<typename efi::efi_internal::get_type_from_macro_input<void datatype>::type>>    \
            (name);                                                                                     \
}                                                                                                       \
                                                                                                        \
                                                                                                        \
                                                                                                        \
template <int dim, class Number = double>                                                               \
inline                                                                                                  \
bool                                                                                                    \
stores_##dataname (                                                                                     \
    ScratchData<dim>  &scratch_data,                                                                    \
    const std::string &global_vector_name,                                                              \
    const Number dummy = Number(0))                                                                     \
{                                                                                                       \
        const std::string name = efi_internal::get_unique_name (global_vector_name,                         \
                                              EFI_STRINGIFY(EFI_COMBINE_NAMES(dataname,_q)),            \
                                              n_quadrature_points (scratch_data),                       \
                                              dummy);                                                   \
        return scratch_data.get_general_data_storage().stores_object_with_name(name);                   \
}






namespace efi {


template <int dim>
using ScratchData = dealii::MeshWorker::ScratchData<dim>;


// This namespace provides an interface for a number of
// frequently (user-)defined fields stored in
// ScratchData::user_data_scratch_data.
// TODO current and neighbor cell?
// Should all functions here be to work on
// the current cell.
namespace
ScratchDataTools
{

    // Enumeration of ad-helper types used in the
    // scratch data together with automatic differentiation.
    enum ADHelperType {none,
                       residual_linearization,
                       energy_functional};

    // Convert an ADHelperType to a string
    inline
    std::string
    ad_helper_type_to_string (const ADHelperType type);

    // Output type for function values etc.
    // TODO: Eclipse cannot parse
    //       typename dealii::FEValuesViews::View<dim,dim,Extractor>::template OutputType<Number>;
    //       As long this is the case the equivalent type
    //       dealii::efi_internal::FEValuesViews::ViewType<dim,dim,Extractor>::type::template OutputType<Number>;
    //       will be uase. However, it uses the 'internal' namespace of the deal.II
    //       library, which should be avoided in general.
    template <int dim, class Number, class Extractor>
    using OutputType
    =
    typename dealii::internal::FEValuesViews::ViewType<dim,dim,Extractor>::type::template OutputType<Number>;


    //////////////////////////////////////////////////
    // re-implementation of scratch data functions
    //////////////////////////////////////////////////

    // Reinit the ScratchData object for the given cell.
    template <class CellIteratorType, int dim>
    void
    reinit (ScratchData<dim>       &scratch_data,
            const CellIteratorType &cell);

    // Reinit the ScratchData object for the given cell.
    template <class CellIteratorType, int dim>
    void
    reinit (ScratchData<dim>       &scratch_data,
            const CellIteratorType &cell,
            const unsigned int      face_no);

    // Get the number of quadrature points of
    // the number of quadrature points from the
    // current FEValues object
    // (see also ScratchData<dim>::get_current_fe_values ()
    // in the deal.II documentation).
    // Note that one of the ScratchData::reinit (...) functions
    // must have been called before.
    template <int dim>
    unsigned int
    n_quadrature_points (const ScratchData<dim> &scratch_data);

    // Get the number of dofs per cell from the
    // current FEValues object
    // (see also ScratchData<dim>::get_current_fe_values ()
    // in the deal.II documentation).
    // Note that one of the ScratchData::reinit (...) functions
    // must have been called before.
    template <int dim>
    unsigned int
    dofs_per_cell (const ScratchData<dim> &scratch_data);

    // Extract the local dof values.
    // If Number is an ad number a substitution failure occurs and
    // The other version of this function is used.
    template <int dim, class VectorType, class Number = double>
    std::enable_if_t<AD::is_ad_number<Number>::value>
    extract_local_dof_values (ScratchData<dim>  &scratch_data,
                              const std::string &global_vector_name,
                              const VectorType  &input_vector,
                              const Number       dummy = Number(0));

    // Extract the local dof values.
    // If Number is an ad number a substitution failure occurs and
    // The other version of this function is used.
    // The extracted local dof values are directly registered as
    // automatic-differentiation-variables.
    template <int dim, class VectorType, class Number = double>
    std::enable_if_t<!AD::is_ad_number<Number>::value>
    extract_local_dof_values (ScratchData<dim>  &scratch_data,
                              const std::string &global_vector_name,
                              const VectorType  &input_vector,
                              const Number       dummy = Number(0));

    // Return a reference to the local dof values.
    // If Number is an ad number a substitution failure occurs and
    // The other version of this function is used.
    // ScratchDataTools::extract_local_dof_values must have
    // been called first.
    template <int dim, class Number = double>
    const std::enable_if_t<AD::is_ad_number<Number>::value, std::vector<Number>> &
    get_local_dof_values (const ScratchData<dim> &scratch_data,
                          const std::string      &global_vector_name,
                          const Number dummy = Number(0));

    // Return a reference to the local dof values.
    // If Number is an ad number a substitution failure occurs and
    // The other version of this function is used.
    // ScratchDataTools::extract_local_dof_values must have
    // been called first.
    template <int dim, class Number = double>
    const std::enable_if_t<!AD::is_ad_number<Number>::value, std::vector<Number>> &
    get_local_dof_values (const ScratchData<dim> &scratch_data,
                          const std::string      &global_vector_name,
                          const Number dummy = Number(0));

    // Return the JxW values of the current element.
    template <int dim>
    const dealii::FEValuesBase<dim> &
    get_current_fe_values (const ScratchData<dim> &scratch_data);

    // Return the normal vectors of the current element.
    template <int dim>
    const std::vector<dealii::Point<dim>>&
    get_quadrature_points (const ScratchData<dim> &scratch_data);

    // Return the JxW values of the current element.
    template <int dim>
    const std::vector<double>&
    get_JxW_values (const ScratchData<dim> &scratch_data);

    // Return the normal vectors of the current element.
    template <int dim>
    const std::vector<dealii::Tensor<1,dim>>&
    get_normal_vectors (const ScratchData<dim> &scratch_data);

    // Return the dof indices of the current element.
    template <int dim>
    const std::vector<dealii::types::global_dof_index>&
    get_local_dof_indices (const ScratchData<dim> &scratch_data);

    // Return the values of the selected components of the finite
    // element function, characterized by local degrees-of-freedom vector
    // input_vector, at the quadrature points of the cell, face or
    // subface selected the last time the reinit function of the
    // ScratchData object was called.
    // If Number is an non-ad-number a substitution failure occurs and
    // The other version of this function is used.
    // ScratchDataTools::extract_local_dof_values must have
    // been called first.
    template <int dim, class Extractor, class Number = double>
    const std::enable_if_t<AD::is_ad_number<Number>::value,
        std::vector<typename OutputType<dim,Number,Extractor>::value_type>
    > &
    get_values (ScratchData<dim> &scratch_data,
                const std::string      &global_vector_name,
                const Extractor        &variable,
                const Number            dummy = Number(0));

    // Return the values of the selected components of the finite
    // element function, characterized by local degrees-of-freedom vector
    // input_vector, at the quadrature points of the cell, face or
    // subface selected the last time the reinit function of the
    // ScratchData object was called.
    // If Number is an ad-number a substitution failure occurs and
    // The other version of this function is used.
    // ScratchDataTools::extract_local_dof_values must have
    // been called first.
    template <int dim, class Extractor, class Number = double>
    const std::enable_if_t<!AD::is_ad_number<Number>::value,
        std::vector<typename OutputType<dim,Number,Extractor>::value_type>
    > &
    get_values (ScratchData<dim> &scratch_data,
                const std::string      &global_vector_name,
                const Extractor        &variable,
                const Number            dummy = Number(0));

    // Return the gradients of the selected components of the finite
    // element function, characterized by local degrees-of-freedom vector
    // input_vector, at the quadrature points of the cell, face or
    // subface selected the last time the reinit function of the
    // ScratchData object was called.
    // If Number is a non-ad-number a substitution failure occurs and
    // The other version of this function is used.
    // ScratchDataTools::extract_local_dof_values must have
    // been called first.
    template <int dim, class Extractor, class Number = double>
    const std::enable_if_t<AD::is_ad_number<Number>::value,
        std::vector<typename OutputType<dim,Number,Extractor>::gradient_type>
    > &
    get_gradients (ScratchData<dim> &scratch_data,
                   const std::string      &global_vector_name,
                   const Extractor        &variable,
                   const Number            dummy = Number(0));

    // Return the gradients of the selected components of the finite
    // element function, characterized by local degrees-of-freedom vector
    // input_vector, at the quadrature points of the cell, face or
    // subface selected the last time the reinit function of the
    // ScratchData object was called.
    // If Number is an ad-number a substitution failure occurs and
    // The other version of this function is used.
    // ScratchDataTools::extract_local_dof_values must have
    // been called first.
    template <int dim, class Extractor, class Number = double>
    const std::enable_if_t<!AD::is_ad_number<Number>::value,
        std::vector<typename OutputType<dim,Number,Extractor>::gradient_type>
    > &
    get_gradients (ScratchData<dim> &scratch_data,
                   const std::string      &global_vector_name,
                   const Extractor        &variable,
                   const Number            dummy = Number(0));


    /////////////////////////////////////////////////////////////////
    // cell data storage
    /////////////////////////////////////////////////////////////////

    // Get a reference to the cell data storage. If it does not
    // exist yet, it is created.
    template <int dim>
    void
    attach_history_data_storage (ScratchData<dim> &scratch_data,
                                 GeneralCellDataStorage& cell_data_storage);

    // Get a reference to the cell data storage. If it does not
    // exist yet, it is created.
    template <int dim>
    void
    attach_tmp_history_data_storage (ScratchData<dim> &scratch_data,
                                     GeneralCellDataStorage& cell_data_storage);

    // Get a reference to the cell data storage.
    template <int dim>
    dealii::GeneralDataStorage&
    get_history_data (ScratchData<dim> &scratch_data);

    // Get a reference to the cell data storage.
    template <int dim>
    dealii::GeneralDataStorage&
    get_tmp_history_data (ScratchData<dim> &scratch_data);

    // Get a reference to the cell data storage.
    template <int dim>
    double&
    get_time_step_size (ScratchData<dim> &scratch_data);

    // Get a reference to the cell data storage.
    template <int dim>
    double&
    get_or_add_time_step_size (ScratchData<dim> &scratch_data);

    /////////////////////////////////////////////////////////////////
    // AD functions
    //
    // (see dealii::Differentiation::AD::CellLevelBase
    //      dealii::Differentiation::AD::ResidualLinearization
    //  and dealii::Differentiation::AD::EnergyFunctional)
    /////////////////////////////////////////////////////////////////


    // Return a reference to a dealii::Differentiation::AD::CellLevelBase
    // automatic differentiation helper.
    // If it does not exist, then the it is constructed
    // and a reference to this new object then be returned.
    template <int dim, class ADNumberType>
    AD::CellLevelBase<AD::ADNumberTraits<ADNumberType>::type_code,
        typename AD::ADNumberTraits<ADNumberType>::scalar_type> &
    get_or_add_ad_helper (ScratchData<dim>   &scratch_data,
                          const std::string  &global_vector_name,
                          const ADNumberType  dummy);

    // Set the type of the AD-helper object to be used internally (
    // dealii::Differentiation::AD::ResidualLinearization or
    // dealii::Differentiation::AD::EnergyFunctional) in order to
    // handle AD numbers of type Number. For different global_vector_names
    // different ad_helper_types can be set. The allowed strings for helper
    // type are "residual_linearization" and "energy_functional".
    // Note that ScratchDataTools::extract_local_dof_values() must be called
    // again. All fields relying on the data of the previously set
    // helper_type get invalidated (more precisely, all fields that were
    // initialized the with global_vector_name and the same number type of
    // the dummy variable).
    template <int dim, class ADNumberType>
    void
    set_ad_helper_type (ScratchData<dim>   &scratch_data,
                        const std::string  &global_vector_name,
                        const ADNumberType  dummy,
                        const ADHelperType  helper_type);

    // Return the string specifying the ad_helper type.
    template <int dim, class ADNumberType>
    ADHelperType
    get_ad_helper_type ( ScratchData<dim>  &scratch_data,
                        const std::string &global_vector_name,
                        const ADNumberType       dummy);


    // Return a reference to a dealii::Differentiation::AD::CellLevelBase
    // automatic differentiation helper.
    // Note that one of the ScratchData::reinit (...) functions
    // must have been called before.
    template <int dim, class ADNumberType>
    AD::CellLevelBase<AD::ADNumberTraits<ADNumberType>::type_code,
        typename AD::ADNumberTraits<ADNumberType>::scalar_type> &
    get_ad_helper (ScratchData<dim>   &scratch_data,
                   const std::string  &global_vector_name,
                   const ADNumberType  dummy);

    // Return a reference to a dealii::Differentiation::AD::CellLevelBase
    // automatic differentiation helper.
    // Note that one of the ScratchData::reinit (...) functions
    // must have been called before.
    template <int dim, class ADNumberType>
    AD::CellLevelBase<AD::ADNumberTraits<ADNumberType>::type_code,
        typename AD::ADNumberTraits<ADNumberType>::scalar_type> &
    get_ad_helper (ScratchData<dim>   &scratch_data,
                   const std::string  &global_vector_name,
                   const ADNumberType  dummy);

    // Return a read-only reference to a dealii::Differentiation::AD::CellLevelBase
    // automatic differentiation helper.
    // Note that one of the ScratchData::reinit (...) functions
    // must have been called before.
    template <int dim, class ADNumberType>
    AD::CellLevelBase<AD::ADNumberTraits<ADNumberType>::type_code,
        typename AD::ADNumberTraits<ADNumberType>::scalar_type> &
    get_ad_helper (const ScratchData<dim>  &scratch_data,
                   const std::string  &global_vector_name,
                   const ADNumberType  dummy);


    // Register the previously extracted local dof values of
    // type ScalarNumberType as variables for automatic
    // differentiation. Internally a vector of local dof values
    // of type ADNumberType is created and these are set as
    // ad-variables.
    // ScratchDataTools::extract_local_dof_values must have
    // been called first for the number type ScalarNumberType.
    // ADNumberType must be an automatically differentiable number
    // type, i.e. AD::is_ad_number must return true.
    // ScalarNumberType must be a non-AD number, i.e. AD::is_ad_number
    // must return false.
    template <int dim, class ScalarNumberType, class ADNumberType>
    std::enable_if_t<!std::is_same<typename AD::ADNumberTraits<ADNumberType>::scalar_type,
        ScalarNumberType>::value>
    register_local_dof_values (ScratchData<dim>       &scratch_data,
                               const std::string      &global_vector_name,
                               const ScalarNumberType  scalar_dummy,
                               const ADNumberType      dummy);

    // Register the previously extracted local dof values of
    // type ScalarNumberType as variables for automatic
    // differentiation. Internally a vector of local dof values
    // of type ADNumberType is created and these are set as
    // ad-variables. This does the same as above, but is a
    // specialization for the case that AD::ADNumberTraits<ADNumberType>::scalar_type
    // and ScalarNumberType are the same. In this case a faster
    // implementation can be used.
    // ScratchDataTools::extract_local_dof_values must have
    // been called first for the number type ScalarNumberType.
    // ADNumberType must be an automatically differentiable number
    // type, i.e. AD::is_ad_number must return true.
    // ScalarNumberType must be a non-AD number, i.e. AD::is_ad_number
    // must return false.
    template <int dim, class ScalarNumberType, class ADNumberType>
    std::enable_if_t<std::is_same<typename AD::ADNumberTraits<ADNumberType>::scalar_type,
        ScalarNumberType>::value>
    register_local_dof_values (ScratchData<dim>       &scratch_data,
                               const std::string      &global_vector_name,
                               const ScalarNumberType  scalar_dummy,
                               const ADNumberType      dummy);

    // Register an energy functional. See the deal.II documentation
    // of dealii::Differentiation::AD::EnergyFunctional::register_energy_functional.
    // ScratchDataTools::extract_local_dof_values must have
    // been called first for the number type ADNumberType or local dof
    // values must have been registered for the number type ADNumberType
    // via ScratchDataTools::register_local_dof_values
    // ADNumberType must be an automatically differentiable number
    // type, i.e. AD::is_ad_number must return true.
    template <int dim, class ADNumberType>
    void
    register_energy_functional (ScratchData<dim>   &scratch_data,
                                const std::string  &global_vector_name,
                                const ADNumberType  dummy,
                                const ADNumberType &functional);

    // Register a residual vector. See the deal.II documentation
    // of dealii::Differentiation::AD::ResidualLinearization::register_residual.
    // ScratchDataTools::extract_local_dof_values must have
    // been called first for the number type ADNumberType or local dof
    // values must have been registered for the number type ADNumberType
    // via ScratchDataTools::register_local_dof_values
    // ADNumberType must be an automatically differentiable number
    // type, i.e. AD::is_ad_number must return true.
    template <int dim, class ADNumberType>
    void
    register_residual (ScratchData<dim>                &scratch_data,
                       const std::string               &global_vector_name,
                       const ADNumberType               dummy,
                       const std::vector<ADNumberType> &residual);

    // Compute the residual vector. See the deal.II documentation
    // of dealii::Differentiation::AD::CellLevelBase::compute_resiudal
    // and its implementations in the derived classes.
    // ScratchDataTools::extract_local_dof_values must have
    // been called first for the number type ADNumberType or local dof
    // values must have been registered for the number type ADNumberType
    // via ScratchDataTools::register_local_dof_values
    // ADNumberType must be an automatically differentiable number
    // type, i.e. AD::is_ad_number must return true.
    template <int dim, class ADNumberType>
    void
    compute_residual (ScratchData<dim>   &scratch_data,
                      const std::string  &global_vector_name,
                      const ADNumberType  dummy,
                      dealii::Vector<typename AD::ADNumberTraits<ADNumberType>::scalar_type> &residual);

    // Compute the linearization of the residual vector. See the deal.II documentation
    // of dealii::Differentiation::AD::CellLevelBase::compute_linearization
    // and its implementations in the derived classes.
    // ScratchDataTools::extract_local_dof_values must have
    // been called first for the number type ADNumberType or local dof
    // values must have been registered for the number type ADNumberType
    // via ScratchDataTools::register_local_dof_values
    // ADNumberType must be an automatically differentiable number
    // type, i.e. AD::is_ad_number must return true.
    template <int dim, class ADNumberType>
    void
    compute_linearization (ScratchData<dim>   &scratch_data,
                           const std::string  &global_vector_name,
                           const ADNumberType  dummy,
                           dealii::FullMatrix<typename AD::ADNumberTraits<ADNumberType>::scalar_type> &linearization);

namespace efi_internal {

    template <class Number = double>
    std::string
    get_unique_name (const std::string  &global_vector_name,
                     const std::string  &object_type,
                     const unsigned int  size,
                     const Number       &exemplar_number);


    template <class Extractor, class Number = double>
    std::string
    get_unique_name (const std::string  &global_vector_name,
                     const Extractor    &variable,
                     const std::string  &object_type,
                     const unsigned int  size,
                     const Number       &exemplar_number);


    template <typename Number = double>
    std::string
    get_unique_dofs_name(const std::string &global_vector_name,
                         const unsigned int size,
                         const Number &     exemplar_number);

}//namespace efi_internal

}//namespace ScratchDataTools



//------------------- INLINE AND TEMPLATE FUNCTIONS -------------------//



std::string
ScratchDataTools::
ad_helper_type_to_string (const ADHelperType type)
{
    switch(type)
    {
        case ADHelperType::residual_linearization : return "residual_linearization";
        case ADHelperType::energy_functional      : return "energy_functional";
        default : return "none";
    }
}



namespace ScratchDataTools {

// Usage of EFI_SCRATCH_DATA_ACCESS_TO macro:
// the macro creates the templated (template<int dim, class Number = double>)
// functions get_or_add_*, get_*, and  stores_* functions for the given fields, where
// * is determined by the first macro-argument.
// The get_or_add_*, get_* return a vector of size ScratchDataTools::n_quadrature_points
// with elements of the type given as second macro-argument. Note that the parentheses
// are mandatory. All functions the arguments take the following arguments (in the given
// order):
//
// \code{.cpp}
// ScratchData<dim>  &scratch_data,           // reference to scratch data
// const std::string &global_vector_name,     // name of the used global vector
// const Number       dummy = Number(0).      // dummy to determine the number type
// \endcode
//
// Note that the template argument Number is determined by type dummy which is double
// by default. See also the documentation at the EFI_SCRATCH_DATA_ACCESS_TO macro
// definition.
EFI_SCRATCH_DATA_ACCESS_TO (deformation_grads, (dealii::Tensor<2,dim,Number>))
EFI_SCRATCH_DATA_ACCESS_TO (jacobians, (Number))
EFI_SCRATCH_DATA_ACCESS_TO (inverse_deformation_grads, (dealii::Tensor<2,dim,Number>))
EFI_SCRATCH_DATA_ACCESS_TO (cauchy_stresses, (dealii::SymmetricTensor<2,dim,Number>))
EFI_SCRATCH_DATA_ACCESS_TO (cauchy_stress_tangents, (dealii::SymmetricTensor<4,dim,Number>))
EFI_SCRATCH_DATA_ACCESS_TO (kirchoff_stresses, (dealii::SymmetricTensor<2,dim,Number>))
EFI_SCRATCH_DATA_ACCESS_TO (kirchoff_stress_tangents, (dealii::SymmetricTensor<4,dim,Number>))
EFI_SCRATCH_DATA_ACCESS_TO (piola_kirchoff_stresses, (dealii::SymmetricTensor<2,dim,Number>))
EFI_SCRATCH_DATA_ACCESS_TO (piola_kirchoff_stress_tangents, (dealii::SymmetricTensor<4,dim,Number>))
EFI_SCRATCH_DATA_ACCESS_TO (piola_stresses, (dealii::Tensor<2,dim,Number>))
EFI_SCRATCH_DATA_ACCESS_TO (piola_stress_tangents, (dealii::Tensor<4,dim,Number>))
EFI_SCRATCH_DATA_ACCESS_TO (right_cauch_green_deformation_tensors, (dealii::SymmetricTensor<2,dim,Number>))
EFI_SCRATCH_DATA_ACCESS_TO (left_cauch_green_deformation_tensors, (dealii::SymmetricTensor<2,dim,Number>))
EFI_SCRATCH_DATA_ACCESS_TO (strain_energy_density, (Number))


EFI_SCRATCH_DATA_ACCESS_TO (piola_kirchoff_stress_tangents_iso, (dealii::SymmetricTensor<4,dim,Number>))
}



template <class CellIteratorType, int dim>
void
ScratchDataTools::
reinit (ScratchData<dim>       &scratch_data,
        const CellIteratorType &cell)
{
    scratch_data.reinit (cell);
}



template <class CellIteratorType, int dim>
void
ScratchDataTools::
reinit (ScratchData<dim>       &scratch_data,
        const CellIteratorType &cell,
        const unsigned int      face_no)
{
    scratch_data.reinit (cell, face_no);
}



template <int dim>
inline
unsigned int
ScratchDataTools::
dofs_per_cell (const ScratchData<dim> &scratch_data)
{
    return scratch_data.get_current_fe_values().dofs_per_cell;
}



template <int dim>
inline
unsigned int
ScratchDataTools::
n_quadrature_points (const ScratchData<dim> &scratch_data)
{
    return scratch_data.get_current_fe_values().n_quadrature_points;
}



template <int dim, class VectorType, class Number>
std::enable_if_t<AD::is_ad_number<Number>::value>
ScratchDataTools::
extract_local_dof_values (ScratchData<dim>  &scratch_data,
                          const std::string &global_vector_name,
                          const VectorType  &input_vector,
                          const Number       dummy)
{
    static_assert (std::is_same<typename AD::ADNumberTraits<Number>::scalar_type, typename VectorType::value_type>::value," ");

    auto &ad_helper = get_or_add_ad_helper (scratch_data,global_vector_name,dummy);

    // before we can set new variables, we have to reset the old
    // ad helper object.
    if (get_ad_helper_type (scratch_data,global_vector_name,dummy)
            == ADHelperType::residual_linearization)
        ad_helper.reset (dofs_per_cell(scratch_data),dofs_per_cell(scratch_data));
    else if (get_ad_helper_type (scratch_data,global_vector_name,dummy)
            == ADHelperType::energy_functional)
        ad_helper.reset (dofs_per_cell(scratch_data));
    else
        Assert(false,dealii::ExcMessage(
                "Set ADHerplerType does not allow automatic differentiation."
                "Use ScratchDataTools::set_ad_helper_type to set"
                "the ADHerplerType type to 'residual_linearization' or "
                "'energy_functional'."));

    ad_helper.register_dof_values (input_vector,scratch_data.get_local_dof_indices());

}



template <int dim, class VectorType, class Number>
inline
std::enable_if_t<!AD::is_ad_number<Number>::value>
ScratchDataTools::
extract_local_dof_values (ScratchData<dim>  &scratch_data,
                          const std::string &global_vector_name,
                          const VectorType  &input_vector,
                          const Number       dummy)
{
    scratch_data.extract_local_dof_values(global_vector_name,input_vector,dummy);
}



template <int dim, class Number>
inline
const std::enable_if_t<AD::is_ad_number<Number>::value,std::vector<Number>> &
ScratchDataTools::
get_local_dof_values (const ScratchData<dim> &scratch_data,
                      const std::string      &global_vector_name,
                      const Number            dummy)
{
    return get_ad_helper (scratch_data,global_vector_name,dummy).get_sensitive_dof_values ();
}



template <int dim, class Number>
inline
const std::enable_if_t<!AD::is_ad_number<Number>::value,std::vector<Number>> &
ScratchDataTools::
get_local_dof_values (const ScratchData<dim> &scratch_data,
                      const std::string      &global_vector_name,
                      const Number            dummy)
{
    return scratch_data.get_local_dof_values(global_vector_name,dummy);
}



template <int dim>
inline
const dealii::FEValuesBase<dim> &
ScratchDataTools::
get_current_fe_values (const ScratchData<dim> &scratch_data)
{
    return scratch_data.get_current_fe_values();
}



template <int dim>
inline
const std::vector<dealii::Point<dim>>&
ScratchDataTools::
get_quadrature_points (const ScratchData<dim> &scratch_data)
{
    return scratch_data.get_quadrature_points();
}



template <int dim>
inline
const std::vector<double>&
ScratchDataTools::
get_JxW_values (const ScratchData<dim> &scratch_data)
{
    return scratch_data.get_JxW_values();
}



template <int dim>
inline
const std::vector<dealii::Tensor<1,dim>>&
ScratchDataTools::
get_normal_vectors (const ScratchData<dim> &scratch_data)
{
    return scratch_data.get_normal_vectors();
}



template <int dim>
inline
const std::vector<dealii::types::global_dof_index>&
ScratchDataTools::
get_local_dof_indices (const ScratchData<dim> &scratch_data)
{
    return scratch_data.get_local_dof_indices();
}



template <int dim, class Extractor, class Number>
const std::enable_if_t<AD::is_ad_number<Number>::value,
    std::vector<typename ScratchDataTools::OutputType<dim,Number,Extractor>::value_type>
> &
ScratchDataTools::
get_values (ScratchData<dim>  &scratch_data,
            const std::string &global_vector_name,
            const Extractor   &variable,
            const Number       dummy)
{
    auto & fe = scratch_data.get_current_fe_values();

    const unsigned int n_q_points = fe.n_quadrature_points;

    const std::string name = efi_internal::get_unique_name(
            global_vector_name, variable, "_values_q", n_q_points, dummy);

    // Now build the return type
    using RetType = std::vector<typename ScratchDataTools::OutputType<dim,Number,Extractor>::value_type>;

    RetType &values = scratch_data.get_general_data_storage().template get_or_add_object_with_name<RetType>(name,fe.n_quadrature_points);

    AssertDimension(values.size(), n_q_points);

    fe[variable].get_function_values_from_local_dof_values (
            get_ad_helper (scratch_data,global_vector_name,dummy).get_sensitive_dof_values(),
            values);

    return values;
}



template <int dim, class Extractor, class Number>
inline
const std::enable_if_t<!AD::is_ad_number<Number>::value,
    std::vector<typename ScratchDataTools::OutputType<dim,Number,Extractor>::value_type>
> &
ScratchDataTools::
get_values (ScratchData<dim>  &scratch_data,
            const std::string &global_vector_name,
            const Extractor   &variable,
            const Number       dummy)
{
    return scratch_data.get_values (global_vector_name,variable,dummy);
}



template <int dim, class Extractor, class Number>
const std::enable_if_t<AD::is_ad_number<Number>::value,
    std::vector<typename ScratchDataTools::OutputType<dim,Number,Extractor>::gradient_type>
> &
ScratchDataTools::
get_gradients (ScratchData<dim>  &scratch_data,
               const std::string &global_vector_name,
               const Extractor   &variable,
               const Number       dummy)
{
    auto & fe = scratch_data.get_current_fe_values();

    const unsigned int n_q_points = fe.n_quadrature_points;

    const std::string name = efi_internal::get_unique_name(
            global_vector_name, variable, "_gradients_q", n_q_points, dummy);

    // Now build the return type
    using RetType = std::vector<typename ScratchDataTools::OutputType<dim,Number,Extractor>::gradient_type>;

    RetType &gradients = scratch_data.get_general_data_storage().template get_or_add_object_with_name<RetType>(name,fe.n_quadrature_points);

    AssertDimension(gradients.size(), n_q_points);

    fe[variable].get_function_gradients_from_local_dof_values (
            get_ad_helper (scratch_data,global_vector_name,dummy).get_sensitive_dof_values(),
            gradients);

    return gradients;
}



template <int dim, class Extractor, class Number>
inline
const std::enable_if_t<!AD::is_ad_number<Number>::value,
    std::vector<typename ScratchDataTools::OutputType<dim,Number,Extractor>::gradient_type>
> &
ScratchDataTools::
get_gradients (ScratchData<dim>  &scratch_data,
               const std::string &global_vector_name,
               const Extractor   &variable,
               const Number       dummy)
{
    return scratch_data.get_gradients (global_vector_name,variable,dummy);
}



template <int dim>
void
ScratchDataTools::
attach_history_data_storage (ScratchData<dim> &scratch_data,
                             GeneralCellDataStorage &cell_data_storage)
{
   scratch_data.get_general_data_storage().add_or_overwrite_reference ("history_data_storage",cell_data_storage);
}



template <int dim>
void
ScratchDataTools::
attach_tmp_history_data_storage (ScratchData<dim> &scratch_data,
                                     GeneralCellDataStorage &cell_data_storage)
{
   scratch_data.get_general_data_storage().add_or_overwrite_reference ("tmp_history_data_storage",cell_data_storage);
}



template <int dim>
dealii::GeneralDataStorage&
ScratchDataTools::
get_history_data (ScratchData<dim> &scratch_data)
{
    Assert (scratch_data.get_general_data_storage().stores_object_with_name("history_data_storage"),
            dealii::ExcMessage ("No object with name history_data_storage "
                                "stored in ScratchData::user_data_storage."));

    return scratch_data.get_general_data_storage().
            template get_object_with_name<GeneralCellDataStorage>
            ("history_data_storage").get_data(scratch_data.get_current_fe_values().get_cell());
}



template <int dim>
dealii::GeneralDataStorage&
ScratchDataTools::
get_tmp_history_data (ScratchData<dim> &scratch_data)
{
    Assert (scratch_data.get_general_data_storage().stores_object_with_name("tmp_history_data_storage"),
            dealii::ExcMessage ("No object with name tmp_history_data_storage "
                                "stored in ScratchData::user_data_storage."));

    return scratch_data.get_general_data_storage().
            template get_object_with_name<GeneralCellDataStorage>
            ("tmp_history_data_storage").get_data(scratch_data.get_current_fe_values().get_cell());
}



template <int dim>
double&
ScratchDataTools::
get_time_step_size (ScratchData<dim> &scratch_data)
{
    Assert (scratch_data.get_general_data_storage().stores_object_with_name("time_step_size"),
            dealii::ExcMessage ("No object with name time_step_size "
                                "stored in ScratchData::user_data_storage."));

    return scratch_data.get_general_data_storage().
            template get_object_with_name<double> ("time_step_size");
}


template <int dim>
double&
ScratchDataTools::
get_or_add_time_step_size (ScratchData<dim> &scratch_data)
{
    return scratch_data.get_general_data_storage().
            template get_or_add_object_with_name<double> ("time_step_size");
}



template <int dim, class ADNumberType>
AD::CellLevelBase<AD::ADNumberTraits<ADNumberType>::type_code,
    typename AD::ADNumberTraits<ADNumberType>::scalar_type> &
ScratchDataTools::
get_or_add_ad_helper (ScratchData<dim>   &scratch_data,
                      const std::string  &global_vector_name,
                      const ADNumberType  dummy)
{
    static_assert ( AD::is_ad_number<ADNumberType>::value,"Number is a non-ad-type.");

    const unsigned int n_dofs_per_cell = scratch_data.get_current_fe_values().dofs_per_cell;

    const std::string name = efi_internal::get_unique_dofs_name (global_vector_name,
                                scratch_data.get_current_fe_values().dofs_per_cell,
                                dummy);

    auto &holder = scratch_data.get_general_data_storage().
            template get_or_add_object_with_name<AD::CellLevelBaseHolder<ADNumberType>>
            (name);

    using residual_linearization_type =
            AD::ResidualLinearization<AD::ADNumberTraits<ADNumberType>::type_code,
            typename AD::ADNumberTraits<ADNumberType>::scalar_type>;

    using energy_functional_type =
            AD::EnergyFunctional<AD::ADNumberTraits<ADNumberType>::type_code,
            typename AD::ADNumberTraits<ADNumberType>::scalar_type>;

    // I'll fix that later
    if (!holder.held || !holder.held->content)
    {
        if (get_ad_helper_type (scratch_data,global_vector_name,dummy)
                == ADHelperType::residual_linearization)
            holder.template set<residual_linearization_type>(n_dofs_per_cell,n_dofs_per_cell);
        else if (get_ad_helper_type (scratch_data,global_vector_name,dummy)
                == ADHelperType::energy_functional)
            holder.template set<energy_functional_type>(n_dofs_per_cell);
        else
            Assert(false,dealii::ExcMessage(
                    "Set ADHerplerType does not allow automatic differentiation."
                    "Use ScratchDataTools::set_ad_helper_type to set"
                    "the ADHerplerType type to 'residual_linearization' or "
                    "'energy_functional'."));
    }
    // TODO The else-case when the object is initialized with
    //      the wrong ad_helper_type is not covered!!!
    //      Some callback would be nice whenever set_ad_helper_type
    //      is called.

    return holder.get();
}



template <int dim, class ADNumberType>
void
ScratchDataTools::
set_ad_helper_type (ScratchData<dim>   &scratch_data,
                    const std::string  &global_vector_name,
                    const ADNumberType  dummy,
                    const ADHelperType  helper_type)
{
    // Typically, dofs_per_cell is used as second argument to create the name,
    // however, the helper type is set independent of the number of dofs of the
    // current cell.
    const std::string name = efi_internal::get_unique_dofs_name (global_vector_name,
                                dealii::numbers::invalid_unsigned_int,
                                dummy);

    if ((helper_type == ADHelperType::residual_linearization)
            || (helper_type == ADHelperType::energy_functional))
        scratch_data.get_general_data_storage().
                template get_or_add_object_with_name<ADHelperType>
                (name) = helper_type;
    else
        AssertThrow(false,dealii::ExcMessage("AD helper-type <"
                +ad_helper_type_to_string(helper_type)+"> not defined."));
}



template <int dim, class ADNumberType>
ScratchDataTools::ADHelperType
ScratchDataTools::
get_ad_helper_type (ScratchData<dim> &scratch_data,
                    const std::string      &global_vector_name,
                    const ADNumberType      dummy)
{
    // Typically, dofs_per_cell is used as second argument to create the name,
    // however, the helper type is set independent of the number of dofs of the
    // current cell.
    const std::string name = efi_internal::get_unique_dofs_name (global_vector_name,
                                dealii::numbers::invalid_unsigned_int,
                                dummy);

    if (scratch_data.get_general_data_storage().stores_object_with_name(name))
        return scratch_data.get_general_data_storage().
                template get_object_with_name<ADHelperType>
                (name);
    else
        return ADHelperType::none;
}



template <int dim, class ADNumberType>
inline
AD::CellLevelBase<AD::ADNumberTraits<ADNumberType>::type_code,
    typename AD::ADNumberTraits<ADNumberType>::scalar_type> &
ScratchDataTools::
get_ad_helper (ScratchData<dim>   &scratch_data,
               const std::string  &global_vector_name,
               const ADNumberType  dummy)
{
    static_assert ( AD::is_ad_number<ADNumberType>::value,"Number is a non-ad-type.");

    const std::string name = efi_internal::get_unique_dofs_name (global_vector_name,
                                scratch_data.get_current_fe_values().dofs_per_cell,
                                dummy);

    Assert (scratch_data.get_general_data_storage().stores_object_with_name(name),
            dealii::ExcMessage ("No object with name " + name +
                                " stored in ScratchData::user_data_storage."));

    return scratch_data.get_general_data_storage().
            template get_object_with_name<AD::CellLevelBaseHolder<ADNumberType>>
            (name).get();
}



template <int dim, class ADNumberType>
inline
AD::CellLevelBase<AD::ADNumberTraits<ADNumberType>::type_code,
    typename AD::ADNumberTraits<ADNumberType>::scalar_type> &
ScratchDataTools::
get_ad_helper (const ScratchData<dim>  &scratch_data,
               const std::string       &global_vector_name,
               const ADNumberType       dummy)
{
    static_assert ( AD::is_ad_number<ADNumberType>::value,"Number is a non-ad-type.");

    const std::string name = efi_internal::get_unique_dofs_name (global_vector_name,
                                scratch_data.get_current_fe_values().dofs_per_cell,
                                dummy);

    Assert (scratch_data.get_general_data_storage().stores_object_with_name(name),
            dealii::ExcMessage ("No object with name " + name +
                                " stored in ScratchData::user_data_storage."));

    return scratch_data.get_general_data_storage().
            template get_object_with_name<AD::CellLevelBaseHolder<ADNumberType>>
            (name).get();
}



template <int dim, class ScalarNumberType, class ADNumberType>
std::enable_if_t<!std::is_same<typename AD::ADNumberTraits<ADNumberType>::scalar_type,
    ScalarNumberType>::value>
ScratchDataTools::
register_local_dof_values (ScratchData<dim>       &scratch_data,
                           const std::string      &global_vector_name,
                           const ScalarNumberType  scalar_dummy,
                           const ADNumberType      ad_dummy)
{
    static_assert (AD::is_ad_number<ADNumberType>::value,"ADNumberType is a non-ad-type.");

    static_assert (std::is_arithmetic<typename AD::ADNumberTraits<ScalarNumberType>::scalar_type>::value,"");
    static_assert (std::is_arithmetic<typename AD::ADNumberTraits<ADNumberType>::scalar_type>::value,"");
    static_assert (std::is_convertible<typename AD::ADNumberTraits<ScalarNumberType>::scalar_type,
                                       typename AD::ADNumberTraits<ADNumberType>::scalar_type>::value,"");

    const unsigned int dofs_per_cell = ScratchDataTools::dofs_per_cell(scratch_data);

    const std::string name = efi_internal::get_unique_dofs_name (
                                global_vector_name,
                                dofs_per_cell,
                                ad_dummy);

    // This object might not exist yet.
    auto &ad_helper = get_or_add_ad_helper (scratch_data,global_vector_name,ad_dummy);

    // before we can set new variables, we have to
    // reset the ad_helper object.
    ad_helper.reset (dofs_per_cell,dofs_per_cell);

    // Get the local dof values of type ScalarNumberType which
    // we want to register as ad-variables.
    auto &local_dof_values = get_local_dof_values(scratch_data,global_vector_name,scalar_dummy);

    std::vector<typename AD::ADNumberTraits<ADNumberType>::scalar_type> tmp_local_dof_values;
    tmp_local_dof_values.reserve (local_dof_values.size());

    for (auto &dof_value : local_dof_values)
        tmp_local_dof_values.push_back (AD::ADNumberTraits<ADNumberType>::get_scalar_value (dof_value));

    ad_helper.register_dof_values (tmp_local_dof_values);
}



template <int dim, class ScalarNumberType, class ADNumberType>
inline
std::enable_if_t<std::is_same<typename AD::ADNumberTraits<ADNumberType>::scalar_type,
    ScalarNumberType>::value>
ScratchDataTools::
register_local_dof_values (ScratchData<dim>       &scratch_data,
                           const std::string      &global_vector_name,
                           const ScalarNumberType  scalar_dummy,
                           const ADNumberType      ad_dummy)
{
    static_assert (AD::is_ad_number<ADNumberType>::value,"ADNumberType is a non-ad-type.");

    static_assert (std::is_arithmetic<typename AD::ADNumberTraits<ScalarNumberType>::scalar_type>::value,"");

    const unsigned int dofs_per_cell = ScratchDataTools::dofs_per_cell(scratch_data);

    const std::string name = efi_internal::get_unique_dofs_name (
                                global_vector_name,
                                dofs_per_cell,
                                ad_dummy);

    // This object might not exist yet.
    auto &ad_helper = get_or_add_ad_helper (scratch_data,global_vector_name,ad_dummy);

    // before we can set new variables, we have to
    // reset the ad_helper object.
    ad_helper.reset (dofs_per_cell,dofs_per_cell);

    ad_helper.register_dof_values (get_local_dof_values(scratch_data,global_vector_name,scalar_dummy));
}



template <int dim, class ADNumberType>
inline
void
ScratchDataTools::
register_energy_functional (ScratchData<dim>   &scratch_data,
                            const std::string  &global_vector_name,
                            const ADNumberType  dummy,
                            const ADNumberType &functional)
{
    static_assert (AD::is_ad_number<ADNumberType>::value, "ADNumberType is a non-ad-type.");

    Assert(get_ad_helper_type(scratch_data, global_vector_name,dummy)
            == ADHelperType::energy_functional,
            dealii::ExcMessage("Requested ad-helper of type <energy_functional> "
                               "(aka AD::EnergyFunctional) but ad-helper type is set to < "
                               + ad_helper_type_to_string(
                                       get_ad_helper_type(scratch_data, global_vector_name,dummy)) + ">."));

    auto &ad_helper = static_cast<AD::EnergyFunctional<AD::ADNumberTraits<ADNumberType>::type_code,
                                  typename AD::ADNumberTraits<ADNumberType>::scalar_type>&>(
           get_ad_helper (scratch_data, global_vector_name, dummy));

    ad_helper.register_energy_functional (functional);
}



template <int dim, class ADNumberType>
inline
void
ScratchDataTools::
register_residual (ScratchData<dim>                &scratch_data,
                   const std::string               &global_vector_name,
                   const ADNumberType               dummy,
                   const std::vector<ADNumberType> &residual)
{
    static_assert (AD::is_ad_number<ADNumberType>::value, "ADNumberType is a non-ad-type.");

    Assert(get_ad_helper_type(scratch_data, global_vector_name,dummy)
            == ADHelperType::residual_linearization,
            dealii::ExcMessage("Requested ad-helper of type <residual_linearization> "
                               "(aka AD::ResidualLinearization) but ad-helper type is set to < "
                               + ad_helper_type_to_string(
                                       get_ad_helper_type(scratch_data, global_vector_name,dummy)) + ">."));

    auto &ad_helper = static_cast<AD::ResidualLinearization<AD::ADNumberTraits<ADNumberType>::type_code,
                                  typename AD::ADNumberTraits<ADNumberType>::scalar_type>&>(
           get_ad_helper (scratch_data, global_vector_name, dummy));

    ad_helper.register_residual_vector (residual);
}



template <int dim, class ADNumberType>
inline
void
ScratchDataTools::
compute_residual (ScratchData<dim>   &scratch_data,
                  const std::string  &global_vector_name,
                  const ADNumberType  dummy,
                  dealii::Vector<typename AD::ADNumberTraits<ADNumberType>::scalar_type> &residual)
{
    static_assert ( AD::is_ad_number<ADNumberType>::value,"ADNumberType is a non-ad-type.");

    auto &ad_helper = get_ad_helper (scratch_data,global_vector_name, dummy);

    ad_helper.compute_residual (residual);
}



template <int dim, class ADNumberType>
inline
void
ScratchDataTools::
compute_linearization (ScratchData<dim>   &scratch_data,
                       const std::string  &global_vector_name,
                       const ADNumberType  dummy,
                       dealii::FullMatrix<typename AD::ADNumberTraits<ADNumberType>::scalar_type> &linearization)
{
    static_assert ( AD::is_ad_number<ADNumberType>::value,"ADNumberType is a non-ad-type.");

    auto &ad_helper = get_ad_helper (scratch_data, global_vector_name, dummy);

    ad_helper.compute_linearization (linearization);
}



template <class Number>
inline
std::string
ScratchDataTools::
efi_internal::
get_unique_name (const std::string  &global_vector_name,
                 const std::string  &object_type,
                 const unsigned int  size,
                 const Number       &exemplar_number)
{
    return global_vector_name + "_" + object_type +
           "_" + dealii::Utilities::int_to_string(size) + "_" +
           dealii::Utilities::type_to_string(exemplar_number);
}



template <class Extractor, class Number>
inline
std::string
ScratchDataTools::
efi_internal::
get_unique_name (const std::string  &global_vector_name,
                 const Extractor    &variable,
                 const std::string  &object_type,
                 const unsigned int  size,
                 const Number       &exemplar_number)
{
    return get_unique_name (global_vector_name,
                            variable.get_name() + "_" + object_type,
                            size,
                            exemplar_number);
}



template <class Number>
inline
std::string
ScratchDataTools::
efi_internal::
get_unique_dofs_name(const std::string  &global_vector_name,
                     const unsigned int  size,
                     const Number       &exemplar_number)
{
    return global_vector_name + "_independent_local_dofs_" +
           dealii::Utilities::int_to_string(size) + "_" +
           dealii::Utilities::type_to_string(exemplar_number);
}


}// namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_WORKER_SCRATCH_DATA_H_ */
