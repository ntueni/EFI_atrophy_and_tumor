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

#ifndef SRC_MYLIB_INCLUDE_EFI_BASE_CELL_SCRATCH_DATA_H_
#define SRC_MYLIB_INCLUDE_EFI_BASE_CELL_SCRATCH_DATA_H_

// stl headers
#include <string>

// efi headers
#include <efi/base/extractor.h>
#include <efi/base/type_traits.h>
#include <efi/worker/copy_data.h>
#include <efi/worker/scratch_data.h>
#include <efi/worker/worker_base.h>


namespace efi {

// This is an abstract class, it implements none of the
// purely virtual functions of MaterialModel. Consequently,
// you can't create an instance of this class.
// All member variables of the worker classes must guarantee
// that there are no data races during a parallel assembly
// process.
template <int dim>
class CellWorker : public WorkerBase<dim>
{

public:

    EFI_REGISTER_AS_BASE;

    /// Type of scalar numbers.
    using scalar_type = typename WorkerBase<dim>::scalar_type;
    using ad_type     = scalar_type;


    // Default constructor;
    CellWorker (const std::string &name = "cell worker");

    // Default destructor.
    virtual
    ~CellWorker ();

    /// This function reinitializes the @p FEValues object, calls the
    /// @p DataProcessor::evaluate function and @p CellWorker::do_fill.
    /// @param[in] data_processor provides the function to evaluate the
    /// @p scratch_data object and stores the results in @p scratch_data.
    /// @param[in] fe_function The global vector of unkonws, that describes the
    /// discrerte FE-function.
    /// @param[in] cell Cell iterator of the current cell.
    /// @param[in,out] scratch_data Provides all data needed by the
    /// @p CellWorker to evaluate the current cell. Internally, it is also
    /// used by the @p data_processor which reads from and writes to the
    /// @p scratch_data.
    /// @param[out] copy_data Output data object for the local rhs, the local
    /// cell matrix and the local DoF indices. These objects are appended to
    /// the corresponding vectors in @p copy_data.
    template <class DataProcessor, class VectorType, class CellIteratorType>
    void
    fill (const DataProcessor    &data_processor,
          const VectorType       &fe_function,
          const CellIteratorType &cell,
          ScratchData<dim>       &scratch_data,
          CopyData               &copy_data) const;

    /// Return which data has to be provided to compute the derived
    /// quantities. The flags returned here are the ones passed to
    /// the constructor of this class.
    virtual
    dealii::UpdateFlags
    get_needed_update_flags () const override;

protected:

    /// This functions computes the local rhs, and the local cell matrix
    /// and writes the results to @p copy_data object. The results are written
    /// to the @p back() of the corresponding vectors in @p copy_data.
    /// @param[in] scratch_data Provides all necessary data to compute the local
    /// rhs and cell matrix.
    /// @param[out] copy_data Output data object for the local rhs, the local
    /// cell matrix.
    virtual
    void
    do_fill (ScratchData<dim> &scratch_data,
             CopyData         &copy_data) const override;
};



// This is an abstract class, it implements none of the
// purely virtual functions of MaterialModel. Consequently,
// you can't create an instance of this class.
// All member variables of the worker classes must guarantee
// that there are no data races during a parallel assembly
// process.
template <int dim>
class CellWorkerAD : public WorkerBase<dim>
{

public:

    EFI_REGISTER_AS_BASE;

    /// Type of scalar numbers.
    using scalar_type = typename WorkerBase<dim>::scalar_type;
    using ad_type     = typename AD::NumberTraits<
                            double,AD::NumberTypes::sacado_dfad>::ad_type;

    // Default constructor;
    CellWorkerAD (const std::string &name = "cell worker");

    // Default destructor.
    virtual
    ~CellWorkerAD ();

    // This function reinitializes the FEValues object and
    // then calls the virtual compute_material_response () function.
    // It is not possible to combine this in one function
    // since virtual template functions are not allowed in c++.
    // The DataProcessor
    template <class DataProcessor, class VectorType, class CellIteratorType>
    void
    fill (const DataProcessor    &data_processor,
          const VectorType       &fe_function,
          const CellIteratorType &cell,
          ScratchData<dim>       &scratch_data,
          CopyData               &copy_data) const;

    /// Return which data has to be provided to compute the derived
    /// quantities. The flags returned here are the ones passed to
    /// the constructor of this class.
    virtual
    dealii::UpdateFlags
    get_needed_update_flags () const override;

protected:

    // This function does most of the work, it requires
    // the scratch_data to be (re)initialized
    // with the current cell. It fills the local_rhs and
    // the local_matrix of the CopyData object.
    virtual
    void
    do_fill (ScratchData<dim> &scratch_data,
             CopyData         &copy_data) const override;
};



//---------------------- INLINE AND TEMPLATE FUNCTIONS -----------------------//



template <int dim>
inline
CellWorker<dim>::
CellWorker (const std::string &name)
:
    WorkerBase<dim>(name)
{ }



template <int dim>
inline
CellWorker<dim>::
~CellWorker ()
{ }



template <int dim>
template <class DataProcessor, class VectorType, class CellIteratorType>
inline
void
CellWorker<dim>::
fill (const DataProcessor     &data_processor,
      const VectorType        &fe_function,
      const CellIteratorType  &cell,
      ScratchData<dim>        &scratch_data,
      CopyData                &copy_data) const
{
    // Check if DataProcessor provides an evaluate-function with
    // valid signature (void evaluate(ScratchData<dim> &) const).
    static_assert(has_member_function_evaluate<
            void(DataProcessor::*)(ScratchData<dim>&) const>::value,
            "DataProcessor must implement the member function "
            "<void evaluate(ScratchData<dim> &) const>");

    auto global_vector_name = Extractor<dim>::global_vector_name();

    ScratchDataTools::reinit(scratch_data,cell);

    ScratchDataTools::extract_local_dof_values (
            scratch_data, global_vector_name, fe_function, ad_type());

    // Get the number of dofs per cell.
    auto dofs_per_cell = ScratchDataTools::dofs_per_cell (scratch_data);

    // Add a new set of copy data objects
    // to the copy data containers.
    copy_data.emplace_back (dofs_per_cell);

    // Create an alias for the dof indices
    // in the CopyData object.
    auto &local_dof_indices = copy_data.local_dof_indices.back();

    // Copy the local_dof_indices.
    local_dof_indices = ScratchDataTools::get_local_dof_indices (scratch_data);

    // Compute the constitutive response
    // at the quadrature points.
    data_processor.evaluate (scratch_data); 
    // if (cell->id().to_string() == "90293_0:")
    // {
    //     std::cout << "Same cell as boundary 5" << std::endl;
    //     std::cout  << "solution here is at dof 228123 is " 
    //         << fe_function(228123) << std::endl;
    // }
    // Now, do the actual job.
    this->do_fill (scratch_data,copy_data);
}



template <int dim>
dealii::UpdateFlags
CellWorker<dim>::
get_needed_update_flags () const
{
    return dealii::update_JxW_values | dealii::update_gradients;
}



template <int dim>
inline
CellWorkerAD<dim>::
CellWorkerAD (const std::string &name)
:
    WorkerBase<dim>(name)
{ }



template <int dim>
inline
CellWorkerAD<dim>::
~CellWorkerAD ()
{ }



template <int dim>
template <class DataProcessor, class VectorType, class CellIteratorType>
inline
void
CellWorkerAD<dim>::
fill (const DataProcessor     &data_processor,
      const VectorType        &fe_function,
      const CellIteratorType  &cell,
      ScratchData<dim>        &scratch_data,
      CopyData                &copy_data) const
{
    // Check if DataProcessor provides an evaluate-function with
    // valid signature (void evaluate(ScratchData<dim> &) const).
    static_assert(has_member_function_evaluate<
            void(DataProcessor::*)(ScratchData<dim>&) const>::value,
            "DataProcessor must implement the member function "
            "<void evaluate(ScratchData<dim> &) const>");

    auto global_vector_name = Extractor<dim>::global_vector_name();

    scratch_data.reinit (cell);

    // TODO the ad_helper_type should be chosen based on the type of the
    // data_processor. If it only provides a strain energy function, the
    // energy_linearisation must be used if it provides stresses as well
    // residual_linearization is the ad_helper we should use.
    ScratchDataTools::set_ad_helper_type (
            scratch_data, global_vector_name, ad_type(), ScratchDataTools::residual_linearization);
    ScratchDataTools::extract_local_dof_values (
            scratch_data, global_vector_name, fe_function, ad_type());

    // Get the number of dofs per cell.
    auto dofs_per_cell = ScratchDataTools::dofs_per_cell (scratch_data);

    // Add a new set of copy data objects
    // to the copy data containers.
    copy_data.emplace_back (dofs_per_cell);

    // Create an alias for the dof indices
    // in the CopyData object.
    auto &local_dof_indices = copy_data.local_dof_indices.back();

    // Copy the local_dof_indices.
    local_dof_indices = ScratchDataTools::get_local_dof_indices (scratch_data);

    // Compute the constitutive response
    // at the quadrature points.
    data_processor.evaluate (scratch_data);

    // Now, do the actual job.
    this->do_fill (scratch_data,copy_data);
}



template <int dim>
dealii::UpdateFlags
CellWorkerAD<dim>::
get_needed_update_flags () const
{
    return dealii::update_JxW_values | dealii::update_gradients;
}

}// namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_BASE_CELL_SCRATCH_DATA_H_ */
