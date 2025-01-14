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

#ifndef SRC_MYLIB_INCLUDE_EFI_BASE_BOUNDARY_WORKER_H_
#define SRC_MYLIB_INCLUDE_EFI_BASE_BOUNDARY_WORKER_H_

// stl headers
#include <string>

// efi headers
#include <efi/base/extractor.h>
#include <efi/base/type_traits.h>
#include <efi/worker/copy_data.h>
#include <efi/worker/scratch_data.h>
#include <efi/worker/worker_base.h>


namespace efi {

/// This is the worker class which working on the boundary of our domain.
/// @author Stefan Kaessmair
template <int dim>
class BoundaryWorker : public WorkerBase<dim>
{
public:

    EFI_REGISTER_AS_BASE;

    /// Type of scalar numbers.
    using scalar_type = typename WorkerBase<dim>::scalar_type;
    using ad_type     = scalar_type;

    /// Default constructor;
    BoundaryWorker (const std::string &subsection_name = "boundary_worker");

    /// Default destructor.
    virtual
    ~BoundaryWorker ();

    /// This function reinitializes the @p FEFaceValues object, calls the
    /// @p DataProcessor::evaluate function and @p BoundaryWorker::do_fill.
    /// @param[in] data_processor provides the function to evaluate the
    /// @p scratch_data object and stores the results in @p scratch_data.
    /// @param[in] fe_function The global vector of unkonws, that describes the
    /// discrerte FE-function.
    /// @param[in] cell Cell iterator of the current cell.
    /// @param[in] face_no Number of the face of the current cell we're
    /// working on.
    /// @param[in,out] scratch_data Provides all data needed by the
    /// @p BoundaryWorker to evaluate the current face. Internally, it is also
    /// used by the @p data_processor which reads from and writes to the
    /// @p scratch_data.
    /// @param[out] copy_data Output data object for the local rhs, the local
    /// cell matrix and the local DoF indices. These objects are appended to
    /// the corresponding vectors in @p copy_data.
    /// @param[in] sparsity_only If the @p sparsity_only flag is set, then only
    /// the local DoF indices of the considered cell are written to
    /// @p copy_data.
    template <class DataProcessor, class VectorType, class CellIteratorType>
    void
    fill (const DataProcessor    &data_processor,
          const VectorType       &fe_function,
          const CellIteratorType &cell,
          const unsigned int      face_no,
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
    /// @param[in] scratch_data Provides all necessary data to compute the
    /// local rhs and cell matrix.
    /// @param[out] copy_data Output data object for the local rhs, the local
    /// cell matrix.
    virtual
    void
    do_fill (ScratchData<dim> &scratch_data,
             CopyData         &copy_data) const;
};




//---------------------- INLINE AND TEMPLATE FUNCTIONS -----------------------//



template <int dim>
inline
BoundaryWorker<dim>::
BoundaryWorker (const std::string &subsection_name)
:
    WorkerBase<dim>(subsection_name)
{ }



template <int dim>
inline
BoundaryWorker<dim>::
~BoundaryWorker ()
{ }



template <int dim>
template <class DataProcessor, class VectorType, class CellIteratorType>
inline
void
BoundaryWorker<dim>::
fill (const DataProcessor    &data_processor,
      const VectorType       &fe_function,
      const CellIteratorType &cell,
      const unsigned int      face_no,
      ScratchData<dim>       &scratch_data,
      CopyData               &copy_data) const
{
    // Check if DataProcessor provides an evaluate-function with
    // valid signature (void evaluate(ScratchData<dim> &) const).
    static_assert(has_member_function_evaluate<
            void(DataProcessor::*)(ScratchData<dim>&) const>::value,
            "DataProcessor must implement the member function "
            "<void evaluate(ScratchData<dim> &) const>");

    auto global_vector_name = Extractor<dim>::global_vector_name();

    ScratchDataTools::reinit (scratch_data, cell, face_no);
    ScratchDataTools::extract_local_dof_values (
            scratch_data, Extractor<dim>::global_vector_name(),
            fe_function, scalar_type());

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
    // const auto &face = cell->face(face_no);
    // // for (const auto &face: cell->face_iterators())
    // if (face->boundary_id() == 5)
    // {
    //     auto &fe_face = scratch_data.get_current_fe_values();
    //     const unsigned int dofs_per_face = 12;
    //     efilog(Verbosity::verbose) << "BoundaryWorker do_fill for face ";
    //     std::cout << cell->id().to_string() << " with " << dofs_per_face 
    //     << " dofs per face" << std::endl;
    //     std::vector<dealii::types::global_dof_index> vertex_dof_indices(dofs_per_face);

        
    //     face->get_dof_indices(vertex_dof_indices);
    //     for (auto & dof : vertex_dof_indices)
    //     {
    //         // std::cout << "Changing dof : " << dof << 
    //         // " from " << fe_function[dof] << " to 10000." << std::endl;
    //         // fe_function[dof] = 10000;
    //     } 
    // }   
        
}



template <int dim>
dealii::UpdateFlags
BoundaryWorker<dim>::
get_needed_update_flags () const
{
    return dealii::update_JxW_values | dealii::update_normal_vectors;
}

}// namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_BASE_BOUNDARY_WORKER_H_ */
