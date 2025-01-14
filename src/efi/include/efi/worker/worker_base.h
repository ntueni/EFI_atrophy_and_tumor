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

#ifndef SRC_MYLIB_INCLUDE_EFI_BASE_WORKER_BASE_H_
#define SRC_MYLIB_INCLUDE_EFI_BASE_WORKER_BASE_H_

// stl headers
#include <vector>

// deal.II headers
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

// efi headers
#include <efi/base/extractor.h>
#include <efi/base/utility.h>
#include <efi/base/automatic_differentiation.h>
#include <efi/base/postprocessor.h>
#include <efi/worker/copy_data.h>
#include <efi/worker/scratch_data.h>
#include <efi/factory/registry.h>


namespace efi
{


// ScratchData class for the use in the deal.ii MehsWorker
// framework. It stores everything needed to compute the
// cell contributions to the
template <int dim>
class WorkerBase
{

public:

    /// Dimension in which this object operates.
    static const unsigned int dimension = dim;

    /// Dimension of the space in which this object operates.
    static const unsigned int space_dimension = dim;

    /// Type of scalar numbers.
    using scalar_type = double;

    // Default constructor.
    WorkerBase (const std::string &name = "worker_base");

    // Copy constructor. Making use of the clone
    // function of the constitutive_model, the
    // copy constructor is able to provide a deep
    // copy of the the WorkerBase class.
    WorkerBase (const WorkerBase <dim> &);

    // Default destructor.
    virtual
    ~WorkerBase ();

    // Declare parameters.
    virtual
    void
    declare_parameters (dealii::ParameterHandler &prm);

    // Parse the parameters.
    virtual
    void
    parse_parameters (dealii::ParameterHandler& prm);

    /// Return which data has to be provided to compute the derived
    /// quantities. The flags returned here are the ones passed to
    /// the constructor of this class.
    virtual
    dealii::UpdateFlags
    get_needed_update_flags () const = 0;

protected:

    // The function is purely virtual and hence must be
    // implemented by the derived class.
    // This function does most of the work, it requires
    // the scratch_data to be (re)initialized
    // with the current cell. It fills the local_rhs and
    // the local_matrix of the CopyData object.
    virtual
    void
    do_fill (ScratchData<dim> &scratch_data,
             CopyData         &copy_data) const = 0;

    // Name of the corresponding subsection in the
    // parameter file.
    const std::string subsection;
};



///////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION
///////////////////////////////////////////////////////////////////////////////



template<int dim>
inline
WorkerBase<dim>::
WorkerBase (const std::string &subsection)
:
    subsection (subsection)
{ }



template<int dim>
inline
WorkerBase<dim>::
WorkerBase (const WorkerBase <dim> &worker)
:
  subsection (worker.subsection)
{}



template<int dim>
inline
WorkerBase<dim>::
~WorkerBase ()
{ }



template<int dim>
inline
void
WorkerBase<dim>::
declare_parameters (dealii::ParameterHandler &)
{
    // By default, do nothing.
}



template<int dim>
inline
void
WorkerBase<dim>::
parse_parameters (dealii::ParameterHandler &)
{
    // By default, do nothing.
}


}//namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_BASE_WORKER_BASE_H_ */
