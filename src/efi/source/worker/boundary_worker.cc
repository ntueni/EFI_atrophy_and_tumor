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
#include <efi/worker/boundary_worker.h>
#include <efi/factory/registry.h>


using namespace dealii;

namespace efi
{

template <int dim>
void
BoundaryWorker<dim>::
do_fill (ScratchData<dim> & /*scratch_data*/,
         CopyData         & /*copy_data*/) const
{
//    Assert (false, dealii::ExcNotImplemented());
}


// Instantiation
template class BoundaryWorker<2>;
template class BoundaryWorker<3>;

// Registration
EFI_REGISTER_OBJECT (EFI_TEMPLATE_CLASS (BoundaryWorker,2));
EFI_REGISTER_OBJECT (EFI_TEMPLATE_CLASS (BoundaryWorker,3));

}//namespace efi


