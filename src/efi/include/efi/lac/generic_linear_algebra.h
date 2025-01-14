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

 * Author: Stefan Kaessmair
 */

#ifndef SRC_MYLIB_INCLUDE_EFI_LAC_GENERIC_LINEAR_ALGEBRA_H_
#define SRC_MYLIB_INCLUDE_EFI_LAC_GENERIC_LINEAR_ALGEBRA_H_

// deal.II headers
#include <deal.II/base/config.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>


namespace efi {

namespace LA {

#if defined(DEAL_II_WITH_TRILINOS)

    using namespace dealii::LinearAlgebraTrilinos;

    using dealii::TrilinosWrappers::PreconditionBlockJacobi;
    using dealii::TrilinosWrappers::SolverDirect;

#else
#  error DEAL_II_WITH_TRILINOS required
#endif

}//namespace LA
}//namespace efi


#endif /* SRC_MYLIB_INCLUDE_EFI_LAC_GENERIC_LINEAR_ALGEBRA_H_ */
