/*
 * Copyright (C) 2019 - 2020 by the emerging fields initiative 'Novel Biopolymer
 * Hydrogels for Understanding Complex Soft Tissue Biomechanics' of the FAU
 *
 * This file is part of the EFI library.
 *
 * The efi library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * Author: Stefan Kaessmair
 */

#ifndef SRC_MYLIB_INCLUDE_EFI_BASE_EXCEPTIONS_H_
#define SRC_MYLIB_INCLUDE_EFI_BASE_EXCEPTIONS_H_

#include <deal.II/base/exceptions.h>

namespace efi {

// Exception raised if a string is differently
// formatted as expected.
//
// arg1 is the string which causes the problem
// and arg2 may be used to give additional hints,
// e.g. what is expected.
DeclException2 (ExcWrongFormat,
        std::string,
        std::string,
        "The string <" + arg1 + "> is formatted incorrectly. "
        + arg2);


// Exception raised if a constructor is not available.
DeclExceptionMsg(ExcNotConstructible,
        "The object you requested can't be "
        "constructed. ");

}// namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_BASE_EXCEPTIONS_H_ */
