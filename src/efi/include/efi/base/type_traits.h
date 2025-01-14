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

#ifndef SRC_EFI_INCLUDE_EFI_BASE_TYPE_TRAITS_H_
#define SRC_EFI_INCLUDE_EFI_BASE_TYPE_TRAITS_H_

// dealii headers
#include <deal.II/meshworker/scratch_data.h>

// boost headers
#include <boost/tti/tti.hpp>


namespace efi
{

// static member data
BOOST_TTI_HAS_STATIC_MEMBER_DATA (dimension)
BOOST_TTI_HAS_STATIC_MEMBER_DATA (space_dimension)
BOOST_TTI_HAS_STATIC_MEMBER_DATA (rank)
BOOST_TTI_HAS_STATIC_MEMBER_DATA (registered_as_base)

// static member funcitons
BOOST_TTI_HAS_STATIC_MEMBER_FUNCTION (declare_parameters)

// member functions
BOOST_TTI_HAS_MEMBER_FUNCTION (evaluate)
BOOST_TTI_HAS_MEMBER_FUNCTION (get_needed_update_flags)
BOOST_TTI_HAS_MEMBER_FUNCTION (declare_parameters)


namespace efi_internal {

    /// Helper to get the correct type when the type in a preprocessor macro
    /// is wrapped in parentheses to avoid errors. For instance
    /// #define EFI_TYPE_WRAPPER (mytype) mytype
    /// fails when ScratchData<dim> is passed, because of '<' and '>' the
    /// passed type is not recognized as single token. Therefore, we use the
    /// #define EFI_TYPE_WRAPPER ((mytype))
    ///     typename get_type_from_macro_input<void type>::type
    /// which returns the exact type for EFI_TYPE_WRAPPER ((ScratchData<dim>))
    /// and does not result in a syntax error.
    template<class T> struct get_type_from_macro_input;

    template<class Ret, class Arg>
    struct get_type_from_macro_input<Ret(Arg)>
    {
        typedef Arg type;
    };
}

}

#endif /* SRC_EFI_INCLUDE_EFI_BASE_TYPE_TRAITS_H_ */
