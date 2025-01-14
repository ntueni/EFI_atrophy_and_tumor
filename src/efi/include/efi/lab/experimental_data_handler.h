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

#ifndef SRC_EFI_INCLUDE_EFI_LAB_EXPERIMENTAL_DATA_HANDLER_H_
#define SRC_EFI_INCLUDE_EFI_LAB_EXPERIMENTAL_DATA_HANDLER_H_

// deal.II headers
#include <deal.II/base/parameter_acceptor.h>

// efi headers
#include <efi/base/csv.h>


namespace efi
{


class
ExperimentalDataHandler : public dealii::ParameterAcceptor
{
public:

    /// Declare parameters to the given parameter handler.
    /// @param prm[out] The parameter handler for which we want to declare
    /// the parameters.
    void
    declare_parameters (dealii::ParameterHandler &prm) final;

    /// Parse the parameters stored in the given parameter handler.
    /// @param prm[in] The parameter handler whose parameters we want to
    /// parse.
    void
    parse_parameters (dealii::ParameterHandler &prm) final;



private:

    std::vector<std::string> column_names;
};


}// namespace efi

#endif /* SRC_EFI_INCLUDE_EFI_LAB_EXPERIMENTAL_DATA_HANDLER_H_ */
