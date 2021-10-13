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

#ifndef SRC_MYLIB_INCLUDE_EFI_LAC_NONLINEAR_SOLVER_CONTROL_H_
#define SRC_MYLIB_INCLUDE_EFI_LAC_NONLINEAR_SOLVER_CONTROL_H_

// stl headers
#include <string>

// deal.II headers
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/lac/solver_control.h>

// efi headers
#include <efi/base/utility.h>


namespace efi {


class NonlinearSolverControl : public dealii::ParameterAcceptorProxy<dealii::ReductionControl>
{
public:

    // Default constructor
    NonlinearSolverControl (const std::string &subsection_name,
                            const std::string &unprocessed_input = "");

    // Destructor
    virtual
    ~NonlinearSolverControl () override = default;

    virtual
    State
    check (const unsigned int step,
           const double check_value);

    // Declare parameters for the parameter
    // handler
    virtual
    void
    declare_parameters (dealii::ParameterHandler &prm) override;

    // Parse the parameter handler.
    virtual
    void
    parse_parameters (dealii::ParameterHandler &prm) override;

    // Return the linear solver type
    std::string
    get_linear_solver_type () const;

protected:

    // Type of the linear solver
    std::string linear_solver;
};



///////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION
///////////////////////////////////////////////////////////////////////////////



inline
NonlinearSolverControl::
NonlinearSolverControl (const std::string &subsection_name,
                        const std::string &)
: dealii::ParameterAcceptorProxy<dealii::ReductionControl>(subsection_name),
  linear_solver ("direct")
{
    this->log_frequency (1);
    this->log_history (true);
    this->log_result (true);
}


inline
std::string
NonlinearSolverControl::
get_linear_solver_type () const
{
    return this->linear_solver;
}

}//namespace efi


#endif /* SRC_MYLIB_INCLUDE_EFI_LAC_NONLINEAR_SOLVER_CONTROL_H_ */
