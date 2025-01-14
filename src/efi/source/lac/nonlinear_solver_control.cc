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

// stl headers
#include <ios>
#include <iomanip>

// efi headers
#include <efi/lac/nonlinear_solver_control.h>
#include <efi/base/logstream.h>


namespace efi {


void
NonlinearSolverControl::
declare_parameters (dealii::ParameterHandler &prm)
{
    prm.declare_entry ("maximum number of iterations", "100",
            dealii::Patterns::Integer());
    prm.declare_entry ("absolute tolerance","1e-12",
            dealii::Patterns::Double(0.));
    prm.declare_entry ("relative tolerance","1e-8",
            dealii::Patterns::Double(0.));
    prm.declare_entry ("linear solver type","direct",
            dealii::Patterns::Selection("direct|CG|GMRES"),
            "options: direct|CG|GMRES");
}



void
NonlinearSolverControl::
parse_parameters (dealii::ParameterHandler &prm)
{
    this->set_max_steps(prm.get_integer ("maximum number of iterations"));
    this->set_tolerance(prm.get_double ("absolute tolerance"));
    this->set_reduction(prm.get_double ("relative tolerance"));

    this->linear_solver = prm.get ("linear solver type");
}



dealii::SolverControl::State
NonlinearSolverControl::
check (const unsigned int step,
       const double check_value)
{
    using namespace dealii;

    // if this is the first time we
    // come here, then store the
    // residual for later comparisons
    if (step == 0)
    {
      this->initial_val = check_value;
      this->reduced_tol = check_value * reduce;
    }

    if (this->m_log_history && ((step % this->m_log_frequency) == 0))
        efilog(Verbosity::normal) << "Check step  " << std::setw(6) << step
                                  << " value  " << std::setw(15) << check_value
                                  << " normalized value  " << std::setw(15)
                                  << (std::isfinite(check_value/this->initial_val)? check_value/this->initial_val : 1.)
                                  << std::endl;

    this->lstep  = step;
    this->lvalue = check_value;

    if (step == 0)
    {
        if (this->check_failure)
            this->failure_residual = this->relative_failure_residual * check_value;
    }

    if (this->history_data_enabled)
        this->history_data.push_back(check_value);

    // check whether desired reduction
    // has been achieved. also check
    // for equality in case initial
    // residual already was zero
    if (check_value <= reduced_tol)
    {
        if (this->m_log_result)
            efilog(Verbosity::normal) << "Convergence step  " << std::setw(6) << step
                                      << " value  " << std::setw(15) << check_value
                                      << " normalized value  " << std::setw(15)
                                      << (std::isfinite(check_value/this->initial_val)? check_value/this->initial_val : 1.)
                                      << std::endl;
        lcheck = success;
        return success;
    }

    if (check_value <= this->tol)
    {
        if (this->m_log_result)
            efilog(Verbosity::normal) << "Convergence step  " << std::setw(6) << step
                                      << " value  " << std::setw(15) << check_value
                                      << " normalized value  " << std::setw(15)
                                      <<  check_value/this->initial_val
                                      << std::endl;
        lcheck = success;
        return success;
    }

    if ((step >= maxsteps) || std::isnan(check_value) ||
        (check_failure && (check_value > failure_residual)))
    {
        if (m_log_result)
            efilog(Verbosity::normal) << "Failure step  " << std::setw(6) << step
                                      << " value  " << std::setw(15) << check_value
                                      << " normalized value  " << std::setw(15)
                                      <<  check_value/this->initial_val
                                      << std::endl;
        lcheck = failure;
        return failure;
    }

    lcheck = iterate;
    return iterate;
}



}//namespace efi
