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

#ifndef SRC_EFI_INCLUDE_EFI_BASE_GLOBAL_PARAMETERS_H_
#define SRC_EFI_INCLUDE_EFI_BASE_GLOBAL_PARAMETERS_H_

// stl headers
#include <string>

// deal.II headers
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parameter_handler.h>

// boost headers
#include <boost/filesystem.hpp>
#include <boost/noncopyable.hpp>


namespace efi
{

/// Initialize the global parameters.
/// @todo Rename this function to <code>efi_init</code> as it initializes
/// all static datastructure such as the logstream and not only the
/// @p GlobalParameters.
/// @warning This function should must be called only once.
void
init_global_parameters(const std::string &filename,
                       MPI_Comm mpi_communicator = MPI_COMM_WORLD);

/// Accessor to global parameters in the global parameters handler
/// @p dealii::ParameterAccessor::prm
/// @author Stefan Kaessmair
class
GlobalParameters: public dealii::ParameterAcceptor,
                  private boost::noncopyable
{
public:

    /// Return the output directory as string
    static
    boost::filesystem::path
    get_output_directory ();

    /// Return the output directory as string
    static
    std::string
    get_output_filename ();

    // boost::filesystem::path
    // get_output_directory ();

    static
    boost::filesystem::path
    get_input_directory ();

    /// Return true of false indicating whether we want to generate paraview
    /// output or not.
    static
    bool
    paraview_output_enabled ();

    /// Return true of false indicating whether we want to visualize the displacement
    /// output or not.
    static
    bool
    create_moved_mesh ();

    /// Destructor.
    ~GlobalParameters ();

protected:

    /// Declare parameters to the given parameter handler.
    /// @param prm[out] The parameter handler for which we want to declare
    /// the parameters.
    void
    declare_parameters (dealii::ParameterHandler &prm) override;

    /// Parse the parameters stored int the given parameter handler.
    /// @param prm[in] The parameter handler whose parameters we want to
    /// parse.
    void
    parse_parameters (dealii::ParameterHandler &prm) override;

private:

    // Make this function a friend to get acces to the protected member
    // functions of the dealii::ParameterAcceptor.
    friend
    void
    init_global_parameters (const std::string &filename,
                            MPI_Comm mpi_communicator);

    /// Constructor.
    GlobalParameters (MPI_Comm mpi_communicator);

    /// Flag indicating whether @p parse_parameters has been called.
    bool parsed;

    boost::filesystem::path output_path;
    boost::filesystem::path input_path;
    std::string output_filename;

    /// MPI communicator
    MPI_Comm mpi_communicator;
};


}// namespace efi

#endif /* SRC_EFI_INCLUDE_EFI_BASE_GLOBAL_PARAMETERS_H_ */
