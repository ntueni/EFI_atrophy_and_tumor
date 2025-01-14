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

#ifndef SRC_MYLIB_INCLUDE_EFI_IMPLEMENTATIONS_VISCO_VIRTUAL_LAB_H_
#define SRC_MYLIB_INCLUDE_EFI_IMPLEMENTATIONS_VISCO_VIRTUAL_LAB_H_

// stl headers
#include <string>

//boost header
#include <boost/filesystem.hpp>

// deal.II headers
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/path_search.h>

// efi headers
#include <efi/base/utility.h>
#include <efi/base/logstream.h>
#include <efi/lab/testing_device.h>
#include <efi/lab/sample.h>
#include <efi/factory/registry.h>


namespace efi {


template <int dim>
class Experiment : public dealii::ParameterAcceptor
{
public:

    EFI_REGISTER_AS_BASE;

    /// Dimension in which this object operates.
    static const unsigned int dimension = dim;

    /// Dimension of the space in which this object operates.
    static const unsigned int space_dimension = dim;

    /// Type of scalar numbers.
    using scalar_type = double;

    // Constructor
    Experiment (const std::string &subsection_name = "/experiment",
                MPI_Comm mpi_communicator = MPI_COMM_WORLD) noexcept;

    // Destructor
    ~Experiment ();

    // declare parameters
    void
    declare_parameters (dealii::ParameterHandler &prm) override;

    // parse parameters
    void
    parse_parameters (dealii::ParameterHandler &prm) override;

    // parse the input file
    void
    parse_input (std::string const &filename);

    // Run the experiment
    void
    run ();

private:

    // MPI communicator
    MPI_Comm mpi_communicator;

    // pointer to the sample
    std::unique_ptr<Sample<dim>> sample;

    /// Store the devices in a ordered map. The key is the number of
    /// the @p TestingDevice. The number is specified in the input file in
    /// the section header <tt>subsection
    /// testing_device@[type=...,instance=num]</tt>. The number @p num
    /// must be convertible to @p int.
    std::map<int, std::unique_ptr<TestingDevice<dim>>, std::less<int>> devices;

    bool reset;
};



//---------------------- INLINE AND TEMPLATE FUNCTIONS -----------------------//



template <int dim>
inline
Experiment<dim>::
Experiment (const std::string &subsection_name,
            MPI_Comm mpi_communicator) noexcept
:
dealii::ParameterAcceptor (subsection_name),
mpi_communicator (mpi_communicator)
{
    using namespace dealii;

    deallog.pop();
    deallog.push("EFI");

    if (efi::MPI::is_root (this->mpi_communicator))
        deallog.depth_console (Verbosity::normal);
    else
        deallog.depth_console (Verbosity::quiet);

    print_efi_header (efilog(Verbosity::normal));

    efilog(Verbosity::verbose) << "New Experiment created ("
                               << subsection_name
                               << ")." << std::endl;
}


template <int dim>
inline
Experiment<dim>::
~Experiment ()
{
    using namespace dealii;

    efilog(Verbosity::debug) << "Experiment deleted" << std::endl;
    
    this->sample.reset(NULL);
    
    deallog << std::flush;

    if (deallog.has_file())
    {
        auto & out = static_cast<std::ofstream&>(deallog.get_file_stream());

        deallog.detach();
        out.close();
    }
}


template <int dim>
inline
void
Experiment<dim>::
run ()
{
    for (auto &device : this->devices)
    {
        if(this->reset)
            this->sample->reset();
        device.second->run(*(this->sample));
    }
}

}// namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_IMPLEMENTATIONS_VISCO_VIRTUAL_LAB_H_ */
