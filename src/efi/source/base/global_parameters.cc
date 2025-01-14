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

// deal.II headers
#include <deal.II/base/logstream.h>

// efi headers
#include <efi/base/global_parameters.h>
#include <efi/base/utility.h>


namespace efi
{

GlobalParameters&
global_parameters(GlobalParameters* new_instance = nullptr)
{
    static thread_local std::unique_ptr<GlobalParameters> singleton;
    if (!singleton && new_instance)
        singleton.reset(new_instance);

    Assert(singleton, dealii::ExcNotInitialized());
    return *singleton;
}



boost::filesystem::path
GlobalParameters::
get_output_directory()
{
    Assert (global_parameters().parsed,
            dealii::ExcMessage("Parameters have not been parsed"));
    return global_parameters().output_path;
}

std::string
GlobalParameters::
get_output_filename()
{
    Assert (global_parameters().parsed,
            dealii::ExcMessage("Parameters have not been parsed"));
    return global_parameters().output_filename;
}

boost::filesystem::path
GlobalParameters::
get_input_directory()
{
    Assert (global_parameters().parsed,
            dealii::ExcMessage("Parameters have not been parsed"));
    return global_parameters().input_path;
}

bool
GlobalParameters::
paraview_output_enabled()
{
    Assert (global_parameters().parsed,
            dealii::ExcMessage("Parameters have not been parsed"));

    return  dealii::ParameterAcceptor::prm.get_bool(
                global_parameters().get_section_path(),
                "paraview output");
}

bool
GlobalParameters::
create_moved_mesh()
{
    Assert (global_parameters().parsed,
            dealii::ExcMessage("Parameters have not been parsed"));

    return  dealii::ParameterAcceptor::prm.get_bool(
                global_parameters().get_section_path(),
                "move mesh");
}

void
GlobalParameters::
declare_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    boost::filesystem::path output_path
        = boost::filesystem::current_path() / "out";
    prm.declare_entry("output directory",output_path.string(),
            Patterns::DirectoryName());

    // std::string output_filename
    //     = boost::filesystem::current_path() / "out";
    prm.declare_entry("output filename","dist-test-",
            Patterns::FileName());

    boost::filesystem::path input_path
        = boost::filesystem::current_path();
    prm.declare_entry("input directory",input_path.string(),
            Patterns::DirectoryName());

    prm.declare_entry ("verbosity console","normal",
            Patterns::Selection("quiet|normal|verbose|very verbose|debug"),
            "options: quiet|normal|verbose|very verbose|debug");

    prm.declare_entry ("verbosity logfile","normal",
            Patterns::Selection("quiet|normal|verbose|very verbose|debug"),
            "options: quiet|normal|verbose|very verbose|debug");

    prm.declare_entry ("paraview output","false",
            Patterns::Bool(),
            "Set to true if you want to write output for paraview.");

    prm.declare_entry ("move mesh","false",
            Patterns::Bool(),
            "Set to true if you want to visualize the displacement.");

}



void
GlobalParameters::
parse_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    this-> output_path = boost::filesystem::absolute(prm.get ("output directory"));
    this-> input_path  = boost::filesystem::absolute(prm.get ("input directory"));
    this-> output_filename = prm.get ("output filename");

    std::string verbosity_console = prm.get ("verbosity console");
    std::string verbosity_logfile = prm.get ("verbosity logfile");

    bool is_root = efi::MPI::is_root (this->mpi_communicator);

    // Make sure we define only one callback function.
    if (global_parameters().parsed)
        return;

    // FIXME Kaessmair: The lambda should not have static ofstream object!
    this->parse_parameters_call_back.connect(
    [this, verbosity_console, verbosity_logfile, is_root]()
    {
        // Only the root process will write the logfile.
        if (is_root)
        {
            if (!boost::filesystem::exists (this->output_path))
            {
                AssertThrow (
                        boost::filesystem::create_directories (this->output_path),
                        dealii::ExcMessage ("Error creating directories."));
            }

            std::string log_filename("efilab-" + this->output_filename + ".log");
            boost::filesystem::path path_to_logfile = output_path / log_filename;

            // the ofstream must be static to guarantee that is
            // lives until the program terminates.
            static std::ofstream logfile;

            // If there is another file open, detach and close it.
            if (dealii::deallog.has_file())
            {
                auto & out = static_cast<std::ofstream&>(
                        dealii::deallog.get_file_stream());
                dealii::deallog.detach();
                out.close();
            }

            // If another file was attached to the deallog we still want to
            // close logfile and open it again using the correct openmode.
            if (logfile.is_open())
                logfile.close();

            logfile.open(path_to_logfile.string(),
                    std::ios::out | std::ios::trunc);

            AssertThrow(logfile.is_open(),
                    ExcFileNotOpen(path_to_logfile.string()));

            deallog.depth_console(string_to_verbosity_level(verbosity_console));
            deallog.depth_file(string_to_verbosity_level(verbosity_logfile));

            dealii::deallog.attach (logfile);
            print_efi_header(logfile);
        }
        else
        {
            deallog.depth_console (Verbosity::quiet);
            deallog.depth_file    (Verbosity::quiet);
        }
    });

    global_parameters().parsed = true;
}



GlobalParameters::
GlobalParameters(MPI_Comm mpi_communicator)
: ParameterAcceptor("/global"),
  parsed(false),
  mpi_communicator(mpi_communicator)
{
    dealii::deallog.pop();
    dealii::deallog.push("EFI");

    // Until we've parsed the parameters, the root process streams to
    // efilog with Verbosity::normal.
    if (efi::MPI::is_root (this->mpi_communicator))
        dealii::deallog.depth_console (Verbosity::normal);
    else
        dealii::deallog.depth_console (Verbosity::quiet);

    print_efi_header (efilog(Verbosity::normal));
}



GlobalParameters::
~GlobalParameters()
{
    dealii::deallog << std::flush;
    if (dealii::deallog.has_file())
    {
        auto & out = static_cast<std::ofstream&>(
                dealii::deallog.get_file_stream());
        dealii::deallog.detach();
        out.close();
    }
}



void
init_global_parameters(const std::string &filename,
                       MPI_Comm mpi_communicator)
{
    static bool global_params_exist = false;
    if (!global_params_exist)
    {
        // Create global parameters
        global_parameters(new GlobalParameters(mpi_communicator));
        global_params_exist = true;
    }
    auto& global_params = global_parameters();

    auto& prm = dealii::ParameterAcceptor::prm;

    global_params.enter_my_subsection (prm);
    global_params.declare_parameters (prm);
    global_params.declare_parameters_call_back ();
    global_params.leave_my_subsection (prm);

    prm.parse_input (filename,"",true);

    global_params.enter_my_subsection (prm);
    global_params.parse_parameters (prm);
    global_params.leave_my_subsection (prm);
    global_params.parse_parameters_call_back ();
}

}// namespace efi


