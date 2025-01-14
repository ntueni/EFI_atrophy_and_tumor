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
#include <efi/lab/experiment.h>
#include <efi/lab/testing_device_factory.h>
#include <efi/base/global_parameters.h>

namespace efi {


template <int dim>
void
Experiment<dim>::
declare_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    prm.declare_entry("reset","false",Patterns::Bool(),
            "reset the solution vector after each testing device");

    prm.declare_entry ("output directory","",Patterns::Anything(),
            "options: set directory or leave blank for current directory");

    prm.declare_entry ("input directory","",Patterns::Anything(),
            "options: set directory or leave blank for current directory");

    prm.declare_entry ("verbosity console","normal",
            Patterns::Selection("quiet|normal|verbose|very verbose|debug"),
            "options: quiet|normal|verbose|very verbose|debug");

    prm.declare_entry ("verbosity logfile","normal",
            Patterns::Selection("quiet|normal|verbose|very verbose|debug"),
            "options: quiet|normal|verbose|very verbose|debug");

    efilog(Verbosity::verbose) << "Experiment finished declaring parameters."
                               << std::endl;
}



template <int dim>
void
Experiment<dim>::
parse_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    this->reset =
        prm.get_bool("reset");

    // Set the path to the output directory
    boost::filesystem::path output_directory =
        GlobalParameters::get_output_directory();
        
    // .back() == this->output_directory.separator?
    //                 prm.get("output directory")
    //                 + "efi_v" + version() + "_" + time_stamp()
    //               : prm.get("output directory")
    //                 + this->output_directory.separator
    //                 + "efi_v" + version() + "_" + time_stamp();

    // Create the directories in the given path that do not exist yet
    MPI::create_directories (output_directory,
                             this->mpi_communicator);
    
    efilog(Verbosity::verbose) << "Experiment finished parsing parameters."
                               << std::endl;
}



template <int dim>
void
Experiment<dim>::
parse_input (std::string const &filename)
{
    using namespace dealii;

    // Sample creator
    FactoryTools::action_type create_sample
    = [&] (const FactoryTools::Specifications &specs,
           const std::string                  &unprocessed_input) -> void
    {
        this->sample.reset (
                SampleFactory<dim>::create (this->get_section_path(),
                                            specs,
                                            unprocessed_input,
                                            this->mpi_communicator));
    };

    // Device creator
    FactoryTools::action_type create_testing_device
    = [&] (const FactoryTools::Specifications &specs,
           const std::string                  &unprocessed_input) -> void
    {
        this->devices.emplace (
                specs.get_integer("instance"),
                std::unique_ptr<TestingDevice<dim>>(TestingDeviceFactory<dim>::create (this->get_section_path(),
                                                   specs,
                                                   unprocessed_input,
                                                   this->mpi_communicator)));
    };


    // Put all actions into a map.
    std::map<std::string,FactoryTools::action_type> actions;

    actions[SampleFactory<dim>::keyword()]        = create_sample;
    actions[TestingDeviceFactory<dim>::keyword()] = create_testing_device;

    // Setup the Experiment<dim> object using an input
    // parameter file and the map of predefined
    // factories. Whenever ParameterTools::setup()
    // is able to parse a key in the input parameter
    // file that matches an entry in factories, then
    // the corresponding.
    FactoryTools::apply (actions,filename);

    // Parse the input file for the declared
    // parameters.
    ParameterAcceptor::initialize (filename);

    this->sample->initialize();

    // Write a copy of the used parameter file to the
    // output directory.
    if (MPI::is_root (this->mpi_communicator))
    {
        //std::string stripped_filename = filename.substr(0,filename.rfind('.'));

        boost::filesystem::path output_directory = 
            GlobalParameters::get_output_directory();

        // Just a shortcut for the separator
        // in the file path
        std::string sep (1,output_directory.separator);

        // Get the paths of the output files
        std::string path_prm  = output_directory.string() + sep + "parameter_file" + ".prm";
        std::string path_json = output_directory.string() + sep + "parameter_file" + ".json";

        // Open the filestreams
        std::ofstream output_txt  (path_prm);
        std::ofstream output_json (path_json);

        AssertThrow (output_txt.is_open(), ExcFileNotOpen(path_prm));
        AssertThrow (output_txt.is_open(), ExcFileNotOpen(path_json));

        // write the copies of the input parameter
        // file in *txt and *.json format to the
        // output folder.
        ParameterAcceptor::prm.print_parameters (output_txt,  ParameterHandler::OutputStyle::Text);
        ParameterAcceptor::prm.print_parameters (output_json, ParameterHandler::OutputStyle::JSON);

        // write to the logstream
        efilog(Verbosity::normal) << "Created file <" << path_prm  << ">." << std::endl;
        efilog(Verbosity::normal) << "Created file <" << path_json << ">." << std::endl;
    }

    // just some output
    efilog(Verbosity::verbose) << "Experiment finished parsing input file <" << filename << ">."<< std::endl;
}



// instantiations
template class Experiment<2>;
template class Experiment<3>;

}// namespace efi


