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
#include <deal.II/base/tensor.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/physics/transformations.h>

// boost headers
#include <boost/signals2.hpp>

// efi headers
#include <efi/base/csv.h>
#include <efi/base/cloneable_function.h>
#include <efi/base/logstream.h>
#include <efi/constitutive/constitutive_base.h>
#include <efi/grid/geometry.h>
#include <efi/lab/atrophy.h>
#include <efi/worker/measure_data_worker.h>
#include <efi/base/global_parameters.h>

namespace efi {


template <int dim>
void
Atrophy<dim>::
declare_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    prm.declare_entry ("input files","",
            Patterns::List (
                    Patterns::FileName (Patterns::FileName::input),
                    0,
                    Patterns::List::max_int_value,
                    ","));

    prm.declare_entry("column name atrophy","atrophy");

    using P = dealii::Patterns::Tools::Convert<Tensor<1,dim>>;        
    Point<dim> point;    
    prm.declare_entry("center",P::to_string(point),*P::to_pattern(),"Documentation");

    prm.declare_entry("max radius","1",Patterns::Double(0));

    efilog(Verbosity::verbose) << "Atrophy finished declaring "
                                  "parameters."
                               << std::endl;
}



template <int dim>
void
Atrophy<dim>::
parse_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    std::vector<std::string> files =
            Utilities::split_string_list(prm.get ("input files"),',');

    std::string column_name_angle = prm.get("column name atrophy");

    this->concentration_radius = prm.get_double("max radius");

    using P = dealii::Patterns::Tools::Convert<Tensor<1,dim>>;
    this->concentration_center = P::to_value(prm.get("center"));

    this->input_data.clear();

    boost::filesystem::path input_directory = GlobalParameters::get_input_directory();
    for (const auto &file : files)
    {
        boost::filesystem::path full_path = input_directory / file;
        this->read_test_protocol (full_path.string(),column_name_angle);
    }

    efilog(Verbosity::normal) << "Atrophy center "
                               << this->concentration_center
                               << std::endl; 
    efilog(Verbosity::normal) << "Atrophy radius "
                               << this->concentration_radius
                               << std::endl; 
    efilog(Verbosity::verbose) << "Atrophy finished parsing "
                                  "parameters."
                               << std::endl;
}



template <int dim>
boost::signals2::connection
Atrophy<dim>::
connect_constraints (Sample<dim> &sample) const
{
    using namespace dealii;

    return sample.signals.make_constraints.connect (
    [&sample]
    (AffineConstraints<typename Sample<dim>::scalar_type> &constraints)
    {
        auto &dof_handler = sample.get_dof_handler ();
        auto u_mask  = Extractor<dim>::displacement_mask();

        GetConstrainedBoundaryIDs get_constr_boundary_ids;

        sample.get_geometry().accept (get_constr_boundary_ids);

        for (auto id : {get_constr_boundary_ids.homogeneous})
            DoFTools::make_zero_boundary_constraints (
                    dof_handler, id, constraints, u_mask);
    });
}



template <int dim>
inline
void
Atrophy<dim>::
read_test_protocol (const std::string &filename, const std::string &column_name_atrophy)
{
    using namespace dealii;

    Assert (boost::filesystem::exists(boost::filesystem::path(filename)),
            ExcFileNotOpen (filename));
    
    
    io::CSVReader<2> in (filename);

    in.read_header(io::ignore_extra_column,"time",column_name_atrophy);

    this->input_data.emplace_back();

    InputData& indata = this->input_data.back();
    indata.filename = filename;

    double time, atrophy;
    while (in.read_row (time, atrophy))
    {
        indata.data.emplace_back (time, atrophy);
    }

    efilog(Verbosity::verbose) << "Atrophy finished reading experimental "
                                  "data from file <"
                               << filename << ">."<< std::endl;
}




template <int dim>
inline
void
Atrophy<dim>::
run (Sample<dim> &sample)
{
    using namespace dealii;

    // Set the constraints.
    boost::signals2::connection connection_constraints =
            this->connect_constraints(sample);
    sample.reinit_constraints ();

    GetConstrainedBoundaryIDs constr_boundary_ids;
    sample.get_geometry().accept (constr_boundary_ids);

    MeasureBoundaryTorqueXWorker<dim> torque_worker;

    torque_worker.set_active (constr_boundary_ids.inhomogeneous);

    Tensor<1,dim,double> torque;
    auto torque_copier = torque_worker.create_copier (torque);

    auto connection_torque = sample.connect_boundary_loop (
                                       torque_worker,
                                       torque_copier,
                                       sample.signals.post_nonlinear_solve);

    std::map<types::global_dof_index, double> boundary_values;

    std::vector<double> times;
    std::vector<double> torques;
    std::vector<double> angles;

    auto& dof_handler = sample.get_dof_handler();
    auto& cell_data_history = sample.get_cell_data_history_storage();
    
    for (const auto &cell : dof_handler.active_cell_iterators()){
        dealii::Point<dim> center = cell->center();
        double radius = center.distance(this->concentration_center);
        double concentration = 0.;
        if (radius <= this->concentration_radius){
            concentration = 1 - (radius/this->concentration_radius);
        }
        double heaviSide =1.0/(1. + std::exp(-1*100*(concentration-0.5)));
        concentration = concentration *heaviSide;
        dealii::GeneralDataStorage& cell_data = cell_data_history.get_data(cell);
        cell_data.template add_or_overwrite_copy("concentration", concentration);
    }
    efilog(Verbosity::debug) << "CONCENTRATION SET" << std::endl;

    // for (const auto &cell : dof_handler.active_cell_iterators()){
    //     auto cell_data = cell_data_history.get_data(cell);
    //     double c = cell_data.template get_object_with_name<double>("concentration");
    //     if (c > 0.0){
    //         efilog(Verbosity::debug) << "CONCENTRATION: " << c << std::endl;
    //     }
    // }
    // When the prescribed displacement per step is too large, the algorithm
    // reduces the step width and tries to apply the the boundary conditions
    // in smaller steps. However, sometimes this wont help, therefore the number
    // of allowed refinements of the step width is limited to 5.
    unsigned int refinement_lvl = 0;

    for (InputData &input : this->input_data)
    {
        times.clear();
        angles.clear();
        torques.clear();

        for (unsigned int step = 0; step < input.data.size(); ++step)
        {
            torque = 0;
            boundary_values.clear(); 

            double time          = input.data[step].first;
            double previous_time = step>0? input.data[step-1].first : time;

            Assert (time > previous_time - 1e-10,
                    ExcMessage("Negative time step size detected: "
                               "corrupted test protocol."))

            // Compute the step size
            double dt = time-previous_time;

            double load = input.data[step].second;

            efi::efilog(Verbosity::normal) << "Total time: " << time << std::endl;

            if (sample.run(boundary_values, dt))
            {
                // When the refinement level is zero, then we have reached a
                // point in time for which experimental data is available.
                // Hence, to be able to compare our results with the
                // experimental data, we write the simulation results to the
                // output arrays.
                if (refinement_lvl == 0)
                {
                    // Write to data arrays
                    times.push_back (time);
                    angles.push_back (input.data[step].second);
                    torques.push_back (0.0);
                }

                if (refinement_lvl > 0)
                    --refinement_lvl;
            }
            else
            {
                if (step == 0)
                {
                    // Insert a new intermediate step (linearly interpolated)
                    input.data.insert(input.data.begin(),
                            std::make_pair (
                                 0.5 * (input.data[0].first),
                                 0.5 * (input.data[0].second)));
                }
                else
                {
                    // TODO Use a higher order interpolation scheme to guess
                    // intermediate values.
                    // Insert a new intermediate step (linearly interpolated)
                    input.data.insert(input.data.begin() + step,
                            std::make_pair (
                                 0.5 * (input.data[step-1].first
                                      + input.data[step].first),
                                 0.5 * (input.data[step-1].second
                                      + input.data[step].second)));
                }

                AssertThrow (++refinement_lvl < 100,
                        dealii::ExcMessage ("Time step refinement level > 5."));
                --step;
            }
        }

        if (MPI::is_root(this->mpi_communicator))
        {
            boost::filesystem::path infilename  = input.filename;

            boost::filesystem::path outdir = GlobalParameters::get_output_directory();

            boost::filesystem::path outfilename
                = //infilename.parent_path()
                    ///
                    outdir /
                    boost::filesystem::path(
                            infilename.stem().string());

            io::CSVWriter<3> out(outfilename.string()); // TODO parametrize this
            out.write_headers("time","angle","torque");
            out.write_rows(times,angles,torques);
        }
    }

//        // Open a gnuplot stream. Only the root process is allowed
//        // to send data.
//        GnuplotStream gnuplot (MPI::is_root(this->mpi_communicator));
//
//        gnuplot << "set term wxt noraise size 1000,400\n";
//        gnuplot << "set multiplot layout 1, 2 title 'shear test'\n";
//        gnuplot << "set bmargin 5\n";
//
//        gnuplot << "set title 'stress vs. shear'\n";
//        gnuplot << "set xlabel 'angle [rad]'\n";
//        gnuplot << "set ylabel 'torque'\n";
//        GnuplotStream::plot (gnuplot, angles, torques);
//
//        gnuplot << "set title 'stress vs. time'\n";
//        gnuplot << "set xlabel 'time [s]'\n";
//        gnuplot << "set ylabel 'torque'\n";
//        GnuplotStream::plot (gnuplot, times, torques);
//
//        gnuplot << "unset multiplot\n";
    

    // Disconnect the remaining signals such that they do not interfere with
    // other testing devices that might be run afterwards or we might run
    // into other problems when e.g. the workers of the boundary loop go out
    // of scope.
    connection_constraints.disconnect();
    connection_torque.disconnect();
}



// Instantiation
template class Atrophy<2>;
template class Atrophy<3>;

// Registration
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Atrophy,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Atrophy,3));
}
