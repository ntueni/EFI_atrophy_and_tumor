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
#ifdef DEBUG
#include <boost/filesystem.hpp>
#endif

// efi headers
#include <efi/base/csv.h>
#include <efi/base/gnuplot_stream.h>
#include <efi/base/logstream.h>
#include <efi/constitutive/constitutive_base.h>
#include <efi/lab/tension_compression_testing_device.h>
#include <efi/worker/measure_data_worker.h>
#include <efi/base/global_parameters.h>

namespace efi {


template <int dim>
void
TensionCompressionTestingDevice<dim>::
declare_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    prm.declare_entry ("input files","",
            Patterns::List (
                    Patterns::FileName (Patterns::FileName::input),
                    0,
                    Patterns::List::max_int_value,
                    ","));

    prm.declare_entry ("uniaxial","false", Patterns::Bool ());
    prm.declare_entry ("direction","1.0, 0.0, 0.0",
                      Patterns::List (Patterns::Double(),
                              dim, 3, ",")); 

    prm.declare_entry("column name displacement","displacement");

    efilog(Verbosity::verbose) << "TensileCompressiveTestingDecive"
                                  " finished declaring parameters."
                               << std::endl;
}



template <int dim>
void
TensionCompressionTestingDevice<dim>::
parse_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    std::vector<std::string> files =
            Utilities::split_string_list(prm.get ("input files"),',');

    this->column_name_displacement = prm.get("column name displacement");

    boost::filesystem::path input_directory = GlobalParameters::get_input_directory();
    for (const auto &file : files){
        boost::filesystem::path full_path = input_directory / file;
        this->read_test_protocol (full_path.string(),this->column_name_displacement);
    }
        

    this->is_uniaxial = prm.get_bool("uniaxial");

    this->direction = Utilities::string_to_double (
                        Utilities::split_string_list(
                            prm.get("direction"),','));
    
    double direction_magnitude =  0.;
    for (unsigned int d = 0; d<dim; d++)
        direction_magnitude += this->direction[d]*this->direction[d];
    direction_magnitude = std::sqrt(direction_magnitude);
    for (unsigned int d = 0; d<dim; d++)
        this->direction[d] = this->direction[d]/direction_magnitude;
    efilog(Verbosity::verbose) << "Direction of load: ";
    for (unsigned int d = 0; d<dim; d++)
        efilog(Verbosity::verbose) << this->direction[d] << ", ";
    efilog(Verbosity::verbose) << std::endl;

    efilog(Verbosity::verbose) << "TensionCompressionTestingDecive "
                                  "finished parsing parameters."
                               << std::endl;
}



template <int dim>
boost::signals2::connection
TensionCompressionTestingDevice<dim>::
connect_constraints (Sample<dim> &sample) const
{
    using namespace dealii;

    if (this->is_uniaxial)
    {
        return this->connect_constraints_uniaxial (sample);
    }
    else
    {
        return sample.signals.make_constraints.connect (
        [&sample]
        (AffineConstraints<typename Sample<dim>::scalar_type> &constraints)
        {
            auto &dof_handler = sample.get_dof_handler ();
            auto u_mask  = Extractor<dim>::displacement_mask();

            GetConstrainedBoundaryIDs constr_boundary_ids;

            sample.get_geometry().accept (constr_boundary_ids);

            for (auto id : {constr_boundary_ids.homogeneous,
                            constr_boundary_ids.inhomogeneous})
                DoFTools::make_zero_boundary_constraints (
                        dof_handler, id, constraints, u_mask);
        });
    }

}



template <int dim>
boost::signals2::connection
TensionCompressionTestingDevice<dim>::
connect_constraints_uniaxial (Sample<dim> &sample) const
{
    using namespace dealii;

    return sample.signals.make_constraints.connect (
    [&sample]
    (AffineConstraints<typename Sample<dim>::scalar_type> &constraints)
    {
        auto &dof_handler = sample.get_dof_handler ();
        auto &mapping = sample.get_mapping ();

        std::vector<bool> selector(Extractor<dim>::n_components,false);
        selector[Extractor<dim>::first_displacement_component] = true;

        GetConstrainedBoundaryIDs constr_boundary_ids;
        sample.get_geometry().accept (constr_boundary_ids);

        IsReflectionSymmetric refelction_symmetry;
        sample.get_geometry().accept (refelction_symmetry);

        for (auto id : {constr_boundary_ids.homogeneous,
                        constr_boundary_ids.inhomogeneous})
            DoFTools::make_zero_boundary_constraints (
                    dof_handler, id, constraints,
                    dealii::ComponentMask(selector));

        std::vector<IndexSet> selected_dofs(dim);
        std::vector<std::map<types::global_dof_index, Point<dim>>>
            support_points (dim);

        std::array<unsigned int,dim> constraints_per_spacedim;

        for (int d = 1; d < dim; ++d)
        {
            Assert (refelction_symmetry.values[d],
                    ExcMessage("Geometry must be symmetric to the plane "
                               "with normal in direction <"
                               + Utilities::int_to_string(d)+ ">."));

            constraints_per_spacedim[d] = 0;

            selector.clear();
            selector.resize(Extractor<dim>::n_components,false);
            selector[Extractor<dim>::first_displacement_component+d] = true;

            map_dofs_to_support_points (mapping,
                                        dof_handler,
                                        dealii::ComponentMask(selector),
                                        support_points[d]);

            extract_locally_relevant_dofs (dof_handler,
                                           selected_dofs[d],
                                           dealii::ComponentMask(selector));

            for (auto dof : selected_dofs[d])
            {
                Assert (support_points[d].find(dof) != support_points[d].end(),
                        dealii::ExcMessage ("Point not found."));

                if (fabs(support_points[d][dof][d]) < 1e-10)
                {
                    Assert (constraints.can_store_line(dof),
                            dealii::ExcMessage (
                                    "Constraint matrix can't store line <" +
                                    Utilities::int_to_string(dof) + ">"));

                    constraints.add_line (dof);

                    // Increase the counter
                    ++constraints_per_spacedim[d];
                }
            }

            Assert (constraints_per_spacedim[d]>1,
                    ExcMessage ("Rigid body motion not suppressed in "
                                " direction <" + Utilities::int_to_string(d)
                                + ">. Use a mesh which has vertex dofs at "
                                "the coordinate planes y=0 and z=0."));
        }
    });
}



template <int dim>
inline
void
TensionCompressionTestingDevice<dim>::
read_test_protocol (const std::string &filename,const std::string& column_name_displacement)
{
    using namespace dealii;

    Assert (boost::filesystem::exists(boost::filesystem::path(filename)),
            ExcFileNotOpen (filename));

    io::CSVReader<2> in (filename);

    in.read_header(io::ignore_extra_column,"time",column_name_displacement);

    this->input_data.emplace_back();

    InputData& indata = this->input_data.back();
    indata.filename = filename;

    double time, displacement;
    while (in.read_row (time, displacement))
    {
        indata.data.emplace_back (time, displacement);
    }

    efilog(Verbosity::verbose) << "TensionCompressionTestingDecive "
                                  "finished reading experimental "
                                  "data from file <"
                               << filename << ">."<< std::endl;
}



template <int dim>
inline
void
TensionCompressionTestingDevice<dim>::
run (Sample<dim> &sample)
{
    using namespace dealii;

    // Sets zero dirichlet the constraints
    boost::signals2::connection connection_constraints =
            this->connect_constraints (sample);
    sample.reinit_constraints ();

    std::map<dealii::types::global_dof_index,double> prescribed;

    auto &dof_handler = sample.get_dof_handler ();
    auto &mapping     = sample.get_mapping ();

    GetConstrainedBoundaryIDs constr_boundary_ids;
    sample.get_geometry().accept (constr_boundary_ids);

    MeasureBoundaryForceWorker<dim> force_worker;
    force_worker.set_active (constr_boundary_ids.inhomogeneous);

    Tensor<1,dim,double> force;
    auto force_copier = force_worker.create_copier (force);

    auto connection_force = sample.connect_boundary_loop (
                                       force_worker,
                                       force_copier,
                                       sample.signals.post_nonlinear_solve);

    std::map<types::global_dof_index, double> boundary_values;

    std::vector<double> times;
    std::vector<double> forces;
    std::vector<double> displacements;

    // When the prescribed displacement per step is too large, the algorithm
    // reduces the step width and tries to apply the the boundary conditions
    // in smaller steps. However, sometimes this wont help, therefore the number
    // of allowed refinements of the step width is limited to 5.
    unsigned int refinement_lvl = 0;

    
    for (InputData &input : this->input_data)
    {
        times.clear();
        forces.clear();
        displacements.clear();

        for (unsigned int step = 0; step < input.data.size(); ++step)
        {
            // Reset the force.
            force = 0;

            double time          = input.data[step].first;
            double previous_time = step>0? input.data[step-1].first : time;

            Assert (time > previous_time - 1e-10,
                       ExcMessage("Negative time step size detected: "
                               "corrupted test protocol."))
	    
            // Compute the step size
            double dt = time-previous_time;

            std::vector<scalar_type> values(Extractor<dim>::n_components,0);

            // Note that the strain data is given in percent, therefore it must
            // be multiplied by the height of the sample.
            double displacement = input.data[step].second;
            efilog(Verbosity::normal) << "total displacement: " << displacement << std::endl;
            // Calculate displacement normal to the surface and add these values to the 'values' vector
            for (unsigned int d =0; d<dim; ++d)
            {
                values[Extractor<dim>::first_displacement_component+d] =
                    displacement*this->direction[d];
            }
            
            Functions::ConstantFunction<dim> boundary_function (values);

            std::vector<bool> selector (Extractor<dim>::n_components,false);
            efilog(Verbosity::normal) << "vectorized displacement: ";
            for (unsigned int d = 0; d < (this->is_uniaxial? 1 : dim); ++d)
                {
                    selector[Extractor<dim>::first_displacement_component+d] = true;
                    efilog(Verbosity::normal) << values[d] << ", ";
                }
            efilog(Verbosity::normal) << std::endl;

            dealii::ComponentMask u_mask (selector);
            VectorTools::interpolate_boundary_values (
                    mapping, dof_handler, constr_boundary_ids.inhomogeneous,
                    boundary_function, boundary_values,
                    u_mask);

            if (sample.run (boundary_values, dt))
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
                    displacements.push_back (input.data[step].second);
                    forces.push_back (
                        Utilities::MPI::sum(force[0], this->mpi_communicator));
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

                AssertThrow (++refinement_lvl < 6,
                        dealii::ExcMessage ("Time step refinement level > 5."));
                --step;
            }
        }
    

	if (MPI::is_root(this->mpi_communicator))
	{
	    boost::filesystem::path infilepath  = input.filename;

        boost::filesystem::path outdir = GlobalParameters::get_output_directory();

        std::string output_filename = 
        GlobalParameters::get_output_filename();

	    boost::filesystem::path outfilename
    		= outdir / infilepath.filename() ;

        std::string force_output_name(outfilename.string() + "_" + output_filename);

	    io::CSVWriter<3> out(force_output_name);
	    out.write_headers("time",this->column_name_displacement,"force");
	    out.write_rows(times,displacements,forces);
	}

    }
    // Disconnect the remaining signals such that they do not interfere with
    // other testing devices that might be run afterwards or we might run
    // into other problems when e.g. the workers of the boundary loop go out
    // of scope.
    connection_constraints.disconnect();
    connection_force.disconnect();
}




// instantiation
template class TensionCompressionTestingDevice<2>;
template class TensionCompressionTestingDevice<3>;

// Registration
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(TensionCompressionTestingDevice,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(TensionCompressionTestingDevice,3));
}// namespace efi

