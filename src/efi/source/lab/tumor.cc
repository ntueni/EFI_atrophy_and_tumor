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
#include <efi/lab/tumor.h>
#include <efi/worker/measure_data_worker.h>
#include <efi/base/global_parameters.h>

namespace efi {


template <int dim>
void
Tumor<dim>::
declare_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    prm.declare_entry ("input files","",
            Patterns::List (
                    Patterns::FileName (Patterns::FileName::input),
                    0,
                    Patterns::List::max_int_value,
                    ","));

    prm.declare_entry("column name displacement","displacement");

    efilog(Verbosity::verbose) << "Tumor growth finished declaring "
                                  "parameters."
                               << std::endl;
}



template <int dim>
void
Tumor<dim>::
parse_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    std::vector<std::string> files =
            Utilities::split_string_list(prm.get ("input files"),',');

    this->input_data.clear();

    this->column_name_displacement = prm.get("column name displacement");

    boost::filesystem::path input_directory = GlobalParameters::get_input_directory();

    for (const auto &file : files)
    {
        boost::filesystem::path fullpath = input_directory / file; 
        this->read_test_protocol (fullpath.string(),this->column_name_displacement);
    }
        

    efilog(Verbosity::verbose) << "Tumor growth finished parsing "
                                  "parameters."
                               << std::endl;
}



template <int dim>
boost::signals2::connection
Tumor<dim>::
connect_constraints (Sample<dim> &sample) const
{
    using namespace dealii;

    return sample.signals.make_constraints.connect (
    [&sample](AffineConstraints<typename Sample<dim>::scalar_type> &constraints)
    {
        auto &dof_handler = sample.get_dof_handler ();
        auto u_mask  = Extractor<dim>::displacement_mask();

        GetConstrainedBoundaryIDs get_constr_boundary_ids;

        sample.get_geometry().accept (get_constr_boundary_ids);

        // for (auto id : {get_constr_boundary_ids.homogeneous,
        //                 get_constr_boundary_ids.inhomogeneous})
        //     DoFTools::make_zero_boundary_constraints (
        //             dof_handler, id, constraints, u_mask);

        DoFTools::make_zero_boundary_constraints (
                dof_handler, get_constr_boundary_ids.homogeneous, 
                constraints, u_mask);

    //     std::vector<bool> selectorX (Extractor<dim>::n_components,false);
    //     selectorX[Extractor<dim>::first_displacement_component] = true;
    //     dealii::ComponentMask inhom_mask_x (selectorX);
    //     DoFTools::make_zero_boundary_constraints (
    //             dof_handler, 400, constraints, inhom_mask_x);

    //     std::vector<bool> selectorY (Extractor<dim>::n_components,false);
    //     selectorY[Extractor<dim>::first_displacement_component+1] = true;
    //     dealii::ComponentMask inhom_mask_y (selectorY);
    //     DoFTools::make_zero_boundary_constraints (
    //             dof_handler, 401, constraints, inhom_mask_y);

    //     std::vector<bool> selectorZ (Extractor<dim>::n_components,false);
    //     selectorZ[Extractor<dim>::first_displacement_component+2] = true;
    //     dealii::ComponentMask inhom_mask_z (selectorZ);
    //     DoFTools::make_zero_boundary_constraints (
    //             dof_handler, 402, constraints, inhom_mask_z);
    });
}



template <int dim>
inline
void
Tumor<dim>::
read_test_protocol (const std::string &filename,const std::string & column_name_displacement)
{
    using namespace dealii;

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
    efilog(Verbosity::verbose) << "Tumor growth finished reading "
                                  "experimental data from file <"
                               << filename << ">."<< std::endl;
}



template <int dim>
inline
void
Tumor<dim>::
run (Sample<dim> &sample)
{
    using namespace dealii;

    // Set the constraints (no tumor boundary constraints now)
    boost::signals2::connection connection_constraints =
            this->connect_constraints(sample);
    sample.reinit_constraints ();

    // Force measurement setup (unchanged)
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

    // Store boundary normal vectors and growth load
    std::map<types::global_dof_index, Vector<double>> boundary_normal;
    double current_growth_load = 0.0;
    
    // Get boundary normals (keep existing code)
    auto &dof_handler = sample.get_dof_handler ();
    auto &fe = sample.get_fe();
    Quadrature<dim-1> face_quadrature(fe.get_unit_face_support_points());
    FEFaceValues<dim> fe_values_face(fe, face_quadrature, update_quadrature_points | update_normal_vectors);

    const unsigned int dofs_per_face = fe.n_dofs_per_face();
    const unsigned int n_face_q_points = face_quadrature.size();

    std::vector<types::global_dof_index> dof_indices(dofs_per_face);
    
    // Calculate boundary normals (keep existing normal calculation code)
    for (const auto & cell : dof_handler.active_cell_iterators())
        if(!cell->is_artificial() && cell->at_boundary())
            for (const auto & face : cell->face_iterators())
                if (face->at_boundary())
                    if (face->boundary_id() == constr_boundary_ids.inhomogeneous)
                    {
                        fe_values_face.reinit(cell, face);
                        face->get_dof_indices(dof_indices);
                        for (unsigned int q_point = 0; q_point<n_face_q_points; q_point += dim)
                        {
                            const int index = dof_indices[q_point];
                            auto face_normal = fe_values_face.normal_vector(q_point);
                            auto position = boundary_normal.find(index);
                            if (position == boundary_normal.end()){
                                Vector<double> normal(dim);
                                for (int d =0; d<dim; d++){
                                    normal[d] = face_normal[d];
                                }
                                boundary_normal.insert(std::pair<dealii::types::global_dof_index,Vector<double>>(index, normal));
                            } else {
                                Vector<double> normal = position->second;
                                for (int d =0; d<dim; d++){
                                    normal(d) += face_normal[d];
                                }
                                position->second = normal;
                            }
                        }
                    }
                                
    // Normalize the boundary normals
    for (auto iter = boundary_normal.begin(); iter != boundary_normal.end(); iter++){
        double norm = iter->second.l2_norm();
        for (int d =0; d<dim; d++){
            iter->second(d) = iter->second(d)/norm;
        }
    }

    // Store boundary normals and growth info in sample for access during assembly
    // You'll need to add these as member variables or pass them through ScratchData
    sample.set_tumor_boundary_normals(boundary_normal);
    sample.set_tumor_boundary_id(constr_boundary_ids.inhomogeneous);

    std::vector<double> times;
    std::vector<double> forces;
    std::vector<double> displacements;

    unsigned int refinement_lvl = 0;

    for (InputData &input : this->input_data)
    {
        times.clear();
        forces.clear();
        displacements.clear();

        for (unsigned int step = 0; step < input.data.size(); ++step)
        {
            force = 0;

            double time = input.data[step].first;
            double previous_time = step>0? input.data[step-1].first : time;

            Assert (time > previous_time - 1e-20,
                    ExcMessage("Negative time step size detected."));

            double dt = time - previous_time;
            double amount_of_growth = input.data[step].second;
            
            // Store the current growth load for use in assembly
            current_growth_load = amount_of_growth;
            sample.set_tumor_growth_load(current_growth_load);

            efilog(Verbosity::normal) << "Growth load: " << amount_of_growth << std::endl;

            // No boundary values to set - growth is now handled as body force
            std::map<dealii::types::global_dof_index,double> empty_boundary_values;

            if (sample.run (empty_boundary_values, dt, amount_of_growth))
            {
                if (refinement_lvl == 0)
                {
                    // Store results for output
                    times.push_back (time);
                    displacements.push_back (input.data[step].second);
                    forces.push_back (
                            Utilities::MPI::sum(force[0], // assuming growth in x-direction
                                             this->mpi_communicator));
                }

                if (refinement_lvl > 0)
                    --refinement_lvl;
            }
            else
            {
                // Handle failed convergence with time step refinement
                if (step == 0)
                {
                    input.data.insert(input.data.begin(),
                            std::make_pair (
                                 0.5 * (input.data[0].first),
                                 0.5 * (input.data[0].second)));
                }
                else
                {
                    input.data.insert(input.data.begin() + step,
                            std::make_pair (
                                 0.5 * (input.data[step-1].first
                                      + input.data[step].first),
                                 0.5 * (input.data[step-1].second
                                      + input.data[step].second)));
                }

                AssertThrow (++refinement_lvl < 10,
                        dealii::ExcMessage ("Time step refinement level > 10."));
                --step;
            }
        }

        // Output results (unchanged)
        if (MPI::is_root(this->mpi_communicator))
        {
            boost::filesystem::path infilepath = input.filename;
            boost::filesystem::path outdir = GlobalParameters::get_output_directory();
            boost::filesystem::path outfilename = outdir / infilepath.filename();

            io::CSVWriter<3> out(outfilename.string());
            out.write_headers(this->column_name_displacement,"force","time");
            out.write_rows(displacements,forces,times);
        }
    }

    connection_constraints.disconnect();
    connection_force.disconnect();
}

// Instantiation
template class Tumor<2>;
template class Tumor<3>;

// Registration
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Tumor,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Tumor,3));
}


