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
#include <efi/lab/retraction_expansion_tube.h>
#include <efi/worker/measure_data_worker.h>
#include <efi/base/global_parameters.h>

namespace efi {


template <int dim>
void
RetractionExpansionTube<dim>::
declare_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    prm.declare_entry ("input files","",
            Patterns::List (
                    Patterns::FileName (Patterns::FileName::input),
                    0,
                    Patterns::List::max_int_value,
                    ","));

    prm.declare_entry("diameter","diameter");

    using P = dealii::Patterns::Tools::Convert<Tensor<1,dim>>;        
    Point<dim> point;
    
    prm.declare_entry("center",P::to_string(point),*P::to_pattern(),"Documentation");
    prm.declare_entry("length","50",Patterns::Double());

    efilog(Verbosity::verbose) << "RetractionExpansionTube"
                                  " finished declaring parameters."
                               << std::endl;
}

template <int dim>
void
RetractionExpansionTube<dim>::
parse_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    std::vector<std::string> files =
            Utilities::split_string_list(prm.get ("input files"),',');

    this->diameter_column_name = prm.get("diameter");
    this->length = prm.get_double("length");

    using P = dealii::Patterns::Tools::Convert<Tensor<1,dim>>;
    this->center = P::to_value(prm.get("center"));

    boost::filesystem::path input_directory = GlobalParameters::get_input_directory();
    for (const auto &file : files){
        boost::filesystem::path full_path = input_directory / file;
        this->read_test_protocol (full_path.string(),this->diameter_column_name);
    }

    efilog(Verbosity::verbose) << "RetractionExpansionTube "
                                  "finished parsing parameters."
                               << std::endl;
}



template <int dim>
boost::signals2::connection
RetractionExpansionTube<dim>::
connect_constraints (Sample<dim> &sample) const
{
    using namespace dealii;

    return sample.signals.make_constraints.connect (
    [&sample]
    (AffineConstraints<typename Sample<dim>::scalar_type> &constraints)
    {
        auto &dof_handler = sample.get_dof_handler ();
        auto u_mask  = Extractor<dim>::displacement_mask();

        GetConstrainedBoundaryIDs constr_boundary_ids;

        sample.get_geometry().accept (constr_boundary_ids);

        for (auto id : {constr_boundary_ids.homogeneous,
                        constr_boundary_ids.inhomogeneousA,
                        constr_boundary_ids.inhomogeneousB})
            DoFTools::make_zero_boundary_constraints (
                    dof_handler, id, constraints, u_mask);
    });

}


template <int dim>
inline
void
RetractionExpansionTube<dim>::
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
        indata.data.emplace_back(time, displacement);
    }

    efilog(Verbosity::verbose) << "RetractionExpansionTube "
                                  "finished reading experimental "
                                  "data from file <"
                               << filename << ">."<< std::endl;
}


template <int dim>
inline
void
RetractionExpansionTube<dim>::
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

    this->boundary_set.set_size(dof_handler.n_dofs());

    GetConstrainedBoundaryIDs constr_boundary_ids;
    sample.get_geometry().accept(constr_boundary_ids);

    MeasureBoundaryForceWorker<dim> force_worker;
    force_worker.set_active (constr_boundary_ids.inhomogeneousA);

    Tensor<1,dim,double> force;
    auto force_copier = force_worker.create_copier (force);

    auto connection_force = sample.connect_boundary_loop (
                                       force_worker,
                                       force_copier,
                                       sample.signals.post_nonlinear_solve);

    std::map<types::global_dof_index, double> boundary_values;

    std::vector<double> times;
    std::vector<double> forces;
    std::vector<double> diameters;

    // When the prescribed displacement per step is too large, the algorithm
    // reduces the step width and tries to apply the the boundary conditions
    // in smaller steps. However, sometimes this wont help, therefore the number
    // of allowed refinements of the step width is limited to 5.
    unsigned int refinement_lvl = 0;

    
    this->boundary_set.clear();
    IndexSet tmp_boundary_set(this->boundary_set);

    
    for (InputData &input : this->input_data)
    {
        times.clear();
        forces.clear();
        diameters.clear();
        this->boundary_set.clear();

        double depth_of_tube =  this->center[0]-this->length;
        for (unsigned int step = 0; step < input.data.size(); ++step)
        {
            double scale_factor = 1;
            boundary_values.clear();
            // tmp_boundary_set = this->boundary_set;
            // Reset the force.
            force = 0;

            double time          = input.data[step].first;
            double previous_time = step>0? input.data[step-1].first : 0;

            Assert (time > previous_time - 1e-10,
                       ExcMessage("Negative time step size detected: "
                               "corrupted test protocol."));
	    
            // Compute the step size
            double dt = time-previous_time;
            // std::vector<scalar_type> valuesA(Extractor<dim>::n_components,0);
            // std::vector<scalar_type> valuesB(Extractor<dim>::n_components,0);

            // Note that the strain data is given in percent, therefore it must
            // be multiplied by the height of the sample.
            double diameter = input.data[step].second;
            efi::efilog(Verbosity::normal) << "Diameter of Tube: " << diameter << std::endl;
            
            
            // std::vector<std::pair<double,double>> scale_factor;
            // scale_factor.insert(std::make_pair<double,double>(0.,1.));
            // for (unsigned int radius_factor = 0; radius_factor < percentage_diameter.size(); ++radius_factor)
            // {

            // Testing for interaction with reatraction spatulars
            auto &fe = sample.get_fe();

            Quadrature<dim-1> face_quadrature(fe.get_unit_face_support_points());
            FEFaceValues<dim> fe_values_face(fe, face_quadrature, update_quadrature_points |
                                                update_normal_vectors |
                                                update_gradients);

            const unsigned int dofs_per_face = fe.n_dofs_per_face();
            const unsigned int n_face_q_points = face_quadrature.size();

            std::vector<types::global_dof_index> dof_indices(dofs_per_face);
            std::vector<bool> touched_dofs(dof_handler.n_dofs(),false);

            double x_value, y_value, z_value;
            for (const auto & cell : dof_handler.active_cell_iterators())
                if(!cell->is_artificial() && cell->at_boundary())
                    for (const auto & face : cell->face_iterators())
                        if (face->at_boundary())
                                if (face->boundary_id() == 501 || face->boundary_id() == 503)
                                {
                                    fe_values_face.reinit(cell, face);
                                    face->get_dof_indices(dof_indices);
                                    for (unsigned int q_point = 0; q_point<n_face_q_points; q_point += dim)
                                    {
                                        const int index = dof_indices[q_point];                            
                                        if (!touched_dofs[index])
                                        {
                                            touched_dofs[index] = true;
                                            dealii::Point<dim> support_pnt = fe_values_face.quadrature_point(q_point);
                                            Point<dim> current_point_disp;
                                            sample.get_slave_pnt(support_pnt,current_point_disp,index);
                                            
                                            x_value = current_point_disp[0];
                                            y_value = current_point_disp[1];
                                            z_value = current_point_disp[2];
                                            if (x_value >= depth_of_tube)
                                            {                                                
                                                double c0 = this->center[1];
                                                double c1 = this->center[2];
                                                double radius_1 = (diameter/2);
                                                double radius_0 = (diameter/2);
                                                double point_in_ellipse = ((z_value-c1)*(z_value-c1))/((radius_1)*(radius_1)) + 
                                                                            ((y_value-c0)*(y_value-c0))/((radius_0)*(radius_0));

                                                if ( point_in_ellipse <= 1.0 )
                                                {
                                                    double y0 = y_value;
                                                    double y1 = z_value;
                                                    // std::cout << "current tube diameter: " << radius_0 << std::endl;
                                                    // std::cout << " index : " << index << std::endl;
                                                    // std::cout << " y0 : " << y0 << ", y1 : " << y1 << std::endl;
                                                    std::vector<int> multipliers(2);
                                                    multipliers[0] = 1;
                                                    multipliers[1] = 1;
                                                    y0 = y0 - c0;
                                                    y1 = y1 - c1;
                                                    if (y0<0)
                                                    {
                                                        multipliers[0] = -1;
                                                    }
                                                    if (y1<0)
                                                    {
                                                        multipliers[1] = -1;
                                                    }
                                                    double x0, x1, distance;
                                                    distance = this->distance_point_ellipse(radius_0, radius_1, std::fabs(y0), std::fabs(y1), x0, x1);

                                                    x0 = (x0 * multipliers[0]) + c0;
                                                    x1 = (x1 * multipliers[1]) + c1;
                                                    
                                                    // std::cout << " x0 : " << x0 << ", x1 : " << x1 << std::endl;
                                                    
                                                    dealii::Point<dim> master_pnt;
                                                    master_pnt[0] = support_pnt[0];
                                                    master_pnt[1] = x0;//*radius_factor;
                                                    master_pnt[2] = x1;//*radius_factor;

                                                    Tensor<1,dim> inhom_value = master_pnt - support_pnt;

                                                    // std::cout << " dz: " << dz << std::endl;
                                                    // std::cout << " dy: " << dy << std::endl;
                                                    // std::cout << " master_pnt: " << master_pnt << std::endl;
                                                    // std::cout << " support_pnt: " << support_pnt << std::endl;
                                                    // std::cout << " INHOM_VALUE: " << inhom_value << std::endl;
                                                    // std::cout << " current_point: " << current_point_disp << std::endl;
                                                    Point<dim> new_point = current_point_disp;
                                                    for (int d = 1; d<dim; d++)
                                                    {
                                                        if (std::fabs(inhom_value[d]) > 1e-9){

                                                            // std::cout << " dof: " << index+d << ", dimension: " << d << std::endl;
                                                            double curr_difference = (current_point_disp[d]-support_pnt[d]);
                                                            // if (std::fabs(inhom_value[d]) < std::fabs(curr_difference))
                                                            // {
                                                            //     std::cout << " new difference: " << inhom_value[d] << std::endl;
                                                            //     std::cout << " current difference: " << curr_difference << std::endl;
                                                            // }
                                                            // if ( (std::fabs(difference) - std::fabs(inhom_value[d])) < 1e-8)
                                                            {
                                                                new_point[d] = support_pnt[d] + inhom_value[d];
                                                                // if ( ((index+d) != 2782) && ((index+d) != 2743))
                                                                // {
                                                                    boundary_values.insert(std::pair<dealii::types::global_dof_index,double>(index+d, inhom_value[d]));
                                                                    if (!this->boundary_set.is_element(index+d))
                                                                    {
                                                                        this->boundary_set.add_index(index+d);
                                                                    }
                                                                // }
                                                            }
                                                        }
                                                    }
                                                    // std::cout << " new_point: " << new_point << std::endl;
                                                    std::cout << std::endl;
                                                }                       
                                            }
                                        }
                                    }
                                }
        //     if (sample.run (boundary_values, dt, this->boundary_set))
        //     {
        //             // Test nothing
        //     }
        //     else
        //     {
        //         if (std::fabs(radius_factor - 1) < 1e-6)
        //         {
        //             // Insert a new intermediate step (linearly interpolated)
        //             scale_factor.insert(cale_factor.begin(),
        //                     std::make_pair (
        //                          0.5 * (scale_factor[0].first),
        //                          0.5 * (scale_factor[0].second)));
        //         }
        //         else
        //         {
        //             // TODO Use a higher order interpolation scheme to guess
        //             // intermediate values.
        //             // Insert a new intermediate step (linearly interpolated)
        //             scale_factor.insert(scale_factorbegin() + step,
        //                     std::make_pair (
        //                          0.5 * (scale_factor[step-1].first
        //                               + scale_factor[step].first),
        //                          0.5 * (scale_factor[step-1].second
        //                               + scale_factor[step].second)));
        //         }
        //     }
        // }                    

            // if (sample.run (boundary_values, dt, this->boundary_set))
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
                    diameters.push_back (input.data[step].second);
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
                this->boundary_set = tmp_boundary_set;
            }
        }
    

        if (MPI::is_root(this->mpi_communicator))
        {
            boost::filesystem::path infilepath  = input.filename;

            boost::filesystem::path outdir = GlobalParameters::get_output_directory();

            boost::filesystem::path outfilename
                = outdir / infilepath.filename();

            io::CSVWriter<3> out(outfilename.string());
            out.write_headers("time",this->diameter_column_name,"force");
            out.write_rows(times,diameters,forces);
        }

    }
    // Disconnect the remaining signals such that they do not interfere with
    // other testing devices that might be run afterwards or we might run
    // into other problems when e.g. the workers of the boundary loop go out
    // of scope.
    connection_constraints.disconnect();
    connection_force.disconnect();
}

template <int dim>
double
RetractionExpansionTube<dim>::
sqr( const double x)
{
    return x*x;    
}


template <int dim>
double
RetractionExpansionTube<dim>::
robust_length (const double v0, const double v1)
{
    double magV0 = std::fabs(v0);
    double magV1 = std::fabs(v1);
    if (magV0 > magV1)
        return magV0*std::sqrt(1+sqr(v1/v0));
    else
        return magV1*std::sqrt(1+sqr(v0/v1));    
}

template <int dim>
double
RetractionExpansionTube<dim>::
get_root (const double r0, const double z0, const double z1, double g)
{
    const unsigned int maxIter = 200;
    double n0 = r0*z0;
    double s0 = z1 - 1;
    double s1;
    if (g < 0)
        s1 = 0;
    else
    {
        s1 = this->robust_length(n0, z1) -1;
    }
    double s = 0;
    for (unsigned int i = 0; i < maxIter; i++)
    {
        s = (s0 + s1)/2;
        if (s == s0 || s == s1)
            break;
        double ratio0 = n0/(s + r0);
        double ratio1 = z1/(s + 1);
        g = sqr(ratio0) + sqr(ratio1)  - 1;
        if (g > 0)
            s0 = s;
        else if (g < 0)
            s1 = s;
        else    
            break;
        
        if (i == (maxIter-1))
        {
            std::cout << "Max num iterations is reached" << std::endl;
            std::cout << "g value: " << g << std::endl;
        }
    }
    return s;
}

template <int dim>
double
RetractionExpansionTube<dim>::
distance_point_ellipse (const double e0, const double e1, const double y0, const double y1, double &x0, double &x1)
{
    double distance = 0;
    if (y1 > 0)
    {
        if (y0 > 0)
        {
            double z0 = y0/e0;
            double z1 = y1/e1;
            double g = sqr(z0) + sqr(z1) - 1;
            if (g != 0)
            {
                double r0 = sqr(e0/e1);
                double sbar = this->get_root(r0, z0, z1, g);
                x0 = r0*y0/(sbar+ r0);
                x1 = y1/(sbar + 1);
                distance = std::sqrt(sqr(x0 - y0) + sqr(x1 - y1));
            }
            else
            {
                x0 = y0;
                x1 = y1;
                distance = 0;
            }
        } 
        else 
        {
            x0 = 0;
            x1  = e1;
            distance = std::fabs(y1-e1);
        }
    }
    else
    {
        double numer0 = e0*y0;
        double denom0 = sqr(e0) - sqr(e1);
        if (numer0 < denom0)
        {
            double xde0 = numer0/denom0;
            x0 = e0*xde0;
            x1 = e1*std::sqrt(1 - sqr(xde0));
            distance = std::sqrt(sqr(x0-y0) + sqr(x1));
        }
        else
        {
            x0 = e0;
            x1 = 0;
            distance = std::fabs(y0-e0);   
        }
    }

    return distance;
}

// instantiation
template class RetractionExpansionTube<2>;
template class RetractionExpansionTube<3>;

// Registration
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(RetractionExpansionTube,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(RetractionExpansionTube,3));
}// namespace efi

