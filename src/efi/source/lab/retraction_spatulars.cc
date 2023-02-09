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
#include <efi/lab/retraction_spatulars.h>
#include <efi/worker/measure_data_worker.h>
#include <efi/base/global_parameters.h>

namespace efi {


template <int dim>
void
RetractionSpatulars<dim>::
declare_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    prm.declare_entry ("input files","",
            Patterns::List (
                    Patterns::FileName (Patterns::FileName::input),
                    0,
                    Patterns::List::max_int_value,
                    ","));

    using P = dealii::Patterns::Tools::Convert<Tensor<1,dim>>;        
    Point<dim> point;

    // Spatular parameters
    prm.declare_entry("directionA", "0.0,0.0,-1.0",
                      Patterns::List (Patterns::Double(),
                              dim, 3, ",")); 
    prm.declare_entry("directionB", "0.0,0.0,1.0",
                      Patterns::List (Patterns::Double(),
                              dim, 3, ","));

    prm.declare_entry("displacementA","displacementA");
    prm.declare_entry("displacementB","displacementB");

    efilog(Verbosity::verbose) << "TensileCompressiveTestingDecive"
                                  " finished declaring parameters."
                               << std::endl;
}

template <int dim>
void
RetractionSpatulars<dim>::
parse_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    std::vector<std::string> files =
            Utilities::split_string_list(prm.get ("input files"),',');

    this->retractor_displacementA = prm.get("displacementA");
    this->retractor_displacementB = prm.get("displacementB");

    // Spatular location

    this->directionA = Utilities::string_to_double (
                Utilities::split_string_list(
                        prm.get("directionA"),','));
    this->directionB = Utilities::string_to_double (
                Utilities::split_string_list(
                        prm.get("directionB"),','));

    boost::filesystem::path input_directory = GlobalParameters::get_input_directory();
    for (const auto &file : files){
        boost::filesystem::path full_path = input_directory / file;
        this->read_test_protocol (full_path.string(),this->retractor_displacementA, this->retractor_displacementB);
    }

    efilog(Verbosity::verbose) << "RetractionSpatulars "
                                  "finished parsing parameters."
                               << std::endl;
}



template <int dim>
boost::signals2::connection
RetractionSpatulars<dim>::
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

        for (auto id : {constr_boundary_ids.homogeneous})
            DoFTools::make_zero_boundary_constraints (
                    dof_handler, id, constraints, u_mask);
        
        // auto inhom_mask  = Extractor<dim>::displacement_mask_inhom(0);
        std::vector<bool> selector (Extractor<dim>::n_components,false);
        selector[Extractor<dim>::first_displacement_component+2] = true;

        dealii::ComponentMask inhom_mask (selector);
        for (auto id : {constr_boundary_ids.inhomogeneousA,
                        constr_boundary_ids.inhomogeneousB })
            DoFTools::make_zero_boundary_constraints (
                    dof_handler, id, constraints, inhom_mask);
    });

}

template <int dim>
void
RetractionSpatulars<dim>::
calcuate_normal(const std::vector<dealii::Point<dim>> & vertices, dealii::Tensor<1,dim> & normal)
{
    using namespace dealii;

    Tensor<1,dim> p1p2 = vertices[0]-vertices[1];
    Tensor<1,dim> p1p3 = vertices[0]-vertices[2];

    normal = cross_product_3d(p1p2, p1p3);
    normal = normal/normal.norm();
}


template <int dim>
inline
void
RetractionSpatulars<dim>::
read_test_protocol (const std::string &filename,const std::string& column_name_displacementA,const std::string& column_name_displacementB)
{
    using namespace dealii;

    Assert (boost::filesystem::exists(boost::filesystem::path(filename)),
            ExcFileNotOpen (filename));

    io::CSVReader<3> in (filename);

    in.read_header(io::ignore_extra_column,"time",column_name_displacementA, column_name_displacementB);

    this->input_data.emplace_back();

    InputData& indata = this->input_data.back();
    indata.filename = filename;

    double time, displacementA, displacementB;
    while (in.read_row (time, displacementA, displacementB))
    {
        std::vector<double> vec{time, displacementA, displacementB};
        indata.data.emplace_back(vec);
    }

    efilog(Verbosity::verbose) << "RetractionSpatulars "
                                  "finished reading experimental "
                                  "data from file <"
                               << filename << ">."<< std::endl;
}

template <int dim>
inline
bool
RetractionSpatulars<dim>::
point_on_spatular(const dealii::Point<dim>& master_pnt, const std::vector<dealii::Point<dim>> &vertices) const
{
    using namespace dealii;
    Tensor<1,dim> vert_a, vert_b, vert_c, vert_d; 
    Tensor<1,dim> pnt(master_pnt);

    vert_a = vertices[0];
    vert_b = vertices[1];
    vert_c = vertices[2];
    vert_d = vertices[3];

    double vert_amab = scalar_product(pnt-vert_a,vert_b-vert_a);
    double vert_abab = scalar_product(vert_b-vert_a,vert_b-vert_a);

    double vert_amad = scalar_product(pnt-vert_a,vert_d-vert_a);
    double vert_adad = scalar_product(vert_d-vert_a,vert_d-vert_a);

    if ( ( (0. < vert_amab) && (vert_amab < vert_abab))  && ( (0. < vert_amad) && (vert_amad < vert_adad) ) )
        { 
            // std::cout << "Point in plane "  << pnt << std::endl;
            return true;
        } 
    // std::cout << "Point NOT in plane "  << pnt << std::endl;
    return false;
}



template <int dim>
inline
void
RetractionSpatulars<dim>::
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

    // this->boundary_set.set_size(dof_handler.n_dofs());

    GetConstrainedBoundaryIDs constr_boundary_ids;
    sample.get_geometry().accept(constr_boundary_ids);

    MeasureBoundarySpatulaWorker<dim> force_workerA;
    force_workerA.set_active (constr_boundary_ids.inhomogeneousA);

    Tensor<1,dim,double> forceA;
    auto force_copierA = force_workerA.create_copier (forceA);

    auto connection_forceA = sample.connect_boundary_loop (
                                       force_workerA,
                                       force_copierA,
                                       sample.signals.post_nonlinear_solve);

    MeasureBoundarySpatulaWorker<dim> force_workerB;
    force_workerB.set_active (constr_boundary_ids.inhomogeneousB);

    Tensor<1,dim,double> forceB;
    auto force_copierB = force_workerB.create_copier (forceB);

    auto connection_forceB = sample.connect_boundary_loop (
                                       force_workerB,
                                       force_copierB,
                                       sample.signals.post_nonlinear_solve);

    std::map<types::global_dof_index, double> boundary_values;

    std::vector<double> times;
    std::vector<double> displacementsA;
    std::vector<double> displacementsB;
    std::vector<double> forcesA;
    std::vector<double> forcesB;
    std::vector<double> areasA;
    std::vector<double> areasB;
    std::vector<double> pressuresA;
    std::vector<double> pressuresB;

    // When the prescribed displacement per step is too large, the algorithm
    // reduces the step width and tries to apply the the boundary conditions
    // in smaller steps. However, sometimes this wont help, therefore the number
    // of allowed refinements of the step width is limited to 5.
    unsigned int refinement_lvl = 0;

    
    // this->boundary_set.clear();
    // IndexSet tmp_boundary_set(this->boundary_set);

    
    for (InputData &input : this->input_data)
    {
        times.clear();
        displacementsA.clear();
        displacementsB.clear();
        forcesA.clear();
        forcesB.clear();
        areasA.clear();
        areasB.clear();
        pressuresA.clear();
        pressuresB.clear();
        // this->boundary_set.clear();

        for (unsigned int step = 0; step < input.data.size(); ++step)
        {
            boundary_values.clear();
            // tmp_boundary_set = this->boundary_set;
            // this->boundary_set.clear();
            // Reset the force.
            forceA = 0;
            forceB = 0;

            double time          = input.data[step][0];
            double previous_time = step>0? input.data[step-1][0] : 0;

            Assert (time > previous_time - 1e-10,
                       ExcMessage("Negative time step size detected: "
                               "corrupted test protocol."));
	    
            // Compute the step size
            double dt = time-previous_time;
            efi::efilog(Verbosity::normal) << "Current time: " << time << std::endl;

            std::vector<scalar_type> valuesA(Extractor<dim>::n_components,0);
            std::vector<scalar_type> valuesB(Extractor<dim>::n_components,0);

            // Note that the strain data is given in percent, therefore it must
            // be multiplied by the height of the sample.
            double displacementA = input.data[step][1];
            double displacementB = input.data[step][2];
            // efilog(Verbosity::debug) << "Inhomogenous A boundary indicator: " << constr_boundary_ids.inhomogeneousA << std::endl;
            efi::efilog(Verbosity::normal) << "Retractor A displacement: " << displacementA << std::endl;
            // efilog(Verbosity::debug) << "Inhomogenous B boundary indicator: " << constr_boundary_ids.inhomogeneousB << std::endl;
            efi::efilog(Verbosity::normal) << "Retractor B displacement: " << displacementB << std::endl;
            
            // SPATULAR A
            for (unsigned int d = 0; d < dim; ++d)
                valuesA[Extractor<dim>::first_displacement_component+d] = displacementA * this->directionA[d];
            Functions::ConstantFunction<dim> boundary_functionA(valuesA);

            std::vector<bool> selectorA (Extractor<dim>::n_components,false);

            // for (unsigned int d = 1; d < dim; ++d)
            //     selectorA[Extractor<dim>::first_displacement_component+d] = true;
            selectorA[Extractor<dim>::first_displacement_component+2] = true;
            dealii::ComponentMask u_mask_A (selectorA);
            VectorTools::interpolate_boundary_values (
                    mapping, dof_handler, constr_boundary_ids.inhomogeneousA,
                    boundary_functionA, boundary_values,
                    u_mask_A);
            // efilog(Verbosity::debug) << "Inhomogenous boundary map created with " << boundary_values.size() << " created." << std::endl;
            // SPATULAR B
            for (unsigned int d = 0; d < dim; ++d)
                valuesB[Extractor<dim>::first_displacement_component+d] = displacementB * this->directionB[d];
            Functions::ConstantFunction<dim> boundary_functionB(valuesB);

            std::vector<bool> selectorB (Extractor<dim>::n_components,false);
            selectorB[Extractor<dim>::first_displacement_component+2] = true;

            // for (unsigned int d = 1; d < dim; ++d)
            //     selectorB[Extractor<dim>::first_displacement_component+d] = true;

            dealii::ComponentMask u_mask_B (selectorB);
            VectorTools::interpolate_boundary_values (
                    mapping, dof_handler, constr_boundary_ids.inhomogeneousB,
                    boundary_functionB, boundary_values,
                    u_mask_B);
            // efilog(Verbosity::debug) << "Inhomogenous boundary map created with " << boundary_values.size() << " created." << std::endl;
            // OLD CODE
            // //////////////////// Retractor A values ////////////////////
            // Tensor<1,dim> disp_dot_normalA = this->normalA*displacementA;            
            // std::vector<dealii::Point<dim>> updated_verticesA(4);            
            // // std::cout << "updated_verticesA ";
            // for ( int i = 0; i < this->verticesA.size(); i++)
            //     {
            //         dealii::Point<dim> newPoint(this->verticesA[i]);
            //         updated_verticesA[i] = newPoint + disp_dot_normalA;
            //         // std::cout << ", " << updated_verticesA[i];
            //     }
            // // std::cout << std::endl;

            // //////////////////// Retractor B values ////////////////////
            // Tensor<1,dim> disp_dot_normalB = this->normalB*displacementB;            
            // std::vector<dealii::Point<dim>> updated_verticesB(4);
            // // std::cout << "updated_verticesB ";
            // for ( int i = 0; i < this->verticesB.size(); i++)
            //     {
            //         dealii::Point<dim> newPoint(this->verticesB[i]);
            //         updated_verticesB[i] = newPoint + disp_dot_normalB;
            //         // std::cout << ", " << updated_verticesB[i];
            //     }
            // // std::cout << std::endl;


            // // Testing for interaction with reatraction spatulars
            
            // auto &fe = sample.get_fe();

            // Quadrature<dim-1> face_quadrature(fe.get_unit_face_support_points());
            // FEFaceValues<dim> fe_values_face(fe, face_quadrature, update_quadrature_points |
            //                                     update_normal_vectors |
            //                                     update_gradients);

            // const unsigned int dofs_per_face = fe.n_dofs_per_face();
            // const unsigned int n_face_q_points = face_quadrature.size();

            // std::vector<types::global_dof_index> dof_indices(dofs_per_face);
            // std::vector<bool> touched_dofs(dof_handler.n_dofs(),false);

            // for (const auto & cell : dof_handler.active_cell_iterators())
            //     if(!cell->is_artificial() && cell->at_boundary() && cell->is_locally_owned())
            //         for (const auto & face : cell->face_iterators())
            //             if (face->at_boundary())
            //                     if (face->boundary_id() == 501 || face->boundary_id() == 503)
            //                     {
            //                         fe_values_face.reinit(cell, face);
            //                         face->get_dof_indices(dof_indices);
            //                         for (unsigned int q_point = 0; q_point<n_face_q_points; q_point += dim)
            //                         {
            //                             const int index = dof_indices[q_point]; 
            //                             if ( (tmp_boundary_set.n_elements() == 0) || (tmp_boundary_set.is_element(index+2)))
            //                             {                                                                                     
            //                                 if (!touched_dofs[index])                                                    
            //                                 {
                                                
            //                                     touched_dofs[index] = true;
            //                                     dealii::Point<dim> support_pnt = fe_values_face.quadrature_point(q_point);
            //                                     Point<dim> current_point_disp;
            //                                     sample.get_slave_pnt(support_pnt,current_point_disp,index);
                                                        
            //                                     Tensor<1,dim> current_disp;
            //                                     std::vector<dealii::Point<dim>> vertices;
            //                                     dealii::Tensor<1, dim> normal;
            //                                     if (face->boundary_id() == 501)
            //                                     {
            //                                         current_disp = disp_dot_normalA;
            //                                         vertices = updated_verticesA;
            //                                         normal = this->normalA;
            //                                     }
            //                                     else
            //                                     {
            //                                         current_disp = disp_dot_normalB;
            //                                         vertices = updated_verticesB;
            //                                         normal = this->normalB;
            //                                     }

            //                                     if (this->point_on_spatular(current_point_disp, vertices))
            //                                     {
            //                                         dealii::Point<dim> master_pnt;
            //                                         this->get_master_point(master_pnt,current_point_disp,normal,vertices);

            //                                         double inhom_value = master_pnt[2] - support_pnt[2];
            //                                         Point<dim> new_point = current_point_disp;
            //                                         int d = 2;
            //                                         // for (int d = 0; d<dim; d++)
            //                                         // {
            //                                         if (std::fabs(inhom_value) > 1e-6){
            //                                             new_point[d] = support_pnt[d] + inhom_value;
            //                                             boundary_values.insert(std::pair<dealii::types::global_dof_index,double>(index+2, inhom_value));
            //                                             if (!this->boundary_set.is_element(index+d))
            //                                             {
            //                                                 this->boundary_set.add_index(index+d);
            //                                             }
            //                                         }
            //                                         // }
            //                                         // std::cout << " new_point: " << new_point << std::endl;
            //                                         // std::cout << std::endl;
            //                                     }
            //                                 }
            //                             }
            //                         }
            //                     }
            
            // efilog(Verbosity::debug) << "Inhomogenous boundary map created with " << boundary_values.size() << " created." << std::endl;
            // for (const auto & pair: boundary_values)
            // {
            //     std::cout << "{ " << pair.first << ": " << pair.second   <<" }\n";
            // }

            // efi::efilog(Verbosity::normal) << "ACC | Active set size: " 
            //                     << Utilities::MPI::sum( (this->boundary_set & locally_owned_dofs).n_elements(), mpi_communicator) << std::endl;

            if (sample.run (boundary_values, dt))
            {
                // When the refinement level is zero, then we have reached a
                // point in time for which experimental data is available.
                // Hence, to be able to compare our results with the
                // experimental data, we write the simulation results to the
                // output arrays.
                if (refinement_lvl == 0)
                {
                    double total_forceA = Utilities::MPI::sum(forceA[0], this->mpi_communicator);
                    double total_forceB = Utilities::MPI::sum(forceB[0], this->mpi_communicator);
                    double total_areaA = Utilities::MPI::sum(forceA[1], this->mpi_communicator);
                    double total_areaB = Utilities::MPI::sum(forceB[1], this->mpi_communicator);
                    // Write to data arrays
                    times.push_back (time);
                    displacementsA.push_back (input.data[step][1]);
                    displacementsB.push_back (input.data[step][2]);
                    forcesA.push_back (total_forceA);
                    forcesB.push_back (total_forceB);
                    areasA.push_back (total_areaA);
                    areasB.push_back (total_areaB);
                    pressuresA.push_back (total_forceA/total_areaA);
                    pressuresB.push_back (total_forceB/total_areaB);

                    // efilog(Verbosity::normal)  << "SPATULA A: ";
                    // efilog(Verbosity::normal)  << "Total force: " << total_forceA;
                    // efilog(Verbosity::normal)  << ", total area: " << total_areaA;
                    // efilog(Verbosity::normal)  << ", total pressure: " << total_forceA/total_areaA << std::endl;

                    // efilog(Verbosity::normal)  << "SPATULA B: ";
                    // efilog(Verbosity::normal)  << "Total force: " << total_forceB;
                    // efilog(Verbosity::normal)  << ", total area: " << total_areaB;
                    // efilog(Verbosity::normal)  << ", total pressure: " << total_forceB/total_areaB << std::endl;
                    
                }
                // tmp_boundary_set = this->boundary_set;

                if (refinement_lvl > 0)
                    --refinement_lvl;
            }
            else
            {
                if (step == 0)
                {
                    // Insert a new intermediate step (linearly interpolated)
                    std::vector<double> vec{0.5 * (input.data[0][0]),
                                            0.5 * (input.data[0][1]),
                                            0.5 * (input.data[0][2])};
                    input.data.insert(input.data.begin(),
                            vec);
                }
                else
                {
                    // TODO Use a higher order interpolation scheme to guess
                    // intermediate values.
                    // Insert a new intermediate step (linearly interpolated)
                    std::vector<double> vec{0.5 * (input.data[step-1][0] + input.data[step][0]),
                                            0.5 * (input.data[step-1][1] + input.data[step][1]),
                                            0.5 * (input.data[step-1][2] + input.data[step][2])};
                    input.data.insert(input.data.begin() + step,
                            vec);
                }

                AssertThrow (++refinement_lvl < 6,
                        dealii::ExcMessage ("Time step refinement level > 5."));
                --step;
                // this->boundary_set = tmp_boundary_set;
            }
        }
    

	if (MPI::is_root(this->mpi_communicator))
	{
	    boost::filesystem::path infilepath  = input.filename;

        boost::filesystem::path outdir = GlobalParameters::get_output_directory();

	    boost::filesystem::path outfilename
    		= outdir / infilepath.filename();

	    io::CSVWriter<9> out(outfilename.string());
	    out.write_headers("time",this->retractor_displacementA,this->retractor_displacementB,"forceA","forceB","areaA","areaB","pressureA","pressureB");
	    out.write_rows(times, displacementsA, displacementsB, forcesA, forcesB, areasA, areasB, pressuresA, pressuresB);
	}

    }
    // Disconnect the remaining signals such that they do not interfere with
    // other testing devices that might be run afterwards or we might run
    // into other problems when e.g. the workers of the boundary loop go out
    // of scope.
    connection_constraints.disconnect();
    connection_forceA.disconnect();
    connection_forceB.disconnect();
}
template <int dim>
void
RetractionSpatulars<dim>::
get_master_point(dealii::Point<dim> &master_pnt, 
                const dealii::Point<dim> &slave_pnt,
                const dealii::Tensor<1, dim> &normal, 
                const std::vector<dealii::Point<dim>> &updated_vertices)
{
    double a, b, c, d, e, f, x, y, z;
    a = normal[0];
    b = normal[1];
    c = normal[2];

    d = updated_vertices[0](0);
    e = updated_vertices[0](1);
    f = updated_vertices[0](2);

    x = slave_pnt(0);
    y = slave_pnt(1);
    z = slave_pnt(2);
    double t = (a*d - a*x + b*e - b*y + c*f - c*z)/(a*a + b*b + c*c);

    master_pnt(0) = x + t*a;
    master_pnt(1) = y + t*b;
    master_pnt(2) = z + t*c;
}

// instantiation
template class RetractionSpatulars<2>;
template class RetractionSpatulars<3>;

// Registration
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(RetractionSpatulars,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(RetractionSpatulars,3));
}// namespace efi

