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
#include <string>
#include <ios>
#include <iostream>
#include <vector>
#include <type_traits>

// deal.II headers
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

// efi headers
#include <efi/base/logstream.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <efi/grid/geometry.h>
#include <efi/factory/registry.h>
#include <efi/lab/sample.h>
#include <efi/base/global_parameters.h>

namespace efi {


template <int dim>
inline
Cylinder<dim>::
Cylinder (const std::string &subsection_name,
          const std::string &unprocessed_input)
:
    Geometry<dim> (subsection_name, unprocessed_input),
    radius (.5),
    height (1.),
    relative_max_element_size (1.)
{
    using namespace dealii;

    this->add_parameter("radius", this->radius, "", ParameterAcceptor::prm,
            Patterns::Double(std::nextafter(0.,1.)));
    this->add_parameter("height", this->height, "", ParameterAcceptor::prm,
            Patterns::Double(std::nextafter(0.,1.)));
    this->add_parameter("relative maxium element size",
            this->relative_max_element_size,
            "Maximum allowed element size relative to the radius in (0,1].",
            ParameterAcceptor::prm,
            Patterns::Double(std::nextafter(0.,1.),1.));

    efilog(Verbosity::verbose) << "New Cylinder created ("
                               << subsection_name
                               << ")." << std::endl;
}



template <int dim>
inline
void
Cylinder<dim>::
create_triangulation (dealii::Triangulation<dim> &tria)
{
    dealii::GridGenerator::cylinder (tria, this->radius, this->height/2.);

    // FIXME this is only a bugfix for the way how I assign inhomogeneous
    // dirichlet boundary conditions. 'My way' only works when the boundary
    // elements do not have hanging nodes constraints between them. However,
    // a boundray cell and an interior cell can have hanging node constraints
    // at their common interface. If the bug is fixed use:
    // refine (tria, this->relative_max_element_size, false)
    refine_without_hanging_boundary_nodes (
            tria, this->relative_max_element_size*this->radius, false);
}



template <int dim>
inline
Block<dim>::
Block (const std::string &subsection_name,
       const std::string &unprocessed_input)
:
    Geometry<dim> (subsection_name, unprocessed_input),
    relative_max_element_size (1.)
{
    efilog(Verbosity::verbose) << "New Block created ("
                               << subsection_name
                               << ")." << std::endl;
}



template <int dim>
void
Block<dim>::
create_triangulation (dealii::Triangulation<dim> &tria)
{
    using namespace dealii;

    std::vector<unsigned int> repetitions(dim,1);

    double min_edge_length = 2. *
            *(std::min_element (this->opposite_corner.begin_raw(),
                                this->opposite_corner.end_raw()));

    for (unsigned int i = 0; i < dim; ++i)
    {
        AssertIsFinite (this->opposite_corner[i] / min_edge_length);
        repetitions[i] =
                std::round (2. * this->opposite_corner[i] / min_edge_length);

        Assert (repetitions[i]>0,
                ExcMessage ("Number of repetitions must be > 0"));
    }

    dealii::GridGenerator::subdivided_hyper_rectangle (
            tria, repetitions, this->corner, this->opposite_corner, true);



    // Setup the periodic boundary conditions.
    std::vector< GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>> periodic_faces;

    for (auto direction : this->periodicity_directions)
    {
        types::boundary_id id1 = 2 * direction;
        types::boundary_id id2 = 2 * direction + 1;

        GridTools::collect_periodic_faces
            (tria, id1, id2, direction, periodic_faces);
    }
    tria.add_periodicity(periodic_faces);


    // FIXME this is only a bugfix for the way how I assign inhomogeneous
    // dirichlet boundary conditions. 'My way' only works when the boundary
    // elements do not have hanging nodes constraints between them. However,
    // a boundray cell and an interior cell can have hanging node constraints
    // at their common interface. If the bug is fixed use:
    // refine (tria, this->relative_max_element_size, false)
    refine_without_hanging_boundary_nodes (
            tria, this->relative_max_element_size*min_edge_length, false);

    efilog(Verbosity::verbose) << "Block tirangulation created."
                               << std::endl;
}



template <int dim>
boost::signals2::connection
Block<dim>::
connect_constraints (Sample<dim> &sample) const
{
    using namespace dealii;

    if (this->periodicity_directions.empty ())
        return boost::signals2::connection ();

    const auto &dof_handler = sample.get_dof_handler();

    using DoFHandlerType = std::decay_t<decltype(dof_handler)>;

    // Create a vector for the periodic face pairs.
    std::vector<dealii::GridTools::PeriodicFacePair<typename
        DoFHandlerType::cell_iterator>> periodic_faces;

    for (unsigned int i = 0; i < this->periodicity_directions.size (); ++i)
    {
        unsigned int direction = this->periodicity_directions[i];

        Tensor<1,dim> offset;
        offset[direction]
               = -this->opposite_corner[direction] + this->corner[direction];

        dealii::GridTools::collect_periodic_faces (
                dof_handler,
                dealii::types::boundary_id (2 * direction),
                dealii::types::boundary_id (2 * direction + 1),
                direction,
                periodic_faces,
                offset);
    }

    // Connect the the periodicty constraints to the
    return sample.signals.make_constraints.connect(
    [periodic_faces]
    (AffineConstraints<typename Sample<dim>::scalar_type> &constraints)
    {
        dealii::DoFTools::make_periodicity_constraints<DoFHandlerType>
            (periodic_faces, constraints);
    });
}



template <int dim>
void
Block<dim>::
declare_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    prm.declare_entry("edge lengths", "1.0,1.0,1.0",
                      Patterns::List (Patterns::Double(std::nextafter(0.,1.)),
                              dim, 3, ","));
    prm.declare_entry("periodicity directions", "",
                      Patterns::List (Patterns::Integer(0,2),0, dim, ","));
    prm.declare_entry("relative maxium element size", "1.0",
                      Patterns::Double (std::nextafter(0.,1.), 1.),
                      "Maximum allowed element size relative to the "
                      "radius in (0,1].");

    efilog(Verbosity::verbose) << "Block finished declaring parameters."
                               << std::endl;
}



template <int dim>
void
Block<dim>::
parse_parameters (dealii::ParameterHandler &param)
{
    using namespace dealii;

    this->relative_max_element_size  =
            param.get_double("relative maxium element size");

    this->periodicity_directions
        = Utilities::string_to_int (
                Utilities::split_string_list(
                        param.get("periodicity directions"),','));

    std::vector<double> edge_lengths
        = Utilities::string_to_double (
                Utilities::split_string_list(
                        param.get("edge lengths"),','));

    for (int i = 0; i < dim; ++i)
    {
        this->corner[i]          = -0.5*edge_lengths[i];
        this->opposite_corner[i] =  0.5*edge_lengths[i];
    }

    efilog(Verbosity::verbose) << "Block finished parsing parameters."
                               << std::endl;
}



template <int dim>
inline
ImportedGeometry<dim>::
ImportedGeometry (const std::string &subsection_name,
          const std::string &unprocessed_input)
:
    Geometry<dim> (subsection_name, unprocessed_input)
{
    using namespace dealii;

    this->add_parameter("inpFile", this->inpFile, "", ParameterAcceptor::prm,
            Patterns::Anything());


    efilog(Verbosity::verbose) << "New Geometry imported ("
                               << subsection_name
                               << ")." << std::endl;
}

template <int dim>
void
ImportedGeometry<dim>::
declare_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    prm.declare_entry("minimums", "-10000,-10000,-10000",
                      Patterns::List (Patterns::Double(),
                              dim, 3, ","));
    prm.declare_entry("maximums", "10000,10000.0,10000.0",
                      Patterns::List (Patterns::Double(),
                              dim, 3, ","));    

    prm.declare_entry ("inhomogenous boundary id","1", Patterns::Integer ());
    prm.declare_entry ("type","maxmin", 
            dealii::Patterns::Selection("maxmin|location"), "options: maxmin|type");

    efilog(Verbosity::verbose) << "Imported Geometry finished declaring parameters."
                               << std::endl;
}



template <int dim>
void
ImportedGeometry<dim>::
parse_parameters (dealii::ParameterHandler &param)
{
    using namespace dealii;

    this->minimums
        = Utilities::string_to_double (
                Utilities::split_string_list(
                        param.get("minimums"),','));
    
    this->maximums
        = Utilities::string_to_double (
                Utilities::split_string_list(
                        param.get("maximums"),','));

    this->inhom_bc = param.get_integer ("inhomogenous boundary id");

    this->type = param.get ("type");

    efilog(Verbosity::verbose) << "Imported Geometry finished parsing parameters."
                               << std::endl;
}





template <int dim>
inline
void
ImportedGeometry<dim>::
create_triangulation (dealii::Triangulation<dim> &tria)
{
        using namespace dealii;
        std::string inputFileName = this->inpFile;

        boost::filesystem::path input_directory = 
            GlobalParameters::get_input_directory();

        // Just a shortcut for the separator
        // in the file path
        std::string sep (1,input_directory.separator);

        // Get the paths of the output files
        std::string path_inp  = input_directory.string() + sep + inputFileName;
        efilog(Verbosity::verbose) << "Importing geometry <"
                                    << path_inp
                                    << ">" << std::endl;
        std::ifstream istream(path_inp);
        dealii::GridIn<dim> gridIn;
        gridIn.attach_triangulation(tria);
        gridIn.read_ucd(istream);

        efilog(Verbosity::verbose) << "New Geometry imported." << std::endl;

        this->setNumberOfCells(tria.n_active_cells());
        // unsigned int num_mat_1 = 0;        

        // // Square_for_A
        // Point<dim> vertex_1_A(42.0480, 46.6839, -45.0336);
        // Point<dim> vertex_2_A(24.4111, 42.8497, -27.7466);
        // Point<dim> vertex_4_A(47.5290, 27.5003, -43.6934);
        // Point<dim> vertex_5_A(48.5790, 49.0599, -37.8436);

        // std::vector<Point<dim>> vertices1245_A = {vertex_1_A, vertex_2_A, vertex_4_A,vertex_5_A};

        // std::vector<Tensor<1, dim>> vertex_vectors_A(3);
        // std::vector< double> vector_dot_vertex_A(6);
        
        // this->calculate_square_vector(vertices1245_A, vertex_vectors_A, vector_dot_vertex_A);

        // double u_dot_x_A;
        // double v_dot_x_A;
        // double w_dot_x_A;

        // // Square_for_B
        // Point<dim> vertex_1_B(41.1181, 46.3631, -46.0495);
        // Point<dim> vertex_2_B(46.5992, 27.1795, -44.7093);
        // Point<dim> vertex_4_B(23.7045, 42.5777, -27.8183);
        // Point<dim> vertex_5_B(34.5871, 43.9871, -53.2395);

        // std::vector<Point<dim>> vertices1245_B = {vertex_1_B, vertex_2_B, vertex_4_B,vertex_5_B};

        // std::vector<Tensor<1, dim>> vertex_vectors_B(3);
        // std::vector< double> vector_dot_vertex_B(6);
        
        // this->calculate_square_vector(vertices1245_B, vertex_vectors_B, vector_dot_vertex_B);

        // double u_dot_x_B;
        // double v_dot_x_B;
        // double w_dot_x_B;

        // Point<dim> maximums;
        // Point<dim> minimums;

        // for (int i=0; i<this->minimums.size(); i++)
        // {
        //     maximums(i) = this->maximums[i];
        //     minimums(i) = this->minimums[i];
        // }

        // efilog(Verbosity::debug) << "Maximums: " << maximums << std::endl;
        // efilog(Verbosity::debug) << "Minimums: " << minimums << std::endl;

        // efilog(Verbosity::verbose) << "Creating dirichlet and inhomogeneous boundaries." << std::endl;

        // for (const auto &cell: tria.active_cell_iterators() )
        //     if(cell->at_boundary())
        //         for (const auto &face: cell->face_iterators() )
        //             if (face->at_boundary())
        //             {
        //                 Point<dim> x = face->center();
        //                 if (x[0] < -4.2)
        //                 {
        //                     face->set_boundary_id(2);
        //                 }   
        //                 else if ( (this->type == "maxmin") &&
        //                           ( (x[0] > minimums[0]) &&
        //                             (x[1] > minimums[1]) && (x[1] < maximums[1]) &&
        //                             (x[2] > minimums[2]) && (x[2] < maximums[2]) )) 
        //                 {
        //                     face->set_boundary_id(1);                           
        //                 }
        //                 else if ( (this->type == "location") &&
        //                           (face->boundary_id() == this->inhom_bc) )
        //                 {
        //                     face->set_boundary_id(1);
        //                 }
        //             }
    //     for (const auto &cell: tria.active_cell_iterators() )
    //         if(cell->at_boundary())
    //             for (const auto &face: cell->face_iterators() )
    //                 if (face->at_boundary())
    //                 {
    //                     Point<dim> x = face->center();
    //                     if ( (x[0] > this->minimums[0]) && (x[0] < this->maximums[0]) 
    //                         &&   (x[1] > this->minimums[1]) && (x[1] < this->maximums[1])
    //                         &&   (x[2] > this->minimums[2]) && (x[2] < this->maximums[2]) )
    //                     {
    //                         double z_minus_min = std::fabs(x[2] - this->minimums[2]);
    //                         double z_minus_max = std::fabs(x[2] - this->maximums[2]);
    //                         // std::cout << "face center: " << x << std::endl;
    //                         // std::cout << "z_minus_min: " << z_minus_min << std::endl;
    //                         // std::cout << "z_minus_max: " << z_minus_max << std::endl;
    //                         if (z_minus_min > z_minus_max) // left face (retractor A), top
    //                         {
    //                                 face->set_boundary_id(501);
    //                                 cell->set_material_id(501);                           
    //                         }
    //                         else if (z_minus_min < z_minus_max) // right face (retractor B), bottmom
    //                         {
    //                                 face->set_boundary_id(503);
    //                                 cell->set_material_id(503);                           
    //                         }
    //                     }
    //                     if ( (cell->material_id() == 16) )  // Dirichlet at brain stem
    //                     {
    //                         // cell->set_material_id(24);
    //                         face->set_boundary_id(2);
    //                     // } else if (x[2] > 3) {              // Dirichlet on right hemisphere
    //                     //     // cell->set_material_id(24);
    //                     //     face->set_boundary_id(2);
    //                     }
                    
    //                 // else if (x[0] > 0.45)
    //                 // {
    //                 //     face->set_boundary_id(1);
    //                 // }
    // //             // else if (cellB == 0)
    // //             // {
    // //             //     face->set_boundary_id(3);
    // //             //     cell->set_material_id(503);
    // //             //     cellB++;

    // //             // }
    //                 }

        this->printMeshInformation(tria);
}


template <int dim>
inline
void
ImportedGeometry<dim>::
calculate_square_vector(const std::vector<dealii::Point<dim>> & vertex1245, 
                                std::vector<dealii::Tensor<1, dim>> & vertex_vectors,
                                std::vector<double> & vector_dot_vertex)
{
    using namespace dealii;

    Point<dim> vertex_1 = vertex1245[0];
    Point<dim> vertex_2 = vertex1245[1];
    Point<dim> vertex_4 = vertex1245[2];
    Point<dim> vertex_5 = vertex1245[3];

        //     6-------7
        //    /|      /|
        //   / |     / |
        //  /  2----/--3
        // 5__/____8  /
        // | /     | /
        // |/      |/  
        // 1-------4

    Tensor<1, dim> u = vertex_1 - vertex_2;
    Tensor<1, dim> v = vertex_1 - vertex_4;
    Tensor<1, dim> w = vertex_1 - vertex_5;

    double u_dot_v1 = scalar_product(u,vertex_1);
    double u_dot_v2 = scalar_product(u,vertex_2);
    double v_dot_v1 = scalar_product(v,vertex_1);
    double v_dot_v4 = scalar_product(v,vertex_4);
    double w_dot_v1 = scalar_product(w,vertex_1);
    double w_dot_v5 = scalar_product(w,vertex_5);

    std::vector<double> u_dots(2);
    u_dots[0] = u_dot_v1;
    u_dots[1] = u_dot_v2;
    if (u_dot_v1 > u_dot_v2)
        std::reverse(u_dots.begin(), u_dots.end());
    std::vector<double> v_dots(2);
    v_dots[0] = v_dot_v1;
    v_dots[1] = v_dot_v4;
    if (v_dot_v1 > v_dot_v4)
        std::reverse(v_dots.begin(), v_dots.end());
    std::vector<double> w_dots(2);
    w_dots[0] = w_dot_v1;
    w_dots[1] = w_dot_v5;
    if (w_dot_v1> w_dot_v5)
        std::reverse(w_dots.begin(), w_dots.end()); 
    
    vertex_vectors[0] = u;
    vertex_vectors[1] = v;
    vertex_vectors[2] = w;
    vector_dot_vertex[0] = u_dots[0];
    vector_dot_vertex[1] = u_dots[1];
    vector_dot_vertex[2] = v_dots[0];
    vector_dot_vertex[3] = v_dots[1];
    vector_dot_vertex[4] = w_dots[0];
    vector_dot_vertex[5] = w_dots[1];
        
}

// Instantiation
template class Block<2>;
template class Block<3>;

template class Cylinder<2>;
template class Cylinder<3>;

template class ImportedGeometry<2>;
template class ImportedGeometry<3>;

// Registration
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Block,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Block,3));

EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Cylinder,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Cylinder,3));

EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(ImportedGeometry,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(ImportedGeometry,3));

}// namespace efi

