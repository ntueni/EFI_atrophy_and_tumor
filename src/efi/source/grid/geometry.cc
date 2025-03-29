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
    
    this->add_parameter("muFile",  this->muFile,  "", ParameterAcceptor::prm,
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
    
    efilog(Verbosity::verbose) << "Imported Geometry finished declaring parameters."
                               << std::endl;
}



template <int dim>
void
ImportedGeometry<dim>::
parse_parameters (dealii::ParameterHandler &param)
{
    using namespace dealii;
    
    efilog(Verbosity::verbose) << "Imported Geometry finished parsing parameters."
                               << std::endl;
}





template <int dim>
inline
void
ImportedGeometry<dim>::create_triangulation (dealii::Triangulation<dim> &tria)
{
    using namespace dealii;
    std::string inputFileName = this->inpFile;  // Semikolon hinzugefügt

    boost::filesystem::path input_directory = GlobalParameters::get_input_directory();
    // Shortcut: Verwende den statischen Member für das bevorzugte Pfadtrennzeichen
    std::string sep(1, boost::filesystem::path::preferred_separator);
    // Erzeugen des vollständigen Pfads zur Eingabedatei
    std::string path_inp = input_directory.string() + sep + inputFileName;
    
    efilog(Verbosity::verbose) << "Importing geometry <" << path_inp << ">" << std::endl;

    std::ifstream istream(path_inp);
    dealii::GridIn<dim> gridIn;
    gridIn.attach_triangulation(tria);
    gridIn.read_ucd(istream);

    efilog(Verbosity::verbose) << "New Geometry imported." << std::endl;

    this->setNumberOfCells(tria.n_active_cells());
    this->printMeshInformation(tria);

     // Reading of the FA-Values from rampp_UCD2.inp
    if (!this->muFile.empty())
    {
        std::string path_mu = input_directory.string() + sep + this->muFile;
        efilog(Verbosity::verbose) << "Importing mu values from <" << path_mu << ">" << std::endl;
        std::ifstream muStream(path_mu);
        if (!muStream)
        {
            efilog(Verbosity::normal) << "Could not open mu file: " << path_mu << std::endl;
        }
        else
        {
            std::string line;
            while (std::getline(muStream, line))
            {
                // Jump over empty lines and header (Header lines begin with '#')
                if (line.empty() || line[0]=='#')
                    continue;
                std::istringstream iss(line);
                std::vector<std::string> tokens;
                std::string token;
                // Divide the lines into tokens
                while (iss >> token)
                    tokens.push_back(token);
                // Jump over lines with the wrong number of tokens (Knot table has only 5 tokens: Number, x,y,z, FA-Value)
                if (tokens.size() < 6)
                    continue;
                try
                {
                    // FA-Value is written in the last column
                    double fa_value = std::stod(tokens.back());
                    double mu_element = 0.0;

                    if (fa_value == 0.0) {
                        //Falls der FA-Wert 0.0 ist, setze den mu-Wert auf 10e-6
                        mu_element = 10e-6;
                    } else {
                        // Andernfalls berechne den mu-Wert mit der Formel
                        mu_element = fa_value;
                        //mu_element = (-(fa_value / 0.0037) + 182.4)*1e-6;
                    }

                    // Speichere den berechneten mu-Wert in den Container
                    this->mu_values.push_back(mu_element);
                }
                catch (const std::invalid_argument &e)
                {
                    efilog(Verbosity::normal) << "Conversion failed for token: " 
                                            << tokens.back() << std::endl;
                }
                catch (const std::out_of_range &e)
                {
                    efilog(Verbosity::normal) << "Token out of range: " 
                                            << tokens.back() << std::endl;
                }
            }
            efilog(Verbosity::verbose) << "mu values imported, total count: " 
                                    << this->mu_values.size() << std::endl;
        }
    }
    // Debug .txt file to compare the imported mu values with the .inp File
    //std::ofstream debugFile("/workspace/src/debug_mu_geometry.txt");
    //if (debugFile.is_open())
    //{
    //    debugFile << "mu values imported, total count: " << this->mu_values.size() << "\n";
    //    for (size_t i = 0; i < this->mu_values.size(); ++i)
    //    {
    //        debugFile << "Element " << (i+1) << ": " << this->mu_values[i] << "\n";
    //    }
    //    debugFile.close();
    //}
    //else
    //{
    //    efilog(Verbosity::normal) << "Could not open debug file for writing." << std::endl;
    //}
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

