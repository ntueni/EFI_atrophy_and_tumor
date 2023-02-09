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

#ifndef SRC_MYLIB_INCLUDE_EFI_GRID_GEOMETRY_H_
#define SRC_MYLIB_INCLUDE_EFI_GRID_GEOMETRY_H_

// deal.II headers
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/utilities.h>

// boost headers
#include <boost/signals2.hpp>

// efi headers
#include <efi/grid/refinement.h>
#include <efi/base/utility.h>
#include <efi/factory/registry.h>


namespace efi {


// Prototype for the visitor class.
template <int dim>
class GeometryVisitor;

// Prototype for the sample class
template <int dim>
class Sample;


/// Base class for geometry descriptions.
/// @author Stefan Kaessmair
template <int dim>
class Geometry : public dealii::ParameterAcceptor
{

public:

    EFI_REGISTER_AS_BASE;

    static const unsigned int dimension = dim;
    static const unsigned int space_dimension = dim;

    /// Constructor.
    /// @param[in] subsection_name The name of the subsection in the parameter
    /// file which defines the parameters of the present class instance.
    /// @param[in] unprocessed_input The unprocessed parts of the parameter
    /// file is everything between the begin of the corresponding subsection
    /// of the to-be-constructed neo-Hooke constitutive model and its end.
    Geometry (const std::string &subsection_name,
              const std::string &unprocessed_input);

    // Destructor.
    virtual
    ~Geometry () = default;

    /// Create a triangulation of the geometry.
    /// @param[out] tria The created triangulation.
    virtual
    void
    create_triangulation (dealii::Triangulation<dim> &tria) = 0;

    /// Accept a geometry operation.
    virtual
    void
    accept (GeometryVisitor<dim> &) const = 0;

    /// Connect geometric constraints to the sample, e.g. periodicity.
    /// By default, this function does nothing;
    virtual
    boost::signals2::connection
    connect_constraints (Sample<dim> &sample) const;
};



/// Description of a block geometry determined by a pair of points:
/// one for the bottom and one for the top corner. The block may have periodic
/// faces in several directions.
/// Notice that the sides are always parallel to the respective axis.
/// @author Stefan Kaessmair
template <int dim>
class Block : public Geometry<dim>
{
public:

    /// Constructor.
    /// @param[in] subsection_name The name of the subsection in the parameter
    /// file which defines the parameters of the present class instance.
    /// @param[in] unprocessed_input The unprocessed parts of the parameter
    /// file is everything between the begin of the corresponding subsection
    /// of the to-be-constructed @p Block geometry and its end.
    /// @note The center of gravity of the @p Block is placed at the origin.
    Block (const std::string &subsection_name,
           const std::string &unprocessed_input);

    /// Vritual destructor.
    virtual
    ~Block () = default;

    /// Create a triangulation of block geometry. See the documentation of
    /// <code>dealii::GridGenerator::subdivided_hyper_rectangle()</code>
    /// for details on the @p boundary_ids of the created triangulation.
    /// @param[out] tria The created triangulation.
    virtual
    void
    create_triangulation (dealii::Triangulation<dim> &tria) override;

    /// Declare parameters to the given parameter handler.
    /// @param[out] prm The parameter handler for which we want to declare
    /// the parameters.
    virtual
    void
    declare_parameters (dealii::ParameterHandler &param) override;

    /// Parse the parameters stored int the given parameter handler.
    /// @param[in] prm The parameter handler whose parameters we want to
    /// parse.
    virtual
    void
    parse_parameters (dealii::ParameterHandler& param) override;

    /// Accept a visitor.
    virtual
    void
    accept (GeometryVisitor<dim> &) const override;

    /// Connect periodicity constraints to the sample. The block is periodic in
    /// the directions defined in @p periodic_directions.
    /// @param[in] sample Sample to which we want to connect periodicity
    /// constraints.
    virtual
    boost::signals2::connection
    connect_constraints (Sample<dim> &sample) const override;

    /// Get the bounding box.
    std::pair<dealii::Point<dim>,dealii::Point<dim>>
    get_boundary_points () const;

protected:

    /// One of the corners of the hyper-rectangle.
    dealii::Point<dim> corner;

    /// The opposite corner to 'corner' of the hyper-rectangle.
    dealii::Point<dim> opposite_corner;

    /// Maximum element size relative to the radius.
    double relative_max_element_size;

    /// List of directions in which the block has periodic boundaries.
    std::vector<int> periodicity_directions;
};



/// Geometric description of a cylinder. The $x$-axis serves as the axis of
/// the cylinder. Here, a cylinder is defined as a (@p dim - 1) dimensional
/// disk of given @p radius, extruded along the axis of the cylinder (which is
/// the first coordinate direction). Consequently, in three dimensions, the
/// cylinder extends from `x=-half_height` to `x=+half_height` and its
/// projection into the @p yz-plane is a circle of radius @p radius. In two
/// dimensions, the cylinder is a rectangle from `x=-half_length` to
/// `x=+half_length` and from `y=-radius` to `y=radius`. Taken from dealii
/// manual for the function GridGenerator::cylinder.
/// @author Stefan Kaessmair
template <int dim>
class Cylinder : public Geometry<dim>
{
public:

    /// Constructor.
    /// @param[in] subsection_name The name of the subsection in the parameter
    /// file which defines the parameters of the present class instance.
    /// @param[in] unprocessed_input The unprocessed parts of the parameter
    /// file is everything between the begin of the corresponding subsection
    /// of the to-be-constructed @p Cylinder geometry and its end.
    /// @note The center of gravity of the @p Cylinder is placed at the origin.
    Cylinder (const std::string &subsection_name,
              const std::string &unprocessed_input);

    /// Destructor.
    virtual
    ~Cylinder () = default;

    /// Create a triangulation of the cylinder. See the documentation of
    /// <tt>dealii::GridGenerator::cylinder()</tt> for the details on the
    /// @p boundary_ids of the created triangulation.
    /// @param[out] tria The created triangulation.
    virtual
    void
    create_triangulation (dealii::Triangulation<dim> &tria) override;

    /// Accept a visitor.
    virtual
    void
    accept (GeometryVisitor<dim> &) const override;

    /// Return the line segment of the center axis contained in the hull of the
    /// cylinder.
    std::pair<dealii::Point<dim>,dealii::Point<dim>>
    get_axis_segment () const;

    /// Get the radius of the cylinder.
    double
    get_radius () const;

    /// Get the height of the cylinder
    double
    get_height () const;

protected:

    // Conter of the cylinder
    dealii::Point<dim> center;

    // radius
    double radius;

    // height
    double height;

    // Maximum element size relative to the radius.
    double relative_max_element_size;
};

/// Geometric description of an imported geometery. 
/// Utilizing  dealii GridGenerator::GridIn function.
/// @author Stefan Kaessmair
template <int dim>
class ImportedGeometry : public Geometry<dim>
{
public:

    /// Constructor.
    /// @param[in] subsection_name The name of the subsection in the parameter
    /// file which defines the parameters of the present class instance.
    /// @param[in] unprocessed_input The unprocessed parts of the parameter
    /// file is everything between the begin of the corresponding subsection
    /// of the to-be-constructed 
    ImportedGeometry (const std::string &subsection_name,
              const std::string &unprocessed_input);

    /// Destructor.
    virtual
    ~ImportedGeometry () = default;

    /// Declare parameters to the given parameter handler.
    /// @param[out] prm The parameter handler for which we want to declare
    /// the parameters.
    virtual
    void
    declare_parameters (dealii::ParameterHandler &param) override;

    /// Parse the parameters stored int the given parameter handler.
    /// @param[in] prm The parameter handler whose parameters we want to
    /// parse.
    virtual
    void
    parse_parameters (dealii::ParameterHandler& param) override;

    /// Create a triangulation of the imported geometry. 
    virtual
    void
    create_triangulation (dealii::Triangulation<dim> &tria) override;

    /// Accept a visitor.
    virtual
    void
    accept (GeometryVisitor<dim> &) const override;

    unsigned int numberOfCells;

    // Set number of cells in imported Geometry
    void setNumberOfCells(unsigned int);

    // Get number of cells in imported Geometry
    unsigned int getNumberOfCells();
    void printMeshInformation(const dealii::Triangulation<dim> &);

    protected:

    void calculate_square_vector(const std::vector<dealii::Point<dim>> &, 
                                std::vector<dealii::Tensor<1, dim>> &,
                                std::vector<double> &);

    std::string
    inpFile;

    std::vector<double>
    minimums;

    std::vector<double>
    maximums;

    dealii::types::boundary_id
    inhom_bc;

    std::string
    type;

};


/// Visitor class for geometry objects. This class is particular useful for
/// classes derived from @p TestingDevice which need to know about the geometry
/// in order to construct constraints. However, given @p Sample as input they
/// only get an object of type @p Geometry returned, i.e. double dispatching
/// via the visitor pattern allows to access the functions of the derived
/// classes @p Block and @p Cylinder.
/// @author Stefan Kaessmair
template <int dim>
class GeometryVisitor
{
public:

    /// Destructor.
    virtual ~GeometryVisitor () = default;

    /// Read only visit a @p Block geometry object.
    virtual
    void
    visit (const Block<dim> &) = 0;

    /// Visit a @p Cylinder geometry object.
    virtual
    void
    visit (const Cylinder<dim> &) = 0;

    /// Visit a @p Import geometry object.
    virtual
    void
    visit (const ImportedGeometry<dim> &) = 0;
};





//---------------------- INLINE AND TEMPLATE FUNCTIONS -----------------------//



template <int dim>
inline
Geometry<dim>::
Geometry (const std::string &subsection_name,
          const std::string &)
:
    dealii::ParameterAcceptor (subsection_name)
{ }


template <int dim>
inline
boost::signals2::connection
Geometry<dim>::
connect_constraints (Sample<dim> &) const
{
    return boost::signals2::connection();
}



template <int dim>
inline
void
Block<dim>::
accept (GeometryVisitor<dim> &v) const
{
    v.visit(*this);
}



template <int dim>
inline
std::pair<dealii::Point<dim>,dealii::Point<dim>>
Block<dim>::
get_boundary_points () const
{
    return std::make_pair (this->corner, this->opposite_corner);
}



template <int dim>
inline
void
Cylinder<dim>::
accept (GeometryVisitor<dim> &v) const
{
    v.visit(*this);
}



template <int dim>
inline
std::pair<dealii::Point<dim>,dealii::Point<dim>>
Cylinder<dim>::
get_axis_segment () const
{
    dealii::Point<dim> p1(this->center);
    dealii::Point<dim> p2(this->center);

    p1[0] -= 0.5*this->height;
    p1[0] += 0.5*this->height;

    return std::make_pair(p1,p2);
}



template <int dim>
inline
double
Cylinder<dim>::
get_radius () const
{
    return this->radius;
}



template <int dim>
inline
double
Cylinder<dim>::
get_height () const
{
    return this->height;
}

template <int dim>
inline
void
ImportedGeometry<dim>::
accept (GeometryVisitor<dim> &v) const
{
    v.visit(*this);
}

template <int dim>
unsigned int 
ImportedGeometry<dim>::
getNumberOfCells()
{
    return this->numberOfCells;
}


template <int dim>
void
ImportedGeometry<dim>::
setNumberOfCells(unsigned int num)
{
    this->numberOfCells = num;
}

/// Function to give mesh information for an imported geometry.
/// Taken from deal.ii step-49 tutorial
template <int dim>
inline
void
ImportedGeometry<dim>::
printMeshInformation (const dealii::Triangulation<dim> &triangulation)
{
    efilog(Verbosity::debug) << "Mesh info:" << std::endl
            << " dimension: " << dim << std::endl
            << " no. of cells: " << triangulation.n_active_cells() << std::endl;

    std::map<dealii::types::boundary_id, unsigned int> boundary_count;
    for (const auto &face : triangulation.active_face_iterators())
      if (face->at_boundary())
        boundary_count[face->boundary_id()]++;

    efilog(Verbosity::debug) << " boundary indicators: ";
    for (const std::pair<const dealii::types::boundary_id, unsigned int> &pair :
         boundary_count)
      {
        efilog(Verbosity::debug) << pair.first << "(" << pair.second << " times) ";
      }
    efilog(Verbosity::debug) << "||" << std::endl;

}



}// namespace efi


#endif /* SRC_MYLIB_INCLUDE_EFI_GRID_GEOMETRY_H_ */
