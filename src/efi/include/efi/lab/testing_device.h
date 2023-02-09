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

#ifndef SRC_MYLIB_INCLUDE_EFI_LAB_TESTING_DEVICE_H_
#define SRC_MYLIB_INCLUDE_EFI_LAB_TESTING_DEVICE_H_

// stl headers
#include <string>
#include <array>

// deal.II headers
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>

// boost headers
#include <boost/signals2.hpp>
#include <boost/filesystem.hpp>

// efi headers
#include <efi/grid/geometry.h>
#include <efi/lab/sample.h>
#include <efi/worker/boundary_worker.h>


namespace efi {


// Base class for the setup of the
// device which will do the testing.
// It should provide boundary conditions,
// be able to read external files etc. ...
template <int dim>
class TestingDevice : public dealii::ParameterAcceptor
{
public:

    /// Dimension in which this object operates.
    static const unsigned int dimension = dim;

    /// Dimension of the space in which this object operates.
    static const unsigned int space_dimension = dim;

    /// Type of scalar numbers.
    using scalar_type = double;

    // Constructor
    TestingDevice (const std::string &subsection_name,
                   const std::string &unprocessed_input = "",
                   MPI_Comm mpi_communicator = MPI_COMM_WORLD);

    // Destructor
    virtual
    ~TestingDevice () = default;

    // The testing device is ran with the
    // sample attached to it.
    virtual
    void
    run (Sample<dim> &sample) = 0;

protected:

    /// Check whether the geometry is symmetric with respect to the
    /// planes $x=0$, $y=0$, $z=0$.
    struct IsReflectionSymmetric : public GeometryVisitor<dim>
    {
        std::array<bool,dim> values;

        void visit (const Block<dim> &) final;
        void visit (const Cylinder<dim> &) final;
        void visit (const ImportedGeometry<dim> &) final;
    };

    // Make the constraints required by the used
    // testing device connect this function to
    // sample.signals.add_constrainnts;
    virtual
    boost::signals2::connection
    connect_constraints (Sample<dim> &sample) const;
    
    virtual
    boost::signals2::connection
    connect_constraints2 (Sample<dim> &sample) const;

    /// The mpi communicator of this object
    MPI_Comm mpi_communicator;
    
    std::string output_file;
};



//---------------------- INLINE AND TEMPLATE FUNCTIONS -----------------------//



template <int dim>
TestingDevice<dim>::
TestingDevice (const std::string &subsection_name,
               const std::string &,
               MPI_Comm mpi_communicator)
: dealii::ParameterAcceptor (subsection_name),
  mpi_communicator (mpi_communicator)
{ }



template <int dim>
inline
boost::signals2::connection
TestingDevice<dim>::
connect_constraints (Sample<dim> &) const
{
    // By default no constraints are added.
    return boost::signals2::connection();
}

template <int dim>
inline
boost::signals2::connection
TestingDevice<dim>::
connect_constraints2 (Sample<dim> &) const
{
    // By default no constraints are added.
    return boost::signals2::connection();
}



template <int dim>
void
TestingDevice<dim>::IsReflectionSymmetric::
visit (const Block<dim> &)
{
    for (unsigned int d = 0; d < dim; ++d)
        this->values[d] = true;
}



template <int dim>
void
TestingDevice<dim>::IsReflectionSymmetric::
visit (const Cylinder<dim> &)
{
    for (unsigned int d = 0; d < dim; ++d)
        this->values[d] = true;
}


template <int dim>
void
TestingDevice<dim>::IsReflectionSymmetric::
visit (const ImportedGeometry<dim> &)
{
    for (unsigned int d = 0; d < dim; ++d)
        this->values[d] = false;
}


}// namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_LAB_TESTING_DEVICE_H_ */
