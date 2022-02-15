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

#ifndef SRC_EFI_INCLUDE_EFI_LAB_TRANSLATIONAL_RHEOMETER_H_
#define SRC_EFI_INCLUDE_EFI_LAB_TRANSLATIONAL_RHEOMETER_H_

// stl headers
#include <vector>
#include <utility>
#include <string>

// deal.II headers
#include <deal.II/base/parameter_handler.h>

// efi headers
#include <efi/base/logstream.h>
#include <efi/lab/sample.h>
#include <efi/lab/testing_device.h>


namespace efi {

/// Translational rheometer for shear testing. Translational motion along
/// the y-axis.
/// @author Stefan Kaessmair
template <int dim>
class TranslationalRheometer : public TestingDevice <dim>
{
public:

    /// Type of scalar numbers.
    using scalar_type = double;

    /// Constructor
    TranslationalRheometer (const std::string &subsection_name,
               const std::string &unprocessed_input = "",
               MPI_Comm mpi_communicator = MPI_COMM_WORLD);

    /// Declare parameters.
    void
    declare_parameters (dealii::ParameterHandler &prm) final;

    /// Declare parameters.
    void
    parse_parameters (dealii::ParameterHandler &prm) final;

    /// The testing device is ran with the
    /// sample attached to it.
    void
    run (Sample<dim> &sample) final;

private:

    static const unsigned int translation_axis = 1;

    /// Helper struct for storing input data.
    struct InputData
    {
        /// Filename of the input data.
        std::string filename;

        /// Time (first) and rotation angle (second) data.
        std::vector<std::pair<double,double>> data;
    };

    /// Get the boundary IDs of the homogeneous and inhomogeneous
    /// constrained boundaries.
    struct GetConstrainedBoundaryIDs : public GeometryVisitor<dim>
    {
        dealii::types::boundary_id homogeneous;
        dealii::types::boundary_id inhomogeneous;

        void visit (const Block<dim> &) final;
        void visit (const Cylinder<dim> &) final;
        void visit (const ImportedGeometry<dim> &) final;
    };

    // Make the constraints required by the used
    // testing device connect this function to
    // sample.signals.add_constrainnts;
    boost::signals2::connection
    connect_constraints (Sample<dim> &sample) const final;

    // Read the test protocol from a file.
    void
    read_test_protocol (const std::string &filename,const std::string & column_name_displacement);

    // Input data
    std::vector<InputData> input_data;

    std::string column_name_displacement;
};



//---------------------- INLINE AND TEMPLATE FUNCTIONS -----------------------//



template <int dim>
TranslationalRheometer<dim>::
TranslationalRheometer (const std::string &subsection_name,
           const std::string &unprocessed_input,
           MPI_Comm mpi_communicator)
: TestingDevice<dim> (subsection_name, unprocessed_input,mpi_communicator)
{
    efilog(Verbosity::verbose) << "New Rheometer created ("
                              << subsection_name
                              << ")." << std::endl;
}



template <int dim>
void
TranslationalRheometer<dim>::GetConstrainedBoundaryIDs::
visit (const Block<dim> &)
{
    this->homogeneous = 0;
    this->inhomogeneous = 1;
}



template <int dim>
void
TranslationalRheometer<dim>::GetConstrainedBoundaryIDs::
visit (const Cylinder<dim> &)
{
    this->homogeneous = 1;
    this->inhomogeneous = 2;
}

template <int dim>
void
TranslationalRheometer<dim>::GetConstrainedBoundaryIDs::
visit (const ImportedGeometry<dim> &)
{
    this->homogeneous = 2;
    this->inhomogeneous = 1;
}

}//namespace efi

#endif /* SRC_EFI_INCLUDE_EFI_LAB_TRANSLATIONAL_RHEOMETER_H_ */
