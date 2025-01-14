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

#ifndef SRC_EFI_INCLUDE_EFI_LAB_RETRACTION_SPATULARS_H_
#define SRC_EFI_INCLUDE_EFI_LAB_RETRACTION_SPATULARS_H_

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

/// Retraction using retraction spatulars (variable dimenions). Translational motion along
/// the z-axis.
/// @author Stefan Kaessmair
template <int dim>
class RetractionSpatulars : public TestingDevice <dim>
{
public:

    /// Type of scalar numbers.
    using scalar_type = double;

    /// Constructor
    RetractionSpatulars (const std::string &subsection_name,
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

        /// Time (first) and (second) data vector:retractor displacement 1 and retractor displacement 2.
        std::vector<std::vector<double>> data;
    };

    /// Get the boundary IDs of the homogeneous and inhomogeneous
    /// constrained boundaries.
    struct GetConstrainedBoundaryIDs : public GeometryVisitor<dim>
    {
        dealii::types::boundary_id homogeneous;
        dealii::types::boundary_id inhomogeneousA;
        dealii::types::boundary_id inhomogeneousB;

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
    read_test_protocol (const std::string &filename,const std::string & column_nameA_displacement,const std::string & column_nameB_displacement);

    bool
    point_on_spatular(const dealii::Point<dim> &, 
                    const std::vector<dealii::Point<dim>> &verticesA) const;

    void calcuate_normal(const std::vector<dealii::Point<dim>> &, 
                        dealii::Tensor<1,dim> &);

    void get_master_point(dealii::Point<dim> &master_pnt_A, 
                const dealii::Point<dim> &slave_pnt,
                const dealii::Tensor<1, dim> &normalA, 
                const std::vector<dealii::Point<dim>> &updated_verticesA);

    // Input data
    std::vector<InputData> input_data;

    std::string retractor_displacementA;
    std::string retractor_displacementB;

    // dealii::Tensor<1, dim> normalA;
    // dealii::Tensor<1, dim> normalB;
    std::vector<double> directionA;
    std::vector<double> directionB;

    // dealii::IndexSet boundary_set;
};



//---------------------- INLINE AND TEMPLATE FUNCTIONS -----------------------//



template <int dim>
RetractionSpatulars<dim>::
RetractionSpatulars (const std::string &subsection_name,
           const std::string &unprocessed_input,
           MPI_Comm mpi_communicator)
: TestingDevice<dim> (subsection_name, unprocessed_input,mpi_communicator),
directionA(4),
directionB(4)
{
    efilog(Verbosity::verbose) << "New Retraction spatular created ("
                              << subsection_name
                              << ")." << std::endl;
}



template <int dim>
void
RetractionSpatulars<dim>::GetConstrainedBoundaryIDs::
visit (const Block<dim> &)
{
    this->homogeneous = 0;
    this->inhomogeneousA = 1;
}


template <int dim>
void
RetractionSpatulars<dim>::GetConstrainedBoundaryIDs::
visit (const Cylinder<dim> &)
{
    this->homogeneous = 1;
    this->inhomogeneousA = 2;
}

template <int dim>
void
RetractionSpatulars<dim>::GetConstrainedBoundaryIDs::
visit (const ImportedGeometry<dim> &)
{
    this->homogeneous = 2;
    this->inhomogeneousA = 1;
    this->inhomogeneousB = 3;
}

}//namespace efi

#endif /* SRC_EFI_INCLUDE_EFI_LAB_RETRACTION_SPATULARS_H_ */
