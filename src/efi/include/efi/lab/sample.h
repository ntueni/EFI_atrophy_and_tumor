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

#ifndef SRC_MYLIB_INCLUDE_EFI_LAB_SAMPLE_H_
#define SRC_MYLIB_INCLUDE_EFI_LAB_SAMPLE_H_

// dealii headers
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_selector.h>
#include <deal.II/base/timer.h>
#include <deal.II/meshworker/assemble_flags.h>
//#include <deal.II/meshworker/mesh_loop.h>
#include <efi/base/bugfix_mesh_loop.h>
#include <deal.II/fe/fe_system.h>

// boost headers
#include <boost/signals2.hpp>

// efi headers
#include <efi/grid/geometry_factory.h>
#include <efi/lac/nonlinear_solver_control.h>
#include <efi/lac/generic_linear_algebra.h>
#include <efi/worker/cell_worker.h>
#include <efi/worker/copy_data.h>
#include <efi/worker/boundary_worker.h>
#include <efi/worker/general_cell_data_storage.h>
#include <efi/worker/scratch_data.h>
#include <efi/grid/obstacle.h>

namespace efi {


// compare this to the solid class in
// Esters poro-visco-elastic problem.
// The specimen has a geometry, consists of
// a material, and responds in a certain
// way if it experiences external loads.
template <int dim>
class Sample : public dealii::ParameterAcceptor
{
public:

    EFI_REGISTER_AS_BASE;

    /// Dimension in which this object operates.
    static const unsigned int dimension = dim;

    /// Dimension of the space in which this object operates.
    static const unsigned int space_dimension = dim;

    /// Type of scalar numbers.
    using scalar_type = double;

    // Default constructor
    Sample (const std::string &subsection_name,
            const std::string &unprocessed_input,
            MPI_Comm           mpi_communicator = MPI_COMM_WORLD);

    // Default destructor
    virtual
    ~Sample ();

    // Declare parameters
    virtual
    void
    declare_parameters (dealii::ParameterHandler &prm) override;

    // Parse the parameters.
    virtual
    void
    parse_parameters (dealii::ParameterHandler &prm) override;

    // Run the simulation where at time t+dt, the dirichlet
    // boundary dof values are incremented by the prescribed
    // value.
    bool
    run (const std::map<dealii::types::global_dof_index,double> &prescribed,
         const double time_step_size);

    // Initialize the members.
    // TODO Integrate set_output_directory in initialize.
    virtual
    void
    initialize ();

    virtual
    void
    reset ();

    //
    virtual
    void
    reinit_constraints ();

    // Return a constant reference to the
    // dof handler object. From the dof handler
    // the triangulation and the finite element
    // (system) can be obtained.
    const dealii::DoFHandler<dim>&
    get_dof_handler () const;

    // Return a constant reference to the used map.
    const dealii::Mapping<dim>&
    get_mapping () const;

        // Return a constant reference to the used map.
    const dealii::FESystem<dim>&
    get_fe () const;

    // Return a constant reference to the used geometry.
    const Geometry<dim> &
    get_geometry () const;

    // Return a constant reference to the contact obstacle.
    const Obstacle<dim> &
    get_obstacle () const;

    const std::vector<dealii::types::material_id>
    get_material_ids() const;

    void
    set_material_ids();

    void
    set_obstacle_displacement(double);


    void
    get_slave_pnt(const dealii::Point<dim> &, dealii::Point<dim> &, dealii::types::global_dof_index);

    // // Return a constant reference to the used constitutive model.
    // const ConstitutiveBase<dim> &
    // get_constitutive_model () const;

    // Return a constant reference to the used constitutive model when given material_id.
    const ConstitutiveBase<dim> &
    get_constitutive_model (const int) const;

    /// Create a mesh loop using the @p Sample data structures and
    /// connect it to the given signal.
    /// This function uses @p constitutive_model as data processor for
    /// the workers @p external_cell_worker and @p external_boundary_worker.
    boost::signals2::connection
    connect_mesh_loop (
            const CellWorker<dim>            &external_cell_worker,
            const BoundaryWorker<dim>        &external_boundary_worker,
            const std::function<void(const CopyData&)> &external_copier,
            boost::signals2::signal<void()>  &signal,
            const dealii::MeshWorker::AssembleFlags flags =
                    dealii::MeshWorker::assemble_own_cells
                  | dealii::MeshWorker::assemble_boundary_faces);

    /// Create a mesh loop using the @p Sample data structures and
    /// connect it to the given signal.
    /// This function uses @p external_cell_data_processor and
    /// @p external_boundary_data_processor as data processors for
    /// the @p external_cell_worker and @p external_boundary_worker,
    /// respectively.
    template <class CellDataProcessorType,
              class BoundaryDataProcessorType>
    boost::signals2::connection
    connect_mesh_loop (
            const CellDataProcessorType     &external_cell_data_processor,
            const CellWorker<dim>           &external_cell_worker,
            const BoundaryDataProcessorType &external_boundary_data_processor,
            const BoundaryWorker<dim>       &external_boundary_worker,
            const std::function<void(const CopyData&)> &external_copier,
            boost::signals2::signal<void()> &signal,
            const dealii::MeshWorker::AssembleFlags flags =
                    dealii::MeshWorker::assemble_own_cells
                  | dealii::MeshWorker::assemble_boundary_faces);

    /// Create a boundary mesh loop using the @p Sample data structures and
    /// connect it to the given signal.
    /// This function uses @p external_boundary_data_processor as data
    /// processor for @p external_boundary_worker.
    template <class BoundaryDataProcessorType>
    boost::signals2::connection
    connect_boundary_loop (
            const BoundaryDataProcessorType &external_boundary_data_processor,
            const BoundaryWorker<dim>       &external_boundary_worker,
            const std::function<void(const CopyData&)> &external_copier,
            boost::signals2::signal<void()> &signal);

    /// Create a boundary mesh loop using the @p Sample data structures and
    /// connect it to the given signal.
    /// This function uses @p external_boundary_data_processor as data
    /// processor for @p external_boundary_worker.
    boost::signals2::connection
    connect_boundary_loop (
            const BoundaryWorker<dim>       &external_boundary_worker,
            const std::function<void(const CopyData&)> &external_copier,
            boost::signals2::signal<void()> &signal);

//    /// Create a boundary mesh loop using the @p Sample data structures and
//    /// connect it to the given signal.
//    /// This function uses @p external_boundary_data_processor as data
//    /// processor for @p external_boundary_worker.
//    template <class CellDataProcessorType>
//    boost::signals2::connection
//    connect_cell_loop (
//            const CellDataProcessorType &external_boundary_data_processor,
//            const CellWorker<dim>       &cell_boundary_worker,
//            const std::function<void(const CopyData&)> &external_copier,
//            boost::signals2::signal<void()> &signal);

    // A structure that has boost::signal objects for a number of
    // actions that a sample can do to itself.
    //
    // For documentation on signals, see
    // http://www.boost.org/doc/libs/release/libs/signals2 .
    struct Signals
    {
        // This signal is triggered when efi
        // initializes the sample, i.e. when initilize()
        // is called. By default no constraints are
        // applied.
        boost::signals2::signal<
            void(dealii::AffineConstraints<scalar_type> &)>
        make_constraints;
        boost::signals2::signal<
            void(dealii::AffineConstraints<scalar_type> &, LA::MPI::Vector &)>
        make_constraints2;

        // This signal is triggered when the nonlinear solver
        // has converged.
        boost::signals2::signal<void()> pre_nonlinear_solve;

        // This signal is triggered when the nonlinear solver
        // has converged.
        boost::signals2::signal<void()> post_nonlinear_solve;

        // Disconnect all slots of all signals.
        void
        disconnect_all_slots ();
    };

    // Signals for the various actions that a sample
    // can do to itself.
    mutable Signals signals;

private:

    // Create a sparsity pattern for
    // the system matrix.
    void
    reinit_sparsity ();

    // Application of contact via 
    // updating solution and constraints object.
    void
    apply_contact_constraints ();

    void 
    compute_residual();

    // Assemble the linear system
    // characterized by system_matrix
    // and system_vector.
    void
    assemble ();

    // Solve the linear system.
    void
    solve_linear ();

    // Solve the nonlinear system.
    void
    solve_nonlinear ();

    // Perform an allreduce on the state.
    void
    all_reduce_state ();

    // Write output
    void
    write_output (const unsigned int step,
                  const double       time);

    //Move mesh
    void
    move_mesh(const  LA::MPI::Vector &displacement) const;

    double 
    calculate_area(dealii::types::boundary_id id) const;

    void
    compute_contact_force(dealii::IndexSet &active_set);

protected:

    // Instantiate the protected members based
    // on the given input, in particular the
    // workers, constitutive model, and geometry.
    virtual
    void
    instantiate (std::istream &unprocessed_input);

    // workers
    std::unique_ptr<CellWorker<dim>>     cell_worker;
    std::unique_ptr<BoundaryWorker<dim>> boundary_worker;

    // constitutive model
    // std::unique_ptr<ConstitutiveBase<dim>> constitutive_model;

    // constitutive model
    std::map<int, std::unique_ptr<ConstitutiveBase<dim>>> constitutive_model_map;

    // geometry
    std::unique_ptr<Geometry<dim>> geometry;

    // contact object
    std::unique_ptr<Obstacle<dim>> obstacle;

private:

    // output times and names required
    // to write the *.pvd file.
    std::vector<std::pair<scalar_type,std::string>> times_and_names;

    // MPI communicator
    MPI_Comm mpi_communicator;

    // triangulation
    dealii::parallel::distributed::Triangulation<dim> tria;

    // dof handler and constraints
    dealii::DoFHandler<dim>                dof_handler;
    dealii::AffineConstraints<scalar_type> constraints;
    dealii::AffineConstraints<scalar_type> contact_constraints;
    dealii::AffineConstraints<scalar_type> empty_constraints;

    // index sets
    dealii::IndexSet locally_owned_dofs;
    dealii::IndexSet locally_relevant_dofs;
    dealii::IndexSet active_set;

    // local objects
    std::unique_ptr<dealii::FESystem<dim>>        fe;
    std::unique_ptr<dealii::MappingQGeneric<dim>> mapping;

    // quadrature formulae
    std::unique_ptr<dealii::QuadratureSelector<dim>>   qf_cell;
    std::unique_ptr<dealii::QuadratureSelector<dim-1>> qf_face;

    // linear system
    LA::MPI::SparseMatrix system_matrix;
    LA::MPI::Vector       system_vector;
    LA::MPI::Vector       system_increment;
    LA::MPI::Vector       diag_mass_matrix_vector;
    LA::MPI::Vector       uncondensed_rhs;
    LA::MPI::Vector       locally_owned_solution;
    LA::MPI::Vector       locally_relevant_solution;

    // nonlinear system
    NonlinearSolverControl solver_control;

    // time step size
    scalar_type time_step_size;
    scalar_type elapsed_time;

    // Sample of a scratch data object. It contains all
    // data structures and temporary objects required
    // by each thread to evaluate the contribution
    // of a single cell/face/boundary. We have to make
    // this class a pointer as there is no default
    // constructor for MeshWorker::ScratchData class,
    // but the required information to initialize the
    // class has to be parsed from a parameter file.
    // See the deal.II documentation for the
    // MeshWorker::ScratchData class for further
    // details.
    std::unique_ptr<ScratchData<dim>> sample_scratch_data;

    // Sample of a copy data object. It stores all the
    // "per task data". That is everything which is
    // needed to assemble the cell contributions to
    // the global linear system.
    // Even though not necessary here - CopyData is
    // default constructible - for the sake of
    // uniformity, only pointer to the CopyData object is
    // stored just like for the ScratchData.
    std::unique_ptr<CopyData> sample_copy_data;

    // Cell data storage. A reference of it is stored in
    // sample_scratch_data.
    std::unique_ptr<GeneralCellDataStorage> cell_data_history_storage;

    // Cell data storage. A reference of it is stored in
    // sample_scratch_data.
    std::unique_ptr<GeneralCellDataStorage> tmp_cell_data_history_storage;

    // Alias
    using State = dealii::SolverControl::State;

    // State of the simulation
    State state;

    // Timer for code profiling (i.e. to keep track
    // how much time was spend by which function).
    std::unique_ptr<dealii::ConditionalOStream> ccond;
    std::unique_ptr<dealii::TimerOutput> timer;

    std::vector<dealii::types::material_id> material_ids;

    bool apply_contact;

};



// This factory exists just to unify the interface.
template <int dim>
struct SampleFactory
{
    // Create a new Sample object
    static
    Sample<dim>*
    create (const std::vector<std::string>     &section_path,
            const FactoryTools::Specifications &specs,
            const std::string                  &unprocessed_input,
            MPI_Comm mpi_communicator = MPI_COMM_WORLD);

    // Return the keyword the Factory expects
    // when specs.get("type) is called.
    static
    std::string
    keyword ();
};



//---------------------- INLINE AND TEMPLATE FUNCTIONS -----------------------//


template <int dim>
inline
const dealii::DoFHandler<dim>&
Sample<dim>::
get_dof_handler () const
{
    return this->dof_handler;
}



template <int dim>
inline
const dealii::Mapping<dim>&
Sample<dim>::
get_mapping () const
{
    Assert (this->mapping, dealii::ExcNotInitialized());
    return *(this->mapping);
}

template <int dim>
inline
const dealii::FESystem<dim>&
Sample<dim>::
get_fe () const
{
    Assert (this->fe, dealii::ExcNotInitialized());
    return *(this->fe);
}



template <int dim>
inline
const Geometry<dim>&
Sample<dim>::
get_geometry () const
{
    Assert (this->geometry, dealii::ExcNotInitialized());
    return *(this->geometry);
}


// template <int dim>
// inline
// const Obstacle<dim> &
// Sample<dim>::
// get_obstacle () const;
// {

// }


template <int dim>
inline
void
Sample<dim>::
set_material_ids()
{    
    for (const auto & cell : this->tria.active_cell_iterators())
        if (!std::count(material_ids.begin(), material_ids.end(),cell->material_id()))
            material_ids.push_back(cell->material_id());
}

template <int dim>
inline
void
Sample<dim>::
set_obstacle_displacement(double disp)
{
    this->obstacle->update(disp);
}

 
template <int dim>
inline
const std::vector<dealii::types::material_id>
Sample<dim>::
 get_material_ids() const
{
    return material_ids;
}  

template <int dim>
inline
const ConstitutiveBase<dim>&
Sample<dim>::
get_constitutive_model (int material_id) const
{
    // Assert (this->constitutive_model, dealii::ExcNotInitialized());
    return *(this->constitutive_model_map.at(material_id));
}



template <int dim>
inline
boost::signals2::connection
Sample<dim>::
connect_mesh_loop (
        const CellWorker<dim>     &external_cell_worker,
        const BoundaryWorker<dim> &external_boundary_worker,
        const std::function<void(const CopyData&)> &external_copier,
        boost::signals2::signal<void()> &signal,
        const dealii::MeshWorker::AssembleFlags flags)
{
    // Assert (this->constitutive_model, dealii::ExcNotInitialized());


    return this->connect_mesh_loop (
                *(this->constitutive_model_map.at(3)),
                external_cell_worker,
                *(this->constitutive_model_map.at(3)),
                external_boundary_worker,
                external_copier,
                signal,
                flags);
}



template <int dim>
template <class CellDataProcessorType,
          class BoundaryDataProcessorType>
inline
boost::signals2::connection
Sample<dim>::
connect_mesh_loop (
        const CellDataProcessorType     &external_cell_data_processor,
        const CellWorker<dim>           &external_cell_worker,
        const BoundaryDataProcessorType &external_boundary_data_processor,
        const BoundaryWorker<dim>       &external_boundary_worker,
        const std::function<void(const CopyData&)> &external_copier,
        boost::signals2::signal<void()> &signal,
        const dealii::MeshWorker::AssembleFlags flags)
{
    using namespace dealii;

    // Check if CellDataProcessorType provides an evaluate-function with
    // valid signature (void evaluate(ScratchData<dim> &) const).
    static_assert(has_member_function_evaluate<
            void(CellDataProcessorType::*)(ScratchData<dim>&)const>::value,
            "DataProcessor must implement the member function "
            "<void evaluate(ScratchData<dim> &) const>");

    static_assert(has_member_function_get_needed_update_flags<
            dealii::UpdateFlags(CellDataProcessorType::*)()const>::value,
            "DataProcessor must implement the member function "
            "<dealii::UpdateFlags get_needed_update_flags() const>.");

    // Check if BoundaryDataProcessorType provides an evaluate-function with
    // valid signature (void evaluate(ScratchData<dim> &) const).
    static_assert(has_member_function_evaluate<
            void(BoundaryDataProcessorType::*)(ScratchData<dim>&)const>::value,
            "DataProcessor must implement the member function "
            "<void evaluate(ScratchData<dim> &) const>");

    static_assert(has_member_function_get_needed_update_flags<
            dealii::UpdateFlags(BoundaryDataProcessorType::*)()const>::value,
            "DataProcessor must implement the member function "
            "<dealii::UpdateFlags get_needed_update_flags() const>.");

    // Check if the sample copy data object is initialized.
    Assert (this->sample_copy_data, ExcNotInitialized());

    // An alias for the cell
    using CellIteratorType = decltype(this->dof_handler.begin_active());

    // Create a cell_worker compatible with the dealii::Meshworker interface.
    std::function<void(const CellIteratorType &cell,
                       ScratchData<dim>       &scratch_data,
                       CopyData               &copy_data)> cell_worker;

    // Create a cell_worker compatible with the dealii::Meshworker interface.
    std::function<void(const CellIteratorType &cell,
                       const unsigned int      face_no,
                       ScratchData<dim>       &scratch_data,
                       CopyData               &copy_data)> boundary_worker;

    // If we want to work on the cells, initialize the cell_worker.
    if (flags & dealii::MeshWorker::work_on_cells)
    {
        cell_worker =
            [&](const CellIteratorType &cell,
                ScratchData<dim>       &scratch_data,
                CopyData               &copy_data)
                {

                    if (this->state == State::failure)
                        return;

                    try
                    {
                        external_cell_worker.fill (
                                external_cell_data_processor,
                                this->locally_relevant_solution,
                                cell,
                                scratch_data,
                                copy_data);
                    }
                    catch (ExceptionBase &exec)
                    {
                        this->state = State::failure;
                        efilog(Verbosity::normal) << "CellWorker failed sample.h line 650."
                                              << std::endl;
                    }
                };
    }

    // If we want to work on the boundary, initialize the boundary_worker.
    if (flags & MeshWorker::work_on_boundary)
    {
        boundary_worker =
            [&](const CellIteratorType &cell,
                const unsigned int      face_no,
                ScratchData<dim>       &scratch_data,
                CopyData               &copy_data)
                {
                    if (this->state == State::failure)
                        return;

                    try
                    {
                        external_boundary_worker.fill (
                                external_boundary_data_processor,
                                this->locally_relevant_solution,
                                cell,
                                face_no,
                                scratch_data,
                                copy_data);
                    }
                    catch (ExceptionBase &exec)
                    {
                        this->state = State::failure;
                        efilog(Verbosity::normal) << "BoundaryWorker failed."
                                              << std::endl;
                    }
                };
    }

    // Create a sample scratch data object
    ScratchData<dim> external_sample_scratch_data (
            *(this->mapping),
            *(this->fe),
            *(this->qf_cell),
              external_cell_worker.get_needed_update_flags()
            | external_cell_data_processor.get_needed_update_flags (),
            *(this->qf_face),
              external_boundary_worker.get_needed_update_flags ()
            | external_boundary_data_processor.get_needed_update_flags ());

    // Create a new mesh loop
    std::function<void()> connectable_mesh_loop =
            [this, cell_worker, boundary_worker,
              &external_copier, flags, external_sample_scratch_data]() mutable
            {
                // Add a reference to the cell_data_storage to the
                // sample_scratch_data, such that it can be accessed
                // by the worker, constitutive, and other objects,
                // which have internal variables to store.
                ScratchDataTools::attach_history_data_storage (
                        external_sample_scratch_data,
                      *(this->cell_data_history_storage));
                ScratchDataTools::attach_tmp_history_data_storage (
                        external_sample_scratch_data,
                      *(this->tmp_cell_data_history_storage));

                // Set the time step size.
                ScratchDataTools::get_or_add_time_step_size(
                        external_sample_scratch_data) = this->time_step_size;

                mesh_loop (
                        this->dof_handler.begin_active(),
                        this->dof_handler.end(),
                        cell_worker,
                        external_copier,
                        external_sample_scratch_data,
                        *(this->sample_copy_data),
                        flags,
                        boundary_worker);

                // Perform all reduce the state such that the state
                // is consistent for all processors.
                this->all_reduce_state ();

                // just some output
                if (this->state == State::success)
                    efilog(Verbosity::debug) << "Registered mesh loop."
                                             << std::endl;
            };

    // Connect the new mesh loop and return the connection.
    return signal.connect (connectable_mesh_loop);
}


template <int dim>
inline
boost::signals2::connection
Sample<dim>::
connect_boundary_loop (
        const BoundaryWorker<dim>       &external_boundary_worker,
        const std::function<void(const CopyData&)> &external_copier,
        boost::signals2::signal<void()> &signal)
{

    // Check if the sample copy data object is initialized.
    Assert (this->sample_copy_data, dealii::ExcNotInitialized());

    // An alias for the cell
    using CellIteratorType = decltype(this->dof_handler.begin_active());

    // TODO: loop over boundary cells of different material types

    // Create a cell_worker compatible with the dealii::Meshworker interface.
    std::function<void(const CellIteratorType &cell,
                       ScratchData<dim>       &scratch_data,
                       CopyData               &copy_data)> dummy_cell_worker;

    // Create a cell_worker compatible with the dealii::Meshworker interface.
    std::function<void(const CellIteratorType &cell,
                       const unsigned int      face_no,
                       ScratchData<dim>       &scratch_data,
                       CopyData               &copy_data)>
    boundary_worker =
        [&](const CellIteratorType &cell,
            const unsigned int      face_no,
            ScratchData<dim>       &scratch_data,
            CopyData               &copy_data)
            {
                if (this->state == State::failure)
                    return;

                try
                {
                    // get cell material_id()
                    int material_id = cell->material_id();

                    external_boundary_worker.fill (
                            this->get_constitutive_model(material_id),
                            this->locally_relevant_solution,
                            cell,
                            face_no,
                            scratch_data,
                            copy_data);
                }
                catch (dealii::ExceptionBase &exec)
                {
                    this->state = State::failure;
                    efilog(Verbosity::normal) << "BoundaryWorker failed in"
                                                 "connected boundary loop."
                                              << std::endl;
                }
            };

    dealii::UpdateFlags updateFlags = external_boundary_worker.get_needed_update_flags ();
    for (const auto & cm: this->constitutive_model_map)
    {
        updateFlags = updateFlags | cm.second->get_needed_update_flags ();
    }

    // get update flags for all material models
    // Create a sample scratch data object
    ScratchData<dim> external_sample_scratch_data (
            *(this->mapping),
            *(this->fe),
            *(this->qf_cell),
              dealii::update_default,
            *(this->qf_face),
              updateFlags);

    // Create a new mesh loop
    std::function<void()> connectable_mesh_loop =
            [this, dummy_cell_worker, boundary_worker,
              &external_copier, external_sample_scratch_data]() mutable
            {
                // Add a reference to the cell_data_storage to the
                // sample_scratch_data, such that it can be accessed
                // by the worker, constitutive, and other objects,
                // which have internal variables to store.
                ScratchDataTools::attach_history_data_storage (
                        external_sample_scratch_data,
                      *(this->cell_data_history_storage));
                ScratchDataTools::attach_tmp_history_data_storage (
                        external_sample_scratch_data,
                      *(this->tmp_cell_data_history_storage));

                // Set the time step size.
                ScratchDataTools::get_or_add_time_step_size(
                        external_sample_scratch_data) = this->time_step_size;

                // Now run the mesh loop with the specified worker classes.
                mesh_loop (
                        this->dof_handler.begin_active(),
                        this->dof_handler.end(),
                        dummy_cell_worker,
                        external_copier,
                        external_sample_scratch_data,
                        *(this->sample_copy_data),
                        dealii::MeshWorker::work_on_boundary,
                        boundary_worker);

                // Perform all reduce the state such that the state
                // is consistent for all processors.
                this->all_reduce_state ();

                // just some output
                if (this->state == State::success)
                    efilog(Verbosity::debug) << "Registered mesh loop."
                                             << std::endl;
            };

    // Connect the new mesh loop and return the connection.
    return signal.connect (connectable_mesh_loop);
}



template <int dim>
template <class BoundaryDataProcessorType>
inline
boost::signals2::connection
Sample<dim>::
connect_boundary_loop (
        const BoundaryDataProcessorType &external_boundary_data_processor,
        const BoundaryWorker<dim>       &external_boundary_worker,
        const std::function<void(const CopyData&)> &external_copier,
        boost::signals2::signal<void()> &signal)
{
    // Check if BoundaryDataProcessorType provides an evaluate-function with
    // valid signature (void evaluate(ScratchData<dim> &) const).
    static_assert(has_member_function_evaluate<
            void(BoundaryDataProcessorType::*)(ScratchData<dim>&)const>::value,
            "DataProcessor must implement the member function "
            "<void evaluate(ScratchData<dim> &) const>.");

    static_assert(has_member_function_get_needed_update_flags<
            dealii::UpdateFlags(BoundaryDataProcessorType::*)()const>::value,
            "DataProcessor must implement the member function "
            "<dealii::UpdateFlags get_needed_update_flags() const>.");

    // Check if the sample copy data object is initialized.
    Assert (this->sample_copy_data, dealii::ExcNotInitialized());

    // An alias for the cell
    using CellIteratorType = decltype(this->dof_handler.begin_active());

    // TODO: loop over boundary cells of different material types

    // Create a cell_worker compatible with the dealii::Meshworker interface.
    std::function<void(const CellIteratorType &cell,
                       ScratchData<dim>       &scratch_data,
                       CopyData               &copy_data)> dummy_cell_worker;

    // Create a cell_worker compatible with the dealii::Meshworker interface.
    std::function<void(const CellIteratorType &cell,
                       const unsigned int      face_no,
                       ScratchData<dim>       &scratch_data,
                       CopyData               &copy_data)>
    boundary_worker =
        [&](const CellIteratorType &cell,
            const unsigned int      face_no,
            ScratchData<dim>       &scratch_data,
            CopyData               &copy_data)
            {
                if (this->state == State::failure)
                    return;

                try
                {
                    // get cell material_id()
                    int material_id = cell->material_id();

                    external_boundary_worker.fill (
                            this->get_constitutive_model(material_id),
                            this->locally_relevant_solution,
                            cell,
                            face_no,
                            scratch_data,
                            copy_data);
                }
                catch (dealii::ExceptionBase &exec)
                {
                    this->state = State::failure;
                    efilog(Verbosity::normal) << "BoundaryWorker failed in"
                                                 "connected boundary loop."
                                              << std::endl;
                }
            };

    dealii::UpdateFlags updateFlags = external_boundary_worker.get_needed_update_flags ();
    for (const auto & cm: this->constitutive_model_map)
    {
        updateFlags = updateFlags | cm.second->get_needed_update_flags ();
    }

    // get update flags for all material models
    // Create a sample scratch data object
    ScratchData<dim> external_sample_scratch_data (
            *(this->mapping),
            *(this->fe),
            *(this->qf_cell),
              dealii::update_default,
            *(this->qf_face),
              updateFlags);

    // Create a new mesh loop
    std::function<void()> connectable_mesh_loop =
            [this, dummy_cell_worker, boundary_worker,
              &external_copier, external_sample_scratch_data]() mutable
            {
                // Add a reference to the cell_data_storage to the
                // sample_scratch_data, such that it can be accessed
                // by the worker, constitutive, and other objects,
                // which have internal variables to store.
                ScratchDataTools::attach_history_data_storage (
                        external_sample_scratch_data,
                      *(this->cell_data_history_storage));
                ScratchDataTools::attach_tmp_history_data_storage (
                        external_sample_scratch_data,
                      *(this->tmp_cell_data_history_storage));

                // Set the time step size.
                ScratchDataTools::get_or_add_time_step_size(
                        external_sample_scratch_data) = this->time_step_size;

                // Now run the mesh loop with the specified worker classes.
                mesh_loop (
                        this->dof_handler.begin_active(),
                        this->dof_handler.end(),
                        dummy_cell_worker,
                        external_copier,
                        external_sample_scratch_data,
                        *(this->sample_copy_data),
                        dealii::MeshWorker::work_on_boundary,
                        boundary_worker);

                // Perform all reduce the state such that the state
                // is consistent for all processors.
                this->all_reduce_state ();

                // just some output
                if (this->state == State::success)
                    efilog(Verbosity::debug) << "Registered mesh loop."
                                             << std::endl;
            };

    // Connect the new mesh loop and return the connection.
    return signal.connect (connectable_mesh_loop);
}


// template <int dim>
// template <class BoundaryDataProcessorType>
// inline
// boost::signals2::connection
// Sample<dim>::
// connect_boundary_loop (
//         const BoundaryWorker<dim>       &external_boundary_worker,
//         const std::function<void(const CopyData&)> &external_copier,
//         boost::signals2::signal<void()> &signal)
// {

//     // Check if the sample copy data object is initialized.
//     Assert (this->sample_copy_data, dealii::ExcNotInitialized());

//     // An alias for the cell
//     using CellIteratorType = decltype(this->dof_handler.begin_active());

//     // TODO: loop over boundary cells of different material types

//     // Create a cell_worker compatible with the dealii::Meshworker interface.
//     std::function<void(const CellIteratorType &cell,
//                        ScratchData<dim>       &scratch_data,
//                        CopyData               &copy_data)> dummy_cell_worker;

//     // Create a cell_worker compatible with the dealii::Meshworker interface.
//     std::function<void(const CellIteratorType &cell,
//                        const unsigned int      face_no,
//                        ScratchData<dim>       &scratch_data,
//                        CopyData               &copy_data)>
//     boundary_worker =
//         [&](const CellIteratorType &cell,
//             const unsigned int      face_no,
//             ScratchData<dim>       &scratch_data,
//             CopyData               &copy_data)
//             {
//                 if (this->state == State::failure)
//                     return;

//                 try
//                 {
//                     // get cell material_id()
//                     int material_id = cell->material_id();

//                     external_boundary_worker.fill (
//                             this->get_constitutive_model(material_id),
//                             this->locally_relevant_solution,
//                             cell,
//                             face_no,
//                             scratch_data,
//                             copy_data);
//                 }
//                 catch (dealii::ExceptionBase &exec)
//                 {
//                     this->state = State::failure;
//                     efilog(Verbosity::normal) << "BoundaryWorker failed in"
//                                                  "connected boundary loop."
//                                               << std::endl;
//                 }
//             };

//     dealii::UpdateFlags updateFlags = external_boundary_worker.get_needed_update_flags ();
//     for (const auto & cm: this->constitutive_model_map)
//     {
//         updateFlags = updateFlags | cm.second->get_needed_update_flags ();
//     }

//     // get update flags for all material models
//     // Create a sample scratch data object
//     ScratchData<dim> external_sample_scratch_data (
//             *(this->mapping),
//             *(this->fe),
//             *(this->qf_cell),
//               dealii::update_default,
//             *(this->qf_face),
//               updateFlags);

//     // Create a new mesh loop
//     std::function<void()> connectable_mesh_loop =
//             [this, dummy_cell_worker, boundary_worker,
//               &external_copier, external_sample_scratch_data]() mutable
//             {
//                 // Add a reference to the cell_data_storage to the
//                 // sample_scratch_data, such that it can be accessed
//                 // by the worker, constitutive, and other objects,
//                 // which have internal variables to store.
//                 ScratchDataTools::attach_history_data_storage (
//                         external_sample_scratch_data,
//                       *(this->cell_data_history_storage));
//                 ScratchDataTools::attach_tmp_history_data_storage (
//                         external_sample_scratch_data,
//                       *(this->tmp_cell_data_history_storage));

//                 // Set the time step size.
//                 ScratchDataTools::get_or_add_time_step_size(
//                         external_sample_scratch_data) = this->time_step_size;

//                 // Now run the mesh loop with the specified worker classes.
//                 mesh_loop (
//                         this->dof_handler.begin_active(),
//                         this->dof_handler.end(),
//                         dummy_cell_worker,
//                         external_copier,
//                         external_sample_scratch_data,
//                         *(this->sample_copy_data),
//                         dealii::MeshWorker::work_on_boundary,
//                         boundary_worker);

//                 // Perform all reduce the state such that the state
//                 // is consistent for all processors.
//                 this->all_reduce_state ();

//                 // just some output
//                 if (this->state == State::success)
//                     efilog(Verbosity::debug) << "Registered mesh loop."
//                                              << std::endl;
//             };

//     // Connect the new mesh loop and return the connection.
//     return signal.connect (connectable_mesh_loop);
// }


template <int dim>
inline
void
Sample<dim>::
all_reduce_state ()
{
    unsigned int bitmask = (this->state == State::iterate)? 0x0 :
                           (this->state == State::success)? 0x1 :
                                                            0x2;
    // bitwise or
    bitmask = MPI::bor (bitmask,this->mpi_communicator);

    this->state = (bitmask & State::failure)? State::failure :
                  (bitmask & State::success)? State::success :
                                              State::iterate;
}



template <int dim>
inline
void
Sample<dim>::Signals::
disconnect_all_slots ()
{
    this->make_constraints.disconnect_all_slots ();
    this->make_constraints2.disconnect_all_slots ();
    this->pre_nonlinear_solve.disconnect_all_slots ();
    this->post_nonlinear_solve.disconnect_all_slots ();
}



template <int dim>
inline
Sample<dim>*
SampleFactory<dim>::
create (const std::vector<std::string>     &section_path,
        const FactoryTools::Specifications &specs,
        const std::string                  &unprocessed_input,
        MPI_Comm                            mpi_communicator)
{
    return new Sample<dim> (
            get_section_path_str(section_path) + "/"
                + FactoryTools::get_subsection_name (keyword(),specs),
            unprocessed_input,
            mpi_communicator);
}



template <int dim>
inline
std::string
SampleFactory<dim>::
keyword ()
{
    return "sample";
}


}// namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_LAB_SAMPLE_H_ */
