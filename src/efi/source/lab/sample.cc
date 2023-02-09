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
#include <deal.II/base/path_search.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/fe/fe_values_extractors.h>
//#include <deal.II/meshworker/mesh_loop.h>

// efi headers
#include <efi/lab/sample.h>
#include <efi/constitutive/constitutive_factory.h>
#include <efi/base/factory_tools.h>
#include <efi/factory/registry.h>
#include <efi/base/logstream.h>
#include <efi/base/postprocessor.h>
#include <efi/worker/scratch_data.h>
#include <efi/base/global_parameters.h>
#include <efi/grid/obstacle_factory.h>

namespace efi {


template <int dim>
Sample<dim>::
Sample (const std::string &subsection_name,
        const std::string &unprocessed_input,
        MPI_Comm mpi_communicator)
:
dealii::ParameterAcceptor (subsection_name),
mpi_communicator (mpi_communicator),
tria (mpi_communicator),
solver_control (get_section_path_str(this->get_section_path())
                + "/" + "nonlinear_solver_control",
                unprocessed_input),
time_step_size(0),
elapsed_time(0),
state (State::iterate)
{
    using namespace dealii;

    std::istringstream iss (unprocessed_input);
    this->instantiate (iss);

    //TODO move this to a sparate function
    std::ostream*  tStream=NULL;
    if (dealii::deallog.has_file())
    {
      tStream= &dealii::deallog.get_file_stream();
    }else
    {
      tStream= &std::cout;
    }
    dealii::ConditionalOStream* oStr=new dealii::ConditionalOStream
      (*tStream, MPI::is_root(this->mpi_communicator));
    this->ccond.reset(oStr);
    dealii::TimerOutput* tOut= new dealii::TimerOutput(this->mpi_communicator,*(this->ccond),
      dealii::TimerOutput::summary, dealii::TimerOutput::cpu_and_wall_times );
    this->timer.reset(tOut);

    efilog(Verbosity::verbose) << "New Sample created ("+ subsection_name +")."
                               << std::endl;
}



template <int dim>
Sample<dim>::
~Sample ()
{
    using namespace dealii;
    efilog(Verbosity::debug) << "Sample deleted"
      << std::endl;
    if (efi::MPI::is_root (this->mpi_communicator))
    {
        static std::ofstream tFile;

	std::map< std::string, double > timeMap=
	  (*(this->timer)).get_summary_data(TimerOutput::total_cpu_time);  
	std::string k;
	double v;
	/*tFile.open(this->output_directory.string()+
		    std::string(1,this->output_directory.separator)+
		    "cpu_timing", std::ios::out);
	*/
	tFile.open("cpu_timing",std::ios::out);
	for (const auto &p : timeMap){ 
	//for (const auto &[k, v] : timeMap)
	  std::tie(k,v) = p; 
	  tFile << k <<","<< v << std::endl;
	}
	tFile.close();
    }    
    this->timer.reset(NULL); 
    
    this->dof_handler.clear();
}



template <int dim>
void
Sample<dim>::
declare_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    //TimerOutput::Scope timer_section(*(this->timer), EFI_PRETTY_FUNCTION);

    // finite element related parameters
    prm.enter_subsection ("finite element");
        prm.declare_entry ("polynomial degree","1",Patterns::Integer());
    prm.leave_subsection ();

    // quadrature related parameters
    prm.enter_subsection ("quadrature");
        prm.declare_entry ("quadrature type","gauss",
                Patterns::Selection(QuadratureSelector<dim>::get_quadrature_names()),
                "options: " + QuadratureSelector<dim>::get_quadrature_names());
        prm.declare_entry ("quadrature points per space direction","auto",
                Patterns::Anything(),
                "options: 'auto' or integer values > 0");
    prm.leave_subsection ();

    // quadrature related parameters
    prm.enter_subsection ("contact");
        prm.declare_entry ("apply contact","false",Patterns::Bool());
    prm.leave_subsection ();

    // just some output
    efilog(Verbosity::verbose) << "Sample finished declaring parameters"
                               << std::endl;
 }



template <int dim>
void
Sample<dim>::
parse_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    //TimerOutput::Scope timer_section(*(this->timer), EFI_PRETTY_FUNCTION);

    // Check if the worker and constitutive model
    // have been initialized.
    Assert (this->cell_worker, ExcNotInitialized ());
    Assert (this->boundary_worker, ExcNotInitialized ());

    // finite element related parameters
    prm.enter_subsection ("finite element");
        int pdegree = prm.get_integer ("polynomial degree");

        // initialize the FESystem and the
        this->fe.reset      (new FESystem<dim>(FE_Q<dim>(pdegree),dim));
        this->mapping.reset (new MappingQGeneric<dim>(pdegree));
    prm.leave_subsection ();

    // quadrature related parameters
    prm.enter_subsection ("quadrature");
        auto quadrature_str =  prm.get ("quadrature type");
        auto order =
                (prm.get ("quadrature points per space direction") == "auto"?
                         (quadrature_str=="gauss"? pdegree + 1 : 0) :
                          Utilities::string_to_int(
                          prm.get ("quadrature points per space direction")));

        this->qf_cell.reset (new QuadratureSelector<dim>  (quadrature_str,order));
        this->qf_face.reset (new QuadratureSelector<dim-1>(quadrature_str,order));
    prm.leave_subsection ();

    // contact related parameters
    prm.enter_subsection ("contact");
        this->apply_contact =  prm.get_bool ("apply contact");

        boost::filesystem::path input_directory = 
                GlobalParameters::get_input_directory();

        // input directory
        std::string directory (input_directory.string()
                            + std::string(1,input_directory.separator));

    prm.leave_subsection ();

    // just some output
    efilog(Verbosity::verbose) << "Sample finished parsing parameters"
                               << std::endl;
}


template <int dim>
bool
Sample<dim>::
run (const std::map<dealii::types::global_dof_index,double> &prescribed,
     const double dt)
{
    using namespace dealii;

    // Set the time step size
    this->time_step_size = dt;
    efi::efilog(Verbosity::normal) << "dt: " << dt << std::endl;

    bool output_enabled = GlobalParameters::paraview_output_enabled();


    // All changes are written to a temporary
    // vector such that in case of failure, the
    // original data can be restored from
    // the locally_owned solution.
    LA::MPI::Vector tmp_locally_owned_solution (this->locally_owned_solution);

    this->locally_owned_solution.compress(VectorOperation::insert);

    dealii::types::global_dof_index dof;
    double value;
    // Re-sets solution value to boundary value
    this->contact_constraints.reinit(this->locally_relevant_dofs);
    this->active_set.clear();
    this->active_set.set_size(this->dof_handler.n_dofs());
    for (auto &el : prescribed)
    {
        std::tie(dof,value) = el;

        if (this->locally_owned_dofs.is_element (dof))
            {
                this->locally_owned_solution.set(1, &dof, &value);
                this->active_set.add_index(dof);
            }

    }
    this->contact_constraints.close();
    this->contact_constraints.merge(this->constraints);

    this->locally_owned_solution.compress(VectorOperation::insert);
    
    this->solve_nonlinear ();

    this->compute_residual();
    // dealii::IndexSet boundary_set;
    // this->compute_contact_force(boundary_set);

    if (this->state == State::success)
    {
        this->elapsed_time += dt;
        if (output_enabled)
            this->write_output(this->times_and_names.size(),this->elapsed_time);
        return true;
    }
    else
    {
        // restore original solution
        this->locally_owned_solution = tmp_locally_owned_solution;
        return false;
    }
}

template <int dim>
void
Sample<dim>::
instantiate (std::istream &unprocessed_input)
{
    using namespace dealii;

    unsigned int n_mpi_processes =
            Utilities::MPI::n_mpi_processes(mpi_communicator);

    //TimerOutput::Scope timer_section(*(this->timer), EFI_PRETTY_FUNCTION);

    FactoryTools::action_type create_contsitutive
    = [&] (const FactoryTools::Specifications &specs,
           const std::string                  &unprocessed_input) -> void
    {

        int material_id = specs.get_integer("material_id");
        std::string type = specs.get("type");

        
        efilog(Verbosity::quiet) << "Material_id: " << material_id;
        efilog(Verbosity::quiet) << ", Type: " << type << std::endl;

        this->constitutive_model_map.emplace (
                specs.get_integer("material_id"),
                ConstitutiveFactory<dim>::create (
                        this->get_section_path(), specs, unprocessed_input));
    };

    FactoryTools::action_type create_geometry
    = [&] (const FactoryTools::Specifications &specs,
           const std::string                  &unprocessed_input) -> void
    {
        this->geometry.reset (
                GeometryFactory<dim>::create (
                        this->get_section_path(), specs, unprocessed_input));
    };


    // Put all actions into a map.
    std::map<std::string,FactoryTools::action_type> actions;

    actions[ConstitutiveFactory<dim>::keyword()] = create_contsitutive;
    actions[GeometryFactory<dim>::keyword()]     = create_geometry;

    if (this->apply_contact )
    {
        FactoryTools::action_type create_obstacle
        = [&] (const FactoryTools::Specifications &specs,
            const std::string                  &unprocessed_input) -> void
        {
            this->obstacle.reset (
                    ObstacleFactory<dim>::create (
                            this->get_section_path(), specs, unprocessed_input));
        };
        actions[ObstacleFactory<dim>::keyword()]     = create_obstacle;
    }

    // Setup the Sample<dim> object using an input
    // parameter file and the map of predefined
    // factories. Whenever ParameterTools::setup()
    // is able to parse a key in the input parameter
    // file that matches an entry in factories, then
    // the corresponding.
    FactoryTools::apply (actions, unprocessed_input);

    // set the workers:
    this->cell_worker.reset (new CellWorker<dim>());
    this->boundary_worker.reset (new BoundaryWorker<dim>());

    if (this->apply_contact )
    {
    // Initialize contact geometry object
    dealii::types::boundary_id contact_boundary_id = 5;
    this->obstacle->set_contact_boundary(contact_boundary_id);
    }

    // just some output
    efilog(Verbosity::verbose) << "Sample parses unprocessed input."
                               << std::endl;
}



template <int dim>
void
Sample<dim>::
initialize ()
{
    using namespace dealii;

    TimerOutput::Scope timer_section(*(this->timer), EFI_PRETTY_FUNCTION);

    this->tria.clear ();
    this->constraints.clear ();
    this->dof_handler.clear ();
    this->locally_owned_dofs.clear ();
    this->locally_relevant_dofs.clear ();

    this->signals.disconnect_all_slots ();

    Assert (this->boundary_worker,    ExcNotInitialized ());
    Assert (this->cell_worker,        ExcNotInitialized ());
    // Assert (this->constitutive_model, ExcNotInitialized ());
    // Assert (this->constitutive_model_map, ExcNotInitialized ());
    Assert (this->geometry,           ExcNotInitialized ());
    Assert (this->mapping,            ExcNotInitialized ());



    dealii::UpdateFlags updateFlags = this->cell_worker->get_needed_update_flags ();
    for (const auto & cm: this->constitutive_model_map)
    {
        updateFlags = updateFlags | cm.second->get_needed_update_flags ();            
    }

    // After we have gathered all information
    // create the copy and scratch data objects.
    this->sample_copy_data.reset (new CopyData ());
    this->sample_scratch_data.reset (new ScratchData<dim> (
            *(this->mapping),
            *(this->fe),
            *(this->qf_cell),
            updateFlags,
            *(this->qf_face),
              this->boundary_worker->get_needed_update_flags ()));

    this->cell_data_history_storage.reset(new GeneralCellDataStorage ());
    this->tmp_cell_data_history_storage.reset(new GeneralCellDataStorage ());

    // After we have gathered all information
    // create the triangulation.
    this->geometry->create_triangulation (this->tria);

    this->dof_handler.initialize (this->tria, *(this->fe));

    // The geometry requires the dof_handler to be initialized.
    this->geometry->connect_constraints (*this);

    if (this->apply_contact)
        this->obstacle->create();

    // Find the set of locally owned dofs.
    this->locally_owned_dofs = this->dof_handler.locally_owned_dofs ();

    // Find the set of locally relevant dofs.
    extract_locally_relevant_dofs (
            this->dof_handler,
            this->locally_relevant_dofs);

    // compute the ghosted locally_relevant_solution vector
    this->locally_relevant_solution = this->locally_owned_solution;

    this->reinit_constraints();

    this->reset();

    efilog(Verbosity::very_verbose) << "Sample<dim>::constraints has  "
                                    << this->constraints.n_constraints()
                                    << " constraints"
                                    << std::endl;

    efilog(Verbosity::very_verbose) << "Sample<dim>::dof_handler has  "
                                    << this->dof_handler.n_dofs()
                                    << " degrees of freedom"
                                    << std::endl;
    // just some output
    efilog(Verbosity::verbose) << "Sample finished initialization." << std::endl;
    
}

template <int dim>
void
Sample<dim>::
reset()
{
     // Initialize the vectors (locally owned and locally relevant)
    this->locally_relevant_solution.reinit (
            this->locally_owned_dofs,
            this->locally_relevant_dofs,
            this->mpi_communicator);

    this->locally_owned_solution.reinit (
            this->locally_owned_dofs,
            this->mpi_communicator);

    this->system_increment.reinit (
            this->locally_owned_dofs,
            this->mpi_communicator);

    this->system_vector.reinit (
            this->locally_owned_dofs,
            this->mpi_communicator);

    this->uncondensed_rhs.reinit(
            this->locally_owned_dofs,
            this->mpi_communicator);

    this->diag_mass_matrix_vector.reinit (
            this->locally_owned_dofs,
            this->mpi_communicator);

    // Initialize the history data
    this->cell_data_history_storage->initialize (
            this->dof_handler.active_cell_iterators());
    this->tmp_cell_data_history_storage->initialize (
            this->dof_handler.active_cell_iterators());
}

template <int dim>
void
Sample<dim>::
reinit_constraints ()
{
    // Create the constraints.
    this->constraints.clear ();
    this->constraints.reinit (this->locally_relevant_dofs);

    // dealii::DoFTools::make_hanging_node_constraints (this->dof_handler,
    //                                                  this->constraints);

    this->signals.make_constraints (this->constraints);

    this->constraints.close();

    efilog(Verbosity::very_verbose) << "Sample<dim>::constraints has  "
                                    << this->constraints.n_constraints()
                                    << " constraints"
                                    << std::endl;

    efilog(Verbosity::very_verbose) << "Sample<dim>::dof_handler has  "
                                    << this->dof_handler.n_dofs()
                                    << " degrees of freedom"
                                    << std::endl;

    // Notify us know if the constraints were.
    efilog(Verbosity::verbose) << "Sample finished initializing constraints."
                               << std::endl;

    // Whenever the constraints change, we must also reitialize the
    // sparsity pattern of the system matrix.
    this->reinit_sparsity ();
}



template <int dim>
void
Sample<dim>::
reinit_sparsity ()
{
    using namespace dealii;

    Assert (this->boundary_worker,     ExcNotInitialized());
    Assert (this->cell_worker,         ExcNotInitialized());
    Assert (this->sample_scratch_data, ExcNotInitialized());
    Assert (this->sample_copy_data,    ExcNotInitialized());

    // number of degrees of freedom
    const types::global_dof_index n_dofs = this->dof_handler.n_dofs();

    dealii::DynamicSparsityPattern dynamic_sparsity_pattern (n_dofs,n_dofs);

    using CellIteratorType = decltype(this->dof_handler.begin_active());

    std::vector<std::vector<dealii::types::global_dof_index>>
            sample_sparsity_copy_data;

    auto cell_woker =
            [&](const CellIteratorType &cell,
                ScratchData<dim>       &/*dummy*/,
                std::vector<std::vector<dealii::types::global_dof_index>>
                    &sparsity_copy_data)
                {
                    sparsity_copy_data.emplace_back (
                            cell->get_fe().dofs_per_cell);
                    cell->get_dof_indices (sparsity_copy_data.back());
                };

    auto boundary_woker =
            [&](const CellIteratorType &cell,
                const unsigned int      /*dummy*/,
                ScratchData<dim>       &/*dummy*/,
                std::vector<std::vector<dealii::types::global_dof_index>>
                    &sparsity_copy_data)
                {
                    sparsity_copy_data.emplace_back (
                            cell->get_fe().dofs_per_cell);
                    cell->get_dof_indices (sparsity_copy_data.back());
                };

    auto copier =
            [&](const std::vector<std::vector<dealii::types::global_dof_index>>
                    &sparsity_copy_data)
                {
                    for (const auto &local_dof_indices : sparsity_copy_data)
                    {
                        this->constraints.add_entries_local_to_global (
                                local_dof_indices,
                                dynamic_sparsity_pattern,
                                false);
                    }
                };

    mesh_loop (this->dof_handler.begin_active(),
               this->dof_handler.end(),
               cell_woker,
               copier,
               *(this->sample_scratch_data),
               sample_sparsity_copy_data,
               MeshWorker::assemble_own_cells
               | MeshWorker::assemble_boundary_faces ,
               boundary_woker);

    // Since running in parallel, the sparsity pattern
    // the following function is necessary.
    SparsityTools::distribute_sparsity_pattern(
            dynamic_sparsity_pattern,
            this->dof_handler.n_locally_owned_dofs_per_processor(),
            this->mpi_communicator,
            this->locally_relevant_dofs);

    // Reset the system matrix.
    this->system_matrix.clear();

    // Initialize the system matrix.
    this->system_matrix.reinit(this->locally_owned_dofs,
                               this->locally_owned_dofs,
                               dynamic_sparsity_pattern,
                               this->mpi_communicator);

    this->diag_mass_matrix_vector.reinit(this->locally_owned_dofs,this->mpi_communicator);

    // Free storage.
    dynamic_sparsity_pattern.reinit (0,0);
    if (this->apply_contact )
    {
        // Assemble mass matrix
        efilog(Verbosity::verbose) << "Assembling mass matrix"
                                << std::endl;
        LA::MPI::SparseMatrix &mass_matrix = this->system_matrix;
        // Quadrature<dim-1> face_quadrature_formula(this->fe->get_unit_face_support_points());

        QGaussLobatto<dim-1> face_quadrature_formula(this->fe->degree + 1);
        FEFaceValues<dim> fe_values_face(*fe,
                                        face_quadrature_formula,
                                        update_values | update_JxW_values);

        const unsigned int dofs_per_cell = this->fe->n_dofs_per_cell();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        const FEValuesExtractors::Vector displacement(0);

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        // efilog(Verbosity::verbose) << "contact boundary ID = "
        //                            << (this->obstacle->get_contact_boundary_id())
        //                            << std::endl;
        for (const auto & cell: dof_handler.active_cell_iterators())
            if (cell->is_locally_owned())
                for (const auto & face: cell->face_iterators())
                    if (face->at_boundary() && ((face->boundary_id() == 501) || (face->boundary_id() == 503) ))
                        {
                            fe_values_face.reinit(cell, face);
                            cell_matrix = 0;

                            for (unsigned int q_point = 0; q_point < n_face_q_points; q_point++)
                                for (unsigned int i = 0; i < dofs_per_cell; i++)
                                    {
                                        cell_matrix(i,i) += (fe_values_face[displacement].value(i, q_point) * 
                                                            fe_values_face[displacement].value(i, q_point) *
                                                            fe_values_face.JxW(q_point));
                                    }
                            cell->get_dof_indices(local_dof_indices);

                            for (unsigned int i = 0; i< dofs_per_cell; i++)
                                mass_matrix.add(local_dof_indices[i], local_dof_indices[i], cell_matrix(i,i));
                        }
        mass_matrix.compress(VectorOperation::add);

        const unsigned int start = this->system_vector.local_range().first;
        const unsigned int end = this->system_vector.local_range().second;
        for (unsigned int j = start; j < end; j++)
            this->diag_mass_matrix_vector(j) = mass_matrix.diag_element(j);
            
        this->diag_mass_matrix_vector.compress(VectorOperation::insert);
        mass_matrix = 0;

        this->active_set.set_size(this->dof_handler.n_dofs());

    }

    // Notify us know if this was successful.
    efilog(Verbosity::verbose) << "Sample finished initializing the"
                                  " sparsity pattern of the system_matrix."
                               << std::endl;
}


template <int dim>
void    
Sample<dim>::
apply_contact_constraints ()
{
    using namespace dealii;


    this->contact_constraints.reinit(this->locally_relevant_dofs);

    this->active_set.clear();

if (this->apply_contact)

    {

    TimerOutput::Scope timer_section(*(this->timer), EFI_PRETTY_FUNCTION);

    FEValuesExtractors::Vector displacements(0);

    std::vector<bool> touched_dofs(this->dof_handler.n_dofs(),false);
    LA::MPI::Vector distributed_solution(
            this->locally_owned_dofs,
            this->mpi_communicator);
            
    distributed_solution = this->locally_relevant_solution;
            
    LA::MPI::Vector diag_mm_vector_relvent(   
            this->locally_relevant_dofs,
            this->mpi_communicator);

    diag_mm_vector_relvent = this->diag_mass_matrix_vector;
            
    LA::MPI::Vector lambda(   
            this->locally_relevant_dofs,
            this->mpi_communicator);

    lambda = this->uncondensed_rhs;
            
    LA::MPI::Vector residual(   
            this->locally_relevant_dofs,
            this->mpi_communicator);

    residual = this->system_vector;
    // QGaussLobatto<dim-1> face_quadrature(this->fe->degree + 1);
    Quadrature<dim-1> face_quadrature(this->fe->get_unit_face_support_points());
    FEFaceValues<dim> fe_values_face(*fe, face_quadrature, update_quadrature_points |
                                           update_normal_vectors |
                                           update_gradients);

    const unsigned int dofs_per_face = this->fe->n_dofs_per_face();
    const unsigned int n_face_q_points = face_quadrature.size();

    std::vector<types::global_dof_index> dof_indices(dofs_per_face);

    bool display = true;
    
    double c = this->obstacle->get_penalty_parameter();
    efi::efilog(Verbosity::verbose) << "penalty parameter: " << c << std::endl;
    for (const auto & cell : dof_handler.active_cell_iterators())
        if(!cell->is_artificial())
            for (const auto & face : cell->face_iterators())
                if (face->at_boundary())
                        {
                        //   efi::efilog(Verbosity::verbose) << "Face at boundary ==  "<< face->boundary_id() << std::endl;  
                        if (face->boundary_id() == this->obstacle->get_contact_boundary_id())
                        {
                        // efi::efilog(Verbosity::verbose) << "Face at boundary ==  "<< this->obstacle->get_contact_boundary_id() << std::endl;
                        fe_values_face.reinit(cell, face);
                        face->get_dof_indices(dof_indices);
                        std::vector<Tensor<2, dim>> grad_u(n_face_q_points);
                        fe_values_face[displacements].get_function_gradients(this->locally_relevant_solution, grad_u);
                        for (unsigned int q_point = 0; q_point<n_face_q_points; q_point += dim)
                        {
                            const int index = dof_indices[q_point];                            
                            if (!touched_dofs[index])
                            { 
                                std::vector<types::global_dof_index> vertex_dof_indices(&dof_indices[q_point],(&dof_indices[q_point])+dim);
                                if (!constraints.is_constrained(vertex_dof_indices[0]) 
                                    && !constraints.is_constrained(vertex_dof_indices[1])
                                    && !constraints.is_constrained(vertex_dof_indices[2]))
                                {
                                    // std::cout << "face_vertex: index " << index << std::endl;
                                    // Need to calculate normal vector of displaced body 
                                    //      n = J*F_inv*N (N = undeformed normal vector, J = det(F))
                                    Tensor<2,dim> F = Physics::Elasticity::Kinematics::F(grad_u[q_point]); // deformation gradient at quad point
                                    Tensor<2,dim> F_inv = invert(F);
                                    double J = determinant(F);
                                    Tensor<1,dim> normal_vector =  fe_values_face.normal_vector(q_point);

                                    Tensor<1,dim> def_normal_vector = J*transpose(F_inv)*normal_vector;
                                    def_normal_vector = def_normal_vector/def_normal_vector.norm();

                                    touched_dofs[index] = true;
                                    Point<dim> support_pnt = fe_values_face.quadrature_point(q_point);
                                    Point<dim> slave_pnt = support_pnt;

                                    Tensor<1,dim> def_at_point;
                                    Tensor<1,dim> lamda_at_point;
                                    Tensor<1,dim> mass_matrix_at_point;
                                    Tensor<1,dim> contact_force;
                                    Tensor<1,dim> residual_at_point;
                                    for (unsigned int v_index=0; v_index<dim;v_index++)
                                    {
                                        const unsigned int vertex_dof_index = vertex_dof_indices[v_index];

                                        slave_pnt(v_index) += this->locally_relevant_solution(vertex_dof_index);
                                        def_at_point[v_index] = this->locally_relevant_solution(vertex_dof_index);
                                        contact_force[v_index] = lambda(vertex_dof_index)/diag_mm_vector_relvent(vertex_dof_index);
                                        lamda_at_point[v_index] = lambda(vertex_dof_index);
                                        mass_matrix_at_point[v_index] = diag_mm_vector_relvent(vertex_dof_index);
                                        residual_at_point[v_index] = residual(vertex_dof_index);
                                    }

                                    double u_dot_n = scalar_product(def_at_point,def_normal_vector);
                                    double force_at_point = scalar_product(contact_force,def_normal_vector);

                                    // Find min gap distance (and mater _pnt) to contact surface 
                                    Point<dim> master_pnt; 
                                    double gap = this->obstacle->find_master_pnt(slave_pnt, master_pnt, false);
                                    if (gap>1e5)
                                        master_pnt = support_pnt;
                                    // efi::efilog(Verbosity::verbose) << "gap: " 
                                    //                                 << gap << std::endl;
                                    // efi::efilog(Verbosity::verbose) << "contact force: " 
                                    //                                 << force_at_point << std::endl;
                                    // std::cout << "(support_pnt): " 
                                    //                                 << (support_pnt) << std::endl;

                                    // if ( ((u_dot_n-gap)) > -1e-6)
                                    for (unsigned int d =0; d<dim; d++)
                                    {
                                        // if (force_at_point[d]+c*(def_at_pont[d]-gap))
                                        {    
                                            if (display)
                                            {
                                
                                                std::cout << "gap: " 
                                                                                << gap << std::endl;
                                
                                                std::cout << "residual from system vector: " 
                                                                                << residual_at_point << std::endl; 

                                                std::cout << "contact force: " 
                                                                                << force_at_point << std::endl;

                                                std::cout << "displacement: " 
                                                                            << u_dot_n << std::endl;

                                                std::cout << "support point: " 
                                                                            << support_pnt << std::endl; 

                                                std::cout << "slave point: " 
                                                                            << slave_pnt << std::endl; 

                                                std::cout << "master point: " 
                                                                                << master_pnt << std::endl; 

                                                std::cout << "constrained solution: " 
                                                                                << (master_pnt - support_pnt) << std::endl;                                         
                                                display = false; 
                                            }
                                            for (unsigned int i = 0; i<vertex_dof_indices.size(); i++)  
                                            {
                                                unsigned int idx = vertex_dof_indices[i];
                                                Tensor<1,dim> constrained_solution = master_pnt - support_pnt;
                                                
                                                // Add constraint
                                                this->contact_constraints.add_line(idx);
                                                this->contact_constraints.set_inhomogeneity(idx, 0);
                                                // Change distibuted solution
                                                distributed_solution(idx) = constrained_solution[i];
                                                // Add to active_set
                                                this->active_set.add_index(idx); 
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        }
                    }
    
    distributed_solution.compress(VectorOperation::insert);
    this->locally_owned_solution = distributed_solution;

    efi::efilog(Verbosity::normal) << "ACC | Active set size: " 
                                << Utilities::MPI::sum( (active_set & locally_owned_dofs).n_elements(), mpi_communicator) << std::endl;
    }
    this->contact_constraints.close();
    this->contact_constraints.merge(this->constraints);    
}

template <int dim>
void
Sample<dim>::
compute_residual()
{

    
    using namespace dealii;

    TimerOutput::Scope timer_section(*(this->timer), EFI_PRETTY_FUNCTION);

    Assert (this->boundary_worker,     ExcNotInitialized());
    Assert (this->cell_worker,         ExcNotInitialized());
    // Assert (this->constitutive_model,  ExcNotInitialized());
    // Assert (this->constitutive_model_map,  ExcNotInitialized());
    Assert (this->sample_scratch_data, ExcNotInitialized());
    Assert (this->sample_copy_data,    ExcNotInitialized());

    // reset system matrix and rhs
    system_matrix = 0;
    uncondensed_rhs = 0;
    
    this->empty_constraints.reinit(this->locally_relevant_dofs);
    
    auto u_mask  = Extractor<dim>::displacement_mask();

    dealii::types::boundary_id id = 2;
    DoFTools::make_zero_boundary_constraints(this->dof_handler, id, this->empty_constraints, u_mask);
        
    this->empty_constraints.close();
    // Loop over material types and set up system?
    using CellIteratorType = decltype(this->dof_handler.begin_active());
    auto cell_woker =
            [&](const CellIteratorType &cell,
                ScratchData<dim>       &scratch_data,
                CopyData               &copy_data)
                {
                    if (this->state == State::failure)
                        return;
                    try
                    {
                        this->cell_worker->fill (
                              *(this->constitutive_model_map.at(cell->material_id())),
                                this->locally_relevant_solution,
                                cell,
                                scratch_data,
                                copy_data);
                    }
                    catch (ExceptionBase &exec)
                    {
                        this->state = State::failure;
                        efilog(Verbosity::normal) << "CellWorker failed sample.cc line 975."
                                                  << std::endl;
                    }
                };


    auto boundary_woker =
            [&](const CellIteratorType &cell,
                const unsigned int      face_no,
                ScratchData<dim>       &scratch_data,
                CopyData               &copy_data)
                {
                    if (this->state == State::failure)
                        return;
                    try
                    {
                        this->boundary_worker->fill (
                                DataProcessorDummy (),
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

    
    auto copier = create_residual_data_copier (
                        this->uncondensed_rhs,
                        this->system_matrix,
                        this->state,
                        this->empty_constraints);

    mesh_loop (this->dof_handler.begin_active(),
               this->dof_handler.end(),
               cell_woker,
               copier,
               *(this->sample_scratch_data),
               *(this->sample_copy_data),
               MeshWorker::assemble_own_cells | 
               MeshWorker::assemble_boundary_faces,
               boundary_woker);

    // }
    // Perform all reduce the state such that the state
    // is consistent for all processors.
    this->all_reduce_state ();

    system_matrix.compress(VectorOperation::add);
    uncondensed_rhs.compress(VectorOperation::add);

    // just some output
    if (this->state == State::success)
        efilog(Verbosity::normal) << "CR | " << std::endl;
}


template <int dim>
void
Sample<dim>::
assemble ()
{
    using namespace dealii;

    TimerOutput::Scope timer_section(*(this->timer), EFI_PRETTY_FUNCTION);

    Assert (this->boundary_worker,     ExcNotInitialized());
    Assert (this->cell_worker,         ExcNotInitialized());
    // Assert (this->constitutive_model,  ExcNotInitialized());
    // Assert (this->constitutive_model_map,  ExcNotInitialized());
    Assert (this->sample_scratch_data, ExcNotInitialized());
    Assert (this->sample_copy_data,    ExcNotInitialized());

    // reset system matrix and rhs
    system_matrix = 0;
    system_vector = 0;

    
    using CellIteratorType = decltype(this->dof_handler.begin_active());

    // Loop over material types and set up system?
    auto cell_woker =
            [&](const CellIteratorType &cell,
                ScratchData<dim>       &scratch_data,
                CopyData               &copy_data)
                {
                    if (this->state == State::failure)
                        return;
                    try
                    {
                        this->cell_worker->fill (
                              *(this->constitutive_model_map.at(cell->material_id())),
                                this->locally_relevant_solution,
                                cell,
                                scratch_data,
                                copy_data);
                    }
                    catch (ExceptionBase &exec)
                    {
                        this->state = State::failure;
                        efilog(Verbosity::normal) << "CellWorker failed HERE "
                                                  << std::endl;
                    }
                };


    auto boundary_woker =
            [&](const CellIteratorType &cell,
                const unsigned int      face_no,
                ScratchData<dim>       &scratch_data,
                CopyData               &copy_data)
                {
                    if (this->state == State::failure)
                        return;
                    try
                    {
                        this->boundary_worker->fill (
                                DataProcessorDummy (),
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

    
    auto copier = create_assembly_data_copier (
                        this->system_vector,
                        this->system_matrix,
                        this->state,
                        this->contact_constraints);

    mesh_loop (this->dof_handler.begin_active(),
               this->dof_handler.end(),
               cell_woker,
               copier,
               *(this->sample_scratch_data),
               *(this->sample_copy_data),
               MeshWorker::assemble_own_cells
                | MeshWorker::assemble_boundary_faces
               | MeshWorker::cells_after_faces,
               boundary_woker);

    // }
    // Perform all reduce the state such that the state
    // is consistent for all processors.
    this->all_reduce_state ();

    system_matrix.compress(VectorOperation::add);
    system_vector.compress(VectorOperation::add);

    // just some output
    if (this->state == State::success)
        efilog(Verbosity::normal) << "ASM | " << std::flush;
}



template <int dim>
void
Sample<dim>::
solve_linear ()
{
    using namespace dealii;

    TimerOutput::Scope timer_section(*(this->timer), EFI_PRETTY_FUNCTION);

    ReductionControl linear_solver_control (
            10000, 1e-10, 1e-12, /*log_history*/ false, /*log_result*/ false);

    if (this->solver_control.get_linear_solver_type() == "direct")
    {
        try
        {
            LA::SolverDirect::AdditionalData additional_data (false,"Amesos_Mumps");

            LA::SolverDirect solver (linear_solver_control, additional_data);

            solver.initialize (this->system_matrix);
            solver.solve (this->system_increment, this->system_vector);
        }
        catch (ExceptionBase & e)
        {
            efilog(Verbosity::debug) << e.what() << std::endl;
            this->state = State::failure;
        }
    }
    else if (this->solver_control.get_linear_solver_type() == "CG")
    {
        try
        {
            LA::PreconditionBlockJacobi::AdditionalData additional_data;
            LA::PreconditionBlockJacobi preconditioner;

            LA::SolverCG solver (linear_solver_control);

            preconditioner.initialize(this->system_matrix, additional_data);
            solver.solve(this->system_matrix,
                         this->system_increment,
                         this->system_vector,
                         preconditioner);
        }
        catch (ExceptionBase & e)
        {
            efilog(Verbosity::debug) << e.what() << std::endl;
            this->state = State::failure;
        }
    }
    else if (this->solver_control.get_linear_solver_type() == "GMRES")
    {
        try
        {
            LA::PreconditionBlockJacobi::AdditionalData additional_data;
            LA::PreconditionBlockJacobi preconditioner;

            LA::SolverGMRES solver (linear_solver_control);

            preconditioner.initialize(this->system_matrix, additional_data);
            solver.solve(this->system_matrix,
                         this->system_increment,
                         this->system_vector,
                         preconditioner);
        }
        catch (ExceptionBase &e)
        {
            efilog(Verbosity::debug) << e.what() << std::endl;
            this->state = State::failure;
        }
    }
    else
        AssertThrow(false, dealii::ExcMessage("Solver type not supported"));

    this->all_reduce_state ();

    efilog(Verbosity::normal) << "SLV | " << std::flush;

    if (this->state == State::failure)
    {
        efilog(Verbosity::normal) << "Linear solver failed." << std::endl;
        return;
    }

    this->contact_constraints.distribute (this->system_increment);
}



template <int dim>
void
Sample<dim>::
solve_nonlinear ()
{
    using namespace dealii;

    LA::MPI::Vector residual(   
            this->locally_owned_dofs,
            this->mpi_communicator);

    unsigned int step  = 0;

    this->state = State::success;

    efilog(Verbosity::normal) << std::string(85,'_') << std::endl;
    efilog(Verbosity::normal) << "    | ";

    // Copy the cell data history storage. The copy will be used
    // To store the updated cell data history. If the nonlinear
    // solver was successful, the updated history <tt>cdh<\tt> is
    // swapped with <tt>cell_data_history_storage<\tt>, otherwise
    // the update is discarded.
    this->tmp_cell_data_history_storage.reset(
        new GeneralCellDataStorage(*(this->cell_data_history_storage)));

    // Add a reference to the cell_data_storage to the
    // sample_scratch_data, such that it can be accessed
    // by the worker, constitutive, and other objects,
    // which have internal variables to store.
    ScratchDataTools::attach_history_data_storage (
          *(this->sample_scratch_data),
          *(this->cell_data_history_storage));
    ScratchDataTools::attach_tmp_history_data_storage (
           *(this->sample_scratch_data),
           *(this->tmp_cell_data_history_storage));

    // Set the time step size
    ScratchDataTools::get_or_add_time_step_size (
            *(this->sample_scratch_data)) = this->time_step_size;
    
    // Set up active set
    // this->active_set.clear();
    // this->active_set.set_size(this->dof_handler.n_dofs());

    IndexSet old_active_set(active_set);
    do
    {
        // Get the locally relevant solution with
        // ghost cells etc. from the
        // locally relevant solution.
        this->locally_relevant_solution = this->locally_owned_solution;

        // this->apply_contact_constraints(); 

        // this->locally_relevant_solution = this->locally_owned_solution;

        // Obstacle<dim>::cellCount = 0;
        this->assemble ();

        // const unsigned int start_vec = (this->system_vector.local_range().first),
        //                     end_vec = (this->system_vector.local_range().second);
        // for (unsigned int n = start_vec; n < end_vec; ++n)
        //     {

        //         if (this->active_set.is_element(n))
        //         {                 
        //             this->system_vector(n) = 0;
        //             // std::cout << n << " is contact_constraints.is_constrained "<< std::endl;
        //         } 
        //     }
        // this->system_vector.compress(VectorOperation::insert);

        // Check if an assembly error occurred.
        if (this->state != State::success)
            break;

        this->solve_linear ();

        // Check if a linear solver error occurred.
        if (this->state != State::success)
            break;

        // Update the locally owned solution.
        this->locally_owned_solution += this->system_increment;
        this->locally_relevant_solution = this->locally_owned_solution;

        double res_norm = this->system_vector.l2_norm();
        // // efilog(Verbosity::debug) << "res norm pre re-compute: "<< res_norm << std::endl;

        // this->compute_residual();
        // double res_norm_new = this->uncondensed_rhs.l2_norm();
        // efilog(Verbosity::normal) << "residual norm_new before removing constraints: " << res_norm_new << std::endl;
        // // // // Remove indices from system_vector that are in the active set
        // LA::MPI::Vector temp_contact_force (this->uncondensed_rhs);
        // residual = this->uncondensed_rhs;
        // const unsigned int start_res = (residual.local_range().first),
        //                     end_res = (residual.local_range().second);
        // for (unsigned int n = start_res; n < end_res; ++n)
        //     {

                // if (this->active_set.is_element(n))
        //         {
        //             // if ( (n == 27310) || (n == 27311) || (n == 27580) || (n == 27581) )
        //             // {
        //             //     std::cout << n << " this->system_increment " << n << " : " << this->system_increment(n) << std::endl;
        //             //     std::cout << n << " nonlinear residual at " << n << " : " << residual(n) << std::endl;
        //             //     std::cout << n << " system vector at " << n << " : " << this->system_vector(n) << std::endl;
        //             // }
        //             // if (this->system_vector(n) != 0)
        //             // {
        //             //     std::cout << n << " is not properly constrained"<< std::endl;
        //             // }
        //             //     std::cout << n << " solution at " << n << " : " << this->locally_owned_solution(n) << std::endl;
        //             //     std::cout << n << " system vector at " << n << " : " << this->system_vector(n) << std::endl;
        //             //     std::cout << n << " solution increment " << n << " : " << this->system_increment(n) << std::endl;
                    
                    // residual(n) = 0;
        //             // std::cout << n << " is contact_constraints.is_constrained "<< std::endl;
        //         } 
        //     }
        //     if (!this->active_set.is_element(n))
        //     {
        //         temp_contact_force(n) = 0;
        //     }
        // // residual.compress(VectorOperation::insert);
        // temp_contact_force.compress(VectorOperation::insert);
        
        // res_norm_new = residual.l2_norm();
        // // // const double force_norm = temp_contact_force.l2_norm();
        // // // const double system_norm = this->system_vector.l2_norm();
        // efilog(Verbosity::normal) << "residual norm_new: " << res_norm_new << std::endl;
        // efilog(Verbosity::debug) << "temp_contact_force norm: " << force_norm << std::endl;
        // efilog(Verbosity::debug) <<  "num_non_zero_res = " << num_non_zero_res << std::endl;

        // Check if convergence is obtained.
        this->all_reduce_state ();
        if ((this->solver_control.check (step++, res_norm)
                != State::iterate) )
            {
                // if (Utilities::MPI::sum( (active_set == old_active_set) ? 0 : 1, this->mpi_communicator) == 0)
                // {
                    // efilog(Verbosity::normal) << "Full Convergence: Active_set did not change"<< std::endl;
                        break;
                // } 
            }
        old_active_set = this->active_set;       

    } while (true);

    // this->compute_residual();
    // If the nonlinear solver was successful, update the solution fields and
    // the cell data history.
    if ((this->state == State::success)
      &&(this->solver_control.last_check () == State::success))
    {
        // Now trigger post_nonlinear_solve signals. If a one of the signals
        // causes the state not being equal to success, we do update the
        // history and the the locally_relevant_solution.
        this->signals.post_nonlinear_solve();
        if (this->state == State::success)
        {
            efilog(Verbosity::debug)
                    << "Sample::nonlinear_solve & "
                       "Sample::signals.post_nonlinear_solve success."
                    << std::endl;

            this->locally_relevant_solution = this->locally_owned_solution;

            std::swap(this->cell_data_history_storage,
                      this->tmp_cell_data_history_storage);
        }
    }
    else
        this->state = State::failure;

}



template <int dim>
void
Sample<dim>::
compute_contact_force(dealii::IndexSet &active_set)
{
    // using namespace dealii;

    // LA::MPI::Vector distributed_lambda(this->locally_owned_dofs,this->mpi_communicator);
    // const unsigned int start_res = this->uncondensed_rhs.local_range().first;
    // const unsigned int end_res = this->uncondensed_rhs.local_range().second;
    // for (unsigned int n = start_res; n < end_res; ++n)
    //     {
    //         if (this->active_set.is_element(n))
    //         {
    //             distributed_lambda(n) = this->uncondensed_rhs(n);
    //             // / this->diag_mass_matrix_vector(n);
    //         }
    //         // else 
    //         // {
    //         //     distributed_lambda(n) = 0;
    //         // }
    //     }
    // distributed_lambda.compress(VectorOperation::insert);
    // LA::MPI::Vector lambda(this->locally_relevant_dofs,this->mpi_communicator);
    // lambda = distributed_lambda;

    // Tensor<1,dim> contact_force_retract_A;
    // Tensor<1,dim> contact_force_retract_B;
    // contact_force_retract_A = 0;
    // contact_force_retract_B = 0;
    // Tensor<1,dim> res_contact_force_retract_A;
    // Tensor<1,dim> res_contact_force_retract_B;
    // double resultant_force_A = 0;
    // double resultant_force_B = 0;

    // Quadrature<dim-1> face_quadrature(fe->get_unit_face_support_points());
    // FEFaceValues<dim> fe_values_face(*fe, face_quadrature, update_quadrature_points);

    // const unsigned int dofs_per_face = fe->n_dofs_per_face(); 
    // std::vector<types::global_dof_index> dof_indices(dofs_per_face);
    // std::vector<bool> touched_dofs(dof_handler.n_dofs(),false);

    // const unsigned int n_face_q_points = face_quadrature.size();
    // const FEValuesExtractors::Vector displacement(0);

    // for (const auto & cell : dof_handler.active_cell_iterators())
    //     if(cell->is_locally_owned() & cell->at_boundary())
    //         for (const auto & face : cell->face_iterators())
    //             if (face->at_boundary())
    //                 if ( (face->boundary_id() == 1) || (face->boundary_id() == 3))
    //                 {
    //                     fe_values_face.reinit(cell, face);
    //                     face->get_dof_indices(dof_indices);
    //                     Tensor<1,dim> test;
    //                     test = 0.;
    //                     for (unsigned int q_point = 0; q_point<n_face_q_points; q_point += dim)
    //                     {
    //                         const int index = dof_indices[q_point];
    //                         for (unsigned int d=0; d<dim; d++){
    //                             test[d] += this->uncondensed_rhs(index+d);
    //                         }
    //                         efilog(Verbosity::normal) << "idx: " << index << std::endl;
    //                         efilog(Verbosity::normal) << "test force: " << test << std::endl;
    //                     }

                        // res_contact_force_retract_A = 0;
                        // res_contact_force_retract_B = 0;
                        // std::vector<Tensor<1,dim>> lambda_values(n_face_q_points);
                        // fe_values_face[displacement].get_function_values(lambda, lambda_values);
                        // for (unsigned int q = 0; q<n_face_q_points; q++)
                        // {   
                        //     for (int d = 0; d< dim; d++)
                        //     {
                        //         if (face->boundary_id() == 1) 
                        //         {
                        //             contact_force_retract_A[d] += lambda_values[q][d]*fe_values_face.JxW(q);
                        //             res_contact_force_retract_A[d] += lambda_values[q][d]*fe_values_face.JxW(q);
                        //         } else {
                        //             contact_force_retract_B[d] += lambda_values[q][d]*fe_values_face.JxW(q);
                        //             res_contact_force_retract_B[d] += lambda_values[q][d]*fe_values_face.JxW(q);
                        //         }
                        //     }
                        // }
                        // resultant_force_A += res_contact_force_retract_A.norm();
                        // resultant_force_B += res_contact_force_retract_B.norm();
                    // }

    // contact_force_retract_A = Utilities::MPI::sum(contact_force_retract_A, this->mpi_communicator);
    // contact_force_retract_B = Utilities::MPI::sum(contact_force_retract_B, this->mpi_communicator);
    // resultant_force_A = Utilities::MPI::sum(resultant_force_A, this->mpi_communicator);
    // resultant_force_B = Utilities::MPI::sum(resultant_force_B, this->mpi_communicator);

    // efilog(Verbosity::normal) << "Contact_force at A: " << contact_force_retract_A << std::endl;
    // efilog(Verbosity::normal) << "Contact_force at B: " << contact_force_retract_B << std::endl;
    // efilog(Verbosity::normal) << "Resultant Contact_force at A: " << resultant_force_A << std::endl;
    // efilog(Verbosity::normal) << "Resultant Contact_force at B: " << resultant_force_B << std::endl;
    

}

template <int dim>
void
Sample<dim>::
write_output (const unsigned int step,
              const double       time)
{
    // this->all_reduce_state ();
    using namespace dealii;

    TimerOutput::Scope timer_section(*(this->timer), EFI_PRETTY_FUNCTION);
    bool create_moved_mesh = GlobalParameters::create_moved_mesh();

    //move_mesh
    if (create_moved_mesh){
        this->move_mesh(this->locally_relevant_solution);
    }

    unsigned int n_mpi_processes =
            Utilities::MPI::n_mpi_processes(mpi_communicator);

    // Create the post-processor before the
    // DataOut object since the post-processor
    // has to survive longer. Otherwise an
    // error will be thrown in Debug mode.
    CellDataPostProcessor<dim> post_processor (
            this->constitutive_model_map, this->cell_data_history_storage.get());

    // setup the data out object
    DataOut<dim> out;
    out.attach_dof_handler (dof_handler);
    out.add_data_vector (this->locally_relevant_solution, post_processor);

    // add material_id filter
    Vector<double> distributed_material_id(tria.n_active_cells());
    distributed_material_id = 0.;
    for( const auto cell : tria.active_cell_iterators())
    {
        distributed_material_id[cell->active_cell_index()] = cell->material_id();
    } 

    out.add_data_vector(distributed_material_id,"material_ids");
            
    // if (!this->active_set.is_empty())
    // {
    //     // Add Active set
    //     LA::MPI::Vector distributed_active_set_vector(this->locally_owned_dofs,this->mpi_communicator);
    //     distributed_active_set_vector = 0.;
    //     for (const auto index: this->active_set)
    //         distributed_active_set_vector[index] = 1.;
    //     distributed_active_set_vector.compress(VectorOperation::insert);
    //     LA::MPI::Vector active_set_vector(this->locally_relevant_dofs, this->mpi_communicator);
    //     active_set_vector = distributed_active_set_vector;
    //     out.add_data_vector(active_set_vector,"active_set");

        LA::MPI::Vector distributed_reaction_force(this->locally_owned_dofs,this->mpi_communicator);
        const unsigned int start_res = this->uncondensed_rhs.local_range().first;
        const unsigned int end_res = this->uncondensed_rhs.local_range().second;
        for (unsigned int n = start_res; n < end_res; ++n)
            {
                if (this->active_set.is_element(n))
                {
                    distributed_reaction_force(n) = this->uncondensed_rhs(n);
                }
                else
                {
                    distributed_reaction_force(n) = 0.;
                }
            }
        distributed_reaction_force.compress(VectorOperation::insert);
        LA::MPI::Vector reaction_force(this->locally_relevant_dofs,this->mpi_communicator);
        reaction_force = distributed_reaction_force;
        out.add_data_vector(reaction_force,"reaction_force");



    double y_reaction_force_total_1 = 0.;
    double z_reaction_force_total_1 = 0.;
    double z_reaction_force_total_3 = 0.;

    std::vector<bool> touched_dofs(this->dof_handler.n_dofs(),false);
    LA::MPI::Vector reaction_tmp(this->locally_relevant_dofs, this->mpi_communicator);
    reaction_tmp = this->uncondensed_rhs;
    for(const auto &cell: dof_handler.active_cell_iterators())
        if (cell->is_locally_owned() & cell->at_boundary())
            for (const auto & face : cell->face_iterators())
                if (face->at_boundary() && ((face->boundary_id() == 1) || (face->boundary_id() == 3)))
                    for (const auto v: face->vertex_indices())
                        {
                            dealii::Point<dim> vertex = face->vertex(v);
                            std::vector<int> multipliers(2);
                            multipliers[0] = 1.0;
                            multipliers[1] = 1.0;
                            if (vertex(1) > 0){
                                multipliers[0] = -1.0;
                            } 
                            if (vertex(2) > 0){
                                multipliers[1] = -1.0;
                            }
                            unsigned int y_dof = face->vertex_dof_index(v, 1);
                            unsigned int z_dof = face->vertex_dof_index(v, 2);
                            double reaction_force_y = reaction_tmp(y_dof)*multipliers[0];
                            double reaction_force_z = reaction_tmp(z_dof)*multipliers[1];
                            if ((face->boundary_id() == 1)){
                                y_reaction_force_total_1 += reaction_force_y;
                                z_reaction_force_total_1 += reaction_force_z;
                            } else {
                                z_reaction_force_total_3 += reaction_force_z;
                            }
                        }
    z_reaction_force_total_1 = Utilities::MPI::sum(z_reaction_force_total_1, this->mpi_communicator);
    double area_1 = this->calculate_area(1);
    double area_1_total = Utilities::MPI::sum(area_1, this->mpi_communicator);
    double area_3 = this->calculate_area(3);
    double area_3_total = Utilities::MPI::sum(area_3, this->mpi_communicator);

    if (area_3_total > 1e-9){
        z_reaction_force_total_3 = Utilities::MPI::sum(z_reaction_force_total_3, this->mpi_communicator);
        efilog(Verbosity::normal) << "z_reaction_force_total_1: " << z_reaction_force_total_1 << std::endl;
        efilog(Verbosity::normal) << "z_reaction_force_total_3: " << z_reaction_force_total_3 << std::endl;
        efilog(Verbosity::normal) << "Total area 1: " << area_1_total << std::endl;
        efilog(Verbosity::normal) << "Total area 3: " << area_3_total << std::endl;
        efilog(Verbosity::normal) << "Average pressure 1: " << (z_reaction_force_total_1/area_1_total) << std::endl;    
        efilog(Verbosity::normal) << "Average pressure 3: " << (z_reaction_force_total_3/area_3_total) << std::endl;
    } else {
        y_reaction_force_total_1 = Utilities::MPI::sum(y_reaction_force_total_1, this->mpi_communicator);
        efilog(Verbosity::normal) << "y_reaction_force_total_1: " << y_reaction_force_total_1 << std::endl;
        efilog(Verbosity::normal) << "z_reaction_force_total_1: " << z_reaction_force_total_1 << std::endl;
        efilog(Verbosity::normal) << "Total area 1: " << area_1_total << std::endl;
        double resultant_force = std::sqrt((y_reaction_force_total_1*y_reaction_force_total_1) + (z_reaction_force_total_1)*z_reaction_force_total_1);
        efilog(Verbosity::normal) << "Resultant force: " << resultant_force << std::endl;
        efilog(Verbosity::normal) << "Average pressure: " << (resultant_force/area_1_total) << std::endl;
    }

    // get the subdomain IDs
    types::subdomain_id locally_owned_subdomain =
            this->tria.locally_owned_subdomain ();

    if (n_mpi_processes > 0)
    {
        Vector<float> subdomain(this->tria.n_active_cells());

        std::fill (subdomain.begin(), subdomain.end(), locally_owned_subdomain);

        out.add_data_vector(subdomain, "subdomain");
        out.build_patches(fe->tensor_degree());
    }
    else
        out.build_patches();

    boost::filesystem::path output_directory = 
        GlobalParameters::get_output_directory();

    // output directory
    std::string directory (output_directory.string()
                         + std::string(1,output_directory.separator));


    std::string output_filename = 
        GlobalParameters::get_output_filename();

    // common name of the output files
    std::string name (output_filename
                    + Utilities::int_to_string(dim,1)
                    + "d-"
                    + Utilities::int_to_string(step, 3));


    // write the owned by this processor
    std::ofstream output(directory + name
                        + "."
                        + Utilities::int_to_string(locally_owned_subdomain, 4)
                        + ".vtu");

    out.write_vtu(output);

    // GridOut gridOut;
    // std::ofstream output_stream(directory + name + "grid.vtu");
    // gridOut.write_vtu(tria,output_stream);

    // Push the master file into the times_and_names
    // vector to be able to write the pvd record later.
    this->times_and_names.push_back ({time, name + ".pvtu"});

    // If this is the root process write the master file
    // gathering the files from the single processes.
    if (MPI::is_root(this->mpi_communicator))
    {
        std::vector<std::string> partition_names;

        for (unsigned int i = 0; i < n_mpi_processes; ++i)
            partition_names.push_back (name + "." +
                    Utilities::int_to_string(i, 4) + ".vtu");

        // Write the parallel (partitioned) VTK unstructured Data (pvtu)
        std::ofstream master_output(directory + name + ".pvtu");
        out.write_pvtu_record(master_output, partition_names);

        // write pvd
        std::ofstream pvd_output (directory
                                + output_filename +
                                + "_solution-"
                                + Utilities::int_to_string(dim,1)
                                + "d.pvd");

        DataOutBase::write_pvd_record (pvd_output, this->times_and_names);
    }

    if (create_moved_mesh)
    {
        //move_mesh
        LA::MPI::Vector tmp(this->locally_relevant_solution);
        tmp *= -1;
        this->move_mesh(tmp);
    }
}


template <int dim>
void
Sample<dim>::
move_mesh(const  LA::MPI::Vector &displacement) const
{
    std::vector<bool> vertex_touched(this->tria.n_vertices(), false);

    for(const auto &cell: dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
            for (const auto v: cell->vertex_indices())
                if (vertex_touched[cell->vertex_index(v)] == false)
                {
                    vertex_touched[cell->vertex_index(v)] = true;

                    dealii::Point<dim> vertex_displacement;
                    for (unsigned int d = 0; d< dim; ++d)
                        vertex_displacement[d] = displacement(cell->vertex_dof_index(v, d));
                    cell->vertex(v) += vertex_displacement;
                }
    
}

template <int dim>
double
Sample<dim>::
calculate_area(dealii::types::boundary_id id) const
{
    using namespace dealii;
    double area_total = 0.;

    for(const auto &cell: dof_handler.active_cell_iterators())
        if (cell->is_locally_owned() && cell->at_boundary())
            for (const auto & face : cell->face_iterators())
                if ((face->at_boundary()) && (face->boundary_id() == id))
                    area_total += face->measure();
    
    return area_total;
    
}

template <int dim>
void
Sample<dim>::
get_slave_pnt(const dealii::Point<dim> &support_pnt, 
                dealii::Point<dim> &slave_pnt,
                dealii::types::global_dof_index face_dof_indices)
{
    slave_pnt = support_pnt;
    for (unsigned int v_index=0; v_index<dim;v_index++)
    {
        const unsigned int vertex_dof_index = face_dof_indices + v_index;
        slave_pnt(v_index) += this->locally_relevant_solution(vertex_dof_index);
    }                                       

}



// Instantiation
template class Sample<2>;
template class Sample<3>;

// Registration
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Sample,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Sample,3));

}// namespace efi


