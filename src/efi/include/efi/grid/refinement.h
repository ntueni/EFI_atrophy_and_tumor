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

#ifndef SRC_MYLIB_INCLUDE_EFI_GRID_REFINEMENT_H_
#define SRC_MYLIB_INCLUDE_EFI_GRID_REFINEMENT_H_

// c++ headers
#include <fstream>

// deal.II headers
#include <deal.II/base/geometry_info.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/manifold_lib.h>


namespace efi {


// Refine until all cell-edges are shorter than the max_mesh_size.
// Note that the anisotropic refinement is restricted to serial
// meshes, i.e. each processor may refine a serial triangulation,
// a distributed triangulation with this kind of refinement is not
// allowed.
template <class MeshType>
void refine_without_hanging_boundary_nodes (
        MeshType     &triangulation,
        const double  max_mesh_size,
        const bool    anisotropic = false)
{
    Assert (max_mesh_size > 1e-10,
            dealii::ExcMessage ("Given mesh size too small."));

    // Check if we deal with a distributed triangulation.
    bool is_distributed = dynamic_cast <dealii::parallel::distributed::Triangulation<MeshType::dimension>*>(&triangulation) != nullptr;

    // Anisotropic refinement is only allowed in serial applications.
    bool allow_anisotropic_refinement = is_distributed? false : anisotropic;

    // we use this flag to check if refinement is required.
    bool refine = false;
    bool refine_boundary_cells = false;

    unsigned int count = 0;

    do {
        refine = false;
        refine_boundary_cells = false;

        auto cell = triangulation.begin_active();
        auto endc = triangulation.end();

        for (; cell!=endc; ++cell)
            if(cell->is_locally_owned())
            {
                dealii::RefinementCase<MeshType::dimension> refinement = cell->refine_flag_set();

                for (unsigned int i = 0; i < MeshType::dimension; ++i)
                    if (cell->extent_in_direction (i) > max_mesh_size)
                    {
                        refine = true;
                        if (allow_anisotropic_refinement && (cell->at_boundary()))
                            refinement = refinement | dealii::RefinementCase<MeshType::dimension>::cut_axis(i);
                        else
                        {
                            // If the cell is at the boundary, set the
                            // refine_boundary_cell flag.
                            if (cell->at_boundary())
                                refine_boundary_cells = true;

                            refinement = dealii::RefinementCase<MeshType::dimension>::isotropic_refinement;
                            break;
                        }
                    }

                // On a periodic boundary, we use only isotropic refinement. Hence, if a cell
                // is marked for refinement and has a periodic neighbor, we set the refinement
                // case to isotropic refinement.
                if (refinement != dealii::RefinementCase<MeshType::dimension>::no_refinement)
                {
                    for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<MeshType::dimension>::faces_per_cell; ++face_no)
                        if (cell->has_periodic_neighbor(face_no))
                        {
                            refinement = dealii::RefinementCase<MeshType::dimension>::isotropic_refinement;
                        }
                }

                // set the refinement flag
                cell->set_refine_flag (refinement);
            }

        cell = triangulation.begin_active();
        endc = triangulation.end();

        for (; cell!=endc; ++cell)
            if(cell->is_locally_owned())
            {
                for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<MeshType::dimension>::faces_per_cell; ++face_no)
                    if (cell->has_periodic_neighbor(face_no))
                    {
                        if (cell->periodic_neighbor(face_no)->has_children())
                        {
                            // If the cell is at the boundary, set the
                            // refine_boundary_cell flag.
                            if (cell->at_boundary())
                                refine_boundary_cells = true;

                            refine = true;
                            cell->set_refine_flag (dealii::RefinementCase<MeshType::dimension>::isotropic_refinement);
                        }
                    }

                // If one cell at the boundary is refined refine all boundary cells
                if (refine_boundary_cells && cell->at_boundary())
                    cell->set_refine_flag (dealii::RefinementCase<MeshType::dimension>::isotropic_refinement);
            }

        // This function is at least called once to mace sure that
        // pre- and post-refinement signals of the triangulation
        // are triggered.
        triangulation.execute_coarsening_and_refinement ();

        if (is_distributed)
        {
            auto &distributed_tria = dynamic_cast <dealii::parallel::distributed::Triangulation<MeshType::dimension>&>(triangulation);
            unsigned int refinement_state = refine? 1 : 0;

            refine = dealii::Utilities::MPI::sum (refinement_state, distributed_tria.get_communicator()) > 0;
        }

        ++count;
         if (count == 100)
             Assert (false, dealii::ExcMessage ("Exceeds max number of iterations."));

    } while (refine);
}



// Refine until all cell-edges are shorter than the max_mesh_size.
// Note that the anisotropic refinement is restricted to serial
// meshes, i.e. each processor may refine a serial triangulation,
// a distributed triangulation with this kind of refinement is not
// allowed.
template <class MeshType>
void refine (MeshType     &triangulation,
             const double  max_mesh_size,
             const bool    anisotropic = false)
{
    Assert (max_mesh_size > 1e-10,
            dealii::ExcMessage ("Given mesh size too small."));

    // Check if we deal with a distributed triangulation.
    bool is_distributed = dynamic_cast <dealii::parallel::distributed::Triangulation<MeshType::dimension>*>(&triangulation) != nullptr;

    // Anisotropic refinement is only allowed in serial applications.
    bool allow_anisotropic_refinement = is_distributed? false : anisotropic;

    // we use this flag to check if refinement is required.
    bool refine = false;

    unsigned int count = 0;

    do {
        refine = false;
        auto cell = triangulation.begin_active();
        auto endc = triangulation.end();

        for (; cell!=endc; ++cell)
            if(cell->is_locally_owned())
            {
                dealii::RefinementCase<MeshType::dimension> refinement = cell->refine_flag_set();

                for (unsigned int i = 0; i < MeshType::dimension; ++i)
                    if (cell->extent_in_direction (i) > max_mesh_size)
                    {
                        refine = true;
                        if (allow_anisotropic_refinement)
                            refinement = refinement | dealii::RefinementCase<MeshType::dimension>::cut_axis(i);
                        else
                        {
                            refinement = dealii::RefinementCase<MeshType::dimension>::isotropic_refinement;
                            break;
                        }
                    }

                // On a periodic boundary, we use only isotropic refinement. Hence, if a cell
                // is marked for refinement and has a periodic neighbor, we set the refinement
                // case to isotropic refinement.
                if (refinement != dealii::RefinementCase<MeshType::dimension>::no_refinement)
                {
                    for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<MeshType::dimension>::faces_per_cell; ++face_no)
                        if (cell->has_periodic_neighbor(face_no))
                        {
                            refinement = dealii::RefinementCase<MeshType::dimension>::isotropic_refinement;
                        }
                }

                // set the refinement flag
                cell->set_refine_flag (refinement);
            }

        cell = triangulation.begin_active();
        endc = triangulation.end();

        for (; cell!=endc; ++cell)
            if(cell->is_locally_owned())
            {
                for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<MeshType::dimension>::faces_per_cell; ++face_no)
                    if (cell->has_periodic_neighbor(face_no))
                    {
                        if (cell->periodic_neighbor(face_no)->has_children())
                        {
                            refine = true;
                            cell->set_refine_flag (dealii::RefinementCase<MeshType::dimension>::isotropic_refinement);
                        }
                    }
            }
        // This function is at least called once to mace sure that
        // pre- and post-refinement signals of the triangulation
        // are triggered.
        triangulation.execute_coarsening_and_refinement ();

        if (is_distributed)
        {
            auto &distributed_tria = dynamic_cast <dealii::parallel::distributed::Triangulation<MeshType::dimension>&>(triangulation);
            unsigned int refinement_state = refine? 1 : 0;

            refine = dealii::Utilities::MPI::sum (refinement_state, distributed_tria.get_communicator()) > 0;
        }

        ++count;
         if (count == 100)
             Assert (false, dealii::ExcMessage ("Exceeds max number of iterations."));

    } while (refine);
}




// Refine until all cell-edges are shorter than the max_mesh_size.
// Note that the anisotropic refinement is restricted to serial
// meshes, i.e. each processor may refine a serial triangulation,
// a distributed triangulation with this kind of refinement is not
// allowed.
template <class MeshType>
void refine (MeshType     &triangulation,
             const std::function<double(const dealii::Point<MeshType::dimension>&)>  &max_mesh_size,
             const bool    anisotropic = false)
{
    // Check if we deal with a distributed triangulation.
    bool is_distributed = dynamic_cast <dealii::parallel::distributed::Triangulation<MeshType::dimension>*>(&triangulation) != nullptr;

    // Anisotropic refinement is only allowed in serial applications.
    bool allow_anisotropic_refinement = is_distributed? false : anisotropic;

    // we use this flag to check if refinement is required.
    bool refine = false;

    unsigned int count = 0;

    do {
        refine = false;
        auto cell = triangulation.begin_active();
        auto endc = triangulation.end();

        for (; cell!=endc; ++cell)
            if(cell->is_locally_owned())
            {
                dealii::RefinementCase<MeshType::dimension> refinement = cell->refine_flag_set();

                Assert (max_mesh_size (cell->center()) > 1e-10, dealii::ExcMessage ("Given mesh size too small."));

                for (unsigned int i = 0; i < MeshType::dimension; ++i)
                    if (cell->extent_in_direction (i) > max_mesh_size(cell->center()))
                    {
                        refine = true;
                        if (allow_anisotropic_refinement)
                            refinement = refinement | dealii::RefinementCase<MeshType::dimension>::cut_axis(i);
                        else
                        {
                            refinement = dealii::RefinementCase<MeshType::dimension>::isotropic_refinement;
                            break;
                        }
                    }

                // On a periodic boundary, we use only isotropic refinement. Hence, if a cell
                // is marked for refinement and has a periodic neighbor, we set the refinement
                // case to isotropic refinement.
                if (refinement != dealii::RefinementCase<MeshType::dimension>::no_refinement)
                {
                    for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<MeshType::dimension>::faces_per_cell; ++face_no)
                        if (cell->has_periodic_neighbor(face_no))
                        {
                            refinement = dealii::RefinementCase<MeshType::dimension>::isotropic_refinement;
                        }
                }

                // set the refinement flag
                cell->set_refine_flag (refinement);
            }

        cell = triangulation.begin_active();
        endc = triangulation.end();

        for (; cell!=endc; ++cell)
            if(cell->is_locally_owned())
            {
                for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<MeshType::dimension>::faces_per_cell; ++face_no)
                    if (cell->has_periodic_neighbor(face_no))
                    {
                        if (cell->periodic_neighbor(face_no)->has_children())
                        {
                            refine = true;
                            cell->set_refine_flag (dealii::RefinementCase<MeshType::dimension>::isotropic_refinement);
                        }
                    }
            }
        // This function is at least called once to mace sure that
        // pre- and post-refinement signals of the triangulation
        // are triggered.
        triangulation.execute_coarsening_and_refinement ();

        if (is_distributed)
        {
            auto &distributed_tria = dynamic_cast <dealii::parallel::distributed::Triangulation<MeshType::dimension>&>(triangulation);
            unsigned int refinement_state = refine? 1 : 0;

            refine = dealii::Utilities::MPI::sum (refinement_state, distributed_tria.get_communicator()) > 0;
        }

        ++count;
         if (count == 100)
             Assert (false, dealii::ExcMessage ("Exceeds max number of iterations."));

    } while (refine);
}




// Refine until all cell-edges are shorter than the max_mesh_size.
// Note that the anisotropic refinement is restricted to serial
// meshes without periodic boundaries.
template <class MeshType>
void refine (MeshType &triangulation,
             std::map<dealii::types::material_id, double>  max_mesh_size_map,
             const bool anisotropic = false)
{
    // Check if we deal with a distributed triangulation.
    bool is_distributed = (dynamic_cast <dealii::parallel::distributed::Triangulation<MeshType::dimension>*>(&triangulation) != nullptr);

    // Anisotropic refinement is only allowed in serial applications.
    bool allow_anisotropic_refinement = is_distributed? false : anisotropic;

    // we use this flag to check if refinement is required.
    bool refine = false;

    unsigned int count = 0;

    do {
        refine = false;
        auto cell = triangulation.begin_active();
        auto endc = triangulation.end();

        for (; cell!=endc; ++cell)
            if(cell->is_locally_owned())
            {
                dealii::RefinementCase<MeshType::dimension> refinement = cell->refine_flag_set();

                Assert (max_mesh_size_map.find(cell->material_id()) != max_mesh_size_map.end(),
                        dealii::ExcMessage ("Mesh size not specified for material id "
                                           + dealii::Utilities::int_to_string(cell->material_id())));
                double max_mesh_size = max_mesh_size_map[cell->material_id()];

                Assert (max_mesh_size > 1e-10,
                        dealii::ExcMessage ("Given mesh size too small."));

                for (unsigned int i = 0; i < MeshType::dimension; ++i)
                    if (cell->extent_in_direction (i) > max_mesh_size)
                    {
                        refine = true;
                        if (allow_anisotropic_refinement)
                            refinement = refinement | dealii::RefinementCase<MeshType::dimension>::cut_axis(i);
                        else
                        {
                            refinement = dealii::RefinementCase<MeshType::dimension>::isotropic_refinement;
                            break;
                        }
                    }

                // On a periodic boundary, we use only isotropic refinement. Hence, if a cell
                // is marked for refinement and has a periodic neighbor, we set the refinement
                // case to isotropic refinement.
                if (refinement != dealii::RefinementCase<MeshType::dimension>::no_refinement)
                {
                    for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<MeshType::dimension>::faces_per_cell; ++face_no)
                        if (cell->has_periodic_neighbor(face_no))
                        {
                            refinement = dealii::RefinementCase<MeshType::dimension>::isotropic_refinement;
                        }
                }
                // set the refinement flag
                cell->set_refine_flag (refinement);
            }

        cell = triangulation.begin_active();
        endc = triangulation.end();

        for (; cell!=endc; ++cell)
            if(cell->is_locally_owned())
            {
                for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<MeshType::dimension>::faces_per_cell; ++face_no)
                    if (cell->has_periodic_neighbor(face_no))
                    {
                        if (cell->periodic_neighbor(face_no)->has_children())
                        {
                            refine = true;
                            cell->set_refine_flag (dealii::RefinementCase<MeshType::dimension>::isotropic_refinement);
                        }
                    }
            }

        // This function is at least called once to make sure that
        // pre- and post-refinement signals of the triangulation
        // are triggered.
        triangulation.execute_coarsening_and_refinement ();

        if (is_distributed)
        {
            auto &distributed_tria = dynamic_cast <dealii::parallel::distributed::Triangulation<MeshType::dimension>&>(triangulation);
            unsigned int refinement_state = refine? 1 : 0;

            refine = dealii::Utilities::MPI::sum (refinement_state, distributed_tria.get_communicator()) > 0;
        }

        ++count;
        if (count == 100)
            Assert (false, dealii::ExcMessage ("Exceeds max number of iterations."));

    } while (refine);
}

}// close efi

#endif /* SRC_MYLIB_INCLUDE_EFI_GRID_REFINEMENT_H_ */
