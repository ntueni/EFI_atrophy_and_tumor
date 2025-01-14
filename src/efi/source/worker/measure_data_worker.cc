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

// efi headers
#include <efi/worker/measure_data_worker.h>


namespace efi {


template <int dim>
void
MeasureBoundaryForceWorker<dim>::
do_fill (ScratchData<dim> &scratch_data,
         CopyData         &copy_data) const
{
    using namespace dealii;
    using namespace dealii::Physics::Elasticity;

    // If the boundary_id of the current face iterator is in the set of
    // active_boundary_ids continue, otherwise, do nothing.
    if (this->is_active (scratch_data))
    {
        auto global_vector_name = Extractor<dim>::global_vector_name();

        // Get the number of quadrature points and
        // the number of dofs per cell.
        auto n_q_points = ScratchDataTools::n_quadrature_points (scratch_data);

        // Get the material quantities:
        // - JxW area elements
        // - N normal vectors
        // - tau Kirchoff stresses
        // - Grad_u material displacement gradients
        auto &JxW    = ScratchDataTools::get_JxW_values (scratch_data);
        auto &N      = ScratchDataTools::get_normal_vectors (scratch_data);
        auto &tau    = ScratchDataTools::get_kirchoff_stresses (
                           scratch_data, global_vector_name, ad_type());
        auto &Grad_u = ScratchDataTools::get_gradients (
                           scratch_data, global_vector_name,
                           Extractor<dim>::displacement(),ad_type(0));

        // loop over all quadrature points
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            // deformation gradient
            Tensor<2,dim,ad_type> F = StandardTensors<dim>::I + Grad_u[q];

            this->add_measured_data (
                    copy_data, JxW[q]*(tau[q]*transpose(invert(F))*N[q]));
        }
    }
}

template <int dim>
void
MeasureBoundaryRadialYZWorker<dim>::
do_fill (ScratchData<dim> &scratch_data,
         CopyData         &copy_data) const
{
    using namespace dealii;
    using namespace dealii::Physics::Elasticity;

    // If the boundary_id of the current face iterator is in the set of
    // active_boundary_ids continue, otherwise, do nothing.
    if (this->is_active (scratch_data))
    {
        auto global_vector_name = Extractor<dim>::global_vector_name();

        // Get the number of quadrature points and
        // the number of dofs per cell.
        auto n_q_points = ScratchDataTools::n_quadrature_points (scratch_data);

        // Get the material quantities:
        // - JxW area elements
        // - N normal vectors
        // - tau Kirchoff stresses
        // - Grad_u material displacement gradients
        auto &JxW    = ScratchDataTools::get_JxW_values (scratch_data);
        auto &N      = ScratchDataTools::get_normal_vectors (scratch_data);
        auto &tau    = ScratchDataTools::get_kirchoff_stresses (
                           scratch_data, global_vector_name, ad_type());
        auto &Grad_u = ScratchDataTools::get_gradients (
                           scratch_data, global_vector_name,
                           Extractor<dim>::displacement(),ad_type(0));

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            Tensor<1,dim,ad_type> force_area;
            force_area = 0;
            // deformation gradient
            Tensor<2,dim,ad_type> F = StandardTensors<dim>::I + Grad_u[q];

            Tensor<1,dim,ad_type> force =
                    JxW[q]*(tau[q]*transpose(invert(F))*N[q]);

            Tensor<1,dim,ad_type> area_vector = JxW[q]*(transpose(invert(F))*N[q]);

            double area = std::sqrt((area_vector[0]*area_vector[0]) + (area_vector[1]*area_vector[1]) + (area_vector[2]*area_vector[2]));

            double resultant_force = std::sqrt((force[1]*force[1]) + (force[2]*force[2]));
            force_area[0] = resultant_force;
            force_area[1] = area;

            this->add_measured_data (copy_data, force_area);
        }
    }
}

template <int dim>
void
MeasureBoundarySpatulaWorker<dim>::
do_fill (ScratchData<dim> &scratch_data,
         CopyData         &copy_data) const
{
    using namespace dealii;
    using namespace dealii::Physics::Elasticity;

    // If the boundary_id of the current face iterator is in the set of
    // active_boundary_ids continue, otherwise, do nothing.
    if (this->is_active (scratch_data))
    {
        auto global_vector_name = Extractor<dim>::global_vector_name();

        // Get the number of quadrature points and
        // the number of dofs per cell.
        auto n_q_points = ScratchDataTools::n_quadrature_points (scratch_data);

        // Get the material quantities:
        // - JxW area elements
        // - N normal vectors
        // - tau Kirchoff stresses
        // - Grad_u material displacement gradients
        auto &JxW    = ScratchDataTools::get_JxW_values (scratch_data);
        auto &N      = ScratchDataTools::get_normal_vectors (scratch_data);
        auto &tau    = ScratchDataTools::get_kirchoff_stresses (
                           scratch_data, global_vector_name, ad_type());
        auto &Grad_u = ScratchDataTools::get_gradients (
                           scratch_data, global_vector_name,
                           Extractor<dim>::displacement(),ad_type(0));

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            Tensor<1,dim,ad_type> force_area;
            force_area = 0;
            // deformation gradient
            Tensor<2,dim,ad_type> F = StandardTensors<dim>::I + Grad_u[q];

            Tensor<1,dim,ad_type> force =
                    JxW[q]*(tau[q]*transpose(invert(F))*N[q]);     

            Tensor<1,dim,ad_type> area_vector = JxW[q]*(transpose(invert(F))*N[q]);

            double area = std::sqrt((area_vector[0]*area_vector[0]) + (area_vector[1]*area_vector[1]) + (area_vector[2]*area_vector[2]));

            double resultant_force = force[2];
            force_area[0] = resultant_force;
            force_area[1] = area;

            this->add_measured_data (copy_data, force_area);
        }
    }
}



template <int dim>
void
MeasureBoundaryTorqueXWorker<dim>::
do_fill (ScratchData<dim> &scratch_data,
         CopyData         &copy_data) const
{
    using namespace dealii;
    using namespace dealii::Physics::Elasticity;

    // If the boundary_id of the current face iterator is in the set of
    // active_boundary_ids continue, otherwise, do nothing.
    if (this->is_active (scratch_data))
    {
        auto global_vector_name = Extractor<dim>::global_vector_name();

        // Get the number of quadrature points and
        // the number of dofs per cell.
        auto n_q_points = ScratchDataTools::n_quadrature_points (scratch_data);

        // Get the material quantities:
        // - JxW area elements
        // - N normal vectors
        // - tau Kirchoff stresses
        // - Grad_u material displacement gradients
        auto &JxW    = ScratchDataTools::get_JxW_values (scratch_data);
        auto &N      = ScratchDataTools::get_normal_vectors (scratch_data);
        auto &tau    = ScratchDataTools::get_kirchoff_stresses (
                           scratch_data, global_vector_name, ad_type());
        auto &u      = ScratchDataTools::get_values (
                           scratch_data, global_vector_name,
                           Extractor<dim>::displacement(),ad_type(0));
        auto &Grad_u = ScratchDataTools::get_gradients (
                           scratch_data, global_vector_name,
                           Extractor<dim>::displacement(),ad_type(0));

        std::vector<Point<dim>> x
            = ScratchDataTools::get_quadrature_points (scratch_data);

        // loop over all quadrature points
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            // deformation gradient
            Tensor<2,dim,ad_type> F = StandardTensors<dim>::I + Grad_u[q];

            // get coordinates in spatial configuration: x = X + u;
            x[q] += u[q];

            // The distance vector between the point where the force is acting
            // and the x-axis is simply obtained by setting the x-component of
            // our spatial coordinates to zero.
            x[q][0] = 0;
            Tensor<1,dim,ad_type> force =
                    JxW[q]*(tau[q]*transpose(invert(F))*N[q]);

            this->add_measured_data (copy_data, cross_product_3d(x[q],force));
        }
    }
}



// Instantiations
template class MeasureBoundaryForceWorker<2>;
template class MeasureBoundaryForceWorker<3>;

template class MeasureBoundaryTorqueXWorker<2>;
template class MeasureBoundaryTorqueXWorker<3>;

template class MeasureBoundaryRadialYZWorker<2>;
template class MeasureBoundaryRadialYZWorker<3>;

template class MeasureBoundarySpatulaWorker<2>;
template class MeasureBoundarySpatulaWorker<3>;


// Registration
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(MeasureBoundaryForceWorker,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(MeasureBoundaryForceWorker,3));

EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(MeasureBoundaryTorqueXWorker,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(MeasureBoundaryTorqueXWorker,3));

EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(MeasureBoundaryRadialYZWorker,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(MeasureBoundaryRadialYZWorker,3));

EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(MeasureBoundarySpatulaWorker,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(MeasureBoundarySpatulaWorker,3));

}// namespace efi




