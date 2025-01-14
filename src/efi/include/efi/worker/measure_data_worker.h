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
 * Author: Stefan Kaessmairr
 */

#ifndef SRC_EFI_INCLUDE_EFI_WORKER_MEASURE_DATA_WORKER_H_
#define SRC_EFI_INCLUDE_EFI_WORKER_MEASURE_DATA_WORKER_H_

// stl headers
#include <string>
#include <vector>
#include <memory>

// deal.II headers
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#if DEBUG
// boost headers
#include <boost/type_index.hpp>
#endif

// efi headers
#include <efi/base/type_traits.h>
#include <efi/worker/boundary_worker.h>
#include <efi/worker/copy_data.h>
#include <efi/worker/scratch_data.h>


/// Preprocessor macro to provide basic boundary measurement worker
/// implementation measuring a single quantity. The quantity can be of any
/// @p datatype, it only needs to provide the += operator.
/// @note The functions @p get_needed_update_flags and @p do_fill must be
/// implemented by the user.
#define EFI_MEASURE_BOUNDARY_DATA_WORKER(dataname,datatype)                   \
template <int dim>                                                            \
class MeasureBoundary##dataname##Worker                                       \
    : public MeasureBoundaryDataWorker<                                       \
      typename efi_internal::get_type_from_macro_input<void datatype>::type,  \
      dim>                                                                    \
{                                                                             \
public:                                                                       \
                                                                              \
    using data_type   =                                                       \
        typename efi_internal::get_type_from_macro_input<void datatype>::type;\
    using scalar_type =                                                       \
        typename MeasureBoundaryDataWorker<data_type, dim>::scalar_type;      \
    using ad_type     = scalar_type;                                          \
                                                                              \
    /* Default constructor */                                                 \
    MeasureBoundary##dataname##Worker (                                       \
            const std::string &subsection_name = "",                          \
            const std::string &unprocessed_input = "")                        \
    : MeasureBoundaryDataWorker<data_type,dim> (                              \
            subsection_name, unprocessed_input) { }                           \
                                                                              \
    /* Return which data has to be provided to compute the derived
     * quantities. The flags returned here are the ones passed to
     * the constructor of this class.*/                                       \
    dealii::UpdateFlags get_needed_update_flags () const final;               \
                                                                              \
private:                                                                      \
                                                                              \
    /* Create a unique name used to access data fields in the
     * <tt>copy_data.general_storages</tt>*/                                  \
    std::string get_unique_name () const final                                \
    {   std::string name = "measured_" + std::string(EFI_STRINGIFY(dataname)) \
            + "_on_boundaries";                                               \
        for (auto id : this->active_set)                                      \
        {   name += '_';                                                      \
            name += dealii::Utilities::to_string (id); }                      \
        return name;                                                          \
    }                                                                         \
                                                                              \
    /* This function computes the forces acting on the boundary and
     * writes the local contribution to the @copy_data
     * @param[in] scratch_data Provides all necessary data to do the
     * requested computations on the boundary.
     * @param[out] copy_data Output container.*/                              \
    void                                                                      \
    do_fill (ScratchData<dim> &scratch_data,                                  \
             CopyData         &copy_data) const final;                        \
};



namespace efi {


/// Base class for measurement workers.
template <int dim>
class MeasureBoundaryDataWorkerBase : public BoundaryWorker<dim>
{
public:

    /// Default constructor.
    MeasureBoundaryDataWorkerBase (const std::string &subsection_name,
                                   const std::string &unprocessed_input);

    /// Default constructor
    virtual
    ~MeasureBoundaryDataWorkerBase () = default;

    /// Add a boundary ID to the set of active boundary IDs.
    void
    set_active (const dealii::types::boundary_id id);

    /// Check whether the field with the name returned by
    /// <tt>get_unique_name()</tt> is stored by
    /// <tt>copy_data.general_storages[idx]</tt> or not.
    bool
    measured_data_is_stored (const CopyData    &copy_data,
                             const unsigned int idx =
                             dealii::numbers::invalid_unsigned_int) const;

    /// Create a copier funciton that takes a CopyData object, extracts the
    /// data field given by <tt>get_unique_name()</tt> of type @p DataType
    /// and adds it to dst.
    /// @note The object @p dst must live at least as long as the returned
    /// copier function is in use.
    template <class DataType>
    std::function<void(const CopyData&)>
    create_copier (DataType &dst) const;

protected:

    /// Check whether the boundary ID of the face the @p scratch_data is
    /// initialized with is in the @p active_set.
    bool
    is_active (const ScratchData<dim> &scratch_data) const;

    /// Create a unique name used to access data fields in the
    /// <tt>copy_data.general_storages</tt>
    virtual
    std::string
    get_unique_name () const = 0;

    /// Set of active boundary IDs.
    std::set<dealii::types::boundary_id> active_set;
};



template <class DataType, int dim>
class MeasureBoundaryDataWorker : public MeasureBoundaryDataWorkerBase<dim>
{
public:

    /// Default constructor.
    MeasureBoundaryDataWorker (const std::string &subsection_name,
                               const std::string &unprocessed_input);

    /// Default destructor.
    virtual
    ~MeasureBoundaryDataWorker () = default;

    /// Return a reference to the data field stored in
    /// <tt>copy_data.general_storages[idx]</tt>
    /// @note the name of the field in @p copy_data is given by
    /// <tt>get_unique_name()</tt>.
    const DataType&
    get_measured_data (const CopyData    &copy_data,
                       const unsigned int idx =
                               dealii::numbers::invalid_unsigned_int) const;

protected:

    /// Add @p input_data to the data field @p dataname stored in
    /// <tt>copy_data.general_storages[idx]</tt>
    /// @note the name of the field in @p copy_data is not @p dataname, the
    /// exact name is given by <tt>get_unique_name()</tt>.
    void
    add_measured_data (CopyData       &copy_data,
                       const DataType &measured_data,
                       const unsigned int idx =
                               dealii::numbers::invalid_unsigned_int) const;
};



template <int dim>
class BoundaryWorkerGroup : public BoundaryWorker<dim>
{
public:

    BoundaryWorkerGroup ();

    dealii::UpdateFlags
    get_needed_update_flags () const final;

    void
    register_worker (const BoundaryWorker<dim>& worker);

private:

    void
    do_fill (ScratchData<dim> &scratch_data,
             CopyData &copy_data) const final;

    // Vector of all registered workers.
    std::vector<std::reference_wrapper<
        const BoundaryWorker<dim>>> workers;
};



//------------------- MEASUREMENT WORKER IMPLEMENTATIONS ---------------------//



// Measure the force acting on a boundary element.
EFI_MEASURE_BOUNDARY_DATA_WORKER (
        Force, (dealii::Tensor<1,dim,double>))

// Measure the torque around the x-axis acting on boundary element.
EFI_MEASURE_BOUNDARY_DATA_WORKER (
        TorqueX, (dealii::Tensor<1,dim,double>))

// Measure the resultant force acting radially and area.
EFI_MEASURE_BOUNDARY_DATA_WORKER (
        RadialYZ, (dealii::Tensor<1,dim,double>))

// Measure the resultant force acting perpendicularly and area.
EFI_MEASURE_BOUNDARY_DATA_WORKER (
        Spatula, (dealii::Tensor<1,dim,double>))



//---------------------- INLINE AND TEMPLATE FUNCTIONS -----------------------//



template <int dim>
MeasureBoundaryDataWorkerBase<dim>::
MeasureBoundaryDataWorkerBase (const std::string &subsection_name,
                               const std::string &/*unprocessed_input*/)
: BoundaryWorker<dim>(subsection_name)
{ }



template <int dim>
void
MeasureBoundaryDataWorkerBase<dim>::
set_active (const dealii::types::boundary_id id)
{
    this->active_set.insert(id);
}



template <int dim>
bool
MeasureBoundaryDataWorkerBase<dim>::
measured_data_is_stored (const CopyData    &copy_data,
                         const unsigned int idx) const
{
    if (idx != dealii::numbers::invalid_unsigned_int)
    {
        AssertIndexRange (idx,copy_data.size());
        return copy_data.general_storages[idx].stores_object_with_name
            (this->get_unique_name ());
    }
    else
    {
        return copy_data.general_storages.back().stores_object_with_name
            (this->get_unique_name ());
    }
}


template <int dim>
template <class DataType>
std::function<void(const CopyData&)>
MeasureBoundaryDataWorkerBase<dim>::
create_copier (DataType &dst) const
{
    using DerivedType = MeasureBoundaryDataWorker<DataType,dim>;

    Assert ((dynamic_cast<const DerivedType*>(this)!= nullptr),
            dealii::ExcMessage (
                    "Invalid pointer cast. Object measuring "
                    + this->get_unique_name()
                    + "cannot be cast to <"
                    + boost::typeindex::type_id<DerivedType>().pretty_name()
                    + ">."));

    return [this,&dst](const CopyData& copy_data)
            {
                for (unsigned int i = 0; i < copy_data.size(); ++i)
                    if (this->measured_data_is_stored (copy_data,i))
                    {
                        const auto &worker
                            = static_cast<const DerivedType&> (*this);

                        dst += worker.get_measured_data (copy_data,i);
                    }
            };
}



template <int dim>
bool
MeasureBoundaryDataWorkerBase<dim>::
is_active (const ScratchData<dim> &scratch_data) const
{
    using namespace dealii;

    Assert (dynamic_cast<const dealii::FEFaceValues<dim>*>(
            &ScratchDataTools::get_current_fe_values (scratch_data))
            != nullptr,
            ExcMessage (
                    "FEValuesBase cannot be cast into FEFaceValues. "
                    "Try to reinitialize scratch_data with the "
                    "current cell and face number before calling "
                    "this function."));

     // If the actieve_set is empty this function assumes that the entire
     // boundary is selected
     if (this->active_set.empty())
         return true;

     auto &fe  = static_cast<const dealii::FEFaceValues<dim>&> (
             ScratchDataTools::get_current_fe_values (scratch_data));

     // TODO The way how to get the face iterator we are working on is
     // stupid. However, I haven't found a better solution, yet. Adding a
     // function to dealii::FEFaceValues::get_face() would be the best
     // solution. Similar to dealii::FEValues::get_cell()
     const unsigned int face_index = fe.get_face_index();
     unsigned int i = 0;
     for (; i < dealii::GeometryInfo<dim>::faces_per_cell; ++i)
         if (face_index == fe.get_cell()->face_index(i))
             break;

     Assert (i < dealii::GeometryInfo<dim>::faces_per_cell,
             ExcMessage ("Face not found."));

     const auto &face = fe.get_cell()->face(i);

     return std::find(std::begin(this->active_set),std::end(this->active_set),
             face->boundary_id())
        != std::end(this->active_set);
}



template <class DataType, int dim>
MeasureBoundaryDataWorker<DataType,dim>::
MeasureBoundaryDataWorker (const std::string &subsection_name,
                           const std::string &unprocessed_input)
: MeasureBoundaryDataWorkerBase<dim>(subsection_name, unprocessed_input)
{ }



template <class DataType, int dim>
const DataType&
MeasureBoundaryDataWorker<DataType,dim>::
get_measured_data (const CopyData    &copy_data,
                   const unsigned int idx) const
{
    if (idx != dealii::numbers::invalid_unsigned_int)
    {
        AssertIndexRange (idx,copy_data.size());
        return copy_data.general_storages[idx].get_object_with_name<DataType>
            (this->get_unique_name ());
    }
    else
    {
        return copy_data.general_storages.back().get_object_with_name<DataType>
            (this->get_unique_name ());
    }
}



template <class DataType, int dim>
void
MeasureBoundaryDataWorker<DataType,dim>::
add_measured_data (CopyData        &copy_data,
                   const DataType  &measured_data,
                   const unsigned int idx) const
{
    if (idx != dealii::numbers::invalid_unsigned_int)
    {
        AssertIndexRange (idx,copy_data.size());
        copy_data.general_storages[idx].get_or_add_object_with_name<DataType>
            (this->get_unique_name ()) += measured_data;
    }
    else
    {
        copy_data.general_storages.back().get_or_add_object_with_name<DataType>
            (this->get_unique_name ()) += measured_data;
    }
}



template <int dim>
inline
dealii::UpdateFlags
MeasureBoundaryForceWorker<dim>::
get_needed_update_flags () const
{
    return dealii::update_JxW_values | dealii::update_normal_vectors
            | dealii::update_gradients;
}



template <int dim>
inline
dealii::UpdateFlags
MeasureBoundaryTorqueXWorker<dim>::
get_needed_update_flags () const
{
    return dealii::update_JxW_values | dealii::update_normal_vectors
            | dealii::update_gradients | dealii::update_quadrature_points
            | dealii::update_values;
}



template <int dim>
inline
dealii::UpdateFlags
MeasureBoundaryRadialYZWorker<dim>::
get_needed_update_flags () const
{
    return dealii::update_JxW_values | dealii::update_normal_vectors
            | dealii::update_gradients | dealii::update_quadrature_points
            | dealii::update_values;
}


template <int dim>
inline
dealii::UpdateFlags
MeasureBoundarySpatulaWorker<dim>::
get_needed_update_flags () const
{
    return dealii::update_JxW_values | dealii::update_normal_vectors
            | dealii::update_gradients | dealii::update_quadrature_points
            | dealii::update_values;
}


}//namespace efi



#endif /* SRC_EFI_INCLUDE_EFI_WORKER_MEASURE_DATA_WORKER_H_ */
