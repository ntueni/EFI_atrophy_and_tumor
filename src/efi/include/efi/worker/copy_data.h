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

#ifndef SRC_MYLIB_INCLUDE_EFI_WORKER_COPY_DATA_H_
#define SRC_MYLIB_INCLUDE_EFI_WORKER_COPY_DATA_H_



// deal.II headers
#include <deal.II/base/exceptions.h>
#include <deal.II/algorithms/general_data_storage.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/meshworker/copy_data.h>

// efi headers
#include <efi/base/logstream.h>


namespace efi {

struct CopyData
{
    /// CopyData constructor. The size of all data fields is set to @p m.
    /// Each of the elements of @p vectors, @p macrices, and
    /// @p local_dof_indices has the size @p size. All @p vectors and @p matrixs
    /// are zeroed and the @p local_dof_indices vectors are initialized with
    /// dealii::numbers::invalid_dof_index.
    /// @param[in] m Number of elements in @p vectors, @p matrices,
    /// @p local_dof_indices, and @p general_strorages.
    /// @param[in] size Size of the elements of @p vectors, @p matrices and
    /// @p local_dof_indices.
    CopyData (const unsigned int m = 0,
              const unsigned int size = 0);

    /// Copy constructor (deep copy).
    CopyData (const CopyData &) = default;

    /// Assignment operator (deep copy).
    CopyData&
    operator= (const CopyData &);

    /// Assign a zero to all vectors, and matricesAll dof indices are set to
    /// dealii::numbers::invalid_dof_index. The elements of general_storages
    /// are reseted via @p GeneralDataStorage::reset().
    /// @param[in]
    template <class OtherNumber,
              bool = std::enable_if<
                  std::is_convertible<double,OtherNumber>::value>::value>
    CopyData&
    operator= (const OtherNumber &val);

    /// Reinitialize this object. The size of all data fields is set to @p m.
    /// Each of the elements of @p vectors, @p matrices, and
    /// @p local_dof_indices is resized to @p size. If omit_zeroing_entries flag
    /// is set to false, all @p vectors and @p matrices are zeroed and the
    /// @p local_dof_indices vectors are set to
    /// @p dealii::numbers::invalid_dof_index.
    /// @param[in] m New number of elements in @p vectors, @p matrices,
    /// @p local_dof_indices, and @p general_strorages.
    /// @param[in] size New size of the elements of @p vectors, @p matrices and
    /// @p local_dof_indices.
    /// @param[in] omit_zeroing_entries As the name says zeroing (or equivalent
    /// actions) is omitted if this flag is set.
    void
    reinit (const unsigned int m,
            const unsigned int size,
            const bool omit_zeroing_entries = false);

    /// Emplace a new elements at the back of @p vectors, @p matrices,
    /// @p local_dof_indices and @p general_strorages. The new elements of the
    /// former three fields have the size @p size. If omit_zeroing_entries flag
    /// is set to false, all @p vectors and @p matrices are set to zero and the
    /// @p local_dof_indices vectors are set to
    /// @p dealii::numbers::invalid_dof_index.
    /// @param[in] size Size of the new element of @p vectors, @p matrices
    /// and @p local_dof_indices.
    /// @param[in] omit_zeroing_entries As the name says zeroing (or equivalent
    /// actions) is omitted if this flag is set.
    void
    emplace_back (const unsigned int size,
                  const bool omit_zeroing_entries = false);

    /// Return the size of the data vectors. If not all data fields have the
    /// same size an assertion is triggered.
    unsigned int
    size () const;

    // Vector of local vectors.
    std::vector<dealii::Vector<double>> vectors;

    // Vector of local matrices.
    std::vector<dealii::FullMatrix<double>> matrices;

    // Vector of local dof indices.
    std::vector<std::vector<dealii::types::global_dof_index>>local_dof_indices;

    // Vector of local general data storages.
    std::vector<dealii::GeneralDataStorage> general_storages;
};



/// Create a standard copier function that copies the local @p matrices and
/// @p vectors to @p vec and @p mat. The returned copier function throws a
/// dealii::ExcNumberNotFinite exception when a NaN or infinite value is found
/// in one of the @p vectors.
/// @param[out] vec Global vector into which the local @p vectors are assembled.
/// @param[out] mat Global matrix into which the local @p mactices are
/// assembled.
/// @param[in] constraints The constraints of the global objects.
template <class VectorType, class MatrixType, class Number>
std::function<void(const CopyData &)>
create_assembly_data_copier (VectorType &vec,
                             MatrixType &mat,
                             const dealii::AffineConstraints<Number>
                                &constraints);

            



//---------------------- INLINE AND TEMPLATE FUNCTIONS -----------------------//



inline
CopyData::
CopyData (const unsigned int m,
          const unsigned int size)
:
    vectors (),
    matrices (),
    local_dof_indices (),
    general_storages ()
{
    reinit (m,size);
}


inline
CopyData&
CopyData::
operator= (const CopyData &c)
{
    vectors = c.vectors;
    matrices = c.matrices;
    local_dof_indices = c.local_dof_indices;

    general_storages.resize(0);
    for (const auto &s : c.general_storages)
        general_storages.push_back (s);

    return *this;
}



template <class OtherNumber, bool>
CopyData &
CopyData::operator= (const OtherNumber &val)
{
    Assert(dealii::numbers::value_is_zero(val),
           dealii::ExcMessage("Only assignment with zero is allowed"));
    (void)val;

    for (auto &v : vectors)
        v = 0.0;
    for (auto &m : matrices)
        m = 0.0;
    for (auto &idx : local_dof_indices)
        std::fill (std::begin(idx),std::end(idx),
                        dealii::numbers::invalid_dof_index);
    for (auto &s : general_storages)
        s.reset();

    return *this;
}



void
inline
CopyData::
reinit (const unsigned int m,
        const unsigned int size,
        const bool omit_zeroing_entries)
{
    vectors.resize(m);
    matrices.resize(m);
    local_dof_indices.resize(m);
    general_storages.resize(m);

    for (auto &v : vectors)
        v.reinit (size,omit_zeroing_entries);
    for (auto &m : matrices)
        m.reinit (size,size,omit_zeroing_entries);
    for (auto &idx : local_dof_indices)
    {
        if (omit_zeroing_entries)
            idx.resize (size);
        else
        {
            idx.resize (0);
            idx.resize (size,dealii::numbers::invalid_dof_index);
        }
    }
    for (auto s : general_storages)
        s.reset();
}



void
inline
CopyData::
emplace_back (const unsigned int size,
              const bool omit_zeroing_entries)
{
    vectors.emplace_back(size);
    matrices.emplace_back(size,size);
    general_storages.emplace_back();
    if (omit_zeroing_entries)
        local_dof_indices.emplace_back(size);
    else
    {
        local_dof_indices.emplace_back(size,dealii::numbers::invalid_dof_index);
        matrices.back() = 0;
    }


}



unsigned int
inline
CopyData::
size () const
{
    AssertDimension (vectors.size(),matrices.size());
    AssertDimension (vectors.size(),local_dof_indices.size());
    AssertDimension (vectors.size(),general_storages.size());

    return vectors.size();
}



template <class VectorType, class MatrixType, class Number>
std::function<void(const CopyData &)>
create_assembly_data_copier (VectorType &vec,
                             MatrixType &mat,
                             const dealii::AffineConstraints<Number>
                                &constraints)
{
    return [&vec, &mat, &constraints] (const CopyData &copy_data)
            {
                for (unsigned int i = 0; i < copy_data.size(); ++i)
                {
                    // Check if all numbers in the vectors
                    // are finite. If not throw an error.
                    for(auto &value : copy_data.vectors[i])
                        AssertThrow (dealii::numbers::is_finite(value),
                                     dealii::ExcNumberNotFinite(value));

                    constraints.distribute_local_to_global(
                            copy_data.matrices[i],
                            copy_data.vectors[i],
                            copy_data.local_dof_indices[i],
                            mat,
                            vec, true);
                }
            };
}



template <class VectorType, class MatrixType, class Number>
std::function<void(const CopyData &)>
create_assembly_data_copier (VectorType &vec,
                             MatrixType &mat,
                             dealii::SolverControl::State &state,
                             const dealii::AffineConstraints<Number>
                                &constraints)
{
    return [&vec, &mat, &constraints, &state] (const CopyData &copy_data)
            {
                if (state == dealii::SolverControl::State::failure)
                    return;

                try
                {
                    for (unsigned int i = 0; i < copy_data.size(); ++i)
                    {
                        // Check if all numbers in the vectors
                        // are finite. If not throw an error.

                        for(auto &value : copy_data.vectors[i])
                            {
                                // std::cout << "VALUE:  " << value << std::endl;
                                AssertThrow (dealii::numbers::is_finite(value),
                                         dealii::ExcNumberNotFinite(value));
                            }

                        constraints.distribute_local_to_global(
                                copy_data.matrices[i],
                                copy_data.vectors[i],
                                copy_data.local_dof_indices[i],
                                mat,
                                vec, true);
                    }
                }
                catch (dealii::ExceptionBase &exec)
                {
                    state = dealii::SolverControl::State::failure;
                    efilog(Verbosity::normal) << "Copier failed."
                                              << std::endl;
                    efilog(Verbosity::normal) << exec.what()
                                              << std::endl;
                    efilog(Verbosity::normal) << "Copier failed."
                    << std::endl;
                    state = dealii::SolverControl::State::failure;
                    return;
                }
            };
}

template <class VectorType, class MatrixType, class Number>
std::function<void(const CopyData &)>
create_residual_data_copier (VectorType &vec,
                             MatrixType &mat,
                             dealii::SolverControl::State &state,
                             const dealii::AffineConstraints<Number>
                                &constraints)
{
    return [&vec, &mat, &constraints, &state] (const CopyData &copy_data)
            {
                if (state == dealii::SolverControl::State::failure)
                    return;

                try
                {
                    
                    for (unsigned int i = 0; i < copy_data.size(); ++i)
                    {
                        // Check if all numbers in the vectors
                        // are finite. If not throw an error.

                        for(auto &value : copy_data.vectors[i])
                            {
                                // std::cout << "VALUE:  " << value << std::endl;
                                AssertThrow (dealii::numbers::is_finite(value),
                                         dealii::ExcNumberNotFinite(value));
                            }

                        constraints.distribute_local_to_global(
                                copy_data.vectors[i],
                                copy_data.local_dof_indices[i],
                                vec);
                    }
                }
                catch (dealii::ExceptionBase &exec)
                {
                    state = dealii::SolverControl::State::failure;
                    efilog(Verbosity::normal) << "Copier failed."
                                              << std::endl;
                    efilog(Verbosity::normal) << exec.what()
                                              << std::endl;
                    efilog(Verbosity::normal) << "Copier failed."
                    << std::endl;
                    state = dealii::SolverControl::State::failure;
                    return;
                }
            };
}
}// namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_WORKER_COPY_DATA_H_ */
