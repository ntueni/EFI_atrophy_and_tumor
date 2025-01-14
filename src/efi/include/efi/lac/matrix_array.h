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

#ifndef SRC_MYLIB_INCLUDE_EFI_LAC_MATRIX_ARRAY_H_
#define SRC_MYLIB_INCLUDE_EFI_LAC_MATRIX_ARRAY_H_

// deal.II headers
#include <deal.II/base/utilities.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_indices.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/base/numbers.h>

// boost headers
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/collection_size_type.hpp>


BOOST_SERIALIZATION_SPLIT_FREE(dealii::BlockIndices)

namespace boost {
namespace serialization {

template <class Archive>
inline
void save (Archive& ar,
           const dealii::BlockIndices& g,
           const unsigned int /* file_version */) {

    std::vector<typename dealii::BlockIndices::size_type> block_sizes (g.size());

    for(unsigned int i = 0; i < block_sizes.size (); ++i)
        block_sizes[i] = g.block_size (i);
    ar << block_sizes;
}



template <class Archive>
inline
void load (Archive& ar,
           dealii::BlockIndices& g,
           const unsigned int /* file_version */)
{
    std::vector<typename dealii::BlockIndices::size_type> block_sizes;
    ar >> block_sizes;
    g.reinit (block_sizes);
}



template <class Archive, class number>
inline
void save (Archive & ar,
           const dealii::BlockVector<number> & v,
           const unsigned int /* file_version */)
{
    typedef typename dealii::BlockVector<number>::BlockType block_type;
    const collection_size_type count(v.n_blocks());
    ar << BOOST_SERIALIZATION_NVP(count);

    if (v.n_blocks() > 0)
        ar << serialization::make_array<const block_type, collection_size_type>(
            static_cast<const block_type *>(&v.block(0)),
            count
        );
}



template <class Archive, class number>
inline
void load (Archive & ar,
           dealii::BlockVector<number> & v,
           const unsigned int /* file_version */)
{
    typedef typename dealii::BlockVector<number>::BlockType block_type;
    collection_size_type count(v.n_blocks());
    ar >> BOOST_SERIALIZATION_NVP(count);
    v.reinit(count);

    if (v.n_blocks() > 0)
        ar >> serialization::make_array<block_type, collection_size_type>(
            static_cast<block_type *>(&v.block(0)),
            count
        );
    v.collect_sizes();
}



template<class Archive, class number>
inline
void serialize (Archive & ar,
                dealii::BlockVector<number> & t,
                unsigned int file_version)
{
        split_free(ar, t, file_version);
}

} // namespace serialization
} // namespace boost




namespace efi {

namespace efi_internal {

// This is just a very basic
template <class T>
class Table
{
public:

    typedef dealii::types::global_dof_index  size_type;

    Table (const size_type n_rows = 0,
           const size_type n_cols = 0);

    Table (const Table &) = default;

    T &operator() (const size_type i,
                   const size_type j);

    const T &operator() (const size_type i,
                         const size_type j) const;

    bool empty () const;

    size_type n_rows () const;
    size_type n_cols () const;
    size_type n_elements () const;

    void reinit (const size_type n_rows,
                 const size_type n_cols);

private:

    friend class boost::serialization::access;

    template <class Archive>
    void serialize (Archive &ar, const unsigned int);


    dealii::TableIndices<2> table_size;
    std::vector<T> values;
};

}//close internal



template <class number>
class MatrixArray {

public:

    typedef dealii::FullMatrix<number> block_type;

    typedef typename block_type::value_type   value_type;
    typedef value_type                      *pointer;
    typedef const value_type                *const_pointer;
    typedef value_type                      &reference;
    typedef const value_type                &const_reference;
    typedef dealii::types::global_dof_index  size_type;

    class const_iterator;

//    class Accessor
//    {
//    public:
//
//      Accessor (const MatrixArray<number> *matrix,
//                const size_type row,
//                const size_type col);
//
//      size_type row() const;
//      size_type column() const;
//      number value() const;
//
//    protected:
//
//      const MatrixArray<number> *matrix;
//      size_type a_row;
//      size_type a_col;
//
//      friend class const_iterator;
//    };
//
//
//    class const_iterator
//    {
//    public:
//
//      const_iterator(const MatrixArray<number> *matrix,
//                     const size_type row,
//                     const size_type col);
//
//      const_iterator &operator++ ();
//      const_iterator operator++ (int);
//
//      const Accessor &operator* () const;
//      const Accessor *operator-> () const;
//
//      bool operator == (const const_iterator &) const;
//      bool operator != (const const_iterator &) const;
//
//      bool operator < (const const_iterator &) const;
//      bool operator > (const const_iterator &) const;
//
//    private:
//
//      Accessor accessor;
//    };

    ~MatrixArray ();

    MatrixArray (const std::vector<size_type> &block_sizes = {});

    MatrixArray (const std::vector<size_type> &row_block_sizes,
                 const std::vector<size_type> &col_block_sizes);

    MatrixArray (const dealii::BlockIndices &block_indices);

    MatrixArray (const dealii::BlockIndices &row_block_indices,
                 const dealii::BlockIndices &column_block_indices);

    MatrixArray (const MatrixArray<number> &) = default;

    MatrixArray<number> & operator= (const MatrixArray<number> &) = default;
    MatrixArray<number> & operator= (const double d);

    MatrixArray<number> & operator*= (const double );
    MatrixArray<number> & operator/= (const double );

    bool empty () const;

    void clear ();

    void reinit (const std::vector<size_type> &row_block_sizes);

    void reinit (const std::vector<size_type> &row_block_sizes,
                 const std::vector<size_type> &col_block_sizes);

    void reinit (const dealii::BlockIndices &block_indices);

    void reinit (const dealii::BlockIndices &row_block_indices,
                 const dealii::BlockIndices &column_block_indices);

    void copy_to (dealii::FullMatrix<double> &matrix);

    // add the (transposed) matrix to the block (row, column)
    template <class MatrixType>
    void block_add (const unsigned int row,
                    const unsigned int column,
                    const MatrixType  &matrix,
                    const bool         transposed = false);

    // add the (transposed) matrix to the block (row, column)
    template <class MatrixType>
    void block_add (const unsigned int row,
                    const unsigned int column,
                    const number       factor,
                    const MatrixType  &matrix,
                    const bool         transposed = false);

    // set the block (row, column) equal to the (transposed) matrix
    template <class MatrixType>
    void block_set (const unsigned int row,
                    const unsigned int column,
                    const MatrixType  &matrix,
                    const bool         transposed = false);

    // set value
    void set (const size_type i,
              const size_type j,
              const value_type value);

    // add value
    void add (const size_type i,
              const size_type j,
              const value_type value);

    void add (const number d,
              const MatrixArray<number> &src);

    void symmetrize ();

    // add a matrix
    template <class MatrixType>
    void add (const MatrixType &matrix,
              const size_type dst_offset_i,
              const size_type dst_offset_j);

    // add a matrix
    template <class MatrixType>
    void add (const number      factor,
              const MatrixType &matrix,
              const size_type dst_offset_i,
              const size_type dst_offset_j);

    // set a matrix
    template <class MatrixType>
    void set (const MatrixType &matrix,
              const size_type dst_offset_i,
              const size_type dst_offset_j);

    // add a matrix
    template <class MatrixType>
    void add (const MatrixType &matrix,
              const std::pair<unsigned int, size_type> &dst_offset_i,
              const std::pair<unsigned int, size_type> &dst_offset_j);

    // add a matrix
    template <class MatrixType>
    void add (const number      factor,
              const MatrixType &matrix,
              const std::pair<unsigned int, size_type> &dst_offset_i,
              const std::pair<unsigned int, size_type> &dst_offset_j);

    // set a matrix
    template <class MatrixType>
    void set (const MatrixType &matrix,
              const std::pair<unsigned int, size_type> &dst_offset_i,
              const std::pair<unsigned int, size_type> &dst_offset_j);

    value_type operator () (const size_type i,
                            const size_type j) const;

    value_type el (const size_type i,
                   const size_type j) const;

    size_type m () const;
    size_type n () const;

    unsigned int n_block_rows () const;
    unsigned int n_block_cols () const;

    template <class BlockVectorType>
    void vmult (BlockVectorType       &dst,
                const BlockVectorType &src) const;

    template <class BlockVectorType>
    void Tvmult (BlockVectorType       &dst,
                 const BlockVectorType &src) const;

    template <class BlockVectorType>
    void vmult_add (BlockVectorType       &dst,
                    const BlockVectorType &src) const;

    template <class BlockVectorType>
    void Tvmult_add (BlockVectorType       &dst,
                     const BlockVectorType &src) const;

    template <class number2>
    void mmult (MatrixArray<number2>& C,
                const MatrixArray<number2>& B,
                const bool adding = false) const;

    template <class number2>
    void Tmmult (MatrixArray<number2>& C,
                 const MatrixArray<number2>& B,
                 const bool adding = false) const;

    template <class number2>
    void mTmult (MatrixArray<number2>& C,
                 const MatrixArray<number2>& B,
                 const bool adding = false) const;

    template <class number2>
    void TmTmult (MatrixArray<number2>& C,
                  const MatrixArray<number2>& B,
                  const bool adding = false) const;

    // print the matrix as a whole
    template <typename StreamType>
    void print (StreamType         &s,
                const unsigned int  width = 5,
                const unsigned int  precision = 2) const;

    // Technically, all blocks always exist. However,
    // call a block as non-existing if it is empty.
    bool block_exists (const unsigned int row,
                       const unsigned int column) const;

    void create_block (const unsigned int row,
                       const unsigned int column,
                       const bool omit_zeroing_entries = true);

    block_type &
    block (const unsigned int row,
           const unsigned int column);

    const block_type &
    block (const unsigned int row,
           const unsigned int column) const;




    DeclException0 (ExcBlockDimensionMismatch);
    DeclException0 (ExcEmptyMatrix);

    const dealii::BlockIndices& get_row_block_indices () const;
    const dealii::BlockIndices& get_column_block_indices () const;

private:

    friend class boost::serialization::access;

    template <class Archive>
    inline
    void serialize (Archive &ar, const unsigned int);

    dealii::BlockIndices row_block_indices;
    dealii::BlockIndices column_block_indices;

    efi_internal::Table<block_type> sub_objects;
};



namespace efi_internal {

template <class T>
inline
Table<T>::Table (const size_type n_rows,
                 const size_type n_cols)
    :
    table_size (n_rows, n_cols),
    values (n_rows * n_cols)
{
    if (n_elements() == 0)
        table_size = dealii::TableIndices<2>();
}



template <class T>
inline
T& Table<T>::operator() (const size_type i,
                         const size_type j)
{
    AssertIndexRange (i, n_rows());
    AssertIndexRange (j, n_cols());

    return values [i * n_cols() + j];
}

template <class T>
inline
const T& Table<T>::operator() (const size_type i,
                               const size_type j) const
{
    AssertIndexRange (i, n_rows());
    AssertIndexRange (j, n_cols());

    return values [i * n_cols() + j];
}

template <class T>
bool
Table<T>::empty () const
{
    return values.empty();
}

template <class T>
inline
typename Table<T>::size_type
Table<T>::n_rows () const
{
    return table_size[0];
}

template <class T>
inline
typename Table<T>::size_type
Table<T>::n_cols () const
{
    return table_size[1];
}

template <class T>
inline
typename Table<T>::size_type
Table<T>::n_elements () const
{
    return table_size[0] * table_size[1];
}

template <class T>
inline
void
Table<T>::reinit (const size_type n_rows,
                  const size_type n_cols)
{
    table_size = dealii::TableIndices<2>(n_rows, n_cols);

    unsigned int new_size = n_elements();

    if (new_size == 0)
    {
        values.resize (0);
        table_size = dealii::TableIndices<2>();
        return;
    }

    values.clear();
    values.resize (new_size);
}

template <class T>
template <class Archive>
inline
void
Table<T>::serialize (Archive &ar, const unsigned int)
{
    ar & table_size;
    ar & values;
}

}//close internal



template <class number>
inline
MatrixArray<number>::~MatrixArray ()
{ }



template <class number>
inline
MatrixArray<number>::MatrixArray (const std::vector<size_type> &block_sizes)
    :
    MatrixArray<number>(block_sizes, block_sizes)
{ }



template <class number>
inline
MatrixArray<number>::MatrixArray (const std::vector<size_type> &row_block_sizes,
                                  const std::vector<size_type> &col_block_sizes)
    :
    row_block_indices (row_block_sizes),
    column_block_indices (col_block_sizes),
    sub_objects (row_block_sizes.size(), col_block_sizes.size())
{ }



template <class number>
inline
MatrixArray<number>::MatrixArray (const dealii::BlockIndices &block_indices)
    :
    MatrixArray<number>(block_indices,block_indices)
{ }



template <class number>
inline
MatrixArray<number>::MatrixArray (const dealii::BlockIndices &row_block_indices,
                                  const dealii::BlockIndices &column_block_indices)
    :
    row_block_indices (row_block_indices),
    column_block_indices (column_block_indices),
    sub_objects (row_block_indices.size(), column_block_indices.size())
{ }



template <class number>
inline
MatrixArray<number>&
MatrixArray<number>::operator = (const double d)
{
    Assert (d==0, dealii::ExcScalarAssignmentOnlyForZeroValue());


    sub_objects.reinit (row_block_indices.size(), column_block_indices.size());

    return *this;

    // avoid compiler warning
    (void)d;
}



template <class number>
inline
MatrixArray<number>&
MatrixArray<number>::operator *= (const double d)
{
    if (d==0) return operator=(d);

    for (size_type r=0; r<this->n_block_rows(); ++r)
        for (size_type c=0; c<this->n_block_cols(); ++c)
            if (block_exists(r,c))
                block(r,c) *= d;
    return *this;
}



template <class number>
inline
MatrixArray<number>&
MatrixArray<number>::operator /= (const double d)
{
    AssertIsFinite(1./d);

    for (size_type r=0; r<this->n_block_rows(); ++r)
        for (size_type c=0; c<this->n_block_cols(); ++c)
            if (block_exists(r,c))
                block(r,c) /= d;
    return *this;
}



template <class number>
inline
bool
MatrixArray<number>::empty () const
{
    return sub_objects.empty();
}



template <class number>
inline
void
MatrixArray<number>::reinit (const std::vector<size_type> &block_sizes)
{
    reinit (block_sizes, block_sizes);
}



template <class number>
inline
void
MatrixArray<number>::reinit (const std::vector<size_type> &row_block_sizes,
                             const std::vector<size_type> &col_block_sizes)
{
    this->row_block_indices.reinit (row_block_sizes);
    this->column_block_indices.reinit (col_block_sizes);
    this->sub_objects.reinit (row_block_sizes.size(), col_block_sizes.size());
}



template <class number>
inline
void
MatrixArray<number>::reinit (const dealii::BlockIndices &block_indices)
{
    reinit (block_indices, block_indices);
}



template <class number>
inline
void
MatrixArray<number>::reinit (const dealii::BlockIndices &row_block_indices,
                             const dealii::BlockIndices &col_block_indices)
{
    this->row_block_indices = row_block_indices;
    this->column_block_indices = col_block_indices;
    this->sub_objects.reinit (row_block_indices.size(), col_block_indices.size());
}



template <class number>
inline
void
MatrixArray<number>::copy_to (dealii::FullMatrix<double> &matrix)
{
    AssertDimension (m(), matrix.m());
    AssertDimension (n(), matrix.n());

    matrix = 0;

    for (unsigned int r = 0; r < n_block_rows(); ++r)
        for(unsigned int c = 0; c < n_block_cols(); ++c)
        {
            if(block_exists(r,c))
            {
                matrix.fill(block(r,c),
                            row_block_indices.block_start(r),
                            column_block_indices.block_start(c));
            }
        }
}



template <class number>
template <class MatrixType>
inline
void
MatrixArray<number>::block_add (const unsigned int row,
                                const unsigned int column,
                                const MatrixType  &matrix,
                                const bool         transposed)
{
    AssertIndexRange (row, n_block_rows());
    AssertIndexRange (column, n_block_cols());

    create_block(row, column, false);

    if(!transposed)
    {
        AssertDimension (matrix.m(), row_block_indices.block_size(row));
        AssertDImension (matrix.n(), column_block_indices.block_size(column));

        for(size_type i = 0; i < matrix.m(); ++i)
            for(size_type j = 0; j < matrix.n(); ++j)
                block(row,column).add(i, j, matrix(i,j));
    }
    else
    {
        AssertDimension (matrix.m(), column_block_indices.block_size(column));
        AssertDimension (matrix.n(), row_block_indices.block_size(row));

        for(size_type i = 0; i < matrix.n(); ++i)
            for(size_type j = 0; j < matrix.m(); ++j)
                block(row,column).add(i, j, matrix(j,i));
    }
}



template <class number>
template <class MatrixType>
inline
void
MatrixArray<number>::block_add (const unsigned int row,
                                const unsigned int column,
                                const number       factor,
                                const MatrixType  &matrix,
                                const bool         transposed)
{
    AssertIndexRange (row, n_block_rows());
    AssertIndexRange (column, n_block_cols());

    create_block(row, column, false);

    if(!transposed)
    {
        AssertDimension (matrix.m(), row_block_indices.block_size(row));
        AssertDImension (matrix.n(), column_block_indices.block_size(column));

        for(size_type i = 0; i < matrix.m(); ++i)
            for(size_type j = 0; j < matrix.n(); ++j)
                block(row,column).add(i, j, factor*matrix(i,j));
    }
    else
    {
        AssertDimension (matrix.m(), column_block_indices.block_size(column));
        AssertDimension (matrix.n(), row_block_indices.block_size(row));

        for(size_type i = 0; i < matrix.n(); ++i)
            for(size_type j = 0; j < matrix.m(); ++j)
                block(row,column).add(i, j, factor*matrix(j,i));
    }
}



template <class number>
template <class MatrixType>
inline
void
MatrixArray<number>::block_set (const unsigned int row,
                                const unsigned int column,
                                const MatrixType  &matrix,
                                const bool         transposed)
{
    AssertIndexRange (row, n_block_rows());
    AssertIndexRange (column, n_block_cols());

    create_block(row, column);

    if(!transposed)
    {
        AssertDimension (matrix.m(), row_block_indices.block_size(row));
        AssertDimension (matrix.n(), column_block_indices.block_size(column));

        for(size_type i = 0; i < matrix.m(); ++i)
            for(size_type j = 0; j < matrix.n(); ++j)
                block(row,column).set(i, j, matrix(i,j));
    }
    else
    {
        AssertDimension (matrix.m(), column_block_indices.block_size(column));
        AssertDimension (matrix.n(), row_block_indices.block_size(row));

        for(size_type i = 0; i < matrix.n(); ++i)
            for(size_type j = 0; j < matrix.m(); ++j)
                block(row,column).set(i, j, matrix(j,i));
    }
}



template <class number>
inline
void
MatrixArray<number>::set (const size_type i,
                          const size_type j,
                          const value_type value)
{
    AssertIsFinite(value);

    const std::pair<unsigned int,size_type>
    row_index = row_block_indices.global_to_local (i),
    col_index = column_block_indices.global_to_local (j);

    create_block(row_index.first,col_index.first,false);
    block(row_index.first,col_index.first).set (row_index.second,
                                                col_index.second,
                                                value);
}


template <class number>
inline
void
MatrixArray<number>::add (const size_type i,
                          const size_type j,
                          const value_type value)
{
    AssertIsFinite(value);

    const std::pair<unsigned int,size_type>
    row_index = row_block_indices.global_to_local (i),
    col_index = column_block_indices.global_to_local (j);

    create_block(row_index.first,col_index.first,false);
    block(row_index.first,col_index.first).add (row_index.second,
                                                col_index.second,
                                                value);
}



template <class number>
inline
void
MatrixArray<number>::add(const number d,
                         const MatrixArray<number> &src)
{
    AssertDimension (src.n_block_rows(), n_block_rows());
    AssertDimension (src.n_block_cols(), n_block_cols());
    AssertDimension (src.m(), m());
    AssertDimension (src.n(), n());

    for (size_type r=0; r<src.n_block_rows(); ++r)
        for (size_type c=0; c<src.n_block_cols(); ++c)
            if (src.block_exists(r,c))
            {
                create_block(r,c,false);
                block(r,c).add(d,src.block(r,c));
            }
}



template <class number>
inline
void
MatrixArray<number>::symmetrize ()
{
    AssertDimension (m(), n());

    for (size_type r=0; r< n_block_rows(); ++r)
        for (size_type c=r+1; c< n_block_cols(); ++c)
        {
            if (block_exists(c,r))
            {
                if(block_exists(r,c))
                {
                    block(r,c) *= 0.5;
                    block(r,c).Tadd(block(c,r), 0.5);
                    block(c,r).copy_transposed(block(r,c));
                }
                else
                {
                    create_block(r,c,false);
                    block(r,c).Tadd(block(c,r), 0.5);
                    block(c,r).copy_transposed(block(r,c));
                }
            }
            else if(block_exists(r,c))
            {
                block(r,c) *= 0.5;
                create_block(c,r,false);
                block(c,r).copy_transposed(block(r,c));
            }
        }
}



template <class number>
template <class MatrixType>
inline
void
MatrixArray<number>::add (const MatrixType &matrix,
                          const size_type dst_offset_i,
                          const size_type dst_offset_j)
{
    AssertIndexRange (dst_offset_i+matrix.m()-1, m());
    AssertIndexRange (dst_offset_j+matrix.n()-1, n());

    add (matrix,
         row_block_indices.global_to_local (dst_offset_i),
         column_block_indices.global_to_local (dst_offset_j));
}


template <class number>
template <class MatrixType>
inline
void
MatrixArray<number>::add (const number      factor,
                          const MatrixType &matrix,
                          const size_type dst_offset_i,
                          const size_type dst_offset_j)
{
    AssertIndexRange (dst_offset_i+matrix.m()-1, m());
    AssertIndexRange (dst_offset_j+matrix.n()-1, n());

    add (factor,
         matrix,
         row_block_indices.global_to_local (dst_offset_i),
         column_block_indices.global_to_local (dst_offset_j));
}


template <class number>
template <class MatrixType>
inline
void
MatrixArray<number>::set (const MatrixType &matrix,
                          const size_type dst_offset_i,
                          const size_type dst_offset_j)
{
    AssertIndexRange (dst_offset_i+matrix.m()-1, m());
    AssertIndexRange (dst_offset_j+matrix.n()-1, n());

    set (matrix,
         row_block_indices.global_to_local (dst_offset_i),
         column_block_indices.global_to_local (dst_offset_j));
}


template <class number>
template <class MatrixType>
inline
void
MatrixArray<number>::add (const MatrixType &matrix,
                 const std::pair<unsigned int, size_type> &dst_offset_i,
                 const std::pair<unsigned int, size_type> &dst_offset_j)
{
    AssertIndexRange (row_block_indices.local_to_global(
            dst_offset_i.first,dst_offset_i.second)+matrix.m()-1, m());
    AssertIndexRange (column_block_indices.local_to_global(
            dst_offset_j.first,dst_offset_j.second)+matrix.n()-1, n());

    size_type local_row, local_col, local_col_begin;
    unsigned int block_row, block_col, block_col_begin;

    std::tie(block_row, local_row) = dst_offset_i;
    std::tie(block_col_begin, local_col_begin) = dst_offset_j;

    size_type i = 0;

    for(; i<matrix.m(); ++block_row)
    {
        size_type local_row_end = std::min (row_block_indices.block_size(block_row),
                                            local_row + matrix.m() - i);

        for(; local_row != local_row_end; ++local_row, ++i)
        {
            size_type j = 0;

            local_col = local_col_begin;
            block_col = block_col_begin;

            for(; j<matrix.n(); ++block_col)
            {
                create_block (block_row, block_col, false);

                size_type local_col_end = std::min (column_block_indices.block_size(block_col),
                                                    local_col + matrix.n() - j);

                for(; local_col != local_col_end; ++local_col, ++j) {
                    block(block_row, block_col).set (local_row, local_col, matrix(i,j));
                }
                local_col = 0;
            }
        }
        local_row = 0;
    }
}


template <class number>
template <class MatrixType>
inline
void
MatrixArray<number>::add (const number      factor,
                          const MatrixType &matrix,
                          const std::pair<unsigned int, size_type> &dst_offset_i,
                          const std::pair<unsigned int, size_type> &dst_offset_j)
{
    AssertIndexRange (row_block_indices.local_to_global(
            dst_offset_i.first,dst_offset_i.second)+matrix.m()-1, m());
    AssertIndexRange (column_block_indices.local_to_global(
            dst_offset_j.first,dst_offset_j.second)+matrix.n()-1, n());

    size_type local_row, local_col, local_col_begin;
    unsigned int block_row, block_col, block_col_begin;

    std::tie(block_row, local_row) = dst_offset_i;
    std::tie(block_col_begin, local_col_begin) = dst_offset_j;

    size_type i = 0;

    for(; i<matrix.m(); ++block_row)
    {
        size_type local_row_end = std::min (row_block_indices.block_size(block_row),
                                            local_row + matrix.m() - i);

        for(; local_row != local_row_end; ++local_row, ++i)
        {
            size_type j = 0;

            local_col = local_col_begin;
            block_col = block_col_begin;

            for(; j<matrix.n(); ++block_col)
            {
                create_block (block_row, block_col, false);

                size_type local_col_end = std::min (column_block_indices.block_size(block_col),
                                                    local_col + matrix.n() - j);

                for(; local_col != local_col_end; ++local_col, ++j) {
                    block(block_row, block_col).add (local_row, local_col, factor * matrix(i,j));
                }
                local_col = 0;
            }
        }
        local_row = 0;
    }
}


template <class number>
template <class MatrixType>
inline
void
MatrixArray<number>::set (const MatrixType &matrix,
                          const std::pair<unsigned int, size_type> &dst_offset_i,
                          const std::pair<unsigned int, size_type> &dst_offset_j)
{
    AssertIndexRange (row_block_indices.local_to_global(
            dst_offset_i.first,dst_offset_i.second)+matrix.m()-1, m());
    AssertIndexRange (column_block_indices.local_to_global(
            dst_offset_j.first,dst_offset_j.second)+matrix.n()-1, n());

    size_type local_row, local_col, local_col_begin;
    unsigned int block_row, block_col, block_col_begin;

    std::tie(block_row, local_row) = dst_offset_i;
    std::tie(block_col_begin, local_col_begin) = dst_offset_j;

    size_type i = 0;

    for(; i<matrix.m(); ++block_row)
    {
        size_type local_row_end = std::min (row_block_indices.block_size(block_row),
                                            local_row + matrix.m() - i);

        for(; local_row != local_row_end; ++local_row, ++i)
        {
            size_type j = 0;

            local_col = local_col_begin;
            block_col = block_col_begin;

            for(; j<matrix.n(); ++block_col)
            {
                create_block (block_row, block_col);

                size_type local_col_end = std::min (column_block_indices.block_size(block_col),
                                                    local_col + matrix.n() - j);

                for(; local_col != local_col_end; ++local_col, ++j) {
                    block(block_row, block_col).set (local_row, local_col, matrix(i,j));
                }
                local_col = 0;
            }
        }
        local_row = 0;
    }
}


template <class number>
inline
typename MatrixArray<number>::value_type
MatrixArray<number>::operator () (const size_type i,
                                  const size_type j) const
{
    const std::pair<unsigned int,size_type>
    row_index = row_block_indices.global_to_local (i),
    col_index = column_block_indices.global_to_local (j);

    Assert(block_exists(row_index.first,col_index.first),
            dealii::ExcMessage ("Block "
                    + dealii::Utilities::int_to_string(row_index.first) +", "
                    + dealii::Utilities::int_to_string(col_index.first)
                    + " does not exist."));

    return block(row_index.first,col_index.first) (row_index.second,
                                                   col_index.second);
}



template <class number>
typename MatrixArray<number>::value_type
MatrixArray<number>::el (const size_type i,
                         const size_type j) const
{
    const std::pair<unsigned int,size_type>
    row_index = row_block_indices.global_to_local (i),
    col_index = column_block_indices.global_to_local (j);

    return block_exists(row_index.first,col_index.first) ?
            block(row_index.first,col_index.first)(row_index.second,col_index.second)
            : number(0);
}



template <class number>
inline
typename MatrixArray<number>::block_type&
MatrixArray<number>::block (const unsigned int row,
                            const unsigned int column)
{
    AssertIndexRange (row,n_block_rows());
    AssertIndexRange (column,n_block_cols());
    Assert(block_exists(row,column),
            dealii::ExcMessage ("Block "
                    +dealii::Utilities::int_to_string(row)+", "
                    +dealii::Utilities::int_to_string(column)
                    +" does not exist."));

    return sub_objects (row,column);
}



template <class number>
inline
const typename MatrixArray<number>::block_type&
MatrixArray<number>::block (const unsigned int row,
                            const unsigned int column) const
{
    AssertIndexRange (row,n_block_rows());
    AssertIndexRange (column,n_block_cols());
    Assert(block_exists(row,column),
            dealii::ExcMessage ("Block "
                    +dealii::Utilities::int_to_string(row)+", "
                    +dealii::Utilities::int_to_string(column)
                    +" does not exist."));

    return sub_objects (row,column);
}



template <class number>
inline
typename MatrixArray<number>::size_type
MatrixArray<number>::m () const
{
    return row_block_indices.total_size();
}



template <class number>
inline
typename MatrixArray<number>::size_type
MatrixArray<number>::n () const
{
  return column_block_indices.total_size();
}



template <class number>
inline
unsigned int
MatrixArray<number>::n_block_rows () const
{
  return row_block_indices.size();
}



template <class number>
inline
unsigned int
MatrixArray<number>::n_block_cols () const
{
  return column_block_indices.size();
}


template <class number>
template <class BlockVectorType>
inline
void
MatrixArray<number>::vmult (BlockVectorType       &dst,
                            const BlockVectorType &src) const
{
    Assert (!empty(), dealii::ExcEmptyObject());
    AssertDimension (src.n_blocks(), this->n_block_cols());
    AssertDimension (dst.n_blocks(), this->n_block_rows());

    for(unsigned int r = 0; r < n_block_rows(); ++r) {
        if(block_exists(r,0))
            block(r,0).vmult(dst.block(r),src.block(0));
        else
            dst.block(r) = 0;

        for(unsigned int c = 0; c < n_block_cols(); ++c)
        {
            if(block_exists(r,c))
                block(r,c).vmult_add(dst.block(r),src.block(c));
        }
    }
}



template <class number>
template <class BlockVectorType>
inline
void
MatrixArray<number>::Tvmult (BlockVectorType       &dst,
                             const BlockVectorType &src) const
{
    Assert (!empty(), dealii::ExcEmptyObject());
    AssertDimension (src.n_blocks(), this->n_block_rows());
    AssertDimension (dst.n_blocks(), this->n_block_cols());

    for(unsigned int c = 0; c < n_block_cols(); ++c) {
        if(block_exists(0,c))
            block(0,c).Tvmult(dst.block(c),src.block(0));
        else
            dst.block(c) = 0;

        for(unsigned int r = 0; r < n_block_rows(); ++r)
        {
            if(block_exists(r,c))
                block(r,c).Tvmult_add(dst.block(c),src.block(r));
        }
    }
}



template <class number>
template <class BlockVectorType>
inline
void
MatrixArray<number>::vmult_add (BlockVectorType       &dst,
                                const BlockVectorType &src) const
{
    Assert (!empty(), dealii::ExcEmptyObject());
    AssertDimension (src.n_blocks(), this->n_block_cols());
    AssertDimension (dst.n_blocks(), this->n_block_rows());

    for(unsigned int r = 0; r < n_block_rows(); ++r) {
        for(unsigned int c = 0; c < n_block_cols(); ++c)
        {
            if(block_exists(r,c))
                block(r,c).vmult_add(dst.block(r),src.block(c));
        }
    }
}

template <class number>
template <class BlockVectorType>
inline
void
MatrixArray<number>::Tvmult_add (BlockVectorType       &dst,
                                 const BlockVectorType &src) const
{
    Assert (!empty(), dealii::ExcEmptyObject());
    AssertDimension (src.n_blocks(), this->n_block_rows());
    AssertDimension (dst.n_blocks(), this->n_block_cols());

    for(unsigned int c = 0; c < n_block_cols(); ++c) {
        for(unsigned int r = 0; r < n_block_rows(); ++r)
        {
            if(block_exists(r,c))
                block(r,c).Tvmult_add(dst.block(c),src.block(r));
        }
    }
}

template <class number>
template <class number2>
inline
void
MatrixArray<number>::mmult (MatrixArray<number2>& C,
                            const MatrixArray<number2>& B,
                            const bool adding) const
{
    Assert (!empty(), dealii::ExcEmptyObject());
    AssertDimension (C.n_block_cols(), B.n_block_cols());
    AssertDimension (C.n_block_rows(), this->n_block_rows());
    AssertDimension (B.n_block_rows(), this->n_block_cols());

    const unsigned int n_block_contr = this->n_block_cols();

    if(!adding) C = 0;

    for (unsigned int r = 0; r < C.n_block_rows(); ++r) {
        for(unsigned int c = 0; c < C.n_block_cols(); ++c) {
            for (unsigned int i = 0; i < n_block_contr; ++i) {

                if(block_exists(r,i) && B.block_exists(i,c)) {
                    C.create_block(r,c,false);
                    block(r,i).mmult(C.block(r,c),B.block(i,c),true);
                }
            }
        }
    }
}



template <class number>
template <class number2>
inline
void
MatrixArray<number>::Tmmult (MatrixArray<number2>& C,
                             const MatrixArray<number2>& B,
                             const bool adding) const
{
    Assert (!empty(), dealii::ExcEmptyObject());
    AssertDimension (C.n_block_cols(), B.n_block_cols());
    AssertDimension (C.n_block_rows(), this->n_block_cols());
    AssertDimension (B.n_block_rows(), this->n_block_rows());

    const unsigned int n_block_contr = this->n_block_rows();

    if(!adding) C = 0;

    for (unsigned int r = 0; r < C.n_block_rows(); ++r) {
        for(unsigned int c = 0; c < C.n_block_cols(); ++c) {
            for (unsigned int i = 0; i < n_block_contr; ++i) {

                if(block_exists(i,r) && B.block_exists(i,c)) {
                    C.create_block(r,c);
                    block(i,r).Tmmult(C.block(r,c),B.block(i,c),true);
                }
            }
        }
    }
}

template <class number>
template <class number2>
inline
void
MatrixArray<number>::mTmult (MatrixArray<number2>& C,
                             const MatrixArray<number2>& B,
                             const bool adding) const
{
    Assert (!empty(), dealii::ExcEmptyObject());
    AssertDimension (C.n_block_cols(), B.n_block_rows());
    AssertDimension (C.n_block_rows(), this->n_block_rows());
    AssertDimension (B.n_block_cols(), this->n_block_cols());

    const unsigned int n_block_contr = this->n_block_cols();

    if(!adding) C = 0;

    for (unsigned int r = 0; r < C.n_block_rows(); ++r) {
        for(unsigned int c = 0; c < C.n_block_cols(); ++c) {
            for (unsigned int i = 0; i < n_block_contr; ++i) {

                if(block_exists(r,i) && B.block_exists(c,i)) {
                    C.create_block(r,c,false);
                    block(r,i).mTmult(C.block(r,c),B.block(c,i),true);
                }
            }
        }
    }
}

template <class number>
template <class number2>
inline
void
MatrixArray<number>::TmTmult (MatrixArray<number2>& C,
                              const MatrixArray<number2>& B,
                              const bool adding) const
{
    Assert (!empty(), dealii::ExcEmptyObject());
    AssertDimension (C.n_block_cols(), B.n_block_rows());
    AssertDimension (C.n_block_rows(), this->n_block_cols());
    AssertDimension (B.n_block_cols(), this->n_block_rows());

    const unsigned int n_block_contr = this->n_block_rows();

    if(!adding) C = 0;

    for (unsigned int r = 0; r < C.n_block_rows(); ++r) {
        for(unsigned int c = 0; c < C.n_block_cols(); ++c) {
            for (unsigned int i = 0; i < n_block_contr; ++i) {

                if(block_exists(i,r) && B.block_exists(c,i)) {
                    C.create_block(r,c,false);
                    block(i,r).TmTmult(C.block(r,c),B.block(c,i),true);
                }
            }
        }
    }
}


template <typename number>
template <class StreamType>
inline
void
MatrixArray<number>::print (StreamType         &s,
                                const unsigned int  w,
                                const unsigned int  p) const
{
    Assert (!this->empty(), ExcEmptyMatrix());

    // save the state of out stream
    const unsigned int old_precision = s.precision (p);
    const unsigned int old_width = s.width (w);

    for (size_type i=0; i<this->m(); ++i)
    {
        for (size_type j=0; j<this->n(); ++j)
        {
            s.width(w);
            s.precision(p);
            s << el(i,j);
        }
        s << std::endl;
    }

    // reset output format
    s.precision(old_precision);
    s.width(old_width);
}



template <class number>
template <class Archive>
inline
void
MatrixArray<number>::serialize (Archive &ar, const unsigned int)
{
  ar & row_block_indices;
  ar & column_block_indices;
  ar & sub_objects;
}



template <class number>
bool
MatrixArray<number>::block_exists (const unsigned int row,
                                   const unsigned int column) const
{
    return !sub_objects(row,column).empty();
}



template <class number>
inline
void
MatrixArray<number>::create_block (const unsigned int row,
                                   const unsigned int column,
                                   const bool omit_zeroing_entries)
{
    AssertIndexRange (row, n_block_rows());
    AssertIndexRange (column, n_block_cols());

    if(block_exists(row,column)) return;

    sub_objects(row,column).reinit (row_block_indices.block_size(row),
                                    column_block_indices.block_size(column));

    if(!omit_zeroing_entries)
        sub_objects(row,column) = number(0);
}



template <class number>
inline
const dealii::BlockIndices&
MatrixArray<number>::get_row_block_indices () const
{
    return row_block_indices;
}



template <class number>
inline
const dealii::BlockIndices&
MatrixArray<number>::get_column_block_indices () const
{
    return column_block_indices;
}


}// namespace efi


#endif /* SRC_MYLIB_INCLUDE_EFI_LAC_MATRIX_ARRAY_H_ */
