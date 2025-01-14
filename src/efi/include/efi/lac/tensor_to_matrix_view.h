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

#ifndef SRC_MYLIB_INCLUDE_EFI_LAC_TENSOR_TO_MATRIX_VIEW_H_
#define SRC_MYLIB_INCLUDE_EFI_LAC_TENSOR_TO_MATRIX_VIEW_H_

// deal.II headers
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_accessors.h>


namespace efi {


namespace efi_internal {

// Helper class to extract elements form a Tensor using a matrix like
// interface, i.e. for given row and column index the corresponding
// tensor element is returned.
template <int row_rank, int col_rank, class Tensor> struct ExtractHelper;

template <int row_rank, int col_rank, int rank, int dim, class number>
struct ExtractHelper<row_rank, col_rank, dealii::Tensor<rank,dim,number>> {

    static
    number
    extract_matrix_element (const dealii::Tensor<rank,dim,number> &tensor, const unsigned int i, const unsigned int j)
    {
        AssertIndexRange (i,(dealii::Tensor<row_rank,dim>::n_independent_components));
        AssertIndexRange (j,(dealii::Tensor<col_rank,dim>::n_independent_components));

        return dealii::TensorAccessors::extract<row_rank>(
                       tensor, dealii::Tensor<row_rank, dim>::unrolled_to_component_indices(i))[
                               dealii::Tensor<col_rank, dim>::unrolled_to_component_indices(j)];
    }
};



template <int col_rank, int rank, int dim, class number>
struct ExtractHelper<0, col_rank, dealii::Tensor<rank,dim,number>> {

    static
    number
    extract_matrix_element (const dealii::Tensor<rank,dim,number> &tensor, const unsigned int i, const unsigned int j)
    {
        AssertIndexRange (i,1);
        AssertIndexRange (j,(dealii::Tensor<col_rank,dim>::n_independent_components));

        return tensor[dealii::Tensor<col_rank, dim>::unrolled_to_component_indices(j)];
        (void)i;
    }
};



template <int row_rank, int rank, int dim, class number>
struct ExtractHelper<row_rank, 0, dealii::Tensor<rank,dim,number>> {

    static
    number
    extract_matrix_element (const dealii::Tensor<rank,dim,number> &tensor, const unsigned int i, const unsigned int j)
    {
        AssertIndexRange (i,(dealii::Tensor<row_rank,dim>::n_independent_components));
        AssertIndexRange (j,1);

        return tensor[dealii::Tensor<row_rank, dim>::unrolled_to_component_indices(i)];
        (void)j;
    }
};



template <int rank, int dim, class number>
struct ExtractHelper<0, 0, dealii::Tensor<rank,dim,number>> {

    static
    number
    extract_matrix_element (const dealii::Tensor<rank,dim,number> &tensor, const unsigned int i, const unsigned int j)
    {
        AssertIndexRange (i,1);
        AssertIndexRange (j,1);

        return number(tensor);
        (void)i;
        (void)j;
    }
};

template <int row_rank, int col_rank, int rank, int dim, class number>
number
extract_matrix_element (const dealii::Tensor<rank,dim,number> &tensor, const unsigned int i, const unsigned int j)
{
    return ExtractHelper<row_rank,col_rank,dealii::Tensor<rank,dim,number>>::extract_matrix_element(tensor,i,j);
}

}

// Create a view to a tensor with a matrix interface.
//
// The dimensions of the matrix are m x n where m is the number of components of an object
// Tensor<row_rank,dim> and n is the number of components of an object Tensor<col_rank,dim>,
// i.e. m x n = dim**row_rank x dim**col_rank.
// Consider the tensor<rank,dim,number> as tensor<row_rank,dim,tensor<col_rank,dim,number>>.
// Now one can view the matrix as tensor<row_rank,...> unrolled to a vector whose elements
// tensor<col_rank> itself are unrolled to a vector, i.e. a vector of vectors which is the
// 2D array we need.
//
// Example:
// Tensor<3,3> A;
// TensorToMatrixView<2,1,3,3> view(A);
// view.print(std::cout)
//
// Output for A_ijk:
//
// A111 A112 A113
// A121 A122 A123
// A131 A132 A133
// A211 A212 A213
// A221 A222 A223
// A231 A232 A233
// A311 A312 A313
// A321 A322 A323
// A331 A332 A333
//
// The row tensor indices are the first 'row_rank' indices the next 'col_rank' indices
// are the column tensor indices, i.e. here A_(ik)(k)
//                          ~~~~~~~~~~~~~~~~~~~^   ^~~~~~~~~~~~~~~~~~~~~~
//                          row tensor indices      colmun tensor indices
//
// row tensor indices (i,j, since row_rank = 2)    11, 12, 13, 21, 22, 23, 31, 32, 33
// matrix row index                                1,  2,  3,  4,  5,  6,  7,  8,  9
// col tensor indices (k, since col_rank = 1)      1,  2,  3
// matrix col index                                1,  2,  3
//
// As usual the rightmost index of the row and column tensor indices varies fastest.
template <int row_rank, int col_rank, int rank, int dim, class number = double>
class TensorToMatrixView {

public:

    static_assert (row_rank + col_rank == rank, "Rank mismatch.");

    TensorToMatrixView (const dealii::Tensor<rank,dim,number> &tensor);

    typedef unsigned int  size_type;

    inline size_type m () const;
    inline size_type n () const;

    inline number operator() (const size_type i, const size_type j) const;
    inline number el (const size_type i, const size_type j) const;


    template <class OutVector, class InVector>
    inline
    void vmult (OutVector      &dst,
                const InVector &src) const;

    template <class OutVector, class InVector>
    inline
    void Tvmult (OutVector      &dst,
                 const InVector &src) const;

    template <class OutVector, class InVector>
    inline
    void vmult_add (OutVector      &dst,
                    const InVector &src) const;

    template <class OutVector, class InVector>
    inline
    void Tvmult_add (OutVector      &dst,
                     const InVector &src) const;

    //TODO iterators, begin, end, etc. Without iterators some
    //     of the dealii function do not work with this view class.

    // print the matrix as a whole
    template <typename StreamType>
    inline
    void print (StreamType         &s,
                const unsigned int  width = 5,
                const unsigned int  precision = 2) const;

private:

    static const size_type n_rows = dealii::Tensor<row_rank,dim,number>::n_independent_components;
    static const size_type n_cols = dealii::Tensor<col_rank,dim,number>::n_independent_components;

    const dealii::Tensor<rank,dim,number> &tensor;
};



template <int row_rank, int col_rank, int rank, int dim, class number>
TensorToMatrixView<row_rank,col_rank,rank,dim,number>::
TensorToMatrixView (const dealii::Tensor<rank,dim,number> &tensor)
    :
    tensor(tensor)
{ }



template <int row_rank, int col_rank, int rank, int dim, class number>
unsigned int
TensorToMatrixView<row_rank,col_rank,rank,dim,number>::m () const
{
    return n_rows;
}

template <int row_rank, int col_rank, int rank, int dim, class number>
unsigned int
TensorToMatrixView<row_rank,col_rank,rank,dim,number>::n () const
{
    return n_cols;
}



template <int row_rank, int col_rank, int rank, int dim, class number>
number
TensorToMatrixView<row_rank,col_rank,rank,dim,number>::
operator() (const size_type i, const size_type j) const
{
    AssertIndexRange (i,n_rows);
    AssertIndexRange (j,n_cols);

    return  efi_internal::extract_matrix_element<row_rank,col_rank>(tensor,i,j);
}



template <int row_rank, int col_rank, int rank, int dim, class number>
number
TensorToMatrixView<row_rank,col_rank,rank,dim,number>::
el (const size_type i, const size_type j) const
{
    AssertIndexRange (i,n_rows);
    AssertIndexRange (j,n_cols);

    return  efi_internal::extract_matrix_element<row_rank,col_rank>(tensor,i,j);
}



template <int row_rank, int col_rank, int rank, int dim, class number>
template <class OutVector, class InVector>
void
TensorToMatrixView<row_rank,col_rank,rank,dim,number>::
vmult (OutVector      &dst,
       const InVector &src) const {
    AssertThrow(false,dealii::ExcMessage("Its on my to do list, "
                                         "sorry for being lazy ;)"));
}



template <int row_rank, int col_rank, int rank, int dim, class number>
template <class OutVector, class InVector>
void
TensorToMatrixView<row_rank,col_rank,rank,dim,number>::
Tvmult (OutVector      &dst,
        const InVector &src) const {
    AssertThrow(false,dealii::ExcMessage("Its on my to do list, "
                                         "sorry for being lazy ;)"));
}



template <int row_rank, int col_rank, int rank, int dim, class number>
template <class OutVector, class InVector>
void
TensorToMatrixView<row_rank,col_rank,rank,dim,number>::
vmult_add (OutVector      &dst,
                const InVector &src) const {
    AssertThrow(false,dealii::ExcMessage("Its on my to do list, "
                                         "sorry for being lazy ;)"));
}



template <int row_rank, int col_rank, int rank, int dim, class number>
template <class OutVector, class InVector>
void
TensorToMatrixView<row_rank,col_rank,rank,dim,number>::
Tvmult_add (OutVector      &dst,
                 const InVector &src) const {
    AssertThrow(false,dealii::ExcMessage("Its on my to do list, "
                                         "sorry for being lazy ;)"));
}



template <int row_rank, int col_rank, int rank, int dim, class number>
template <class StreamType>
void
TensorToMatrixView<row_rank,col_rank,rank,dim,number>::
print (StreamType         &s,
       const unsigned int  w,
       const unsigned int  p) const
{
    // save the state of out stream
    const unsigned int old_precision = s.precision (p);
    const unsigned int old_width = s.width (w);

    for (size_type i=0; i<this->m(); ++i)
    {
        for (size_type j=0; j<this->n(); ++j)
        {
            s.width(w);
            s.precision(p);
            s << this->operator()(i,j);
        }
        s << std::endl;
    }
    // reset output format
    s.precision(old_precision);
    s.width(old_width);
}



// For given row_rank and col_rank a view to a tensor
// with a matrix interface is created. See @TensorToMatrixView.
template <int row_rank, int col_rank, int rank, int dim, class number>
inline
TensorToMatrixView<row_rank,col_rank,rank,dim,number>
get_matrix_view (const dealii::Tensor<rank,dim,number> &tensor)
{
    return TensorToMatrixView<row_rank,col_rank,rank,dim,number>(tensor);
}

}

#endif /* SRC_MYLIB_INCLUDE_EFI_LAC_TENSOR_TO_MATRIX_VIEW_H_ */
