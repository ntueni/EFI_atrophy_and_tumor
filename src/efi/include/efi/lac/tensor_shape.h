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

#ifndef SRC_MYLIB_INCLUDE_EFI_LAC_TENSOR_SHAPE_H_
#define SRC_MYLIB_INCLUDE_EFI_LAC_TENSOR_SHAPE_H_

// deal.II headers
#include <deal.II/base/tensor.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/table_indices.h>


namespace efi {


// This class takes an array and provides the
// same access to the data members as a tensor.
template <int rank, int dim, class number = double, int max_rank = rank>
class TensorShape;


template <int dim, class number, int max_rank>
class TensorShape<0,dim,number,max_rank> {

public:

    typedef number value_type;

    TensorShape (number *ptr, unsigned int size = dealii::Tensor<max_rank,dim>::n_independent_components)
        :
        values(ptr,size)
    { }

    TensorShape (dealii::Vector<typename std::remove_cv<number>::type> &vector)
        :
        values(vector.begin(),vector.size())
    { }

    TensorShape (const dealii::Vector<typename std::remove_cv<number>::type> &vector)
        :
        values(vector.begin(),vector.size())
    { }

    TensorShape (dealii::Table<2,typename std::remove_cv<number>::type> &table, const unsigned int row)
        :
        values(&table[row][0], table.size()[1])
    { }

    TensorShape (const dealii::Table<2,typename std::remove_cv<number>::type> &table, const unsigned int row)
        :
        values(&table[row][0], table.size()[1])
    { }

    TensorShape<0,dim,number,max_rank>&
    operator= (const dealii::Tensor<0,dim, typename std::decay<number>::type> &tensor)
    {
        AssertIndexRange ((dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)),values.size());
        values[dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)] = tensor;
        return *this;
    }

    TensorShape<0,dim,number,max_rank>&
    operator+= (const dealii::Tensor<0,dim, typename std::decay<number>::type> &tensor)
    {
        AssertIndexRange ((dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)),values.size());
        values[dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)] += tensor;
        return *this;
    }

    TensorShape<0,dim,number,max_rank>&
    operator-= (const dealii::Tensor<0,dim, typename std::decay<number>::type> &tensor)
    {
        AssertIndexRange ((dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)),values.size());
        values[dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)] -= tensor;
        return *this;
    }


    TensorShape<0,dim,number,max_rank>&
    operator= (const number d)
    {
        AssertIndexRange ((dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)),values.size());
        values[dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)] = d;
        return *this;
    }

    TensorShape<0,dim,number,max_rank>&
    operator+= (const number d)
    {
        AssertIndexRange ((dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)),values.size());
        values[dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)] += d;
        return *this;
    }

    TensorShape<0,dim,number,max_rank>&
    operator-= (const number d)
    {
        AssertIndexRange ((dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)),values.size());
        values[dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)] -= d;
        return *this;
    }

    operator value_type& () {

        AssertIndexRange ((dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)),values.size());
        return values[dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)];
    }

    operator const value_type& () const {

        AssertIndexRange ((dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)),values.size());
        return values[dealii::Tensor<max_rank,dim>::component_to_unrolled_index(indices)];
    }

protected:

    // make it mutable, otherwise we cannot have
    // a const version of operator[] in the derived
    // classes and an conversion operator in this
    // class.
    mutable dealii::TableIndices<max_rank> indices;

    dealii::ArrayView<number> values;
};



template <int rank, int dim, class number, int max_rank>
class TensorShape : public TensorShape<rank-1,dim,number,max_rank> {

public:

    typedef TensorShape<rank-1,dim,number,max_rank> value_type;

    TensorShape (number *ptr, unsigned int size = dealii::Tensor<max_rank,dim>::n_independent_components)
        :
        TensorShape<rank-1,dim,number,max_rank>(ptr,size)
    { }

    TensorShape (dealii::Vector<typename std::remove_cv<number>::type> &vector)
        :
        TensorShape<rank-1,dim,number,max_rank>(vector)
    { }

    TensorShape (const dealii::Vector<typename std::remove_cv<number>::type> &vector)
        :
        TensorShape<rank-1,dim,number,max_rank>(vector)
    { }

    TensorShape (dealii::Table<2,typename std::remove_cv<number>::type> &table, const unsigned int row)
        :
        TensorShape<rank-1,dim,number,max_rank>(table,row)
    { }

    TensorShape (const dealii::Table<2,typename std::remove_cv<number>::type> &table, const unsigned int row)
        :
        TensorShape<rank-1,dim,number,max_rank>(table,row)
    { }


    TensorShape<rank,dim,number,max_rank>&
    operator= (const dealii::Tensor<rank,dim, typename std::remove_cv<number>::type> &tensor)
    {
        for(unsigned int i=0; i<dim; ++i)
            (*this)[i] = tensor[i];
        return *this;
    }

    TensorShape<rank,dim,number,max_rank>&
    operator= (const number d)
    {
        for(unsigned int i=0; i<dim; ++i)
            (*this)[i] = d;
        return *this;
    }

    TensorShape<rank,dim,number,max_rank>&
    operator+= (const dealii::Tensor<rank,dim, typename std::remove_cv<number>::type> &tensor)
    {
        for(unsigned int i=0; i<dim; ++i)
            (*this)[i] += tensor[i];
        return *this;
    }

    TensorShape<rank,dim,number,max_rank>&
    operator-= (const dealii::Tensor<rank,dim, typename std::remove_cv<number>::type> &tensor)
    {
        for(unsigned int i=0; i<dim; ++i)
            (*this)[i] -= tensor[i];
        return *this;
    }

    value_type& operator[] (const unsigned int i) {
        AssertIndexRange (i,dim);
        this->indices[max_rank-rank] = i;
        return *this;
    }

    const value_type& operator[] (const unsigned int i) const {
        AssertIndexRange (i,dim);
        this->indices[max_rank-rank] = i;
        return *this;
    }


//    operator const dealii::Tensor<rank,dim, typename std::remove_cv<number>::type> () const
//    {
//        dealii::Tensor<rank,dim, typename std::remove_cv<number>::type> tensor;
//        for(unsigned int i=0; i<dim; ++i)
//            tensor[i] = (*this)[i];
//        return tensor;
//    }
};


template <int dim, class number>
class TensorShape<0,dim,number,0> {

public:

    typedef number value_type;

    TensorShape (number *ptr, unsigned int
#ifdef DEBUG
            size
#endif
            = 1)
        :
        value(*ptr)
    {
        AssertDimension(size,1);
    }

    TensorShape (dealii::Vector<typename std::remove_cv<number>::type> &vector)
        :
        value(vector[0])
    {
        AssertDimension(vector.size(),1);
    }

    TensorShape (const dealii::Vector<typename std::remove_cv<number>::type> &vector)
        :
        value(vector[0])
    {
        AssertDimension(vector.size(),1);
    }

    TensorShape (dealii::Table<2,typename std::remove_cv<number>::type> &table, const unsigned int row)
        :
        value(table[row][0])
    {
        AssertDimension(table.size()[1],1);
    }

    TensorShape (const dealii::Table<2,typename std::remove_cv<number>::type> &table, const unsigned int row)
        :
        value(table[row][0])
    {
        AssertDimension(table.size()[1],1);
    }


    TensorShape<0,dim,number,0>&
    operator= (const dealii::Tensor<0,dim, typename std::decay<number>::type> &tensor)
    {
        value = tensor;
        return *this;
    }

    TensorShape<0,dim,number,0>&
    operator+= (const dealii::Tensor<0,dim, typename std::decay<number>::type> &tensor)
    {
        value += tensor;
        return *this;
    }

    TensorShape<0,dim,number,0>&
    operator-= (const dealii::Tensor<0,dim, typename std::decay<number>::type> &tensor)
    {
        value -= tensor;
        return *this;
    }


    TensorShape<0,dim,number,0>&
    operator= (const number d)
    {
        value = d;
        return *this;
    }

    TensorShape<0,dim,number,0>&
    operator+= (const number d)
    {
        value += d;
        return *this;
    }

    TensorShape<0,dim,number,0>&
    operator-= (const number d)
    {
        value -= d;
        return *this;
    }


    operator value_type& () {
        return value;
    }


    operator const value_type& () const {
        return value;
    }

protected:

    number &value;
};


}//close efi



#endif /* SRC_MYLIB_INCLUDE_EFI_LAC_TENSOR_SHAPE_H_ */
