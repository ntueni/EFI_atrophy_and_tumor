/*
 * Copyright (C) 2019 - 2020 by the emerging fields initiative 'Novel Biopolymer
 * Hydrogels for Understanding Complex Soft Tissue Biomechanics' of the FAU
 *
 * This file is part of the EFI library.
 *
 * The efi library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * Author: Stefan Kaessmair
 */

#ifndef SRC_MYLIB_INCLUDE_EFI_BASE_CLONEABLE_FUNCTION_H_
#define SRC_MYLIB_INCLUDE_EFI_BASE_CLONEABLE_FUNCTION_H_

// stl headers
#include <vector>
#include <random>

// deal.II headers
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

// efi headers
#include <efi/base/gnuplot_stream.h>


namespace efi {



template <int dim>
class CloneableFunction : public dealii::Function<dim>
{
public:

    CloneableFunction (const unsigned int n_components = 1,
                       const double       initial_time = 0.0);

    virtual CloneableFunction<dim>* clone () const = 0;

    virtual void print (GnuplotStream& stream,
                        const unsigned int component = 0);

    virtual ~CloneableFunction () = default;

    CloneableFunction (const CloneableFunction<dim>&) = default;
    CloneableFunction<dim>& operator= (const CloneableFunction<dim>&) = default;
};



template <int dim>
class FunctionWrapper : public CloneableFunction<dim>
{
public:

    /// Constructor
    FunctionWrapper (const std::function<double(const dealii::Point<dim> &,
                                                const unsigned int,
                                                const double)>  &function_object,
                     const unsigned int n_components,
                     const double       initial_time = 0.0);

    /// Destructor.
    virtual
    ~FunctionWrapper () = default;

    /// Assignment operator.
    FunctionWrapper<dim>&
    operator= (const FunctionWrapper<dim>&) = default;

    /// Return the value of the function at the given point. Unless
    /// there is only one component (i.e. the function is scalar),
    /// you should state the component you want to have evaluated;
    /// it defaults to zero, i.e. the first component.
    /// @param[in] p Point at which the function is evaluated.
    /// @param[in] component Component of the function which is
    /// evaluated.
    virtual
    double
    value (const dealii::Point<dim> &p,
           const unsigned int component = 0) const override;

    /// Returns a clone.
    virtual
    FunctionWrapper<dim>*
    clone () const override;


private :

    /// The wrapped function.
    std::function<double(const dealii::Point<dim> &, const unsigned int, const double)>
        wrapped_function;
};



/// Depending on the input data, this function is a linear or
/// quadratic function. The coefficients are determined by a
/// least square fit of the input data.
template <int dim>
class FittedFunction : public CloneableFunction<dim>
{
public:

    /// Constructor.
    /// Note that <tt>x</tt> and <tt>y</tt> must have the same size.
    /// @param[in] y Scalar valued data to be fitted.
    /// @param[in] x Evaluation points of the provided data.
    template<int rank, class Number>
    FittedFunction (const std::vector<Number> &y,
                    const std::vector<dealii::Point<dim>> &x,
                    const double initial_time = 0);

    /// Constructor.
    /// Note that <tt>x</tt> and <tt>y</tt> must have the same size.
    /// @param[in] y Tensor-valued data to be fitted.
    /// @param[in] x Evaluation points of the provided data.
    template<int rank, class Number>
    FittedFunction (const std::vector<dealii::Tensor<rank,dim,Number>> &y,
                    const std::vector<dealii::Point<dim>> &x,
                    const double initial_time = 0);

    /// Constructor.
    /// As the constructor above, just for symmetric tensors. Internally,
    /// the SymmetricTensor is converted into a regular tensor, i.e.
    /// the function has <tt>Tensor<rank,dim,Number>::n_independent_components
    /// </tt> components.
    template<int rank, class Number>
    FittedFunction (const std::vector<dealii::SymmetricTensor<rank,dim,Number>> &y,
                    const std::vector<dealii::Point<dim>> &x,
                    const double initial_time = 0);

    /// Default destructor.
    virtual
    ~FittedFunction () = default;

    /// Assignment operator.
    FittedFunction<dim>&
    operator= (const FittedFunction<dim>&) = default;

    /// Return the value of the function at the given point. Unless
    /// there is only one component (i.e. the function is scalar),
    /// you should state the component you want to have evaluated;
    /// it defaults to zero, i.e. the first component.
    /// @param[in] p Point at which the function is evaluated.
    /// @param[in] component Component of the function which is
    /// evaluated.
    virtual
    double
    value (const dealii::Point<dim> &p,
           const unsigned int component = 0) const override;

    /// Return a clone of this function.
    virtual
    FittedFunction<dim>*
    clone () const override;

private :

    /// Constructor.
    /// The data points (x[i],y[i]) are least square fitted to
    /// a linear or a quadratic function. For less than six data
    /// points, a linear function is fitted, otherwise
    /// a quadratic function is fitted.
    /// @param[in] y Array of the vector-valued data to be fitted.
    /// Access a scalar value via y[component][evaluation point].
    /// @param[in] x Vector of the evaluation points of the vector
    /// valued data.
    FittedFunction (const std::vector<std::vector<double>> &y,
                    const std::vector<dealii::Point<dim>>  &x,
                    const double initial_time = 0);

    std::function<double(const dealii::Point<dim> &, const unsigned int, const double)> fitted_function;
};


// create a random real number [a,b) with uniform probability
// density function P(i|a,b) = 1/(b-a).
template <int dim>
class UniformlyDistRandomNumber : public CloneableFunction<dim>
{
public:

    UniformlyDistRandomNumber (const double a,
                               const double b,
                               const unsigned int n_components = 1);

    virtual double value (const dealii::Point<dim> &,
                          const unsigned int       = 0) const;

    virtual UniformlyDistRandomNumber<dim>* clone () const override {
        return new UniformlyDistRandomNumber(distribution.a(), distribution.b(), this->n_components);
    };

    virtual ~UniformlyDistRandomNumber () = default;

private:

    // seed of the random number engine
    std::random_device rd;
    // Uniform random number generator.
    // We have to use a pointer, since a non-constness of the generator
    // is required to create random numbers.
    mutable std::mt19937 gen;
    // Transform the random unsigned int generated by gen into a double.
    // As above, the distribution must not be constant in order
    // to get random numbers.
    mutable std::uniform_real_distribution<double> distribution;
};



//------------------- INLINE AND TEMPLATE FUNCTIONS -------------------//



template<int dim>
template<int rank, class Number>
inline
FittedFunction<dim>::
FittedFunction (const std::vector<Number> &y,
                const std::vector<dealii::Point<dim>> &x,
                const double initial_time)
: CloneableFunction<dim> (1, initial_time)
{
    using namespace dealii;

    AssertDimension(x.size(),y.size())

    unsigned int n_data_components = 1;

    std::vector<std::vector<double>> y_vectorized (1);

    y[0].resize(x.size());
    for (unsigned int q=0; q< x.size(); ++q)
        y_vectorized[0][q] = y[q];

    *this = FittedFunction<dim>(y_vectorized,x,initial_time);
}



template<int dim>
template<int rank, class Number>
inline
FittedFunction<dim>::
FittedFunction (const std::vector<dealii::Tensor<rank,dim,Number>> &y,
                const std::vector<dealii::Point<dim>> &x,
                const double initial_time)
: CloneableFunction<dim> (dealii::Tensor<rank,dim,Number>::n_independent_components, initial_time)
{
    using namespace dealii;

    AssertDimension(x.size(),y.size())

    unsigned int n_data_components =
            Tensor<rank,dim,Number>::n_independent_components;

    std::vector<std::vector<double>> y_vectorized (n_data_components);
    for (unsigned int c = 0; c < n_data_components; ++c)
        y_vectorized[c].resize (x.size(), 0);

    Vector<double> tmp(n_data_components);

    for (unsigned int q=0, c=0; q< x.size(); ++q, c=0)
    {
        y[q].unroll(tmp);
        for (unsigned int i=0; i < tmp.size(); ++i, ++c)
            y_vectorized[c][q] = tmp[i];
    }

    *this = FittedFunction<dim>(y_vectorized,x,initial_time);
}



template<int dim>
template<int rank, class Number>
inline
FittedFunction<dim>::
FittedFunction (const std::vector<dealii::SymmetricTensor<rank,dim,Number>> &y,
                const std::vector<dealii::Point<dim>>      &x,
                const double initial_time)
: CloneableFunction<dim> (dealii::Tensor<rank,dim,Number>::n_independent_components, initial_time)
{
    using namespace dealii;

    AssertDimension(x.size(),y.size())

    unsigned int n_data_components =
            Tensor<rank,dim,Number>::n_independent_components;

    std::vector<std::vector<double>> y_vectorized (n_data_components);
    for (unsigned int c = 0; c < n_data_components; ++c)
        y_vectorized[c].resize (x.size(), 0);

    Vector<double> tmp(n_data_components);

    for (unsigned int q=0, c=0; q< x.size(); ++q, c=0)
    {
        Tensor<rank,dim,Number> yq;
        yq = y[q];
        yq.unroll(tmp);
        for (unsigned int i=0; i < tmp.size(); ++i, ++c)
            y_vectorized[c][q] = tmp[i];
    }

    *this = FittedFunction<dim>(y_vectorized,x,initial_time);
}


}//close efi

#endif /* SRC_MYLIB_INCLUDE_EFI_BASE_CLONEABLE_FUNCTION_H_ */
