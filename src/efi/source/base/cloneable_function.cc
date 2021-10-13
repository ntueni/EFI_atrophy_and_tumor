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
#include <efi/base/cloneable_function.h>


using namespace dealii;

namespace efi {

template <int dim>
CloneableFunction<dim>::CloneableFunction (const unsigned int n_components,
                                           const double       initial_time)
    :
    Function<dim>(n_components,initial_time)
{ }



template <int dim>
void
CloneableFunction<dim>::print (GnuplotStream& stream,
                               const unsigned int component)

{
    unsigned int n_spatial_steps = 21;
    double dx = 1.0/(double)(n_spatial_steps-1);

    Point<dim> x;

    stream << "set cbtics\n"
           << "set term wxt 1 noraise size 800, 800\n"
           << "set autoscale fix\n"
           << "set hidden3d front\n"
           << "set pm3d interpolate 5,5\n"
           << "splot '-' using 1:2:3 with pm3d title 't=" << Utilities::to_string(this->get_time()) << "'\n";

    for(unsigned int i = 0; i < n_spatial_steps; ++i) {
        for (unsigned int j = 0; j < n_spatial_steps; ++j) {

            stream << Utilities::to_string(x[0]) << " "
                   << Utilities::to_string(x[1]) << " "
                   << Utilities::to_string(this->value(x,component)) <<"\n";

            x[0] += dx;
        }
        stream << "\n";
        x[0] = 0;
        x[1] += dx;
    }
    stream << "\ne\n";
}


template<int dim>
FunctionWrapper<dim>::
FunctionWrapper (const std::function<double(const dealii::Point<dim> &,
                                            const unsigned int,
                                            const double)>  &function_object,
                 const unsigned int n_components,
                 const double       initial_time)
: CloneableFunction<dim> (n_components, initial_time),
  wrapped_function (function_object)
{ }



template<int dim>
double
FunctionWrapper<dim>::
value (const dealii::Point<dim> &p,
       const unsigned int component) const
{
    return this->wrapped_function (p,component,this->get_time());
}



template<int dim>
FunctionWrapper<dim>*
FunctionWrapper<dim>::
clone () const
{
    return new FunctionWrapper<dim> (this->wrapped_function,
                                     this->n_components,
                                     this->get_time());
}



template<int dim>
FittedFunction<dim>::
FittedFunction (const std::vector<std::vector<double>> &y,
                const std::vector<dealii::Point<dim>>  &x,
                const double initial_time)
: CloneableFunction<dim> (y.size(), initial_time)
{
    unsigned int n_coeffs = SymmetricTensor<2,dim>::n_independent_components+dim+1;

    bool quadratic_least_squares = (x.size() >= n_coeffs);
    bool linear_least_squares    = (x.size() >= dim+1);

    if (!quadratic_least_squares)
        n_coeffs = dim+1;
    if (!linear_least_squares)
        n_coeffs = 1;

    Assert (x.size() >= n_coeffs, ExcLowerRange (x.size(), n_coeffs));

    std::vector<Vector<double>> monomials (x.size());

    FullMatrix<double> A (n_coeffs);
    FullMatrix<double> Ainv (n_coeffs);
    Vector<double> src (n_coeffs);
    std::vector<Vector<double>> coefficients (y.size());

    std::function<Vector<double>(const Point<dim>&)> compute_monomials =
            [n_coeffs,quadratic_least_squares,linear_least_squares]
            (const Point<dim> & p)->Vector<double>
    {
        Vector<double> monomials (n_coeffs);

        unsigned int n=0;

        if (quadratic_least_squares)
        {
            for (unsigned int r=0; r<dim; ++r, ++n)
                monomials[n] = p[r]*p[r];
            for (unsigned int r=0; r<dim; ++r)
                for (unsigned int s=r+1; s<dim; ++s, ++n)
                    monomials[n] = p[r]*p[s];
        }

        if (linear_least_squares || quadratic_least_squares)
        {
            for (unsigned int r=0; r<dim; ++r, ++n)
                monomials[n] = p[r];
        }
        monomials[n] = 1.;
        return monomials;
    };

    A = 0;
    for (unsigned int i = 0; i < x.size(); ++i)
    {
        monomials[i] = compute_monomials (x[i]);
        for (unsigned int r=0; r<n_coeffs; ++r)
        {
            A(r,r) += monomials[i][r]*monomials[i][r];
            for (unsigned int s=r+1; s<n_coeffs; ++s)
                A(r,s) += monomials[i][r]*monomials[i][s];
        }
    }
    // Copy symmetric entries.
    for (unsigned int r=0; r<n_coeffs; ++r)
        for (unsigned int s=0; s<r; ++s)
            A(r,s) = A(s,r);

    // Invert the monomial matrix.
    Ainv.invert(A);

    // loop over all components.
    for (unsigned int c = 0; c < y.size(); ++c)
    {
        src = 0;
        Assert (y[c].size() > dim, ExcLowerRange (y[c].size(), dim+1));
        for (unsigned int i = 0; i < x.size(); ++i)
            src.add (y[c][i], monomials[i]);

        coefficients[c].reinit(n_coeffs,true);

        // Compute the coefficients.
        Ainv.vmult(coefficients[c],src);
    }

    this->fitted_function =
        [coefficients,compute_monomials](const Point<dim>& p,
                                         const unsigned int c,
                                         const double) -> double
        {
            AssertIndexRange (c,coefficients.size());
            return coefficients[c]*compute_monomials(p);
        };
}



template<int dim>
double
FittedFunction<dim>::
value (const dealii::Point<dim> &p,
       const unsigned int component) const
{
    return this->fitted_function (p, component, this->get_time());
}



template<int dim>
FittedFunction<dim>*
FittedFunction<dim>::
clone () const
{
    return nullptr;
}



template <int dim>
UniformlyDistRandomNumber<dim>::UniformlyDistRandomNumber (const double a,
                                                           const double b,
                                                           const unsigned int n_components)
    :
    CloneableFunction<dim>(n_components,0),
    rd(),
    gen(rd()),
    distribution(a,b)
{
    srand (0);
}



template <int dim>
double
UniformlyDistRandomNumber<dim>::value (const Point<dim> &,
                               const unsigned int         ) const {
    return distribution(gen);
}


template class CloneableFunction<1>;
template class CloneableFunction<2>;
template class CloneableFunction<3>;

template class FunctionWrapper<1>;
template class FunctionWrapper<2>;
template class FunctionWrapper<3>;

template class FittedFunction<1>;
template class FittedFunction<2>;
template class FittedFunction<3>;

template class UniformlyDistRandomNumber<1>;
template class UniformlyDistRandomNumber<2>;
template class UniformlyDistRandomNumber<3>;

}//close efi


