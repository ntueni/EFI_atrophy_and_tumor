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

#ifndef SRC_MYLIB_INCLUDE_EFI_CONSTITUTIVE_CONSTITUTIVE_BASE_H_
#define SRC_MYLIB_INCLUDE_EFI_CONSTITUTIVE_CONSTITUTIVE_BASE_H_


// c++ headers
#include <vector>
#include <string>

// deal.II headers
#include <deal.II/algorithms/general_data_storage.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parameter_acceptor.h>

// efi headers
#include <efi/base/utility.h>
#include <efi/base/postprocessor.h>
#include <efi/base/factory_tools.h>
#include <efi/base/postprocessor.h>
#include <efi/factory/registry.h>
#include <efi/worker/scratch_data.h>

namespace efi
{

class DataProcessorDummy
{
public:

    template <int dim>
    void
    evaluate (ScratchData<dim> &scratch_data) const;

    /// Return which data has to be provided to compute the derived
    /// quantities. The flags returned here are the ones passed to the
    /// constructor of this class.
    dealii::UpdateFlags
    get_needed_update_flags () const;
};

// Purely virtual base class for constitutive models.
// Derived classes need to implement the functions:
// - declare_parameters
// - parse_parameters
// - evaluate
// - get_needed_update_flags
// Since the interface of this class provides
// void evaluate (ScratchData<dim> &), it can be used
// as DataProcessor in the CellWorker::reinit ()
// and BoundaryWorker::reinit () functions.
template <int dim>
class ConstitutiveBase : public dealii::ParameterAcceptor
{

public:

    EFI_REGISTER_AS_BASE;

    /// Dimension in which this object operates.
    static const unsigned int dimension = dim;

    /// Dimension of the space in which this object operates.
    static const unsigned int space_dimension = dim;

    /// Type of scalar numbers.
    using scalar_type = double;


    /// Constructor.
    /// @param[in] subsection_name The name of the subsection in the parameter
    /// file which defines the parameters for the constructed instance
    /// of this class.
    /// @param[in] unprocessed_input The unprocessed parts of the parameter
    /// file is everything between the begin of the corresponding subsection
    /// of the to-be-constructed Ogden constitutive model and its end.
    ConstitutiveBase (const std::string &subsection_name,
                      const std::string &unprocessed_input);

    /// Default destructor.
    virtual
    ~ConstitutiveBase ();

    /// This function computes the constitutive response (e.g. stresses)
    /// at the quadrature points and writes the reults to the given
    /// <tt>scratch_data<\tt> object.
    /// This function is purely virtual and hence must be implemented by
    /// any derived class which represents a concrete constitutive model.
    /// @param[in,out] scratch_data the data object from which we get
    /// the all required data (displacement, etc.) and to which we write
    /// the constitutive response of the constitutive material.
    virtual
    void
    evaluate (ScratchData<dim> &scratch_data) const = 0;

    /// Return which data has to be provided to compute the derived
    /// quantities. The flags returned here are the ones passed to the
    /// constructor of this class.
    /// This function is purely virtual and hence must be implemented by
    /// any derived class which represents a concrete constitutive model.
    virtual
    dealii::UpdateFlags
    get_needed_update_flags () const = 0;

    /// Return the data interpretation information like the tensorial
    /// orders of the fields and the names of the fields computed by
    /// evaluate_vector_field. This information is returned as vector
    /// of @p DataInterpretation objects.
    /// By default, this function returns an empty vector. However, it
    /// is meant to be overridden by derived classes.
    virtual
    std::vector<DataInterpretation>
    get_data_interpretation () const;

    /// This function computes the post-processor information we want
    /// to write to the output files e.g. for paraview. By default, this
    /// function does not compute any quantities, but is meant to be
    /// overridden by derived classes.
    /// @param[in] input_data It provides all quantities necessary for the
    /// post processing.
    /// @param[out] computed_quantities The <tt>std::vector<\tt> has
    /// as many entries as we have evaluation points, the internal <tt>
    /// dealii::Vector<double><\tt> stores the individual components of
    /// the postprocessed data fields.
    /// @param[in] additional_input_data Further input data for the cell which
    /// is about to be evaluated. This might be useful when the output
    /// cannot be computed from the <tt>input_data<\tt>, e.g. when history
    /// data is required.
    virtual
    void
    evaluate_vector_field (const dealii::DataPostprocessorInputs::Vector<dim> &input_data,
                           std::vector<dealii::Vector<double>> &computed_quantities,
                           const dealii::GeneralDataStorage* additional_input_data = nullptr) const;

    /// Compute the first derivatives of the free energy with to the
    /// principal stretches. The label principal stress might appear
    /// a bit sloppy, but the derivative of the free energy with respect
    /// to a principal stretch yields a principal-stress-like
    /// quantity. The principal stresses computed separately for the
    /// volumetric and the isochoric part of the free energy.This function
    /// is particularly useful for visco-elastic models.
    /// @param[in] lambda The principal stretches.
    /// @param[out] tangent_iso The isochoric principal Kirchoff stresses.
    /// @param[out] tangent_vol The volumetric principal Kirchoff stresses.
    virtual
    void
    compute_principal_stresses (const std::array<double,dim> &lambda,
                                dealii::Tensor<1,dim,double> &principal_stresses_iso,
                                dealii::Tensor<1,dim,double> &principal_stresses_vol) const;

    /// Compute the second derivatives of the free energy with to the
    /// principal stretches. The label principal stress tangents might
    /// appear a bit sloppy, but the derivative of the free energy with
    /// respect to a principal stretch yields a principal-stress-like
    /// quantity. The principal stresses computed separately for the
    /// volumetric and the isochoric part of the free energy. This function
    /// is particularly useful for visco-elastic models.
    /// @param[in] lambda The principal stretches.
    /// @param[out] tangent_iso The derivative of the isochoric principal
    /// Kirchoff stresses with respect to the principal stretches.
    /// @param[out] tangent_vol The derivative of the volumetric principal
    /// Kirchoff stresses with respect to the principal stretches.
    virtual
    void
    compute_principal_stress_tangents (const std::array<double,dim> &lambda,
                                       dealii::SymmetricTensor<2,dim,double> &principal_stress_tangent_iso,
                                       dealii::SymmetricTensor<2,dim,double> &principal_stress_tangent_vol) const;

protected:

    std::string section_path_str;
};



//----------------------- INLINE AND TEMPLATE FUNCTIONS ----------------------//



template <int dim>
inline
void
DataProcessorDummy::
evaluate (ScratchData<dim> &) const
{
    // do nothing
}



inline
dealii::UpdateFlags
DataProcessorDummy::
get_needed_update_flags () const
{
    return dealii::update_default;
}



template<int dim>
inline
ConstitutiveBase<dim>::
ConstitutiveBase (const std::string &subsection_name,
                  const std::string &)
:
    dealii::ParameterAcceptor (subsection_name),
    section_path_str (get_section_path_str(this->get_section_path()))
{ }



template<int dim>
inline
ConstitutiveBase<dim>::
~ConstitutiveBase ()
{ }



template<int dim>
inline
std::vector<DataInterpretation>
ConstitutiveBase<dim>::
get_data_interpretation () const
{
   return std::vector<DataInterpretation>();
}



template<int dim>
inline
void
ConstitutiveBase<dim>::
evaluate_vector_field (const dealii::DataPostprocessorInputs::Vector<dim> &,
                       std::vector<dealii::Vector<double>> &,
                       const dealii::GeneralDataStorage*) const
{
    // do nothing
}



template<int dim>
inline
void
ConstitutiveBase<dim>::
compute_principal_stresses (const std::array<double,dim> &,
                            dealii::Tensor<1,dim,double> &,
                            dealii::Tensor<1,dim,double> &) const
{
    Assert(false,dealii::ExcNotImplemented());
}



template<int dim>
inline
void
ConstitutiveBase<dim>::
compute_principal_stress_tangents (const std::array<double,dim> &,
                                   dealii::SymmetricTensor<2,dim,double> &,
                                   dealii::SymmetricTensor<2,dim,double> &) const
{
    Assert(false,dealii::ExcNotImplemented());
}

}//namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_CONSTITUTIVE_CONSTITUTIVE_BASE_H_ */
