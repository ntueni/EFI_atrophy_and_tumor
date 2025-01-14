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

#ifndef SRC_MYLIB_INCLUDE_EFI_CONSTITUTIVE_NEO_HOOKE_H_
#define SRC_MYLIB_INCLUDE_EFI_CONSTITUTIVE_NEO_HOOKE_H_

// stl headers
#include <string>
#include <vector>
#include <iostream>

// deal.II headers
#include <deal.II/base/parameter_handler.h>

// efi headers
#include <efi/constitutive/constitutive_base.h>
#include <efi/base/factory_tools.h>
#include <efi/base/logstream.h>
#include <efi/base/automatic_differentiation.h>
#include <efi/base/postprocessor.h>
#include <efi/worker/scratch_data.h>


namespace efi
{


/// Neo-Hookean material model. For given deformations, it computes the
/// Piola stresses and its tangent with respect to the deformation
/// gradient.
/// @author Stefan Kaessmair
template <int dim>
class NeoHooke : public ConstitutiveBase<dim>
{
public:

    using scalar_type = typename ConstitutiveBase<dim>::scalar_type;

    /// Constructor.
    /// @param[in] subsection_name The name of the subsection in the parameter
    /// file which defines the parameters of the present class instance.
    /// @param[in] unprocessed_input The unprocessed parts of the parameter
    /// file is everything between the begin of the corresponding subsection
    /// of the to-be-constructed neo-Hooke constitutive model and its end.
    NeoHooke (const std::string &subsection_name,
              const std::string &unprocessed_input);

    /// Default destructor.
    ~NeoHooke () = default;

    /// Declare parameters to the given parameter handler.
    /// @param[out] prm The parameter handler for which we want to declare
    /// the parameters.
    void
    declare_parameters (dealii::ParameterHandler &prm) final;

    /// Parse the parameters stored int the given parameter handler.
    /// @param[in] prm The parameter handler whose parameters we want to
    /// parse.
    void
    parse_parameters (dealii::ParameterHandler &prm) final;

    /// This function computes the constitutive response and writes
    /// all fields to the given scratch_data object.
    /// @param scratch_data the data object from which we get the
    /// all required data (displacement, etc.) and to which we write
    /// the constitutive response of the @p Ogden material.
    void
    evaluate (ScratchData<dim> &scratch_data) const final;

    /// Return which data has to be provided to compute the derived
    /// quantities. The flags returned here are the ones passed to
    /// the constructor of this class.
    dealii::UpdateFlags
    get_needed_update_flags () const final;

    /// Return the data interpretation information like the tensorial
    /// orders of the fields and the names of the fields computed by
    /// evaluate_vector_field. This information is returned as vector
    /// of @p DataInterpretation objects.
    std::vector<DataInterpretation>
    get_data_interpretation () const final;

    /// This function computes the post-processor information we want
    /// to write to the output files e.g. for paraview. Here, the
    /// displacements and the Piola stress tensor are computed.
    /// @param[in] input_data It provides all quantities necessary for
    /// the post processing.
    /// @param[out] computed_quantities The displacement and the Piola
    /// stress tensor are written to this vector. The <tt>std::vector<\tt>
    /// has as many entries as we have evaluation points, the internal
    /// <tt>dealii::Vector<double><\tt> stores the individual components
    /// of the displacement and the Piola stress tensor.
    /// @param[in] dummy Dummy variable
    void
    evaluate_vector_field (const dealii::DataPostprocessorInputs::Vector<dim> &input_data,
                           std::vector<dealii::Vector<double>> &computed_quantities,
                           const dealii::GeneralDataStorage* dummy = nullptr) const final;

private:

    /// First Lame parameter.
    double mu;

    /// Second Lame parameter.
    double lambda;
};



//----------------------- INLINE AND TEMPLATE FUNCTIONS ----------------------//



template<int dim>
inline
NeoHooke<dim>::
NeoHooke(const std::string &subsection_name,
         const std::string &unprocessed_input)
:
    ConstitutiveBase<dim>(subsection_name,unprocessed_input),
    mu (0),
    lambda (0)
{
    efilog(Verbosity::verbose) << "New NeoHooke constitutive model created ("
                               << subsection_name
                               << ")."<< std::endl;
}



template<int dim>
inline
void
NeoHooke<dim>::
declare_parameters (dealii::ParameterHandler &prm)
{
    prm.declare_entry("young's modulus","2.1e5",
            dealii::Patterns::Double(0));
    prm.declare_entry("poisson ratio","0.3",
            dealii::Patterns::Double(std::numeric_limits<double>::lowest(),0.5));

    efilog(Verbosity::verbose) << "NeoHooke constitutive model finished "
                                  "declaring parameters."
                               << std::endl;
}



template<int dim>
inline
void
NeoHooke<dim>::
parse_parameters (dealii::ParameterHandler &prm)
{
    double E  = prm.get_double("young's modulus");
    double nu = prm.get_double("poisson ratio");

    // Lame parameters
    this->lambda = E*nu/((1.+nu)*(1.-2.*nu));
    this->mu     = E/(2.*(1.+nu));

    efilog(Verbosity::debug) << "lambda: " << this->lambda << std::endl;
    efilog(Verbosity::debug) << "mu:     " << this->mu << std::endl;

    efilog(Verbosity::verbose) << "NeoHooke constitutive model finished parsing parameters."
                               << std::endl;
}

}//namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_CONSTITUTIVE_NEO_HOOKE_H_ */
