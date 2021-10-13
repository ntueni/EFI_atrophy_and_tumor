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

#ifndef SRC_EFI_INCLUDE_EFI_CONSTITUTIVE_MAXWELL_ELEMENT_H_
#define SRC_EFI_INCLUDE_EFI_CONSTITUTIVE_MAXWELL_ELEMENT_H_

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
#include <efi/base/postprocessor.h>
#include <efi/worker/scratch_data.h>


namespace efi {


/// Implementation of a Maxwell element, i.e. a spring and a dashpod in
/// serial. For given deformations, it computes the viscoelastic
/// response given in terms of Piola-Kirchoff stresses and their tangents
/// with respect to the right Cauchy-Green deformation tensor.
/// @author Stefan Kaessmair
template <int dim>
class MaxwellElement : public ConstitutiveBase<dim>
{
public:

    using scalar_type = typename ConstitutiveBase<dim>::scalar_type;

    /// Constructor.
    /// @param[in] subsection_name The name of the subsection in the parameter
    /// file which defines the parameters for the constructed instance
    /// of this class.
    /// @param[in] unprocessed_input The unprocessed parts of the parameter
    /// file is everything between the begin of the corresponding subsection
    /// of the to-be-constructed Maxwell element and its end.
    MaxwellElement (const std::string &subsection_name,
                    const std::string &unprocessed_input);

    /// Default destructor.
    ~MaxwellElement () = default;

    /// Declare parameters to the given parameter handler.
    /// @param prm[out] The parameter handler for which we want to declare
    /// the parameters.
    void
    declare_parameters (dealii::ParameterHandler &prm) final;

    /// Parse the parameters stored int the given parameter handler.
    /// @param prm[in] The parameter handler whose parameters we want to
    /// parse.
    void
    parse_parameters (dealii::ParameterHandler &prm) final;

    /// This function computes the constitutive response and writes
    /// all fields to the given scratch_data object.
    /// For details, see Reese, S., Govindjee, S. A theory of finite
    /// viscoelasticity and numerical aspects, Int. J. Solids
    /// Structures, 35:34555-3482 (1998).
    /// @param scratch_data the data object from which we get the
    /// all required data (displacement, etc.) and to which we write
    /// the constitutive response of the @p MaxwellElement.
    void
    evaluate (ScratchData<dim> &scratch_data) const final;

    /// Return which data has to be provided to compute the derived
    /// quantities. The flags returned here are the ones passed to
    /// the constructor of this class.
    dealii::UpdateFlags
    get_needed_update_flags () const  final;

private:

    /// Update the internal equilibrium.
    /// @param[in,out] scratch_data the data object from which we get
    /// the all required data (e.g. the kinematics) and to which we
    /// write the local equilibrium fields.
    void
    update_internal_equilibrium (ScratchData<dim> &scratch_data) const;

    /// Spring of the maxwell element.
    std::unique_ptr<ConstitutiveBase<dim>> spring;

    /// Deviatoric viscosity of the dashpot.
    double eta_d;

    /// Volumetric viscosity of the dashpot.
    double eta_v;
};



//----------------------- INLINE AND TEMPLATE FUNCTIONS ----------------------//



template<int dim>
inline
void
MaxwellElement<dim>::
declare_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    prm.declare_entry ("deviatoric viscosity", "0", Patterns::Double(0));
    prm.declare_entry ("volumetric viscosity", "0", Patterns::Double(0));

    efilog(Verbosity::verbose) << "MaxwellElement constitutive model finished "
                                  "declaring parameters."
                               << std::endl;
}



template<int dim>
inline
void
MaxwellElement<dim>::
parse_parameters (dealii::ParameterHandler &prm)
{
    using namespace dealii;

    this->eta_d = prm.get_double ("deviatoric viscosity");
    this->eta_v = prm.get_double ("volumetric viscosity");

    efilog(Verbosity::verbose) << "MaxwellElement constitutive model finished "
                                  "parsing parameters."
                               << std::endl;
}



template<int dim>
inline
dealii::UpdateFlags
MaxwellElement<dim>::
get_needed_update_flags () const
{
    Assert(this->spring,dealii::ExcNotInitialized());
    return this->spring->get_needed_update_flags();
}


}// namespace efi



#endif /* SRC_EFI_INCLUDE_EFI_CONSTITUTIVE_MAXWELL_ELEMENT_H_ */
