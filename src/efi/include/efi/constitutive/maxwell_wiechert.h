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

#ifndef SRC_EFI_INCLUDE_EFI_CONSTITUTIVE_MAXWELL_WIECHERT_H_
#define SRC_EFI_INCLUDE_EFI_CONSTITUTIVE_MAXWELL_WIECHERT_H_

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


/// Implementation of a generalized Maxwell model, also known
/// Maxwell-Wiechert model. For given deformations, it computes
/// the viscoelastic response given in terms of Piola-Kirchoff
/// stresses and their tangents with respect to the right Cauchy-
/// Green deformation tensor.
/// @author Stefan Kaessmair
template <int dim>
class MaxwellWiechert : public ConstitutiveBase<dim>
{
public:

    using scalar_type = double;

    /// Constructor.
    /// @param[in] subsection_name The name of the subsection in the parameter
    /// file which defines the parameters for the constructed instance
    /// of this class.
    /// @param[in] unprocessed_input The unprocessed parts of the parameter
    /// file is everything between the begin of the corresponding subsection
    /// of the to-be-constructed visco-elastic model and its end.
    MaxwellWiechert (const std::string &subsection_name,
                     const std::string &unprocessed_input);

    /// Default destructor.
    ~MaxwellWiechert () = default;

    /// This function computes the constitutive response and writes
    /// all fields to the given scratch_data object.
    /// For details, see Reese S., Govindjee, S. A theory of finite
    /// viscoelasticity and numerical aspects, Int. J. Solids
    /// Structures, 35:34555-3482 (1998).
    /// @param scratch_data the data object from which we get the
    /// all required data (displacement, etc.) and to which we write
    /// the constitutive response of the @p MaxwellWiechert model.
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
    /// @param[in] additional_input_data Additional input data for the cell
    /// which is about to be evaluated. This might be useful when the output
    /// cannot be computed from the <tt>input_data<\tt>, e.g. when history
    /// data is required.
    void
    evaluate_vector_field (const dealii::DataPostprocessorInputs::Vector<dim> &input_data,
                           std::vector<dealii::Vector<double>> &computed_quantities,
                           const dealii::GeneralDataStorage* additional_input_data = nullptr) const final;

private:

    // The components of the generalized maxwell element
    std::vector<std::unique_ptr<ConstitutiveBase<dim>>> components;
};



//------------------- INLINE AND TEMPLATE FUNCTIONS -------------------//



template<int dim>
inline
dealii::UpdateFlags
MaxwellWiechert<dim>::
get_needed_update_flags () const
{
    // The quadrature points positions will be used for a
    // least squares fit of the stresses on an element, such
    // that the stresses can be written to the output.
    dealii::UpdateFlags update_flags = dealii::update_quadrature_points;
    for (auto &c : components)
        update_flags = update_flags | c->get_needed_update_flags ();
    return update_flags;
}

}// namespace efi



#endif /* SRC_EFI_INCLUDE_EFI_CONSTITUTIVE_MAXWELL_WIECHERT_H_ */
