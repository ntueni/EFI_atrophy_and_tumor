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

#ifndef SRC_MYLIB_INCLUDE_EFI_BASE_EXTRACTOR_H_
#define SRC_MYLIB_INCLUDE_EFI_BASE_EXTRACTOR_H_

// deal.II headers
#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_values_extractors.h>


namespace efi
{

// The Extractor class provides an unified
// access to data fields of the viscoelastic
// problem.
template <int dim>
struct Extractor
{
    enum component {first_displacement_component = 0};
    enum global_vector {current};

    // Get the dealii::FEValuesExtractor
    // of the displacement field.
    static dealii::FEValuesExtractors::Vector displacement ();

    // Get the dealii::ComponentMask
    // of the displacement field.
    static dealii::ComponentMask displacement_mask ();
    // Get the dealii::ComponentMask
    // of the displacement field.
    static dealii::ComponentMask displacement_mask_inhom (int unconstr_dim);

    // The number of components of the
    // considered viscoelastic problem.
    static const unsigned int n_components = dim;

    // Global vector name used to store data
    // in/retrieve data from the ScratchData
    // objects.
    static constexpr std::string global_vector_name (const global_vector = current);
};



///////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION
///////////////////////////////////////////////////////////////////////////////



template <int dim>
inline
dealii::FEValuesExtractors::Vector
Extractor<dim>::
displacement ()
{
    return dealii::FEValuesExtractors::Vector(first_displacement_component);
}



template <int dim>
inline
dealii::ComponentMask
Extractor<dim>::displacement_mask ()
{
    std::vector<bool> mask (n_components,false);
    for (unsigned int c = first_displacement_component; c < first_displacement_component+dim; ++c)
        mask[c] = true;
    return mask;
}

template <int dim>
inline
dealii::ComponentMask
Extractor<dim>::displacement_mask_inhom (int unconstr_dim)
{
    std::vector<bool> mask (n_components,false);
    for (unsigned int c = 0; c < first_displacement_component+dim; ++c)
        if (c != unconstr_dim)
            mask[c] = true;
    return mask;
}

template <int dim>
inline
constexpr std::string
Extractor<dim>::global_vector_name (const global_vector)
{
    return "global_vector";
}

}// namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_BASE_EXTRACTOR_H_ */
