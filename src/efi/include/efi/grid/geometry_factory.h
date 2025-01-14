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

#ifndef SRC_MYLIB_INCLUDE_EFI_GRID_GEOMETRY_FACTORY_H_
#define SRC_MYLIB_INCLUDE_EFI_GRID_GEOMETRY_FACTORY_H_

// efi header
#include <efi/base/utility.h>
#include <efi/base/factory_tools.h>
#include <efi/grid/geometry.h>


namespace efi
{


// Factory pattern implementation for
// Geometries.
template <int dim>
class GeometryFactory
{
public:

    // Create a new Geometry instance of
    // specified by the "type" stored in the
    // Sepcifications.
    static
    Geometry<dim>*
    create (const std::vector<std::string>     &section_path,
            const FactoryTools::Specifications &specs,
            const std::string                  &unprocessed_input);

    // Create a new Geometry instance of
    // specified by the string type_str.
    // The arguments args... are forwarded to the
    // contitutive model constructor.
    // An error of type efi::ExcNotConstructible
    // is thrown if the given string does not match
    // any option.
    template <class ... Args>
    static
    Geometry<dim>*
    create (const std::string& type_str,
            Args &&... args);

    // return the keyword which
    static
    std::string
    keyword ();

    // Return a string that specifies the list of
    // of allowed options. The options are separated
    // by '|' no addional spaces are added (e.g.
    // "model1|model2|model3").
    static
    std::string
    get_names ();
};



///////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION
///////////////////////////////////////////////////////////////////////////////



template <int dim>
Geometry<dim>*
GeometryFactory<dim>::
create (const std::vector<std::string>     &section_path,
        const FactoryTools::Specifications &specs,
        const std::string                  &unprocessed_input)
{
    return GeometryFactory<dim>::create (
            specs.get("type"),
            get_section_path_str(section_path) + "/" +
                FactoryTools::get_subsection_name (keyword(),specs),
            unprocessed_input);
}



template <int dim>
template <class ... Args>
Geometry<dim>*
GeometryFactory<dim>::
create (const std::string& type_str,
        Args &&... args)
{
    if (type_str == "block")
    {
        using geometry_type = Block<dim>;

        return make_new_if_constructible<geometry_type>(std::forward<Args>(args)...);
    }
    else if (type_str == "cylinder")
    {
        using geometry_type = Cylinder<dim>;

        return make_new_if_constructible<geometry_type>(std::forward<Args>(args)...);
    }
    else if (type_str == "import")
    {
        using geometry_type = ImportedGeometry<dim>;

        return make_new_if_constructible<geometry_type>(std::forward<Args>(args)...);
    }
    else
        AssertThrow (false, ExcNotConstructible ());
}

template <int dim>
inline
std::string
GeometryFactory<dim>::
keyword ()
{
    return "geometry";
}



template <int dim>
inline
std::string
GeometryFactory<dim>::
get_names ()
{
    return "block|cylinder|import";
}

}//namespace efi


#endif /* SRC_MYLIB_INCLUDE_EFI_GRID_GEOMETRY_FACTORY_H_ */
