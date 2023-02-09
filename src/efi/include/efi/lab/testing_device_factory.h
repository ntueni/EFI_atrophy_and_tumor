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

#ifndef SRC_MYLIB_INCLUDE_EFI_LAB_TESTING_DEVICE_FACTORY_H_
#define SRC_MYLIB_INCLUDE_EFI_LAB_TESTING_DEVICE_FACTORY_H_

// efi headers
#include <efi/base/exceptions.h>
#include <efi/base/utility.h>
#include <efi/base/factory_tools.h>
#include <efi/lab/testing_device.h>
#include <efi/lab/tension_compression_testing_device.h>
#include <efi/lab/rotational_rheometer.h>
#include <efi/lab/translational_rheometer.h>
#include <efi/lab/retraction_spatulars.h>
#include <efi/lab/retraction_ellipse.h>
#include <efi/lab/retraction_expansion_tube.h>


namespace efi
{


// Factory pattern implementation for
// constitutive models.
template <int dim>
class TestingDeviceFactory
{
public:

    // Create a new testing device instance of
    // specified by the "type" stored in the
    // Sepcifications.
    static
    TestingDevice<dim>*
    create (const std::vector<std::string>     &section_path,
            const FactoryTools::Specifications &specs,
            const std::string                  &unprocessed_input,
            MPI_Comm mpi_communicator);

    // Create a new testing device instance
    // specified by the string type_str.
    // The arguments args... are forwarded to the
    // tesing device constructor.
    // An error of type efi::ExcNotConstructible
    // is thrown if the given string does not match
    // any option.
    template <class ... Args>
    static
    TestingDevice<dim>*
    create (const std::string& type_str,
            Args &&... args);

    // return the keyword which
    static
    std::string
    keyword ();

    // Return a string that specifies the list of
    // of allowed options. The options are separated
    // by '|'.
    static
    std::string
    get_names ();
};



//---------------------- INLINE AND TEMPLATE FUNCTIONS -----------------------//



template <int dim>
inline
TestingDevice<dim>*
TestingDeviceFactory<dim>::
create (const std::vector<std::string>     &section_path,
        const FactoryTools::Specifications &specs,
        const std::string                  &unprocessed_input,
        MPI_Comm mpi_communicator)
{
    return TestingDeviceFactory<dim>::create (
            specs.get("type"),
            get_section_path_str(section_path) + "/" +
                FactoryTools::get_subsection_name (keyword(),specs),
            unprocessed_input,
            mpi_communicator);
}



template <int dim>
template <class ... Args>
inline
TestingDevice<dim>*
TestingDeviceFactory<dim>::
create (const std::string& type_str,
        Args &&... args)
{
    if (type_str == "rotational_rheometer")
    {
        using model_type = RotationalRheometer<dim>;

        return make_new_if_constructible<model_type>(std::forward<Args>(args)...);
    }
    if (type_str == "translational_rheometer")
    {
        using model_type = TranslationalRheometer<dim>;

        return make_new_if_constructible<model_type>(std::forward<Args>(args)...);
    }
    if (type_str == "tension_compression_testing_device")
    {
        using model_type = TensionCompressionTestingDevice<dim>;

        return make_new_if_constructible<model_type>(std::forward<Args>(args)...);
    }
    if (type_str == "retraction_spatulars")
    {
        using model_type = RetractionSpatulars<dim>;

        return make_new_if_constructible<model_type>(std::forward<Args>(args)...);
    }
    if (type_str == "retraction_ellipse")
    {
        using model_type = RetractionEllipse<dim>;

        return make_new_if_constructible<model_type>(std::forward<Args>(args)...);
    }
    if (type_str == "retraction_expansion_tube")
    {
        using model_type = RetractionExpansionTube<dim>;

        return make_new_if_constructible<model_type>(std::forward<Args>(args)...);
    }
    else
        AssertThrow (false, ExcNotConstructible ());
}



template <int dim>
inline
std::string
TestingDeviceFactory<dim>::
keyword ()
{
    return "testing_device";
}



template <int dim>
inline
std::string
TestingDeviceFactory<dim>::
get_names ()
{
    return "rotational_rheometer|translational_rheometer|"
           "tension_compression_testing_device|"
           "retraction_spatulars|retraction_ellipse|retraction_expansion_tube";
}

}//namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_LAB_TESTING_DEVICE_FACTORY_H_ */
