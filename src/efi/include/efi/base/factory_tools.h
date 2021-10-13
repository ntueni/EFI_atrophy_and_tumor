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

#ifndef SRC_MYLIB_INCLUDE_EFI_BASE_FACTORY_TOOLS_H_
#define SRC_MYLIB_INCLUDE_EFI_BASE_FACTORY_TOOLS_H_

// c++ headers
#include <map>
#include <string>
#include <istream>
#include <functional>
#include <vector>

// deal.II headers
#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

// efi headers
#include <efi/base/exceptions.h>


namespace efi {


// Toolbox for deal.II parameter files.
// So far, only the format ParameterHandler::OutputType::Text
// is supported by this toolbox.
namespace FactoryTools {


// This is like a minimal version of
// dealii's ParameterHandler.
class Specifications
{
public:

    // Copy constructor
    Specifications (const Specifications &) = default;

    // Constructor
    Specifications (const std::string &specifications = "");

    // Copy assignment
    Specifications& operator= (const Specifications&) = default;

    // Destructor
    virtual ~Specifications () = default;

    // Return the value of the specification entry
    // entry_string as double. An assertion is thrown
    // when the stored value cannot be converted to
    // double or if the entry does not exist.
    double
    get_double (const std::string &entry_string) const;

    // Return the value of the specification entry
    // entry_string as int. An assertion is thrown
    // when the stored value cannot be converted to
    // int or if the entry does not exist.
    int
    get_integer (const std::string &entry_string) const;

    // Return the value of the specification entry
    // entry_string as bool. An assertion is thrown
    // when the stored value cannot be converted to
    // bool or if the entry does not exist.
    bool
    get_bool (const std::string &entry_string) const;

    // Return the value of the specification entry
    // entry_string as string. An assertion is thrown
    // if the entry does not exist.
    std::string
    get (const std::string &entry_string) const;

    // Return the specifications as string.
    const std::string &
    get () const;

    // Return the specifications as string.
    void
    set (const std::string &specifications);

    // Returns if the specifications are empty
    bool
    empty () const;

    // Clear the object
    void
    clear ();

    // delimiter
    static const char delimiter = ',';

private:

    // Split the comma separated list of specs
    // given as a single string into its
    // elements, which are stored in a vector.
    // Leading and trailing whitespace is removed.
    std::vector<std::string>
    split (const std::string &specs,
           const char         delimiter = Specifications::delimiter) const;

    // specifications
    std::string specifications;

    // processed specifications
    std::map<std::string,std::string> specifications_map;
};



namespace efi_internal {

// Consider an example of a deal.II parameter file as
// produced with the ParameterHandler::OutputType::Text
// option:
//
// subsection header
//   set param = ...
//   ...
// end
//
// Note that the header string is expected to
// have the following format:
// keyword@[specs]
// e.g. constitutive@[type=neo-hooke]
// where @[specs] is optional and may be omitted.
// The specifications (specs) are instructions
// for the factory methods creating the requested
// objects at runtime, i.e. for the given example
// ConstitutiveFactory<dim>::create ("neo-hooke",...)
// is called.
//
// This function parses the header and reads
// the (sub-)section keyword and the given specs.
// When parsing the header, we always search for
// the first "@[" and the last "]" to appear.
// That is, within the brackets "@[" and "]" may
// appear again.
// TODO also make this work for other file types like
//      *.json or *.xml
void
parse_section_header (const std::string &header,
                      std::string       &keyword,
                      Specifications    &specs);



// Consider an example of a deal.II parameter file as
// produced with the ParameterHandler::OutputStyle::Text
// option:
//
// subsection header
//   set param = ...
//   ...
// end
//
// Note that the header of a subsection is
// expected to have the following format:
//
// keyword@[specs]
//
// where @[specs] is optional and may be omitted.
// All lines between "subsection header" and the
// corresponding "end" are written to unprocessed_input.
// If the structure_only flag is set, then only
// the lines between "subsection header" and
// and the corresponding "end" that start with
// one of the keywords  "subsection" or "end" are
// written to unprocessed_input.
//
// TODO also make this work for other file types like
//      *.json or *.xml
std::istream&
getsection (std::istream   &input_stream,
            std::string    &keyword,
            Specifications &specs,
            std::string    &unprocessed_input,
            const bool      structure_only = false);
}


// Combine the given keyword and specs to get
// the subsection name originally read from
// the dealii input file.
std::string
get_subsection_name (const std::string    &keyword,
                     const Specifications &specs);



// typedef for the actions used in apply () below.
using action_type = std::function<void(const Specifications &,const std::string &)>;



// Consider an example of a deal.II parameter file as
// produced with the ParameterHandler::OutputType::Text
// option:
//
// subsection header
//   set param = ...
//   ...
// end
//
// Note that the header of a subsection is
// expected to have the following format:
//
// keyword@[specs]
//
// where @[specs] is optional and may be omitted.
// This function reads the input stream is and parses
// the section a section headers. When the parsed
// keyword matches a key in the map actions, the
// corresponding function is carried out.
// The functions take two arguments:
// - the parsed 'specs'
// - the 'unprocessed_input' i.e. every thing
//   between of 'subsection header' and the corresponding
//   'end' of the processed section.
// These arguments are internally provided
// by calling @ParameterTools::getsection.
void
apply (const std::map<std::string,action_type> actions,
       std::istream &input_stream);



// Like the function above, but instead of a given
// stream that is parsed, it opens the file 'filename'
// and parses its input.
void
apply (const std::map<std::string,action_type> actions,
       const std::string &filename);



//---------------------- INLINE AND TEMPLATE FUNCTIONS -----------------------//



inline
const std::string &
Specifications::
get () const
{
    return specifications;
}



inline
bool
Specifications::
empty () const
{
    return this->specifications.empty();
}



inline
void
Specifications::
clear ()
{
    this->specifications.clear();
    this->specifications_map.clear();
}

}// namespace ParameterTools

}// namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_BASE_FACTORY_TOOLS_H_ */
