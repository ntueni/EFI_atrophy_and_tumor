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

// c++ headers
#include <ios>
#include <iomanip>
#include <fstream>
#include <algorithm>

// efi headers
#include <efi/base/factory_tools.h>


namespace efi {


FactoryTools::
Specifications::
Specifications (const std::string &specifications)
: specifications ()
{
    this->set (specifications);
}



double
FactoryTools::
Specifications::
get_double (const std::string &entry_string) const
{
    const std::string value_string = this->get(entry_string);

    try
    {
        return dealii::Utilities::string_to_double(value_string);
    }
    catch (...)
    {
        AssertThrow(false,
                    dealii::ExcMessage("Can't convert the value <" +
                                       value_string + "> for entry <" +
                                       entry_string +
                                       "> to a double precision variable."));
        return 0;
    }
}



int
FactoryTools::
Specifications::
get_integer (const std::string &entry_string) const
{
    const std::string value_string = this->get(entry_string);

    try
    {
        return dealii::Utilities::string_to_int(value_string);
    }
    catch (...)
    {
        AssertThrow(false,
                    dealii::ExcMessage("Can't convert the value <" +
                                       value_string + "> for entry <" +
                                       entry_string +
                                       "> to a signed integer variable."));
        return 0;
    }
}



bool
FactoryTools::
Specifications::
get_bool (const std::string &entry_string) const
{
    const std::string value_string = this->get(entry_string);

    if ((value_string=="true")|| (value_string=="yes")||
        (value_string=="y")   || (value_string=="1"))
        return true;
    else if ((value_string=="false")|| (value_string=="no")||
            (value_string=="n")     || (value_string=="0"))
        return false;
    else
    {
        AssertThrow(false,
                    dealii::ExcMessage("Can't convert the value <" +
                                       value_string + "> for entry <" +
                                       entry_string +
                                       "> to a boolean variable."));
        return false;
    }
}



std::string
FactoryTools::
Specifications::
get (const std::string &entry_string) const
{
    try
    {
        return this->specifications_map.at(entry_string);
    }
    catch (...)
    {
        auto splitted = split (specifications,Specifications::delimiter);

        if (std::find(std::begin(splitted),std::end(splitted),entry_string)
            != std::end(splitted))
        {
            AssertThrow (false,
                        dealii::ExcMessage ("Entry <" + entry_string +
                                            "> exists but is not meant to return "
                                            "anything (I could not find '=' in "
                                            "input string)."));
        }
        else
        {
            AssertThrow (false,
                        dealii::ExcMessage ("Entry <" + entry_string +
                                            "> not found."));
        }

        return "";
    }
}



void
FactoryTools::
Specifications::
set (const std::string &specifications)
{
    // set the new specification string
    this->specifications = specifications;

    // reset the map
    specifications_map.clear();

    if (!specifications.empty())
    {
        auto splitted = split (specifications,Specifications::delimiter);

        std::vector<std::vector<std::string>> tmp;

        for (auto &str : splitted)
            tmp.emplace_back(split (str,'='));

        for (auto &vec : tmp)
        {
            // Only entries formatted 'variable = value'
            // are written to the map
            if (vec.size()==2)
                specifications_map[vec[0]] = vec[1];
        }
    }
}




std::vector<std::string>
FactoryTools::
Specifications::
split (const std::string &specs,
       const char         delimiter) const
{
    std::vector<std::string> splitted;

    if (!specs.empty())
        splitted = dealii::Utilities::split_string_list (specs,delimiter);

    // remove leading and trailing whitespace including
    for (auto &str : splitted)
        dealii::Utilities::trim(str);

    return splitted;
}


void
FactoryTools::
efi_internal::
parse_section_header (const std::string &header,
                      std::string       &keyword,
                      FactoryTools::Specifications &specs)
{
    auto opt_begin = header.find("@[");
    auto opt_end   = header.rfind("]");

    if ((opt_begin != std::string::npos) && (opt_end != std::string::npos))
    {
        specs.set(header.substr(opt_begin+2,opt_end-opt_begin-2));
        keyword  = header.substr(0,opt_begin);
    }
    else
    {
        keyword  = header;
        specs.set("");
    }

    // Throw an expection is the section name is
    // empty.
    Assert (!keyword.empty(), dealii::ExcEmptyObject());
}



std::istream&
FactoryTools::
efi_internal::
getsection (std::istream &input_stream,
            std::string  &keyword,
            FactoryTools::Specifications &specs,
            std::string  &unprocessed_input,
            const bool    structure_only)
{
    using namespace dealii;

    // reset the strings
    keyword.clear();
    specs.clear();
    unprocessed_input.clear();

    std::string line;

    int  depth = 0;
    bool found = false;

    while(((depth>0)||(!found)) && std::getline (input_stream,line))
    {
        if (found && !structure_only)
            unprocessed_input += line + '\n';

        // TODO remove leading whitespace only?!
        auto trimmed_line = Utilities::trim(line);

        // check the first word of each line
        // whether it is 'section' or 'end'
        // or something else.
        if (Utilities::match_at_string_start(trimmed_line, "SUBSECTION ") ||
            Utilities::match_at_string_start(trimmed_line, "subsection "))
        {
            if (depth == 0)
            {
                // delete this prefix
                trimmed_line.erase(0, std::string("subsection").length() + 1);

                // parse the (sub-)section header
                parse_section_header (trimmed_line,keyword,specs);
            }
            else if ((depth>0) && structure_only)
                unprocessed_input += line + '\n';

            ++depth;
            found = true;
        }
        else if (Utilities::match_at_string_start(trimmed_line, "END") ||
                 Utilities::match_at_string_start(trimmed_line, "end"))
        {
            --depth;

            if ((depth>0) && structure_only)
                unprocessed_input += line + '\n';
        }
    }
    Assert(depth == 0, ExcMessage("Unbalanced subsections."));

    return input_stream;
}



std::string
FactoryTools::
get_subsection_name (const std::string &name,
                     const FactoryTools::Specifications &specs)
{
    return name+ (specs.empty()?"":"@["+specs.get()+"]");
}


void
FactoryTools::
apply (const std::map<std::string,action_type> actions,
       std::istream &is)
{
    using namespace dealii;

    std::string keyword;
    std::string unprocessed_input;

    FactoryTools::Specifications specs;

    std::istringstream cs;

    while (efi_internal::getsection(is,keyword,specs,unprocessed_input,true))
    {
        if (actions.find(keyword) != actions.end())
        {
            auto &action = actions.at(keyword);

            action (specs,unprocessed_input);
        }
        else
        {
            cs.clear();
            cs.str(unprocessed_input);

            FactoryTools::apply (actions,cs);
        }
    }
}



void
FactoryTools::
apply (const std::map<std::string,action_type> actions,
       const std::string &filename)
{
    std::ifstream file (filename);

    Assert (file.is_open(), dealii::ExcFileNotOpen (filename));

    FactoryTools::apply (actions,file);
}


}// namespace efi
