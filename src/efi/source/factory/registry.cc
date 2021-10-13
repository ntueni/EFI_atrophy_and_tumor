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

// stl headers
#include <memory>
#include <iostream>
#include <ios>
#include <iomanip>

// deal.II headers
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>

// boost headers
#include <boost/filesystem.hpp>

// efi headers
#include <efi/factory/registry.h>
#include <efi/base/logstream.h>


namespace efi {


// Return a reference to the registry singleton.
Registry&
get_registry ();



///////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION
///////////////////////////////////////////////////////////////////////////////



Registry&
get_registry ()
{
    static std::unique_ptr<Registry> singleton;
    if (!singleton)
        singleton = std::make_unique<Registry>();
    return *singleton;
}



void
Registry::add_helper (const Entry &entry)
{
    auto &r = get_registry();

    if(r.name_to_entry.find(entry.classname) == r.name_to_entry.end())
        r.name_to_entry[entry.classname] = entry;
}



template <class OStreamType>
void
Registry::
print (OStreamType &&out)
{
    auto &r = get_registry();

    unsigned int max_length = 1;
    for (const auto &el : r.name_to_entry)
    {
        std::string::size_type pos = el.second.file.find("/efi/");

        if(max_length<(el.second.file.size()-(pos!=std::string::npos? pos : 0)))
            max_length = el.second.file.size()-(pos!=std::string::npos? pos : 0);
    }

    max_length += 131;

    // Write the table header...
    out << std::string(max_length,'_')
        <<"\n"
        << "Registry entries:        class name"
        << std::setw(35) << "alias"
        << std::setw(35) << "factory key"
        << "       registered at\n"
        << std::string(max_length,'_')
        <<"\n";

    // and now write the table content
    for (const auto &el : r.name_to_entry)
        out << el.second << '\n';
    out << std::string(max_length,'_') << std::endl;;
}



//void
//Registry::
//print_parameter_description (const std::string &output_dir)
//{
//    using namespace dealii;
//
//    boost::filesystem::path base_dir = output_dir;
//
//    AssertThrow (boost::filesystem::exists (base_dir),
//            ExcMessage("Directory <" + output_dir +"> does not exist."));
//    AssertThrow (boost::filesystem::is_directory (base_dir),
//            ExcMessage("Path <" + output_dir +"> does not name a directory."));
//
//    auto &r = get_registry();
//
//
//
//    for (auto &value : r.name_to_entry)
//    {
//        boost::filesystem::path sub_dir = base_dir;
//
//        auto &key   = value.first;
//        auto &entry = value.second;
//
//        try
//        {
//            entry.build_ptr("");
//        }
//        catch (ExcNotConstructible &)
//        {
//            std::cerr << "Entry <"<< entry
//                      << "> is not constructible by the given constructor. "
//                      << "No parameter description has been printed. "
//                      << std::endl;
//        }
//    }
//}



std::ostream &
operator<< (std::ostream &out, const Registry::Entry &entry)
{
    auto tmp = dealii::Utilities::split_string_list(entry.file,boost::filesystem::path::separator);

    auto it = std::find(std::begin(tmp),std::end(tmp),"efi");

    std::string file;

    for (;it!=std::end(tmp);++it)
        file += (file.empty()? "" : std::string(1,boost::filesystem::path::separator)) + *it;

    out << std::setw(35) << (entry.classname.size()   < 31? entry.classname   : std::string(&(entry.classname[0]),31)+"...")
        << std::setw(35) << (entry.alias.size()       < 31? entry.alias       : std::string(&(entry.alias[0]),31)+"...")
        << std::setw(35) << (entry.factory_key.size() < 31? entry.factory_key : std::string(&(entry.factory_key[0]),31)+"...")
        << "       line " << std::setw(6)  << entry.line
        << " in file "  << file;
    return out;
}



// Instantiation
template void Registry::print<std::ostream&>(std::ostream &);
template void Registry::print<dealii::ConditionalOStream&>(dealii::ConditionalOStream &);
template void Registry::print<dealii::LogStream&>(dealii::LogStream &);
template void Registry::print<efi_internal::LogStreamWrapper>(efi_internal::LogStreamWrapper &&);

}// namespace efi
