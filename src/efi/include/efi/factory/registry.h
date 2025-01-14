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
#include <functional>
#include <map>
#include <string>

// dealii headers
#include <deal.II/base/types.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_acceptor.h>

// boost headers
#include <boost/any.hpp>
#include <boost/tti/tti.hpp>

// efi headers
#include <efi/base/utility.h>
#include <efi/base/type_traits.h>


#ifndef SRC_EFI_INCLUDE_EFI_FACTORY_REGISTRY_H_
#define SRC_EFI_INCLUDE_EFI_FACTORY_REGISTRY_H_

#define EFI_TEMPLATE_CLASS_AUX(X,...) X<__VA_ARGS__>
#define EFI_TEMPLATE_CLASS(X,...) EFI_TEMPLATE_CLASS_AUX(X,__VA_ARGS__)

#define _EFI_TEMPLATE_CLASS(X,...) _##X


// Preprocessor macro to register a class.
// NOTE that he macro must be called in a
// source file (*.cc).
#define EFI_REGISTER_OBJECT(classname) \
    static char EFI_COMBINE_NAMES (dummyvar_for_registration_efi_obj,EFI_COMBINE_NAMES(_##classname,__LINE__)) \
        EFI_UNUSED_ATTRIBUTE =                                                                                 \
        Registry::add<classname>(Registry::Entry{EFI_STRINGIFY(classname),                                     \
                                                 EFI_STRINGIFY(classname),                                     \
                                                 "",                                                           \
                                                 __LINE__,                                                     \
                                                 __FILE__,                                                     \
                                                 boost::any(nullptr)})


// Preprocessor macro to register a class. In
// addition to the above macro an alias name
// can be defined.
// NOTE that he macro must be called in a
// source file (*.cc).
#define EFI_REGISTER_ALIASED_OBJECT(classname,alias) \
    static char EFI_COMBINE_NAMES (dummyvar_for_registration_efi_obj,EFI_COMBINE_NAMES(_##classname,__LINE__)) \
        EFI_UNUSED_ATTRIBUTE =                                                                                 \
        Registry::add<classname>(Registry::Entry{EFI_STRINGIFY(classname),                                     \
                                                 alias,                                                        \
                                                 "",                                                           \
                                                 __LINE__,                                                     \
                                                 __FILE__,                                                     \
                                                 boost::any(nullptr)})


// This macro must be placed in a class body.
// It allows RegistryTools::RegisteredBaseTypeOf<T>
// to return the base of T which has REGISTER_AS_BASE
// in its body e.g.:
//
// struct Base {
//     EFI_REGISTER_AS_BASE;
//     virtual ~Base() = default;
// }
//
// struct Derived1 {
//     virtual ~Derived1() = default;
// }
//
// struct Derived2 {
//     virtual ~Derived2() = default;
// }
//
// Now RegistryTools::RegisteredBaseTypeOf<T> for
// T = Derived1 and T=Derived2 is equal to Base.
// However, if EFI_REGISTER_AS_BASE
// is missing in the Base class body, then
// RegistryTools::RegisteredBaseTypeOf<T> yields
// Derived1 and Derived2 for T=Derived1 and T=Derived2,
// respectively. Of course, on could simply define
// using registered_base_type = Base in Base, but I
//  preferred this macro (we also might add some
// more function to this later).
#define EFI_REGISTER_AS_BASE                                        \
        static const char registered_as_base = 1;                  \
        auto get_registered_base_object () const ->decltype(*this)


namespace efi
{
namespace RegistryTools
{
namespace efi_internal
{


// See template RegistryTools::base_type_of<T>
template <class T, class Enable = void>
struct RegisteredBaseTypeOfImpl;

template <class T>
struct RegisteredBaseTypeOfImpl<T,
    std::enable_if_t<has_static_member_data_registered_as_base<
        std::remove_pointer_t<std::decay_t<T>>,const char
    >::value>>
{
    using type =
        std::decay_t<
            decltype(std::declval<std::remove_pointer_t<std::decay_t<T>>>
                    ().get_registered_base_object())>;
};



// See template RegistryTools::base_type_of<T>
template <class T>
struct RegisteredBaseTypeOfImpl<T,
    std::enable_if_t<!has_static_member_data_registered_as_base<
        std::remove_pointer_t<std::decay_t<T>>,const char
    >::value>>
{
    using type = std::remove_pointer_t<std::decay_t<T>>;
};

}// namespace efi_internal



// Given a derived class T, base_type_of<Derived> is the
// type the base class that has the REGISTER_AS_BASE
// macro defined in its class body. If the macro is not defined
// the id of the type T is returned.
// Removes const, volatile &&, & modifiers and * modifiers from returned Type.
template <class T>
using RegisteredBaseTypeOf
        = typename efi_internal::RegisteredBaseTypeOfImpl<T>::type;



// Given a derived class T, the function returns the
// type_id of the Base class that has the REGISTER_AS_BASE
// defined macro in its class body. If the macro is not defined
// the id of the type T is returned.
// If trim is set to true only the pure name of the base class
// is returned, i.e. if trim is active the id "...::base_class_name<...>"
// is trimmed to "base_class_name<...>".
// Removes const, volatile &&, & modifiers and * modifiers from Type.
template <class T>
inline
std::string
registered_base_type_id (const bool trim = false)
{
    using registered_base_type = RegisteredBaseTypeOf<T>;

    std::string tmp =
            boost::typeindex::type_id<registered_base_type>().pretty_name();
    std::string::size_type first = trim? tmp.rfind("::") : 0;

    if (first == std::string::npos)
        first = 0;
    else if (trim)
        first += 2;

    return tmp.substr (first, tmp.size()-first);
}

}// namespace RegistryTools



// Registry that stores information of the
// self-registering classes of the efi-library.
// See also the Registry class of the moose framework
// (mooseframework.org).
//
// WARNING: when compiling a static library we require the options:
// For GCC:   -whole-archive for LD
// For MSVC:  /WHOLEARCHIVE:CompressionMethodsLib.lib
//            in the additional linker options
class Registry
{
public:

    // Registry entry which contains
    // all necessary information about
    // the registered class
    struct Entry
    {
        // Name of the registered class.
        std::string  classname;

        // Alias of the registered class. If no alias is
        // defined it is this field stores the same string
        // as classname.
        std::string  alias;

        // Key required to choose the correct factory to
        // create the desired an object of type classname
        std::string  factory_key;

        // For debugging: Store the linewhere the
        // register_..._object () has been called.
        unsigned int line;

        // For debugging: Store the file where the
        // register_..._object () has been called.
        std::string  file;

        // std::functions with signature
        // T*(const std::string&, const std::string&)
        // are stored in build_ptr with their types
        // erased. Note that the type T can be arbitrary.
        // To access the function pointer and run the
        // function use the following syntax:
        //
        // boost::hof::apply(boost::any_cast<std::function<T*(...)>>(m_any), ...)
        //
        // See also https://stackoverflow.com/questions/45715219/store-functions-with-different-signatures-in-a-map/45718187
        boost::any  build_ptr;
    };

    // Add a new entry to the registry. The entry fields
    // classname, alias, line, file are filled directly
    // by the registration macros (register_object(T) OR
    // register_aliased_object(T,"my alias")).
    // Internally, the Entry::factory_key and the build_ptr
    // are added by this function and the result is copied
    // to the map Registry::name_to_entry of a static
    //  Registry-singleton.
    // The Entry::build_ptr holds an object of type
    // std::function<base_type*(const std::string&, const std::string&)>
    // where base_type* = RegistryTools::RegisteredBaseTypeOf<T>.
    template <class T>
    static char
    add (const Entry &entry);

    // Print the entire registry.
    template <class OStreamType>
    static void
    print (OStreamType &&out);

    //TODO:
//    /// Write the parameter descriptions of the registered entries.
//    /// This function creates a directory which has the following structure
//    /// param[in] output_dir The output directory.
//    static void
//    print_parameter_description (const std::string &output_dir = ".");

private:

    // Create the factory key from a given
    // type. The key is simple the name of the
    // class represented by RegistryTools::base_type_of<T>
    // modified in the following way:
    //
    // - remove modifiers (const, volatile, &, &&, *, **)
    // - remove template arguments
    //   (MyClassExampleBase<...> -> MyClassExampleBase)
    // - Convert the class name from snake-case
    //   (MyClassExampleBase -> my_class_example_base)
    // - remove possible trailing '_base' strings
    //   (my_class_example_base -> my_class_example)
    //
    // Now, my_class_example is the generated factory key.
    template <class T>
    static std::string
    get_factory_key ();

    // Helper function to add a new entry to
    // the registry. The get_registry() function
    // which gives access to the global Registry
    // object is only defined in the source file
    // registry.cc. Since the template arguments
    // of add<T> cannot be specified yet, it must
    // remain in the header file. However, the
    // body of add_helper is defined in the source
    // file and copies the entry 'entry' to the
    // global registry.
    static void
    add_helper (const Entry &entry);

    // Maps class names to registred entries.
    std::map<std::string,Entry> name_to_entry;
};



template <class T>
char
Registry::add (const Entry &entry)
{
    Entry copy = entry;
    copy.factory_key = get_factory_key<T>();

    using base_type = RegistryTools::RegisteredBaseTypeOf<T>;

    // Initialize the build_ptr. The type of the function ptr, i.e.
    // base_type*(const std::string&, const std::string&)
    // is erased when stored in the build_ptr.
    copy.build_ptr = std::function<base_type*(
            const std::string&, const std::string&)>
                     ([](const std::string& subsection_name,
                         const std::string& unporcessed_input)
                             ->base_type* // Return type
                      {
                           return static_cast<base_type*>(
                                   make_new_if_constructible<T>(
                                           subsection_name,unporcessed_input));
                      });
    // copy 'copy' to the name_to_entry map of
    // a Registry singleton
    add_helper (copy);

    return 0;
}



template <class T>
std::string
Registry::get_factory_key ()
{
    // Convert the typename from camel- to snake-case and remove the
    // template arguments and other modifiers (cv,&,&&,*,**).
    std::string tmp = RegistryTools::registered_base_type_id<T>(true);

    // remove the template arguments and convert the class name to snake case.
    tmp = to_snake_case (tmp.substr(0, tmp.rfind("<")));

    // Any appended 'base' string is removed.
    if (tmp.rfind("base")!=std::string::npos)
        tmp = tmp.substr(0,tmp.size()-4);
    // Trailing underscores are removed.
    while (tmp.back()=='_')
        tmp = tmp.substr(0,tmp.size()-1);

    return tmp;
}



// Stream operator for Registry::entry.
std::ostream &
operator<< (std::ostream &out, const Registry::Entry &entry);


}// namespace efi

#endif /* SRC_EFI_INCLUDE_EFI_FACTORY_REGISTRY_H_ */
