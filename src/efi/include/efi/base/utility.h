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

#ifndef SRC_MYLIB_INCLUDE_EFI_BASE_UTILITY_H_
#define SRC_MYLIB_INCLUDE_EFI_BASE_UTILITY_H_

// c++ headers
#include <string>
#include <ios>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <ctime>
#include <regex>

// deal.II headers
#include <deal.II/base/mpi.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/base/timer.h>

// boost headers
#include <boost/filesystem.hpp>
#include <boost/type_index.hpp>

// efi headers
#include <efi/base/exceptions.h>
#include <efi/base/logstream.h>
#include <efi/base/type_traits.h>


// some macro definitions
#define EFI_VERSION_MAJOR 0
#define EFI_VERSION_MINOR 1
#define EFI_PATCHLEVEL 0

#if (defined(__GNUC__ ) || defined(_MSC_VER))
#define EFI_FUNCTION __func__
#else
#define EFI_FUNCTION __FUNCTION__
#endif


#if defined(__GNUC__ )
#define EFI_PRETTY_FUNCTION __PRETTY_FUNCTION__
#else
#define EFI_PRETTY_FUNCTION __FUNCTION__
#endif


#if defined(__GNUC__)
#define EFI_UNUSED_ATTRIBUTE  __attribute__((unused))
#else
#define EFI_UNUSED_ATTRIBUTE
#endif


#define EFI_COMBINE_NAMES_IMPL(X,Y) X##Y
#define EFI_COMBINE_NAMES(X,Y) EFI_COMBINE_NAMES_IMPL(X,Y)


#define EFI_STRINGIFY_IMPL(...) #__VA_ARGS__
#define EFI_STRINGIFY(...) EFI_STRINGIFY_IMPL(__VA_ARGS__)



namespace efi {


// Convert a string from camel to snake case
inline
std::string
to_snake_case (const std::string& camel_case)
{
    // Convert lowerCamelCase and UpperCamelCase strings to
    // lower_with_underscore.
    std::string str(1, std::tolower(camel_case[0]));

    // First place underscores between contiguous lower and upper case letters.
    // For example, `_LowerCamelCase` becomes `_Lower_Camel_Case`.
    for (auto it = camel_case.begin() + 1; it != camel_case.end(); ++it)
    {
        if (std::isupper(*it) && *(it-1) != '_' && std::islower(*(it-1)))
            str += "_";
        str += *it;
    }
    // Then convert it to lower case.
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);

    return str;
}



// Stream the parameters of a class T to out
template <class T>
inline
std::enable_if_t<(has_member_function_declare_parameters<void(T::*)(dealii::ParameterHandler &)>::value
               || has_static_member_function_declare_parameters<T,void(dealii::ParameterHandler &)>::value)
>
print_parameters(T& obj,
                 std::ostream & out,
                 dealii::ParameterHandler::OutputStyle style)
{
    using namespace dealii;
    ParameterHandler prm;

    obj.declare_parameters (prm);
    prm.print_parameters(out,style);
}

// Stream the parameters of a class T to out
template <class T>
inline
std::enable_if_t<!(has_member_function_declare_parameters<void(T::*)(dealii::ParameterHandler &)>::value
                || has_static_member_function_declare_parameters<T,void(dealii::ParameterHandler &)>::value)
>
print_parameters(T&,
                 std::ostream &,
                 dealii::ParameterHandler::OutputStyle)
{
    // Nothing to print
}

// Print the parameters of a class.
template <class T>
inline
void
print_parameters_to_file (T& obj,
                          dealii::ParameterHandler::OutputStyle style = dealii::ParameterHandler::OutputStyle::ShortText,
                          std::string output_directory = boost::filesystem::current_path().string())
{
    using namespace dealii;

    boost::filesystem::path output_path = output_directory;

    Assert (boost::filesystem::exists(output_path),
            ExcMessage("<"+output_path.string()+"> not found."));

    std::string tmp = boost::typeindex::type_id<T>().pretty_name();
    std::string::size_type first = tmp.rfind("::");
    std::string::size_type last;//  = tmp.rfind("<");

    if (first == std::string::npos)
        first = 0;
    else
        first += 2;

//    if (last == std::string::npos)
        last = tmp.size();

    std::string filename = output_path.string()
                         + std::string(1,boost::filesystem::path::separator)
                         + to_snake_case(tmp.substr(first,last-first)) + ".prm";

    std::ofstream out (filename);

    AssertThrow(out.is_open(),ExcFileNotOpen(filename));
    print_parameters (obj,out,style);

    out.close();
}

// Return the absolute path of a
// dealii::ParameterAcceptor object
// as string.
inline
std::string
get_section_path_str (const std::vector<std::string> &section_path)
{
    std::string path = "";

    for (auto &section_name : section_path)
        path += "/" + section_name;

    return path;
}



template <class T>
inline
constexpr std::enable_if_t<has_static_member_data_dimension<T,unsigned int>::value,unsigned int>
get_dimension()
{
    return T::dimension;
}



template <class T>
inline
constexpr std::enable_if_t<has_static_member_data_space_dimension<T,unsigned int>::value,unsigned int>
get_space_dimension()
{
    return T::space_dimension;
}



template <class T>
inline
constexpr std::enable_if_t<!has_static_member_data_dimension<T,unsigned int>::value,unsigned int>
get_dimension()
{
    return dealii::numbers::invalid_unsigned_int;
}



template <class T>
inline
constexpr std::enable_if_t<!has_static_member_data_space_dimension<T,unsigned int>::value,unsigned int>
get_space_dimension()
{
    return dealii::numbers::invalid_unsigned_int;
}


// Return the version number of this library
// as string
inline
std::string
version ()
{
    return dealii::Utilities::int_to_string(EFI_VERSION_MAJOR) + "." +
           dealii::Utilities::int_to_string(EFI_VERSION_MINOR) + "." +
           dealii::Utilities::int_to_string(EFI_PATCHLEVEL);
}


// Print the efi headers
template <class OStream>
inline
void
print_efi_header (OStream &&out)
{

    std::time_t printable_time =
            std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    out << "using efi " << version() << '\n'
        << "    on: " << std::ctime(&printable_time)
        << "author: Stefan Kaessmair\n\n";
}



inline
std::string
highlight (const std::string &str)
{
#if defined(__GNUC__ )
    return "\033[31m" + str + "\033[0m";
#else
    return str;
#endif
}



namespace efi_internal {

// Return a pointer to an object of type Type.
// The object is constructed via 'new' from
// the given arguments.
// If no matching constructor is available, sfinae kicks
// in and the version of the function below
// is used.
// Depending on which function is
// enabled, it only calls the constructors
// that really exist for the given arguments,
// otherwise it just returns a nullptr
// of the requested type but does not result
// in a compiler error.
template <class Type, class ... Args>
inline
std::enable_if_t<std::is_constructible<Type,Args...>::value, Type*>
make_new_if_constructible_impl (Args&&... args)
{
    return new Type (std::forward<Args>(args)...);
}



// Retrun a nullptr. An exception of type
// ExcNotConstructible is thrown.
template <class Type, class ... Args>
inline
std::enable_if_t<!std::is_constructible<Type,Args...>::value, Type*>
make_new_if_constructible_impl (Args&&...)
{
    AssertThrow (false, ExcNotConstructible ())
    return nullptr;
}

}// namespace efi_internal



// Return a pointer to an object of type Type.
// The object is constructed via 'new' from
// the given arguments.
// If no matching constructor is available, an
// exception of type ExcNotConstructible is
// thrown.
template <class Type, class ... Args>
inline
Type*
make_new_if_constructible (Args&&...args)
{
    return efi_internal::make_new_if_constructible_impl<Type>(std::forward<Args>(args)...);
}



inline
const std::string
time_stamp ()
{
    time_t now = time(0);
    tm     tstruct;
    char   buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d_%Hh%Mm%Ss", &tstruct);

    return buf;
}



inline
std::vector<unsigned int>
unrolled_to_component_index (const unsigned int i,
                             const unsigned int rank,
                             const unsigned int dim)
{
    // Check if i is a valid index.
    AssertIndexRange (i, dealii::Utilities::pow(dim,rank));

    std::vector<unsigned int> indices (rank);

    unsigned int remainder = i;
    for (int r=rank-1; r>=0; --r)
      {
        indices[r] = (remainder % dim);
        remainder /= dim;
      }
    Assert (remainder == 0, dealii::ExcInternalError());
    return indices;
}



// Extract the relevant indices of the degrees of freedom belonging to certain vector
// certain vector components of a vector-valued finite element. If the DoFHander is
// ordinary, all dofs are relevant. If a parallel::distributed::Triangulation
// was used, only a subset of the indices is locally relevant.
// TODO This is not very efficient.
//
// WARNING: This function is only works for primitive elements!
template <class DoFHandlerType>
inline
void
extract_locally_relevant_dofs (const DoFHandlerType        &dof_handler,
                               dealii::IndexSet            &selected_dofs,
                               const dealii::ComponentMask &component_mask = dealii::ComponentMask ())
{
    if (component_mask.represents_the_all_selected_mask())
    {
        dealii::DoFTools::extract_locally_relevant_dofs (dof_handler, selected_dofs);
    }
    else
    {
        // TODO It works also for vector-values problems if no component of a
        // non-primitive element is selected.

        // Extracting dofs component-wise only works for components of primitive elements.
        Assert(dof_handler.get_fe ().is_primitive (),
               typename dealii::FiniteElement<DoFHandlerType::dimension>::ExcFENotPrimitive());

        dealii::IndexSet locally_relevant_dofs;

        dealii::DoFTools::extract_locally_relevant_dofs (dof_handler, locally_relevant_dofs);

        selected_dofs.set_size (locally_relevant_dofs.size());

        std::vector<dealii::types::global_dof_index> dof_indices;
        std::vector<dealii::types::global_dof_index> dofs_on_ghosts;

        typename DoFHandlerType::active_cell_iterator cell = dof_handler.begin_active(),
                                                      endc = dof_handler.end();

        for (; cell!=endc; ++cell)
            if (cell->is_locally_owned() || cell->is_ghost())
            {
                dof_indices.resize(cell->get_fe().dofs_per_cell);
                cell->get_dof_indices(dof_indices);

                for (unsigned int i = 0; i < dof_indices.size(); ++i)
                    if (locally_relevant_dofs.is_element(dof_indices[i]))
                        if (component_mask[cell->get_fe().system_to_component_index(i).first])
                            selected_dofs.add_index(dof_indices[i]);
            }
    }
}



// Return a map of support points for the locally relevant degrees of freedom,
// which are selected by the component mask and are handled by this DoF handler object.
// This function, of course, only works if the finite elements,
// whose components are selected by the component mask, provide support points, i.e. no edge
// elements or Raviart-Thomas elements. The components represented by
// these elements cannot be selected. Otherwise, an exception is thrown.
template <int dim, int spacedim>
void
map_dofs_to_support_points (const dealii::Mapping<dim, spacedim>                               &mapping,
                            const dealii::DoFHandler<dim, spacedim>                            &dof_handler,
                            const dealii::ComponentMask                                        &component_mask,
                            std::map<dealii::types::global_dof_index, dealii::Point<spacedim>> &support_points)
{
    support_points.clear();

    const dealii::FiniteElement<dim,spacedim> &fe = dof_handler.get_fe ();

    if (component_mask.represents_the_all_selected_mask ())
    {
        Assert(fe.has_support_points(),
               typename dealii::FiniteElement<dim>::ExcFEHasNoSupportPoints());
    }
    else
    {
        // check if mask is valid, i.e. all corresponding base elements have support points.
        for (unsigned int c = 0; c < component_mask.size(); ++c)
            if (component_mask[c])
                Assert(fe.base_element(fe.component_to_base_index(c).first).has_support_points (),
                       typename dealii::FiniteElement<dim>::ExcFEHasNoSupportPoints());
    }


    std::vector<bool> active_system_indices (fe.n_dofs_per_cell(), false);
    std::vector<dealii::Point<dim> > q_points (fe.n_dofs_per_cell());

    for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        if (fe.base_element(fe.system_to_base_index (i).first.first).is_primitive()
         && fe.base_element(fe.system_to_base_index (i).first.first).has_support_points())
        {
            active_system_indices[i] = component_mask[fe.system_to_component_index(i).first];
            q_points[i] = fe.unit_support_point(i);
        }

    dealii::Quadrature<dim> q_dummy (q_points);

    // Now loop over all cells and enquire the support points on each
    // of these. we use dummy quadrature formulas where the quadrature
    // points are located at the unit support points to enquire the
    // location of the support points in real space.
    //
    // The weights of the quadrature rule have been set to invalid
    // values by the used constructor.
    dealii::FEValues<dim, spacedim> fe_values(mapping, fe, q_dummy, dealii::update_quadrature_points);

    typename dealii::DoFHandler<dim,spacedim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                                    endc = dof_handler.end();

    std::vector<dealii::types::global_dof_index> local_dof_indices;
    for (; cell != endc; ++cell)
        // only work on locally relevant cells
        if (!cell->is_artificial())
        {
            fe_values.reinit(cell);

            local_dof_indices.resize(cell->get_fe().dofs_per_cell);
            cell->get_dof_indices(local_dof_indices);

            const std::vector<dealii::Point<spacedim>> &points = fe_values.get_quadrature_points();

            for (unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
                // insert the values into the map
                if (active_system_indices[i])
                    support_points[local_dof_indices[i]] = points[i];
        }
}



namespace MPI
{

//FIXME use this for band and bor: #if !defined(DEAL_II_WITH_MPI) && !defined(DEAL_II_WITH_PETSC)
// Bitwise AND performed for t over all processors.
template <typename T>
inline
T band (const T &t,
       const MPI_Comm &mpi_communicator)
{
  T return_value;
  dealii::Utilities::MPI::internal::all_reduce (
          MPI_BAND, dealii::ArrayView<const T>(&t,1),
          mpi_communicator, dealii::ArrayView<T>(&return_value,1));
  return return_value;
}



// Bitwise OR performed for t over all processors.
template <typename T>
inline
T bor (const T &t,
       const MPI_Comm &mpi_communicator)
{
  T return_value;
  dealii::Utilities::MPI::internal::all_reduce (
          MPI_BOR, dealii::ArrayView<const T>(&t,1),
          mpi_communicator, dealii::ArrayView<T>(&return_value,1));
  return return_value;
}



// Get the rank of the root process.
inline
unsigned int
root_mpi_process (const MPI_Comm &mpi_communicator = MPI_COMM_WORLD)
{
    return 0;
}



// Check if a process is the root process (i.e. has rank 0).
inline
bool
is_root (const MPI_Comm &mpi_communicator)
{
    return dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == root_mpi_process (mpi_communicator);
}



// The processor with rank root creates all directories
// path which do not exist, yet. The path is broadcasetd
// to all other processor.
inline
void
create_directories (boost::filesystem::path& path,
                    const MPI_Comm &mpi_communicator,
                    const unsigned int root)
{
    unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process (mpi_communicator);
    unsigned int n_mpi_processes  = dealii::Utilities::MPI::n_mpi_processes (mpi_communicator);

    std::map<unsigned int, std::string> objects_to_send;

    if (this_mpi_process == root)
    {
        if (!boost::filesystem::exists(path))
            AssertThrow (boost::filesystem::create_directories (path),
                         dealii::ExcMessage ("Error creating directories."));

        for(unsigned int i = 1; i < n_mpi_processes; ++i)
            objects_to_send.insert (std::make_pair(i, path.string()));
    }

    auto objects_to_recv = dealii::Utilities::MPI::some_to_some (mpi_communicator, objects_to_send);

    if (this_mpi_process != root)
        path = objects_to_recv[0];
}



inline
void
create_directories (boost::filesystem::path& path,
       const MPI_Comm &mpi_communicator = MPI_COMM_WORLD)
{
    create_directories (path,mpi_communicator,MPI::root_mpi_process(mpi_communicator));
}



// Processor root creates a file if it does
// not exist. The path used by the root processor
// is broadcasted to all other processors.
inline
void
create_file (boost::filesystem::path& path,
             const MPI_Comm         &mpi_communicator,
             const unsigned int      root)
{
    unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process (mpi_communicator);
    unsigned int n_mpi_processes  = dealii::Utilities::MPI::n_mpi_processes (mpi_communicator);

    AssertThrow (path.has_filename(), dealii::ExcMessage ("Error creating file" ));

    std::map<unsigned int, std::string> objects_to_send;

    if (this_mpi_process == root)
    {
        if (path.has_parent_path() && !boost::filesystem::exists(path.parent_path()))
            AssertThrow (boost::filesystem::create_directories (path.parent_path()),
                         dealii::ExcMessage ("Error creating directories."));

        std::ofstream(path.string());

        AssertThrow (boost::filesystem::exists(path), dealii::ExcMessage ("Error creating file" ));

        for(unsigned int i = 1; i < n_mpi_processes; ++i)
            objects_to_send.insert (std::make_pair(i, path.string()));
    }

    auto objects_to_recv = dealii::Utilities::MPI::some_to_some (mpi_communicator, objects_to_send);

    if (this_mpi_process != root)
        path = objects_to_recv[0];
}


// Processor with rank MPI::root_mpi_process(mpi_communicator)
// creates a file if it does not exist. The path used by the
// root processor is broadcasted to all other processors.
inline
void
create_file (boost::filesystem::path &path,
             const MPI_Comm          &mpi_communicator = MPI_COMM_WORLD)
{
    create_file (path,mpi_communicator,MPI::root_mpi_process(mpi_communicator));
}



// Processor with rank MPI::root_mpi_process(mpi_communicator)
// creates a file if it does not exist. The path used by the
// root processor is broadcasted to all other processors.
inline
void
create_file (std::string    &path_to_file,
             const MPI_Comm &mpi_communicator = MPI_COMM_WORLD)
{
    boost::filesystem::path path(path_to_file);
    create_file (path,mpi_communicator,MPI::root_mpi_process(mpi_communicator));
    path_to_file = path.string();
}

}// namespace MPI

}//close namespace


#endif /* SRC_MYLIB_INCLUDE_EFI_BASE_UTILITY_H_ */
