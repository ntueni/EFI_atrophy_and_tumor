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

#ifndef SRC_MYLIB_INCLUDE_EFI_BASE_LOGSTREAM_H_
#define SRC_MYLIB_INCLUDE_EFI_BASE_LOGSTREAM_H_

// stl headers
#include <ostream>

// deal.II headers
#include <deal.II/base/logstream.h>


namespace efi {


// Verbosity levels
// -q or --quiet      0   quiet
// (none)             1   normal
// -v                 2   verbose
// -vv                3   very_verbose
// -vvv               4   debug
enum Verbosity {quiet        = 0,
                normal       = 1,
                verbose      = 2,
                very_verbose = 3,
                debug        = 4};

// Cast a valid string (see the list below) to a
// verbosity level. If an invalid string is given
// then the function returns Verbosity::quiet.
//
// lvl   string
//-------------
//  0    quiet
//  1    normal
//  2    verbose
//  3    very_verbose
//  4    debug
Verbosity
string_to_verbosity_level (const std::string &str);



// some prototype
namespace efi_internal {
class LogStreamWrapper;
}// namespace efi_internal



// Create an LogStreamWrapper with a certain verbosity level.
// This is just a wrapper to the deallog object but
// allows to directly set the verbosity level without the
// need to create dealii::LogStream::Prefix objects.
// Using this function allows to write streams in a
// natural way, e.g.
//
// efilog(Verbosity::normal) << "say hello"
//                           << "say goodbye"
//                           << std::endl;
efi_internal::LogStreamWrapper
efilog(int verbosity_level);



namespace efi_internal {


// Create a wrapper to the dealii::LogStream.
// Everything streamed to this wrapper class
// is finally streamed to the static
// dealii::deallog object.
class LogStreamWrapper
{
public:

    // Move constructor
    LogStreamWrapper (LogStreamWrapper&&) = default;

    // Copy constructor is forbidden
    LogStreamWrapper(const LogStreamWrapper&) = delete;

    // Copy assignment is forbidden
    LogStreamWrapper&
    operator= (const LogStreamWrapper&) = delete;

    // Stream operator
    LogStreamWrapper &
    operator<<(std::ostream &(*p)(std::ostream &));

    // Stream operator
    LogStreamWrapper &
    operator<<(const char* t);

private:

    // Constructor. It is private such that only
    // the friend function efilog can create such an
    // object.
    LogStreamWrapper (int verbosity_level);

    // Store the prefixes from which deallog determines
    // what is written to the logstream.
    std::vector<std::unique_ptr<dealii::LogStream::Prefix>> prefixes;

    // make the constructor function a friend class.
    friend
    LogStreamWrapper
    efi::efilog(int l);
};

}// namespace internanl



// Stream operator, taking an rvalue reference
// to a LogStreamWrapper as input. Everything
// streamed using this function ends up in
// dealii::deallog.
template <typename T>
inline
efi_internal::LogStreamWrapper &
operator<<(efi_internal::LogStreamWrapper &&log, const T &t);



// Stream operator, taking an lvalue reference
// to a LogStreamWrapper as input. Everything
// streamed using this function ends up in
// dealii::deallog.
template <typename T>
efi_internal::LogStreamWrapper &
operator<<(efi_internal::LogStreamWrapper &log, const T &t);



//---------------------- INLINE AND TEMPLATE FUNCTIONS -----------------------///



inline
Verbosity
string_to_verbosity_level (const std::string &str)
{
    if (str == "normal")
        return Verbosity::normal;
    else if (str == "verbose")
        return Verbosity::verbose;
    else if (str == "very verbose")
        return Verbosity::very_verbose;
    else if (str == "debug")
        return Verbosity::debug;
    else
        return Verbosity::quiet;
}



inline
efi_internal::
LogStreamWrapper::
LogStreamWrapper (int verbosity_level)
:
    prefixes (std::max(verbosity_level-1,0))
{
    AssertThrow (!((verbosity_level<0)||(verbosity_level > 4)),
            dealii::ExcIndexRange(verbosity_level,0,4));

    for (auto &prefix : this->prefixes)
        prefix.reset(new dealii::LogStream::Prefix(std::string(4,' ')));
}



inline
efi_internal::LogStreamWrapper &
efi_internal::
LogStreamWrapper::
operator<<(std::ostream &(*p)(std::ostream &))
{
    dealii::deallog<< p;
    return *this;
}



inline
efi_internal::LogStreamWrapper &
efi_internal::
LogStreamWrapper::
operator<<(const char* t)
{
  // print to the internal stringstream
  dealii::deallog << t;
  return  *this;
}



template <typename T>
inline
efi_internal::LogStreamWrapper &
operator<<(efi_internal::LogStreamWrapper &&log, const T &t)
{
    // print to the internal stringstream
    dealii::deallog << t;
    return log;
}



template <typename T>
inline
efi_internal::LogStreamWrapper &
operator<<(efi_internal::LogStreamWrapper &log, const T &t)
{
    // print to the internal stringstream
    dealii::deallog << t;
    return log;
}



inline
efi_internal::LogStreamWrapper
efilog(int verbosity_level)
{
    return efi_internal::LogStreamWrapper(verbosity_level);
}

}// namespace efi



#endif /* SRC_MYLIB_INCLUDE_EFI_BASE_LOGSTREAM_H_ */
