## Copyright (C) 2019 - 2020 by the emerging fields initiative 'Novel Biopolymer
## Hydrogels for Understanding Complex Soft Tissue Biomechanics' of the FAU
##
## This file is part of the EFI library.
##
## The EFI library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
##
## Author: Stefan Kaessmair

INCLUDE_DIRECTORIES(include)

FILE(GLOB MYFILES "source/*/*.cc"
                  "source/*/*/*.cc")

# DYNAMIC VERSION
ADD_LIBRARY(efi SHARED ${MYFILES})

DEAL_II_SETUP_TARGET(efi)

# find the static Boost libraries
set (BOOST_ROOT ${DEAL_II_BOOST_DIR})

set(BUILD_SHARED_LIBS ON)
set(Boost_USE_STATIC_LIBS OFF)

find_package (Boost REQUIRED COMPONENTS filesystem)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

TARGET_LINK_LIBRARIES(efi ${Boost_LIBRARIES}) 



# STATIC VERSION
#ADD_LIBRARY(mylib ${MYFILES})
#
#DEAL_II_SETUP_TARGET(mylib)
#
# find the static Boost libraries
#set (BOOST_ROOT ${DEAL_II_BOOST_DIR})
#
#find_package (Boost REQUIRED COMPONENTS filesystem)
#
#INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
#LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
#
#TARGET_LINK_LIBRARIES(mylib ${Boost_LIBRARIES}) 
