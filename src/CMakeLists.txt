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

CMAKE_MINIMUM_REQUIRED( VERSION 3.9 FATAL_ERROR )

FIND_PACKAGE( deal.II 9.1.1 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF( NOT ${deal.II_FOUND} )
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

INCLUDE( ProcessorCount )
ProcessorCount(N)

IF(NOT N EQUAL 0)
	
  ADD_CUSTOM_TARGET( debug
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all -- -j${N}
    COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
    )

  ADD_CUSTOM_TARGET( release
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all -- -j${N} 
    COMMENT "Switch CMAKE_BUILD_TYPE to Release"
    )

  INCLUDE( ${DEAL_II_FEATURE_CONFIG} )

  PROJECT( VirtualLabratory )

  SET( CMAKE_C_COMPILER mpicc )
  SET( CMAKE_CXX_COMPILER mpicxx )
  SET( CMAKE_CXX_STANDARD 14 )
  #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -lboost_filesystem -lboost_system -fext-numeric-literals")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  ## Binaries will be saved in given directory
  #SET( PROJECT_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin )
  ## executables will be saved in given directory
  #SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR} )

  ## Add the efi library
  ADD_SUBDIRECTORY( efi )

  SET( EFI_VLAB_src
       ${CMAKE_CURRENT_SOURCE_DIR}/main.cc )

  ADD_EXECUTABLE( efi_vlab ${EFI_VLAB_src} )

  TARGET_INCLUDE_DIRECTORIES( 
      efi_vlab 
      PUBLIC	
      ${CMAKE_CURRENT_SOURCE_DIR}/efi/include )

  DEAL_II_SETUP_TARGET( efi_vlab )

  TARGET_LINK_LIBRARIES(efi_vlab efi)
	

ENDIF()

message(STATUS "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
