cmake_minimum_required(VERSION 3.22)
project(spectral_flups_ns LANGUAGES C CXX)

include(ExternalProject)
include(CTest)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(SPECTRAL_USE_FLUPS_BACKEND "Use FLUPS instead of the direct FFTW backend" OFF)

find_package(MPI REQUIRED COMPONENTS C CXX)

find_path(FFTW_INCLUDE_DIR fftw3.h REQUIRED)
find_library(FFTW_LIBRARY NAMES fftw3 REQUIRED)
find_path(HDF5_INCLUDE_DIR hdf5.h REQUIRED)
find_library(HDF5_LIBRARY NAMES hdf5 REQUIRED)
get_filename_component(FFTW_LIBRARY_DIR ${FFTW_LIBRARY} DIRECTORY)
get_filename_component(HDF5_LIBRARY_DIR ${HDF5_LIBRARY} DIRECTORY)

set(THIRD_PARTY_PREFIX ${CMAKE_BINARY_DIR}/third_party/install)
set(H3LPR_INSTALL_DIR ${THIRD_PARTY_PREFIX}/h3lpr)
set(FLUPS_INSTALL_DIR ${THIRD_PARTY_PREFIX}/flups)
set(THIRD_PARTY_ARCH_DIR ${CMAKE_BINARY_DIR}/third_party/arch)
file(MAKE_DIRECTORY ${THIRD_PARTY_ARCH_DIR})

file(WRITE ${THIRD_PARTY_ARCH_DIR}/h3lpr.cmake.make
"CXX=${MPI_CXX_COMPILER}
CC=${MPI_C_COMPILER}
CXXFLAGS=-O3 -DNDEBUG -std=c++17 -fPIC -fno-omit-frame-pointer
LDFLAGS=
")

file(WRITE ${THIRD_PARTY_ARCH_DIR}/flups.cmake.make
"CXX=${MPI_CXX_COMPILER}
CC=${MPI_C_COMPILER}
CXXFLAGS=-O3 -DNDEBUG -std=c++17 -fPIC -fno-omit-frame-pointer -DWITHOUT_OPENMP -DPROF
CCFLAGS=-O3 -DNDEBUG -std=c99 -fPIC -fno-omit-frame-pointer
LDFLAGS=
H3LPR_DIR=${H3LPR_INSTALL_DIR}
H3LPR_INC=${H3LPR_INSTALL_DIR}/include
H3LPR_LIB=${H3LPR_INSTALL_DIR}/lib
H3LPR_LIBNAME=-lh3lpr
FFTW_INC=${FFTW_INCLUDE_DIR}
FFTW_LIB=${FFTW_LIBRARY_DIR}
FFTW_LIBNAME=-lfftw3
HDF5_INC=${HDF5_INCLUDE_DIR}
HDF5_LIB=${HDF5_LIBRARY_DIR}
HDF5_LIBNAME=-lhdf5
OPTS=-DWITHOUT_OPENMP -DPROF
")

ExternalProject_Add(h3lpr_ep
  GIT_REPOSITORY https://github.com/vanreeslab/h3lpr.git
  GIT_TAG main
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} ARCH_FILE=${THIRD_PARTY_ARCH_DIR}/h3lpr.cmake.make PREFIX=${H3LPR_INSTALL_DIR} lib_static
  BUILD_IN_SOURCE TRUE
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E make_directory ${H3LPR_INSTALL_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E make_directory ${H3LPR_INSTALL_DIR}/include/h3lpr
    COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/h3lpr.a ${H3LPR_INSTALL_DIR}/lib/libh3lpr.a
    COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/src/macros.hpp <SOURCE_DIR>/src/parser.hpp <SOURCE_DIR>/src/profiler.hpp <SOURCE_DIR>/src/ptr.hpp ${H3LPR_INSTALL_DIR}/include/h3lpr
)

ExternalProject_Add(flups_ep
  GIT_REPOSITORY https://github.com/vortexlab-uclouvain/flups.git
  GIT_TAG release
  DEPENDS h3lpr_ep
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} ARCH_FILE=${THIRD_PARTY_ARCH_DIR}/flups.cmake.make PREFIX=${FLUPS_INSTALL_DIR} install_static
  BUILD_IN_SOURCE TRUE
  INSTALL_COMMAND ""
)

add_library(spectral_ns
  src/fftw_fft.cpp
  src/flups_fft.cpp
  src/spectral_solver.cpp
)
add_dependencies(spectral_ns flups_ep)
target_include_directories(spectral_ns
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${FLUPS_INSTALL_DIR}/include
    ${H3LPR_INSTALL_DIR}/include
    ${FFTW_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIR}
)
target_link_directories(spectral_ns PUBLIC ${FLUPS_INSTALL_DIR}/lib ${H3LPR_INSTALL_DIR}/lib)
target_link_libraries(spectral_ns
  PUBLIC
    flups_a2a
    h3lpr
    MPI::MPI_CXX
    ${FFTW_LIBRARY}
    ${HDF5_LIBRARY}
)
if(SPECTRAL_USE_FLUPS_BACKEND)
  target_compile_definitions(spectral_ns PUBLIC SPECTRAL_USE_FLUPS_BACKEND)
endif()
target_compile_options(spectral_ns PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG -fno-omit-frame-pointer>)

add_executable(taylor_green_flups src/main.cpp)
target_link_libraries(taylor_green_flups PRIVATE spectral_ns)
target_compile_options(taylor_green_flups PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG -fno-omit-frame-pointer>)

if(BUILD_TESTING)
  add_executable(spectral_ns_tests cpp_tests/spectral_ns_tests.cpp)
  target_link_libraries(spectral_ns_tests PRIVATE spectral_ns)
  target_compile_options(spectral_ns_tests PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG -fno-omit-frame-pointer>)
  add_test(NAME spectral_ns_tests COMMAND spectral_ns_tests)
  if(SPECTRAL_USE_FLUPS_BACKEND)
    add_test(NAME spectral_ns_tests_mpi2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:spectral_ns_tests>)
  endif()
endif()
