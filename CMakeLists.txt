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
set(AVA_TARGET "CPU" CACHE STRING "AVA execution target for the standalone AVA solver: CPU or CUDA")
set_property(CACHE AVA_TARGET PROPERTY STRINGS CPU CUDA)
set(SPECTRAL_FFT_BACKEND "FFTW" CACHE STRING "FFT backend for the AVA taylor_green executable: FFTW, FLUPS, CUFFT, or CUFFTMP")
set_property(CACHE SPECTRAL_FFT_BACKEND PROPERTY STRINGS FFTW FLUPS CUFFT CUFFTMP)
set(SPECTRAL_PRECISION "DOUBLE" CACHE STRING "Floating point precision for the AVA taylor_green solver: DOUBLE or SINGLE")
set_property(CACHE SPECTRAL_PRECISION PROPERTY STRINGS DOUBLE SINGLE)
if(SPECTRAL_USE_FLUPS_BACKEND AND SPECTRAL_FFT_BACKEND STREQUAL "FFTW")
  set(SPECTRAL_FFT_BACKEND "FLUPS" CACHE STRING "FFT backend for the AVA taylor_green executable: FFTW, FLUPS, or CUFFT" FORCE)
endif()
if(NOT SPECTRAL_PRECISION STREQUAL "DOUBLE" AND NOT SPECTRAL_PRECISION STREQUAL "SINGLE")
  message(FATAL_ERROR "SPECTRAL_PRECISION must be DOUBLE or SINGLE")
endif()
if(SPECTRAL_PRECISION STREQUAL "SINGLE" AND SPECTRAL_FFT_BACKEND STREQUAL "FLUPS")
  message(FATAL_ERROR "SPECTRAL_PRECISION=SINGLE is not supported with FLUPS: the current FLUPS wrapper exposes double buffers.")
endif()

set(SPECTRAL_FLUPS_GIT_REPOSITORY "https://github.com/vortexlab-uclouvain/flups.git" CACHE STRING "FLUPS git repository to build")
set(SPECTRAL_FLUPS_GIT_TAG "release" CACHE STRING "FLUPS git tag or branch to build")
set(SPECTRAL_FLUPS_SOURCE_DIR "" CACHE PATH "Optional existing FLUPS source checkout; skips ExternalProject git clone when set")
set(SPECTRAL_FLUPS_ENABLE_CUDA_DEFAULT OFF)
if(AVA_TARGET STREQUAL "CUDA" AND SPECTRAL_FFT_BACKEND STREQUAL "FLUPS")
  if(SPECTRAL_FLUPS_GIT_REPOSITORY STREQUAL "https://github.com/vortexlab-uclouvain/flups.git")
    set(SPECTRAL_FLUPS_GIT_REPOSITORY "git@git.immc.ucl.ac.be:examples/flups.git" CACHE STRING "FLUPS git repository to build" FORCE)
  endif()
  if(SPECTRAL_FLUPS_GIT_TAG STREQUAL "release")
    set(SPECTRAL_FLUPS_GIT_TAG "dev-gpu" CACHE STRING "FLUPS git tag or branch to build" FORCE)
  endif()
  set(SPECTRAL_FLUPS_ENABLE_CUDA_DEFAULT ON)
endif()
option(SPECTRAL_FLUPS_ENABLE_CUDA "Build FLUPS with its CUDA/cuFFT backend when the selected FLUPS source supports it" ${SPECTRAL_FLUPS_ENABLE_CUDA_DEFAULT})
option(SPECTRAL_THIRD_PARTY_UPDATE_DISCONNECTED "Skip git update steps for ExternalProject dependencies after the initial clone" ON)
set(SPECTRAL_ASSUME_CUDA_AWARE_MPI_DEFAULT OFF)
if(DEFINED ENV{SPECTRAL_ASSUME_CUDA_AWARE_MPI})
  set(SPECTRAL_ASSUME_CUDA_AWARE_MPI_DEFAULT "$ENV{SPECTRAL_ASSUME_CUDA_AWARE_MPI}")
endif()
option(SPECTRAL_ASSUME_CUDA_AWARE_MPI "Allow CUDA+FLUPS builds with the selected MPI; required because FLUPS CUDA passes device buffers to MPI" ${SPECTRAL_ASSUME_CUDA_AWARE_MPI_DEFAULT})
set(SPECTRAL_FLUPS_CUDA_OPTS "-DBACKEND_CUDA -DHAVE_CUFFT -Dm_hipfft_check_errors=m_cufft_check_errors" CACHE STRING "Extra preprocessor options passed to the FLUPS Makefile when SPECTRAL_FLUPS_ENABLE_CUDA=ON")
set(SPECTRAL_FLUPS_CUDA_LIBNAME "-lcufft -lcublas -lcudart" CACHE STRING "CUDA libraries passed to the FLUPS Makefile when SPECTRAL_FLUPS_ENABLE_CUDA=ON")
set(SPECTRAL_CUFFTMP_ROOT "" CACHE PATH "Optional cuFFTMp installation prefix containing include/cufftMp.h and lib/libcufftMp.so")

if(AVA_TARGET STREQUAL "CUDA" AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

set(THIRD_PARTY_PREFIX ${CMAKE_BINARY_DIR}/third_party/install)

find_package(MPI REQUIRED COMPONENTS C CXX)

find_path(FFTW_INCLUDE_DIR fftw3.h REQUIRED)
if(SPECTRAL_PRECISION STREQUAL "SINGLE" AND SPECTRAL_FFT_BACKEND STREQUAL "FFTW")
  find_library(FFTW_LIBRARY NAMES fftw3f REQUIRED)
else()
  find_library(FFTW_LIBRARY NAMES fftw3 REQUIRED)
endif()
find_path(HDF5_INCLUDE_DIR hdf5.h REQUIRED)
find_library(HDF5_LIBRARY NAMES hdf5 REQUIRED)
get_filename_component(FFTW_LIBRARY_DIR ${FFTW_LIBRARY} DIRECTORY)
get_filename_component(HDF5_LIBRARY_DIR ${HDF5_LIBRARY} DIRECTORY)
if(SPECTRAL_FLUPS_ENABLE_CUDA)
  if(NOT SPECTRAL_ASSUME_CUDA_AWARE_MPI)
    message(FATAL_ERROR
      "SPECTRAL_FLUPS_ENABLE_CUDA=ON requires a CUDA-aware MPI: FLUPS CUDA passes "
      "device buffers to MPI. Configure with -DSPECTRAL_ASSUME_CUDA_AWARE_MPI=ON "
      "only when the selected MPI supports CUDA device pointers.")
  endif()
  find_package(CUDAToolkit REQUIRED)
  get_filename_component(CUDAToolkit_ROOT_DIR "${CUDAToolkit_BIN_DIR}/.." ABSOLUTE)
  set(CUDAToolkit_LIBRARY_DIR "${CUDAToolkit_ROOT_DIR}/lib")
  list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDAToolkit_PRIMARY_INCLUDE_DIR)
  set(SPECTRAL_CUDATOOLKIT_INCLUDE_FLAGS "")
  foreach(_cuda_include IN LISTS CUDAToolkit_INCLUDE_DIRS)
    string(APPEND SPECTRAL_CUDATOOLKIT_INCLUDE_FLAGS " -I${_cuda_include}")
  endforeach()
  set(SPECTRAL_FLUPS_CUDA_ARCH_FLAGS "")
  if(CMAKE_CUDA_ARCHITECTURES)
    list(GET CMAKE_CUDA_ARCHITECTURES 0 _spectral_cuda_arch)
    string(REGEX REPLACE "-(real|virtual)$" "" _spectral_cuda_arch "${_spectral_cuda_arch}")
    if(_spectral_cuda_arch MATCHES "^[0-9]+$")
      set(SPECTRAL_FLUPS_CUDA_ARCH_FLAGS "-arch=sm_${_spectral_cuda_arch}")
    endif()
  endif()
endif()

set(CUFFTMP_LIBRARY_DIRS "")
set(CUFFTMP_INCLUDE_DIRS "")
set(CUFFTMP_DEPENDS "")
if(SPECTRAL_FFT_BACKEND STREQUAL "CUFFTMP")
  find_package(CUDAToolkit REQUIRED)

  set(CUFFTMP_SEARCH_PREFIXES "")
  foreach(_root IN ITEMS
      "${SPECTRAL_CUFFTMP_ROOT}"
      "$ENV{CUFFTMP_HOME}"
      "$ENV{NVHPC_ROOT}"
      "$ENV{NVSHMEM_HOME}")
    if(_root)
      list(APPEND CUFFTMP_SEARCH_PREFIXES "${_root}")
      file(GLOB _nvhpc_versions LIST_DIRECTORIES TRUE "${_root}/Linux_x86_64/*")
      if(_nvhpc_versions)
        list(APPEND CUFFTMP_SEARCH_PREFIXES ${_nvhpc_versions})
      endif()
    endif()
  endforeach()
  list(REMOVE_DUPLICATES CUFFTMP_SEARCH_PREFIXES)

  if(CUFFTMP_SEARCH_PREFIXES)
    list(APPEND CMAKE_PREFIX_PATH ${CUFFTMP_SEARCH_PREFIXES})
  endif()

  find_path(CUFFTMP_INCLUDE_DIR cufftMp.h
    HINTS ${CUFFTMP_SEARCH_PREFIXES}
    PATH_SUFFIXES
      include
      math_libs/include
      math_libs/12.2/targets/x86_64-linux/include/cufftmp
      math_libs/12.2/targets/x86_64-linux/include
      math_libs/12.8/targets/x86_64-linux/include/cufftmp
      math_libs/12.8/targets/x86_64-linux/include
      math_libs/11.8/targets/x86_64-linux/include/cufftmp
      math_libs/11.8/targets/x86_64-linux/include
  )
  find_library(CUFFTMP_LIBRARY NAMES cufftMp
    HINTS ${CUFFTMP_SEARCH_PREFIXES}
    PATH_SUFFIXES
      lib
      lib64
      math_libs/lib
      math_libs/lib64
      math_libs/12.2/targets/x86_64-linux/lib
      math_libs/12.8/targets/x86_64-linux/lib
      math_libs/11.8/targets/x86_64-linux/lib
  )
  find_library(CUFFTMP_NVSHMEM_HOST_LIBRARY NAMES nvshmem_host
    HINTS ${CUFFTMP_SEARCH_PREFIXES}
    PATH_SUFFIXES
      lib
      lib64
      comm_libs/nvshmem/lib
      comm_libs/12.2/nvshmem_cufftmp_compat/lib
      comm_libs/12.2/nvshmem/lib
      comm_libs/12.8/nvshmem_cufftmp_compat/lib
      comm_libs/12.8/nvshmem/lib
      comm_libs/11.8/nvshmem_cufftmp_compat/lib
      comm_libs/11.8/nvshmem/lib
  )
  if(CUFFTMP_INCLUDE_DIR AND CUFFTMP_LIBRARY AND CUFFTMP_NVSHMEM_HOST_LIBRARY)
    get_filename_component(_cufftmp_library_dir "${CUFFTMP_LIBRARY}" DIRECTORY)
    get_filename_component(_nvshmem_library_dir "${CUFFTMP_NVSHMEM_HOST_LIBRARY}" DIRECTORY)
    list(APPEND CUFFTMP_INCLUDE_DIRS "${CUFFTMP_INCLUDE_DIR}")
    list(APPEND CUFFTMP_LIBRARY_DIRS "${_cufftmp_library_dir}" "${_nvshmem_library_dir}")
    message(STATUS "Found cuFFTMp include: ${CUFFTMP_INCLUDE_DIR}")
    message(STATUS "Found cuFFTMp library: ${CUFFTMP_LIBRARY}")
    message(STATUS "Found NVSHMEM host library: ${CUFFTMP_NVSHMEM_HOST_LIBRARY}")
  else()
    message(FATAL_ERROR
      "SPECTRAL_FFT_BACKEND=CUFFTMP requires cuFFTMp. Set SPECTRAL_CUFFTMP_ROOT, "
      "CUFFTMP_HOME, NVHPC_ROOT, or NVSHMEM_HOME to an existing cuFFTMp/NVIDIA HPC SDK install. "
      "Searched prefixes: ${CUFFTMP_SEARCH_PREFIXES}")
  endif()
endif()

set(H3LPR_INSTALL_DIR ${THIRD_PARTY_PREFIX}/h3lpr)
set(FLUPS_INSTALL_DIR ${THIRD_PARTY_PREFIX}/flups)
set(AVA_SOURCE_DIR ${CMAKE_BINARY_DIR}/third_party/ava)
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

if(SPECTRAL_FLUPS_ENABLE_CUDA)
  file(APPEND ${THIRD_PARTY_ARCH_DIR}/flups.cmake.make
"CXX=${CUDAToolkit_NVCC_EXECUTABLE}
CXXFLAGS+=-I${MPI_CXX_COMPILER_INCLUDE_DIRS}${SPECTRAL_CUDATOOLKIT_INCLUDE_FLAGS} ${SPECTRAL_FLUPS_CUDA_ARCH_FLAGS} --forward-unknown-to-host-compiler --extended-lambda --expt-relaxed-constexpr
CUDA_INC=${CUDAToolkit_PRIMARY_INCLUDE_DIR}
CUDA_LIB=${CUDAToolkit_LIBRARY_DIR}
CUDA_LIBNAME=${SPECTRAL_FLUPS_CUDA_LIBNAME}
OPTS+=${SPECTRAL_FLUPS_CUDA_OPTS}
LDFLAGS+=-L${CUDAToolkit_LIBRARY_DIR} ${SPECTRAL_FLUPS_CUDA_LIBNAME}
")
endif()

ExternalProject_Add(h3lpr_ep
  GIT_REPOSITORY https://github.com/vanreeslab/h3lpr.git
  GIT_TAG main
  UPDATE_DISCONNECTED ${SPECTRAL_THIRD_PARTY_UPDATE_DISCONNECTED}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} ARCH_FILE=${THIRD_PARTY_ARCH_DIR}/h3lpr.cmake.make PREFIX=${H3LPR_INSTALL_DIR} lib_static
  BUILD_IN_SOURCE TRUE
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E make_directory ${H3LPR_INSTALL_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E make_directory ${H3LPR_INSTALL_DIR}/include/h3lpr
    COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/h3lpr.a ${H3LPR_INSTALL_DIR}/lib/libh3lpr.a
    COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/src/macros.hpp <SOURCE_DIR>/src/parser.hpp <SOURCE_DIR>/src/profiler.hpp <SOURCE_DIR>/src/ptr.hpp ${H3LPR_INSTALL_DIR}/include/h3lpr
)

if(SPECTRAL_FLUPS_SOURCE_DIR)
  ExternalProject_Add(flups_ep
    SOURCE_DIR ${SPECTRAL_FLUPS_SOURCE_DIR}
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    DEPENDS h3lpr_ep
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
      -DFLUPS_SOURCE_DIR=<SOURCE_DIR>
      -DSPECTRAL_FLUPS_ENABLE_CUDA=${SPECTRAL_FLUPS_ENABLE_CUDA}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckFlupsCuda.cmake
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} ARCH_FILE=${THIRD_PARTY_ARCH_DIR}/flups.cmake.make PREFIX=${FLUPS_INSTALL_DIR} install_static
    BUILD_IN_SOURCE TRUE
    INSTALL_COMMAND ""
  )
else()
  ExternalProject_Add(flups_ep
    GIT_REPOSITORY ${SPECTRAL_FLUPS_GIT_REPOSITORY}
    GIT_TAG ${SPECTRAL_FLUPS_GIT_TAG}
    UPDATE_DISCONNECTED ${SPECTRAL_THIRD_PARTY_UPDATE_DISCONNECTED}
    DEPENDS h3lpr_ep
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
      -DFLUPS_SOURCE_DIR=<SOURCE_DIR>
      -DSPECTRAL_FLUPS_ENABLE_CUDA=${SPECTRAL_FLUPS_ENABLE_CUDA}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckFlupsCuda.cmake
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} ARCH_FILE=${THIRD_PARTY_ARCH_DIR}/flups.cmake.make PREFIX=${FLUPS_INSTALL_DIR} install_static
    BUILD_IN_SOURCE TRUE
    INSTALL_COMMAND ""
  )
endif()

ExternalProject_Add(ava_ep
  GIT_REPOSITORY https://git.immc.ucl.ac.be/mema/ava.git
  GIT_TAG main
  SOURCE_DIR ${AVA_SOURCE_DIR}
  UPDATE_DISCONNECTED ${SPECTRAL_THIRD_PARTY_UPDATE_DISCONNECTED}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
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

set(AVA_INCLUDE_DIRS
  ${AVA_SOURCE_DIR}
  ${AVA_SOURCE_DIR}/atomic
  ${AVA_SOURCE_DIR}/reduce
  ${AVA_SOURCE_DIR}/scan
  ${AVA_SOURCE_DIR}/sort
  ${AVA_SOURCE_DIR}/select
  ${AVA_SOURCE_DIR}/bits
  ${AVA_SOURCE_DIR}/iterator
)

set(SPECTRAL_PRECISION_DEFINITION SPECTRAL_REAL_DOUBLE)
if(SPECTRAL_PRECISION STREQUAL "SINGLE")
  set(SPECTRAL_PRECISION_DEFINITION SPECTRAL_REAL_FLOAT)
endif()

function(configure_taylor_green_common target)
  add_dependencies(${target} ava_ep)
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${AVA_INCLUDE_DIRS}
  )
  target_compile_definitions(${target} PRIVATE ${SPECTRAL_PRECISION_DEFINITION})
endfunction()

function(configure_taylor_green_flups target)
  add_dependencies(${target} flups_ep)
  target_sources(${target} PRIVATE src/flups_fft.cpp)
  target_compile_definitions(${target} PRIVATE SPECTRAL_USE_FLUPS_BACKEND)
  target_include_directories(${target} PRIVATE
    ${FLUPS_INSTALL_DIR}/include
    ${H3LPR_INSTALL_DIR}/include
    ${HDF5_INCLUDE_DIR}
  )
  target_link_directories(${target} PRIVATE ${FLUPS_INSTALL_DIR}/lib ${H3LPR_INSTALL_DIR}/lib)
  target_link_libraries(${target} PRIVATE flups_a2a h3lpr ${HDF5_LIBRARY} ${FFTW_LIBRARY})
  if(SPECTRAL_FLUPS_ENABLE_CUDA)
    set_source_files_properties(src/flups_fft.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(${target} PRIVATE BACKEND_CUDA HAVE_CUFFT)
    target_link_libraries(${target} PRIVATE CUDA::cufft CUDA::cublas)
  endif()
endfunction()

if(AVA_TARGET STREQUAL "CUDA")
  if(SPECTRAL_FFT_BACKEND STREQUAL "FFTW")
    message(FATAL_ERROR "SPECTRAL_FFT_BACKEND=FFTW requires -DAVA_TARGET=CPU")
  elseif(NOT SPECTRAL_FFT_BACKEND STREQUAL "CUFFT" AND NOT SPECTRAL_FFT_BACKEND STREQUAL "FLUPS" AND NOT SPECTRAL_FFT_BACKEND STREQUAL "CUFFTMP")
    message(FATAL_ERROR "SPECTRAL_FFT_BACKEND must be FFTW, FLUPS, CUFFT, or CUFFTMP")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(taylor_green
    src/main_ava.cpp
    src/ava_taylor_green.cu
  )
  configure_taylor_green_common(taylor_green)
  target_compile_definitions(taylor_green PRIVATE AVA_TARGET_CUDA)
  target_compile_options(taylor_green PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --generate-line-info -diag-suppress=550>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG>
  )
  set_target_properties(taylor_green PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
  )
  target_link_libraries(taylor_green PRIVATE CUDA::cudart)
  if(SPECTRAL_FFT_BACKEND STREQUAL "CUFFT")
    target_compile_definitions(taylor_green PRIVATE SPECTRAL_USE_CUFFT)
    target_link_libraries(taylor_green PRIVATE CUDA::cufft)
  elseif(SPECTRAL_FFT_BACKEND STREQUAL "CUFFTMP")
    if(CUFFTMP_DEPENDS)
      add_dependencies(taylor_green ${CUFFTMP_DEPENDS})
    endif()
    target_compile_definitions(taylor_green PRIVATE SPECTRAL_USE_CUFFTMP)
    target_include_directories(taylor_green PRIVATE ${CUFFTMP_INCLUDE_DIRS})
    target_link_directories(taylor_green PRIVATE ${CUFFTMP_LIBRARY_DIRS})
    if(CUFFTMP_LIBRARY AND CUFFTMP_NVSHMEM_HOST_LIBRARY)
      target_link_libraries(taylor_green PRIVATE ${CUFFTMP_LIBRARY} ${CUFFTMP_NVSHMEM_HOST_LIBRARY})
    else()
      target_link_libraries(taylor_green PRIVATE cufftMp nvshmem_host)
    endif()
    target_link_libraries(taylor_green PRIVATE MPI::MPI_CXX)
    target_link_options(taylor_green PRIVATE "LINKER:--allow-shlib-undefined")
  elseif(SPECTRAL_FFT_BACKEND STREQUAL "FLUPS")
    configure_taylor_green_flups(taylor_green)
    target_link_libraries(taylor_green PRIVATE MPI::MPI_CXX)
  endif()
elseif(AVA_TARGET STREQUAL "CPU")
  if(SPECTRAL_FFT_BACKEND STREQUAL "CUFFT" OR SPECTRAL_FFT_BACKEND STREQUAL "CUFFTMP")
    message(FATAL_ERROR "SPECTRAL_FFT_BACKEND=CUFFT/CUFFTMP requires -DAVA_TARGET=CUDA")
  endif()
  add_executable(taylor_green
    src/main_ava.cpp
    src/ava_taylor_green.cpp
  )
  configure_taylor_green_common(taylor_green)
  target_include_directories(taylor_green PRIVATE ${FFTW_INCLUDE_DIR})
  target_compile_definitions(taylor_green PRIVATE AVA_TARGET_CPU)
  target_compile_options(taylor_green PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG -fno-omit-frame-pointer>)
  target_link_libraries(taylor_green PRIVATE MPI::MPI_CXX ${FFTW_LIBRARY})
  if(SPECTRAL_FFT_BACKEND STREQUAL "FLUPS")
    configure_taylor_green_flups(taylor_green)
  elseif(NOT SPECTRAL_FFT_BACKEND STREQUAL "FFTW")
    message(FATAL_ERROR "SPECTRAL_FFT_BACKEND must be FFTW, FLUPS, CUFFT, or CUFFTMP")
  endif()
else()
  message(FATAL_ERROR "AVA_TARGET must be CPU or CUDA")
endif()

if(BUILD_TESTING)
  add_executable(spectral_ns_tests cpp_tests/spectral_ns_tests.cpp)
  target_link_libraries(spectral_ns_tests PRIVATE spectral_ns)
  target_compile_options(spectral_ns_tests PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG -fno-omit-frame-pointer>)
  add_test(NAME spectral_ns_tests COMMAND spectral_ns_tests)
  if(SPECTRAL_USE_FLUPS_BACKEND)
    add_test(NAME spectral_ns_tests_mpi2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:spectral_ns_tests>)
  endif()
endif()
