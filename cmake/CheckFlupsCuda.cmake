if(NOT SPECTRAL_FLUPS_ENABLE_CUDA)
  return()
endif()

file(GLOB_RECURSE flups_cuda_sources
  "${FLUPS_SOURCE_DIR}/*.cu"
  "${FLUPS_SOURCE_DIR}/*.cuh"
)

set(has_cuda_backend FALSE)
if(flups_cuda_sources)
  set(has_cuda_backend TRUE)
endif()

if(NOT has_cuda_backend)
  file(GLOB_RECURSE flups_source_files
    "${FLUPS_SOURCE_DIR}/Makefile"
    "${FLUPS_SOURCE_DIR}/src/*.cpp"
    "${FLUPS_SOURCE_DIR}/src/*.hpp"
    "${FLUPS_SOURCE_DIR}/src/*.h"
  )
  foreach(path IN LISTS flups_source_files)
    file(READ "${path}" content)
    if(content MATCHES "#[ \t]*include[^\n]*cufft|cufft[A-Za-z_0-9]*[ \t]*\\(")
      set(has_cuda_backend TRUE)
      break()
    endif()
  endforeach()
endif()

if(NOT has_cuda_backend)
  message(FATAL_ERROR
    "SPECTRAL_FLUPS_ENABLE_CUDA=ON was requested, but the selected FLUPS source "
    "tree does not appear to contain a CUDA/cuFFT backend. Set "
    "-DSPECTRAL_FLUPS_ENABLE_CUDA=OFF or select a CUDA-capable FLUPS branch/tag "
    "with -DSPECTRAL_FLUPS_GIT_TAG=<tag-or-branch>."
  )
endif()
