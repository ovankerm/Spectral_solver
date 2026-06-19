# CMake Build Paths

This project builds one runnable executable:

```bash
taylor_green
```

The solver is always AVA-based. CMake selects:

- `AVA_TARGET=CPU|CUDA`: where AVA kernels run.
- `SPECTRAL_FFT_BACKEND=FFTW|FLUPS|CUFFT|CUFFTMP`: which FFT backend is used.

## CUDA + cuFFT

Single-process GPU path. AVA kernels run on CUDA and FFT calls use cuFFT.

```bash
nix develop -c cmake -S . -B build-cuda \
  -DAVA_TARGET=CUDA \
  -DSPECTRAL_FFT_BACKEND=CUFFT \
  -DCMAKE_BUILD_TYPE=Release

nix develop -c cmake --build build-cuda --target taylor_green -j 8

nix develop -c ./build-cuda/taylor_green \
  --n 64 --re 1600 \
  --adaptive-cfl 1 --dt 0.1 --cfl 0.4 --t-end 20 \
  --output-dt 0.1 --print-every 20 \
  --output-dir output_cuda
```

## CPU + FFTW

Single-process CPU path. AVA kernels run on CPU and FFT calls use FFTW.

```bash
nix develop -c cmake -S . -B build-fftw \
  -DAVA_TARGET=CPU \
  -DSPECTRAL_FFT_BACKEND=FFTW \
  -DCMAKE_BUILD_TYPE=Release

nix develop -c cmake --build build-fftw --target taylor_green -j 8

nix develop -c ./build-fftw/taylor_green \
  --n 64 --re 1600 \
  --adaptive-cfl 1 --dt 0.1 --cfl 0.4 --t-end 20 \
  --output-dt 0.1 --print-every 20 \
  --output-dir output_fftw
```

## CPU + MPI + FLUPS

Distributed CPU path. AVA kernels run on each rank's local CPU chunks, and
FLUPS handles the distributed FFT and MPI communication.

```bash
nix develop -c cmake -S . -B build-flups-cpu \
  -DAVA_TARGET=CPU \
  -DSPECTRAL_FFT_BACKEND=FLUPS \
  -DCMAKE_BUILD_TYPE=Release

nix develop -c cmake --build build-flups-cpu --target taylor_green -j 8

nix develop -c mpirun -np 4 ./build-flups-cpu/taylor_green \
  --n 64 --re 1600 \
  --adaptive-cfl 1 --dt 0.1 --cfl 0.4 --t-end 20 \
  --output-dt 0.1 --print-every 20 \
  --output-dir output_flups_cpu
```

## CUDA + MPI + FLUPS

Distributed GPU path. AVA kernels run on CUDA and FLUPS is built from the
private CUDA-capable branch. The clone uses SSH and assumes the user has a
working key for `git.immc.ucl.ac.be`.

FLUPS CUDA passes device buffers to MPI, so this path requires a CUDA-aware MPI
implementation. The Nix shell uses `openmpi.override { cudaSupport = true; }`
and exports `SPECTRAL_ASSUME_CUDA_AWARE_MPI=ON`. Outside the Nix shell, pass
`-DSPECTRAL_ASSUME_CUDA_AWARE_MPI=ON` only when the MPI library really supports
CUDA device pointers.

When `AVA_TARGET=CUDA` and `SPECTRAL_FFT_BACKEND=FLUPS`, CMake automatically
selects:

- `SPECTRAL_FLUPS_GIT_REPOSITORY=git@git.immc.ucl.ac.be:examples/flups.git`
- `SPECTRAL_FLUPS_GIT_TAG=dev-gpu`
- `SPECTRAL_FLUPS_ENABLE_CUDA=ON`

```bash
nix develop -c cmake -S . -B build-flups-cuda \
  -DAVA_TARGET=CUDA \
  -DSPECTRAL_FFT_BACKEND=FLUPS \
  -DCMAKE_BUILD_TYPE=Release

nix develop -c cmake --build build-flups-cuda --target taylor_green -j 8

nix develop -c mpirun -np 4 ./build-flups-cuda/taylor_green \
  --n 64 --re 1600 \
  --adaptive-cfl 1 --dt 0.1 --cfl 0.4 --t-end 20 \
  --output-dt 0.1 --print-every 20 \
  --output-dir output_flups_cuda
```

## CUDA + MPI + CUFFTMp

Distributed GPU path. AVA kernels run on CUDA and cuFFTMp handles the
multi-process FFT. cuFFTMp is not part of the standard cuFFT package in this
Nix shell; it is normally shipped with the NVIDIA HPC SDK or NVIDIA Developer
Zone packages. The full SDK archive is several GB, so this project does not
download it from CMake.

Use an installed cuFFTMp/HPC SDK prefix:

```bash
nix develop -c cmake -S . -B build-cufftmp \
  -DAVA_TARGET=CUDA \
  -DSPECTRAL_FFT_BACKEND=CUFFTMP \
  -DSPECTRAL_CUFFTMP_ROOT=/path/to/nvhpc/Linux_x86_64/25.5 \
  -DCMAKE_BUILD_TYPE=Release
```

Alternatively, enter the Nix shell with `CUFFTMP_HOME` or `NVHPC_ROOT` pointing
to an existing SDK install. The shell also auto-detects common local install
locations such as `/opt/nvidia/hpc_sdk`, `/usr/local/nvidia/hpc_sdk`, and
`$HOME/nvhpc`.

Then build and run:

```bash
nix develop -c cmake --build build-cufftmp --target taylor_green -j 8

nix develop -c mpirun -np 4 ./build-cufftmp/taylor_green \
  --n 64 --re 1600 \
  --adaptive-cfl 1 --dt 0.1 --cfl 0.4 --t-end 20 \
  --output-dt 0.1 --print-every 20 \
  --output-dir output_cufftmp
```

At runtime, cuFFTMp also needs the NVSHMEM runtime libraries and bootstrap in
`LD_LIBRARY_PATH`; for an HPC SDK install this usually means the matching
`comm_libs/nvshmem/lib` directory.

## Option Summary

| Path | AVA target | FFT backend | MPI |
| --- | --- | --- | --- |
| CUDA + cuFFT | `-DAVA_TARGET=CUDA` | `-DSPECTRAL_FFT_BACKEND=CUFFT` | no |
| CPU + FFTW | `-DAVA_TARGET=CPU` | `-DSPECTRAL_FFT_BACKEND=FFTW` | no |
| CPU + FLUPS | `-DAVA_TARGET=CPU` | `-DSPECTRAL_FFT_BACKEND=FLUPS` | optional |
| CPU + FLUPS + MPI | `-DAVA_TARGET=CPU` | `-DSPECTRAL_FFT_BACKEND=FLUPS` | yes |
| CUDA + FLUPS + MPI | `-DAVA_TARGET=CUDA` | `-DSPECTRAL_FFT_BACKEND=FLUPS` | yes |
| CUDA + CUFFTMp + MPI | `-DAVA_TARGET=CUDA` | `-DSPECTRAL_FFT_BACKEND=CUFFTMP` | yes |

## FLUPS Build Options

These options control the FLUPS external project:

- `SPECTRAL_FLUPS_GIT_TAG`: FLUPS branch or tag to clone. Default: `release`.
- `SPECTRAL_FLUPS_GIT_REPOSITORY`: FLUPS repository to clone. Default:
  `https://github.com/vortexlab-uclouvain/flups.git`.
- `SPECTRAL_FLUPS_ENABLE_CUDA`: request CUDA/cuFFT support in the FLUPS build.
  Defaults to `OFF`.
- `SPECTRAL_FLUPS_CUDA_OPTS`: extra preprocessor flags appended to FLUPS `OPTS`
  when CUDA is enabled. Default: `-DBACKEND_CUDA -DHAVE_CUFFT`.
- `SPECTRAL_FLUPS_CUDA_LIBNAME`: CUDA libraries appended to FLUPS link flags.
  Default: `-lcufft -lcublas -lcudart`.
- `SPECTRAL_ASSUME_CUDA_AWARE_MPI`: required for `SPECTRAL_FLUPS_ENABLE_CUDA=ON`.
  This is an acknowledgement that the selected MPI can communicate CUDA device
  buffers. The Nix shell sets it automatically because it uses CUDA-aware
  OpenMPI.

## CUFFTMp Build Options

These options control cuFFTMp discovery:

- `SPECTRAL_CUFFTMP_ROOT`: cuFFTMp or NVIDIA HPC SDK installation prefix.

CMake also checks `CUFFTMP_HOME`, `NVSHMEM_HOME`, and `NVHPC_ROOT`.
The target links against `libcufftMp`, `libnvshmem_host`, and the CUDA driver.

For `AVA_TARGET=CUDA`, CMake defaults `CMAKE_CUDA_ARCHITECTURES` to `70` when
the user has not set it, because AVA's CUDA headers require `compute_70` or
newer.

## Implementation Notes

- AVA owns initialization, products, projection, phase-shift dealiasing, RK3
  updates, reductions, and diagnostics.
- FFTW, cuFFT, FLUPS, and cuFFTMp are only FFT backends.
- The FLUPS path sizes AVA arrays from FLUPS local physical and spectral
  topologies, so MPI ranks operate on local chunks rather than replicated global
  arrays.
- The cuFFTMp path uses cuFFTMp descriptor allocations for FFT fields and gives
  AVA kernels the local device pointer inside each descriptor. Non-field
  metadata stays on the normal AVA allocator.

The CMake guard currently fails early if `SPECTRAL_FLUPS_ENABLE_CUDA=ON` is
requested for a FLUPS source tree that does not expose CUDA/cuFFT implementation
files.
