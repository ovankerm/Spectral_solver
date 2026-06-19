# Method And Code

This repository contains a standalone Taylor-Green vortex benchmark for the
incompressible Navier-Stokes equations in a periodic cube. The main executable
is always named `taylor_green`; CMake options select the AVA target and the FFT
backend.

## Numerical Method

The solver advances the incompressible velocity field in Fourier space on a
uniform periodic `n^3` grid over `[0, 2 pi]^3`.

The evolved equation is the projected velocity form:

```text
du/dt = P[-u . grad(u) + nu laplacian(u)]
div(u) = 0
```

where `P` is the spectral Leray projection and `nu = 1 / Re` for the
non-dimensional Taylor-Green setup used here.

Main algorithmic pieces:

- Taylor-Green initial condition is created analytically.
- Velocity is transformed to spectral space.
- The mean mode is removed and the field is projected to divergence-free form.
- Nonlinear products are evaluated pseudo-spectrally.
- Phase-shift dealiasing is used for nonlinear products.
- High modes are truncated with a two-thirds style spectral mask.
- Time integration uses a third-order Runge-Kutta scheme.
- Optional adaptive time stepping chooses the main step from the requested CFL,
  while substeps are clipped to hit requested output times exactly.

The diagnostics written to `taylor_green_diagnostics.csv` are:

- `t`: simulation time.
- `kinetic_energy`: domain-averaged kinetic energy.
- `dissipation`: viscous dissipation rate.
- `enstrophy`: `0.5 <|omega|^2>`.
- `max_spectral_divergence`: maximum divergence residual in spectral space.
- `max_velocity`: maximum physical velocity magnitude.
- `dt` and `cfl`: actual last substep and CFL.
- `main_dt` and `main_cfl`: unconstrained CFL step before output-time clipping.

## Main Build Options

The executable is controlled by these CMake cache options:

```text
AVA_TARGET=CPU|CUDA
SPECTRAL_FFT_BACKEND=FFTW|FLUPS|CUFFT|CUFFTMP
SPECTRAL_PRECISION=DOUBLE|SINGLE
```

Supported paths currently exercised by the benchmark:

| Path | AVA target | FFT backend | MPI |
| --- | --- | --- | --- |
| AVA(CPU)+FFTW | CPU | FFTW | no |
| AVA(CPU)+FLUPS(CPU) | CPU | FLUPS | optional/yes |
| AVA(CUDA)+cuFFT | CUDA | cuFFT | no |
| AVA(CUDA)+FLUPS(CUDA) | CUDA | FLUPS | yes, CUDA-aware MPI required |
| AVA(CUDA)+cuFFTMp | CUDA | cuFFTMp | yes, NVSHMEM/cuFFTMp required |

The older direct `spectral_ns` library and tests remain in the tree as a CPU
reference/test implementation, but the runnable benchmark path is the AVA
solver.

## Code Structure

Core executable path:

- `src/main_ava.cpp`: command-line parsing, MPI initialization for distributed
  backends, and call into the AVA solver driver.
- `src/ava_taylor_green.hpp`: public AVA solver configuration, diagnostics,
  and driver declarations.
- `src/ava_taylor_green_impl.hpp`: implementation shared by the CPU and CUDA
  translation units. It contains the RK3 loop, kernels, diagnostics,
  projection, dealiasing, timing, and backend dispatch.
- `src/ava_taylor_green.cpp`: CPU AVA translation unit.
- `src/ava_taylor_green.cu`: CUDA AVA translation unit.

FFT and legacy reference components:

- `src/fftw_fft.hpp` / `src/fftw_fft.cpp`: local FFTW backend.
- `src/flups_fft.hpp` / `src/flups_fft.cpp`: FLUPS wrapper for distributed
  physical/spectral layouts.
- `src/spectral_solver.hpp` / `src/spectral_solver.cpp`: older CPU spectral
  solver used by `cpp_tests/spectral_ns_tests.cpp`.

Build and dependency support:

- `CMakeLists.txt`: single `taylor_green` executable, external builds for AVA,
  h3lpr, and FLUPS, and backend-specific target wiring.
- `cmake/CheckFlupsCuda.cmake`: configure-time guard for FLUPS CUDA source
  support.
- `flake.nix` / `.envrc`: local development shell.
- `scripts/*.sbatch`: Lucia and Meluxina benchmark submissions.
- `scripts/run_local_benchmark_sweep.sh`: local CPU/GPU benchmark sweep.

Analysis and reporting:

- `compare_taylor_green.py`: compares solver diagnostics with
  `spectral_Re1600_512.dat` and generates the validation plot.
- `perf_to_speedscope.py`: converts `perf script` output to Speedscope JSON.
- `Benchmark.md`: current benchmark report.
- `CMAKE_OPTIONS.md`: build option recipes for each execution path.

## Runtime Data Flow

At startup the solver initializes physical and spectral metadata for the local
rank or process. For single-process FFTW/cuFFT, the local domain is the full
grid. For FLUPS and cuFFTMp, the local domain is determined by the distributed
FFT backend.

One RHS evaluation performs:

1. Inverse transforms from spectral velocity to physical velocity.
2. Phase-shifted inverse transforms for the dealiased product.
3. Pointwise nonlinear products using AVA kernels.
4. Forward transforms of nonlinear products.
5. Accumulation of convective terms, viscous term, spectral mask, and
   divergence-free projection.

The cuFFTMp path uses cuFFTMp descriptors for distributed FFT fields. AVA kernels
operate on the local device pointer exposed by the descriptor, avoiding large
host-device transfers except for diagnostics/output.

## Validation Case

The benchmark case is Taylor-Green at `Re=1600`, usually run with:

```text
--re 1600 --adaptive-cfl 1 --dt 0.5 --cfl 0.5 --t-end 20 --output-dt 1
```

For reference-curve comparison, the 512^3 run is repeated with
`--output-dt 0.01` to match `spectral_Re1600_512.dat`.

