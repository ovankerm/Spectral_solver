{
  description = "Standalone C++ pseudo-spectral Navier-Stokes solver using FLUPS FFTs";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system:
        f (import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        }));
    in
    {
      devShells = forAllSystems (pkgs: {
        default =
          let
            cudaAwareOpenmpi = pkgs.openmpi.override { cudaSupport = true; };
            cudaEnv = pkgs.symlinkJoin {
              name = "cuda-dev-env";
              paths = with pkgs.cudaPackages; [
                cuda_cccl
                cuda_cudart
                cuda_nvcc
                libcublas
                libcufft
                libnvshmem
              ];
            };
          in
          pkgs.mkShell {
          packages = with pkgs; [
            cmake
            gnumake
            gcc
            git
            perf
            cudaAwareOpenmpi
            fftw
            fftwFloat
            hdf5
            pkg-config
            nodejs
            python3
            python3Packages.matplotlib
            python3Packages.numpy
          ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux (with pkgs.cudaPackages; [
            cuda_cccl
            cuda_cudart
            cuda_nvcc
            libcublas
            libcufft
            libnvshmem
          ]);

          shellHook = ''
            export OMPI_MCA_rmaps_base_oversubscribe=1
            export OMPI_MCA_mpi_yield_when_idle=1
            export SPECTRAL_ASSUME_CUDA_AWARE_MPI=ON
            export CUDA_HOME=${cudaEnv}
            export CUDA_PATH=${cudaEnv}
            export CUDAToolkit_ROOT=${cudaEnv}
            export CPATH=${cudaEnv}/include''${CPATH:+:$CPATH}
            export LIBRARY_PATH=${cudaEnv}/lib''${LIBRARY_PATH:+:$LIBRARY_PATH}
            export LD_LIBRARY_PATH=${cudaEnv}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            export NVCC_PREPEND_FLAGS="-I${cudaEnv}/include''${NVCC_PREPEND_FLAGS:+ $NVCC_PREPEND_FLAGS}"

            if [ -z "''${CUFFTMP_HOME:-}" ]; then
              for nvhpc_root in "''${NVHPC_ROOT:-}" /opt/nvidia/hpc_sdk /usr/local/nvidia/hpc_sdk "$HOME/nvhpc"; do
                [ -n "$nvhpc_root" ] || continue
                [ -d "$nvhpc_root" ] || continue
                nvhpc_version=$(find "$nvhpc_root" -maxdepth 5 -type f -path '*/math_libs/include/cufftMp.h' -printf '%h\n' 2>/dev/null | sed 's#/math_libs/include$##' | sort -V | tail -1)
                if [ -n "$nvhpc_version" ]; then
                  export NVHPC_ROOT="$nvhpc_root"
                  export CUFFTMP_HOME="$nvhpc_version"
                  break
                fi
              done
            fi

            if [ -n "''${CUFFTMP_HOME:-}" ]; then
              export CPATH="$CUFFTMP_HOME/math_libs/include''${CPATH:+:$CPATH}"
              export LIBRARY_PATH="$CUFFTMP_HOME/math_libs/lib64:$CUFFTMP_HOME/comm_libs/nvshmem/lib''${LIBRARY_PATH:+:$LIBRARY_PATH}"
              export LD_LIBRARY_PATH="$CUFFTMP_HOME/math_libs/lib64:$CUFFTMP_HOME/comm_libs/nvshmem/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
              export NVSHMEM_HOME="''${NVSHMEM_HOME:-$CUFFTMP_HOME/comm_libs/nvshmem}"
            fi
          '';
        };
      });
    };
}
