{
  description = "Standalone C++ pseudo-spectral Navier-Stokes solver using FLUPS FFTs";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            gnumake
            gcc
            git
            perf
            openmpi
            fftw
            hdf5
            pkg-config
            nodejs
            python3
            python3Packages.matplotlib
            python3Packages.numpy
          ];

          shellHook = ''
            export OMPI_MCA_rmaps_base_oversubscribe=1
            export OMPI_MCA_mpi_yield_when_idle=1
          '';
        };
      });
    };
}
