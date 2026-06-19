#include "ava_taylor_green.hpp"

#include <algorithm>
#include <exception>
#include <iostream>
#include <string>

#if defined(SPECTRAL_USE_FLUPS_BACKEND) || defined(SPECTRAL_USE_CUFFTMP)
#include <mpi.h>
#endif

namespace {

int read_int_arg(char** argv, int argc, const std::string& name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return std::stoi(argv[i + 1]);
        }
    }
    return fallback;
}

double read_double_arg(char** argv, int argc, const std::string& name, double fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return std::stod(argv[i + 1]);
        }
    }
    return fallback;
}

std::string read_string_arg(char** argv, int argc, const std::string& name, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(SPECTRAL_USE_FLUPS_BACKEND) || defined(SPECTRAL_USE_CUFFTMP)
    MPI_Init(&argc, &argv);
#endif
    try {
        spectral::ava_solver::RunConfig config;
        config.n = read_int_arg(argv, argc, "--n", config.n);
        config.steps = read_int_arg(argv, argc, "--steps", config.steps);
        config.max_steps = read_int_arg(argv, argc, "--max-steps", config.max_steps);
        config.output_every = std::max(1, read_int_arg(argv, argc, "--output-every", config.output_every));
        config.print_every = std::max(1, read_int_arg(argv, argc, "--print-every", config.print_every));
        config.adaptive_cfl = read_int_arg(argv, argc, "--adaptive-cfl", config.adaptive_cfl ? 1 : 0) != 0;
        config.dt = read_double_arg(argv, argc, "--dt", config.dt);
        config.cfl = read_double_arg(argv, argc, "--cfl", config.cfl);
        config.t_end = read_double_arg(argv, argc, "--t-end", config.steps * config.dt);
        config.output_dt = read_double_arg(argv, argc, "--output-dt", config.output_dt);
        config.reynolds = read_double_arg(argv, argc, "--re", config.reynolds);
        config.output_dir = read_string_arg(argv, argc, "--output-dir", config.output_dir);
        spectral::ava_solver::run_taylor_green(config);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
#if defined(SPECTRAL_USE_FLUPS_BACKEND) || defined(SPECTRAL_USE_CUFFTMP)
        MPI_Finalize();
#endif
        return 1;
    }
#if defined(SPECTRAL_USE_FLUPS_BACKEND) || defined(SPECTRAL_USE_CUFFTMP)
    MPI_Finalize();
#endif
    return 0;
}
