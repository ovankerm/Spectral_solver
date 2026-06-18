#include "spectral_solver.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_fft_roundtrip(spectral::SpectralNavierStokes& solver) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    auto& fft = solver.fft();
    std::vector<double> u(static_cast<std::size_t>(fft.physical_size()), 0.0);
    const auto& n = fft.shape();
    fft.for_each_physical([&](std::size_t id, int ix, int iy, int) {
        u[id] = std::sin(2.0 * pi * (ix + 0.5) / n[0]) + 0.25 * std::cos(2.0 * pi * (iy + 0.5) / n[1]);
    });
    auto roundtrip = fft.backward(fft.forward(u));
    double local_err = 0.0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        local_err = std::max(local_err, std::abs(u[i] - roundtrip[i]));
    }
    double err = 0.0;
    MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_MAX, fft.comm());
    require(err < 1.0e-10, "FFT roundtrip error too large");
}

void test_projection_and_energy(spectral::SpectralNavierStokes& solver) {
    auto u = solver.taylor_green();
    auto u_hat = solver.to_spectral(u);
    solver.project(u_hat);
    solver.truncate(u_hat);
    require(solver.max_spectral_divergence(u_hat) < 1.0e-8, "projection did not remove divergence");
    require(std::abs(solver.kinetic_energy(u) - 0.125) < 1.0e-12, "Taylor-Green energy mismatch");
}

void test_viscous_decay(spectral::SpectralNavierStokes& solver) {
    auto u_hat = solver.to_spectral(solver.taylor_green());
        solver.project(u_hat);
        const double e0 = solver.kinetic_energy_spectral(u_hat);
        auto u1_hat = solver.step_rk3(u_hat, 0.0, 1.0e-3);
    const double e1 = solver.kinetic_energy_spectral(u1_hat);
    require(e1 < e0, "viscous step did not reduce kinetic energy");
    require(solver.max_spectral_divergence(u1_hat) < 1.0e-7, "RK step left nonzero divergence");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    try {
        constexpr double pi = 3.141592653589793238462643383279502884;
        spectral::SpectralNavierStokes solver({12, 12, 12}, {2.0 * pi, 2.0 * pi, 2.0 * pi}, 0.05, MPI_COMM_WORLD);
        test_fft_roundtrip(solver);
        test_projection_and_energy(solver);
        test_viscous_decay(solver);
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cout << "all C++ spectral tests passed\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "test failure: " << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }
    MPI_Finalize();
    return 0;
}
