#pragma once

#include "fftw_fft.hpp"
#include "flups_fft.hpp"

#include <array>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace spectral {

using ScalarPhysical = std::vector<double>;
using ScalarSpectral = std::vector<double>;
using VectorPhysical = std::array<ScalarPhysical, 3>;
using VectorSpectral = std::array<ScalarSpectral, 3>;

#ifdef SPECTRAL_USE_FLUPS_BACKEND
using FftBackend = FlupsFft3D;
#else
using FftBackend = FftwFft3D;
#endif

struct Diagnostics {
    double t = 0.0;
    double energy = 0.0;
    double dissipation = 0.0;
    double enstrophy = 0.0;
    double max_spectral_divergence = 0.0;
};

struct SolverProfile {
    double step_rk3 = 0.0;
    double rhs = 0.0;
    double nonlinear_rhs = 0.0;
    double velocity_ifft = 0.0;
    double shifted_velocity_ifft = 0.0;
    double product_fft = 0.0;
    double projection = 0.0;
    double truncation = 0.0;
    double diagnostics = 0.0;
    int steps = 0;
    int rhs_calls = 0;
    int nonlinear_calls = 0;
};

class SpectralNavierStokes {
public:
    SpectralNavierStokes(std::array<int, 3> shape,
                         std::array<double, 3> lengths,
                         double viscosity,
                         MPI_Comm comm,
                         bool phase_shift_dealiasing = true);

    FftBackend& fft() { return fft_; }
    const FftBackend& fft() const { return fft_; }

    VectorPhysical taylor_green(double amplitude = 1.0, int mode = 1) const;
    VectorSpectral to_spectral(const VectorPhysical& u);
    VectorPhysical to_physical(const VectorSpectral& u_hat);

    void project(VectorSpectral& u_hat) const;
    void truncate(VectorSpectral& u_hat) const;
    ScalarSpectral product_dealiased(const ScalarSpectral& a_hat, const ScalarSpectral& b_hat);
    VectorSpectral nonlinear_rhs(const VectorSpectral& u_hat);
    VectorSpectral rhs(const VectorSpectral& u_hat);
    VectorSpectral step_rk3(const VectorSpectral& u_hat, double t, double dt);

    double kinetic_energy(const VectorPhysical& u) const;
    double kinetic_energy_spectral(const VectorSpectral& u_hat);
    double max_velocity(const VectorSpectral& u_hat);
    VectorSpectral vorticity_hat(const VectorSpectral& u_hat) const;
    double enstrophy(const VectorSpectral& u_hat);
    double dissipation(const VectorSpectral& u_hat);
    double max_spectral_divergence(const VectorSpectral& u_hat) const;
    Diagnostics diagnostics(const VectorSpectral& u_hat, double t);
    void reset_profile();
    const SolverProfile& profile() const { return profile_; }

private:
    bool keep_mode(const SpectralMode& mode) const;
    void zero_mean(VectorSpectral& u_hat) const;

    FftBackend fft_;
    double nu_ = 0.0;
    bool phase_shift_dealiasing_ = true;
    std::vector<std::complex<double>> phase_shift_;
    std::vector<std::uint8_t> keep_mode_mask_;
    SolverProfile profile_;
};

void write_diagnostics_csv(const std::string& path, const std::vector<Diagnostics>& history);
void write_vorticity_ppm(const std::string& path,
                         SpectralNavierStokes& solver,
                         const VectorSpectral& u_hat);

}  // namespace spectral
