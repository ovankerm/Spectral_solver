#include "spectral_solver.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace spectral {

namespace {
constexpr double pi = 3.141592653589793238462643383279502884;
using Clock = std::chrono::steady_clock;

double elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::complex<double> load_complex(const std::vector<double>& a, std::size_t id) {
    return {a[id], a[id + 1]};
}

void store_complex(std::vector<double>& a, std::size_t id, std::complex<double> z) {
    a[id] = z.real();
    a[id + 1] = z.imag();
}

VectorSpectral make_vector_spectral(std::size_t n) {
    return {ScalarSpectral(n, 0.0), ScalarSpectral(n, 0.0), ScalarSpectral(n, 0.0)};
}
}  // namespace

SpectralNavierStokes::SpectralNavierStokes(std::array<int, 3> shape,
                                           std::array<double, 3> lengths,
                                           double viscosity,
                                           MPI_Comm comm,
                                           bool phase_shift_dealiasing)
    : fft_(shape, lengths, comm), nu_(viscosity), phase_shift_dealiasing_(phase_shift_dealiasing) {
    if (nu_ < 0.0) {
        throw std::runtime_error("viscosity must be non-negative");
    }
    phase_shift_.resize(fft_.spectral_storage_size() / 2);
    keep_mode_mask_.resize(fft_.spectral_storage_size() / 2);
    const auto& L = fft_.lengths();
    const auto& n = fft_.shape();
    const double dx = 0.5 * L[0] / n[0];
    const double dy = 0.5 * L[1] / n[1];
    const double dz = 0.5 * L[2] / n[2];
    fft_.for_each_spectral([&](std::size_t id, const SpectralMode& mode) {
        phase_shift_[id / 2] = std::exp(std::complex<double>(0.0, mode.kx * dx + mode.ky * dy + mode.kz * dz));
        keep_mode_mask_[id / 2] = keep_mode(mode) ? 1 : 0;
    });
}

VectorPhysical SpectralNavierStokes::taylor_green(double amplitude, int mode) const {
    const auto& n = fft_.shape();
    const auto& L = fft_.lengths();
    VectorPhysical u = {
        ScalarPhysical(static_cast<std::size_t>(fft_.physical_size()), 0.0),
        ScalarPhysical(static_cast<std::size_t>(fft_.physical_size()), 0.0),
        ScalarPhysical(static_cast<std::size_t>(fft_.physical_size()), 0.0),
    };
    for (int iz = 0; iz < n[2]; ++iz) {
        const double z = (iz + 0.5) * L[2] / n[2];
        for (int iy = 0; iy < n[1]; ++iy) {
            const double y = (iy + 0.5) * L[1] / n[1];
            for (int ix = 0; ix < n[0]; ++ix) {
                const double x = (ix + 0.5) * L[0] / n[0];
                const double ax = 2.0 * pi * mode * x / L[0];
                const double ay = 2.0 * pi * mode * y / L[1];
                const double az = 2.0 * pi * mode * z / L[2];
                const std::size_t id = static_cast<std::size_t>(ix + n[0] * (iy + n[1] * iz));
                u[0][id] = amplitude * std::sin(ax) * std::cos(ay) * std::cos(az);
                u[1][id] = -amplitude * std::cos(ax) * std::sin(ay) * std::cos(az);
            }
        }
    }
    return u;
}

VectorSpectral SpectralNavierStokes::to_spectral(const VectorPhysical& u) {
    VectorSpectral out = make_vector_spectral(fft_.spectral_storage_size());
    for (int c = 0; c < 3; ++c) {
        out[c] = fft_.forward(u[c]);
    }
    return out;
}

VectorPhysical SpectralNavierStokes::to_physical(const VectorSpectral& u_hat) {
    VectorPhysical out;
    for (int c = 0; c < 3; ++c) {
        out[c] = fft_.backward(u_hat[c]);
    }
    return out;
}

void SpectralNavierStokes::project(VectorSpectral& u_hat) const {
    fft_.for_each_spectral([&](std::size_t id, const SpectralMode& mode) {
        if (mode.k2 == 0.0) {
            return;
        }
        const auto ux = load_complex(u_hat[0], id);
        const auto uy = load_complex(u_hat[1], id);
        const auto uz = load_complex(u_hat[2], id);
        const auto dot = mode.kx * ux + mode.ky * uy + mode.kz * uz;
        const auto factor = dot / mode.k2;
        store_complex(u_hat[0], id, ux - mode.kx * factor);
        store_complex(u_hat[1], id, uy - mode.ky * factor);
        store_complex(u_hat[2], id, uz - mode.kz * factor);
    });
    zero_mean(u_hat);
}

void SpectralNavierStokes::truncate(VectorSpectral& u_hat) const {
    fft_.for_each_spectral([&](std::size_t id, const SpectralMode&) {
        if (!keep_mode_mask_[id / 2]) {
            for (auto& component : u_hat) {
                component[id] = 0.0;
                component[id + 1] = 0.0;
            }
        }
    });
}

ScalarSpectral SpectralNavierStokes::product_dealiased(const ScalarSpectral& a_hat,
                                                       const ScalarSpectral& b_hat) {
    const auto a = fft_.backward(a_hat);
    const auto b = fft_.backward(b_hat);
    ScalarPhysical c0(a.size(), 0.0);
    for (std::size_t i = 0; i < c0.size(); ++i) {
        c0[i] = a[i] * b[i];
    }
    auto c0_hat = fft_.forward(c0);

    auto a_shift_hat = a_hat;
    auto b_shift_hat = b_hat;
    fft_.for_each_spectral([&](std::size_t id, const SpectralMode&) {
        const auto phase = phase_shift_[id / 2];
        store_complex(a_shift_hat, id, load_complex(a_shift_hat, id) * phase);
        store_complex(b_shift_hat, id, load_complex(b_shift_hat, id) * phase);
    });

    const auto a_shift = fft_.backward(a_shift_hat);
    const auto b_shift = fft_.backward(b_shift_hat);
    ScalarPhysical cs(a_shift.size(), 0.0);
    for (std::size_t i = 0; i < cs.size(); ++i) {
        cs[i] = a_shift[i] * b_shift[i];
    }
    auto cs_hat = fft_.forward(cs);

    ScalarSpectral out = c0_hat;
    fft_.for_each_spectral([&](std::size_t id, const SpectralMode&) {
        const auto shifted_back = load_complex(cs_hat, id) * std::conj(phase_shift_[id / 2]);
        auto value = 0.5 * (load_complex(c0_hat, id) + shifted_back);
        if (!keep_mode_mask_[id / 2]) {
            value = 0.0;
        }
        store_complex(out, id, value);
    });
    return out;
}

VectorSpectral SpectralNavierStokes::nonlinear_rhs(const VectorSpectral& u_hat) {
    const auto nonlinear_start = Clock::now();
    profile_.nonlinear_calls += 1;
    VectorSpectral rhs = make_vector_spectral(fft_.spectral_storage_size());
    const auto velocity_ifft_start = Clock::now();
    const auto u = to_physical(u_hat);
    profile_.velocity_ifft += elapsed_seconds(velocity_ifft_start);

    VectorPhysical u_shift;
    if (phase_shift_dealiasing_) {
        auto u_shift_hat = u_hat;
        fft_.for_each_spectral([&](std::size_t id, const SpectralMode&) {
            const auto phase = phase_shift_[id / 2];
            for (int c = 0; c < 3; ++c) {
                store_complex(u_shift_hat[c], id, load_complex(u_shift_hat[c], id) * phase);
            }
        });
        const auto shifted_ifft_start = Clock::now();
        u_shift = to_physical(u_shift_hat);
        profile_.shifted_velocity_ifft += elapsed_seconds(shifted_ifft_start);
    }

    std::array<std::array<ScalarSpectral, 3>, 3> product_hat;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            product_hat[a][b] = ScalarSpectral(fft_.spectral_storage_size(), 0.0);
        }
    }

    for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
            ScalarPhysical c0(u[0].size(), 0.0);
            ScalarPhysical cs;
            if (phase_shift_dealiasing_) {
                cs.assign(u[0].size(), 0.0);
            }
            for (std::size_t p = 0; p < c0.size(); ++p) {
                c0[p] = u[a][p] * u[b][p];
                if (phase_shift_dealiasing_) {
                    cs[p] = u_shift[a][p] * u_shift[b][p];
                }
            }

            const auto product_fft_start = Clock::now();
            auto c0_hat = fft_.forward(c0);
            ScalarSpectral cs_hat;
            if (phase_shift_dealiasing_) {
                cs_hat = fft_.forward(cs);
            }
            profile_.product_fft += elapsed_seconds(product_fft_start);
            auto out = ScalarSpectral(fft_.spectral_storage_size(), 0.0);
            fft_.for_each_spectral([&](std::size_t id, const SpectralMode&) {
                auto value = load_complex(c0_hat, id);
                if (phase_shift_dealiasing_) {
                    value = 0.5 * (value + load_complex(cs_hat, id) * std::conj(phase_shift_[id / 2]));
                }
                if (!keep_mode_mask_[id / 2]) {
                    value = 0.0;
                }
                store_complex(out, id, value);
            });
            product_hat[a][b] = out;
            if (a != b) {
                product_hat[b][a] = out;
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        fft_.for_each_spectral([&](std::size_t id, const SpectralMode& mode) {
            const std::complex<double> imag(0.0, 1.0);
            const auto value = -imag * (mode.kx * load_complex(product_hat[0][i], id)
                                      + mode.ky * load_complex(product_hat[1][i], id)
                                      + mode.kz * load_complex(product_hat[2][i], id));
            store_complex(rhs[i], id, value);
        });
    }
    truncate(rhs);
    profile_.nonlinear_rhs += elapsed_seconds(nonlinear_start);
    return rhs;
}

VectorSpectral SpectralNavierStokes::rhs(const VectorSpectral& u_hat) {
    const auto rhs_start = Clock::now();
    profile_.rhs_calls += 1;
    auto out = nonlinear_rhs(u_hat);
    fft_.for_each_spectral([&](std::size_t id, const SpectralMode& mode) {
        for (int c = 0; c < 3; ++c) {
            const auto value = load_complex(out[c], id) - nu_ * mode.k2 * load_complex(u_hat[c], id);
            store_complex(out[c], id, value);
        }
    });
    auto local_start = Clock::now();
    project(out);
    profile_.projection += elapsed_seconds(local_start);
    local_start = Clock::now();
    truncate(out);
    profile_.truncation += elapsed_seconds(local_start);
    profile_.rhs += elapsed_seconds(rhs_start);
    return out;
}

VectorSpectral SpectralNavierStokes::step_rk3(const VectorSpectral& u_hat, double t, double dt) {
    (void)t;
    const auto step_start = Clock::now();
    profile_.steps += 1;
    auto u = u_hat;
    auto g = rhs(u);
    for (int c = 0; c < 3; ++c) {
        for (std::size_t i = 0; i < u[c].size(); ++i) {
            u[c][i] += (dt / 3.0) * g[c][i];
        }
    }
    auto local_start = Clock::now();
    project(u);
    profile_.projection += elapsed_seconds(local_start);
    local_start = Clock::now();
    truncate(u);
    profile_.truncation += elapsed_seconds(local_start);

    auto h = rhs(u);
    for (int c = 0; c < 3; ++c) {
        for (std::size_t i = 0; i < u[c].size(); ++i) {
            g[c][i] = -(5.0 / 9.0) * g[c][i] + h[c][i];
            u[c][i] += (15.0 / 16.0) * dt * g[c][i];
        }
    }
    local_start = Clock::now();
    project(u);
    profile_.projection += elapsed_seconds(local_start);
    local_start = Clock::now();
    truncate(u);
    profile_.truncation += elapsed_seconds(local_start);

    h = rhs(u);
    for (int c = 0; c < 3; ++c) {
        for (std::size_t i = 0; i < u[c].size(); ++i) {
            g[c][i] = -(153.0 / 128.0) * g[c][i] + h[c][i];
            u[c][i] += (8.0 / 15.0) * dt * g[c][i];
        }
    }
    local_start = Clock::now();
    project(u);
    profile_.projection += elapsed_seconds(local_start);
    local_start = Clock::now();
    truncate(u);
    profile_.truncation += elapsed_seconds(local_start);
    profile_.step_rk3 += elapsed_seconds(step_start);
    return u;
}

double SpectralNavierStokes::kinetic_energy(const VectorPhysical& u) const {
    double sum = 0.0;
    for (std::size_t i = 0; i < u[0].size(); ++i) {
        sum += 0.5 * (u[0][i] * u[0][i] + u[1][i] * u[1][i] + u[2][i] * u[2][i]);
    }
    return sum / static_cast<double>(u[0].size());
}

double SpectralNavierStokes::kinetic_energy_spectral(const VectorSpectral& u_hat) {
    return kinetic_energy(to_physical(u_hat));
}

double SpectralNavierStokes::max_velocity(const VectorSpectral& u_hat) {
    const auto u = to_physical(u_hat);
    double max_value = 0.0;
    for (std::size_t i = 0; i < u[0].size(); ++i) {
        const double speed =
            std::sqrt(u[0][i] * u[0][i] + u[1][i] * u[1][i] + u[2][i] * u[2][i]);
        max_value = std::max(max_value, speed);
    }
    return max_value;
}

VectorSpectral SpectralNavierStokes::vorticity_hat(const VectorSpectral& u_hat) const {
    auto omega = make_vector_spectral(fft_.spectral_storage_size());
    fft_.for_each_spectral([&](std::size_t id, const SpectralMode& mode) {
        const std::complex<double> imag(0.0, 1.0);
        const auto ux = load_complex(u_hat[0], id);
        const auto uy = load_complex(u_hat[1], id);
        const auto uz = load_complex(u_hat[2], id);
        store_complex(omega[0], id, imag * (mode.ky * uz - mode.kz * uy));
        store_complex(omega[1], id, imag * (mode.kz * ux - mode.kx * uz));
        store_complex(omega[2], id, imag * (mode.kx * uy - mode.ky * ux));
    });
    return omega;
}

double SpectralNavierStokes::enstrophy(const VectorSpectral& u_hat) {
    auto omega = to_physical(vorticity_hat(u_hat));
    double sum = 0.0;
    for (std::size_t i = 0; i < omega[0].size(); ++i) {
        sum += omega[0][i] * omega[0][i] + omega[1][i] * omega[1][i] + omega[2][i] * omega[2][i];
    }
    return 0.5 * sum / static_cast<double>(omega[0].size());
}

double SpectralNavierStokes::dissipation(const VectorSpectral& u_hat) {
    return 2.0 * nu_ * enstrophy(u_hat);
}

double SpectralNavierStokes::max_spectral_divergence(const VectorSpectral& u_hat) const {
    double max_div = 0.0;
    fft_.for_each_spectral([&](std::size_t id, const SpectralMode& mode) {
        const auto div = mode.kx * load_complex(u_hat[0], id)
                       + mode.ky * load_complex(u_hat[1], id)
                       + mode.kz * load_complex(u_hat[2], id);
        max_div = std::max(max_div, std::abs(div));
    });
    return max_div;
}

Diagnostics SpectralNavierStokes::diagnostics(const VectorSpectral& u_hat, double t) {
    const auto diagnostics_start = Clock::now();
    const double ens = enstrophy(u_hat);
    auto row = Diagnostics{t, kinetic_energy_spectral(u_hat), 2.0 * nu_ * ens, ens, max_spectral_divergence(u_hat)};
    profile_.diagnostics += elapsed_seconds(diagnostics_start);
    return row;
}

void SpectralNavierStokes::reset_profile() {
    profile_ = SolverProfile{};
}

bool SpectralNavierStokes::keep_mode(const SpectralMode& mode) const {
    const auto& n = fft_.shape();
    const double ex = static_cast<double>(mode.mx * mode.mx) / std::pow(n[0] / 2.0, 2.0);
    const double ey = static_cast<double>(mode.my * mode.my) / std::pow(n[1] / 2.0, 2.0);
    const double ez = static_cast<double>(mode.mz * mode.mz) / std::pow(n[2] / 2.0, 2.0);
    return ex + ey + ez <= 1.0;
}

void SpectralNavierStokes::zero_mean(VectorSpectral& u_hat) const {
    fft_.for_each_spectral([&](std::size_t id, const SpectralMode& mode) {
        if (mode.mx == 0 && mode.my == 0 && mode.mz == 0) {
            for (auto& component : u_hat) {
                component[id] = 0.0;
                component[id + 1] = 0.0;
            }
        }
    });
}

void write_diagnostics_csv(const std::string& path, const std::vector<Diagnostics>& history) {
    std::ofstream file(path);
    file << "t,kinetic_energy,dissipation,enstrophy,max_spectral_divergence\n";
    file << std::setprecision(17);
    for (const auto& row : history) {
        file << row.t << ',' << row.energy << ',' << row.dissipation << ',' << row.enstrophy << ','
             << row.max_spectral_divergence << '\n';
    }
}

void write_vorticity_ppm(const std::string& path, SpectralNavierStokes& solver, const VectorSpectral& u_hat) {
    auto omega = solver.to_physical(solver.vorticity_hat(u_hat));
    const auto& n = solver.fft().shape();
    const int mid = n[2] / 2;
    std::vector<double> mag(static_cast<std::size_t>(n[0] * n[1]), 0.0);
    double max_value = 0.0;
    for (int iy = 0; iy < n[1]; ++iy) {
        for (int ix = 0; ix < n[0]; ++ix) {
            const std::size_t src = static_cast<std::size_t>(ix + n[0] * (iy + n[1] * mid));
            const std::size_t dst = static_cast<std::size_t>(ix + n[0] * iy);
            mag[dst] = std::sqrt(omega[0][src] * omega[0][src] + omega[1][src] * omega[1][src] + omega[2][src] * omega[2][src]);
            max_value = std::max(max_value, mag[dst]);
        }
    }
    if (max_value == 0.0) {
        max_value = 1.0;
    }

    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << n[0] << " " << n[1] << "\n255\n";
    for (int iy = n[1] - 1; iy >= 0; --iy) {
        for (int ix = 0; ix < n[0]; ++ix) {
            const double q = std::clamp(mag[static_cast<std::size_t>(ix + n[0] * iy)] / max_value, 0.0, 1.0);
            const unsigned char r = static_cast<unsigned char>(255.0 * std::sqrt(q));
            const unsigned char g = static_cast<unsigned char>(255.0 * q * q);
            const unsigned char b = static_cast<unsigned char>(255.0 * (1.0 - q) * 0.25);
            file.write(reinterpret_cast<const char*>(&r), 1);
            file.write(reinterpret_cast<const char*>(&g), 1);
            file.write(reinterpret_cast<const char*>(&b), 1);
        }
    }
}

}  // namespace spectral
