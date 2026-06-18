#include "spectral_solver.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

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
    MPI_Init(&argc, &argv);
    try {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        const int n = read_int_arg(argv, argc, "--n", 24);
        const int steps = read_int_arg(argv, argc, "--steps", 80);
        const int max_steps = read_int_arg(argv, argc, "--max-steps", 10000000);
        const int output_every = std::max(1, read_int_arg(argv, argc, "--output-every", 1));
        const bool profile = read_int_arg(argv, argc, "--profile", 0) != 0;
        const bool adaptive_cfl = read_int_arg(argv, argc, "--adaptive-cfl", 0) != 0;
        const double dt = read_double_arg(argv, argc, "--dt", 0.0025);
        const double cfl = read_double_arg(argv, argc, "--cfl", 0.5);
        const double t_end = read_double_arg(argv, argc, "--t-end", steps * dt);
        const double output_dt = read_double_arg(argv, argc, "--output-dt", dt);
        const double reynolds = read_double_arg(argv, argc, "--re", 400.0);
        const std::string output_dir = read_string_arg(argv, argc, "--output-dir", "output_cpp");

        constexpr double pi = 3.141592653589793238462643383279502884;
        const double length = 2.0 * pi;
        const double nu = length / (2.0 * pi * reynolds);

        if (rank == 0) {
            std::filesystem::create_directories(output_dir);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const std::string csv_path = output_dir + "/taylor_green_diagnostics.csv";
        std::ofstream csv;
        if (rank == 0) {
            csv.open(csv_path);
            csv << std::setprecision(17);
            csv << "t,kinetic_energy,dissipation,enstrophy,max_spectral_divergence\n";
        }

        spectral::SpectralNavierStokes solver({n, n, n}, {length, length, length}, nu, MPI_COMM_WORLD);
        auto u_hat = solver.to_spectral(solver.taylor_green());
        solver.project(u_hat);
        solver.truncate(u_hat);
        if (profile) {
            solver.reset_profile();
        }

        std::vector<spectral::Diagnostics> history;
        auto write_diagnostics_row = [&](double t) {
            const auto row = solver.diagnostics(u_hat, t);
            history.push_back(row);
            if (rank == 0) {
                csv << row.t << ',' << row.energy << ',' << row.dissipation << ',' << row.enstrophy << ','
                    << row.max_spectral_divergence << '\n';
                csv.flush();
            }
            return row;
        };

        double t = 0.0;
        if (adaptive_cfl) {
            if (dt <= 0.0 || cfl <= 0.0 || t_end < 0.0 || output_dt <= 0.0) {
                throw std::runtime_error("--dt, --cfl, and --output-dt must be positive and --t-end must be non-negative");
            }
            const double dx_min = length / static_cast<double>(n);
            int step = 0;
            double next_output = 0.0;
            auto row = write_diagnostics_row(t);
            if (rank == 0) {
                std::cout << "step " << step << "/" << max_steps << " t=" << t << " dt=0 E=" << row.energy << "\n";
            }
            next_output += output_dt;

            while (t < t_end - 1.0e-14) {
                if (step >= max_steps) {
                    throw std::runtime_error("adaptive CFL run exceeded --max-steps before reaching --t-end");
                }
                const double max_u = solver.max_velocity(u_hat);
                double dt_step = dt;
                if (max_u > 1.0e-14) {
                    dt_step = std::min(dt_step, cfl * dx_min / max_u);
                }
                dt_step = std::min(dt_step, t_end - t);
                if (next_output <= t_end + 1.0e-14) {
                    dt_step = std::min(dt_step, next_output - t);
                }
                if (dt_step <= 1.0e-14) {
                    next_output += output_dt;
                    continue;
                }

                u_hat = solver.step_rk3(u_hat, t, dt_step);
                t += dt_step;
                ++step;

                const bool output_due = t >= next_output - 1.0e-12 || t >= t_end - 1.0e-12;
                if (output_due) {
                    row = write_diagnostics_row(t);
                    if (rank == 0 && (history.size() == 2 || history.size() % 100 == 1 || t >= t_end - 1.0e-12)) {
                        std::cout << "step " << step << "/" << max_steps << " t=" << t << " dt=" << dt_step
                                  << " cfl=" << (max_u * dt_step / dx_min) << " E=" << row.energy << "\n";
                    }
                    while (next_output <= t + 1.0e-12) {
                        next_output += output_dt;
                    }
                }
            }
        } else {
            for (int step = 0; step <= steps; ++step) {
                if (step % output_every == 0 || step == steps) {
                    const auto row = write_diagnostics_row(t);
                    if (rank == 0 && (step % (100 * output_every) == 0 || step == steps)) {
                        std::cout << "step " << step << "/" << steps << " t=" << t << " E=" << row.energy << "\n";
                    }
                }
                if (step != steps) {
                    u_hat = solver.step_rk3(u_hat, t, dt);
                    t += dt;
                }
            }
        }
        if (rank == 0) {
            csv.close();
        }

        spectral::write_vorticity_ppm(output_dir + "/taylor_green_vorticity.ppm", solver, u_hat);

        const auto& last = history.back();
        if (rank == 0) {
            std::cout << std::setprecision(10)
                      << "saved diagnostics: " << csv_path << "\n"
                      << "saved vorticity image: " << output_dir << "/taylor_green_vorticity.ppm\n"
                      << "final energy: " << last.energy << "\n"
                      << "final dissipation: " << last.dissipation << "\n"
                      << "max spectral divergence: " << last.max_spectral_divergence << "\n";
        }
        if (profile && rank == 0) {
            const auto& p = solver.profile();
            std::cout << std::setprecision(6)
                      << "\nprofile_seconds\n"
                      << "steps," << p.steps << "\n"
                      << "rhs_calls," << p.rhs_calls << "\n"
                      << "nonlinear_calls," << p.nonlinear_calls << "\n"
                      << "step_rk3," << p.step_rk3 << "\n"
                      << "rhs," << p.rhs << "\n"
                      << "nonlinear_rhs," << p.nonlinear_rhs << "\n"
                      << "velocity_ifft," << p.velocity_ifft << "\n"
                      << "shifted_velocity_ifft," << p.shifted_velocity_ifft << "\n"
                      << "product_fft," << p.product_fft << "\n"
                      << "projection," << p.projection << "\n"
                      << "truncation," << p.truncation << "\n"
                      << "diagnostics," << p.diagnostics << "\n";
        }
    } catch (const std::exception& e) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cerr << "error: " << e.what() << "\n";
        }
        MPI_Finalize();
        return 1;
    }
    MPI_Finalize();
    return 0;
}
