#pragma once

#include <memory>
#include <string>

namespace spectral::ava_solver {

struct Diagnostics {
    double t = 0.0;
    double kinetic_energy = 0.0;
    double dissipation = 0.0;
    double enstrophy = 0.0;
    double max_spectral_divergence = 0.0;
    double max_velocity = 0.0;
    double dt = 0.0;
    double cfl = 0.0;
    double main_dt = 0.0;
    double main_cfl = 0.0;
};

struct RunConfig {
    int n = 64;
    int steps = 80;
    int max_steps = 10000000;
    int output_every = 1;
    int print_every = 10;
    double dt = 0.0025;
    double cfl = 0.4;
    double t_end = 0.2;
    double output_dt = 0.1;
    double reynolds = 1600.0;
    bool adaptive_cfl = false;
    std::string output_dir = "output_ava";
};

class TaylorGreenSolver {
public:
    struct Impl;

    TaylorGreenSolver(int n, double reynolds);
    ~TaylorGreenSolver();

    TaylorGreenSolver(const TaylorGreenSolver&) = delete;
    TaylorGreenSolver& operator=(const TaylorGreenSolver&) = delete;

    void step_rk3(double dt);
    Diagnostics diagnostics(double t);
    double max_velocity();
    void print_profile(int rank, double init_time, double run_time) const;

private:
    std::unique_ptr<Impl> impl_;
};

void run_taylor_green(const RunConfig& config);

}  // namespace spectral::ava_solver
