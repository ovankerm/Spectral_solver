#include "flups_fft.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace spectral {

FlupsFft3D::FlupsFft3D(std::array<int, 3> shape, std::array<double, 3> lengths, MPI_Comm comm)
    : shape_(shape), lengths_(lengths), comm_(comm) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        throw std::runtime_error("MPI must be initialized before constructing FlupsFft3D");
    }

    int comm_size = 0;
    MPI_Comm_size(comm_, &comm_size);
    if (comm_size != 1) {
        throw std::runtime_error("This standalone Navier-Stokes layer currently supports one MPI rank");
    }

    for (int n : shape_) {
        if (n <= 1) {
            throw std::runtime_error("all grid sizes must be greater than one");
        }
    }
    for (double length : lengths_) {
        if (length <= 0.0) {
            throw std::runtime_error("all domain lengths must be positive");
        }
    }

    const int nglob[3] = {shape_[0], shape_[1], shape_[2]};
    const int nproc[3] = {1, 1, 1};
    topo_in_ = flups_topo_new(0, 1, nglob, nproc, false, nullptr, FLUPS_ALIGNMENT, comm_);

    for (int id = 0; id < 3; ++id) {
        for (int side = 0; side < 2; ++side) {
            bc_[id][side] = static_cast<FLUPS_BoundaryType*>(flups_malloc(sizeof(FLUPS_BoundaryType)));
            bc_[id][side][0] = PER;
        }
    }

    const double h[3] = {
        lengths_[0] / shape_[0],
        lengths_[1] / shape_[1],
        lengths_[2] / shape_[2],
    };
    const double L[3] = {lengths_[0], lengths_[1], lengths_[2]};
    const FLUPS_CenterType centers[3] = {CELL_CENTER, CELL_CENTER, CELL_CENTER};

    solver_ = flups_init(topo_in_, bc_, h, L, NOD, centers);
    flups_set_greenType(solver_, CHAT_2);
    flups_setup(solver_, false);
    buffer_ = flups_get_innerBuffer(solver_);
    topo_spec_ = flups_get_innerTopo_spectral(solver_);

    for (int d = 0; d < 3; ++d) {
        nmem_in_[d] = flups_topo_get_nmem(topo_in_, d);
        nmem_spec_[d] = flups_topo_get_nmem(topo_spec_, d);
    }
    flups_topo_get_istartGlob(topo_in_, istart_in_);
    flups_topo_get_istartGlob(topo_spec_, istart_spec_);
    flups_get_spectralInfo(solver_, kfact_, koffset_, symstart_);

    physical_storage_size_ = flups_topo_get_memsize(topo_in_);
    spectral_storage_size_ = flups_topo_get_memsize(topo_spec_);

    calibrate_inverse_scale();
}

FlupsFft3D::~FlupsFft3D() {
    if (solver_ != nullptr) {
        flups_cleanup_solver(solver_);
    }
    if (topo_in_ != nullptr) {
        flups_topo_free(topo_in_);
    }
    for (auto& dim : bc_) {
        for (auto*& side : dim) {
            if (side != nullptr) {
                flups_free(side);
                side = nullptr;
            }
        }
    }
}

std::vector<double> FlupsFft3D::forward(const std::vector<double>& physical) {
    copy_physical_to_buffer(physical);
    flups_do_FFT(solver_, buffer_, FLUPS_FORWARD);
    std::vector<double> spectral(buffer_, buffer_ + spectral_storage_size_);
    flups_do_FFT(solver_, buffer_, FLUPS_BACKWARD);
    return spectral;
}

std::vector<double> FlupsFft3D::backward(const std::vector<double>& spectral) {
    if (spectral.size() != spectral_storage_size_) {
        throw std::runtime_error("spectral field has wrong storage size");
    }
    std::fill(buffer_, buffer_ + flups_get_allocSize(solver_), 0.0);
    flups_do_FFT(solver_, buffer_, FLUPS_FORWARD);
    std::copy(spectral.begin(), spectral.end(), buffer_);
    flups_do_FFT(solver_, buffer_, FLUPS_BACKWARD);
    auto physical = copy_physical_from_buffer();
    for (double& value : physical) {
        value *= backward_scale_;
    }
    return physical;
}

void FlupsFft3D::copy_physical_to_buffer(const std::vector<double>& physical) {
    if (static_cast<int>(physical.size()) != physical_size()) {
        throw std::runtime_error("physical field has wrong size");
    }
    std::fill(buffer_, buffer_ + flups_get_allocSize(solver_), 0.0);

    for (int iz = 0; iz < shape_[2]; ++iz) {
        for (int iy = 0; iy < shape_[1]; ++iy) {
            for (int ix = 0; ix < shape_[0]; ++ix) {
                const std::size_t src = static_cast<std::size_t>(ix + shape_[0] * (iy + shape_[1] * iz));
                const std::size_t dst = flups_locID(0, ix, iy, iz, 0, 0, nmem_in_, 1);
                buffer_[dst] = physical[src];
            }
        }
    }
}

std::vector<double> FlupsFft3D::copy_physical_from_buffer() const {
    std::vector<double> physical(static_cast<std::size_t>(physical_size()), 0.0);
    for (int iz = 0; iz < shape_[2]; ++iz) {
        for (int iy = 0; iy < shape_[1]; ++iy) {
            for (int ix = 0; ix < shape_[0]; ++ix) {
                const std::size_t dst = static_cast<std::size_t>(ix + shape_[0] * (iy + shape_[1] * iz));
                const std::size_t src = flups_locID(0, ix, iy, iz, 0, 0, nmem_in_, 1);
                physical[dst] = buffer_[src];
            }
        }
    }
    return physical;
}

void FlupsFft3D::calibrate_inverse_scale() {
    std::vector<double> ones(static_cast<std::size_t>(physical_size()), 1.0);
    auto spectrum = forward(ones);
    auto roundtrip = backward(spectrum);
    const double mean = std::accumulate(roundtrip.begin(), roundtrip.end(), 0.0) / roundtrip.size();
    if (std::abs(mean) < 1.0e-30) {
        throw std::runtime_error("failed to calibrate FLUPS inverse FFT scale");
    }
    backward_scale_ = 1.0 / mean;
}

}  // namespace spectral
