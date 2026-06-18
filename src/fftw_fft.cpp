#include "fftw_fft.hpp"

#include <algorithm>
#include <stdexcept>

namespace spectral {

FftwFft3D::FftwFft3D(std::array<int, 3> shape, std::array<double, 3> lengths, MPI_Comm comm)
    : shape_(shape), lengths_(lengths), ncomplex_(static_cast<std::size_t>(shape[0] * shape[1] * shape[2])) {
    int comm_size = 0;
    MPI_Comm_size(comm, &comm_size);
    if (comm_size != 1) {
        throw std::runtime_error("The serial FFTW backend supports one MPI rank");
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

    in_ = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * ncomplex_));
    out_ = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * ncomplex_));
    if (in_ == nullptr || out_ == nullptr) {
        throw std::runtime_error("fftw_malloc failed");
    }
    forward_plan_ = fftw_plan_dft_3d(shape_[2], shape_[1], shape_[0], in_, out_, FFTW_FORWARD, FFTW_MEASURE);
    backward_plan_ = fftw_plan_dft_3d(shape_[2], shape_[1], shape_[0], in_, out_, FFTW_BACKWARD, FFTW_MEASURE);
    if (forward_plan_ == nullptr || backward_plan_ == nullptr) {
        throw std::runtime_error("FFTW plan creation failed");
    }
}

FftwFft3D::~FftwFft3D() {
    if (forward_plan_ != nullptr) {
        fftw_destroy_plan(forward_plan_);
    }
    if (backward_plan_ != nullptr) {
        fftw_destroy_plan(backward_plan_);
    }
    if (in_ != nullptr) {
        fftw_free(in_);
    }
    if (out_ != nullptr) {
        fftw_free(out_);
    }
}

std::vector<double> FftwFft3D::forward(const std::vector<double>& physical) {
    if (physical.size() != ncomplex_) {
        throw std::runtime_error("physical field has wrong size");
    }
    for (std::size_t i = 0; i < ncomplex_; ++i) {
        in_[i][0] = physical[i];
        in_[i][1] = 0.0;
    }
    fftw_execute(forward_plan_);
    std::vector<double> spectral(2 * ncomplex_, 0.0);
    for (std::size_t i = 0; i < ncomplex_; ++i) {
        spectral[2 * i] = out_[i][0];
        spectral[2 * i + 1] = out_[i][1];
    }
    return spectral;
}

std::vector<double> FftwFft3D::backward(const std::vector<double>& spectral) {
    if (spectral.size() != 2 * ncomplex_) {
        throw std::runtime_error("spectral field has wrong storage size");
    }
    for (std::size_t i = 0; i < ncomplex_; ++i) {
        in_[i][0] = spectral[2 * i];
        in_[i][1] = spectral[2 * i + 1];
    }
    fftw_execute(backward_plan_);
    const double scale = 1.0 / static_cast<double>(ncomplex_);
    std::vector<double> physical(ncomplex_, 0.0);
    for (std::size_t i = 0; i < ncomplex_; ++i) {
        physical[i] = scale * out_[i][0];
    }
    return physical;
}

}  // namespace spectral
