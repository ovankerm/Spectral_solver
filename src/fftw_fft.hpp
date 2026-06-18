#pragma once

#include "flups_fft.hpp"

#include <array>
#include <vector>

#include <fftw3.h>
#include <mpi.h>

namespace spectral {

class FftwFft3D {
public:
    FftwFft3D(std::array<int, 3> shape, std::array<double, 3> lengths, MPI_Comm comm);
    ~FftwFft3D();

    FftwFft3D(const FftwFft3D&) = delete;
    FftwFft3D& operator=(const FftwFft3D&) = delete;

    const std::array<int, 3>& shape() const { return shape_; }
    const std::array<double, 3>& lengths() const { return lengths_; }
    int physical_size() const { return shape_[0] * shape_[1] * shape_[2]; }
    int global_physical_size() const { return physical_size(); }
    std::size_t spectral_storage_size() const { return 2 * static_cast<std::size_t>(physical_size()); }
    MPI_Comm comm() const { return MPI_COMM_SELF; }
    int rank() const { return 0; }
    int comm_size() const { return 1; }

    template <class Func>
    void for_each_physical(Func&& func) const {
        for (int iz = 0; iz < shape_[2]; ++iz) {
            for (int iy = 0; iy < shape_[1]; ++iy) {
                for (int ix = 0; ix < shape_[0]; ++ix) {
                    const std::size_t id = static_cast<std::size_t>(ix + shape_[0] * (iy + shape_[1] * iz));
                    func(id, ix, iy, iz);
                }
            }
        }
    }

    std::vector<double> forward(const std::vector<double>& physical);
    std::vector<double> backward(const std::vector<double>& spectral);

    template <class Func>
    void for_each_spectral(Func&& func) const {
        for (int iz = 0; iz < shape_[2]; ++iz) {
            const int mz = signed_mode(iz, 2);
            for (int iy = 0; iy < shape_[1]; ++iy) {
                const int my = signed_mode(iy, 1);
                for (int ix = 0; ix < shape_[0]; ++ix) {
                    const int mx = signed_mode(ix, 0);
                    SpectralMode mode;
                    mode.mx = mx;
                    mode.my = my;
                    mode.mz = mz;
                    mode.kx = two_pi_ * mx / lengths_[0];
                    mode.ky = two_pi_ * my / lengths_[1];
                    mode.kz = two_pi_ * mz / lengths_[2];
                    mode.k2 = mode.kx * mode.kx + mode.ky * mode.ky + mode.kz * mode.kz;
                    const std::size_t id = 2 * static_cast<std::size_t>(ix + shape_[0] * (iy + shape_[1] * iz));
                    func(id, mode);
                }
            }
        }
    }

private:
    int signed_mode(int idx, int dim) const {
        return idx <= shape_[dim] / 2 ? idx : idx - shape_[dim];
    }

    static constexpr double two_pi_ = 6.283185307179586476925286766559005768;
    std::array<int, 3> shape_{};
    std::array<double, 3> lengths_{};
    std::size_t ncomplex_ = 0;
    fftw_complex* in_ = nullptr;
    fftw_complex* out_ = nullptr;
    fftw_plan forward_plan_ = nullptr;
    fftw_plan backward_plan_ = nullptr;
};

}  // namespace spectral
