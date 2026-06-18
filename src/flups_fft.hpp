#pragma once

#include <array>
#include <vector>

#include <mpi.h>

#include <flups.h>

namespace spectral {

struct SpectralMode {
    double kx = 0.0;
    double ky = 0.0;
    double kz = 0.0;
    double k2 = 0.0;
    int mx = 0;
    int my = 0;
    int mz = 0;
};

class FlupsFft3D {
public:
    FlupsFft3D(std::array<int, 3> shape, std::array<double, 3> lengths, MPI_Comm comm);
    ~FlupsFft3D();

    FlupsFft3D(const FlupsFft3D&) = delete;
    FlupsFft3D& operator=(const FlupsFft3D&) = delete;

    const std::array<int, 3>& shape() const { return shape_; }
    const std::array<double, 3>& lengths() const { return lengths_; }
    int physical_size() const { return shape_[0] * shape_[1] * shape_[2]; }
    std::size_t spectral_storage_size() const { return spectral_storage_size_; }

    std::vector<double> forward(const std::vector<double>& physical);
    std::vector<double> backward(const std::vector<double>& spectral);

    template <class Func>
    void for_each_spectral(Func&& func) const {
        const int ax0 = flups_topo_get_axis(topo_spec_);
        const int ax1 = (ax0 + 1) % 3;
        const int ax2 = (ax0 + 2) % 3;
        const int nf = 2;
        for (int i2 = 0; i2 < flups_topo_get_nloc(topo_spec_, ax2); ++i2) {
            for (int i1 = 0; i1 < flups_topo_get_nloc(topo_spec_, ax1); ++i1) {
                const std::size_t base = flups_locID(ax0, 0, i1, i2, 0, ax0, nmem_spec_, nf);
                for (int i0 = 0; i0 < flups_topo_get_nloc(topo_spec_, ax0); ++i0) {
                    const int ie[3] = {
                        istart_spec_[ax0] + i0,
                        istart_spec_[ax1] + i1,
                        istart_spec_[ax2] + i2,
                    };
                    const int raw[3] = {
                        ie[(3 - ax0) % 3],
                        ie[(4 - ax0) % 3],
                        ie[(5 - ax0) % 3],
                    };
                    auto signed_mode = [this](int idx, int dim) {
                        return idx <= shape_[dim] / 2 ? idx : idx - shape_[dim];
                    };
                    SpectralMode mode;
                    mode.mx = signed_mode(raw[0], 0);
                    mode.my = signed_mode(raw[1], 1);
                    mode.mz = signed_mode(raw[2], 2);
                    mode.kx = 2.0 * 3.141592653589793238462643383279502884 * mode.mx / lengths_[0];
                    mode.ky = 2.0 * 3.141592653589793238462643383279502884 * mode.my / lengths_[1];
                    mode.kz = 2.0 * 3.141592653589793238462643383279502884 * mode.mz / lengths_[2];
                    mode.k2 = mode.kx * mode.kx + mode.ky * mode.ky + mode.kz * mode.kz;
                    func(base + static_cast<std::size_t>(i0) * nf, mode);
                }
            }
        }
    }

private:
    void copy_physical_to_buffer(const std::vector<double>& physical);
    std::vector<double> copy_physical_from_buffer() const;
    void calibrate_inverse_scale();

    std::array<int, 3> shape_{};
    std::array<double, 3> lengths_{};
    MPI_Comm comm_ = MPI_COMM_NULL;
    FLUPS_Topology* topo_in_ = nullptr;
    FLUPS_Solver* solver_ = nullptr;
    FLUPS_Topology* topo_spec_ = nullptr;
    double* buffer_ = nullptr;
    FLUPS_BoundaryType* bc_[3][2]{};
    int nmem_in_[3]{};
    int nmem_spec_[3]{};
    int istart_in_[3]{};
    int istart_spec_[3]{};
    double kfact_[3]{};
    double koffset_[3]{};
    double symstart_[3]{};
    std::size_t physical_storage_size_ = 0;
    std::size_t spectral_storage_size_ = 0;
    double backward_scale_ = 1.0;
};

}  // namespace spectral
