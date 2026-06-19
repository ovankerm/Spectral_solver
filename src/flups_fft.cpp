#include "flups_fft.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>

#if defined(BACKEND_CUDA)
#include <cuda_runtime.h>
#endif

namespace spectral {

#if defined(BACKEND_CUDA)
namespace {
void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

__global__ void pack_physical_kernel(const FlupsComplex* physical,
                                     double* buffer,
                                     const std::size_t* storage_index,
                                     std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        buffer[storage_index[i]] = physical[i].x;
    }
}

__global__ void unpack_physical_kernel(const double* buffer,
                                       FlupsComplex* physical,
                                       const std::size_t* storage_index,
                                       std::size_t n,
                                       double scale) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        physical[i] = {scale * buffer[storage_index[i]], 0.0};
    }
}

__global__ void pack_spectral_kernel(const FlupsComplex* spectral,
                                     double* buffer,
                                     const std::size_t* storage_index,
                                     std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const std::size_t dst = storage_index[i];
        buffer[dst] = spectral[i].x;
        buffer[dst + 1] = spectral[i].y;
    }
}

__global__ void unpack_spectral_kernel(const double* buffer,
                                       FlupsComplex* spectral,
                                       const std::size_t* storage_index,
                                       std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const std::size_t src = storage_index[i];
        spectral[i] = {buffer[src], buffer[src + 1]};
    }
}

dim3 grid_for(std::size_t n) {
    constexpr int block = 256;
    return dim3(static_cast<unsigned int>((n + block - 1) / block));
}
}  // namespace
#endif

FlupsFft3D::FlupsFft3D(std::array<int, 3> shape, std::array<double, 3> lengths, MPI_Comm comm)
    : shape_(shape), lengths_(lengths), comm_(comm) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        throw std::runtime_error("MPI must be initialized before constructing FlupsFft3D");
    }

    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &comm_size_);

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
    int nproc[3] = {0, 0, 0};
    MPI_Dims_create(comm_size_, 3, nproc);
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
        nloc_in_[d] = flups_topo_get_nloc(topo_in_, d);
    }
    flups_topo_get_istartGlob(topo_in_, istart_in_);
    flups_topo_get_istartGlob(topo_spec_, istart_spec_);
    flups_get_spectralInfo(solver_, kfact_, koffset_, symstart_);

    physical_storage_size_ = flups_topo_get_memsize(topo_in_);
    spectral_storage_size_ = flups_topo_get_memsize(topo_spec_);
    local_physical_size_ = nloc_in_[0] * nloc_in_[1] * nloc_in_[2];
    initialize_index_maps();

    calibrate_inverse_scale();
}

FlupsFft3D::~FlupsFft3D() {
    if (solver_ != nullptr) {
        flups_cleanup_solver(solver_);
    }
    if (topo_in_ != nullptr) {
        flups_topo_free(topo_in_);
    }
#if defined(BACKEND_CUDA)
    if (physical_storage_index_device_ != nullptr) {
        cudaFree(physical_storage_index_device_);
    }
    if (spectral_storage_index_device_ != nullptr) {
        cudaFree(spectral_storage_index_device_);
    }
#endif
    for (auto& dim : bc_) {
        for (auto*& side : dim) {
            if (side != nullptr) {
                flups_free(side);
                side = nullptr;
            }
        }
    }
}

void FlupsFft3D::forward_device(const FlupsComplex* physical, FlupsComplex* spectral) {
#if defined(BACKEND_CUDA)
    cuda_check(cudaMemset(buffer_, 0, sizeof(double) * static_cast<std::size_t>(flups_get_allocSize(solver_))),
               "clear FLUPS device buffer before device forward");
    constexpr int block = 256;
    pack_physical_kernel<<<grid_for(physical_storage_index_.size()), block>>>(
        physical, buffer_, physical_storage_index_device_, physical_storage_index_.size());
    cuda_check(cudaGetLastError(), "pack physical field into FLUPS device buffer");
    flups_do_FFT(solver_, buffer_, FLUPS_FORWARD);
    unpack_spectral_kernel<<<grid_for(spectral_storage_index_.size()), block>>>(
        buffer_, spectral, spectral_storage_index_device_, spectral_storage_index_.size());
    cuda_check(cudaGetLastError(), "unpack FLUPS spectral device buffer");
    flups_do_FFT(solver_, buffer_, FLUPS_BACKWARD);
#else
    (void)physical;
    (void)spectral;
    throw std::runtime_error("FLUPS device forward requires BACKEND_CUDA");
#endif
}

void FlupsFft3D::backward_device(const FlupsComplex* spectral, FlupsComplex* physical) {
#if defined(BACKEND_CUDA)
    cuda_check(cudaMemset(buffer_, 0, sizeof(double) * static_cast<std::size_t>(flups_get_allocSize(solver_))),
               "clear FLUPS device buffer before device backward");
    flups_do_FFT(solver_, buffer_, FLUPS_FORWARD);
    constexpr int block = 256;
    pack_spectral_kernel<<<grid_for(spectral_storage_index_.size()), block>>>(
        spectral, buffer_, spectral_storage_index_device_, spectral_storage_index_.size());
    cuda_check(cudaGetLastError(), "pack spectral field into FLUPS device buffer");
    flups_do_FFT(solver_, buffer_, FLUPS_BACKWARD);
    unpack_physical_kernel<<<grid_for(physical_storage_index_.size()), block>>>(
        buffer_, physical, physical_storage_index_device_, physical_storage_index_.size(), backward_scale_);
    cuda_check(cudaGetLastError(), "unpack FLUPS physical device buffer");
#else
    (void)spectral;
    (void)physical;
    throw std::runtime_error("FLUPS device backward requires BACKEND_CUDA");
#endif
}

std::vector<double> FlupsFft3D::forward(const std::vector<double>& physical) {
    copy_physical_to_buffer(physical);
    flups_do_FFT(solver_, buffer_, FLUPS_FORWARD);
#if defined(BACKEND_CUDA)
    std::vector<double> spectral(spectral_storage_size_, 0.0);
    cuda_check(cudaMemcpy(spectral.data(), buffer_, sizeof(double) * spectral_storage_size_, cudaMemcpyDeviceToHost),
               "copy FLUPS spectral buffer to host");
#else
    std::vector<double> spectral(buffer_, buffer_ + spectral_storage_size_);
#endif
    flups_do_FFT(solver_, buffer_, FLUPS_BACKWARD);
    return spectral;
}

std::vector<double> FlupsFft3D::backward(const std::vector<double>& spectral) {
    if (spectral.size() != spectral_storage_size_) {
        throw std::runtime_error("spectral field has wrong storage size");
    }
#if defined(BACKEND_CUDA)
    cuda_check(cudaMemset(buffer_, 0, sizeof(double) * static_cast<std::size_t>(flups_get_allocSize(solver_))),
               "clear FLUPS device buffer before backward");
#else
    std::fill(buffer_, buffer_ + flups_get_allocSize(solver_), 0.0);
#endif
    flups_do_FFT(solver_, buffer_, FLUPS_FORWARD);
#if defined(BACKEND_CUDA)
    cuda_check(cudaMemcpy(buffer_, spectral.data(), sizeof(double) * spectral_storage_size_, cudaMemcpyHostToDevice),
               "copy host spectral field to FLUPS device buffer");
#else
    std::copy(spectral.begin(), spectral.end(), buffer_);
#endif
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
#if defined(BACKEND_CUDA)
    std::vector<double> storage(static_cast<std::size_t>(flups_get_allocSize(solver_)), 0.0);

    for (int iz = 0; iz < nloc_in_[2]; ++iz) {
        for (int iy = 0; iy < nloc_in_[1]; ++iy) {
            for (int ix = 0; ix < nloc_in_[0]; ++ix) {
                const std::size_t src = static_cast<std::size_t>(ix + nloc_in_[0] * (iy + nloc_in_[1] * iz));
                const std::size_t dst = flups_locID(0, ix, iy, iz, 0, 0, nmem_in_, 1);
                storage[dst] = physical[src];
            }
        }
    }
    cuda_check(cudaMemcpy(buffer_, storage.data(), sizeof(double) * storage.size(), cudaMemcpyHostToDevice),
               "copy host physical field to FLUPS device buffer");
#else
    std::fill(buffer_, buffer_ + flups_get_allocSize(solver_), 0.0);

    for (int iz = 0; iz < nloc_in_[2]; ++iz) {
        for (int iy = 0; iy < nloc_in_[1]; ++iy) {
            for (int ix = 0; ix < nloc_in_[0]; ++ix) {
                const std::size_t src = static_cast<std::size_t>(ix + nloc_in_[0] * (iy + nloc_in_[1] * iz));
                const std::size_t dst = flups_locID(0, ix, iy, iz, 0, 0, nmem_in_, 1);
                buffer_[dst] = physical[src];
            }
        }
    }
#endif
}

std::vector<double> FlupsFft3D::copy_physical_from_buffer() const {
    std::vector<double> physical(static_cast<std::size_t>(physical_size()), 0.0);
#if defined(BACKEND_CUDA)
    std::vector<double> storage(physical_storage_size_, 0.0);
    cuda_check(cudaMemcpy(storage.data(), buffer_, sizeof(double) * physical_storage_size_, cudaMemcpyDeviceToHost),
               "copy FLUPS physical buffer to host");
#endif
    for (int iz = 0; iz < nloc_in_[2]; ++iz) {
        for (int iy = 0; iy < nloc_in_[1]; ++iy) {
            for (int ix = 0; ix < nloc_in_[0]; ++ix) {
                const std::size_t dst = static_cast<std::size_t>(ix + nloc_in_[0] * (iy + nloc_in_[1] * iz));
                const std::size_t src = flups_locID(0, ix, iy, iz, 0, 0, nmem_in_, 1);
#if defined(BACKEND_CUDA)
                physical[dst] = storage[src];
#else
                physical[dst] = buffer_[src];
#endif
            }
        }
    }
    return physical;
}

void FlupsFft3D::initialize_index_maps() {
    physical_storage_index_.assign(static_cast<std::size_t>(physical_size()), 0);
    for (int iz = 0; iz < nloc_in_[2]; ++iz) {
        for (int iy = 0; iy < nloc_in_[1]; ++iy) {
            for (int ix = 0; ix < nloc_in_[0]; ++ix) {
                const std::size_t dense = static_cast<std::size_t>(ix + nloc_in_[0] * (iy + nloc_in_[1] * iz));
                physical_storage_index_[dense] = flups_locID(0, ix, iy, iz, 0, 0, nmem_in_, 1);
            }
        }
    }

    spectral_storage_index_.clear();
    for_each_spectral([&](std::size_t src, const SpectralMode&) {
        spectral_storage_index_.push_back(src);
    });
    local_spectral_size_ = static_cast<int>(spectral_storage_index_.size());

#if defined(BACKEND_CUDA)
    cuda_check(cudaMalloc(&physical_storage_index_device_,
                          sizeof(std::size_t) * physical_storage_index_.size()),
               "allocate FLUPS physical index map");
    cuda_check(cudaMemcpy(physical_storage_index_device_, physical_storage_index_.data(),
                          sizeof(std::size_t) * physical_storage_index_.size(), cudaMemcpyHostToDevice),
               "copy FLUPS physical index map");
    cuda_check(cudaMalloc(&spectral_storage_index_device_,
                          sizeof(std::size_t) * spectral_storage_index_.size()),
               "allocate FLUPS spectral index map");
    cuda_check(cudaMemcpy(spectral_storage_index_device_, spectral_storage_index_.data(),
                          sizeof(std::size_t) * spectral_storage_index_.size(), cudaMemcpyHostToDevice),
               "copy FLUPS spectral index map");
#endif
}

void FlupsFft3D::calibrate_inverse_scale() {
    std::vector<double> ones(static_cast<std::size_t>(physical_size()), 1.0);
    auto spectrum = forward(ones);
    auto roundtrip = backward(spectrum);
    const double local_sum = std::accumulate(roundtrip.begin(), roundtrip.end(), 0.0);
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm_);
    const double mean = global_sum / static_cast<double>(global_physical_size());
    if (std::abs(mean) < 1.0e-30) {
        throw std::runtime_error("failed to calibrate FLUPS inverse FFT scale");
    }
    backward_scale_ = 1.0 / mean;
}

}  // namespace spectral
