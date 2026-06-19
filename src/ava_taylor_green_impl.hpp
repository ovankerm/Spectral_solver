#include "ava_taylor_green.hpp"

#include "ava.h"
#include "ava_impl.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifdef AVA_TARGET_CUDA
#include <cuda_runtime.h>
#ifdef SPECTRAL_USE_CUFFTMP
#include <mpi.h>
#include <cufftMp.h>
#elif defined(SPECTRAL_USE_CUFFT)
#include <cufft.h>
#endif
#endif

#ifdef SPECTRAL_USE_FLUPS_BACKEND
#include "flups_fft.hpp"
#elif !defined(AVA_TARGET_CUDA)
#include <fftw3.h>
#endif

#if defined(SPECTRAL_USE_FLUPS_BACKEND) || defined(SPECTRAL_USE_CUFFTMP)
#define SPECTRAL_DISTRIBUTED_FFT 1
#else
#define SPECTRAL_DISTRIBUTED_FFT 0
#endif

namespace spectral::ava_solver {
namespace detail {

constexpr double pi = 3.141592653589793238462643383279502884;

#ifdef SPECTRAL_REAL_FLOAT
using Real = float;
#else
using Real = double;
#endif

constexpr Real rpi = static_cast<Real>(pi);

__host__ __device__ Real real_from_int(int value) {
    return static_cast<Real>(value);
}

__host__ __device__ Real real_from_size(std::size_t value) {
    return static_cast<Real>(value);
}

__host__ __device__ Real rsin(Real value) {
#ifdef SPECTRAL_REAL_FLOAT
#ifdef __CUDA_ARCH__
    return __sinf(value);
#else
    return sinf(value);
#endif
#else
    return sin(value);
#endif
}

__host__ __device__ Real rcos(Real value) {
#ifdef SPECTRAL_REAL_FLOAT
#ifdef __CUDA_ARCH__
    return __cosf(value);
#else
    return cosf(value);
#endif
#else
    return cos(value);
#endif
}

__host__ __device__ Real rsqrt(Real value) {
#ifdef SPECTRAL_REAL_FLOAT
    return sqrtf(value);
#else
    return sqrt(value);
#endif
}

double wall_seconds() {
    using clock = std::chrono::steady_clock;
    static const auto start = clock::now();
    const auto now = clock::now();
    return std::chrono::duration<double>(now - start).count();
}

struct Complex {
    Real x;
    Real y;
};

__host__ __device__ Complex czero() { return {Real{0}, Real{0}}; }
__host__ __device__ Complex cadd(Complex a, Complex b) { return {a.x + b.x, a.y + b.y}; }
__host__ __device__ Complex csub(Complex a, Complex b) { return {a.x - b.x, a.y - b.y}; }
__host__ __device__ Complex cmul(Complex a, Complex b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
__host__ __device__ Complex cscale(Complex a, Real s) { return {s * a.x, s * a.y}; }
__host__ __device__ Complex cscale(Complex a, int s) { return cscale(a, real_from_int(s)); }
__host__ __device__ Complex cconj(Complex a) { return {a.x, -a.y}; }
__host__ __device__ Real cabs2(Complex a) { return a.x * a.x + a.y * a.y; }
__host__ __device__ Complex minus_i_k(Complex a, int k) {
    const Real rk = real_from_int(k);
    return {rk * a.y, -rk * a.x};
}

int signed_mode(int i, int n) {
    return i <= n / 2 ? i : i - n;
}

std::size_t compact_index(int ix, int iy, int iz, int n) {
    return static_cast<std::size_t>(ix + n * (iy + n * iz));
}

bool keep_mode_host(int mx, int my, int mz, int n) {
    const double h = 0.5 * static_cast<double>(n);
    return (mx * mx + my * my + mz * mz) / (h * h) <= 1.0;
}

#ifdef AVA_TARGET_CUDA
void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}
void profile_sync(const char* what) {
    cuda_check(cudaGetLastError(), what);
    cuda_check(cudaDeviceSynchronize(), what);
}
#if defined(SPECTRAL_USE_CUFFT) || defined(SPECTRAL_USE_CUFFTMP)
void cufft_check(cufftResult err, const char* what) {
    if (err != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cuFFT error " + std::to_string(static_cast<int>(err)));
    }
}
#endif

#if defined(AVA_TARGET_CUDA) && defined(SPECTRAL_USE_CUFFT)
#ifdef SPECTRAL_REAL_FLOAT
using CufftComplex = cufftComplex;
constexpr cufftType cufft_type = CUFFT_C2C;
constexpr const char* cufft_precision_name = "C2C";
inline cufftResult cufft_exec(cufftHandle plan, CufftComplex* in, CufftComplex* out, int direction) {
    return cufftExecC2C(plan, in, out, direction);
}
#else
using CufftComplex = cufftDoubleComplex;
constexpr cufftType cufft_type = CUFFT_Z2Z;
constexpr const char* cufft_precision_name = "Z2Z";
inline cufftResult cufft_exec(cufftHandle plan, CufftComplex* in, CufftComplex* out, int direction) {
    return cufftExecZ2Z(plan, in, out, direction);
}
#endif
#endif

#if defined(AVA_TARGET_CUDA) && defined(SPECTRAL_USE_CUFFTMP)
#ifdef SPECTRAL_REAL_FLOAT
constexpr cufftType cufftmp_type = CUFFT_C2C;
inline cufftResult cufftmp_exec(cufftHandle plan, cudaLibXtDesc* in, cudaLibXtDesc* out, int direction) {
    return cufftXtExecDescriptorC2C(plan, in, out, direction);
}
#else
constexpr cufftType cufftmp_type = CUFFT_Z2Z;
inline cufftResult cufftmp_exec(cufftHandle plan, cudaLibXtDesc* in, cudaLibXtDesc* out, int direction) {
    return cufftXtExecDescriptorZ2Z(plan, in, out, direction);
}
#endif
#endif
#else
void profile_sync(const char*) {}
#endif

#if !defined(AVA_TARGET_CUDA) && !defined(SPECTRAL_USE_FLUPS_BACKEND)
#ifdef SPECTRAL_REAL_FLOAT
using FftwComplex = fftwf_complex;
using FftwPlan = fftwf_plan;
inline void* fftw_alloc_complex_values(std::size_t n) { return fftwf_malloc(sizeof(fftwf_complex) * n); }
inline void fftw_free_values(void* ptr) { fftwf_free(ptr); }
inline FftwPlan fftw_plan_3d(int n, FftwComplex* in, FftwComplex* out, int sign) {
    return fftwf_plan_dft_3d(n, n, n, in, out, sign, FFTW_ESTIMATE);
}
inline void fftw_execute_plan(FftwPlan plan, FftwComplex* in, FftwComplex* out) {
    fftwf_execute_dft(plan, in, out);
}
inline void fftw_destroy(FftwPlan plan) { fftwf_destroy_plan(plan); }
#else
using FftwComplex = fftw_complex;
using FftwPlan = fftw_plan;
inline void* fftw_alloc_complex_values(std::size_t n) { return fftw_malloc(sizeof(fftw_complex) * n); }
inline void fftw_free_values(void* ptr) { fftw_free(ptr); }
inline FftwPlan fftw_plan_3d(int n, FftwComplex* in, FftwComplex* out, int sign) {
    return fftw_plan_dft_3d(n, n, n, in, out, sign, FFTW_ESTIMATE);
}
inline void fftw_execute_plan(FftwPlan plan, FftwComplex* in, FftwComplex* out) {
    fftw_execute_dft(plan, in, out);
}
inline void fftw_destroy(FftwPlan plan) { fftw_destroy_plan(plan); }
#endif
#endif

double mpi_sum(double local) {
#if SPECTRAL_DISTRIBUTED_FFT
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global;
#else
    return local;
#endif
}

double mpi_max(double local) {
#if SPECTRAL_DISTRIBUTED_FFT
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global;
#else
    return local;
#endif
}

int mpi_rank() {
#if SPECTRAL_DISTRIBUTED_FFT
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
#else
    return 0;
#endif
}

std::pair<int, int> local_slab(int n, int rank, int comm_size) {
    const int base = n / comm_size;
    const int rem = n % comm_size;
    const int count = base + (rank < rem ? 1 : 0);
    const int start = rank * base + std::min(rank, rem);
    return {start, count};
}

class FftPlan {
public:
    explicit FftPlan(int n) : n_(n), global_size_(static_cast<std::size_t>(n) * n * n) {
#ifdef SPECTRAL_USE_FLUPS_BACKEND
        const std::array<int, 3> shape = {n_, n_, n_};
        const std::array<double, 3> lengths = {2.0 * pi, 2.0 * pi, 2.0 * pi};
        flups_ = std::make_unique<spectral::FlupsFft3D>(shape, lengths, MPI_COMM_WORLD);
        physical_size_ = static_cast<std::size_t>(flups_->physical_size());
        flups_->for_each_spectral([&](std::size_t, const spectral::SpectralMode&) { ++spectral_size_; });
#elif defined(SPECTRAL_USE_CUFFTMP)
        MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &comm_size_);
        int device_count = 0;
        cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count <= 0) {
            throw std::runtime_error("SPECTRAL_FFT_BACKEND=CUFFTMP requires at least one CUDA device");
        }
        cuda_check(cudaSetDevice(rank_ % device_count), "cudaSetDevice");
        const auto [x0, nx] = local_slab(n_, rank_, comm_size_);
        const auto [y0, ny] = local_slab(n_, rank_, comm_size_);
        physical_start_x_ = x0;
        spectral_start_y_ = y0;
        physical_size_ = static_cast<std::size_t>(nx) * n_ * n_;
        spectral_size_ = static_cast<std::size_t>(n_) * ny * n_;
        size_t work_size = 0;
        cufft_check(cufftCreate(&plan_), "cufftCreate");
        cufft_check(cufftMpAttachComm(plan_, CUFFT_COMM_MPI, &comm_), "cufftMpAttachComm");
        cufft_check(cufftMakePlan3d(plan_, n_, n_, n_, cufftmp_type, &work_size), "cufftMakePlan3d");
#elif defined(AVA_TARGET_CUDA)
#ifdef SPECTRAL_USE_CUFFT
        physical_size_ = global_size_;
        spectral_size_ = global_size_;
        cufft_check(cufftPlan3d(&plan_, n_, n_, n_, cufft_type), "cufftPlan3d");
#else
        throw std::runtime_error("AVA_TARGET=CUDA requires CUFFT or FLUPS backend");
#endif
#else
        physical_size_ = global_size_;
        spectral_size_ = global_size_;
        auto* in = reinterpret_cast<FftwComplex*>(fftw_alloc_complex_values(global_size_));
        auto* out = reinterpret_cast<FftwComplex*>(fftw_alloc_complex_values(global_size_));
        if (!in || !out) {
            throw std::runtime_error("fftw_malloc failed while creating FFTW plans");
        }
        forward_ = fftw_plan_3d(n_, in, out, FFTW_FORWARD);
        backward_ = fftw_plan_3d(n_, in, out, FFTW_BACKWARD);
        fftw_free_values(in);
        fftw_free_values(out);
        if (!forward_ || !backward_) {
            throw std::runtime_error("fftw_plan_dft_3d failed");
        }
#endif
    }

    ~FftPlan() {
#if defined(SPECTRAL_USE_CUFFTMP)
        cufftDestroy(plan_);
        if (comm_ != MPI_COMM_NULL) {
            MPI_Comm_free(&comm_);
        }
#elif defined(AVA_TARGET_CUDA) && defined(SPECTRAL_USE_CUFFT)
        cufftDestroy(plan_);
#elif !defined(SPECTRAL_USE_FLUPS_BACKEND) && !defined(AVA_TARGET_CUDA)
        fftw_destroy(forward_);
        fftw_destroy(backward_);
#endif
    }

    std::size_t physical_size() const { return physical_size_; }
    std::size_t spectral_size() const { return spectral_size_; }
    std::size_t global_size() const { return global_size_; }

    void allocate_physical(Complex** ptr) {
        ava::malloc(ptr, physical_size_);
    }

    void allocate_spectral(Complex** ptr) {
        ava::malloc(ptr, spectral_size_);
    }

    void free_field(Complex* ptr) {
        ava::free(ptr);
    }

    void fill_physical_indices(std::vector<int>& ix, std::vector<int>& iy, std::vector<int>& iz) const {
        ix.clear();
        iy.clear();
        iz.clear();
        ix.reserve(physical_size_);
        iy.reserve(physical_size_);
        iz.reserve(physical_size_);
#ifdef SPECTRAL_USE_FLUPS_BACKEND
        flups_->for_each_physical([&](std::size_t, int x, int y, int z) {
            ix.push_back(x);
            iy.push_back(y);
            iz.push_back(z);
        });
#elif defined(SPECTRAL_USE_CUFFTMP)
        const int nx = static_cast<int>(physical_size_ / static_cast<std::size_t>(n_ * n_));
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < n_; ++y) {
                for (int z = 0; z < n_; ++z) {
                    ix.push_back(physical_start_x_ + x);
                    iy.push_back(y);
                    iz.push_back(z);
                }
            }
        }
#else
        for (int z = 0; z < n_; ++z) {
            for (int y = 0; y < n_; ++y) {
                for (int x = 0; x < n_; ++x) {
                    ix.push_back(x);
                    iy.push_back(y);
                    iz.push_back(z);
                }
            }
        }
#endif
    }

    void fill_spectral_modes(std::vector<int>& mx, std::vector<int>& my, std::vector<int>& mz) const {
        mx.clear();
        my.clear();
        mz.clear();
        mx.reserve(spectral_size_);
        my.reserve(spectral_size_);
        mz.reserve(spectral_size_);
#ifdef SPECTRAL_USE_FLUPS_BACKEND
        flups_->for_each_spectral([&](std::size_t, const spectral::SpectralMode& mode) {
            mx.push_back(mode.mx);
            my.push_back(mode.my);
            mz.push_back(mode.mz);
        });
#elif defined(SPECTRAL_USE_CUFFTMP)
        const int ny = static_cast<int>(spectral_size_ / static_cast<std::size_t>(n_ * n_));
        for (int x = 0; x < n_; ++x) {
            for (int y = 0; y < ny; ++y) {
                for (int z = 0; z < n_; ++z) {
                    mx.push_back(signed_mode(x, n_));
                    my.push_back(signed_mode(spectral_start_y_ + y, n_));
                    mz.push_back(signed_mode(z, n_));
                }
            }
        }
#else
        for (int z = 0; z < n_; ++z) {
            for (int y = 0; y < n_; ++y) {
                for (int x = 0; x < n_; ++x) {
                    mx.push_back(signed_mode(x, n_));
                    my.push_back(signed_mode(y, n_));
                    mz.push_back(signed_mode(z, n_));
                }
            }
        }
#endif
    }

    void forward(Complex* in, Complex* out) {
        ++forward_calls_;
#ifdef SPECTRAL_USE_FLUPS_BACKEND
        const double t0 = wall_seconds();
#ifdef AVA_TARGET_CUDA
        flups_->forward_device(reinterpret_cast<const spectral::FlupsComplex*>(in),
                               reinterpret_cast<spectral::FlupsComplex*>(out));
        cuda_check(cudaDeviceSynchronize(), "FLUPS CUDA forward synchronize");
#else
        std::vector<Complex> host_in(physical_size_);
        ava::deep_copy(host_in.data(), in, physical_size_);
        std::vector<double> physical(physical_size_, 0.0);
        for (std::size_t i = 0; i < physical_size_; ++i) {
            physical[i] = host_in[i].x;
        }
        const auto spectral = flups_->forward(physical);
        std::vector<Complex> host_out(spectral_size_, {0.0, 0.0});
        std::size_t dense = 0;
        flups_->for_each_spectral([&](std::size_t src, const spectral::SpectralMode&) {
            host_out[dense++] = {spectral[src], spectral[src + 1]};
        });
        ava::deep_copy(out, host_out.data(), spectral_size_);
#endif
        forward_time_ += wall_seconds() - t0;
#elif defined(SPECTRAL_USE_CUFFTMP)
        cudaLibXtDesc* temp = nullptr;
        Complex* temp_ptr = nullptr;
        std::size_t temp_size = 0;
        double t0 = wall_seconds();
        allocate_temp_descriptor(&temp, &temp_ptr, &temp_size, CUFFT_XT_FORMAT_INPLACE);
        cuda_check(cudaDeviceSynchronize(), "cuFFTMp forward descriptor synchronize");
        descriptor_alloc_time_ += wall_seconds() - t0;
        if (temp_size < physical_size_ || temp_size < spectral_size_) {
            cufftXtFree(temp);
            throw std::runtime_error("cuFFTMp temporary descriptor is smaller than the local field");
        }
        t0 = wall_seconds();
        ava::deep_copy(temp_ptr, in, physical_size_);
        cuda_check(cudaDeviceSynchronize(), "cuFFTMp forward copy-in synchronize");
        copy_in_time_ += wall_seconds() - t0;
        t0 = wall_seconds();
        cufft_check(cufftmp_exec(plan_, temp, temp, CUFFT_FORWARD), "cufftXtExecDescriptor forward");
        cuda_check(cudaDeviceSynchronize(), "cuFFTMp forward synchronize");
        forward_time_ += wall_seconds() - t0;
        t0 = wall_seconds();
        ava::deep_copy(out, temp_ptr, spectral_size_);
        cuda_check(cudaDeviceSynchronize(), "cuFFTMp forward copy-out synchronize");
        copy_out_time_ += wall_seconds() - t0;
        t0 = wall_seconds();
        cufft_check(cufftXtFree(temp), "cufftXtFree forward temp");
        cuda_check(cudaDeviceSynchronize(), "cuFFTMp forward free synchronize");
        descriptor_free_time_ += wall_seconds() - t0;
#elif defined(AVA_TARGET_CUDA)
        const double t0 = wall_seconds();
        cufft_check(cufft_exec(plan_,
                               reinterpret_cast<CufftComplex*>(in),
                               reinterpret_cast<CufftComplex*>(out),
                               CUFFT_FORWARD),
                    "cufftExec forward");
        cuda_check(cudaDeviceSynchronize(), "cuFFT forward synchronize");
        forward_time_ += wall_seconds() - t0;
#else
        const double t0 = wall_seconds();
        fftw_execute_plan(forward_, reinterpret_cast<FftwComplex*>(in), reinterpret_cast<FftwComplex*>(out));
        forward_time_ += wall_seconds() - t0;
#endif
    }

    void backward(Complex* in, Complex* out) {
        ++backward_calls_;
#ifdef SPECTRAL_USE_FLUPS_BACKEND
        const double t0 = wall_seconds();
#ifdef AVA_TARGET_CUDA
        flups_->backward_device(reinterpret_cast<const spectral::FlupsComplex*>(in),
                                reinterpret_cast<spectral::FlupsComplex*>(out));
        cuda_check(cudaDeviceSynchronize(), "FLUPS CUDA backward synchronize");
#else
        std::vector<Complex> host_in(spectral_size_);
        ava::deep_copy(host_in.data(), in, spectral_size_);
        std::vector<double> spectral(flups_->spectral_storage_size(), 0.0);
        std::size_t dense = 0;
        flups_->for_each_spectral([&](std::size_t dst, const spectral::SpectralMode&) {
            spectral[dst] = host_in[dense].x;
            spectral[dst + 1] = host_in[dense].y;
            ++dense;
        });
        const auto physical = flups_->backward(spectral);
        std::vector<Complex> host_out(physical_size_, {0.0, 0.0});
        for (std::size_t i = 0; i < physical_size_; ++i) {
            host_out[i] = {physical[i], 0.0};
        }
        ava::deep_copy(out, host_out.data(), physical_size_);
#endif
        backward_time_ += wall_seconds() - t0;
#elif defined(SPECTRAL_USE_CUFFTMP)
        cudaLibXtDesc* temp = nullptr;
        Complex* temp_ptr = nullptr;
        std::size_t temp_size = 0;
        double t0 = wall_seconds();
        allocate_temp_descriptor(&temp, &temp_ptr, &temp_size, CUFFT_XT_FORMAT_INPLACE_SHUFFLED);
        cuda_check(cudaDeviceSynchronize(), "cuFFTMp inverse descriptor synchronize");
        descriptor_alloc_time_ += wall_seconds() - t0;
        if (temp_size < spectral_size_ || temp_size < physical_size_) {
            cufftXtFree(temp);
            throw std::runtime_error("cuFFTMp temporary descriptor is smaller than the local field");
        }
        t0 = wall_seconds();
        ava::deep_copy(temp_ptr, in, spectral_size_);
        cuda_check(cudaDeviceSynchronize(), "cuFFTMp inverse copy-in synchronize");
        copy_in_time_ += wall_seconds() - t0;
        t0 = wall_seconds();
        cufft_check(cufftmp_exec(plan_, temp, temp, CUFFT_INVERSE), "cufftXtExecDescriptor inverse");
        cuda_check(cudaDeviceSynchronize(), "cuFFTMp inverse synchronize");
        backward_time_ += wall_seconds() - t0;
        t0 = wall_seconds();
        ava::deep_copy(out, temp_ptr, physical_size_);
        cuda_check(cudaDeviceSynchronize(), "cuFFTMp inverse copy-out synchronize");
        copy_out_time_ += wall_seconds() - t0;
        t0 = wall_seconds();
        cufft_check(cufftXtFree(temp), "cufftXtFree inverse temp");
        cuda_check(cudaDeviceSynchronize(), "cuFFTMp inverse free synchronize");
        descriptor_free_time_ += wall_seconds() - t0;
        scale_inverse(out);
#elif defined(AVA_TARGET_CUDA)
        const double t0 = wall_seconds();
        cufft_check(cufft_exec(plan_,
                               reinterpret_cast<CufftComplex*>(in),
                               reinterpret_cast<CufftComplex*>(out),
                               CUFFT_INVERSE),
                    "cufftExec inverse");
        cuda_check(cudaDeviceSynchronize(), "cuFFT inverse synchronize");
        backward_time_ += wall_seconds() - t0;
        scale_inverse(out);
#else
        const double t0 = wall_seconds();
        fftw_execute_plan(backward_, reinterpret_cast<FftwComplex*>(in), reinterpret_cast<FftwComplex*>(out));
        backward_time_ += wall_seconds() - t0;
        scale_inverse(out);
#endif
    }

    void scale_inverse(Complex* out) const {
        const Real inv_size = Real{1} / real_from_size(global_size_);
        const auto nval = physical_size_;
        ava::parallel_for<AVA_BLOCK_SIZE>(nullptr, 0, static_cast<int64_t>(nval), [=] __device__(int64_t i) {
            out[i] = cscale(out[i], inv_size);
        });
    }

    void print_profile(int rank) const {
#if SPECTRAL_DISTRIBUTED_FFT
        const double local[8] = {forward_time_, backward_time_, descriptor_alloc_time_, descriptor_free_time_,
                                 copy_in_time_, copy_out_time_,
                                 static_cast<double>(forward_calls_), static_cast<double>(backward_calls_)};
        double global[8] = {};
        MPI_Reduce(local, global, 8, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << std::setprecision(6)
                      << "profile_fft max_rank_seconds"
                      << " forward=" << global[0]
                      << " backward=" << global[1]
                      << " descriptor_alloc=" << global[2]
                      << " descriptor_free=" << global[3]
                      << " copy_in=" << global[4]
                      << " copy_out=" << global[5]
                      << " forward_calls=" << static_cast<long long>(global[6])
                      << " backward_calls=" << static_cast<long long>(global[7])
                      << "\n";
        }
#else
        if (rank == 0) {
            std::cout << std::setprecision(6)
                      << "profile_fft seconds"
                      << " forward=" << forward_time_
                      << " backward=" << backward_time_
                      << " forward_calls=" << forward_calls_
                      << " backward_calls=" << backward_calls_
                      << "\n";
        }
#endif
    }

private:
#ifdef SPECTRAL_USE_CUFFTMP
    void allocate_temp_descriptor(cudaLibXtDesc** desc_out,
                                  Complex** ptr_out,
                                  std::size_t* size_out,
                                  cufftXtSubFormat format) const {
        cufft_check(cufftXtMalloc(plan_, desc_out, format), "cufftXtMalloc");
        auto* desc = *desc_out;
        if (desc == nullptr || desc->descriptor == nullptr || desc->descriptor->nGPUs <= 0) {
            throw std::runtime_error("unexpected cuFFTMp local descriptor shape");
        }
        void* local_ptr = nullptr;
        std::size_t local_size = 0;
        for (int i = 0; i < desc->descriptor->nGPUs; ++i) {
            if (desc->descriptor->data[i] != nullptr) {
                local_ptr = desc->descriptor->data[i];
                local_size = desc->descriptor->size[i] / sizeof(Complex);
                break;
            }
        }
        *ptr_out = reinterpret_cast<Complex*>(local_ptr);
        *size_out = local_size;
        if (*ptr_out == nullptr || *size_out == 0) {
            cufftXtFree(desc);
            throw std::runtime_error("cuFFTMp returned a null local device pointer");
        }
    }
#endif

    int n_ = 0;
    std::size_t global_size_ = 0;
    std::size_t physical_size_ = 0;
    std::size_t spectral_size_ = 0;
    long long forward_calls_ = 0;
    long long backward_calls_ = 0;
    double forward_time_ = 0.0;
    double backward_time_ = 0.0;
    double descriptor_alloc_time_ = 0.0;
    double descriptor_free_time_ = 0.0;
    double copy_in_time_ = 0.0;
    double copy_out_time_ = 0.0;
#ifdef SPECTRAL_USE_FLUPS_BACKEND
    std::unique_ptr<spectral::FlupsFft3D> flups_;
#elif defined(SPECTRAL_USE_CUFFTMP)
    cufftHandle plan_{};
    MPI_Comm comm_ = MPI_COMM_NULL;
    int rank_ = 0;
    int comm_size_ = 1;
    int physical_start_x_ = 0;
    int spectral_start_y_ = 0;
#elif defined(AVA_TARGET_CUDA)
    cufftHandle plan_{};
#else
    FftwPlan forward_{};
    FftwPlan backward_{};
#endif
};

}  // namespace detail

using namespace detail;

struct TaylorGreenSolver::Impl {
    int n = 0;
    Real nu = Real{0};
    FftPlan fft;
    std::size_t nphys = 0;
    std::size_t nspec = 0;
    std::size_t global_size = 0;
    std::array<Complex*, 3> u_hat{};
    std::array<Complex*, 3> rhs{};
    std::array<Complex*, 3> g{};
    std::array<Complex*, 3> phys{};
    std::array<Complex*, 3> shift_hat{};
    std::array<Complex*, 3> shift_phys{};
    Complex* product = nullptr;
    Complex* product_shift = nullptr;
    Complex* product_hat = nullptr;
    Complex* product_shift_hat = nullptr;
    Complex* phase = nullptr;
    unsigned char* mask = nullptr;
    int* mx = nullptr;
    int* my = nullptr;
    int* mz = nullptr;
    int* pix = nullptr;
    int* piy = nullptr;
    int* piz = nullptr;
    Real* scratch = nullptr;
    std::size_t scratch_size = 0;
    double step_time = 0.0;
    double rhs_time = 0.0;
    double diagnostics_time = 0.0;
    double max_velocity_time = 0.0;
    double rhs_velocity_backward_time = 0.0;
    double rhs_phase_shift_time = 0.0;
    double rhs_shift_backward_time = 0.0;
    double rhs_product_kernel_time = 0.0;
    double rhs_product_forward_time = 0.0;
    double rhs_accumulate_time = 0.0;
    double rhs_viscous_project_time = 0.0;
    double rk_update_time = 0.0;
    double rk_project_time = 0.0;
    long long step_calls = 0;
    long long rhs_calls = 0;
    long long diagnostics_calls = 0;
    long long max_velocity_calls = 0;

    Impl(int n_in, double reynolds) : n(n_in), nu(static_cast<Real>(1.0 / reynolds)), fft(n_in) {
        if (n <= 0) {
            throw std::runtime_error("grid size must be positive");
        }
        if (reynolds <= 0.0) {
            throw std::runtime_error("Reynolds number must be positive");
        }
        nphys = fft.physical_size();
        nspec = fft.spectral_size();
        global_size = fft.global_size();
        allocate();
        initialize_metadata();
        initialize_velocity();
    }

    ~Impl() {
        ava::free(scratch);
        ava::free(piz);
        ava::free(piy);
        ava::free(pix);
        ava::free(mz);
        ava::free(my);
        ava::free(mx);
        ava::free(mask);
        ava::free(phase);
        fft.free_field(product_shift_hat);
        fft.free_field(product_hat);
        fft.free_field(product_shift);
        fft.free_field(product);
        for (int c = 0; c < 3; ++c) {
            fft.free_field(shift_phys[c]);
            fft.free_field(shift_hat[c]);
            fft.free_field(phys[c]);
            fft.free_field(g[c]);
            fft.free_field(rhs[c]);
            fft.free_field(u_hat[c]);
        }
    }

    void allocate() {
        for (int c = 0; c < 3; ++c) {
            fft.allocate_spectral(&u_hat[c]);
            fft.allocate_spectral(&rhs[c]);
            fft.allocate_spectral(&g[c]);
            fft.allocate_physical(&phys[c]);
            fft.allocate_spectral(&shift_hat[c]);
            fft.allocate_physical(&shift_phys[c]);
        }
        fft.allocate_physical(&product);
        fft.allocate_physical(&product_shift);
        fft.allocate_spectral(&product_hat);
        fft.allocate_spectral(&product_shift_hat);
        ava::malloc(&phase, nspec);
        ava::malloc(&mask, nspec);
        ava::malloc(&mx, nspec);
        ava::malloc(&my, nspec);
        ava::malloc(&mz, nspec);
        ava::malloc(&pix, nphys);
        ava::malloc(&piy, nphys);
        ava::malloc(&piz, nphys);
        scratch_size = (std::max(nphys, nspec) + AVA_BLOCK_SIZE - 1) / AVA_BLOCK_SIZE;
        ava::malloc(&scratch, scratch_size);
    }

    void initialize_metadata() {
        std::vector<int> h_pix;
        std::vector<int> h_piy;
        std::vector<int> h_piz;
        fft.fill_physical_indices(h_pix, h_piy, h_piz);
        std::vector<int> h_mx;
        std::vector<int> h_my;
        std::vector<int> h_mz;
        fft.fill_spectral_modes(h_mx, h_my, h_mz);
        if (h_pix.size() != nphys || h_mx.size() != nspec) {
            throw std::runtime_error("FFT backend returned inconsistent local sizes");
        }
        std::vector<Complex> h_phase(nspec);
        std::vector<unsigned char> h_mask(nspec);
        for (std::size_t i = 0; i < nspec; ++i) {
            const double angle = (static_cast<double>(h_mx[i] + h_my[i] + h_mz[i]) * pi) / static_cast<double>(n);
            h_phase[i] = {static_cast<Real>(std::cos(angle)), static_cast<Real>(std::sin(angle))};
            h_mask[i] = keep_mode_host(h_mx[i], h_my[i], h_mz[i], n) ? 1 : 0;
        }
        ava::deep_copy(pix, h_pix.data(), nphys);
        ava::deep_copy(piy, h_piy.data(), nphys);
        ava::deep_copy(piz, h_piz.data(), nphys);
        ava::deep_copy(mx, h_mx.data(), nspec);
        ava::deep_copy(my, h_my.data(), nspec);
        ava::deep_copy(mz, h_mz.data(), nspec);
        ava::deep_copy(phase, h_phase.data(), nspec);
        ava::deep_copy(mask, h_mask.data(), nspec);
    }

    void initialize_velocity() {
        const int nn = n;
        auto* x_id = pix;
        auto* y_id = piy;
        auto* z_id = piz;
        auto* p0 = phys[0];
        auto* p1 = phys[1];
        auto* p2 = phys[2];
        const std::size_t local_n = nphys;
        ava::parallel_for<AVA_BLOCK_SIZE>(nullptr, 0, static_cast<int64_t>(local_n), [=] __device__(int64_t id) {
            const Real x = (real_from_int(x_id[id]) + Real{0.5}) * Real{2} * rpi / real_from_int(nn);
            const Real y = (real_from_int(y_id[id]) + Real{0.5}) * Real{2} * rpi / real_from_int(nn);
            const Real z = (real_from_int(z_id[id]) + Real{0.5}) * Real{2} * rpi / real_from_int(nn);
            p0[id] = {rsin(x) * rcos(y) * rcos(z), Real{0}};
            p1[id] = {-rcos(x) * rsin(y) * rcos(z), Real{0}};
            p2[id] = czero();
        });
        for (int c = 0; c < 3; ++c) {
            fft.forward(phys[c], u_hat[c]);
        }
        project_truncate(u_hat[0], u_hat[1], u_hat[2]);
    }

    void project_truncate(Complex* f0, Complex* f1, Complex* f2) {
        auto* keep = mask;
        auto* kx = mx;
        auto* ky = my;
        auto* kz = mz;
        const std::size_t local_n = nspec;
        ava::parallel_for<AVA_BLOCK_SIZE>(nullptr, 0, static_cast<int64_t>(local_n), [=] __device__(int64_t id) {
            const int ix = kx[id];
            const int iy = ky[id];
            const int iz = kz[id];
            const Real k2 = real_from_int(ix * ix + iy * iy + iz * iz);
            Complex ux = f0[id];
            Complex uy = f1[id];
            Complex uz = f2[id];
            if (k2 > Real{0} && keep[id]) {
                const Complex dot = cadd(cadd(cscale(ux, ix), cscale(uy, iy)), cscale(uz, iz));
                const Complex factor = cscale(dot, Real{1} / k2);
                f0[id] = csub(ux, cscale(factor, ix));
                f1[id] = csub(uy, cscale(factor, iy));
                f2[id] = csub(uz, cscale(factor, iz));
            } else {
                f0[id] = czero();
                f1[id] = czero();
                f2[id] = czero();
            }
        });
    }

    void compute_rhs() {
        const double rhs_t0 = wall_seconds();
        double section_t0 = wall_seconds();
        for (int c = 0; c < 3; ++c) {
            fft.backward(u_hat[c], phys[c]);
        }
        rhs_velocity_backward_time += wall_seconds() - section_t0;
        auto* r0 = rhs[0];
        auto* r1 = rhs[1];
        auto* r2 = rhs[2];
        auto* sh0 = shift_hat[0];
        auto* sh1 = shift_hat[1];
        auto* sh2 = shift_hat[2];
        auto* u0 = u_hat[0];
        auto* u1 = u_hat[1];
        auto* u2 = u_hat[2];
        auto* ph = phase;
        const std::size_t local_n = nspec;
        section_t0 = wall_seconds();
        ava::parallel_for<AVA_BLOCK_SIZE>(nullptr, 0, static_cast<int64_t>(local_n), [=] __device__(int64_t id) {
            r0[id] = czero();
            r1[id] = czero();
            r2[id] = czero();
            sh0[id] = cmul(u0[id], ph[id]);
            sh1[id] = cmul(u1[id], ph[id]);
            sh2[id] = cmul(u2[id], ph[id]);
        });
        profile_sync("compute_rhs phase shift");
        rhs_phase_shift_time += wall_seconds() - section_t0;
        section_t0 = wall_seconds();
        for (int c = 0; c < 3; ++c) {
            fft.backward(shift_hat[c], shift_phys[c]);
        }
        rhs_shift_backward_time += wall_seconds() - section_t0;
        accumulate_product(0, 0);
        accumulate_product(0, 1);
        accumulate_product(0, 2);
        accumulate_product(1, 1);
        accumulate_product(1, 2);
        accumulate_product(2, 2);
        add_viscous_project();
        rhs_time += wall_seconds() - rhs_t0;
        ++rhs_calls;
    }

    void accumulate_product(int a, int b) {
        Complex* phys_ptr[3] = {phys[0], phys[1], phys[2]};
        Complex* shift_ptr[3] = {shift_phys[0], shift_phys[1], shift_phys[2]};
        auto* pa = phys_ptr[a];
        auto* pb = phys_ptr[b];
        auto* sa = shift_ptr[a];
        auto* sb = shift_ptr[b];
        auto* p0 = product;
        auto* ps = product_shift;
        const std::size_t physical_n = nphys;
        double section_t0 = wall_seconds();
        ava::parallel_for<AVA_BLOCK_SIZE>(nullptr, 0, static_cast<int64_t>(physical_n), [=] __device__(int64_t id) {
            p0[id] = {pa[id].x * pb[id].x, Real{0}};
            ps[id] = {sa[id].x * sb[id].x, Real{0}};
        });
        profile_sync("accumulate_product product kernel");
        rhs_product_kernel_time += wall_seconds() - section_t0;
        section_t0 = wall_seconds();
        fft.forward(product, product_hat);
        fft.forward(product_shift, product_shift_hat);
        rhs_product_forward_time += wall_seconds() - section_t0;

        Complex* rhs_ptr[3] = {rhs[0], rhs[1], rhs[2]};
        auto* ra = rhs_ptr[a];
        auto* rb = rhs_ptr[b];
        auto* ph0 = product_hat;
        auto* phs = product_shift_hat;
        auto* ph = phase;
        auto* keep = mask;
        auto* kx = mx;
        auto* ky = my;
        auto* kz = mz;
        const std::size_t spectral_n = nspec;
        section_t0 = wall_seconds();
        ava::parallel_for<AVA_BLOCK_SIZE>(nullptr, 0, static_cast<int64_t>(spectral_n), [=] __device__(int64_t id) {
            const int kvals[3] = {kx[id], ky[id], kz[id]};
            Complex value = cscale(cadd(ph0[id], cmul(phs[id], cconj(ph[id]))), Real{0.5});
            if (!keep[id]) {
                value = czero();
            }
            ra[id] = cadd(ra[id], minus_i_k(value, kvals[b]));
            if (a != b) {
                rb[id] = cadd(rb[id], minus_i_k(value, kvals[a]));
            }
        });
        profile_sync("accumulate_product rhs accumulation");
        rhs_accumulate_time += wall_seconds() - section_t0;
    }

    void add_viscous_project() {
        auto* r0 = rhs[0];
        auto* r1 = rhs[1];
        auto* r2 = rhs[2];
        auto* u0 = u_hat[0];
        auto* u1 = u_hat[1];
        auto* u2 = u_hat[2];
        auto* keep = mask;
        auto* kx = mx;
        auto* ky = my;
        auto* kz = mz;
        const Real viscosity = nu;
        const std::size_t local_n = nspec;
        const double section_t0 = wall_seconds();
        ava::parallel_for<AVA_BLOCK_SIZE>(nullptr, 0, static_cast<int64_t>(local_n), [=] __device__(int64_t id) {
            const int ix = kx[id];
            const int iy = ky[id];
            const int iz = kz[id];
            const Real k2 = real_from_int(ix * ix + iy * iy + iz * iz);
            r0[id] = csub(r0[id], cscale(u0[id], viscosity * k2));
            r1[id] = csub(r1[id], cscale(u1[id], viscosity * k2));
            r2[id] = csub(r2[id], cscale(u2[id], viscosity * k2));
            if (k2 > Real{0} && keep[id]) {
                const Complex dot = cadd(cadd(cscale(r0[id], ix), cscale(r1[id], iy)), cscale(r2[id], iz));
                const Complex factor = cscale(dot, Real{1} / k2);
                r0[id] = csub(r0[id], cscale(factor, ix));
                r1[id] = csub(r1[id], cscale(factor, iy));
                r2[id] = csub(r2[id], cscale(factor, iz));
            } else {
                r0[id] = czero();
                r1[id] = czero();
                r2[id] = czero();
            }
        });
        profile_sync("add_viscous_project");
        rhs_viscous_project_time += wall_seconds() - section_t0;
    }

    void step(double dt) {
        const double step_t0 = wall_seconds();
        const Real rdt = static_cast<Real>(dt);
        const std::size_t local_n = nspec;
        auto* u0 = u_hat[0];
        auto* u1 = u_hat[1];
        auto* u2 = u_hat[2];
        auto* r0 = rhs[0];
        auto* r1 = rhs[1];
        auto* r2 = rhs[2];
        auto* g0 = g[0];
        auto* g1 = g[1];
        auto* g2 = g[2];
        compute_rhs();
        double section_t0 = wall_seconds();
        ava::parallel_for<AVA_BLOCK_SIZE>(nullptr, 0, static_cast<int64_t>(local_n), [=] __device__(int64_t id) {
            g0[id] = r0[id];
            g1[id] = r1[id];
            g2[id] = r2[id];
            u0[id] = cadd(u0[id], cscale(g0[id], rdt / Real{3}));
            u1[id] = cadd(u1[id], cscale(g1[id], rdt / Real{3}));
            u2[id] = cadd(u2[id], cscale(g2[id], rdt / Real{3}));
        });
        profile_sync("rk stage 1 update");
        rk_update_time += wall_seconds() - section_t0;
        section_t0 = wall_seconds();
        project_truncate(u_hat[0], u_hat[1], u_hat[2]);
        profile_sync("rk stage 1 project");
        rk_project_time += wall_seconds() - section_t0;
        compute_rhs();
        section_t0 = wall_seconds();
        ava::parallel_for<AVA_BLOCK_SIZE>(nullptr, 0, static_cast<int64_t>(local_n), [=] __device__(int64_t id) {
            g0[id] = cadd(cscale(g0[id], -Real{5} / Real{9}), r0[id]);
            g1[id] = cadd(cscale(g1[id], -Real{5} / Real{9}), r1[id]);
            g2[id] = cadd(cscale(g2[id], -Real{5} / Real{9}), r2[id]);
            u0[id] = cadd(u0[id], cscale(g0[id], Real{15} * rdt / Real{16}));
            u1[id] = cadd(u1[id], cscale(g1[id], Real{15} * rdt / Real{16}));
            u2[id] = cadd(u2[id], cscale(g2[id], Real{15} * rdt / Real{16}));
        });
        profile_sync("rk stage 2 update");
        rk_update_time += wall_seconds() - section_t0;
        section_t0 = wall_seconds();
        project_truncate(u_hat[0], u_hat[1], u_hat[2]);
        profile_sync("rk stage 2 project");
        rk_project_time += wall_seconds() - section_t0;
        compute_rhs();
        section_t0 = wall_seconds();
        ava::parallel_for<AVA_BLOCK_SIZE>(nullptr, 0, static_cast<int64_t>(local_n), [=] __device__(int64_t id) {
            g0[id] = cadd(cscale(g0[id], -Real{153} / Real{128}), r0[id]);
            g1[id] = cadd(cscale(g1[id], -Real{153} / Real{128}), r1[id]);
            g2[id] = cadd(cscale(g2[id], -Real{153} / Real{128}), r2[id]);
            u0[id] = cadd(u0[id], cscale(g0[id], Real{8} * rdt / Real{15}));
            u1[id] = cadd(u1[id], cscale(g1[id], Real{8} * rdt / Real{15}));
            u2[id] = cadd(u2[id], cscale(g2[id], Real{8} * rdt / Real{15}));
        });
        profile_sync("rk stage 3 update");
        rk_update_time += wall_seconds() - section_t0;
        section_t0 = wall_seconds();
        project_truncate(u_hat[0], u_hat[1], u_hat[2]);
        profile_sync("rk stage 3 project");
        rk_project_time += wall_seconds() - section_t0;
#ifdef AVA_TARGET_CUDA
        cuda_check(cudaGetLastError(), "TaylorGreenSolver::step");
#endif
        step_time += wall_seconds() - step_t0;
        ++step_calls;
    }

    template <typename Expr>
    double reduce_sum(std::size_t nval, Expr expr) {
        const std::size_t blocks = (nval + AVA_BLOCK_SIZE - 1) / AVA_BLOCK_SIZE;
        auto* out = scratch;
        ava::parallel_for_block<AVA_BLOCK_SIZE>(nullptr, 0, static_cast<int64_t>(blocks),
                                                [=] __device__(int block, int tid) {
            __shared__ Real cache[AVA_BLOCK_SIZE];
            const std::size_t id = static_cast<std::size_t>(block) * AVA_BLOCK_SIZE + static_cast<std::size_t>(tid);
            cache[tid] = id < nval ? expr(id) : Real{0};
            __syncthreads();
            for (int stride = AVA_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    cache[tid] += cache[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                out[block] = cache[0];
            }
        });
        std::vector<Real> host(blocks, Real{0});
        ava::deep_copy(host.data(), scratch, blocks);
        double sum = 0.0;
        for (Real item : host) {
            sum += static_cast<double>(item);
        }
        return sum;
    }

    template <typename Expr>
    double reduce_max(std::size_t nval, Expr expr) {
        const std::size_t blocks = (nval + AVA_BLOCK_SIZE - 1) / AVA_BLOCK_SIZE;
        auto* out = scratch;
        ava::parallel_for_block<AVA_BLOCK_SIZE>(nullptr, 0, static_cast<int64_t>(blocks),
                                                [=] __device__(int block, int tid) {
            __shared__ Real cache[AVA_BLOCK_SIZE];
            const std::size_t id = static_cast<std::size_t>(block) * AVA_BLOCK_SIZE + static_cast<std::size_t>(tid);
            cache[tid] = id < nval ? expr(id) : Real{0};
            __syncthreads();
            for (int stride = AVA_BLOCK_SIZE / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    cache[tid] = cache[tid] > cache[tid + stride] ? cache[tid] : cache[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                out[block] = cache[0];
            }
        });
        std::vector<Real> host(blocks, Real{0});
        ava::deep_copy(host.data(), scratch, blocks);
        double value = 0.0;
        for (Real item : host) {
            value = std::max(value, static_cast<double>(item));
        }
        return value;
    }

    double compute_max_velocity() {
        const double t0 = wall_seconds();
        for (int c = 0; c < 3; ++c) {
            fft.backward(u_hat[c], phys[c]);
        }
        auto* p0 = phys[0];
        auto* p1 = phys[1];
        auto* p2 = phys[2];
        const double local = reduce_max(nphys, [=] __device__(std::size_t id) {
            return p0[id].x * p0[id].x + p1[id].x * p1[id].x + p2[id].x * p2[id].x;
        });
        const double value = std::sqrt(mpi_max(local));
        max_velocity_time += wall_seconds() - t0;
        ++max_velocity_calls;
        return value;
    }

    Diagnostics compute_diagnostics(double t) {
        const double t0 = wall_seconds();
        for (int c = 0; c < 3; ++c) {
            fft.backward(u_hat[c], phys[c]);
        }
        auto* p0 = phys[0];
        auto* p1 = phys[1];
        auto* p2 = phys[2];
        const double local_energy = reduce_sum(nphys, [=] __device__(std::size_t id) {
            return Real{0.5} * (p0[id].x * p0[id].x + p1[id].x * p1[id].x + p2[id].x * p2[id].x);
        });
        const double local_vmax2 = reduce_max(nphys, [=] __device__(std::size_t id) {
            return p0[id].x * p0[id].x + p1[id].x * p1[id].x + p2[id].x * p2[id].x;
        });

        auto* u0 = u_hat[0];
        auto* u1 = u_hat[1];
        auto* u2 = u_hat[2];
        auto* oh0 = shift_hat[0];
        auto* oh1 = shift_hat[1];
        auto* oh2 = shift_hat[2];
        auto* kx = mx;
        auto* ky = my;
        auto* kz = mz;
        const std::size_t spectral_n = nspec;
        ava::parallel_for<AVA_BLOCK_SIZE>(nullptr, 0, static_cast<int64_t>(spectral_n), [=] __device__(int64_t id) {
            const int ix = kx[id];
            const int iy = ky[id];
            const int iz = kz[id];
            oh0[id] = {-(real_from_int(iy) * u2[id].y - real_from_int(iz) * u1[id].y),
                       real_from_int(iy) * u2[id].x - real_from_int(iz) * u1[id].x};
            oh1[id] = {-(real_from_int(iz) * u0[id].y - real_from_int(ix) * u2[id].y),
                       real_from_int(iz) * u0[id].x - real_from_int(ix) * u2[id].x};
            oh2[id] = {-(real_from_int(ix) * u1[id].y - real_from_int(iy) * u0[id].y),
                       real_from_int(ix) * u1[id].x - real_from_int(iy) * u0[id].x};
        });
        for (int c = 0; c < 3; ++c) {
            fft.backward(shift_hat[c], shift_phys[c]);
        }
        auto* w0 = shift_phys[0];
        auto* w1 = shift_phys[1];
        auto* w2 = shift_phys[2];
        const double local_enstrophy = reduce_sum(nphys, [=] __device__(std::size_t id) {
            return Real{0.5} * (w0[id].x * w0[id].x + w1[id].x * w1[id].x + w2[id].x * w2[id].x);
        });
        const double local_div = reduce_max(nspec, [=] __device__(std::size_t id) {
            const Complex div = cadd(cadd(cscale(u0[id], kx[id]), cscale(u1[id], ky[id])), cscale(u2[id], kz[id]));
            return rsqrt(cabs2(div));
        });
        const double inv_global = 1.0 / static_cast<double>(global_size);
        const double energy = mpi_sum(local_energy) * inv_global;
        const double enstrophy = mpi_sum(local_enstrophy) * inv_global;
        const auto row = Diagnostics{t, energy, 2.0 * nu * enstrophy, enstrophy, mpi_max(local_div),
                                     std::sqrt(mpi_max(local_vmax2))};
        diagnostics_time += wall_seconds() - t0;
        ++diagnostics_calls;
        return row;
    }

    void print_profile(int rank, double init_time, double run_time) const {
        struct LocalProfile {
            double init;
            double run;
            double step;
            double rhs;
            double diagnostics;
            double max_velocity;
            double calls_step;
            double calls_rhs;
            double calls_diag;
            double calls_max_velocity;
        } local{init_time, run_time, step_time, rhs_time, diagnostics_time, max_velocity_time,
                static_cast<double>(step_calls), static_cast<double>(rhs_calls),
                static_cast<double>(diagnostics_calls), static_cast<double>(max_velocity_calls)};
#if SPECTRAL_DISTRIBUTED_FFT
        const double local_values[10] = {local.init, local.run, local.step, local.rhs, local.diagnostics,
                                         local.max_velocity, local.calls_step, local.calls_rhs,
                                         local.calls_diag, local.calls_max_velocity};
        double global[10] = {};
        MPI_Reduce(local_values, global, 10, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << std::setprecision(6)
                      << "profile_solver max_rank_seconds"
                      << " init=" << global[0]
                      << " run=" << global[1]
                      << " step=" << global[2]
                      << " rhs=" << global[3]
                      << " diagnostics=" << global[4]
                      << " max_velocity=" << global[5]
                      << " step_calls=" << static_cast<long long>(global[6])
                      << " rhs_calls=" << static_cast<long long>(global[7])
                      << " diagnostics_calls=" << static_cast<long long>(global[8])
                      << " max_velocity_calls=" << static_cast<long long>(global[9])
                      << "\n";
        }
#else
        if (rank == 0) {
            std::cout << std::setprecision(6)
                      << "profile_solver seconds"
                      << " init=" << local.init
                      << " run=" << local.run
                      << " step=" << local.step
                      << " rhs=" << local.rhs
                      << " diagnostics=" << local.diagnostics
                      << " max_velocity=" << local.max_velocity
                      << " step_calls=" << step_calls
                      << " rhs_calls=" << rhs_calls
                      << " diagnostics_calls=" << diagnostics_calls
                      << " max_velocity_calls=" << max_velocity_calls
                      << "\n";
        }
#endif
        struct Breakdown {
            double rhs_velocity_backward;
            double rhs_phase_shift;
            double rhs_shift_backward;
            double rhs_product_kernel;
            double rhs_product_forward;
            double rhs_accumulate;
            double rhs_viscous_project;
            double rk_update;
            double rk_project;
        } breakdown{rhs_velocity_backward_time, rhs_phase_shift_time, rhs_shift_backward_time,
                    rhs_product_kernel_time, rhs_product_forward_time, rhs_accumulate_time,
                    rhs_viscous_project_time, rk_update_time, rk_project_time};
#if SPECTRAL_DISTRIBUTED_FFT
        const double breakdown_local[9] = {breakdown.rhs_velocity_backward, breakdown.rhs_phase_shift,
                                           breakdown.rhs_shift_backward, breakdown.rhs_product_kernel,
                                           breakdown.rhs_product_forward, breakdown.rhs_accumulate,
                                           breakdown.rhs_viscous_project, breakdown.rk_update,
                                           breakdown.rk_project};
        double breakdown_global[9] = {};
        MPI_Reduce(breakdown_local, breakdown_global, 9, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << std::setprecision(6)
                      << "profile_breakdown max_rank_seconds"
                      << " rhs_velocity_backward=" << breakdown_global[0]
                      << " rhs_phase_shift=" << breakdown_global[1]
                      << " rhs_shift_backward=" << breakdown_global[2]
                      << " rhs_product_kernel=" << breakdown_global[3]
                      << " rhs_product_forward=" << breakdown_global[4]
                      << " rhs_accumulate=" << breakdown_global[5]
                      << " rhs_viscous_project=" << breakdown_global[6]
                      << " rk_update=" << breakdown_global[7]
                      << " rk_project=" << breakdown_global[8]
                      << "\n";
        }
#else
        if (rank == 0) {
            std::cout << std::setprecision(6)
                      << "profile_breakdown seconds"
                      << " rhs_velocity_backward=" << breakdown.rhs_velocity_backward
                      << " rhs_phase_shift=" << breakdown.rhs_phase_shift
                      << " rhs_shift_backward=" << breakdown.rhs_shift_backward
                      << " rhs_product_kernel=" << breakdown.rhs_product_kernel
                      << " rhs_product_forward=" << breakdown.rhs_product_forward
                      << " rhs_accumulate=" << breakdown.rhs_accumulate
                      << " rhs_viscous_project=" << breakdown.rhs_viscous_project
                      << " rk_update=" << breakdown.rk_update
                      << " rk_project=" << breakdown.rk_project
                      << "\n";
        }
#endif
        fft.print_profile(rank);
    }
};

TaylorGreenSolver::TaylorGreenSolver(int n, double reynolds)
    : impl_(std::make_unique<Impl>(n, reynolds)) {}

TaylorGreenSolver::~TaylorGreenSolver() = default;

void TaylorGreenSolver::step_rk3(double dt) {
    impl_->step(dt);
}

Diagnostics TaylorGreenSolver::diagnostics(double t) {
    return impl_->compute_diagnostics(t);
}

double TaylorGreenSolver::max_velocity() {
    return impl_->compute_max_velocity();
}

void TaylorGreenSolver::print_profile(int rank, double init_time, double run_time) const {
    impl_->print_profile(rank, init_time, run_time);
}

void run_taylor_green(const RunConfig& config) {
    if (config.dt <= 0.0 || config.cfl <= 0.0 || config.output_dt <= 0.0) {
        throw std::runtime_error("--dt, --cfl, and --output-dt must be positive");
    }
    const int rank = mpi_rank();
    if (rank == 0) {
        std::filesystem::create_directories(config.output_dir);
    }
#if SPECTRAL_DISTRIBUTED_FFT
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    const std::string csv_path = config.output_dir + "/taylor_green_diagnostics.csv";
    std::ofstream csv;
    if (rank == 0) {
        csv.open(csv_path);
        csv << std::setprecision(17);
        csv << "t,kinetic_energy,dissipation,enstrophy,max_spectral_divergence,max_velocity,dt,cfl,main_dt,main_cfl\n";
    }

    const double init_t0 = wall_seconds();
    TaylorGreenSolver solver(config.n, config.reynolds);
    const double init_time = wall_seconds() - init_t0;
    const double run_t0 = wall_seconds();
    double t = 0.0;
    int step = 0;
    int output_count = 0;
    auto write_row = [&](double time, bool force_print, double last_dt, double last_cfl, double main_dt, double main_cfl) {
        const auto row = solver.diagnostics(time);
        if (rank == 0) {
            csv << row.t << ',' << row.kinetic_energy << ',' << row.dissipation << ',' << row.enstrophy << ','
                << row.max_spectral_divergence << ',' << row.max_velocity << ',' << last_dt << ',' << last_cfl
                << ',' << main_dt << ',' << main_cfl << '\n';
            csv.flush();
            const bool print = force_print || output_count % config.print_every == 0;
            if (print) {
                std::cout << std::setprecision(10) << "step " << step << " t=" << row.t << " E=" << row.kinetic_energy
                          << " eps=" << row.dissipation << " max_u=" << row.max_velocity
                          << " dt=" << last_dt << " cfl=" << last_cfl
                          << " main_dt=" << main_dt << " main_cfl=" << main_cfl << "\n";
            }
        }
        ++output_count;
        return row;
    };

    if (config.adaptive_cfl) {
        double next_output = 0.0;
        write_row(t, true, 0.0, 0.0, 0.0, 0.0);
        next_output += config.output_dt;
        while (t < config.t_end - 1.0e-14) {
            if (step >= config.max_steps) {
                throw std::runtime_error("adaptive CFL run exceeded --max-steps before reaching --t-end");
            }
            const double vmax = solver.max_velocity();
            double main_dt = config.dt;
            if (vmax > 1.0e-14) {
                main_dt = std::min(main_dt, config.cfl * (2.0 * pi / static_cast<double>(config.n)) / vmax);
            }
            const double main_cfl = vmax * main_dt / (2.0 * pi / static_cast<double>(config.n));
            double dt_step = main_dt;
            dt_step = std::min(dt_step, config.t_end - t);
            if (next_output <= config.t_end + 1.0e-14) {
                dt_step = std::min(dt_step, next_output - t);
            }
            if (dt_step <= 1.0e-14) {
                next_output += config.output_dt;
                continue;
            }
            const double actual_cfl = vmax * dt_step / (2.0 * pi / static_cast<double>(config.n));
            solver.step_rk3(dt_step);
            t += dt_step;
            ++step;
            if (t >= next_output - 1.0e-12 || t >= config.t_end - 1.0e-12) {
                write_row(t, t >= config.t_end - 1.0e-12, dt_step, actual_cfl, main_dt, main_cfl);
                while (next_output <= t + 1.0e-12) {
                    next_output += config.output_dt;
                }
            }
        }
    } else {
        for (step = 0; step <= config.steps; ++step) {
            if (step % config.output_every == 0 || step == config.steps) {
                const double fixed_cfl = solver.max_velocity() * config.dt / (2.0 * pi / static_cast<double>(config.n));
                write_row(t, step == config.steps, step == 0 ? 0.0 : config.dt, step == 0 ? 0.0 : fixed_cfl,
                          step == 0 ? 0.0 : config.dt, step == 0 ? 0.0 : fixed_cfl);
            }
            if (step != config.steps) {
                solver.step_rk3(config.dt);
                t += config.dt;
            }
        }
    }
    const double run_time = wall_seconds() - run_t0;
#if SPECTRAL_DISTRIBUTED_FFT
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    solver.print_profile(rank, init_time, run_time);
    if (rank == 0) {
        std::cout << "saved diagnostics: " << csv_path << "\n";
    }
}

}  // namespace spectral::ava_solver
