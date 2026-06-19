#!/bin/bash -l
#SBATCH --job-name=spectral_cuda_flups
#SBATCH --nodes=1
#SBATCH -A p201207
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=out/meluxina-%j.out
#SBATCH --error=out/meluxina-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p out

module load OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0

easybuild_root="/apps/USE/easybuild/release/2025.1/software"
cmake_root="${easybuild_root}/CMake/3.31.3-GCCcore-14.2.0"
fftw_root="${easybuild_root}/FFTW/3.3.10-GCC-14.2.0"
hdf5_root="${easybuild_root}/HDF5/1.14.6-gompi-2025a"
cuda_root="${easybuild_root}/CUDA/12.8.0"

export PATH="${cmake_root}/bin:${PATH}"
export CUDAToolkit_ROOT="${cuda_root}"
export LD_LIBRARY_PATH="${fftw_root}/lib:${hdf5_root}/lib:${cuda_root}/lib64:${LD_LIBRARY_PATH:-}"
export SPECTRAL_ASSUME_CUDA_AWARE_MPI=ON
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "== job =="
date
hostname
nvidia-smi

echo "== toolchain =="
which cmake || true
cmake --version | head -1
which nvcc || true
nvcc --version
mpirun --version | head -5
echo "SPECTRAL_ASSUME_CUDA_AWARE_MPI=${SPECTRAL_ASSUME_CUDA_AWARE_MPI}"

grid_size="${SPECTRAL_GRID_SIZE:-64}"
t_end="${SPECTRAL_T_END:-20}"
output_dt="${SPECTRAL_OUTPUT_DT:-1}"
ranks="${SLURM_NTASKS:-4}"
build_dir="build-meluxina-cuda-flups-job"
bench_root="${SLURM_SUBMIT_DIR}/out/meluxina_cuda_flups_${grid_size}_${SLURM_JOB_ID}"
flups_source_dir="${SPECTRAL_FLUPS_SOURCE_DIR:-${SLURM_SUBMIT_DIR}/third_party_sources/flups-dev-gpu}"
mkdir -p "${bench_root}"

if [ "${SPECTRAL_CLEAN_BUILD:-0}" = "1" ]; then
  rm -rf "${build_dir}"
fi

flups_args=()
if [ -d "${flups_source_dir}" ]; then
  flups_args+=("-DSPECTRAL_FLUPS_SOURCE_DIR=${flups_source_dir}")
fi

cmake -S . -B "${build_dir}" \
  -DAVA_TARGET=CUDA \
  -DSPECTRAL_FFT_BACKEND=FLUPS \
  -DSPECTRAL_FLUPS_ENABLE_CUDA=ON \
  -DSPECTRAL_ASSUME_CUDA_AWARE_MPI=ON \
  -DSPECTRAL_FLUPS_CUDA_OPTS="-DBACKEND_CUDA -DHAVE_CUFFT -Dm_hipfft_check_errors=m_cufft_check_errors" \
  "${flups_args[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER="${cuda_root}/bin/nvcc" \
  -DFFTW_INCLUDE_DIR="${fftw_root}/include" \
  -DFFTW_LIBRARY="${fftw_root}/lib/libfftw3.so" \
  -DHDF5_INCLUDE_DIR="${hdf5_root}/include" \
  -DHDF5_LIBRARY="${hdf5_root}/lib/libhdf5.so"

build_jobs=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))
cmake --build "${build_dir}" --clean-first --target taylor_green -j "${build_jobs:-4}"

name="cuda_flups_${grid_size}_np${ranks}"
out_dir="${bench_root}/${name}"
mkdir -p "${out_dir}"

start="$(date +%s.%N)"
srun --cpu-bind=none -n "${ranks}" "./${build_dir}/taylor_green" \
  --n "${grid_size}" \
  --re 1600 \
  --adaptive-cfl 1 \
  --dt 0.5 \
  --cfl 0.5 \
  --t-end "${t_end}" \
  --output-dt "${output_dt}" \
  --print-every 20 \
  --output-dir "${out_dir}" \
  > "${bench_root}/${name}.log" 2>&1
status="$?"
end="$(date +%s.%N)"
awk -v s="${start}" -v e="${end}" -v st="${status}" \
  'BEGIN{printf "status %d\nreal %.6f\n", st, e-s}' \
  > "${bench_root}/${name}.time"

cat "${bench_root}/${name}.time"
tail -80 "${bench_root}/${name}.log"

echo "== done =="
date
echo "${bench_root}"
