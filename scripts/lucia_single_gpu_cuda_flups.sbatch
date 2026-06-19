#!/bin/bash -l
#SBATCH --partition=debug-gpu
#SBATCH --account=examples
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --time=01:00:00
#SBATCH -e out/slurm-%j.err
#SBATCH -o out/slurm-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module load EasyBuild/2023a
module load GCC/12.3.0
module load FFTW/3.3.10-GCC-12.3.0
module load HDF5/1.14.0-gompi-2023a
module load CUDA/12.2.0
module load NVHPC/23.7-CUDA-12.2.0
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.2.0
module load CMake/3.26.3-GCCcore-12.3.0

mkdir -p out

export SPECTRAL_ASSUME_CUDA_AWARE_MPI=ON
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "== job =="
date
hostname
nvidia-smi

echo "== toolchain =="
cmake --version | head -1
nvcc --version
mpirun --version | head -5
echo "SPECTRAL_ASSUME_CUDA_AWARE_MPI=${SPECTRAL_ASSUME_CUDA_AWARE_MPI}"

grid_size="${SPECTRAL_GRID_SIZE:-64}"
t_end="${SPECTRAL_T_END:-20}"
output_dt="${SPECTRAL_OUTPUT_DT:-1}"
build_dir="build-lucia-cuda-flups-single-job"
bench_root="${SLURM_SUBMIT_DIR}/out/lucia_cuda_flups_single_${grid_size}_${SLURM_JOB_ID}"
flups_source_dir="${SPECTRAL_FLUPS_SOURCE_DIR:-${SLURM_SUBMIT_DIR}/third_party_sources/flups-dev-gpu}"
mkdir -p "${bench_root}"
if [ "${SPECTRAL_CLEAN_BUILD:-0}" = "1" ]; then
  rm -rf "${build_dir}"
fi

cmake -S . -B "${build_dir}" \
  -DAVA_TARGET=CUDA \
  -DSPECTRAL_FFT_BACKEND=FLUPS \
  -DSPECTRAL_FLUPS_ENABLE_CUDA=ON \
  -DSPECTRAL_ASSUME_CUDA_AWARE_MPI=ON \
  -DSPECTRAL_FLUPS_CUDA_OPTS="-DBACKEND_CUDA -DHAVE_CUFFT -Dm_hipfft_check_errors=m_cufft_check_errors" \
  -DSPECTRAL_FLUPS_SOURCE_DIR="${flups_source_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DFFTW_INCLUDE_DIR="${EBROOTFFTW}/include" \
  -DFFTW_LIBRARY="${EBROOTFFTW}/lib/libfftw3.so" \
  -DHDF5_INCLUDE_DIR="${EBROOTHDF5}/include" \
  -DHDF5_LIBRARY="${EBROOTHDF5}/lib/libhdf5.so"

cmake --build "${build_dir}" --target taylor_green -j "${SLURM_CPUS_PER_TASK:-8}"

name="cuda_flups_${grid_size}_np1"
out_dir="${bench_root}/${name}"
mkdir -p "${out_dir}"

start="$(date +%s.%N)"
srun -n 1 --cpu-bind=cores "./${build_dir}/taylor_green" \
  --n "${grid_size}" \
  --re 1600 \
  --adaptive-cfl 1 \
  --dt 0.5 \
  --cfl 0.5 \
  --t-end "${t_end}" \
  --output-dt "${output_dt}" \
  --print-every 1 \
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
