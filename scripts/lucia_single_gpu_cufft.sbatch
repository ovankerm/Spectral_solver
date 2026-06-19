#!/bin/bash -l
#SBATCH --partition=debug-gpu
#SBATCH --account=examples
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --time=00:20:00
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

echo "== job =="
date
hostname
nvidia-smi

echo "== toolchain =="
cmake --version | head -1
nvcc --version
mpirun --version | head -5

grid_size="${SPECTRAL_GRID_SIZE:-128}"
precision="${SPECTRAL_PRECISION:-DOUBLE}"
t_end="${SPECTRAL_T_END:-20}"
output_dt="${SPECTRAL_OUTPUT_DT:-1}"
print_every="${SPECTRAL_PRINT_EVERY:-20}"
precision_label="float64"
if [ "${precision}" = "SINGLE" ]; then
  precision_label="float32"
fi
build_dir="build-lucia-cuda-cufft-${precision_label}"
bench_root="${SLURM_SUBMIT_DIR}/out/lucia_cufft_${precision_label}_${grid_size}_${SLURM_JOB_ID}"
mkdir -p "${bench_root}"

cmake -S . -B "${build_dir}" \
  -DAVA_TARGET=CUDA \
  -DSPECTRAL_FFT_BACKEND=CUFFT \
  -DSPECTRAL_PRECISION="${precision}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DFFTW_INCLUDE_DIR="${EBROOTFFTW}/include" \
  -DFFTW_LIBRARY="${EBROOTFFTW}/lib/libfftw3.so" \
  -DHDF5_INCLUDE_DIR="${EBROOTHDF5}/include" \
  -DHDF5_LIBRARY="${EBROOTHDF5}/lib/libhdf5.so"

cmake --build "${build_dir}" --target taylor_green -j "${SLURM_CPUS_PER_TASK:-8}"

name="cuda_cufft_${precision_label}_${grid_size}"
out_dir="${bench_root}/${name}"
mkdir -p "${out_dir}"

start="$(date +%s.%N)"
srun -n 1 "./${build_dir}/taylor_green" \
  --n "${grid_size}" \
  --re 1600 \
  --adaptive-cfl 1 \
  --dt 0.5 \
  --cfl 0.5 \
  --t-end "${t_end}" \
  --output-dt "${output_dt}" \
  --print-every "${print_every}" \
  --output-dir "${out_dir}" \
  > "${bench_root}/${name}.log" 2>&1
status="$?"
end="$(date +%s.%N)"
awk -v s="${start}" -v e="${end}" -v st="${status}" \
  'BEGIN{printf "status %d\nreal %.6f\n", st, e-s}' \
  > "${bench_root}/${name}.time"
cat "${bench_root}/${name}.time"
tail -40 "${bench_root}/${name}.log"

echo "== done =="
date
echo "${bench_root}"
