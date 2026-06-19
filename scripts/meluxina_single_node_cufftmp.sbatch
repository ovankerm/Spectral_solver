#!/bin/bash -l
#SBATCH --job-name=spectral_cufftmp
#SBATCH --nodes=1
#SBATCH -A p201207
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=out/meluxina-cufftmp-%j.out
#SBATCH --error=out/meluxina-cufftmp-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p out

module load OpenMPI/5.0.7-NVHPC-25.3-CUDA-12.8.0

easybuild_root="/apps/USE/easybuild/release/2025.1/software"
cmake_root="${easybuild_root}/CMake/3.31.3-GCCcore-14.2.0"
fftw_root="${easybuild_root}/FFTW/3.3.10-GCC-14.2.0"
hdf5_root="${easybuild_root}/HDF5/1.14.6-gompi-2025a"
cuda_root="${easybuild_root}/CUDA/12.8.0"
nvhpc_root="${easybuild_root}/NVHPC/25.3-CUDA-12.8.0/Linux_x86_64/25.3"
cufftmp_lib="${nvhpc_root}/math_libs/12.8/targets/x86_64-linux/lib"
cufftmp_include="${nvhpc_root}/math_libs/12.8/targets/x86_64-linux/include"
nvshmem_lib="${nvhpc_root}/comm_libs/12.8/nvshmem/lib"

export PATH="${cmake_root}/bin:${PATH}"
export CUDAToolkit_ROOT="${cuda_root}"
export CUFFTMP_HOME="${nvhpc_root}"
export NVHPC_ROOT="${nvhpc_root}"
export NVSHMEM_HOME="${nvhpc_root}/comm_libs/12.8/nvshmem"
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_MPI_SUPPORT=1
export NVSHMEM_DISABLE_GDRCOPY=1
export LD_LIBRARY_PATH="${fftw_root}/lib:${hdf5_root}/lib:${cuda_root}/lib64:${cufftmp_lib}:${nvshmem_lib}:${LD_LIBRARY_PATH:-}"
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
echo "CUFFTMP_HOME=${CUFFTMP_HOME}"
echo "NVSHMEM_HOME=${NVSHMEM_HOME}"
ls -l "${cufftmp_include}/cufftMp.h" "${cufftmp_lib}/libcufftMp.so" "${nvshmem_lib}/libnvshmem_host.so"

grid_size="${SPECTRAL_GRID_SIZE:-256}"
precision="${SPECTRAL_PRECISION:-SINGLE}"
precision_label="float64"
if [ "${precision}" = "SINGLE" ]; then
  precision_label="float32"
fi
ranks="${SLURM_NTASKS:-4}"
build_dir="build-meluxina-cuda-cufftmp-${precision_label}"
bench_root="${SLURM_SUBMIT_DIR}/out/meluxina_cufftmp_${precision_label}_${grid_size}_${SLURM_JOB_ID}"
mkdir -p "${bench_root}"

if [ "${SPECTRAL_CLEAN_BUILD:-0}" = "1" ]; then
  rm -rf "${build_dir}"
fi

cmake -S . -B "${build_dir}" \
  -DAVA_TARGET=CUDA \
  -DSPECTRAL_FFT_BACKEND=CUFFTMP \
  -DSPECTRAL_PRECISION="${precision}" \
  -DSPECTRAL_CUFFTMP_ROOT="${CUFFTMP_HOME}" \
  -DCUFFTMP_INCLUDE_DIR="${cufftmp_include}" \
  -DCUFFTMP_LIBRARY="${cufftmp_lib}/libcufftMp.so" \
  -DCUFFTMP_NVSHMEM_HOST_LIBRARY="${nvshmem_lib}/libnvshmem_host.so" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER="${cuda_root}/bin/nvcc" \
  -DFFTW_INCLUDE_DIR="${fftw_root}/include" \
  -DFFTW_LIBRARY="${fftw_root}/lib/libfftw3.so" \
  -DHDF5_INCLUDE_DIR="${hdf5_root}/include" \
  -DHDF5_LIBRARY="${hdf5_root}/lib/libhdf5.so"

build_jobs=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))
cmake --build "${build_dir}" --clean-first --target taylor_green -j "${build_jobs:-8}"

name="cuda_cufftmp_${precision_label}_${grid_size}_np${ranks}"
out_dir="${bench_root}/${name}"
mkdir -p "${out_dir}"

start="$(date +%s.%N)"
srun --cpu-bind=none -n "${ranks}" "./${build_dir}/taylor_green" \
  --n "${grid_size}" \
  --re 1600 \
  --adaptive-cfl 1 \
  --dt 0.5 \
  --cfl 0.5 \
  --t-end 20 \
  --output-dt 1 \
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
