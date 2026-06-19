#!/bin/bash -l
#SBATCH --partition=debug-gpu
#SBATCH --account=examples
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:A100:4
#SBATCH --exclusive
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

nvhpc_version="${EBROOTNVHPC}/Linux_x86_64/23.7"
cufftmp_lib="${nvhpc_version}/math_libs/12.2/targets/x86_64-linux/lib"
nvshmem_lib="${nvhpc_version}/comm_libs/12.2/nvshmem_cufftmp_compat/lib"
export CUFFTMP_HOME="${nvhpc_version}"
export NVSHMEM_HOME="${nvhpc_version}/comm_libs/12.2/nvshmem_cufftmp_compat"
export LD_LIBRARY_PATH="${cufftmp_lib}:${nvshmem_lib}:${LD_LIBRARY_PATH:-}"
export NVSHMEM_BOOTSTRAP=MPI
export NVSHMEM_MPI_SUPPORT=1
export NVSHMEM_DISABLE_GDRCOPY=1
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

grid_size="${SPECTRAL_GRID_SIZE:-64}"
build_dir="build-lucia-cuda-cufftmp-job"
bench_root="${SLURM_SUBMIT_DIR}/out/lucia_cufftmp_${grid_size}_${SLURM_JOB_ID}"
mkdir -p "${bench_root}"

cmake -S . -B "${build_dir}" \
  -DAVA_TARGET=CUDA \
  -DSPECTRAL_FFT_BACKEND=CUFFTMP \
  -DSPECTRAL_CUFFTMP_ROOT="${CUFFTMP_HOME}" \
  -DCUFFTMP_INCLUDE_DIR="${nvhpc_version}/math_libs/12.2/targets/x86_64-linux/include/cufftmp" \
  -DCUFFTMP_LIBRARY="${cufftmp_lib}/libcufftMp.so" \
  -DCUFFTMP_NVSHMEM_HOST_LIBRARY="${nvshmem_lib}/libnvshmem_host.so" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DFFTW_INCLUDE_DIR="${EBROOTFFTW}/include" \
  -DFFTW_LIBRARY="${EBROOTFFTW}/lib/libfftw3.so" \
  -DHDF5_INCLUDE_DIR="${EBROOTHDF5}/include" \
  -DHDF5_LIBRARY="${EBROOTHDF5}/lib/libhdf5.so"

build_jobs=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))
cmake --build "${build_dir}" --target taylor_green -j "${build_jobs:-8}"

name="cuda_cufftmp_${grid_size}_np4"
out_dir="${bench_root}/${name}"
mkdir -p "${out_dir}"

start="$(date +%s.%N)"
srun -n 4 --cpu-bind=cores "./${build_dir}/taylor_green" \
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
tail -40 "${bench_root}/${name}.log"

echo "== done =="
date
echo "${bench_root}"
