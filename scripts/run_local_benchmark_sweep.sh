#!/usr/bin/env bash
set -u

bench_root="${1:-/tmp/spectral_benchmark_sweep_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${bench_root}"

common_args=(
  --re 1600
  --adaptive-cfl 1
  --dt 0.5
  --cfl 0.5
  --t-end 20
  --output-dt 1
  --print-every 20
)

run_case() {
  local name="$1"
  local build_dir="$2"
  local ranks="$3"
  local grid="$4"
  shift 4
  local extra_cmd=("$@")
  local case_dir="${bench_root}/${name}"
  mkdir -p "${case_dir}"
  {
    echo "case=${name}"
    echo "build_dir=${build_dir}"
    echo "ranks=${ranks}"
    echo "grid=${grid}"
    echo "start=$(date -Is)"
  } > "${case_dir}/metadata.txt"

  local status=0
  local wall_start wall_end wall_ms
  wall_start="$(date +%s%3N)"
  if [[ "${ranks}" == "1" && "${extra_cmd[0]:-}" != "mpirun" ]]; then
    "${extra_cmd[@]}" --n "${grid}" "${common_args[@]}" --output-dir "${case_dir}/output" \
      > "${case_dir}/stdout.txt" 2> "${case_dir}/stderr.txt" || status=$?
  else
    "${extra_cmd[@]}" --n "${grid}" "${common_args[@]}" --output-dir "${case_dir}/output" \
      > "${case_dir}/stdout.txt" 2> "${case_dir}/stderr.txt" || status=$?
  fi
  wall_end="$(date +%s%3N)"
  wall_ms=$((wall_end - wall_start))
  echo "wall_seconds=$(awk "BEGIN { printf \"%.3f\", ${wall_ms} / 1000.0 }")" >> "${case_dir}/metadata.txt"
  echo "status=${status}" >> "${case_dir}/metadata.txt"
  echo "end=$(date -Is)" >> "${case_dir}/metadata.txt"
  return "${status}"
}

run_case cpu_fftw_64 build-bench-cpu-fftw 1 64 ./build-bench-cpu-fftw/taylor_green
run_case cpu_flups_np1_64 build-bench-cpu-flups 1 64 mpirun -np 1 ./build-bench-cpu-flups/taylor_green
run_case cpu_flups_np2_64 build-bench-cpu-flups 2 64 mpirun -np 2 ./build-bench-cpu-flups/taylor_green
run_case cpu_flups_np4_64 build-bench-cpu-flups 4 64 mpirun -np 4 ./build-bench-cpu-flups/taylor_green
run_case cpu_flups_np8_64 build-bench-cpu-flups 8 64 mpirun -np 8 ./build-bench-cpu-flups/taylor_green
run_case cuda_cufft_64 build-bench-cuda-cufft 1 64 ./build-bench-cuda-cufft/taylor_green
run_case cuda_cufft_128 build-bench-cuda-cufft 1 128 ./build-bench-cuda-cufft/taylor_green

echo "${bench_root}"
