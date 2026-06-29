#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/.diffusion_pipe_env" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.diffusion_pipe_env"
fi
REPO_DIR="${DIFFUSION_PIPE_REPO_DIR:-${SCRIPT_DIR}}"
MAMBA_ROOT="${DIFFUSION_PIPE_MAMBA_ROOT:-/mnt/c/ai/micromamba}"
MAMBA="${MAMBA_ROOT}/bin/micromamba"
ENV_PREFIX="${DIFFUSION_PIPE_ENV_PREFIX:-${MAMBA_ROOT}/envs/diffusion-pipe}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}/root"

CUDA_HOME_CANDIDATE="$(find "${ENV_PREFIX}/lib" -path "*/site-packages/nvidia/cu13" -type d 2>/dev/null | head -n 1 || true)"
if [[ -n "${CUDA_HOME_CANDIDATE}" ]]; then
  export CUDA_HOME="${CUDA_HOME_CANDIDATE}"
  export PATH="${CUDA_HOME}/bin:${ENV_PREFIX}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
else
  export PATH="${ENV_PREFIX}/bin:${PATH}"
  export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
export TRITON_CACHE_DIR="${HOME}/.triton/cache"

mkdir -p "${TRITON_CACHE_DIR}" "${HOME}/.triton/autotune"
cd "${REPO_DIR}"

exec "${MAMBA}" run -p "${ENV_PREFIX}" "$@"
