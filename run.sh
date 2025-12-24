#!/usr/bin/env bash
# Helper script to start the FastAPI server with the correct conda env and model assets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configurable knobs (env vars win, otherwise sensible defaults)
ENV_NAME="${CTC_ENV:-ctc}"
MODEL_DIR="${CTC_MODEL_DIR:-${SCRIPT_DIR}/transcribe_bundle/checkpoint}"
VOCAB_PATH="${CTC_VOCAB_PATH:-${SCRIPT_DIR}/transcribe_bundle/vocab.json}"
HOST="${FASTAPI_HOST:-0.0.0.0}"
PORT="${FASTAPI_PORT:-8011}"
RELOAD="${FASTAPI_RELOAD:-0}"

# Ensure conda is available
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH. Please install Anaconda/Miniconda and try again." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base 2>/dev/null)"
if [[ -z "${CONDA_BASE}" || ! -d "${CONDA_BASE}" ]]; then
  echo "Could not resolve conda base path. Check your conda installation." >&2
  exit 1
fi

if [[ ! -d "${CONDA_BASE}/envs/${ENV_NAME}" ]]; then
  echo "Conda env '${ENV_NAME}' not found. Create it or set CTC_ENV to an existing env." >&2
  exit 1
fi

# Activate env
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

PY_BIN="$(command -v python)"
UVICORN_BIN="$(command -v uvicorn || true)"

# Validate uvicorn is available in the target env
if [[ -z "${UVICORN_BIN}" ]]; then
  echo "uvicorn is not installed in env '${ENV_NAME}'. Install it (e.g., pip install uvicorn[standard])." >&2
  exit 1
fi

# Ensure torch is available before starting
if ! "${PY_BIN}" - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
PY
then
  echo "torch is not installed in env '${ENV_NAME}'. Install it (pip install torch --extra-index-url https://download.pytorch.org/whl/cu121, or CPU wheel)." >&2
  exit 1
fi

# Validate model assets
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Model directory not found: ${MODEL_DIR}. Set CTC_MODEL_DIR to the correct path." >&2
  exit 1
fi

if [[ ! -f "${VOCAB_PATH}" ]]; then
  echo "Vocabulary JSON not found: ${VOCAB_PATH}. Set CTC_VOCAB_PATH to the correct path." >&2
  exit 1
fi

export CTC_MODEL_DIR="${MODEL_DIR}"
export CTC_VOCAB_PATH="${VOCAB_PATH}"

echo "Environment  : ${ENV_NAME}"
echo "Model dir    : ${CTC_MODEL_DIR}"
echo "Vocab path   : ${CTC_VOCAB_PATH}"
echo "Host/Port    : ${HOST}:${PORT}"
echo "Reload mode  : ${RELOAD}"
echo "Python       : ${PY_BIN}"
echo "Uvicorn      : ${UVICORN_BIN}"
echo "Starting FastAPI..."

if [[ "${RELOAD}" == "1" ]]; then
  exec "${UVICORN_BIN}" fastapi_app:APP --host "${HOST}" --port "${PORT}" --reload
else
  exec "${UVICORN_BIN}" fastapi_app:APP --host "${HOST}" --port "${PORT}"
fi
