#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
FP_DIR="${REPO_ROOT}/thirdparty/FoundationPose"
VENV_PATH="${VENV_PATH:-/home/bi_admin/venv_rfm}"
APT_PACKAGES=(
  build-essential
  cmake
  libeigen3-dev
  libboost-system-dev
  libboost-program-options-dev
  pybind11-dev
)

backup_path() {
  local target="$1"
  if [[ -e "$target" ]]; then
    local backup="${target}_backup_$(date +%Y%m%d_%H%M%S)"
    mv "$target" "$backup"
    echo "[INFO] backup: ${target} -> ${backup}"
  fi
}

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] FoundationPose dir: ${FP_DIR}"
echo "[INFO] venv path: ${VENV_PATH}"

if [[ ! -d "${FP_DIR}" ]]; then
  echo "[ERROR] FoundationPose not found at ${FP_DIR}" >&2
  exit 1
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "[INFO] Creating new venv at ${VENV_PATH}"
  python3 -m venv "${VENV_PATH}"
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

echo "[INFO] Python: $(which python)"
python --version

if command -v dpkg-query >/dev/null 2>&1; then
  missing_apt=()
  for pkg in "${APT_PACKAGES[@]}"; do
    if ! dpkg-query -W -f='${Status}' "${pkg}" 2>/dev/null | grep -q "install ok installed"; then
      missing_apt+=("${pkg}")
    fi
  done
  if [[ ${#missing_apt[@]} -gt 0 ]]; then
    echo "[WARN] Missing apt packages: ${missing_apt[*]}"
    echo "[WARN] Install them first:"
    echo "       sudo apt-get update && sudo apt-get install -y ${missing_apt[*]}"
  fi
fi

echo "[INFO] Installing Python dependencies"
python -m pip install wheel
python -m pip install -r "${REPO_ROOT}/requirements_2204.txt"

echo "[INFO] Building mycpp (pybind extension)"
cd "${FP_DIR}/mycpp"
if [[ -d "build" ]]; then
  if [[ -w "build" ]]; then
    rm -rf build/*
  else
    backup_path "build"
  fi
fi
mkdir -p build
cd build
cmake .. -DPYTHON_EXECUTABLE="$(which python)"
make -j"$(nproc)"
ls -1 mycpp*.so

echo "[INFO] Building mycuda (CUDA extensions)"
cd "${FP_DIR}/bundlesdf/mycuda"
if [[ -f "common.cpython-38-x86_64-linux-gnu.so" ]]; then
  backup_path "common.cpython-38-x86_64-linux-gnu.so"
fi
if [[ -f "gridencoder.cpython-38-x86_64-linux-gnu.so" ]]; then
  backup_path "gridencoder.cpython-38-x86_64-linux-gnu.so"
fi
if [[ -d "build" ]]; then
  if [[ -w "build" ]]; then
    rm -rf build/*
  else
    backup_path "build"
  fi
fi
if [[ -d "common.egg-info" ]]; then
  if [[ -w "common.egg-info" ]]; then
    rm -rf common.egg-info
  else
    backup_path "common.egg-info"
  fi
fi
python -m pip install -e . --no-build-isolation

echo "[INFO] Verifying imports"
export PYTHONPATH="${FP_DIR}:${PYTHONPATH:-}"
python - <<'PY'
import torch
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
import Utils

assert Utils.mycpp is not None, "mycpp import failed"
assert hasattr(Utils.mycpp, "cluster_poses"), "mycpp.cluster_poses missing"
print("IMPORT_OK:", FoundationPose.__name__, ScorePredictor.__name__, PoseRefinePredictor.__name__)
print("mycpp:", Utils.mycpp)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
PY

echo "[INFO] Done"
echo "[INFO] If you use pose_tracker.py directly, keep FoundationPose on PYTHONPATH or add it via sys.path."
