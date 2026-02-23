#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VENV_PATH:-/home/bi_admin/venv_rfm}"
WARP_LANG_VERSION="${WARP_LANG_VERSION:-1.0.2}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -d "${VENV_PATH}" ]]; then
    echo "[ERROR] venv not found: ${VENV_PATH}" >&2
    echo "[ERROR] Set VENV_PATH or create the venv first." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

echo "[INFO] Installing warp-lang==${WARP_LANG_VERSION}"
python -m pip install "warp-lang==${WARP_LANG_VERSION}"

echo "[INFO] Verifying warp import"
python - <<'PY'
import warp as wp

wp.init()
print("WARP_OK", getattr(wp, "__version__", "unknown"))
PY
