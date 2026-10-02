#!/usr/bin/env bash
# One-shot environment setup for a GPU server (downloads default to the
# Tsinghua PyPI mirror, which is fast from mainland China).
#
#   bash scripts/setup_server.sh          # uv-managed environment (recommended)
#   bash scripts/setup_server.sh --pip    # install the non-torch deps into the
#                                      # currently active Python (e.g. a conda
#                                      # env that already ships a working torch)
#
# Override the mirror (e.g. outside China):
#   UV_DEFAULT_INDEX=https://pypi.org/simple bash scripts/setup_server.sh
set -euo pipefail

MIRROR="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
DEPS=(
    "numpy>=2.0"
    "pandas>=2.0"
    "scipy>=1.11"
    "scikit-learn>=1.4"
    "matplotlib>=3.8"
    "requests"
)

if [[ "${1:-}" == "--pip" ]]; then
    echo "[setup] pip install from ${MIRROR}"
    echo "[setup] target interpreter: $(python -c 'import sys; print(sys.executable)')"
    python -m pip install -i "${MIRROR}" "${DEPS[@]}" pytest ruff
else
    if ! command -v uv >/dev/null 2>&1; then
        echo "[setup] uv not found. Install it first (no root needed):" >&2
        echo "        curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        exit 1
    fi
    echo "[setup] uv sync from ${MIRROR} (CPython 3.12 + locked dependencies)"
    UV_DEFAULT_INDEX="${MIRROR}" uv sync
fi

echo "[setup] done."
echo "[setup] verify the GPU with:  python scripts/check_devices.py"
