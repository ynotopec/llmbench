#!/usr/bin/env bash
# run.sh — source-compatible, systemd-compatible
# Usage:
#   ./run.sh [IP] [PORT]          # exec and run app.py
#   source run.sh [IP] [PORT]     # configure env only, keep shell
#   systemd: source run.sh 0.0.0.0 8000
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$SCRIPT_DIR")"
VENV_DIR="${HOME}/venv/${PROJECT_NAME}"
PYTHON="${VENV_DIR}/bin/python"

# ── Arguments (IP, PORT) ──────────────────────────────────────────
export SERVER_IP="${1:-}"
export SERVER_PORT="${2:-}"

# ── Activate venv ────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: venv not found. Run ./install.sh first." >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

# ── Load .env (if exists) ────────────────────────────────────────
if [ -f "${SCRIPT_DIR}/.env" ]; then
    export "$(grep -v '^\s*#' "${SCRIPT_DIR}/.env" | xargs)"
fi

# ── GPU defaults (H100 / DGX Spark) ──────────────────────────────
export HF_HUB_DISABLE_TELEMETRY=1

# ── Detect source mode ───────────────────────────────────────────
# When sourced: BASH_LINENO[0] is set, and we return (don't exec Python)
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    export RUN_ARGS="sourced"
    return 0 2>/dev/null || exit 0
fi

# ── Run ──────────────────────────────────────────────────────────
exec "$PYTHON" "${SCRIPT_DIR}/app.py"
