#!/usr/bin/env bash
# install.sh — idempotent, upgrade-compatible
# Usage: ./install.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
VENV_DIR="$HOME/venv/$PROJECT_NAME"
UV_BIN="${VENV_DIR}/bin/uv"

# ── Detect uv ──────────────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    UV="$(command -v uv)"
else
    echo "→ Installing uv…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    UV="$(command -v uv 2>/dev/null || echo uv)"
fi

# ── Create / refresh venv (idempotent) ─────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "→ Venv exists: $VENV_DIR"
else
    echo "→ Creating venv: $VENV_DIR"
    "$UV" venv "$VENV_DIR"
fi

# ── Activate ───────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"

# ── Sync dependencies from pyproject.toml (upgrade-compatible) ────────
echo "→ Syncing dependencies…"
"$UV" sync --active

echo ""
echo "✅ Installed. Activate with:"
echo "   source \"$VENV_DIR/bin/activate\""
