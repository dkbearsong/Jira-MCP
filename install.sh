#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
REQ_FILE="$PROJECT_ROOT/requirements.txt"

echo "==> Jira Sentiment and Data: Installer"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Error: Python 3 is required but not found in PATH." >&2
  exit 1
fi

echo "==> Using Python: $($PY -V)"

if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtual environment at $VENV_DIR"
  $PY -m venv "$VENV_DIR"
else
  echo "==> Virtual environment already exists at $VENV_DIR"
fi

echo "==> Activating virtual environment"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

if [ -f "$REQ_FILE" ]; then
  echo "==> Installing Python dependencies from $REQ_FILE"
  pip install -r "$REQ_FILE"
else
  echo "Warning: requirements.txt not found at $REQ_FILE"
  if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
    echo "==> Installing project (pyproject.toml) in editable mode"
    pip install -e "$PROJECT_ROOT/jira_mcp"
  else
    echo "Error: No dependency specification found. Please add requirements.txt or pyproject.toml." >&2
    exit 1
  fi
fi

echo "==> Creating .env if missing and documenting required variables"
ENV_FILE="$PROJECT_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<'EOF'
# Jira MCP configuration
# Required:
# JIRA_EMAIL=you@example.com
# API_TOKEN=your_api_token
# JIRA_SITE=https://yourorg.atlassian.net
EOF
  echo "    Wrote template .env to $ENV_FILE"
else
  echo "    .env already exists at $ENV_FILE (skipping)"
fi

echo "==> Done. To start using the environment:"
echo "    source .venv/bin/activate"
echo "    # edit .env to include JIRA_EMAIL, API_TOKEN, JIRA_SITE"
echo ""
echo "If you plan to run the MCP server:"
echo "    python -m jira_mcp.main"


