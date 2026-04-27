#!/bin/bash
# Launch the Adaptive Knowledge Selector web UI
# Usage: ./run.sh

VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"
exec "$VENV_PYTHON" -m streamlit run app.py "$@"
