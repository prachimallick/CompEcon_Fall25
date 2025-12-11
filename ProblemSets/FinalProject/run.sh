#!/usr/bin/env bash
# run.sh -- reproducible automation for ECON 833 Final Project
# Usage:
#   ./run.sh   (Git Bash / WSL / Linux / macOS)

set -euo pipefail

echo "=== ECON 833 Final Project Automation ==="

# Ensure script runs from its own directory
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
echo "Working directory: $HERE"

##############################################
# 1) Create & activate local virtual environment
##############################################
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo ">>> Creating virtual environment in $VENV_DIR ..."
    python -m venv "$VENV_DIR"
fi

# Activate venv (Git Bash / WSL / macOS / Linux / Windows Git-Bash)
if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/Scripts/activate"
fi

VENV_PY="$(python -c 'import sys; print(sys.executable)')"
echo "Using Python: $VENV_PY"

##############################################
# 2) Install dependencies
##############################################
if [ -f "requirements.txt" ]; then
    echo ">>> Installing dependencies from requirements.txt ..."
    "$VENV_PY" -m pip install --upgrade pip >/dev/null 2>&1 || true
    "$VENV_PY" -m pip install -r requirements.txt
else
    echo "WARNING: requirements.txt not found — installing minimal dependencies ..."
    "$VENV_PY" -m pip install pandas numpy matplotlib
fi

##############################################
# 3) Run Python script
##############################################
PYFILE="FinalProject.py"

if [ ! -f "$PYFILE" ]; then
    echo "ERROR: Python file $PYFILE not found."
    exit 2
fi

echo ">>> Running $PYFILE ..."
"$VENV_PY" "$PYFILE"
echo "Python analysis complete."

##############################################
# 4) Compile LaTeX → PDF
##############################################
TEXFILE="FinalProject_Mallick.tex"
PDFOUT="FinalProject_Mallick.pdf"

if [ ! -f "$TEXFILE" ]; then
    echo "ERROR: $TEXFILE not found."
    exit 3
fi

if ! command -v pdflatex >/dev/null 2>&1; then
    echo "WARNING: pdflatex not found — skipping PDF compilation."
    exit 0
fi

echo ">>> Compiling LaTeX ..."
pdflatex -interaction=nonstopmode "$TEXFILE" >/dev/null
pdflatex -interaction=nonstopmode "$TEXFILE" >/dev/null

echo ">>> PDF created: $PDFOUT"
echo "=== Done ==="
