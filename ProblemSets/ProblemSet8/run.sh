#!/usr/bin/env bash
# run.sh -- run analysis and compile tex
# Usage:
#   ./run.sh
#
# Requires: Python, pdflatex

set -euo pipefail

# Move to script directory
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "Working directory: $HERE"

##############################
# 1) Run Python analysis
##############################
echo ">>> Running ps8_fixed.py ..."
python -u ps8_fixed.py --full

##############################
# 2) Compile LaTeX
##############################
TEXFILE="ps8_writeup.tex"
OUTPDF="${TEXFILE%.tex}.pdf"

if [ ! -f "$TEXFILE" ]; then
  echo "ERROR: $TEXFILE not found!"
  exit 2
fi

echo ">>> Compiling LaTeX -> PDF ..."
pdflatex -interaction=nonstopmode -halt-on-error "$TEXFILE"
pdflatex -interaction=nonstopmode -halt-on-error "$TEXFILE"

echo ">>> Done."
echo "PDF generated: $OUTPDF"
