#!/bin/bash
# run_ps5.sh - run tests, analysis, and compile LaTeX (for PS5 submission)

set -euo pipefail
cd "$(dirname "$0")"

echo "1) Running unit tests..."
.venv/bin/python -m pytest -q

echo "2) Running analysis script..."
.venv/bin/python PS5_mallick.py

echo "3) Compiling LaTeX (twice for references)..."
if command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode ProblemSet5_Mallick.tex > /dev/null
  pdflatex -interaction=nonstopmode ProblemSet5_Mallick.tex > /dev/null
  echo "ProblemSet5_Mallick.pdf created"
else
  echo "pdflatex not found — skipping LaTeX compile"
fi

