#!/bin/bash
set -euo pipefail

THIS_DIR=$(readlink -f "$(dirname "$0")")

PAPER="${THIS_DIR}/paper.md"
REFS="${THIS_DIR}/refs.bib"
PRESENTATION_DIR="${THIS_DIR}/presentation"

pandoc "${PAPER}" \
  --from markdown \
  --to pdf \
  --citeproc \
  --bibliography "${REFS}" \
  --output "${THIS_DIR}/paper.pdf"

(
  cd "${PRESENTATION_DIR}"
  xelatex -interaction=nonstopmode -halt-on-error presentation.tex
  xelatex -interaction=nonstopmode -halt-on-error presentation.tex
)
