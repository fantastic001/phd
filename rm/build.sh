#!/bin/bash 

THIS_DIR=$(readlink -f "$(dirname "$0")")

PAPER="${THIS_DIR}/overview.md"
REFS="${THIS_DIR}/refs.bib"
CSL="ieee.csl"



pandoc $PAPER \
  --from markdown \
  --to pdf \
  --citeproc \
  --bibliography $REFS \
  --output "${THIS_DIR}/overview.pdf" 