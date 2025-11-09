#!/bin/bash
# Usage: ./run_ingest.sh data/pdfs/sample.pdf docs
PDF=$1
COLL=${2:-docs}
python -m src.ingest --pdf_path "$PDF" --collection_name "$COLL"
