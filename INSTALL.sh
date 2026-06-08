#!/usr/bin/env bash
# One-shot setup for luknn-replication (CPU-only)
set -e

python3 -m venv .venv
source .venv/bin/activate

# CPU-only PyTorch (no CUDA wheel needed)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[dev]"

echo ""
echo "Done. Activate with:  source .venv/bin/activate"
echo "Run tests with:       pytest"
