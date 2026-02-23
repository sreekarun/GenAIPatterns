#!/usr/bin/env bash
set -euo pipefail

# Create and activate a virtual environment, then install dependencies.
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Virtual environment ready. Activate with: source .venv/bin/activate"
