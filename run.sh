#!/usr/bin/env bash
# Create the venv (if missing), install dependencies and run the script.
set -e
cd "$(dirname "$0")"

# Pick the available Python command
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

# Create the virtual environment once
if [ ! -d ".venv" ]; then
    echo ">> Creating virtual environment .venv"
    $PY -m venv .venv
fi

# Activate it (Linux/macOS or Git Bash on Windows)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    source .venv/Scripts/activate
fi

# Install dependencies
echo ">> Installing dependencies"
python -m pip install --upgrade pip -q
python -m pip install scikit-learn matplotlib -q

# Run the script
echo ">> Running bcw_roc_pr_threshold.py"
python bcw_roc_pr_threshold.py