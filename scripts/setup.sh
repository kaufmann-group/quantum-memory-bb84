#!/bin/bash
set -e

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"

rm -rf "$REPO/.venv"

python3.11 -m venv "$REPO/.venv"
source "$REPO/.venv/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$REPO/scripts/requirements.txt"

python -m ipykernel install \
    --user \
    --name=repo-env \
    --display-name "Quantum Simulation Environment (Python 3.11)"

echo ""
echo "Setup complete."
echo "Activate the environment from the project root with:"
echo "source .venv/bin/activate"
