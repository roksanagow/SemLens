#!/usr/bin/env bash
# SemLens setup script
# Creates a virtual environment and installs all dependencies.
#
# Usage:
#   bash setup.sh          # create venv + install
#   bash setup.sh --run    # create venv + install + launch app

set -e

VENV_DIR=".venv"

echo "=== SemLens Setup ==="

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
echo "Activated: $(which python)"

# Upgrade pip
pip install --upgrade pip --quiet

# Install package in editable mode with all extras
echo "Installing semlens and dependencies ..."
pip install -e ".[all]"

# Suppress Streamlit email prompt
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    mkdir -p "$HOME/.streamlit"
    cp .streamlit/credentials.toml "$HOME/.streamlit/credentials.toml"
    echo "Configured Streamlit (suppressed email prompt)"
fi

echo ""
echo "=== Setup complete ==="
echo "To activate the environment:  source $VENV_DIR/bin/activate"
echo "To launch the app:           streamlit run app/app.py"

# Optionally run the app
if [ "$1" = "--run" ]; then
    echo ""
    echo "Launching SemLens ..."
    streamlit run app/app.py
fi
