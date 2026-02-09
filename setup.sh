#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXTERNAL_DIR="$SCRIPT_DIR/external"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=== Setting up diffmjx-al umbrella repo ==="

# Create virtual environment with uv
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
else
    echo "Creating virtual environment..."
    uv venv "$VENV_DIR"
fi

# Create external directory for cloned repos
mkdir -p "$EXTERNAL_DIR"

# Clone repositories if they don't already exist
if [ -d "$EXTERNAL_DIR/mujoco" ]; then
    echo "mujoco already cloned"
else
    echo "Cloning mujoco..."
    git clone git@github.com:a-paulus/mujoco.git "$EXTERNAL_DIR/mujoco"
fi

if [ -d "$EXTERNAL_DIR/mjx_diffrax" ]; then
    echo "mjx_diffrax already cloned"
else
    echo "Cloning mjx_diffrax..."
    git clone git@github.com:a-paulus/mjx_diffrax.git "$EXTERNAL_DIR/mjx_diffrax"
fi

# Install repositories in editable mode
echo "Installing mujoco in editable mode..."
uv pip install -e "$EXTERNAL_DIR/mujoco"

echo "Installing mjx_diffrax in editable mode..."
uv pip install -e "$EXTERNAL_DIR/mjx_diffrax"

echo ""
echo "=== Setup complete ==="
echo "Activate the virtual environment with:"
echo "  source .venv/bin/activate"
