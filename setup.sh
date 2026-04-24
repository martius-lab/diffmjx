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

# Install JAX with CUDA 12 support
echo "Installing JAX with CUDA 12 support..."
uv pip install "jax[cuda12]"

# Clone repositories if they don't already exist
if [ -d "$EXTERNAL_DIR/softjax" ]; then
    echo "softjax already cloned"
else
    echo "Cloning softjax..."
    git clone git@github.com:a-paulus/softjax.git "$EXTERNAL_DIR/softjax"
fi

if [ -d "$EXTERNAL_DIR/mujoco" ]; then
    echo "mujoco already cloned"
else
    echo "Cloning mujoco..."
    git clone -b diffmjx git@github.com:martius-lab/mujoco.git "$EXTERNAL_DIR/mujoco"
fi

if [ -d "$EXTERNAL_DIR/mjx_diffrax" ]; then
    echo "mjx_diffrax already cloned"
else
    echo "Cloning mjx_diffrax..."
    git clone git@github.com:a-paulus/mjx_diffrax.git "$EXTERNAL_DIR/mjx_diffrax"
fi

echo ""
echo "=== Setup complete ==="
echo "Run experiments with:"
echo "  uv run experiments/<experiment>/run.py"
