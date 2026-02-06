#!/bin/bash
# Staged installation script to avoid pip dependency resolution issues
# Usage: ./install.sh

set -e  # Exit on error

echo "=== Installing prometheus-vla dependencies ==="

# Check if conda environment is active
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "Error: No conda environment active. Please run:"
    echo "  conda activate g1"
    exit 1
fi

echo "Using conda environment: $CONDA_DEFAULT_ENV"

# Stage 1: Core packages
echo ""
echo "[1/6] Installing core packages (numpy, torch)..."
pip install "numpy<2.0.0" "torch>=2.4.0"

# Stage 2: Diffusers (pinned to avoid resolution issues)
echo ""
echo "[2/6] Installing diffusers..."
pip install diffusers==0.30.0

# Stage 3: Other dependencies
echo ""
echo "[3/6] Installing datasets, cmake, av, flask, vuer..."
pip install datasets==4.1.0 "cmake>=3.29.0" "av>=15.0.0" flask vuer

# Stage 4: Unitree SDK
echo ""
echo "[4/6] Installing Unitree SDK..."
pip install git+https://github.com/unitreerobotics/unitree_sdk2_python.git

# Stage 5: LeRobot with extras
echo ""
echo "[5/6] Installing LeRobot with extras..."
pip install -e ./lerobot[unitree_g1_dex3,televuer,intelrealsense]

# Stage 6: Verify installation
echo ""
echo "[6/6] Verifying installation..."
python -c "import torch; import diffusers; import lerobot; print('All imports successful!')"

echo ""
echo "=== Installation complete ==="
