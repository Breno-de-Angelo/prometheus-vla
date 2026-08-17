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
echo "[5/7] Installing LeRobot with extras (including PI05 dependencies)..."
pip install -e ./lerobot[unitree_g1_dex3,televuer,intelrealsense,pi]

# Stage 6: televuer — NOSSO FORK, o de lerobot-ext/teleop/televuer
#
# Cuidado: o extra `televuer` do lerobot acima NÃO instala o pacote televuer.
# Ele só puxa vuer/aiortc/opencv. Quem instalar o televuer de upstream
# (silencht) fica com um pacote de MESMA VERSÃO (4.0.0) e conteúdo diferente,
# e o teleop quebra em runtime com erros do tipo:
#   TypeError: TeleVuerWrapper.__init__() got an unexpected keyword argument 'wrist_cam'
# porque o painel de câmera de pulso é adição nossa. -e é obrigatório: sem
# ele, um `git pull` no repo não atualiza o pacote instalado.
echo ""
echo "[6/7] Installing our televuer fork (editable, from lerobot-ext/teleop/televuer)..."
pip uninstall -y televuer 2>/dev/null || true
pip install -e ./lerobot-ext/teleop/televuer

# Stage 7: Verify installation
echo ""
echo "[7/7] Verifying installation..."
python -c "import torch; import diffusers; import lerobot; print('All imports successful!')"
# Não basta importar: tem que ser O NOSSO televuer. Checa pelo CAMINHO (a
# versão é igual à de upstream e não distingue) e por um parâmetro que só
# existe no fork.
python - <<'PYCHECK'
import inspect, sys
from pathlib import Path
import televuer
from televuer import TeleVuerWrapper

caminho = Path(televuer.__file__).resolve()
params = inspect.signature(TeleVuerWrapper.__init__).parameters

if "wrist_cam" not in params:
    print(f"ERRO: televuer instalado NAO e o nosso fork -> {caminho}")
    print("Conserte com:  pip uninstall -y televuer && pip install -e ./lerobot-ext/teleop/televuer")
    sys.exit(1)
print(f"televuer OK (fork, editable) -> {caminho}")
PYCHECK

echo ""
echo "=== Installation complete ==="
