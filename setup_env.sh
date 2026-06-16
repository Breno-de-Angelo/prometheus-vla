#!/bin/bash
# =============================================================================
# setup_env.sh — Cria e configura o ambiente conda para o Prometheus VLA
# Uso:  bash setup_env.sh [nome_do_env]
# Ex.:  bash setup_env.sh          → cria o env "g1"
#       bash setup_env.sh g1_env   → cria o env "g1_env"
# =============================================================================
set -e

ENV_NAME="${1:-g1}"

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step()  { echo -e "\n${GREEN}[$(date +%H:%M:%S)] $*${NC}"; }
warn()  { echo -e "${YELLOW}⚠️  $*${NC}"; }
abort() { echo -e "${RED}❌ ERRO: $*${NC}"; exit 1; }

# =============================================================================
# 0. Verificações iniciais
# =============================================================================
step "0/9 — Verificações iniciais"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Diretório: $SCRIPT_DIR"

[[ -f "lerobot-ext/init_lerobot_record.py" ]] \
    || abort "Execute este script a partir da raiz do repositório prometheus-vla."

[[ -d "lerobot" ]] \
    || abort "Submódulo 'lerobot' não encontrado. Rode: git submodule update --init --recursive"

[[ -d "unitree_sdk2_python" ]] \
    || abort "Submódulo 'unitree_sdk2_python' não encontrado. Rode: git submodule update --init --recursive"

[[ -d "lerobot-ext/teleop/robot_control/dex-retargeting" ]] \
    || abort "Diretório dex-retargeting não encontrado. Rode: git submodule update --init --recursive"

if ! command -v conda &>/dev/null; then
    abort "conda não encontrado. Instale o Miniconda e reinicie o terminal."
fi

echo "conda: $(conda --version)"
echo "Ambiente que será criado: $ENV_NAME"

# =============================================================================
# 1. Criar o ambiente conda
# =============================================================================
step "1/9 — Criar ambiente conda '$ENV_NAME' (Python 3.10)"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    warn "Ambiente '$ENV_NAME' já existe. Pulando criação."
else
    conda create -n "$ENV_NAME" python=3.10 pip -c conda-forge -y
    echo "Ambiente criado."
fi

# Caminhos absolutos — conda activate não atualiza PATH dentro de scripts bash
CONDA_BASE=$(conda info --base)
PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"
PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"
UV="$CONDA_BASE/envs/$ENV_NAME/bin/uv"
echo "python: $PYTHON"

# =============================================================================
# 2. Instalar pacotes científicos via conda
# =============================================================================
step "2/9 — Instalar pinocchio, casadi, proxsuite, numpy via conda-forge"

# numpy<2 é obrigatório: pinocchio 3.1 segfaulta com NumPy 2.x
conda install -n "$ENV_NAME" -c conda-forge \
    "numpy=1.26.4" \
    "pinocchio=3.1.0" \
    "casadi=3.6.7" \
    "proxsuite=0.7.2" \
    -y

# =============================================================================
# 3. Instalar uv (substitui pip — downloads paralelos, 10-100x mais rápido)
# =============================================================================
step "3/9 — Instalar uv no env"

# Garante pip (caso o env tenha sido criado sem ele) e instala uv
"$PYTHON" -m ensurepip --upgrade 2>/dev/null || true
"$PYTHON" -m pip install uv --quiet
echo "uv: $("$UV" --version)"

# uv precisa saber qual python usar dentro do env
UV_CMD="$UV pip install --python $PYTHON"

# =============================================================================
# 4. Instalar LeRobot (submódulo local, editable)
# =============================================================================
# PyTorch é instalado DEPOIS do lerobot para que a versão pinada vença
# qualquer dependência que o lerobot puxe automaticamente.
step "4/9 — Instalar LeRobot (editable, do submódulo local)"

$UV_CMD -e "./lerobot[unitree_g1_dex3,televuer,intelrealsense,pi]"

# =============================================================================
# 5. Fixar PyTorch 2.3.0 (sobrescreve qualquer versão puxada pelo lerobot)
# =============================================================================
step "5/9 — Fixar PyTorch 2.3.0 + torchvision 0.18.0"

if command -v nvidia-smi &>/dev/null; then
    echo "GPU NVIDIA detectada — instalando com suporte CUDA 12.1"
    $UV_CMD \
        torch==2.3.0 \
        torchvision==0.18.0 \
        torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu121
else
    warn "GPU NVIDIA não detectada — instalando versão CPU (treino não funcionará)"
    $UV_CMD torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
fi

# =============================================================================
# 6. Instalar dex-retargeting + Unitree SDK (locais)
# =============================================================================
step "6/9 — Instalar dex-retargeting e unitree_sdk2_python (locais)"

# dex-retargeting: editable
$UV_CMD -e "lerobot-ext/teleop/robot_control/dex-retargeting"

# televuer e teleimager: pacotes locais de teleop VR
# vuer[all] é obrigatório — sem [all] falta aiohttp e Vuer não é exportado
# params-proto 2.13.2 é obrigatório — versões mais novas removeram Flag e quebram vuer
$UV_CMD "params-proto==2.13.2" "vuer[all]==0.0.60" lerobot-ext/teleop/televuer/ lerobot-ext/teleop/teleimager/

# unitree SDK: NÃO usar o PyPI — a versão do PyPI não inclui crc_amd64.so
$UV_CMD ./unitree_sdk2_python

# =============================================================================
# 7. Instalar dependências restantes (em paralelo via uv)
# =============================================================================
step "7/9 — Instalar dependências restantes"

$UV_CMD \
    "diffusers==0.30.0" \
    "datasets==4.1.0" \
    "draccus==0.10.0" \
    "mujoco==3.6.0" \
    "pyzmq" \
    "opencv-python" \
    "pyrealsense2" \
    "av" \
    "pyyaml" \
    "pandas" \
    "wandb" \
    "huggingface-hub" \
    "accelerate" \
    "einops" \
    "safetensors" \
    "transformers" \
    "rerun-sdk" \
    "pynput" \
    "evdev" \
    "pyserial" \
    "omniview" \
    "flask" \
    "scipy" \
    "meshcat" \
    "pygame" \
    "loguru"

# opcionais: controle por voz
echo ""
echo "Instalando opcionais (controle por voz) — pode ignorar erros aqui:"
$UV_CMD pyaudio speechrecognition 2>/dev/null \
    || warn "pyaudio/speechrecognition não instalados (opcional — sem controle por voz)"

# =============================================================================
# 8. Verificar instalação
# =============================================================================
step "8/9 — Verificar instalação"

"$PYTHON" - <<'EOF'
import sys

ok = True
checks = [
    ("numpy",    lambda: __import__("numpy").__version__),
    ("torch",    lambda: __import__("torch").__version__),
    ("mujoco",   lambda: __import__("mujoco").__version__),
    ("cv2",      lambda: __import__("cv2").__version__),
    ("zmq",      lambda: __import__("zmq").__version__),
    ("lerobot",  lambda: __import__("lerobot").__version__),
    ("draccus",  lambda: __import__("draccus").__version__),
    ("robot.unitree_g1.unitree_g1_dex3", lambda: "ok"),
]

for name, fn in checks:
    try:
        ver = fn()
        print(f"  ✅ {name}: {ver}")
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        ok = False

import numpy as np
if int(np.__version__.split(".")[0]) >= 2:
    print(f"  ❌ numpy {np.__version__} — PRECISA ser < 2.0 (pinocchio segfaulta com NumPy 2.x)")
    ok = False
else:
    print(f"  ✅ numpy versão OK ({np.__version__} < 2.0)")

sys.exit(0 if ok else 1)
EOF

# =============================================================================
# 9. Instruções finais
# =============================================================================
step "9/9 — Pronto!"

echo ""
echo "════════════════════════════════════════════════════════════"
echo " Ambiente '$ENV_NAME' configurado com sucesso."
echo ""
echo " Para ativar:"
echo "   conda activate $ENV_NAME"
echo ""
echo " Para rodar em modo simulação:"
echo "   cd $SCRIPT_DIR"
echo "   conda activate $ENV_NAME"
echo "   python lerobot-ext/init_lerobot_record.py \\"
echo "       --config_path lerobot-ext/config/record/record_televuer.yaml \\"
echo "       --sim \\"
echo "       --quest-ip \$(hostname -I | awk '{print \$1}') \\"
echo "       --quest-adb-ip 192.168.68.51"
echo ""
echo " Guia completo: lerobot-ext/README_SETUP_AMBIENTE.md"
echo "════════════════════════════════════════════════════════════"
