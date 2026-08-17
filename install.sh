#!/usr/bin/env bash
#
# Instala o prometheus-vla com LeRobot 0.6.1 / Python 3.12.
#
# É o §0 do docs/INSTALL.md executável. A ordem dos passos NÃO é arbitrária — cada
# desvio dela já custou um ambiente quebrado, e o porquê de cada um está comentado
# aqui e detalhado no guia.
#
#   ./install.sh                          # cria/usa o env `prometheus-vla`
#   ./install.sh --env prometheus-teste   # outro nome (para testar sem risco)
#   ./install.sh --sem-mujoco             # pula a simulação
#   ./install.sh --so-verificar           # não instala nada, só confere o env ativo
#
# Para reconstruir o ambiente ANTIGO (LeRobot 0.4.4 / Python 3.10), que segue sendo o
# rollback, use o install-g1-0.4.4.sh preservado ao lado deste.

set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NOME="prometheus-vla"
COM_MUJOCO=1
SO_VERIFICAR=0
BRANCH_LEROBOT="prometheus-vla/v0.6.1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)          ENV_NOME="$2"; shift 2 ;;
        --sem-mujoco)   COM_MUJOCO=0; shift ;;
        --so-verificar) SO_VERIFICAR=1; shift ;;
        -h|--help)      sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *)              echo "opção desconhecida: $1 (use --help)" >&2; exit 2 ;;
    esac
done

passo() { echo; echo "── $* ──────────────────────────────────────────"; }
erro()  { echo "ERRO: $*" >&2; exit 1; }

# ── Pré-requisitos ───────────────────────────────────────────────────────────
command -v conda >/dev/null 2>&1 || erro "conda não encontrado no PATH."
# `conda activate` não funciona dentro de script sem carregar o hook antes.
source "$(conda info --base)/etc/profile.d/conda.sh"

cd "$RAIZ"

if [[ $SO_VERIFICAR -eq 0 ]]; then
    # ── 1. Submódulos ────────────────────────────────────────────────────────
    passo "1/9  submódulos"
    git submodule update --init --recursive

    # O submódulo vem no SHA registrado, em detached HEAD. A branch é nossa e precisa
    # existir no fork para funcionar numa máquina nova.
    if git -C lerobot rev-parse --verify "$BRANCH_LEROBOT" >/dev/null 2>&1; then
        git -C lerobot checkout "$BRANCH_LEROBOT"
    elif git -C lerobot rev-parse --verify "origin/$BRANCH_LEROBOT" >/dev/null 2>&1; then
        git -C lerobot checkout -b "$BRANCH_LEROBOT" "origin/$BRANCH_LEROBOT"
    else
        erro "branch '$BRANCH_LEROBOT' não existe no submódulo lerobot, nem local nem no origin.
       Se esta é uma máquina nova, ela ainda não foi enviada:
         cd lerobot && git push origin $BRANCH_LEROBOT"
    fi
    echo "lerobot em $(git -C lerobot rev-parse --short HEAD) ($BRANCH_LEROBOT)"

    # ── 2. Interpretador ─────────────────────────────────────────────────────
    passo "2/9  env conda '$ENV_NOME' (Python 3.12)"
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NOME"; then
        echo "já existe — reaproveitando."
    else
        # -c conda-forge --override-channels evita os Terms of Service dos canais
        # repo.anaconda.com, que têm restrição de uso comercial.
        conda create -n "$ENV_NOME" python=3.12 -y -c conda-forge --override-channels
    fi
    conda activate "$ENV_NOME"
    # O conda entra só pelo interpretador; os pacotes vêm todos de pip, para não
    # misturar dois resolvers no mesmo ambiente.
    echo "python: $(python -V) em $(which python)"

    # ── 3. torch ANTES de tudo ───────────────────────────────────────────────
    passo "3/9  torch + torchvision (índice cu128)"
    # A RTX 5070 é Blackwell (sm_120) e precisa de CUDA 12.8+. Se o torch vier depois,
    # como dependência transitiva, o resolver escolhe um wheel CPU ou cu126 e a GPU
    # some sem aviso nenhum — o import continua funcionando igual.
    pip install --index-url https://download.pytorch.org/whl/cu128 \
        "torch>=2.7,<2.12" "torchvision>=0.22.0,<0.27.0"

    # ── 4. SDK da Unitree ────────────────────────────────────────────────────
    passo "4/9  unitree_sdk2py"
    # Fica comentado no pyproject do lerobot upstream ("requires specific installation
    # instructions"), então não vem por extra nenhum.
    if python -c "import unitree_sdk2py" >/dev/null 2>&1; then
        echo "já instalado."
    else
        pip install git+https://github.com/unitreerobotics/unitree_sdk2_python.git
    fi

    # ── 5. lerobot ───────────────────────────────────────────────────────────
    passo "5/9  lerobot (fork) + extras"
    pip install -e "./lerobot[unitree_g1_dex3,televuer,intelrealsense,pi,dataset]"

    # ── 6. Forks vendorizados ────────────────────────────────────────────────
    passo "6/9  televuer e dex_retargeting (forks do repo, editable)"
    # Os dois existem no PyPI com estes nomes e NENHUM dos dois serve:
    #  - o televuer de upstream tem a MESMA versão (4.0.0) com conteúdo diferente, e o
    #    teleop quebra com "unexpected keyword argument 'wrist_cam'";
    #  - o dex_retargeting 0.5.0 mudou o schema do RetargetingConfig, e os nossos .yml
    #    de mão são do formato antigo (INSTALL §2.7).
    # O -e é obrigatório: sem ele, um `git pull` não atualiza o pacote instalado.
    pip uninstall -y televuer dex_retargeting >/dev/null 2>&1 || true
    pip install -e ./lerobot-ext/teleop/televuer
    pip install -e ./lerobot-ext/teleop/robot_control/dex-retargeting

    # ── 7. Resto ─────────────────────────────────────────────────────────────
    passo "7/9  flask, pytest, transformers no range"
    pip install flask pytest
    # A 0.6.1 exige >=5.4,<5.6; o env antigo tinha 5.6.1, fora do range.
    pip install "transformers>=5.4.0,<5.6.0"

    # ── 8. MuJoCo (opcional) ─────────────────────────────────────────────────
    if [[ $COM_MUJOCO -eq 1 ]]; then
        passo "8/9  mujoco (simulação)"
        # NÃO instale unitree-g1-mujoco/requirements.txt: ele pede opencv-python
        # (não-headless), que instala o MESMO módulo cv2 do opencv-python-headless de
        # que o lerobot depende — dois binários brigando pelo mesmo import.
        pip install mujoco loguru msgpack msgpack-numpy matplotlib
    else
        passo "8/9  mujoco — pulado (--sem-mujoco)"
    fi
else
    passo "modo --so-verificar: nada será instalado"
    [[ -n "${CONDA_DEFAULT_ENV:-}" ]] || erro "nenhum env conda ativo."
    echo "env ativo: $CONDA_DEFAULT_ENV"
fi

# ── 9. Verificação ───────────────────────────────────────────────────────────
passo "9/9  verificação"

cd "$RAIZ"
python - <<'PYCHECK'
import importlib, inspect, sys

falhas = []

import numpy, torch, transformers, lerobot
print(f"  python       {sys.version.split()[0]}")
print(f"  lerobot      {lerobot.__version__}   (esperado 0.6.1)")
print(f"  numpy        {numpy.__version__}   (esperado 2.x)")
print(f"  transformers {transformers.__version__}   (esperado >=5.4,<5.6)")
print(f"  torch        {torch.__version__}  cuda={torch.version.cuda} disponivel={torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("  AVISO: torch sem CUDA. Numa máquina com GPU isso é wheel errado (§4.1).")

# Os dois forks precisam vir do REPO, não do PyPI. A versão não distingue — só o
# caminho, e no televuer um parâmetro que só o fork tem.
import televuer, dex_retargeting

if "wrist_cam" not in inspect.signature(televuer.TeleVuerWrapper.__init__).parameters:
    falhas.append(f"televuer NÃO é o nosso fork -> {televuer.__file__}")
else:
    print("  televuer     fork do repo, editable")

if "lerobot-ext/teleop/robot_control/dex-retargeting" not in dex_retargeting.__file__:
    falhas.append(f"dex_retargeting veio do PyPI -> {dex_retargeting.__file__}")
else:
    print("  dex_retarget fork 0.4.7 do repo, editable")

sys.path.insert(0, "lerobot-ext")
modulos = [
    "lerobot.robots.unitree_g1.unitree_g1_dex3",
    "lerobot.teleoperators.televuer",
    "lerobot.cameras.zmq.camera_zmq",
    "lerobot.datasets.lerobot_dataset",
    "robot.unitree_g1.unitree_g1_dex3",
    "robot.unitree_g1.unitree_g1_loco",
    "teleop.xr_g1_arm",
    "teleop.robot_control.robot_arm_ik",
    "teleop.robot_control.hand_retargeting",
    "flask",
]
for m in modulos:
    try:
        importlib.import_module(m)
    except Exception as e:
        falhas.append(f"import {m}: {type(e).__name__}: {e}")
print(f"  imports      {len(modulos) - len(falhas)}/{len(modulos)}")

if falhas:
    print("\nFALHAS:")
    for f in falhas:
        print(f"  - {f}")
    sys.exit(1)
PYCHECK

# Os testes rodam de DENTRO de lerobot-ext: assets e caches são resolvidos por caminho
# relativo ao cwd (INSTALL §5.1).
passo "testes"
cd "$RAIZ/lerobot-ext" && python -m pytest tests/ -q

if [[ $SO_VERIFICAR -eq 1 ]]; then
    # No modo verificação nada foi instalado, e o env é o que já estava ativo — não o
    # $ENV_NOME, que aqui seria só o valor padrão da variável.
    ENV_NOME="${CONDA_DEFAULT_ENV}"
    RESUMO="Verificação concluída no env '$ENV_NOME' — nada foi instalado."
else
    RESUMO="Instalação concluída no env '$ENV_NOME'."
fi

echo
echo "════════════════════════════════════════════════════════════"
echo " $RESUMO"
echo
echo " Para usar:  conda activate $ENV_NOME && cd $RAIZ/lerobot-ext"
echo " Guia:       docs/INSTALL.md — §5 tem a verificação completa (IK e"
echo "             retargeting); §8, o que ainda falta validar com o robô"
echo "             e com o headset."
echo "════════════════════════════════════════════════════════════"
