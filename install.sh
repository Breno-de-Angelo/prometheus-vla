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
    passo "1/10  submódulos"
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
    passo "2/10  env conda '$ENV_NOME' (Python 3.12)"
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NOME"; then
        echo "já existe — reaproveitando."
    else
        # -c conda-forge --override-channels evita os Terms of Service dos canais
        # repo.anaconda.com, que têm restrição de uso comercial.
        conda create -n "$ENV_NOME" python=3.12 -y -c conda-forge --override-channels
    fi
    conda activate "$ENV_NOME"
    # O conda entra pelo interpretador e mais nada, para não misturar dois resolvers no
    # mesmo ambiente. A única exceção é o pinocchio/casadi do passo 7, que não tem
    # equivalente em wheel.
    echo "python: $(python -V) em $(which python)"

    # ── 3. torch ANTES de tudo ───────────────────────────────────────────────
    passo "3/10  torch + torchvision (índice cu128)"
    # A RTX 5070 é Blackwell (sm_120) e precisa de CUDA 12.8+. Se o torch vier depois,
    # como dependência transitiva, o resolver escolhe um wheel CPU ou cu126 e a GPU
    # some sem aviso nenhum — o import continua funcionando igual.
    pip install --index-url https://download.pytorch.org/whl/cu128 \
        "torch>=2.7,<2.12" "torchvision>=0.22.0,<0.27.0"

    # ── 4. SDK da Unitree ────────────────────────────────────────────────────
    passo "4/10  cyclonedds + unitree_sdk2py"
    # Fica comentado no pyproject do lerobot upstream ("requires specific installation
    # instructions"), então não vem por extra nenhum.
    #
    # O install_requires do SDK tem DUAS armadilhas, e as duas são desarmadas pelo
    # --no-deps abaixo:
    #
    #  - `cyclonedds==0.10.2`: não existe wheel dessa versão para o py3.12, então o pip
    #    cai no sdist, tenta compilar e para em "Could not locate cyclonedds. Try to set
    #    CYCLONEDDS_HOME". O 11.0.1 tem wheel e é o que o env validado usa — por isso ele
    #    entra ANTES, e o --no-deps é o que impede o pip de reverter para o 0.10.2.
    #  - `opencv-python` (não-headless): instala o MESMO módulo cv2 do
    #    opencv-python-headless de que o lerobot depende, e os dois binários brigam pelo
    #    mesmo import (§2.1).
    #
    # `numpy`, a terceira dependência, já vem pelo torch/lerobot.
    #
    # E vem do submódulo (editable), não do git+https: o upstream avança sozinho e o
    # SHA que testamos é o registrado no submódulo.
    pip install "cyclonedds==11.0.1"
    pip install --no-deps -e ./unitree_sdk2_python

    # ── 5. lerobot ───────────────────────────────────────────────────────────
    passo "5/10  lerobot (fork) + extras"
    # `training` traz accelerate e wandb — sem ele o `run_train.py` das políticas
    # morre no `from accelerate import Accelerator`. Não é opcional para quem
    # treina; o `dataset` sozinho só dá conta de gravar e ler dataset.
    pip install -e "./lerobot[unitree_g1_dex3,televuer,intelrealsense,pi,dataset,training]"

    # ── 6. Forks vendorizados ────────────────────────────────────────────────
    passo "6/10  televuer e dex_retargeting (forks do repo, editable)"
    # Os dois existem no PyPI com estes nomes e NENHUM dos dois serve:
    #  - o televuer de upstream tem a MESMA versão (4.0.0) com conteúdo diferente, e o
    #    teleop quebra com "unexpected keyword argument 'wrist_cam'";
    #  - o dex_retargeting 0.5.0 mudou o schema do RetargetingConfig, e os nossos .yml
    #    de mão são do formato antigo (INSTALL §2.7).
    # O -e é obrigatório: sem ele, um `git pull` não atualiza o pacote instalado.
    pip uninstall -y televuer dex_retargeting >/dev/null 2>&1 || true
    pip install -e ./lerobot-ext/teleop/televuer
    pip install -e ./lerobot-ext/teleop/robot_control/dex-retargeting

    # ── 7. Cinemática: pinocchio + casadi ────────────────────────────────────
    passo "7/10  pinocchio + casadi (conda-forge)"
    # Estes NÃO vêm de pip: o `pin` do PyPI não traz os bindings de casadi, e o IK do
    # G1 (teleop/robot_control/robot_arm_ik.py) monta o problema em `pinocchio.casadi`.
    # É a única exceção à regra "pacotes só de pip" do passo 2 — não tem wheel
    # equivalente. Vem junto o ipopt, que é o solver que o IK usa.
    #
    # A POSIÇÃO DESTE PASSO É O PONTO DELICADO, por dois motivos:
    #
    #  - o `pin` do PyPI (dependência do dex_retargeting, passo 6) e o `pinocchio` do
    #    conda-forge instalam no MESMO diretório, site-packages/pinocchio/. Não há
    #    conflito declarado — os nomes de distribuição são diferentes, então nenhum dos
    #    dois gerenciadores vê o outro — e quem sobrescreve é simplesmente quem roda por
    #    último. Rodando o conda DEPOIS, o dist-info do `pin` continua lá satisfazendo o
    #    requisito do dex_retargeting, e os arquivos são os do conda, com os bindings de
    #    casadi. Invertido, o pip apaga os bindings e o IK morre com
    #    `ModuleNotFoundError: pinocchio.casadi`;
    #  - o conda substitui o numpy do pip pelo build de conda-forge da MESMA versão
    #    (2.2.6). Depois do lerobot, é o lerobot que fixa a versão e o conda só troca o
    #    build; antes dele, o resolver do conda escolhe sozinho.
    conda install -y -c conda-forge --override-channels pinocchio casadi

    # O ipopt do conda-forge é multithread e, sem limitar, cada chamada de IK gasta
    # ~83 ms em vez de ~0,8 ms — o FPS da teleop afunda. Gravado no env para valer em
    # toda sessão, não só nesta.
    conda env config vars set -n "$ENV_NOME" OMP_NUM_THREADS=1 >/dev/null
    export OMP_NUM_THREADS=1

    # ── 8. Resto ─────────────────────────────────────────────────────────────
    passo "8/10  flask, pytest, transformers no range"
    pip install flask pytest
    # A 0.6.1 exige >=5.4,<5.6; o env antigo tinha 5.6.1, fora do range.
    pip install "transformers>=5.4.0,<5.6.0"

    # ── 9. MuJoCo (opcional) ─────────────────────────────────────────────────
    if [[ $COM_MUJOCO -eq 1 ]]; then
        passo "9/10  mujoco (simulação)"
        # NÃO instale unitree-g1-mujoco/requirements.txt: ele pede opencv-python
        # (não-headless), que instala o MESMO módulo cv2 do opencv-python-headless de
        # que o lerobot depende — dois binários brigando pelo mesmo import.
        pip install mujoco loguru msgpack msgpack-numpy matplotlib
    else
        passo "9/10  mujoco — pulado (--sem-mujoco)"
    fi
else
    passo "modo --so-verificar: nada será instalado"
    [[ -n "${CONDA_DEFAULT_ENV:-}" ]] || erro "nenhum env conda ativo."
    echo "env ativo: $CONDA_DEFAULT_ENV"
fi

# ── 10. Verificação ──────────────────────────────────────────────────────────
passo "10/10  verificação"

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

# O IK do G1 monta o problema em `pinocchio.casadi`. O `pin` de pip não traz esse
# submódulo, então importar o pinocchio sozinho não prova nada — tem que ser este.
try:
    import pinocchio, casadi
    import pinocchio.casadi  # noqa: F401
    print(f"  pinocchio    {pinocchio.__version__} (+casadi {casadi.__version__})")
except ImportError as e:
    falhas.append(
        f"pinocchio/casadi: {e}. Se o pinocchio importa mas `pinocchio.casadi` não, o "
        "`pin` do PyPI (dep do dex_retargeting) sobrescreveu o do conda — reinstale "
        "DEPOIS dele: `conda install -c conda-forge pinocchio casadi` (passo 7).")

# O 0.10.2 que o setup.py do unitree_sdk2py pede não tem wheel para py3.12 e nem
# chega a instalar; se importou, veio o wheel certo.
try:
    import cyclonedds  # noqa: F401
    from importlib.metadata import version
    print(f"  cyclonedds   {version('cyclonedds')}   (esperado 11.x)")
except Exception as e:
    falhas.append(f"cyclonedds: {e}. Use `pip install cyclonedds==11.0.1` (passo 4).")

# opencv-python e opencv-python-headless instalam o MESMO módulo cv2; ter os dois é
# não-determinístico (§2.1). O SDK da Unitree puxa o não-headless se instalado sem
# --no-deps, que é justamente como isso costuma entrar.
from importlib.metadata import distributions
_cv = sorted(d.metadata["Name"] for d in distributions()
             if (d.metadata["Name"] or "").startswith("opencv"))
if "opencv-python" in _cv:
    falhas.append(f"opencv duplicado: {_cv}. Só opencv-python-headless — "
                  "`pip uninstall -y opencv-python` (§2.1).")

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

# Não é falha: o passo 6 grava a variável no env, e ela só passa a valer no próximo
# `conda activate`. Nesta sessão o script exporta na mão.
import os
if os.environ.get("OMP_NUM_THREADS") != "1":
    print("  AVISO: OMP_NUM_THREADS != 1. O ipopt do conda-forge é multithread e o IK"
          " cai de ~0,8 ms para ~83 ms. Reative o env (`conda activate`) antes de rodar"
          " a teleop.")

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
