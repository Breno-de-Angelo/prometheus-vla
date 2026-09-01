#!/usr/bin/env bash
# Grounding do FastWAM-D: onde o TEXTO olha na imagem.
#
#   uso: bash launch_grounding.sh [GPU] [args extras para o grounding_fastwamd.py]
#
#   bash launch_grounding.sh 2
#   bash launch_grounding.sh 2 --frases="white cup,dripper,robot" --quadros=0,90,180
#   bash launch_grounding.sh --por-camada        # sem GPU fixa: pega a mais livre
#
# Não treina e não infere ação nenhuma: carrega o checkpoint, passa um quadro
# gravado e desenha a cross-attention vídeo→texto por cima da imagem. Ver o
# cabeçalho do `grounding_fastwamd.py` para o que o número significa.
set -euo pipefail

# ── Por que o python vem de $HOME e não de um caminho fixo ───────────────
# Este script roda com mais de um login na mesma máquina (hercules e mrwlker,
# cada um com o seu miniconda). Os launchers de treino fixam
# `/home/hercules/miniconda3/...` porque só o hercules treina; aqui não dá.
# E continua sendo caminho absoluto, não `conda activate`: `screen -dmS` e
# `ssh host cmd` não passam pelo `.bashrc` e não teriam o conda no PATH.
PY="$HOME/miniconda3/envs/prometheus-vla/bin/python"
[ -x "$PY" ] || { echo "ERRO: $PY não existe. Rode o ./install.sh." >&2; exit 1; }

# ── Os dois caches do HuggingFace, separados de propósito ────────────────
# `HF_HUB_CACHE` são os PESOS já baixados (Wan2.2-TI2V-5B, umt5-xxl,
# fastwam_base): 25 GB no disco de dados, world-readable, e ninguém precisa
# baixar de novo.
#
# `HF_HOME` é onde a biblioteca ESCREVE — locks do `datasets`, cache de
# metadados. Apontar os dois para o `/data` quebra qualquer login que não seja
# o dono daquela pasta, e o erro sai lá no fim, depois de carregar 6 B de
# pesos por dois minutos:
#
#   PermissionError: [Errno 13] Permission denied:
#   '/data/.cache/huggingface/datasets/..._0.0.0_....lock'
#
# Por isso: leitura compartilhada, escrita no home de quem rodou.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/.cache/huggingface/hub}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Sem isto o `LeRobotDataset` consulta o Hub para resolver a versão do dataset
# MESMO com `root` local, e um repo_id que não existe lá devolve 401.
export HF_HUB_OFFLINE=1

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    GPU="$1"; shift
else
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d " ")
fi
export CUDA_VISIBLE_DEVICES="$GPU"
# O ipopt do conda-forge é multithread e afunda qualquer coisa que compartilhe
# a máquina; 1 thread é o certo aqui e nos launchers de treino.
export OMP_NUM_THREADS=1

LIVRE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader)
echo "== GPU $GPU ($LIVRE livres) | grounding | $(date '+%F %T') =="
# O modelo (DiT de 5B + UMT5-XXL) ocupa ~25 GB em bf16. Aviso em vez de deixar
# estourar depois de dois minutos carregando.
if [ "${LIVRE%% *}" -lt 28000 ] 2>/dev/null; then
    echo "⚠️  menos de 28 GB livres nesta GPU — o modelo pode não caber."
fi

CHECKPOINT="${CHECKPOINT:-/data/train_output/fastwamd_corrida2_step1000/pretrained_model}"
# O dataset vive no home do hercules e é legível por todo mundo (755). Copiar
# 30 GB para cada login seria desperdício.
ROOT="${ROOT:-/home/hercules/DEV/prometheus-vla/lerobot-ext/meu_dataset/white_cup_on_dripper_2026-08-11}"
SAIDA="${SAIDA:-$HOME/saidas/grounding.png}"
mkdir -p "$(dirname "$SAIDA")"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "$PY" grounding_fastwamd.py \
    --checkpoint="$CHECKPOINT" \
    --root="$ROOT" \
    --saida="$SAIDA" \
    "$@"
