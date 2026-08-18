#!/usr/bin/env bash
# Lançamento do FastWAM-D na athena.
#
#   uso: bash launch_fastwamd.sh [GPU] [args extras para o run_train]
#
# Sem argumento, escolhe a GPU com mais memória livre AGORA. As três A100 são
# compartilhadas: fixar a GPU 0 foi o que causou o `OutOfMemoryError` de 13:47,
# com outro job ocupando 57 dos 80 GB dela.
set -euo pipefail

# ── wandb ────────────────────────────────────────────────────────────────
# A conta `hercules` tem um `~/.netrc` com a chave de OUTRA pessoa. Se nada for
# feito, o wandb usa aquilo em silêncio e a corrida sobe na conta errada — foi
# o que aconteceu às 14:04. Por isso a chave sai de um arquivo próprio e o
# script ABORTA se ele não existir, em vez de cair no `.netrc`.
#
#   crie assim (no SEU terminal, para a chave não passar por log nenhum):
#     ssh hercules@10.9.8.252 "cat > ~/.wandb_key && chmod 600 ~/.wandb_key"
#     <cole a chave de https://wandb.ai/authorize, Enter, Ctrl-D>
CHAVE=~/.wandb_key
if [ -f "$CHAVE" ]; then
    WANDB_API_KEY=$(tr -d "[:space:]" < "$CHAVE")
    export WANDB_API_KEY
else
    echo "ERRO: $CHAVE não existe." >&2
    echo "Sem ele o wandb usaria a chave do ~/.netrc, que é de outra conta." >&2
    echo "Crie o arquivo (veja o comentário no topo deste script) ou rode com" >&2
    echo "  bash $0 $* --wandb.enable=false" >&2
    [[ " $* " == *" --wandb.enable=false "* ]] || exit 1
fi

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    GPU="$1"; shift
else
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d " ")
fi
LIVRE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader)
echo "== GPU $GPU ($LIVRE livres) | wandb: ${WANDB_API_KEY:+chave própria}${WANDB_API_KEY:-desligado} =="

cd ~/DEV/prometheus-vla/lerobot-ext
# Pesos e saída no disco de dados: o disco de sistema vive perto de 100%.
export HF_HOME=/data/.cache/huggingface
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "/home/hercules/miniconda3/envs/prometheus-vla/bin/python" -m policies.fastwam_depth.run_train \
    --config_path=config/train/fastwamdepth_white_cup_on_dripper.yaml "$@" \
    2>&1 | tee -a /data/train_output/fastwamdepth.log
