#!/usr/bin/env bash
# Servidor de inferência do FastWAM-D. Roda AO LADO do treino: usa uma cópia
# congelada do checkpoint, e não o `checkpoints/best` do treino — aquele é
# reescrito por rename atômico a cada melhora e o servidor leria o diretório
# sendo trocado embaixo dele.
#
#   uso: bash launch_server_fastwamd.sh [GPU] [checkpoint]
set -euo pipefail
GPU="${1:-0}"
CKPT="${2:-/data/train_output/fastwamd_teste_step1000/pretrained_model}"
ENV=$HOME/miniconda3/envs/prometheus-vla

# Sem isto o processo morre no `import torch` com
#   ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29` not found
#                (required by .../numpy/_core/_multiarray_umath.so)
# e, dependendo de onde o import estoura, vira Segmentation fault sem mensagem
# nenhuma. O libstdc++ do sistema é mais antigo que o exigido pelo numpy; o do
# ambiente conda tem a versão certa, mas o loader só o enxerga se ele vier
# ANTES no LD_LIBRARY_PATH. Ativar o conda não basta — a activate.d não mexe
# nessa variável.
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
# E o LD_PRELOAD por cima: o libzmq puxa o libstdc++ DO SISTEMA antes de
# qualquer coisa, e aí as extensoes nativas do diffusers/transformers (que
# precisam de GLIBCXX_3.4.29) estouram em Segmentation fault sem mensagem.
# Forcar o libstdc++ do conda a ser o primeiro resolve. Teste de bancada:
#   python -c "import zmq, torch, lerobot.policies.fastwam"   <- sem isto, segfault
export LD_PRELOAD="$ENV/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export HF_HOME=/data/.cache/huggingface
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1

cd ~/DEV/prometheus-vla/lerobot-ext
echo "== GPU $GPU | checkpoint $CKPT =="
exec "$ENV/bin/python" -u init_lerobot_inference_fastwamd_server.py \
    --checkpoint="$CKPT" --port=5600 --debug \
    2>&1 | tee -a /data/train_output/inferencia_server.log
