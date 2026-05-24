#!/usr/bin/env bash
# Inferência pi05-D na Atenas -> simulador MuJoCo do Mori (modo distribuído via ZMQ).
# Uso: bash run_atenas_sim.sh   (suba o run_sim.py no Mori ANTES)
set -e

source /home/hercules/miniconda3/etc/profile.d/conda.sh
conda activate ms3
cd /home/hercules/prometheus-vla/lerobot-ext

CKPT=/home/hercules/prometheus-vla/train/output/he_then_cup3_finetune/checkpoints/best/pretrained_model

python init_lerobot_inference_pi05d_v2.py \
  --sim \
  --checkpoint="$CKPT" \
  --cam-robot=10.8.8.52 \
  --port-cam=5555 \
  --task="Pick up the cup" \
  --debug
