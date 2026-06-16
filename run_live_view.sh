#!/usr/bin/env bash
# Abre a comparação AO VIVO: câmera real do robô (com overlay dos ArUcos) | sim.
# Janela cv2 atualiza em tempo real; aperte Q ou ESC pra fechar.
#
# Uso:  ./run_live_view.sh            (robô em 192.168.68.71)
#       ./run_live_view.sh 10.9.8.73  (outro IP)
set -e
HOST="${1:-192.168.68.71}"
cd "$(dirname "$0")"
export MUJOCO_GL=egl
exec ~/miniconda3/envs/g1/bin/python -u compare_sim_real_view.py \
    --host "$HOST" --timeout 20 --live
