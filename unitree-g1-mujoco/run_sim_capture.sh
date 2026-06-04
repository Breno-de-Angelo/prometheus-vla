#!/usr/bin/env bash
# Captura de validação: roda o entrypoint REAL em --sim com diagnóstico in-process.
# Uso: run_sim_capture.sh [segundos] [tagdataset]
set +e
SECS="${1:-50}"
TAG="${2:-run}"
PY=~/miniconda3/envs/g1/bin/python
ROOT=/home/luiz-aumo/I2CA/prometheus-vla
OUT=/tmp/sim_val
mkdir -p "$OUT"

echo "[cap] limpando órfãos (5555 / 6000-6003)..."
pkill -9 -f init_lerobot 2>/dev/null
pkill -9 -f image_publish 2>/dev/null
pkill -9 -f multiprocessing.spawn 2>/dev/null
pkill -9 -f run_g1_server 2>/dev/null
pkill -9 -f sim_validate_host 2>/dev/null
for p in $(ss -tlnp 2>/dev/null | grep -oP ':(5555|600[0-3])\b' >/dev/null; ss -tlnp 2>/dev/null | grep -E ':(5555|600[0-3])' | grep -oP 'pid=\K[0-9]+'); do
  kill -9 "$p" 2>/dev/null && echo "[cap] matei órfão pid=$p"
done
sleep 2

rm -f "$OUT/diag.jsonl" "$OUT/client.log" "$OUT/diag_err.log"
rm -rf "/tmp/sim_val_ds/$TAG"; mkdir -p "/tmp/sim_val_ds"

echo "[cap] rodando --sim por ${SECS}s (G1_ACTION_LOG + SIM_VAL_DIAG)..."
cd "$ROOT"
rm -f "$OUT/ext_action.jsonl"
SIM_VAL_DIAG=1 G1_ACTION_LOG="$OUT/ext_action.jsonl" MUJOCO_GL=egl timeout -s INT "${SECS}" "$PY" \
  lerobot-ext/init_lerobot_record_v2.py \
  --config_path lerobot-ext/config/record/record_televuer.yaml \
  --sim --dataset.root="/tmp/sim_val_ds/$TAG" > "$OUT/client.log" 2>&1

echo "[cap] terminou. limpando órfãos pós-run..."
pkill -9 -f init_lerobot 2>/dev/null
pkill -9 -f image_publish 2>/dev/null
pkill -9 -f multiprocessing.spawn 2>/dev/null
sleep 1
echo "[cap] diag.jsonl: $(wc -l < "$OUT/diag.jsonl" 2>/dev/null || echo 0) linhas"
echo "[cap] FIM"
