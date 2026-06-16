#!/usr/bin/env bash
# Sobe a inferencia da VLA + o dashboard OmniView num terminal so:
#   - live_omniview.py                       (dashboard web :8013, consome camera :5555 + telemetria :5557)
#   - inference_realtime_pi05d_right14.py    (a VLA: le a camera, roda a politica, manda as acoes)
#
# Por DEFAULT roda em --dry-run (a VLA calcula as acoes mas NAO move o robo).
# Pra mover de verdade:   DRY_RUN=0 ./start_inference.sh
#
# Pre-requisito: start_robot.sh ja no ar (camera ZMQ :5555 + ponte DDS).
# Ctrl-C derruba os dois e limpa a porta do dashboard. Log do dashboard em /tmp/omniview.log.
#
# Overrides por env var (todos opcionais):
#   CKPT=<dir pretrained_model[_ema]>   ROBOT_IP=10.9.8.73   TASK="Pick up the white cup"
#   GPU=0   HTTP_PORT=8013   DRY_RUN=1   ENV_NAME=ms3
# Args extras vao direto pra VLA:  ./start_inference.sh --rtc --max-delta 0.03
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$SCRIPT_DIR")}"          # lerobot-ext/start_inference.sh -> repo
ENV_NAME="${ENV_NAME:-ms3}"

# --- parametros (override por env var) ---
CKPT="${CKPT:-$REPO/train_output/cup_pi05_right8_armstate7_run4a_lf/checkpoints/best/pretrained_model_ema}"
ROBOT_IP="${ROBOT_IP:-10.9.8.73}"
TASK="${TASK:-Pick up the white cup}"
GPU="${GPU:-0}"                                   # GPU0 (treino fica na 2)
HTTP_PORT="${HTTP_PORT:-8013}"
HTTP_HOST="${HTTP_HOST:-0.0.0.0}"                 # 0.0.0.0 = acessivel na rede do lab (http://<atena>:8013). 127.0.0.1 = so local (precisa tunel SSH)
RECORD="${RECORD:-1}"                             # 1 = grava RGB+atencao+depth em train/log/run_<ts>/ (p/ replay HTML). 0 = so o log de texto (poupa disco)
DRY_RUN="${DRY_RUN:-1}"                           # 1 = NAO move (default seguro) · 0 = move o robo

VLA="$REPO/inference_realtime_pi05d_right14.py"
OMNI="$REPO/lerobot-ext/tools/live_omniview.py"

# ativa conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || { echo "[FATAL] nao ativei env '$ENV_NAME'"; exit 1; }
echo "[*] env conda ATIVADO: $ENV_NAME"

[ -f "$VLA" ]  || { echo "[FATAL] nao achei a VLA: $VLA"; exit 1; }
[ -f "$OMNI" ] || { echo "[FATAL] nao achei o OmniView: $OMNI"; exit 1; }
[ -d "$CKPT" ] || { echo "[FATAL] checkpoint inexistente: $CKPT"; exit 1; }

export PYTHONPATH="$REPO/lerobot-ext${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/data/huggingface-models}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$REPO"

# espera uma porta entrar em LISTEN (porta, timeout_s) -> 0 ok / 1 timeout
wait_listen() {
  local port=$1 t=${2:-15} i=0
  while ! ss -ltnH "sport = :$port" 2>/dev/null | grep -q .; do
    sleep 0.5; i=$((i+1)); [ "$i" -ge $((t*2)) ] && return 1
  done
  return 0
}

# limpa o :8013 de runs orfaos
fuser -k ${HTTP_PORT}/tcp 2>/dev/null; sleep 1

OMNI_PID=""
cleanup() {
  trap - INT TERM EXIT
  echo; echo "[*] encerrando..."
  [ -n "$OMNI_PID" ] && kill "$OMNI_PID" 2>/dev/null
  fuser -k ${HTTP_PORT}/tcp 2>/dev/null
  wait 2>/dev/null
  echo "[*] fim. dashboard derrubado."
}
trap cleanup INT TERM EXIT

# --- 1) dashboard OmniView (background) ---
echo "[*] subindo OmniView (dashboard :$HTTP_PORT, log: /tmp/omniview.log)..."
python -u "$OMNI" --host "$ROBOT_IP" --http-host "$HTTP_HOST" --http-port "$HTTP_PORT" --no-browser >/tmp/omniview.log 2>&1 &
OMNI_PID=$!
IP_LOCAL="$(hostname -I 2>/dev/null | awk '{print $1}')"
if wait_listen "$HTTP_PORT" 15; then
  if [ "$HTTP_HOST" = "127.0.0.1" ]; then
    echo "✅ OmniView ATIVO (so local) -> abra um tunel: ssh -L $HTTP_PORT:127.0.0.1:$HTTP_PORT $(whoami)@${IP_LOCAL:-<atena>}  e use http://localhost:$HTTP_PORT/inference.html"
  else
    echo "✅ OmniView ATIVO  ->  http://${IP_LOCAL:-<atena>}:$HTTP_PORT/inference.html   (acessivel na rede do lab)"
  fi
else
  echo "⚠️  OmniView nao abriu :$HTTP_PORT em 15s — ver /tmp/omniview.log (segue mesmo assim)"
fi

# --- 2) VLA (foreground) ---
DRY_FLAG=""
if [ "$DRY_RUN" = "1" ]; then
  DRY_FLAG="--dry-run"
  echo "[*] modo DRY-RUN: a VLA calcula as acoes mas NAO move o robo."
  echo "    pra valer:  DRY_RUN=0 $0"
else
  echo "############################################################"
  echo "##  ATENCAO: DRY_RUN=0  ->  O ROBO VAI SE MOVER          ##"
  echo "##  copo posicionado? area livre? e-stop a mao?          ##"
  echo "##  (Ctrl-C nos proximos 4s pra abortar)                 ##"
  echo "############################################################"
  sleep 4
fi

REC_FLAG=""
if [ "$RECORD" = "1" ]; then
  REC_FLAG="--record"
  echo "[*] gravacao LIGADA: RGB+atencao+depth -> train/log/run_<ts>/ (RECORD=0 desliga)"
fi

echo "[*] subindo VLA (GPU$GPU, robot=$ROBOT_IP, ckpt=.../$(basename "$(dirname "$CKPT")")/$(basename "$CKPT"))..."
# foreground (sem exec): ao terminar/Ctrl-C, o trap cleanup derruba o OmniView
CUDA_VISIBLE_DEVICES="$GPU" python -u "$VLA" \
  --checkpoint "$CKPT" \
  --robot-ip "$ROBOT_IP" \
  --task "$TASK" \
  --live $DRY_FLAG $REC_FLAG "$@"
