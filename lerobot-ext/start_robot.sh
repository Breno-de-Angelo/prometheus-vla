#!/usr/bin/env bash
# Sobe os dois servicos do robo num terminal so:
#   - run_g1_server.py      (ponte DDS<->ZMQ: corpo + Dex3, portas 6000-6003)
#   - realsense_server.py   (camera ZMQ :5555, head_camera RGB[+depth])
# Confirma cada servico ativo, avisa quando alguem conecta, e no Ctrl-C
# derruba tudo + limpa portas. Logs em /tmp/g1_server.log e /tmp/g1_camera.log
set -uo pipefail

# REPO = nivel acima deste script (lerobot-ext/start_robot.sh -> repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$SCRIPT_DIR")}"
ENV_NAME="${ENV_NAME:-g1}"
# permite override p/ teste (default = scripts reais)
SRV="${SRV:-$REPO/lerobot-ext/robot/unitree_g1/run_g1_server.py}"
CAM="${CAM:-$REPO/lerobot-ext/Scripts_Prometheus_int/realsense_server.py}"

CAM_PORT=5555
SRV_PORTS=(6000 6001 6002 6003)
WATCH_PORTS=(5555 6000 6001 6002 6003)

# ativa conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || { echo "[FATAL] nao ativei env '$ENV_NAME'"; exit 1; }
echo "[*] env conda ATIVADO: $ENV_NAME"

[ -f "$SRV" ] || { echo "[FATAL] nao achei $SRV"; exit 1; }
[ -f "$CAM" ] || { echo "[FATAL] nao achei $CAM"; exit 1; }

# espera uma porta entrar em LISTEN (porta, timeout_s) -> 0 ok / 1 timeout
wait_listen() {
  local port=$1 t=${2:-20} i=0
  while ! ss -ltnH "sport = :$port" 2>/dev/null | grep -q .; do
    sleep 0.5; i=$((i+1)); [ "$i" -ge $((t*2)) ] && return 1
  done
  return 0
}

# limpa portas de runs orfaos
for port in "${WATCH_PORTS[@]}"; do fuser -k ${port}/tcp 2>/dev/null; done
sleep 1

pids=()
cleanup() {
  trap - INT TERM EXIT   # desarma pra nao reentrar (INT->EXIT, watchdog->TERM)
  echo; echo "[*] encerrando..."
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null; done
  for port in "${WATCH_PORTS[@]}"; do fuser -k ${port}/tcp 2>/dev/null; done
  wait 2>/dev/null
  echo "[*] fim. portas liberadas."
}
trap cleanup INT TERM EXIT

# --- 1) run_g1_server ---
echo "[*] subindo run_g1_server  (log: /tmp/g1_server.log)..."
stdbuf -oL python -u "$SRV" > >(tee /tmp/g1_server.log) 2>&1 &
SRV_PID=$!; pids+=($SRV_PID)
if wait_listen "${SRV_PORTS[1]}" 25; then
  echo "✅ run_g1_server ATIVO (DDS<->ZMQ, portas ${SRV_PORTS[0]}-${SRV_PORTS[-1]})"
else
  echo "⚠️  run_g1_server NAO abriu a porta ${SRV_PORTS[1]} em 25s — ver /tmp/g1_server.log"
fi

# --- 2) realsense_server ---
echo "[*] subindo realsense_server (log: /tmp/g1_camera.log)..."
stdbuf -oL python -u "$CAM" > >(tee /tmp/g1_camera.log) 2>&1 &
CAM_PID=$!; pids+=($CAM_PID)
if wait_listen "$CAM_PORT" 25; then
  echo "✅ realsense_server ATIVO (camera ZMQ :$CAM_PORT)"
else
  echo "⚠️  realsense_server NAO abriu :$CAM_PORT em 25s — ver /tmp/g1_camera.log"
fi

# --- monitor de conexoes: avisa quando alguem conecta/desconecta nas portas ---
conn_monitor() {
  local filt="( sport = :5555 or sport = :6000 or sport = :6001 or sport = :6002 or sport = :6003 )"
  declare -A active
  while :; do
    declare -A now=()
    while read -r laddr peer; do
      [ -z "$peer" ] && continue
      local lport=${laddr##*:}
      local key="$peer @ :$lport"
      now["$key"]=1
      if [ -z "${active["$key"]:-}" ]; then
        echo "🔌 [$(date +%H:%M:%S)] CONECTOU: $peer  ->  porta $lport"
      fi
    done < <(ss -tnH state established "$filt" 2>/dev/null | awk '{print $3, $4}')
    for key in "${!active[@]}"; do
      [ -z "${now["$key"]:-}" ] && echo "❌ [$(date +%H:%M:%S)] DESCONECTOU: $key"
    done
    active=(); for k in "${!now[@]}"; do active["$k"]=1; done
    sleep 2
  done
}
conn_monitor & pids+=($!)
echo "[*] monitor de conexoes ATIVO (avisa conexao/desconexao nas portas ${WATCH_PORTS[*]})"

# --- watchdog: se SRV ou CAM cair, derruba tudo ---
( while :; do
    kill -0 "$SRV_PID" 2>/dev/null || { echo "[!] run_g1_server caiu"; kill -TERM $$; exit; }
    kill -0 "$CAM_PID" 2>/dev/null || { echo "[!] realsense_server caiu"; kill -TERM $$; exit; }
    sleep 2
  done ) &
pids+=($!)

echo "[*] tudo no ar. Ctrl-C para parar."
wait
