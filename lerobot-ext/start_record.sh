#!/usr/bin/env bash
# Wrapper para init_lerobot_record.py — seleciona config e passa flags extras.
# Uso: ./start_record.sh [--config <yaml>] [--sim] [--left-arm-limp] [--debug] [--dry-preflight] [-- ARGS...]
#
# Configs disponíveis (em config/record/):
#   pick_up_the_cup4  (default)
#   pick_up_the_cup3
#   pick_up_the_cup2
#   pick_up_the_cup
#   get_kettle
#   record_televuer
#   record_realsense_mediapipe
#   record_key

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/config/record"
DEFAULT_CONFIG="pick_up_the_cup4"
DEFAULT_ROBOT_IP="192.168.123.164"
ENV_NAME="${ENV_NAME:-g1}"

# ativa conda (mesmo padrão dos outros scripts)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || { echo "[FATAL] não ativei env '$ENV_NAME'"; exit 1; }
echo "[*] env conda ATIVADO: $ENV_NAME"

CONFIG_NAME="$DEFAULT_CONFIG"
ROBOT_IP="$DEFAULT_ROBOT_IP"
EXTRA_ARGS=()

# Parsing manual para capturar --config e repassar o resto
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --config=*)
            CONFIG_NAME="${1#--config=}"
            shift
            ;;
        --robot-ip)
            ROBOT_IP="$2"
            shift 2
            ;;
        --robot-ip=*)
            ROBOT_IP="${1#--robot-ip=}"
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

CONFIG_FILE="$CONFIG_DIR/${CONFIG_NAME}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERRO] Config não encontrada: $CONFIG_FILE" >&2
    echo "Configs disponíveis:" >&2
    ls "$CONFIG_DIR"/*.yaml | xargs -n1 basename | sed 's/\.yaml$//' | sed 's/^/  /' >&2
    exit 1
fi

echo "[start_record] Config:    $CONFIG_NAME"
echo "[start_record] Robot IP:  $ROBOT_IP"
echo "[start_record] Script:    $SCRIPT_DIR/init_lerobot_record.py"
echo

exec python "$SCRIPT_DIR/init_lerobot_record.py" \
    --config_path "$CONFIG_FILE" \
    --robot.robot_ip="$ROBOT_IP" \
    --dataset.push_to_hub=false \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
