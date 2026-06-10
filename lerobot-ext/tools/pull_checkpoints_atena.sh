#!/usr/bin/env bash
# Puxa checkpoints numerados de um treino na Atena pro oldssd do laptop e
# libera o disco remoto (A100 com disco 99% cheio não comporta todos os ckpts
# de 9.1GB; o oldssd comporta).
#
# Só move dirs NUMERADOS (ex.: 001000) com mtime > 5 min (escrita concluída).
# best/ e last/ ficam na Atena. Cada ckpt é verificado com uma segunda
# passada de rsync (itemize vazio = transferência íntegra) antes do rm remoto.
#
# Uso:
#   nohup lerobot-ext/tools/pull_checkpoints_atena.sh <job_name> &
#   ex.: pull_checkpoints_atena.sh cup_pi05_right8_1squeeze_lf
set -u

JOB="${1:?uso: pull_checkpoints_atena.sh <job_name>}"
REMOTE=hercules@10.9.8.252
RDIR="Prometheus/Luiz/prometheus-vla/train_output/${JOB}/checkpoints"
LDIR="/mnt/oldssd/luiz-aumo/prometheus_ckpts/${JOB}/checkpoints"
INTERVAL=300

mkdir -p "$LDIR"
echo "$(date +%F_%T) pull loop iniciado: ${REMOTE}:${RDIR} -> ${LDIR}"

while true; do
    dirs=$(ssh -o ConnectTimeout=10 "$REMOTE" \
        "find ~/$RDIR -maxdepth 1 -mindepth 1 -type d -mmin +5 2>/dev/null" \
        | grep -E '/[0-9]+$' | sort) || dirs=""

    for d in $dirs; do
        name=$(basename "$d")
        echo "$(date +%F_%T) puxando $name ..."
        rsync -a --partial "$REMOTE:$d/" "$LDIR/$name/" || { echo "  rsync falhou, tento na próxima volta"; continue; }
        # segunda passada: itemize vazio = nada faltando/mudado
        pending=$(rsync -ai --dry-run "$REMOTE:$d/" "$LDIR/$name/" | head -1)
        if [ -n "$pending" ]; then
            echo "  verificação acusou pendência ($pending), não removo ainda"
            continue
        fi
        ssh "$REMOTE" "rm -rf '$d'"
        echo "$(date +%F_%T) ✓ $name puxado e liberado da Atena ($(du -sh "$LDIR/$name" | cut -f1))"
    done
    sleep "$INTERVAL"
done
