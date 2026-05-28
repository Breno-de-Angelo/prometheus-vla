#!/usr/bin/env python3
"""Analisa último log de ações para validar pickup."""

import json
import glob
from pathlib import Path

def main():
    # Encontra último log
    logs = glob.glob("/tmp/action_log_*.jsonl")
    if not logs:
        print("❌ Nenhum log encontrado")
        return False

    logfile = sorted(logs)[-1]

    # Analisa eventos
    hand_positions = []
    shoulder_positions = []

    with open(logfile) as f:
        for line in f:
            try:
                event = json.loads(line)
                if event.get("event_type") == "motor_cmd":
                    # Mão
                    for m in event.get("hand_motors", []):
                        if m.get("idx") in [3, 4, 5, 6]:  # dedos
                            hand_positions.append(m.get("q", 0))

                    # Ombro
                    for m in event.get("body_motors", []):
                        if m.get("idx") == 22:  # shoulder pitch
                            shoulder_positions.append(m.get("q", 0))
            except:
                pass

    # Valida
    hand_closed = max(hand_positions) > 0.8 if hand_positions else False
    shoulder_descended = max(shoulder_positions) > 0.5 if shoulder_positions else False

    print(f"📊 Análise: {Path(logfile).name}")
    print(f"  Mão fechou: {max(hand_positions) if hand_positions else 0:.2f}")
    print(f"  Ombro desceu: {max(shoulder_positions) if shoulder_positions else 0:.2f}")

    if hand_closed and shoulder_descended:
        print("✅ Trajetória executada (mas copo pode não ter sido pego)")
        return True
    else:
        print("❌ Trajetória não executada")
        return False

if __name__ == "__main__":
    main()
