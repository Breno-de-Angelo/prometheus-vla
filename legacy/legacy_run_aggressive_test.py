#!/usr/bin/env python3
"""
Loop: roda simulator + test_pick_cup_aggressive + análise até conseguir.
"""

import subprocess
import time
import json
import sys
import os
import glob
from pathlib import Path
from collections import defaultdict

def run_simulator():
    """Inicia o simulator em background."""
    print("[AGGR] 🚀 Iniciando simulator...")
    proc = subprocess.Popen([sys.executable, "unitree-g1-mujoco/run_sim.py"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)
    print("[AGGR] ✅ Simulator pronto (PID={})".format(proc.pid))
    return proc

def run_test():
    """Roda o teste agressivo."""
    print("[AGGR] 🧪 Rodando teste AGRESSIVO...")
    # Delete old log
    old_logs = sorted(glob.glob("/tmp/action_log_*.jsonl"))
    if old_logs:
        try:
            os.remove(old_logs[-1])
        except:
            pass

    result = subprocess.run([sys.executable, "test_pick_cup_aggressive.py"],
                          capture_output=True, text=True, timeout=60)
    print("[AGGR] ✅ Teste completado")
    return result

def analyze_pickup(log_file):
    """Analisa se pegou e levantou."""
    if not log_file.exists():
        print("[AGGR] ❌ Nenhum log encontrado")
        return False

    actions = []
    with open(log_file) as f:
        for line in f:
            try:
                actions.append(json.loads(line))
            except:
                pass

    if not actions:
        print("[AGGR] ❌ Log vazio")
        return False

    print(f"\n[AGGR] 📊 Analisando {len(actions)} eventos...")

    shoulder_pitches = []
    finger_values = defaultdict(list)

    for action in actions:
        if action.get("event_type") == "motor_cmd":
            idx = action.get("idx")
            q_actual = action.get("q_actual", 0.0)

            if idx == 22:
                shoulder_pitches.append(q_actual)
            if 0 <= idx <= 6:
                finger_values[idx].append(q_actual)

    print(f"[AGGR] Shoulder_pitch: {shoulder_pitches[:5]}...{shoulder_pitches[-5:]}")

    success = True
    reasons = []

    # Validação 1: Dedos fecharam FORTE (> 0.8)?
    fingers_closed = False
    if finger_values:
        max_finger_val = max(max(vals) if vals else 0 for vals in finger_values.values())
        print(f"[AGGR] Dedos - máximo: {max_finger_val:.4f}")
        if max_finger_val > 0.8:
            fingers_closed = True
            print("[AGGR] ✅ Dedos fecharam COM FORÇA")
        else:
            success = False
            reasons.append(f"Dedos insuficientes ({max_finger_val:.4f} < 0.8)")
            print("[AGGR] ❌ Dedos não fecharam com força")

    # Validação 2: Movimento descendente-ascendente?
    movement_ok = False
    if len(shoulder_pitches) > 10:
        first_half_max = max(shoulder_pitches[:len(shoulder_pitches)//2])
        second_half_min = min(shoulder_pitches[len(shoulder_pitches)//2:])

        print(f"[AGGR] Shoulder_pitch - 1ª metade max: {first_half_max:.4f}, 2ª metade min: {second_half_min:.4f}")

        # Deve ter descido bem (> 1.0) e depois subido (final < max)
        if first_half_max > 1.0 and second_half_min < first_half_max - 0.2:
            movement_ok = True
            print("[AGGR] ✅ Movimento descendente-ascendente detectado")
        else:
            success = False
            reasons.append(f"Movimento insuficiente (max={first_half_max:.4f}, min={second_half_min:.4f})")
            print("[AGGR] ❌ Movimento inadequado")

    print("\n" + "="*50)
    if success and fingers_closed and movement_ok:
        print("[AGGR] ✅✅✅ SUCESSO! PEGOU E LEVANTOU!")
        return True
    else:
        print("[AGGR] ❌ Falha:")
        for r in reasons:
            print(f"     • {r}")
        return False

def main():
    iteration = 0
    max_iterations = 5

    while iteration < max_iterations:
        iteration += 1
        print(f"\n\n{'='*60}")
        print(f"[AGGR] ITERAÇÃO {iteration}/{max_iterations} (AGRESSIVA)")
        print(f"{'='*60}\n")

        sim_proc = None
        try:
            sim_proc = run_simulator()
            run_test()

            print("[AGGR] 🛑 Parando simulator...")
            sim_proc.terminate()
            sim_proc.wait(timeout=5)
            print("[AGGR] ✅ Simulator parado")

            logs = sorted(glob.glob("/tmp/action_log_*.jsonl"), key=os.path.getmtime)
            if logs:
                latest_log = Path(logs[-1])
                print(f"[AGGR] 📄 Usando log: {latest_log.name}")
                time.sleep(0.5)

                if analyze_pickup(latest_log):
                    print("\n" + "🎉"*20)
                    print("[AGGR] 🎉 CONSEGUIU COM AGRESSIVIDADE!")
                    print("🎉"*20)
                    return True

            time.sleep(2)

        except subprocess.TimeoutExpired:
            print("[AGGR] ❌ Timeout")
        except Exception as e:
            print(f"[AGGR] ❌ Erro: {e}")
        finally:
            if sim_proc and sim_proc.poll() is None:
                sim_proc.kill()
                sim_proc.wait()

    print(f"\n[AGGR] ❌ Atingido máximo de {max_iterations} iterações")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
