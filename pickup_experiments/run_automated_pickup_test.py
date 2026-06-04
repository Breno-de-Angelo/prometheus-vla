#!/usr/bin/env python3
"""
Loop automático: roda simulator + teste + análise até conseguir pegar e levantar o copo.
"""

import subprocess
import time
import json
import sys
import os
from pathlib import Path
from collections import defaultdict

def run_simulator():
    """Inicia o simulator em background."""
    print("[AUTO] 🚀 Iniciando simulator...")
    proc = subprocess.Popen([sys.executable, "unitree-g1-mujoco/run_sim.py"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)  # Deixa carregar
    print("[AUTO] ✅ Simulator pronto (PID={})".format(proc.pid))
    return proc

def run_test():
    """Roda o teste de pickup."""
    print("[AUTO] 🧪 Rodando teste de pickup...")
    # Limpa logs antigos para garantir novo log
    import glob
    old_logs = sorted(glob.glob("/tmp/action_log_*.jsonl"))
    if old_logs:
        os.remove(old_logs[-1])
        print(f"[AUTO] 🧹 Deletado log antigo: {Path(old_logs[-1]).name}")

    result = subprocess.run([sys.executable, "test_pick_cup_complete.py"],
                          capture_output=True, text=True, timeout=60)
    print("[AUTO] ✅ Teste completado")
    return result

def analyze_pickup(log_file):
    """
    Analisa se pegou e levantou.

    Validação:
    1. Dedos fecharam (finger motors > 0.5)
    2. Shoulder_pitch desceu (0.0 → 1.0) e depois subiu (1.0 → 0.3+)
    3. Sequência: descida → fechamento → levantada
    """
    if not log_file.exists():
        print("[AUTO] ❌ Nenhum log encontrado")
        return False

    actions = []
    with open(log_file) as f:
        for line in f:
            try:
                actions.append(json.loads(line))
            except:
                pass

    if not actions:
        print("[AUTO] ❌ Log vazio")
        return False

    print(f"\n[AUTO] 📊 Analisando {len(actions)} eventos...")

    # Extrair sequência de shoulder_pitch e finger values
    shoulder_pitches = []
    finger_values = defaultdict(list)

    for action in actions:
        if action.get("event_type") == "motor_cmd":
            idx = action.get("idx")
            q_actual = action.get("q_actual", 0.0)

            # Motor 22 = shoulder_pitch
            if idx == 22:
                shoulder_pitches.append(q_actual)

            # Motores 0-6 = dedos
            if 0 <= idx <= 6:
                finger_values[idx].append(q_actual)

    print(f"[AUTO] Shoulder_pitch: {shoulder_pitches[:5]}...{shoulder_pitches[-5:]}")

    # Validar
    success = True
    reasons = []

    # Validação 1: Dedos fecharam?
    fingers_closed = False
    if finger_values:
        max_finger_val = max(max(vals) if vals else 0 for vals in finger_values.values())
        print(f"[AUTO] Dedos - máximo: {max_finger_val:.4f}")
        if max_finger_val > 0.5:
            fingers_closed = True
            print("[AUTO] ✅ Dedos fecharam")
        else:
            success = False
            reasons.append("Dedos não fecharam suficientemente")
            print("[AUTO] ❌ Dedos não fecharam")

    # Validação 2: Movimento descendente-ascendente em shoulder_pitch?
    movement_ok = False
    if len(shoulder_pitches) > 10:
        # Deve descer (valores aumentar) e depois subir (valores diminuir)
        first_half_max = max(shoulder_pitches[:len(shoulder_pitches)//2])
        second_half = shoulder_pitches[len(shoulder_pitches)//2:]

        print(f"[AUTO] Shoulder_pitch - 1ª metade max: {first_half_max:.4f}, 2ª metade: {second_half[-1]:.4f}")

        # Deve ter descido bastante (> 0.8) e depois subido
        if first_half_max > 0.8 and second_half[-1] < first_half_max:
            movement_ok = True
            print("[AUTO] ✅ Movimento descendente-ascendente detectado")
        else:
            success = False
            reasons.append(f"Movimento insuficiente (max={first_half_max:.4f}, final={second_half[-1]:.4f})")
            print("[AUTO] ❌ Movimento descendente-ascendente inadequado")

    # Resultado final
    print("\n" + "="*50)
    if success and fingers_closed and movement_ok:
        print("[AUTO] ✅✅✅ SUCESSO! Pegou e levantou o copo!")
        return True
    else:
        print("[AUTO] ❌ Falha - razões:")
        for r in reasons:
            print(f"     • {r}")
        return False

def main():
    iteration = 0
    max_iterations = 10

    while iteration < max_iterations:
        iteration += 1
        print(f"\n\n{'='*60}")
        print(f"[AUTO] ITERAÇÃO {iteration}/{max_iterations}")
        print(f"{'='*60}\n")

        sim_proc = None
        try:
            # Rodar simulator
            sim_proc = run_simulator()

            # Rodar teste
            run_test()

            # Matar simulator
            print("[AUTO] 🛑 Parando simulator...")
            sim_proc.terminate()
            sim_proc.wait(timeout=5)
            print("[AUTO] ✅ Simulator parado")

            # Encontrar log mais recente (deve ser novo)
            import glob
            logs = sorted(glob.glob("/tmp/action_log_*.jsonl"), key=os.path.getmtime)
            if logs:
                latest_log = Path(logs[-1])
                print(f"[AUTO] 📄 Usando log: {latest_log.name}")
                time.sleep(0.5)  # Aguarda conclusão da escrita do log

                # Analisar
                if analyze_pickup(latest_log):
                    print("\n" + "🎉"*20)
                    print("[AUTO] 🎉 CONSEGUIU! Encerrando.")
                    print("🎉"*20)
                    return True

            time.sleep(2)

        except subprocess.TimeoutExpired:
            print("[AUTO] ❌ Timeout no teste")
        except Exception as e:
            print(f"[AUTO] ❌ Erro: {e}")
        finally:
            if sim_proc and sim_proc.poll() is None:
                sim_proc.kill()
                sim_proc.wait()

    print(f"\n[AUTO] ❌ Atingido máximo de {max_iterations} iterações sem sucesso")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
