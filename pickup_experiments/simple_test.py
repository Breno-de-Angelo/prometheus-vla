#!/usr/bin/env python3
"""Roda simulator + teste IK-based e analisa resultado."""

import subprocess
import time
import json
import glob
import sys
from pathlib import Path


def run_test():
    cwd = Path(__file__).parent.parent

    # Mata qualquer simulador anterior
    subprocess.run("pkill -9 -f 'run_sim.py' 2>/dev/null; sleep 1", shell=True)
    time.sleep(1)

    pids = subprocess.run(
        "ps aux | grep 'python.*run_sim\\.py' | grep -v grep",
        shell=True, capture_output=True, text=True
    )
    if pids.stdout.strip():
        print(f"[AVISO] Simulador ainda rodando, abortando!\n{pids.stdout.strip()}")
        return None

    # Remove logs antigos para não confundir análise
    old_logs = glob.glob("/tmp/action_log_*.jsonl")
    for f in old_logs:
        Path(f).unlink()

    # Inicia simulator em modo headless
    print("Iniciando simulator...")
    sim = subprocess.Popen(
        'eval "$(conda shell.bash hook)" && conda activate g1 && '
        'python3 unitree-g1-mujoco/run_sim.py',
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=True,
        cwd=cwd,
    )
    time.sleep(7)  # Aguarda inicializacao completa

    # Executa trajetoria IK
    print("Executando trajetoria IK...")
    result = subprocess.run(
        'eval "$(conda shell.bash hook)" && conda activate g1 && '
        'python3 pickup_experiments/test_pick_cup_optimized.py',
        capture_output=True, text=True, shell=True, cwd=cwd,
        timeout=60,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:300])

    time.sleep(2)  # Aguarda logs serem gravados

    sim.terminate()
    try:
        sim.wait(timeout=3)
    except Exception:
        sim.kill()

    return result


def analyze_logs():
    logs = sorted(glob.glob("/tmp/action_log_*.jsonl"))
    if not logs:
        print("Nenhum log encontrado em /tmp/action_log_*.jsonl")
        return False

    latest_log = logs[-1]
    print(f"\nAnalisando: {Path(latest_log).name}")

    with open(latest_log) as f:
        events = [json.loads(line) for line in f if line.strip()]

    physics_states = [e for e in events if e.get("event_type") == "physics_state"]
    motor_cmds     = [e for e in events if e.get("event_type") == "motor_cmd"]
    actions_rcvd   = [e for e in events if e.get("event_type") == "action_received"]

    print(f"  Eventos physics_state: {len(physics_states)}")
    print(f"  Eventos motor_cmd:     {len(motor_cmds)}")
    print(f"  Acoes recebidas:       {len(actions_rcvd)}")

    if not actions_rcvd:
        print("ERRO: Nenhuma acao recebida — ZMQ nao conectou!")
        return False

    # Extrai posicao do ombro (actuator 29 no body_motors[29])
    shoulder_vals = []
    hand_vals = []
    for cmd in motor_cmds:
        idx = cmd.get("idx")
        if idx == 22:  # right_shoulder_pitch
            shoulder_vals.append(cmd.get("q_actual", 0))

        for bm in cmd.get("body_motors", []):
            if bm.get("idx") == 22:
                shoulder_vals.append(bm.get("q", 0))
        for hm in cmd.get("hand_motors", []):
            hand_vals.append(hm.get("q", 0))

    # Analise de fisica
    cup_heights = [s["cup_height"] for s in physics_states if "cup_height" in s]
    contacts    = [s.get("num_contacts", 0) for s in physics_states]

    if not cup_heights:
        print("AVISO: Sem dados de fisica (cup_height) no log")

    # Cup settled height = media dos primeiros 10% de registros
    settled_idx = max(1, len(cup_heights) // 10)
    cup_settled = float(sum(cup_heights[:settled_idx]) / settled_idx) if cup_heights else 0.799

    cup_max    = max(cup_heights) if cup_heights else 0
    cup_final  = cup_heights[-1] if cup_heights else 0
    cup_delta  = cup_max - cup_settled
    max_contacts = max(contacts) if contacts else 0

    shoulder_max = max(shoulder_vals) if shoulder_vals else 0
    hand_max     = max(hand_vals) if hand_vals else 0

    print(f"\n=== Resultado ===")
    print(f"  Ombro max:        {shoulder_max:.3f}  (esperado ~0.33)")
    print(f"  Mao max:          {hand_max:.3f}  (esperado ~0.9)")
    print(f"  Cup settled Z:    {cup_settled:.3f}")
    print(f"  Cup max Z:        {cup_max:.3f}")
    print(f"  Cup delta Z:      {cup_delta:+.3f}")
    print(f"  Contatos max:     {max_contacts}")

    # Veredicto
    cup_lifted = cup_delta > 0.04  # levantou mais de 4cm
    grip_made  = max_contacts > 0
    traj_ran   = hand_max > 0.5

    if cup_lifted and grip_made:
        print(f"\nSUCESSO! Copo levantado {cup_delta*100:.1f}cm com {max_contacts} contatos!")
        return True
    elif cup_lifted:
        print(f"\nCopo levantou {cup_delta*100:.1f}cm mas sem contato detectado (grip por fricao?)")
        return True
    elif grip_made:
        print(f"\nContato detectado ({max_contacts} contatos) mas copo nao levantou ({cup_delta*100:.1f}cm)")
        return False
    elif traj_ran:
        print(f"\nTrajetoria executada mas sem contato nem levantamento")
        print(f"  Verificar posicionamento do braco (cup em Z={cup_settled:.3f})")
        return False
    else:
        print(f"\nTrajetoria nao executada (hand_max={hand_max:.2f})")
        return False


def main():
    print("=" * 60)
    print("Teste de pickup - Trajetoria IK")
    print("=" * 60)

    run_test()
    success = analyze_logs()

    if success:
        print("\n[DONE] Conseguiu pegar e levantar o copo!")
    else:
        print("\n[DONE] Nao conseguiu levantar o copo ainda.")

    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
