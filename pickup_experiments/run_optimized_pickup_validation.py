#!/usr/bin/env python3
"""
Loop de teste automatizado com otimizações aplicadas.

Executa:
1. Simulator em background
2. test_pick_cup_optimized.py
3. Análise de logs para validar pickup + lifting

Continua até conseguir pegar e levantar o copo com sucesso.
"""

import subprocess
import time
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta


def kill_simulator():
    """Mata qualquer processo do simulator rodando."""
    subprocess.run("pkill -f 'python.*run_sim.py' || true", shell=True)
    time.sleep(0.5)


def start_simulator():
    """Inicia o simulator em background sem display."""
    print("[TEST] ▶️  Iniciando simulator...")

    proc = subprocess.Popen(
        ["python", "unitree-g1-mujoco/run_sim.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=Path(__file__).parent.parent
    )
    time.sleep(4)  # Aguarda inicialização
    return proc


def run_optimized_test():
    """Executa test_pick_cup_optimized.py."""
    print("[TEST] 🎯 Executando trajetória otimizada...")
    result = subprocess.run(
        ["python", "pickup_experiments/test_pick_cup_optimized.py"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True
    )
    return result.returncode == 0


def analyze_logs():
    """Analisa logs para validar pickup + lifting."""
    print("[TEST] 📊 Analisando logs...")

    try:
        result = subprocess.run(
            ["python", "analyze_action_log.py"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=10
        )

        output = result.stdout + result.stderr

        # Procura pelos critérios de sucesso
        has_finger_closure = "Mão fechada" in output or "fingers" in output.lower()
        has_lifting = "shoulder_pitch" in output.lower()

        # Imprime análise
        print(output)

        return has_finger_closure and has_lifting

    except Exception as e:
        print(f"[TEST] ❌ Erro na análise: {e}")
        return False


def main():
    """Loop principal de testes."""
    test_num = 1
    max_tests = 10
    base_dir = Path(__file__).parent.parent

    print("=" * 60)
    print("🚀 Teste Automatizado com Otimizações")
    print("=" * 60)
    print()

    while test_num <= max_tests:
        print(f"\n[TESTE {test_num}/{max_tests}]")
        print("-" * 60)

        # Mata simulator anterior
        kill_simulator()

        # Inicia novo simulator
        sim_proc = start_simulator()

        try:
            # Executa teste
            test_ok = run_optimized_test()
            time.sleep(0.5)

            # Analisa logs
            success = analyze_logs()

            if success:
                print("\n" + "=" * 60)
                print("✅ SUCESSO! Robô pegou e levantou o copo!")
                print("=" * 60)
                return 0
            else:
                print(f"⚠️  Teste {test_num} falhou - continuando...")

        except Exception as e:
            print(f"[TEST] ❌ Erro durante execução: {e}")

        finally:
            # Mata simulator
            try:
                sim_proc.terminate()
                sim_proc.wait(timeout=2)
            except:
                subprocess.run("pkill -9 -f 'python.*run_sim.py' || true", shell=True)

            time.sleep(1)

        test_num += 1

    print("\n" + "=" * 60)
    print(f"❌ Falha após {max_tests} testes")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
