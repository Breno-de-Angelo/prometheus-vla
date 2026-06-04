#!/usr/bin/env python3
"""
Script pra analisar logs de ações.
Uso: python analyze_action_log.py /tmp/action_log_YYYYMMDD_HHMMSS.jsonl
"""
import json
import sys
from pathlib import Path
from collections import defaultdict


def analyze_log(log_file):
    """Analisa arquivo de log."""
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"❌ Arquivo não encontrado: {log_file}")
        return

    events = []
    with open(log_path) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️ Linha inválida: {line[:50]}")

    print(f"\n📊 Análise de Log: {log_path.name}")
    print(f"{'='*60}")
    print(f"Total de eventos: {len(events)}")

    # Contar por tipo
    event_counts = defaultdict(int)
    for event in events:
        event_counts[event.get("event_type")] += 1

    print(f"\n📈 Eventos por tipo:")
    for event_type, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {event_type}: {count}")

    # Ações recebidas
    action_events = [e for e in events if e.get("event_type") == "action_received"]
    if action_events:
        print(f"\n🎯 Ações Recebidas: {len(action_events)}")
        total_motors = sum(e.get("motor_count", 0) for e in action_events)
        print(f"  Total de motores: {total_motors}")
        print(f"  Média de motores/ação: {total_motors / len(action_events):.1f}")

    # Motor commands
    motor_events = [e for e in events if e.get("event_type") == "motor_cmd"]
    if motor_events:
        print(f"\n⚙️ Comandos de Motor: {len(motor_events)}")
        errors = [e.get("error", 0) for e in motor_events]
        if errors:
            print(f"  Erro médio (q_target - q_actual): {sum(errors) / len(errors):.4f}")
            print(f"  Erro máximo: {max(errors):.4f}")

    # Bridge states
    bridge_events = [e for e in events if e.get("event_type") == "bridge_state"]
    if bridge_events:
        print(f"\n🔌 Estados do Bridge: {len(bridge_events)}")

    # Últimas ações
    if action_events:
        print(f"\n📝 Últimas 3 ações:")
        for i, event in enumerate(action_events[-3:], 1):
            motors = event.get("body_motors", [])
            if motors:
                m = motors[0]
                print(f"  {i}. Motor 0: q={m.get('q'):.3f}, kp={m.get('kp', 0):.1f}")

    print(f"\n✅ Log: {log_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python analyze_action_log.py <log_file>")
        print("Exemplo: python analyze_action_log.py /tmp/action_log_20260523_120000.jsonl")
        sys.exit(1)

    analyze_log(sys.argv[1])
