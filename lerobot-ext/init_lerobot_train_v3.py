#!/usr/bin/env python
"""
Training Entry Point V3 — Universal
Funciona com qualquer type: (actdepth, pi05depth, ou qualquer outro registrado)
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Registra TODAS as políticas de uma vez
try:
    import policies  # ativa todos os @register_subclass do __init__.py
except ImportError as e:
    print(f"[ERRO]: Falha ao registrar políticas: {e}")
    sys.exit(1)

# Lê o type do YAML para decidir qual motor usar
import yaml

def get_policy_type(config_path: str) -> str:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("policy", {}).get("type", "")

def display_help():
    print("\n" + "="*70)
    print("LEROBOT TRAINING — UNIVERSAL (V3)")
    print("="*70)
    print("USO:")
    print("  python init_lerobot_train_v3.py --config_path=<YAML>\n")
    print("Tipos suportados:")
    print("  actdepth      ACT + depth 3D + tato            (sem linguagem)")
    print("  pi05depth     PI05 flow matching + depth       (multi-tarefa por texto)")
    print("  openvladepth  OpenVLA 7B + depth, head OFT     (multi-tarefa por texto)")
    print("="*70 + "\n")

if __name__ == "__main__":
    cli_args = sys.argv[:]

    if any(f in cli_args for f in ["-h", "--help"]):
        display_help()
        sys.exit(0)

    config_arg = next((a for a in cli_args if "--config_path" in a), None)
    if not config_arg:
        print("[ERRO]: --config_path obrigatório. Use -h para ajuda.")
        sys.exit(1)

    # Extrai o caminho do YAML
    config_path = config_arg.split("=", 1)[-1]

    # Descobre o tipo e carrega o motor certo
    policy_type = get_policy_type(config_path)
    print(f"[INFO]: Detectado policy.type = '{policy_type}'")

    if policy_type == "actdepth":
        from policies.act_depth.run_train import main as run_train_main
        print("[INFO]: Usando motor ACT-D...")

    elif policy_type == "pi05depth":
        from policies.pi0_depth.run_train import main as run_train_main
        print("[INFO]: Usando motor PI05-Depth...")

    elif policy_type == "openvladepth":
        from policies.openvla_depth.run_train import main as run_train_main
        print("[INFO]: Usando motor OpenVLA-Depth (head OFT paralelo)...")

    else:
        # Fallback: tenta o motor genérico do LeRobot diretamente
        print(f"[AVISO]: Tipo '{policy_type}' sem motor dedicado — usando motor genérico LeRobot.")
        from lerobot.scripts.lerobot_train import main as run_train_main

    try:
        sys.exit(run_train_main())
    except KeyboardInterrupt:
        print("\n[SISTEMA]: Treinamento encerrado pelo usuário.")
        sys.exit(0)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)