#!/usr/bin/env python
"""
Training Entry Point V2 for Unitree G1 with Dex3 Hands.
Activa o treinamento do ACT-D (Depth-Augmented) com PointNet e Fusão Tátil.
"""

import sys
import os

# 1. Garante que o Python enxergue a pasta 'train'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 2. Register custom G1 modules into the global registry
try:
    import robot.unitree_g1
except ImportError as e:
    print(f"\n[IMPORT ERROR]: Failed to load custom G1 modules: {e}")
    sys.exit(1)

# 3. Importa o motor de treinamento customizado
try:
    # A injeção da PointNet ocorre lá dentro do run_train.py!
    from train.run_train import main as run_train_main
except ImportError as e:
    print(f"\n[IMPORT ERROR]: Não encontrei o motor customizado: {e}")
    print("Verifique se a pasta 'train' existe e possui o arquivo 'run_train.py'.")
    sys.exit(1)

def display_help():
    """Prints a professional usage guide for training."""
    print("\n" + "="*70)
    print("LEROBOT TRAINING INTERFACE - ACT-D (CUSTOM ENGINE)")
    print("="*70)
    print("\nUSAGE:")
    print("  python init_lerobot_train_v2.py --config_path=<PATH_TO_YAML_CONFIG> [OPTIONS]")
    print("\nARGUMENTS:")
    print("  --config_path     Path to the training YAML configuration file.")
    print("  -h, --help        Show this help message and exit.")
    
    # --- LISTAGEM DINÂMICA DOS ARQUIVOS YAML ---
    print("\nAVAILABLE CONFIG FILES:")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, "config", "train")
    
    if os.path.exists(config_dir) and os.path.isdir(config_dir):
        yaml_files = [f for f in os.listdir(config_dir) if f.endswith(('.yaml', '.yml'))]
        if yaml_files:
            for yml in sorted(yaml_files):
                print(f"  - config/train/{yml}")
        else:
            print("  (No .yaml files found in 'config/train/' folder)")
    else:
        print("  (Folder 'config/train/' not found)")

    print("\nEXAMPLE:")
    print("  python init_lerobot_train_v2.py --config_path=config/train/train_push_cup.yaml\n")
    print("="*70 + "\n")

if __name__ == "__main__":
    cli_args = sys.argv[:]
    
    # 1. Help Check
    if any(flag in cli_args for flag in ["-h", "--help", "-help"]):
        display_help()
        sys.exit(0)

    # 2. Check for --config_path
    has_config_path = any("--config_path" in arg for arg in cli_args)
    if not has_config_path:
        print("\n[CRITICAL ERROR]: Mandatory '--config_path' argument is missing.")
        print("Use '-h' or '--help' for usage instructions.")
        sys.exit(1)

    print(f"[INFO]: Initializing Custom ACT-D Training Pipeline...")

    # 3. Inicia o processo de treinamento passando os argumentos
    try:
        sys.exit(run_train_main())
    except KeyboardInterrupt:
        print("\n[SYSTEM]: Training session terminated by user.")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\n[RUNTIME ERROR]:")
        traceback.print_exc()
        sys.exit(1)