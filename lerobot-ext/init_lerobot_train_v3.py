#!/usr/bin/env python
"""
Training Entry Point V3 (Arquitetura Nativa com Registry)
Carrega o YAML, lê o __init__.py e chama o treinamento.
"""

import sys
import os

# 1. Garante que o Python enxergue as pastas locais
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 2. Carrega os registros nativos da pasta policies
try:
    import policies  # Isso ativa os decoradores @PreTrainedConfig.register_subclass
except ImportError as e:
    print(f"\n[ERRO DE IMPORTAÇÃO]: Falha ao ler o seu __init__.py: {e}")
    sys.exit(1)

# =====================================================================
# 3. MOTOR DE TREINAMENTO (CORRIGIDO PARA O NOVO LOCAL)
# =====================================================================
try:
    # Mudamos de "train.run_train" para "policies.act_depth.run_train"
    from policies.act_depth.run_train import main as run_train_main
except ImportError as e:
    print(f"\n[ERRO]: Motor de treino (run_train.py) não encontrado no novo local: {e}")
    sys.exit(1)

def display_help():
    print("\n" + "="*70)
    print("LEROBOT TRAINING INTERFACE - NATIVO (V3)")
    print("="*70)
    print("USO:")
    print("  python init_lerobot_train_v3.py --config_path=<CAMINHO_PARA_O_YAML>\n")

if __name__ == "__main__":
    if any(flag in sys.argv for flag in ["-h", "--help"]) or len(sys.argv) < 2:
        display_help()
        sys.exit(0 if "-h" in sys.argv else 1)

    print("[INFO]: Iniciando LeRobot Train Pipeline via motor ACT-D...")
    
    try:
        sys.exit(run_train_main())
    except KeyboardInterrupt:
        print("\n[SISTEMA]: Treinamento cancelado pelo usuário.")
        sys.exit(0)