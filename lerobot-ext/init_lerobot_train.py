#!/usr/bin/env python
"""
Training Entry Point V3 (Arquitetura Nativa com Registry)
Carrega o YAML, lê o __init__.py (que registra o actdepth) e chama o treinamento.
"""

import sys
import os

# 1. Garante que o Python enxergue as pastas locais
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# =====================================================================
# 2. O PULO DO GATO: A MÁGICA DO __init__.py
# Ao fazer este import, o Python lê o seu __init__.py.
# O __init__.py faz o import do ACTConfig, que por sua vez ativa o
# decorador @PreTrainedConfig.register_subclass("actdepth").
# =====================================================================
try:
    # IMPORTANTE: Se o seu __init__.py estiver dentro de uma pasta chamada
    # 'policies', troque a linha abaixo para 'import policies'.
    # Se estiver dentro da pasta 'train', use 'import train'.
    import policies  # <- Ajuste para o nome da pasta do seu __init__.py
    #print("[INFO]: Registro nativo 'actdepth' carregado com sucesso via __init__.py!")
except ImportError as e:
    print(f"\n[ERRO DE IMPORTAÇÃO]: Falha ao ler o seu __init__.py: {e}")
    sys.exit(1)

# =====================================================================
# 3. MOTOR DE TREINAMENTO OFICIAL
# Não injetamos nada, apenas chamamos o fluxo padrão do seu run_train.
# =====================================================================
try:
    from train.run_train import main as run_train_main
except ImportError as e:
    print(f"\n[ERRO]: Motor de treino (run_train.py) não encontrado: {e}")
    sys.exit(1)

def display_help():
    print("\n" + "="*70)
    print("LEROBOT TRAINING INTERFACE - NATIVO (V3)")
    print("="*70)
    print("USO:")
    print("  python init_lerobot_train_v3.py --config_path=<CAMINHO_PARA_O_YAML>\n")

if __name__ == "__main__":
    # Verifica se o usuário pediu ajuda ou esqueceu o config
    if any(flag in sys.argv for flag in ["-h", "--help"]) or len(sys.argv) < 2:
        display_help()
        sys.exit(0 if "-h" in sys.argv else 1)

    print("[INFO]: Iniciando LeRobot Train Pipeline...")
    
    # O motor 'run_train_main' vai ler o sys.argv nativamente, achar o seu YAML,
    # ler "type: actdepth" e procurar no registro. Como o __init__.py já 
    # cadastrou ele, a mágica acontece sozinha!
    try:
        sys.exit(run_train_main())
    except KeyboardInterrupt:
        print("\n[SISTEMA]: Treinamento cancelado pelo usuário.")
        sys.exit(0)