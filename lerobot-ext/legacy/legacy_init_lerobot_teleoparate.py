"""
Teleoperation Entry Point for Unitree G1 with Dex3 Hands.
Ensures hardware and simulation modules are registered within the LeRobot framework.
"""

import sys
import os

# 1. Register custom G1 modules into the global registry
try:
    import robot.unitree_g1
    import teleop.unitree_g1
    import teleop
except ImportError as e:
    print(f"\n[IMPORT ERROR]: Failed to load custom G1 modules: {e}")
    sys.exit(1)

from lerobot.scripts.lerobot_teleoperate import main

def display_help():
    """Prints a professional usage guide and CLI documentation."""
    print("\n" + "="*70)
    print("LEROBOT CUSTOM TELEOPERATION INTERFACE - UNITREE G1")
    print("="*70)
    print("\nUSAGE:")
    print("  python init_lerobot_teleoparate.py --config_path=<PATH_TO_YAML> [OPTIONS]")
    print("\nARGUMENTS:")
    print("  --config_path     Path to the YAML configuration file.")
    print("  --sim             Short flag to force simulation mode.")
    print("  --simulation=true Force simulation mode.")
    print("  -h, --help        Show this help message and exit.")
    
    # --- LISTAGEM DINÂMICA DOS ARQUIVOS YAML ---
    print("\nAVAILABLE CONFIG FILES:")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, "config")
    
    if os.path.exists(config_dir) and os.path.isdir(config_dir):
        # Filtra apenas os arquivos .yaml ou .yml
        yaml_files = [f for f in os.listdir(config_dir) if f.endswith(('.yaml', '.yml'))]
        if yaml_files:
            for yml in sorted(yaml_files):
                print(f"  - config/{yml}")
        else:
            print("  (No .yaml files found in 'config/' folder)")
    else:
        print("  (Folder 'config/' not found)")
    # -------------------------------------------

    print("\nEXAMPLE:")
    print("  python init_lerobot_teleoparate.py --config_path=config/teleop.yaml --sim\n")
    print("="*70 + "\n")

if __name__ == "__main__":
    # Capturamos os argumentos originais
    cli_args = sys.argv[:]
    
    # 1. Check for help flags
    if any(flag in cli_args for flag in ["-h", "--help", "-help"]):
        display_help()
        sys.exit(0)

    # 2. Resilient check for --config_path
    if not any("--config_path" in arg for arg in cli_args):
        print("\n[CRITICAL ERROR]: Mandatory '--config_path' argument is missing.")
        sys.exit(1)

    # 3. LÓGICA DE INTERCEPTAÇÃO E MODIFICAÇÃO
    # Aqui o Python "edita" o que o LeRobot vai entender do YAML
    
    force_sim = False
    # Verificamos se o usuário pediu simulação de qualquer forma (--sim ou --simulation=true)
    for arg in cli_args:
        if arg in ["--sim", "--simulation=true"]:
            force_sim = True
            # Removemos a nossa flag customizada para não dar erro no parser do LeRobot
            if arg in sys.argv:
                sys.argv.remove(arg)

    if force_sim:
        # Injetamos a modificação com os dois traços (Padrão do draccus/argparse)
        sys.argv.append("--robot.is_simulation=true")
        sys.argv.append("--teleop.is_simulation=true")
        print("[INFO]: Overriding YAML config: robot.is_simulation set to TRUE")
    else:
        # Modo robô real com os dois traços
        sys.argv.append("--robot.is_simulation=false")
        sys.argv.append("--teleop.is_simulation=false")
        print("[INFO]: Using Real Robot mode (robot.is_simulation=false)")

    # 4. Launch teleoperation loop
    try:
        # O main() do LeRobot agora recebe a lista sys.argv já modificada pelo Python
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[SYSTEM]: Session terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[RUNTIME ERROR]: {e}")
        sys.exit(1)