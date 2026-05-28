"""
Interactive Dataset Visualization Entry Point for Unitree G1.
Fix: Supports --root=path and --repo=path formats.
"""

import sys
import os
import subprocess
import pandas as pd

# =========================================================================
# MODO 2: O PROCESSO TRABALHADOR
# =========================================================================
if "--internal-run" in sys.argv:
    sys.argv.remove("--internal-run")
    try:
        import robot.unitree_g1
        import teleop.unitree_g1
    except ImportError:
        pass
    from lerobot.scripts.lerobot_dataset_viz import main
    sys.exit(main())

# =========================================================================
# MODO 1: O GERENCIADOR INTERATIVO
# =========================================================================

def check_episode_exists(root_dir, ep_index):
    """Verifica a existência do episódio no arquivo de metadados."""
    if not root_dir or not os.path.exists(root_dir):
        return False, 0
    
    episodes_file = os.path.join(root_dir, "episodes.parquet")
    if os.path.exists(episodes_file):
        try:
            df = pd.read_parquet(episodes_file)
            total = len(df)
            return int(ep_index) < total, total
        except Exception:
            pass
    return True, "?"

if __name__ == "__main__":
    cli_args = sys.argv[:]
    
    # 1. Defaults
    repo_id = "Mrwlker/teste3"
    root_dir = "meu_dataset/teste3"
    episode_idx = "0"
    display_compressed = True

    # 2. Parser robusto (Aceita --root=X e --root X)
    for i, arg in enumerate(cli_args):
        # ROOT
        if arg.startswith("--root="):
            root_dir = arg.split("=")[1]
        elif arg == "--root" and i + 1 < len(cli_args):
            root_dir = cli_args[i+1]
        
        # REPO
        if arg.startswith("--repo="):
            repo_id = arg.split("=")[1]
        elif arg == "--repo" and i + 1 < len(cli_args):
            repo_id = cli_args[i+1]

        # EPISODE
        if arg.startswith("--ep="):
            episode_idx = arg.split("=")[1]
        elif arg == "--ep" and i + 1 < len(cli_args):
            episode_idx = cli_args[i+1]

    # Ajuste inteligente: Se você mudou a pasta root mas não o repo, 
    # tentamos adivinhar o nome do repo pela pasta para evitar erro de mismatch no LeRobot
    if "teste2" not in root_dir and repo_id == "Mrwlker/teste2":
        repo_id = f"Mrwlker/{os.path.basename(root_dir.strip('/'))}"

    current_process = None

    def launch_viz(ep):
        global current_process
        if current_process is not None:
            current_process.terminate()
            current_process.wait()

        cmd = [sys.executable, __file__, "--internal-run", "--repo-id", repo_id, "--episode-index", str(ep)]
        if root_dir:
            cmd.extend(["--root", root_dir])
        if display_compressed:
            cmd.append("--display-compressed-images")

        # Abre sem silenciar o erro para você ver o que acontece
        current_process = subprocess.Popen(cmd)

    print("\n" + "="*60)
    print("📺 VISUALIZADOR INTERATIVO DO G1")
    print("="*60)
    print(f"Dataset : {repo_id}")
    print(f"Pasta   : {root_dir}")
    
    existe, total = check_episode_exists(root_dir, episode_idx)
    launch_viz(episode_idx)

    try:
        while True:
            val = input(f"\n👉 Ep. atual: {episode_idx} (Total: {total}). Próximo ep (ou 'q'): ").strip()
            if val.lower() in ['q', 'sair']: break
            if not val.isdigit(): continue
            
            existe, total = check_episode_exists(root_dir, val)
            if existe or total == "?":
                episode_idx = val
                print(f"🔄 Trocando para o Episódio {episode_idx}...")
                launch_viz(episode_idx)
            else:
                print(f"❌ Erro: Episódio {val} não existe nesta pasta!")
    except KeyboardInterrupt:
        pass
    finally:
        if current_process: current_process.terminate()
        sys.exit(0)