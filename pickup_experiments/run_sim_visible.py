#!/usr/bin/env python3
"""Lanca o simulador com JANELA VISIVEL e sem publicacao de imagens (carga menor)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "unitree-g1-mujoco"))

from run_sim import main

if __name__ == "__main__":
    # publish_images=False evita o subprocesso de cameras (menos carga/risco)
    main(publish_images=False)
