#!/usr/bin/env python
"""
Entry point de gravação com teleop por câmera RealSense D435i + MediaPipe.

É um launcher FINO: reaproveita TODA a maquinaria do init_lerobot_record.py
(auto-env conda 'g1', monkeypatches de dataset/save/tátil/depth, terminal enxuto,
proxy ZMQ da câmera do robô, OmniView LIVE, async-save, etc.) — só troca a fonte
de pose do Quest para a câmera, via config_path.

A teleop em si vive em teleop/realsense_mediapipe_arm.py (classe
'realsense_mediapipe_arm'); o YAML config/record/record_realsense_mediapipe.yaml
aponta teleop.type pra ela. Como o teleop é selecionado por config, NADA do
init_lerobot_record.py precisou mudar.

Uso:
  python lerobot-ext/realsense_mediapipe_record.py --sim          # simulador
  python lerobot-ext/realsense_mediapipe_record.py                # robô real

Controles (teclado — ver realsense_mediapipe_arm.py):
  espaço=destrava/trava  f=fecha mão  j=pinça  s=salva  d=descarta  q=encerra

Aceita os mesmos flags do init_lerobot_record.py (--sim, --dataset.root=...,
--debug, --dry-preflight, etc.). Se nenhum --config_path for passado, usa o
record_realsense_mediapipe.yaml por padrão.
"""
import os
import sys
import runpy
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _HERE / "config" / "record" / "record_realsense_mediapipe.yaml"
_INIT_RECORD = _HERE / "init_lerobot_record.py"

# Marca a fonte de entrada (futuro: gatear no init os passos só-Quest no robô real).
os.environ.setdefault("G1_INPUT_SOURCE", "mediapipe")

# Injeta o config padrão se o usuário não passou um.
if "--config_path" not in sys.argv:
    sys.argv += ["--config_path", str(_DEFAULT_CONFIG)]

# Roda o init_lerobot_record.py como se fosse o __main__ (mantém todo o boot dele).
runpy.run_path(str(_INIT_RECORD), run_name="__main__")
