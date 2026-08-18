#!/usr/bin/env python
"""Treino do FastWAM-D.

Ao contrário do `policies/act_depth/run_train.py`, este não é um fork do laço
de treino do LeRobot — ele só REGISTRA a política e chama o `lerobot-train` de
fábrica. O motivo é deliberado:

  - o laço forkeado do ACT-D existe por causa de coisas que só ele tem
    (curriculum de posição neutra, gate de incerteza, freeze seletivo por
    encoder). O FastWAM-D não precisa de nada disso;
  - um modelo de 5B depende de `accelerate`, gradient checkpointing e
    checkpoint/retomada exatos por amostra, que o laço de fábrica já resolve e
    um fork precisaria perseguir a cada rebase;
  - validação em episódios separados existe nativamente na 0.6.1 via
    `dataset.eval_split` + `eval_steps` — não é preciso o `val_dataset` do fork.

    cd lerobot-ext
    python -m policies.fastwam_depth.run_train \\
        --config_path=config/train/fastwamdepth_white_cup_on_dripper.yaml
"""

from __future__ import annotations

import os
import sys

# O `cwd` precisa estar no path para o `make_policy` reimportar
# `policies.fastwam_depth.modeling_fastwam_depth` pela convenção de nomes
# (ver `lerobot/policies/factory.py::_get_policy_cls_from_policy_name`).
sys.path.append(os.getcwd())

# Import com efeito colateral, e é o ponto de existir deste arquivo: o
# decorador `@PreTrainedConfig.register_subclass("fastwamdepth")` só roda
# quando o módulo é importado. Sem isto o draccus rejeita
# `policy.type: fastwamdepth` como tipo desconhecido.
from policies.fastwam_depth.configuration_fastwam_depth import FastWAMDepthConfig  # noqa: F401
from lerobot.scripts.lerobot_train import train


def main() -> None:
    train()


if __name__ == "__main__":
    main()
