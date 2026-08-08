#!/usr/bin/env python
"""
OpenVLA-Depth — VLA multi-tarefa com profundidade para o Unitree G1 + Dex3.

Registra o tipo `openvladepth` no LeRobot. Uso:

    policy:
      type: openvladepth
      pretrained_backbone: openvla/openvla-7b
"""

from .configuration_openvla import OPENVLADEPTHConfig
from .modeling_openvla import OPENVLADEPTHPolicy
from .processor_openvla import make_openvladepth_pre_post_processors

__all__ = [
    "OPENVLADEPTHConfig",
    "OPENVLADEPTHPolicy",
    "make_openvladepth_pre_post_processors",
]
