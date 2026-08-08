"""Registro central das políticas customizadas do Prometheus-VLA.

Importar este pacote ativa todos os `@PreTrainedConfig.register_subclass(...)`,
o que faz o `make_policy` do LeRobot reconhecer os tipos abaixo pelo nome:

    actdepth      → ACT + PointNet/PointTransformer + tato        (sem linguagem)
    pi05depth     → PI05 flow matching + depth + tato             (multi-tarefa)
    openvladepth  → OpenVLA 7B + depth + tato, head OFT paralelo  (multi-tarefa)
"""

from .act_depth.configuration_act import ACTConfig as ACTConfig
from .openvla_depth.configuration_openvla import OPENVLADEPTHConfig as OPENVLADEPTHConfig
from .pi0_depth.configuration_pi05 import PI05DEPTHConfig as PI05DEPTHConfig
