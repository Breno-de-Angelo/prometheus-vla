#!/usr/bin/env python
"""
OpenVLA-Depth — VLA multi-tarefa com profundidade para o Unitree G1 + Dex3.

Registra o tipo `openvladepth` no LeRobot. Uso:

    policy:
      type: openvladepth
      pretrained_backbone: openvla/openvla-7b

Só o `configuration_openvla` é importado na hora — é ele que carrega o
`@PreTrainedConfig.register_subclass`, que é tudo que o LeRobot precisa para
reconhecer o tipo. `modeling_openvla` e `processor_openvla` são resolvidos sob
demanda (PEP 562), por dois motivos:

  1. `import policies` deixa de arrastar torch/timm/peft só para registrar nomes.
  2. Evita o RuntimeWarning de `python -m policies.openvla_depth.backbone`, que
     acontece quando o `__init__` já colocou `backbone` em `sys.modules` antes de
     o `-m` tentar executá-lo.

O `make_policy` do LeRobot importa `modeling_openvla` dinamicamente
(`_get_policy_cls_from_policy_name`), então nada disso muda o comportamento.
"""

from .configuration_openvla import OPENVLADEPTHConfig

__all__ = [
    "OPENVLADEPTHConfig",
    "OPENVLADEPTHPolicy",
    "make_openvladepth_pre_post_processors",
]

_LAZY = {
    "OPENVLADEPTHPolicy": ".modeling_openvla",
    "make_openvladepth_pre_post_processors": ".processor_openvla",
}


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
