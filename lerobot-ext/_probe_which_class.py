#!/usr/bin/env python3
"""Probe: descobrir QUAL classe UnitreeG1Dex3 o pipeline instancia de fato,
imitando a ordem de import do init_lerobot_record_v2.py, e se os monkeypatches
do init (aplicados em robot.unitree_g1 = lerobot-ext) atingem essa instância."""
import inspect

# 1) Mesma ordem do entrypoint: lerobot-ext primeiro
import robot.unitree_g1
import teleop.unitree_g1
from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3 as EXT_Dex3
print("EXT  UnitreeG1Dex3:", EXT_Dex3.__module__, inspect.getfile(EXT_Dex3))

# 2) lerobot/src
from lerobot.robots.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3 as SRC_Dex3
print("SRC  UnitreeG1Dex3:", SRC_Dex3.__module__, inspect.getfile(SRC_Dex3))
print("EXT is SRC?", EXT_Dex3 is SRC_Dex3)

# 3) Registro draccus para o type "unitree_g1_dex3"
try:
    from lerobot.robots.config import RobotConfig
    reg = getattr(RobotConfig, "_registered_subclasses", None) or \
          getattr(RobotConfig, "_choice_registry", None) or {}
    print("\nRegistro RobotConfig:")
    for k, v in (reg.items() if hasattr(reg, "items") else []):
        if "g1" in str(k).lower() or "g1" in str(v).lower():
            print(f"   {k!r} -> {v} ({getattr(v,'__module__','?')})")
except Exception as e:
    print("registro:", repr(e))

# 4) O que make_robot_from_config REALMENTE instancia
from lerobot.robots.utils import make_robot_from_config
# Constrói o config como o entrypoint (--robot.type=unitree_g1_dex3 --robot.is_simulation=true)
CfgClass = None
for C in (EXT_Dex3, SRC_Dex3):
    cc = getattr(C, "config_class", None)
    if cc:
        CfgClass = cc
        print(f"\nconfig_class de {C.__module__} = {cc.__module__}.{cc.__name__}")

# Usa o config_class de cada classe e vê o que make_robot devolve
for tag, C in (("EXT", EXT_Dex3), ("SRC", SRC_Dex3)):
    try:
        cfg = C.config_class(is_simulation=True)
        robot = make_robot_from_config(cfg)
        print(f"\n[cfg={tag}] make_robot_from_config -> {type(robot).__module__} "
              f"({inspect.getfile(type(robot))})")
        print(f"          type(robot) is SRC? {type(robot) is SRC_Dex3} | is EXT? {type(robot) is EXT_Dex3}")
        sa = type(robot).send_action
        print(f"          send_action qualname={sa.__qualname__} module={sa.__module__}")
    except Exception as e:
        import traceback
        print(f"[cfg={tag}] erro:", repr(e))
        traceback.print_exc()
