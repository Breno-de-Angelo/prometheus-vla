#!/usr/bin/env python3
"""Ponte de telemetria ao vivo entre o processo de gravação e o dashboard web.

O `init_lerobot_record_v2.py` já tem, a cada frame de controle:
  • o estado das juntas (obs do robô: braços por enum kXxx.q, mãos por nome URDF .q),
  • a última action enviada (`_last_action_for_log`),
  • a pressão tátil das duas mãos (`buffer_pressao`, 108 valores/mão).

As câmeras (RGB + Depth) já saem num PUB ZMQ em :5555 (consumido pelo
`view_cam_live.py`). O que FALTAVA pro dashboard estilo OmniView era um canal com
state/action/tátil. Este módulo abre um segundo PUB ZMQ (:5557, JSON) e publica,
por frame, um pacote compacto agrupado por braço/mão — exatamente o que o webapp
(`tools/live_webapp/`) precisa pra montar rampas, trajetória 3D (FK) e o mapa tátil.

Tudo aqui é best-effort e NÃO pode quebrar a gravação: qualquer exceção é engolida.
"""

from __future__ import annotations

import json
import logging
import threading

import numpy as np
import zmq

logger = logging.getLogger(__name__)

TELEMETRY_PORT = 5557

# --- Inferencia right8: deriva o squeeze (0..1) dos dedos reconstruidos ---
# A inferencia right8 NAO publica *_grasp_squeeze.q (chave do teleop); reconstroi os
# dedos via hand_q = squeeze x RIGHT_TARGET. Sem isso o medidor de grasp do OmniView
# fica cravado em 0%. Aqui derivamos o squeeze de volta dos dedos (junta/alvo, media; clip 0..1).
_RIGHT_GRASP_JOINTS = [("right_hand_index_1_joint.q", 1.74),
                       ("right_hand_thumb_2_joint.q", -1.74),
                       ("right_hand_middle_1_joint.q", 1.74)]
_LEFT_GRASP_JOINTS = [("left_hand_index_1_joint.q", 1.74),
                      ("left_hand_thumb_2_joint.q", -1.74),
                      ("left_hand_middle_1_joint.q", 1.74)]

# ---- ordem das juntas (espelha o packing do OmniView: braço esq / dir / mão esq / dir) ----
# Braços: chaves de obs/action usam o NOME DO ENUM (G1_29_JointArmIndex), ex. "kRightElbow.q".
# Mãos: chaves usam o NOME URDF (RIGHT_HAND_JOINT_NAMES), ex. "right_hand_thumb_0_joint.q".
try:
    from robot.unitree_g1.g1_utils import (
        G1_29_JointArmIndex,
        LEFT_HAND_JOINT_NAMES,
        RIGHT_HAND_JOINT_NAMES,
    )

    _ARM = [m.name for m in G1_29_JointArmIndex]          # 14: 7 esq + 7 dir
    LEFT_ARM_KEYS = [f"{n}.q" for n in _ARM[:7]]
    RIGHT_ARM_KEYS = [f"{n}.q" for n in _ARM[7:]]
    LEFT_HAND_KEYS = [f"{n}.q" for n in LEFT_HAND_JOINT_NAMES]
    RIGHT_HAND_KEYS = [f"{n}.q" for n in RIGHT_HAND_JOINT_NAMES]
except Exception as e:  # pragma: no cover - só falha fora do ambiente do robô
    logger.warning("[live_bridge] não consegui importar g1_utils (%s); telemetria desativada", e)
    LEFT_ARM_KEYS = RIGHT_ARM_KEYS = LEFT_HAND_KEYS = RIGHT_HAND_KEYS = []


def _get(src, key: str) -> float:
    """Lê src[key]/src.get(key) de um obs/action dict-like, tolerante a tipos."""
    try:
        if hasattr(src, "get"):
            v = src.get(key, 0.0)
        else:
            v = src[key]
        return float(v)
    except Exception:
        return 0.0



def _derive_squeeze(d, joints):
    vals = []
    for k, tgt in joints:
        present = (k in d) if hasattr(d, "__contains__") else True
        if present and abs(tgt) > 1e-6:
            vals.append(_get(d, k) / tgt)
    if not vals:
        return 0.0
    return max(0.0, min(1.0, sum(vals) / len(vals)))


def _grasp_signal(d, squeeze_key, derive_joints):
    """teleop publica *_grasp_squeeze.q; inferencia right8 nao -> deriva dos dedos."""
    try:
        if squeeze_key in d:
            return _get(d, squeeze_key)
    except TypeError:
        pass
    return _derive_squeeze(d, derive_joints)

def _group(src, keys) -> list:
    return [_get(src, k) for k in keys]


def _press_list(p):
    a = np.asarray(p, dtype=np.float32).ravel()
    if a.size < 108:
        a = np.pad(a, (0, 108 - a.size))
    return [round(float(x), 2) for x in a[:108]]


class TelemetryPublisher:
    """PUB ZMQ de telemetria com custo ~nulo no loop de controle.

    `publish()` (chamado no get_observation, no caminho crítico) SÓ extrai os
    valores (leitura barata de ~42 floats + pressão) e deposita o pacote num
    mailbox de slot único. Uma THREAD de fundo faz o `json.dumps` + o envio ZMQ
    — ou seja, serialização e socket NUNCA acontecem no loop de controle. Se a
    thread não der conta, o slot é simplesmente sobrescrito (drop do frame
    antigo), então o produtor jamais bloqueia.
    """

    def __init__(self, port: int = TELEMETRY_PORT, host: str = "127.0.0.1"):
        self._ok = bool(RIGHT_ARM_KEYS)
        self.socket = None
        self._slot = None
        self._slot_extra = None   # pacotes avulsos (ex.: attn_jpg da inferência)
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        if not self._ok:
            return
        try:
            self._ctx = zmq.Context.instance()
            self.socket = self._ctx.socket(zmq.PUB)
            self.socket.setsockopt(zmq.SNDHWM, 4)          # não acumula backlog
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.bind(f"tcp://{host}:{port}")
            self._thread = threading.Thread(target=self._sender_loop, daemon=True)
            self._thread.start()
            logger.info("[live_bridge] telemetria publicando em tcp://%s:%d", host, port)
        except Exception as e:
            logger.warning("[live_bridge] falha ao abrir PUB de telemetria: %s", e)
            self.socket = None

    def _sender_loop(self):
        while not self._stop.is_set():
            if not self._event.wait(timeout=0.5):
                continue
            self._event.clear()
            for attr in ("_slot", "_slot_extra"):
                pkt = getattr(self, attr)     # snapshot (atribuição é atômica via GIL)
                if pkt is None:
                    continue
                setattr(self, attr, None)
                try:
                    self.socket.send_string(json.dumps(pkt), flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
                except Exception as e:
                    logger.debug("[live_bridge] send falhou: %s", e)

    def publish(self, obs, action, pressure_left, pressure_right, episode=0, frame=0, t=0.0,
                robot_phase="unlocked"):
        """No caminho crítico: só extrai e enfileira. Nunca levanta nem bloqueia.

        robot_phase: 'softstart' (rampa inicial) · 'locked' (bloqueado/congelado —
        aperte X p/ liberar) · 'unlocked' (desbloqueado/teleoperando).
        """
        if self.socket is None:
            return
        try:
            self._slot = {
                "type": "tele",
                "t": float(t),
                "episode": int(episode),
                "frame": int(frame),
                "robot": {"phase": robot_phase},
                "state": {
                    "leftArm": _group(obs, LEFT_ARM_KEYS),
                    "rightArm": _group(obs, RIGHT_ARM_KEYS),
                    "leftHand": _group(obs, LEFT_HAND_KEYS),
                    "rightHand": _group(obs, RIGHT_HAND_KEYS),
                },
                "action": {
                    "leftArm": _group(action, LEFT_ARM_KEYS),
                    "rightArm": _group(action, RIGHT_ARM_KEYS),
                    "leftHand": _group(action, LEFT_HAND_KEYS),
                    "rightHand": _group(action, RIGHT_HAND_KEYS),
                },
                "pressure": {"left": _press_list(pressure_left), "right": _press_list(pressure_right)},
                # sinais de grasp do controle (0=solto, 1=fechado). squeeze=fecha a mão
                # toda (grasp); trigger=pinça fina do dedo.
                "grasp": {
                    "left": _grasp_signal(action, "left_grasp_squeeze.q", _LEFT_GRASP_JOINTS),
                    "right": _grasp_signal(action, "right_grasp_squeeze.q", _RIGHT_GRASP_JOINTS),
                    "leftTrigger": _get(action, "left_grasp_trigger.q"),
                    "rightTrigger": _get(action, "right_grasp_trigger.q"),
                },
            }
            self._event.set()
        except Exception as e:  # nunca derruba a gravação
            logger.debug("[live_bridge] publish falhou: %s", e)

    def publish_extra(self, pkt: dict):
        """Pacote JSON avulso no mesmo PUB (ex.: {'attn_jpg': dataURL} da inferência).
        Mesmo contrato do publish: slot único, nunca bloqueia nem levanta."""
        if self.socket is None:
            return
        try:
            self._slot_extra = dict(pkt)
            self._event.set()
        except Exception as e:
            logger.debug("[live_bridge] publish_extra falhou: %s", e)

    def close(self):
        self._stop.set()
        self._event.set()
        if self.socket is not None:
            try:
                self.socket.close(0)
            except Exception:
                pass
            self.socket = None
