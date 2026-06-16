#!/usr/bin/env python3

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DDS-to-ZMQ bridge server for Unitree G1 robot with Dex3 hands.

This server runs on the robot and forwards:
- Robot state (LowState) from DDS to ZMQ (for remote clients)
- Robot commands (LowCmd) from ZMQ to DDS (from remote clients)
- Dex3 hand state from DDS to ZMQ
- Dex3 hand commands from ZMQ to DDS

Uses JSON for secure serialization instead of pickle.
"""

import base64
import contextlib
import json
import threading
import time
from typing import Any

import zmq
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.utils.crc import CRC

# DDS topic names follow Unitree SDK naming conventions
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"  # action to robot
kTopicLowState = "rt/lowstate"  # observation from robot

# Dex3 hand topics
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"

# ZMQ ports
LOWCMD_PORT = 6000
LOWSTATE_PORT = 6001
HANDSTATE_PORT = 6002
HANDCMD_PORT = 6003

NUM_MOTORS = 35
NUM_HAND_MOTORS = 7

# =============================================================================
# RAMPA ONBOARD (jeito Unitree). O cliente (laptop) manda o ALVO a ~30Hz; aqui no
# robô uma thread de alta frequência interpola até o alvo com clip de velocidade e
# publica no DDS. Isso elimina o stair-stepping/tremor SEM depender de mandar 250Hz
# pela ponte JSON+ZMQ (que o Jetson não aguenta — json.loads vira gargalo e o
# CONFLATE descarta a rampa). Aqui o json.loads roda a 30Hz (receiver) e só a
# cópia-struct + CRC + DDS Write (barato) roda a 250Hz. Réplica do G1_29_ArmController.
import os as _os_top

# Juntas do braço no LowCmd de 29 DoF: 15-21 esquerdo, 22-28 direito.
ARM_JOINT_IDS = tuple(range(15, 29))
# Juntas da mão Dex3 (7 por mão).
HAND_JOINT_IDS = tuple(range(NUM_HAND_MOTORS))

ARM_VEL_LIMIT = float(_os_top.environ.get("G1_ARM_VEL_LIMIT", "20.0"))   # rad/s
ARM_STREAM_HZ = float(_os_top.environ.get("G1_ARM_STREAM_HZ", "250.0"))
HAND_VEL_LIMIT = float(_os_top.environ.get("G1_HAND_VEL_LIMIT", "10.0"))  # rad/s
HAND_STREAM_HZ = float(_os_top.environ.get("G1_HAND_STREAM_HZ", "100.0"))
# Liga/desliga a rampa onboard (no robô). DEFAULT=0: a rampa já é feita NO HOST
# (laptop) pelo arm-streamer 250Hz + hand-streamer 100Hz, que clipam contra o
# COMANDO anterior (convergem ao alvo, mantêm a força do aperto). Ligar aqui (=1)
# cria rampa DUPLA: o modo "unitree" re-clipa contra o MEDIDO, prendendo o erro de
# posição em <=vmax/ciclo → mão lenta E fraca (torque = kp * erro_minúsculo). Só
# ligue (=1) se o host NÃO estiver rodando os streamers (G1_HAND_STREAMER=0).
RAMP_ONBOARD = _os_top.environ.get("G1_RAMP_ONBOARD", "0") not in ("", "0", "false", "False")
# Modo da rampa:
#   "interp"  = interpolação temporal q_atual→q_alvo em ~INTERP_S (suaviza o staircase
#               de 30Hz em qualquer velocidade). Clip de velocidade como backstop.
#   "unitree" = FIEL ao G1_29_ArmController: clip de velocidade puro contra a posição
#               MEDIDA (get_current_dual_arm_q), sem interpolação temporal.
RAMP_MODE = _os_top.environ.get("G1_RAMP_MODE", "unitree").strip().lower()

# WATCHDOG DE SEGURANÇA: se o laptop para de mandar comando por > STALE_S (encoding
# de save, Ctrl-C, crash, queda de rede), a rampa SOLTA o braço suavemente decaindo
# kp→0 em RELEASE_S (em vez de segurar a pose rígida pra sempre, que obrigava a matar
# o server). kd é mantido → o braço desce amortecido, não em queda livre.
STALE_S = float(_os_top.environ.get("G1_STALE_S", "2.0"))
RELEASE_S = float(_os_top.environ.get("G1_RELEASE_S", "1.5"))



def _scale_clip(cur, tgt, vmax):
    """Clip de velocidade VETORIAL: se alguma junta excede vmax (por ciclo), escala
    TODAS proporcionalmente — preserva a direção do movimento no espaço de juntas
    (igual ao clip_arm_q_target da Unitree). cur/tgt: listas de float."""
    if vmax <= 0:
        return list(tgt)
    mx = 0.0
    for c, t in zip(cur, tgt):
        d = t - c
        if d < 0:
            d = -d
        if d > mx:
            mx = d
    if mx <= vmax:
        return list(tgt)
    scale = mx / vmax
    return [c + (t - c) / scale for c, t in zip(cur, tgt)]


class _Holder:
    """Caixa thread-safe para o último alvo recebido (dict do JSON). As versões
    (incrementadas pelo receiver a cada alvo novo) deixam a rampa saber quando
    reconstruir o struct — assim o dict_to_*cmd (caro) só roda a 30Hz, não a 250Hz."""
    __slots__ = (
        "lock", "left", "right", "body", "left_ver", "right_ver", "body_ver",
        "meas_body", "meas_left", "meas_right", "body_ts", "left_ts", "right_ts",
    )

    def __init__(self):
        self.lock = threading.Lock()
        self.left = None
        self.right = None
        self.body = None
        self.left_ver = 0
        self.right_ver = 0
        self.body_ver = 0
        # perf_counter do último comando recebido (watchdog de staleness/release).
        self.body_ts = 0.0
        self.left_ts = 0.0
        self.right_ts = 0.0
        # Posição MEDIDA (q) p/ o modo "unitree" (clip contra o medido). Listas
        # alinhadas a ARM_JOINT_IDS / HAND_JOINT_IDS; None até o 1º estado chegar.
        self.meas_body = None   # q medido das juntas do braço (ordem ARM_JOINT_IDS)
        self.meas_left = None   # q medido dos dedos esq (ordem HAND_JOINT_IDS)
        self.meas_right = None  # q medido dos dedos dir


def lowstate_to_dict(msg: hg_LowState) -> dict[str, Any]:
    """Convert LowState SDK message to a JSON-serializable dictionary."""
    motor_states = []
    for i in range(NUM_MOTORS):
        temp = msg.motor_state[i].temperature
        avg_temp = float(sum(temp) / len(temp)) if isinstance(temp, list) else float(temp)
        motor_states.append(
            {
                "q": float(msg.motor_state[i].q),
                "dq": float(msg.motor_state[i].dq),
                "tau_est": float(msg.motor_state[i].tau_est),
                "temperature": avg_temp,
            }
        )

    return {
        "motor_state": motor_states,
        "imu_state": {
            "quaternion": [float(x) for x in msg.imu_state.quaternion],
            "gyroscope": [float(x) for x in msg.imu_state.gyroscope],
            "accelerometer": [float(x) for x in msg.imu_state.accelerometer],
            "rpy": [float(x) for x in msg.imu_state.rpy],
            "temperature": float(msg.imu_state.temperature),
        },
        # Encode bytes as base64 for JSON compatibility
        "wireless_remote": base64.b64encode(bytes(msg.wireless_remote)).decode("ascii"),
        "mode_machine": int(msg.mode_machine),
    }


def handstate_to_dict(msg: HandState_, side: str) -> dict[str, Any]:
    """Convert HandState SDK message to a JSON-serializable dictionary."""
    motor_states = []
    for i in range(NUM_HAND_MOTORS):
        motor_states.append(
            {
                "q": float(msg.motor_state[i].q),
                "dq": float(msg.motor_state[i].dq),
                "tau_est": float(msg.motor_state[i].tau_est),
            }
        )
    # FIX: inclui a pressão tátil (press_sensor_state). Sem isto o servidor
    # publicava só side+motor_state e o cliente gravava pressure=0 — sensor tátil
    # "zerado" no dataset, apesar do firmware Dex3 reportar valores reais.
    press_sensors = []
    if hasattr(msg, "press_sensor_state"):
        for p in msg.press_sensor_state:
            press_sensors.append(
                {
                    "pressure": list(p.pressure),
                    "temperature": list(p.temperature),
                }
            )
    return {
        "side": side,
        "motor_state": motor_states,
        "press_sensor_state": press_sensors,
    }


def dict_to_lowcmd(data: dict[str, Any]) -> hg_LowCmd:
    """Convert dictionary back to LowCmd SDK message."""
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = data.get("mode_pr", 0)
    cmd.mode_machine = data.get("mode_machine", 0)

    for i, motor_data in enumerate(data.get("motor_cmd", [])):
        cmd.motor_cmd[i].mode = motor_data.get("mode", 0)
        cmd.motor_cmd[i].q = motor_data.get("q", 0.0)
        cmd.motor_cmd[i].dq = motor_data.get("dq", 0.0)
        cmd.motor_cmd[i].kp = motor_data.get("kp", 0.0)
        cmd.motor_cmd[i].kd = motor_data.get("kd", 0.0)
        cmd.motor_cmd[i].tau = motor_data.get("tau", 0.0)

    return cmd


def dict_to_handcmd(data: dict[str, Any]) -> HandCmd_:
    """Convert dictionary back to HandCmd SDK message."""
    cmd = unitree_hg_msg_dds__HandCmd_()
    for i, motor_data in enumerate(data.get("motor_cmd", [])):
        cmd.motor_cmd[i].mode = motor_data.get("mode", 0)
        cmd.motor_cmd[i].q = motor_data.get("q", 0.0)
        cmd.motor_cmd[i].dq = motor_data.get("dq", 0.0)
        cmd.motor_cmd[i].kp = motor_data.get("kp", 0.0)
        cmd.motor_cmd[i].kd = motor_data.get("kd", 0.0)
        cmd.motor_cmd[i].tau = motor_data.get("tau", 0.0)
    return cmd


def state_forward_loop(
    lowstate_sub: ChannelSubscriber,
    lowstate_sock: zmq.Socket,
    state_period: float,
    shutdown_event: threading.Event,
    holder: _Holder | None = None,
) -> None:
    """Read observation from DDS and forward to ZMQ clients."""
    last_state_time = 0.0

    while not shutdown_event.is_set():
        # read from DDS
        msg = lowstate_sub.Read()
        if msg is None:
            continue

        # Captura o q MEDIDO das juntas do braço p/ o modo "unitree" (clip contra medido).
        if holder is not None:
            try:
                mb = [float(msg.motor_state[j].q) for j in ARM_JOINT_IDS]
                with holder.lock:
                    holder.meas_body = mb
            except Exception:
                pass

        now = time.time()
        # optional downsampling (if robot dds rate > state_period)
        if now - last_state_time >= state_period:
            # Convert to dict and serialize with JSON
            state_dict = lowstate_to_dict(msg)
            payload = json.dumps({"topic": kTopicLowState, "data": state_dict}).encode("utf-8")
            # if no subscribers / tx buffer full, just drop
            with contextlib.suppress(zmq.Again):
                lowstate_sock.send(payload, zmq.NOBLOCK)
            last_state_time = now


def handstate_forward_loop(
    left_sub: ChannelSubscriber,
    right_sub: ChannelSubscriber,
    handstate_sock: zmq.Socket,
    state_period: float,
    shutdown_event: threading.Event,
    holder: _Holder | None = None,
) -> None:
    """Read hand state from DDS and forward to ZMQ clients."""
    last_left_time = 0.0
    last_right_time = 0.0

    while not shutdown_event.is_set():
        now = time.time()

        # Read left hand state
        msg_left = left_sub.Read()
        if msg_left is not None:
            if holder is not None:
                try:
                    ml = [float(msg_left.motor_state[j].q) for j in HAND_JOINT_IDS]
                    with holder.lock:
                        holder.meas_left = ml
                except Exception:
                    pass
            if now - last_left_time >= state_period:
                state_dict = handstate_to_dict(msg_left, "left")
                payload = json.dumps({"topic": kTopicDex3LeftState, "data": state_dict}).encode("utf-8")
                with contextlib.suppress(zmq.Again):
                    handstate_sock.send(payload, zmq.NOBLOCK)
                last_left_time = now

        # Read right hand state
        msg_right = right_sub.Read()
        if msg_right is not None:
            if holder is not None:
                try:
                    mr = [float(msg_right.motor_state[j].q) for j in HAND_JOINT_IDS]
                    with holder.lock:
                        holder.meas_right = mr
                except Exception:
                    pass
            if now - last_right_time >= state_period:
                state_dict = handstate_to_dict(msg_right, "right")
                payload = json.dumps({"topic": kTopicDex3RightState, "data": state_dict}).encode("utf-8")
                with contextlib.suppress(zmq.Again):
                    handstate_sock.send(payload, zmq.NOBLOCK)
                last_right_time = now

        time.sleep(0.001)  # Small sleep to avoid busy loop


def cmd_receiver_loop(
    lowcmd_sock: zmq.Socket,
    holder: _Holder,
    shutdown_event: threading.Event,
) -> None:
    """Recebe o ALVO do braço (LowCmd) do ZMQ a ~30Hz, faz o json.loads (caro) AQUI
    e guarda o dict mais recente. Quem publica no DDS é a thread de rampa (250Hz)."""
    while not shutdown_event.is_set():
        try:
            payload = lowcmd_sock.recv()
        except zmq.ContextTerminated:
            break
        except Exception:
            continue
        try:
            msg_dict = json.loads(payload.decode("utf-8"))
        except Exception:
            continue
        if msg_dict.get("topic", "") != kTopicLowCommand_Debug:
            continue
        with holder.lock:
            holder.body = msg_dict.get("data", {})
            holder.body_ver += 1
            holder.body_ts = time.perf_counter()


def cmd_ramp_loop(
    holder: _Holder,
    lowcmd_pub_debug: ChannelPublisher,
    crc: CRC,
    shutdown_event: threading.Event,
) -> None:
    """Publica no DDS a ARM_STREAM_HZ (250Hz) INTERPOLANDO NO TEMPO entre alvos de 30Hz.

    Interpolação temporal (não só clip de velocidade): a cada alvo novo, faz uma rampa
    LINEAR de q_atual → q_alvo ao longo de INTERP_S (= período do alvo, ~33ms). Isso
    suaviza o staircase de 30Hz em QUALQUER velocidade — o clip de velocidade puro só
    suavizaria movimentos mais rápidos que o limite. _scale_clip fica como backstop de
    segurança contra glitch de IK. Só cópia-struct + CRC + Write rodam a 250Hz (barato)."""
    dt = 1.0 / ARM_STREAM_HZ if ARM_STREAM_HZ > 0 else 1.0 / 250.0
    interp_s = 1.0 / float(_os_top.environ.get("G1_ARM_INTERP_HZ", "30.0"))
    vmax = ARM_VEL_LIMIT * dt  # backstop por ciclo (glitch)
    cur_q = None      # q comandado atual (continuidade entre segmentos)
    start_q = None    # q no início do segmento de interpolação
    goal_q = None     # alvo do segmento
    seg_t0 = 0.0
    cmd = None
    base_kp = None    # kp original do braço (p/ o decay do watchdog)
    seen_ver = -1
    released = False  # já avisou que soltou?
    fails = 0
    while not shutdown_event.is_set():
        t0 = time.perf_counter()
        try:
            with holder.lock:
                data = holder.body
                ver = holder.body_ver
                body_ts = holder.body_ts
            stale_for = (t0 - body_ts) if body_ts else 0.0

            # WATCHDOG: laptop parou de comandar → solta o braço suave (kp→0).
            if cmd is not None and body_ts and stale_for > STALE_S:
                rel = 1.0 - (stale_for - STALE_S) / RELEASE_S if RELEASE_S > 0 else 0.0
                if rel < 0.0:
                    rel = 0.0
                if base_kp is not None:
                    for k, j in enumerate(ARM_JOINT_IDS):
                        cmd.motor_cmd[j].kp = base_kp[k] * rel  # q mantido (cur_q)
                cmd.crc = crc.Crc(cmd)
                lowcmd_pub_debug.Write(cmd)
                if not released:
                    print(f"[arm-ramp] WATCHDOG: sem comando há {stale_for:.1f}s — "
                          f"soltando o braço (kp→0 em {RELEASE_S:.1f}s).", flush=True)
                    released = True
                fails = 0
                elapsed = time.perf_counter() - t0
                if dt - elapsed > 0:
                    shutdown_event.wait(dt - elapsed)
                continue
            released = False

            if data is not None:
                if ver != seen_ver:
                    # Alvo novo (30Hz): json já foi no receiver; reconstrói struct 1x.
                    cmd = dict_to_lowcmd(data)
                    goal_new = [cmd.motor_cmd[j].q for j in ARM_JOINT_IDS]
                    base_kp = [cmd.motor_cmd[j].kp for j in ARM_JOINT_IDS]
                    if cur_q is None:
                        cur_q = list(goal_new)  # 1º alvo ≈ pose medida → sem salto.
                    start_q = list(cur_q)       # novo segmento começa do q atual
                    goal_q = goal_new
                    seg_t0 = t0
                    seen_ver = ver
                if goal_q is not None:
                    if RAMP_MODE == "unitree":
                        # FIEL: clip de velocidade puro contra o MEDIDO (clip_arm_q_target).
                        with holder.lock:
                            meas = holder.meas_body
                        base = meas if meas is not None else cur_q
                        cur_q = _scale_clip(base, goal_q, vmax)
                    else:
                        # interp: rampa temporal q_atual→q_alvo em interp_s (+ backstop).
                        alpha = (t0 - seg_t0) / interp_s if interp_s > 0 else 1.0
                        if alpha > 1.0:
                            alpha = 1.0
                        interp_q = [s + alpha * (g - s) for s, g in zip(start_q, goal_q)]
                        cur_q = _scale_clip(cur_q, interp_q, vmax)
                    for k, j in enumerate(ARM_JOINT_IDS):
                        cmd.motor_cmd[j].q = cur_q[k]
                    cmd.crc = crc.Crc(cmd)
                    lowcmd_pub_debug.Write(cmd)
            fails = 0
        except Exception as e:  # noqa: BLE001
            fails += 1
            if fails <= 3 or fails % int(ARM_STREAM_HZ) == 0:
                print(f"[arm-ramp] erro no worker (#{fails}): {type(e).__name__}: {e}", flush=True)
        elapsed = time.perf_counter() - t0
        if dt - elapsed > 0:
            shutdown_event.wait(dt - elapsed)


def cmd_forward_loop(
    lowcmd_sock: zmq.Socket,
    lowcmd_pub_debug: ChannelPublisher,
    crc: CRC,
) -> None:
    """LEGADO (G1_RAMP_ONBOARD=0): forward direto ZMQ->DDS, degrau cru a 30Hz."""
    while True:
        try:
            payload = lowcmd_sock.recv()
        except zmq.ContextTerminated:
            break
        msg_dict = json.loads(payload.decode("utf-8"))

        topic = msg_dict.get("topic", "")
        cmd_data = msg_dict.get("data", {})

        # Reconstruct LowCmd object from dict
        cmd = dict_to_lowcmd(cmd_data)

        # recompute crc
        cmd.crc = crc.Crc(cmd)

        if topic == kTopicLowCommand_Debug:
            lowcmd_pub_debug.Write(cmd)


def handcmd_receiver_loop(
    handcmd_sock: zmq.Socket,
    holder: _Holder,
    shutdown_event: threading.Event,
) -> None:
    """Recebe os ALVOS das mãos (esq/dir) do ZMQ a ~30Hz e guarda o dict mais recente
    de cada lado. A publicação no DDS fica na thread de rampa (100Hz)."""
    while not shutdown_event.is_set():
        try:
            payload = handcmd_sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.001)
            continue
        except zmq.ContextTerminated:
            break
        except Exception:
            continue
        try:
            msg_dict = json.loads(payload.decode("utf-8"))
        except Exception:
            continue
        topic = msg_dict.get("topic", "")
        data = msg_dict.get("data", {})
        if topic == kTopicDex3LeftCommand:
            with holder.lock:
                holder.left = data
                holder.left_ver += 1
                holder.left_ts = time.perf_counter()
        elif topic == kTopicDex3RightCommand:
            with holder.lock:
                holder.right = data
                holder.right_ver += 1
                holder.right_ts = time.perf_counter()


def handcmd_ramp_loop(
    holder: _Holder,
    left_pub: ChannelPublisher,
    right_pub: ChannelPublisher,
    shutdown_event: threading.Event,
) -> None:
    """Publica os dedos no DDS a HAND_STREAM_HZ (100Hz) interpolando até o alvo com
    clip de velocidade. Mantém a publicação sem alvo novo → segura o grip (watchdog
    do firmware Dex3) durante a pausa de encode entre episódios."""
    dt = 1.0 / HAND_STREAM_HZ if HAND_STREAM_HZ > 0 else 1.0 / 100.0
    interp_s = 1.0 / float(_os_top.environ.get("G1_HAND_INTERP_HZ", "30.0"))
    vmax = HAND_VEL_LIMIT * dt  # backstop por ciclo

    # estado por mão: (cur, start, goal, seg_t0, cmd, seen_ver)
    st = {"l": [None, None, None, 0.0, None, -1], "r": [None, None, None, 0.0, None, -1]}

    def _step(side, data, ver, meas, pub, t0):
        s = st[side]
        if data is None:
            return
        if ver != s[5]:
            s[4] = dict_to_handcmd(data)            # cmd (rebuild 1x por alvo)
            goal_new = [s[4].motor_cmd[j].q for j in HAND_JOINT_IDS]
            if s[0] is None:
                s[0] = list(goal_new)               # cur seed sem salto
            s[1] = list(s[0])                       # start = cur
            s[2] = goal_new                         # goal
            s[3] = t0                               # seg_t0
            s[5] = ver
        if s[2] is None:
            return
        if RAMP_MODE == "unitree":
            base = meas if meas is not None else s[0]
            s[0] = _scale_clip(base, s[2], vmax)
        else:
            alpha = (t0 - s[3]) / interp_s if interp_s > 0 else 1.0
            if alpha > 1.0:
                alpha = 1.0
            interp_q = [a + alpha * (g - a) for a, g in zip(s[1], s[2])]
            s[0] = _scale_clip(s[0], interp_q, vmax)
        for k, j in enumerate(HAND_JOINT_IDS):
            s[4].motor_cmd[j].q = s[0][k]
        pub.Write(s[4])

    while not shutdown_event.is_set():
        t0 = time.perf_counter()
        try:
            with holder.lock:
                dl, vl = holder.left, holder.left_ver
                dr, vr = holder.right, holder.right_ver
                ml, mr = holder.meas_left, holder.meas_right
            _step("l", dl, vl, ml, left_pub, t0)
            _step("r", dr, vr, mr, right_pub, t0)
        except Exception as e:  # noqa: BLE001
            print(f"[hand-ramp] erro: {type(e).__name__}: {e}", flush=True)
        elapsed = time.perf_counter() - t0
        if dt - elapsed > 0:
            shutdown_event.wait(dt - elapsed)


def handcmd_forward_loop(
    handcmd_sock: zmq.Socket,
    left_pub: ChannelPublisher,
    right_pub: ChannelPublisher,
    shutdown_event: threading.Event,
) -> None:
    """LEGADO (G1_RAMP_ONBOARD=0): forward direto ZMQ->DDS das mãos."""
    while not shutdown_event.is_set():
        try:
            payload = handcmd_sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.001)
            continue
        except zmq.ContextTerminated:
            break

        msg_dict = json.loads(payload.decode("utf-8"))
        topic = msg_dict.get("topic", "")
        cmd_data = msg_dict.get("data", {})

        # Reconstruct HandCmd object from dict
        cmd = dict_to_handcmd(cmd_data)

        if topic == kTopicDex3LeftCommand:
            left_pub.Write(cmd)
        elif topic == kTopicDex3RightCommand:
            right_pub.Write(cmd)


def main() -> None:
    """Main entry point for the robot server bridge."""
    # DDS na interface $G1_DDS_IFACE, se setado
    import os as _os
    _iface = _os.environ.get("G1_DDS_IFACE")
    if _iface:
        ChannelFactoryInitialize(0, _iface)
    else:
        ChannelFactoryInitialize(0)

    # stop all active publishers on the robot
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    status, result = msc.CheckMode()
    while result is not None and "name" in result and result["name"]:
        msc.ReleaseMode()
        status, result = msc.CheckMode()
        time.sleep(1.0)

    crc = CRC()

    # =========================================================================
    # Body DDS channels
    # =========================================================================
    lowcmd_pub_debug = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
    lowcmd_pub_debug.Init()

    lowstate_sub = ChannelSubscriber(kTopicLowState, hg_LowState)
    lowstate_sub.Init()

    # =========================================================================
    # Dex3 Hand DDS channels
    # =========================================================================
    left_hand_cmd_pub = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
    left_hand_cmd_pub.Init()
    right_hand_cmd_pub = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
    right_hand_cmd_pub.Init()

    left_hand_state_sub = ChannelSubscriber(kTopicDex3LeftState, HandState_)
    left_hand_state_sub.Init()
    right_hand_state_sub = ChannelSubscriber(kTopicDex3RightState, HandState_)
    right_hand_state_sub.Init()

    # =========================================================================
    # ZMQ sockets
    # =========================================================================
    ctx = zmq.Context.instance()

    # Body command: receive from remote client
    lowcmd_sock = ctx.socket(zmq.PULL)
    lowcmd_sock.bind(f"tcp://0.0.0.0:{LOWCMD_PORT}")

    # Body state: publish to remote clients
    lowstate_sock = ctx.socket(zmq.PUB)
    lowstate_sock.bind(f"tcp://0.0.0.0:{LOWSTATE_PORT}")

    # Hand state: publish to remote clients
    handstate_sock = ctx.socket(zmq.PUB)
    handstate_sock.bind(f"tcp://0.0.0.0:{HANDSTATE_PORT}")

    # Hand command: receive from remote client
    handcmd_sock = ctx.socket(zmq.PULL)
    handcmd_sock.bind(f"tcp://0.0.0.0:{HANDCMD_PORT}")

    state_period = 0.002  # ~500 hz
    shutdown_event = threading.Event()
    holder = _Holder()

    # =========================================================================
    # Start forwarding threads
    # =========================================================================

    # Body state forwarding (passa holder p/ capturar o q medido no modo "unitree")
    t_state = threading.Thread(
        target=state_forward_loop,
        args=(lowstate_sub, lowstate_sock, state_period, shutdown_event, holder),
        name="BodyStateForward",
    )
    t_state.start()

    # Hand state forwarding
    t_handstate = threading.Thread(
        target=handstate_forward_loop,
        args=(left_hand_state_sub, right_hand_state_sub, handstate_sock, state_period, shutdown_event, holder),
        name="HandStateForward",
    )
    t_handstate.start()

    aux_threads = [t_state, t_handstate]

    if RAMP_ONBOARD:
        # MÃOS: receiver (json @30Hz) + rampa (DDS @100Hz com clip de velocidade).
        t_handcmd = threading.Thread(
            target=handcmd_receiver_loop,
            args=(handcmd_sock, holder, shutdown_event),
            name="HandCmdReceiver",
        )
        t_handcmd.start()
        t_hand_ramp = threading.Thread(
            target=handcmd_ramp_loop,
            args=(holder, left_hand_cmd_pub, right_hand_cmd_pub, shutdown_event),
            name="HandCmdRamp",
        )
        t_hand_ramp.start()

        # BRAÇO: receiver (json @30Hz) + rampa (DDS @250Hz). A rampa fica em thread
        # própria; o receiver no main thread (bloqueia em recv).
        t_arm_ramp = threading.Thread(
            target=cmd_ramp_loop,
            args=(holder, lowcmd_pub_debug, crc, shutdown_event),
            name="ArmCmdRamp",
        )
        t_arm_ramp.start()
        aux_threads += [t_handcmd, t_hand_ramp, t_arm_ramp]

        _mode_desc = ("interp (rampa temporal + clip backstop)" if RAMP_MODE != "unitree"
                      else "unitree (clip de velocidade puro contra o MEDIDO)")
        print(
            f"bridge running [RAMPA ONBOARD · modo={RAMP_MODE} → {_mode_desc}: "
            f"braço {ARM_STREAM_HZ:.0f}Hz/{ARM_VEL_LIMIT:.0f}rad/s, "
            f"mãos {HAND_STREAM_HZ:.0f}Hz/{HAND_VEL_LIMIT:.0f}rad/s] — laptop manda alvo @30Hz",
            flush=True,
        )
        try:
            cmd_receiver_loop(lowcmd_sock, holder, shutdown_event)
        except KeyboardInterrupt:
            print("shutting down bridge...")
        finally:
            shutdown_event.set()
            ctx.term()
            for t in aux_threads:
                t.join(timeout=2.0)
    else:
        # LEGADO: forward direto (degrau cru a 30Hz).
        t_handcmd = threading.Thread(
            target=handcmd_forward_loop,
            args=(handcmd_sock, left_hand_cmd_pub, right_hand_cmd_pub, shutdown_event),
            name="HandCmdForward",
        )
        t_handcmd.start()
        aux_threads.append(t_handcmd)

        print("bridge running (body + hands: lowstate/handstate -> zmq, lowcmd/handcmd -> dds)")

        try:
            cmd_forward_loop(lowcmd_sock, lowcmd_pub_debug, crc)
        except KeyboardInterrupt:
            print("shutting down bridge...")
        finally:
            shutdown_event.set()
            ctx.term()  # terminates blocking zmq.recv() calls
            for t in aux_threads:
                t.join(timeout=2.0)


if __name__ == "__main__":
    main()

