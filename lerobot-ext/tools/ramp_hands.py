#!/usr/bin/env python
"""
Rampa manual para as maos Dex3 do G1.
Uso:
  python lerobot-ext/tools/ramp_hands.py             # fechar -> abrir
  python lerobot-ext/tools/ramp_hands.py open        # so abre
  python lerobot-ext/tools/ramp_hands.py close       # so fecha
  python lerobot-ext/tools/ramp_hands.py --ip 10.9.8.73

Requer que o teleop NAO esteja rodando (conflito de porta 6003).
"""
import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("G1_LEFT_ARM_LIMP", "0")

from robot.unitree_g1.g1_utils import (
    Dex3_1_Left_JointIndex,
    Dex3_1_Right_JointIndex,
    DEX3_LEFT_LOWER_LIMITS,
    DEX3_LEFT_UPPER_LIMITS,
    DEX3_RIGHT_LOWER_LIMITS,
    DEX3_RIGHT_UPPER_LIMITS,
)

# ── targets ───────────────────────────────────────────────────────────
OPEN_LEFT  = np.zeros(7)
OPEN_RIGHT = np.zeros(7)
CLOSE_LEFT  = np.array([ 0.0,  0.920,  1.74, -1.57, -1.74, -1.57, -1.74])
CLOSE_RIGHT = np.array([ 0.0, -0.920, -1.74,  1.57,  1.74,  1.57,  1.74])

# kp/kd dos dedos (igual ao connect do UnitreeG1Dex3)
KP = 2.0
KD = 0.2

RATE_HZ   = 100   # frequencia da rampa
RAMP_SECS = 1.5   # duracao de cada rampa (abrir ou fechar)

def make_msg(joints, q_arr, kp, kd):
    import importlib
    sock_mod = importlib.import_module("robot.unitree_g1.unitree_sdk2_socket")
    msg = sock_mod.HandCmdMsg()
    for idx, jid in enumerate(joints):
        mode = (jid & 0x0F) | (0x01 << 4)
        msg.motor_cmd[jid].mode = mode
        msg.motor_cmd[jid].kp   = kp
        msg.motor_cmd[jid].kd   = kd
        msg.motor_cmd[jid].q    = float(q_arr[idx])
    return msg

def send_ramp(pub_left, pub_right, start_l, start_r, end_l, end_r, secs, rate):
    steps = int(secs * rate)
    dt = 1.0 / rate
    for i in range(steps + 1):
        alpha = i / steps
        q_l = start_l + alpha * (end_l - start_l)
        q_r = start_r + alpha * (end_r - start_r)
        q_l = np.clip(q_l, DEX3_LEFT_LOWER_LIMITS,  DEX3_LEFT_UPPER_LIMITS)
        q_r = np.clip(q_r, DEX3_RIGHT_LOWER_LIMITS, DEX3_RIGHT_UPPER_LIMITS)
        pub_left.Write(make_msg(Dex3_1_Left_JointIndex,  q_l, KP, KD))
        pub_right.Write(make_msg(Dex3_1_Right_JointIndex, q_r, KP, KD))
        time.sleep(dt)
    return q_l, q_r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", nargs="?", default="both",
                    choices=["open", "close", "both"],
                    help="open | close | both (default: both = fechar depois abrir)")
    ap.add_argument("--ip", default="10.9.8.73", help="IP do robo (default: 10.9.8.73)")
    args = ap.parse_args()

    import importlib
    sock_mod = importlib.import_module("robot.unitree_g1.unitree_sdk2_socket")
    sock_mod.ChannelFactoryInitialize(0, args.ip)
    time.sleep(0.3)

    pub_left  = sock_mod.ChannelPublisher(sock_mod.kTopicDex3LeftCommand,  None)
    pub_right = sock_mod.ChannelPublisher(sock_mod.kTopicDex3RightCommand, None)
    pub_left.Init(); pub_right.Init()

    print(f"[ramp_hands] conectado em {args.ip}:6003  KP={KP}  ramp={RAMP_SECS}s")

    if args.action in ("close", "both"):
        print(">>> FECHANDO maos...")
        q_l, q_r = send_ramp(pub_left, pub_right,
                              OPEN_LEFT, OPEN_RIGHT,
                              CLOSE_LEFT, CLOSE_RIGHT,
                              RAMP_SECS, RATE_HZ)
        print(f"    fechado: L={np.round(q_l,2)}  R={np.round(q_r,2)}")
        if args.action == "both":
            time.sleep(1.0)

    if args.action in ("open", "both"):
        start_l = CLOSE_LEFT  if args.action == "both" else OPEN_LEFT
        start_r = CLOSE_RIGHT if args.action == "both" else OPEN_RIGHT
        print(">>> ABRINDO maos...")
        q_l, q_r = send_ramp(pub_left, pub_right,
                              start_l, start_r,
                              OPEN_LEFT, OPEN_RIGHT,
                              RAMP_SECS, RATE_HZ)
        print(f"    aberto: L={np.round(q_l,2)}  R={np.round(q_r,2)}")

    print("[ramp_hands] concluido.")

if __name__ == "__main__":
    main()
