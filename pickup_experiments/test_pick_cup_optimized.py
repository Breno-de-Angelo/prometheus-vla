#!/usr/bin/env python3
"""
Trajetoria de pickup com indices de motor CORRETOS (braco direito = motores 22-28).
Poses calculadas via IK na pose real de standing (pickup_experiments/ik_poses.json):
  - REACH: middle_1 a 1.45cm do copo
  - LIFT:  levanta o copo ~12cm

Mantem pernas/cintura/braco-esquerdo em q=0 (standing) com kp alto.
"""

import zmq
import json
import time
from pathlib import Path

# Indices de motor do braco direito (CONFIRMADO: body_joint_index)
R_SHOULDER_PITCH = 22
R_SHOULDER_ROLL  = 23
R_SHOULDER_YAW   = 24
R_ELBOW          = 25
R_WRIST_ROLL     = 26
R_WRIST_PITCH    = 27
R_WRIST_YAW      = 28

# Mao direita (indices locais 0-6)
HAND_OPEN  = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
HAND_CLOSE = {0: 0.7, 1: 0.7, 2: 0.7, 3: 0.9, 4: 0.9, 5: 0.9, 6: 0.9}


def send_action(sock, body_motors, hand_motors=None):
    payload = {"body_motors": body_motors}
    if hand_motors:
        payload["right_hand"] = hand_motors
    sock.send_string(json.dumps(payload))


def build_body_motors(arm_dict, leg_kp=100.0, leg_kd=2.5, arm_kp=150.0, arm_kd=3.0):
    """29 motores. Pernas/cintura/braco-esq em q=0 (standing). Braco dir = arm_dict."""
    motors = []
    for i in range(29):
        if i in arm_dict:
            motors.append({"idx": i, "q": arm_dict[i], "kp": arm_kp, "kd": arm_kd})
        else:
            # Mantem standing (zero config) com ganho alto para estabilidade
            motors.append({"idx": i, "q": 0.0, "kp": leg_kp, "kd": leg_kd})
    return motors


def build_hand_motors(hand_dict, kp=20.0, kd=1.0):
    return [{"idx": i, "q": hand_dict.get(i, 0.0), "kp": kp, "kd": kd} for i in range(7)]


def lerp_arm(q_from, q_to, t):
    """Interpola dict de braco {idx: q}."""
    return {idx: q_from.get(idx, 0.0) + (q_to.get(idx, 0.0) - q_from.get(idx, 0.0)) * t
            for idx in set(q_from) | set(q_to)}


def main():
    # Carrega poses IK
    poses = json.loads(Path("pickup_experiments/ik_poses.json").read_text())
    REACH = {int(k): v for k, v in poses["reach"].items()}
    LIFT  = {int(k): v for k, v in poses["lift"].items()}
    HOME  = {i: 0.0 for i in range(22, 29)}

    print("[Pickup] Conectando a 192.168.15.111:6001...")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect("tcp://192.168.15.111:6001")
    time.sleep(0.5)
    print("[Pickup] Conectado!\n")

    def send_phase(name, arm, hand, hand_kp, hand_kd, n_steps, arm_kp=150.0):
        print(f"[{name}] arm[22]={arm.get(22,0):.2f} arm[27]={arm.get(27,0):.2f} "
              f"hand={'fechada' if any(v>0.3 for v in hand.values()) else 'aberta'}")
        bm = build_body_motors(arm, arm_kp=arm_kp)
        hm = build_hand_motors(hand, hand_kp, hand_kd)
        for _ in range(n_steps):
            send_action(sock, bm, hm)
            time.sleep(0.02)

    # Fase 1: HOME (estabiliza standing, mao aberta) — 0.5s
    send_phase("1-Home", HOME, HAND_OPEN, 20.0, 1.0, 25)
    time.sleep(0.3)

    # Fase 2: aproximacao gradual HOME→REACH (mao aberta) — 5 sub-passos, 1.5s
    print("[2-Approach] HOME -> REACH gradual")
    for i in range(1, 6):
        t = i / 5.0
        arm = lerp_arm(HOME, REACH, t)
        send_phase(f"  approach {t:.1f}", arm, HAND_OPEN, 20.0, 1.0, 15)
    time.sleep(0.3)

    # Fase 3: estabiliza na REACH — 1s
    send_phase("3-Reach-settle", REACH, HAND_OPEN, 20.0, 1.0, 50)
    time.sleep(0.3)

    # Fase 4: fecha mao complacente (kp baixo) — 1.2s
    send_phase("4-Grasp", REACH, HAND_CLOSE, 5.0, 3.0, 60)
    time.sleep(0.2)

    # Fase 5: aperta garra (kp maior) — 0.6s
    send_phase("5-Tighten", REACH, HAND_CLOSE, 18.0, 2.0, 30)
    time.sleep(0.2)

    # Fase 6: levanta REACH→LIFT gradual, mao fechada — 5 sub-passos, 2s
    print("[6-Lift] REACH -> LIFT gradual (mao fechada)")
    for i in range(1, 6):
        t = i / 5.0
        arm = lerp_arm(REACH, LIFT, t)
        send_phase(f"  lift {t:.1f}", arm, HAND_CLOSE, 20.0, 2.0, 16, arm_kp=150.0)
    time.sleep(0.2)

    # Fase 7: mantem elevado — 1.5s
    send_phase("7-Hold", LIFT, HAND_CLOSE, 20.0, 2.0, 75, arm_kp=150.0)

    print("\n[Pickup] Trajetoria completa!")
    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
