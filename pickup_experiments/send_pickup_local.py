#!/usr/bin/env python3
"""
Sender ZMQ da trajetoria de pickup pro SIM REAL (run_sim_visible.py), em localhost.

Streama os 29 motores do corpo + 7 da mao direita continuamente (senao o robo
desaba). Sequencia top-down (nao derruba o copo):
  1. SETTLE  - braco erguido (shoulder_pitch=-1.5), copo assenta sem ser empurrado
  2. ABOVE   - mao ~15cm acima do copo
  3. REACH   - baixa reto ate a altura de pega
  4. GRASP   - fecha a mao (complacente -> aperta)
  5. LIFT    - levanta + segura

Uso (com run_sim_visible.py ja rodando):
    conda activate g1
    python pickup_experiments/send_pickup_local.py
"""
import zmq
import json
import time
from pathlib import Path

HERE = Path(__file__).parent
ADDR = "tcp://127.0.0.1:6001"

ARM_IDS = [22, 23, 24, 25, 26, 27, 28]
# Motores da mao (0-6): thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1
# Mao ABERTA (polegar parado/fora do caminho)
HAND_OPEN = {i: 0.0 for i in range(7)}
# Mao FECHADA (ref. teleop RIGHT_HAND_CLOSED_TARGETS): polegar fecha no NEGATIVO
HAND_CLOSE = {0: 0.0, 1: -1.5, 2: -1.5, 3: 1.5, 4: 1.5, 5: 1.5, 6: 1.5}
# So os dedos (indicador+medio) fechados, polegar ainda aberto
HAND_FINGERS_ONLY = {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}


def build_body(arm, leg_kp=120.0, leg_kd=2.5, arm_kp=200.0, arm_kd=4.0):
    out = []
    for i in range(29):
        if i in arm:
            out.append({"idx": i, "q": arm[i], "kp": arm_kp, "kd": arm_kd})
        else:
            out.append({"idx": i, "q": 0.0, "kp": leg_kp, "kd": leg_kd})
    return out


def build_hand(hand, kp, kd):
    return [{"idx": i, "q": hand.get(i, 0.0), "kp": kp, "kd": kd} for i in range(7)]


def lerp(a, b, t):
    return {k: a.get(k, 0.0) + (b.get(k, 0.0) - a.get(k, 0.0)) * t for k in set(a) | set(b)}


def main():
    poses = json.loads((HERE / "ik_poses.json").read_text())
    START = {int(k): v for k, v in poses["start"].items()}  # recuado, aprox. frontal
    REACH = {int(k): v for k, v in poses["reach"].items()}
    LIFT = {int(k): v for k, v in poses["lift"].items()}

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(ADDR)
    print(f"[sender] conectado em {ADDR}")
    time.sleep(0.5)

    def send(arm, hand, hand_kp, hand_kd, arm_kp=200.0, secs=1.0, hz=100):
        n = int(secs * hz)
        bm = build_body(arm, arm_kp=arm_kp)
        hm = build_hand(hand, hand_kp, hand_kd)
        for _ in range(n):
            sock.send_string(json.dumps({"body_motors": bm, "right_hand": hm}))
            time.sleep(1.0 / hz)

    def ramp(a, b, hand, hand_kp, hand_kd, arm_kp=200.0, secs=2.0, hz=100):
        n = int(secs * hz)
        for i in range(1, n + 1):
            arm = lerp(a, b, i / n)
            bm = build_body(arm, arm_kp=arm_kp)
            hm = build_hand(hand, hand_kp, hand_kd)
            sock.send_string(json.dumps({"body_motors": bm, "right_hand": hm}))
            time.sleep(1.0 / hz)

    print("[1] START recuado (braco pouco erguido, copo assenta sem ser empurrado)")
    send(START, HAND_OPEN, 20, 1, secs=3.0)

    print("[2] START -> REACH (move o braco PRA FRENTE ate encostar indicador+medio)")
    ramp(START, REACH, HAND_OPEN, 20, 1, secs=3.0)
    send(REACH, HAND_OPEN, 20, 1, secs=1.0)

    print("[3] Fecha os dedos (indicador+medio), polegar ainda parado")
    send(REACH, HAND_FINGERS_ONLY, 12, 2, secs=1.0)

    print("[4] Agora fecha TUDO incluindo o polegar (complacente -> aperta)")
    send(REACH, HAND_CLOSE, 8, 3, secs=1.5)
    send(REACH, HAND_CLOSE, 25, 2, secs=1.0)

    print("[5] REACH -> LIFT + hold")
    ramp(REACH, LIFT, HAND_CLOSE, 25, 2, secs=2.5)
    send(LIFT, HAND_CLOSE, 25, 2, secs=3.0)

    print("[sender] trajetoria completa")
    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
