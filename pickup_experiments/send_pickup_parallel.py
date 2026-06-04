#!/usr/bin/env python3
"""
Sender ZMQ - variante PARALELA (mao paralela ao copo, movimento minimo).

Le ik_poses_parallel.json (start + trajetoria densa integrada por passos
cartesianos retos). Coreografia pedida:
  1. START   - mao acima da mesa, paralela ao copo, sem encostar (copo assenta)
  2. APPROACH- move o braco RETO PRA FRENTE mantendo paralelismo (mao aberta,
               polegar parado) ate encostar indicador+medio
  3. fecha dedos (indicador+medio), depois fecha TUDO incluindo o polegar
  4. LIFT    - sobe RETO, mao fechada

Uso (com run_sim_visible.py rodando):
    conda activate g1
    python pickup_experiments/send_pickup_parallel.py
"""
import zmq
import json
import time
from pathlib import Path

HERE = Path(__file__).parent
ADDR = "tcp://127.0.0.1:6001"

HAND_OPEN = {i: 0.0 for i in range(7)}
HAND_CLOSE = {0: 0.0, 1: -1.5, 2: -1.5, 3: 1.5, 4: 1.5, 5: 1.5, 6: 1.5}
HAND_FINGERS_ONLY = {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.5, 4: 1.5, 5: 1.5, 6: 1.5}


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


def main():
    poses = json.loads((HERE / "ik_poses_parallel.json").read_text())
    START = {int(k): v for k, v in poses["start"].items()}
    APPROACH = [{int(k): v for k, v in wp.items()} for wp in poses["approach"]]
    LIFT = [{int(k): v for k, v in wp.items()} for wp in poses["lift"]]

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(ADDR)
    print(f"[sender||] conectado em {ADDR}")
    time.sleep(0.5)

    def hold(arm, hand, hand_kp, hand_kd, secs, hz=100):
        bm = build_body(arm)
        hm = build_hand(hand, hand_kp, hand_kd)
        for _ in range(int(secs * hz)):
            sock.send_string(json.dumps({"body_motors": bm, "right_hand": hm}))
            time.sleep(1.0 / hz)

    def play(wps, hand, hand_kp, hand_kd, secs_per_wp=0.25, hz=100):
        hm = build_hand(hand, hand_kp, hand_kd)
        for arm in wps:
            bm = build_body(arm)
            for _ in range(int(secs_per_wp * hz)):
                sock.send_string(json.dumps({"body_motors": bm, "right_hand": hm}))
                time.sleep(1.0 / hz)

    print("[1] START (mao paralela acima da mesa, copo assenta)")
    hold(START, HAND_OPEN, 20, 1, secs=3.0)

    print("[2] APPROACH reto pra frente (mantendo paralelismo, mao aberta)")
    play(APPROACH, HAND_OPEN, 20, 1, secs_per_wp=0.35)
    hold(APPROACH[-1], HAND_OPEN, 20, 1, secs=0.8)

    print("[3] fecha dedos (indicador+medio), polegar parado")
    hold(APPROACH[-1], HAND_FINGERS_ONLY, 12, 2, secs=1.0)

    print("[4] fecha TUDO incluindo o polegar (complacente -> aperta)")
    hold(APPROACH[-1], HAND_CLOSE, 8, 3, secs=1.2)
    hold(APPROACH[-1], HAND_CLOSE, 25, 2, secs=1.0)

    print("[5] LIFT reto pra cima (mao fechada) + hold")
    play(LIFT, HAND_CLOSE, 25, 2, secs_per_wp=0.18)
    hold(LIFT[-1], HAND_CLOSE, 25, 2, secs=3.0)

    print("[sender||] trajetoria completa")
    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
