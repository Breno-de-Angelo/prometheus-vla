#!/usr/bin/env python3
"""
Replay de uma trajetoria GRAVADA na teleoperacao real (dataset LeRobot
Mrwlker/pick_up_the_cup3) dentro do SIM REAL (run_sim_visible.py), localhost.

A teleop real pega o copo com o BRACO/MAO ESQUERDA. O dataset grava `action`
com 28 juntas (so bracos + maos). Mapeamento p/ os motores do sim:

  ds 0-6   = braco ESQ (Lshoulder p/r/y, Lelbow, Lwrist r/p/y)  -> body_motors 15-21
  ds 7-13  = braco DIR                                          -> body_motors 22-28
  ds 14-20 = mao ESQ  (thumb0,1,2, mid0,1, idx0,1)              -> left_hand 0-6 (mesma ordem)
  ds 21-27 = mao DIR  (thumb0,1,2, idx0,1, mid0,1)              -> right_hand 0-6 (idx/mid TROCADOS!)

Pernas/cintura (motores 0-14) seguram em pe (q=0, kp alto).
Streama na FPS do dataset (30Hz). No fim segura a ultima pose.

Uso (com run_sim_visible.py rodando):
    conda activate g1
    python pickup_experiments/replay_dataset.py [--ep N] [--speed 1.0] [--loop]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zmq

HERE = Path(__file__).parent
DATASETS = {
    "left": HERE.parent / "datasets/pick_up_the_cup3_ref/data.parquet",
    "right": HERE.parent / "datasets/pick_up_the_cup_right_ref/data.parquet",
}
ADDR = "tcp://127.0.0.1:6001"

# indices no vetor de action (28)
DS_LARM = list(range(0, 7))    # -> body_motors 15..21
DS_RARM = list(range(7, 14))   # -> body_motors 22..28
DS_LHAND = list(range(14, 21)) # -> left_hand 0..6 (ordem igual ao sim)
# mao dir: sim[0,1,2]=ds21,22,23 (thumb); sim[3,4]=ds26,27 (mid); sim[5,6]=ds24,25 (idx)
DS_RHAND_TO_SIM = [21, 22, 23, 26, 27, 24, 25]


def build_body(larm, rarm, torso=0.0, leg_kp=120.0, leg_kd=2.5, arm_kp=200.0, arm_kd=8.0):
    import math
    out = []
    for i in range(29):
        if 15 <= i <= 21:
            q = larm[i - 15]; kp, kd = arm_kp, arm_kd
        elif 22 <= i <= 28:
            q = rarm[i - 22]; kp, kd = arm_kp, arm_kd
        elif i == 14:                      # waist_pitch -> inclina o tronco pra frente
            q = math.radians(torso); kp, kd = arm_kp, arm_kd
        else:
            q = 0.0; kp, kd = leg_kp, leg_kd
        out.append({"idx": i, "q": float(q), "kp": kp, "kd": kd})
    return out


def build_hand(vals7, kp=20.0, kd=1.0):
    return [{"idx": i, "q": float(vals7[i]), "kp": kp, "kd": kd} for i in range(7)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=list(DATASETS), default="right",
                    help="qual dataset: 'right' (mao direita) ou 'left' (mao esquerda)")
    ap.add_argument("--ep", type=int, default=18, help="indice do episodio")
    ap.add_argument("--speed", type=float, default=1.0, help="multiplicador de velocidade")
    ap.add_argument("--loop", action="store_true", help="repetir em loop")
    ap.add_argument("--hand-kp", type=float, default=20.0)
    ap.add_argument("--arm-kp", type=float, default=200.0,
                    help="kp do braco; alto reduz o sag (sem compensacao de gravidade no sim)")
    ap.add_argument("--close-frame", type=int, default=-1,
                    help="força fechar a mao a partir deste frame (-1 = usa o do dataset, ~97)")
    ap.add_argument("--author-grasp", action="store_true",
                    help="fase final autorada: congela o braco no copo, fecha a mao e levanta reto")
    ap.add_argument("--lift-rad", type=float, default=0.5,
                    help="quanto subir o ombro (rad) no lift autorado")
    ap.add_argument("--torso", type=float, default=0.0,
                    help="inclinacao do tronco (waist pitch) em graus, + = pra frente. "
                         "0 = robo reto (o dataset nao grava a cintura).")
    args = ap.parse_args()

    df = pd.read_parquet(DATASETS[args.data])
    e = df[df["episode_index"] == args.ep].reset_index(drop=True)
    if len(e) < 2:
        raise SystemExit(f"episodio {args.ep} tem {len(e)} frames (escolha outro com --ep)")
    A = np.stack(e["action"].values)
    fps = 30.0
    dt = 1.0 / (fps * args.speed)
    print(f"[replay] ep{args.ep}: {len(A)} frames @ {fps}fps  (speed x{args.speed}, {len(A)/fps:.1f}s reais)")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(ADDR)
    print(f"[replay] conectado em {ADDR}")
    time.sleep(0.5)

    HAND_CLOSE = [0.0, -1.5, -1.5, 1.5, 1.5, 1.5, 1.5]
    HAND_OPEN = [0.0] * 7

    def send_frame(a, fi=None):
        larm = a[DS_LARM]
        rarm = a[DS_RARM]
        lhand = a[DS_LHAND]
        # fecha a mao direita mais cedo se --close-frame setado
        if args.close_frame >= 0 and fi is not None and fi >= args.close_frame:
            rhand = np.array(HAND_CLOSE)
        else:
            rhand = a[DS_RHAND_TO_SIM]
        msg = {
            "body_motors": build_body(larm, rarm, torso=args.torso, arm_kp=args.arm_kp),
            "left_hand": build_hand(lhand, kp=args.hand_kp),
            "right_hand": build_hand(rhand, kp=args.hand_kp),
        }
        sock.send_string(json.dumps(msg))

    def send_custom(rarm, rhand, lhand, secs):
        bm = build_body(A[0][DS_LARM], rarm, torso=args.torso, arm_kp=args.arm_kp)
        hm_r = build_hand(rhand, kp=args.hand_kp)
        hm_l = build_hand(lhand, kp=args.hand_kp)
        msg = json.dumps({"body_motors": bm, "left_hand": hm_l, "right_hand": hm_r})
        for _ in range(int(secs * 100)):
            sock.send_string(msg)
            time.sleep(0.01)

    # segura a 1a pose 2s pra estabilizar antes de comecar o movimento
    print("[replay] indo pra pose inicial (2s)...")
    for _ in range(int(2 * fps)):
        send_frame(A[0]); time.sleep(dt)

    def send_reset():
        sock.send_string(json.dumps({"reset_cup": True}))

    if args.author_grasp:
        cf = args.close_frame if args.close_frame >= 0 else 88
        grasp_arm = np.array(A[cf][DS_RARM], float)     # pose de pega (grip no copo)
        above_arm = grasp_arm.copy(); above_arm[0] -= args.lift_rad  # ombro erguido = mao acima
        lhand0 = A[0][DS_LHAND]
        start_arm = np.array(A[0][DS_RARM], float)

        def lerp(a, b, rhand, secs, steps=40):
            for i in range(1, steps + 1):
                send_custom(list(a + (b - a) * (i / steps)), rhand, lhand0, secs=secs / steps)

        attempt = 0
        while True:
            attempt += 1
            print(f"\n===== TENTATIVA {attempt} =====")
            print("[replay] reset do copo (pra posicao das setas) + braco acima")
            send_reset()
            # leva o braco pra 'acima' com mao aberta enquanto o copo assenta
            lerp(start_arm, above_arm, HAND_OPEN, secs=2.0)
            send_reset()  # garante copo no home depois de assentar
            send_custom(list(above_arm), HAND_OPEN, lhand0, secs=1.0)
            print("[replay] DESCE (aberta) -> FECHA -> LEVANTA")
            lerp(above_arm, grasp_arm, HAND_OPEN, secs=1.8)
            send_custom(list(grasp_arm), HAND_CLOSE, lhand0, secs=1.5)
            lerp(grasp_arm, above_arm, HAND_CLOSE, secs=1.8)
            send_custom(list(above_arm), HAND_CLOSE, lhand0, secs=2.5)
            if not args.loop:
                break
            print("[replay] (ajuste o copo com as setas; nova tentativa em 2s)")
            send_custom(list(above_arm), HAND_OPEN, lhand0, secs=2.0)
    else:
        while True:
            for k in range(len(A)):
                send_frame(A[k], fi=k)
                if k % 30 == 0:
                    print(f"  frame {k}/{len(A)}  (t={k/fps:.1f}s)")
                time.sleep(dt)
            print("[replay] fim do episodio; segurando ultima pose 3s")
            for _ in range(int(3 * fps)):
                send_frame(A[-1], fi=len(A)); time.sleep(dt)
            if not args.loop:
                break

    sock.close(); ctx.term()
    print("[replay] encerrado")


if __name__ == "__main__":
    main()
