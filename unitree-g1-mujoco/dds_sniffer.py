#!/usr/bin/env python3
"""
Sniffer DDS para validar o que o laptop publica nos braços e mãos do G1.

Subscreve (na interface DDS escolhida, default 'lo'):
  - rt/lowcmd            (corpo/braços: 35 motores)  -> kp/q por motor
  - rt/dex3/left/cmd     (mão esquerda: 7 motores)
  - rt/dex3/right/cmd    (mão direita: 7 motores)

Para cada mensagem grava uma linha JSONL com timestamp relativo, o tópico e os
campos (q, kp, kd, tau, mode) de cada motor. No fim imprime um resumo:
  - kp médio/min/max da mão ESQUERDA e DIREITA
  - kp dos motores de braço (índices configuráveis) no 1o frame vs último
  - delta de q (alvo) do braço entre o 1o e o último frame (detecta "salto")

Uso:
  python dds_sniffer.py --iface lo --seconds 20 --out /tmp/dds_sniff.jsonl
"""
import argparse
import json
import time
from collections import defaultdict

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, LowCmd_

# Índices de motor do G1 29DOF que correspondem aos BRAÇOS (shoulder/elbow/wrist).
# left arm: 15-21, right arm: 22-28 (convenção Unitree G1 29dof). Ajustável por --arm-idx.
DEFAULT_LEFT_ARM = list(range(15, 22))
DEFAULT_RIGHT_ARM = list(range(22, 29))


def _motors(msg, n):
    out = []
    for i in range(n):
        m = msg.motor_cmd[i]
        out.append({
            "i": i,
            "q": round(float(m.q), 4),
            "kp": round(float(m.kp), 4),
            "kd": round(float(m.kd), 4),
            "tau": round(float(m.tau), 4),
            "mode": int(m.mode),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="lo", help="interface DDS (lo p/ sim local)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", default="/tmp/dds_sniff.jsonl")
    ap.add_argument("--left-arm", default=",".join(map(str, DEFAULT_LEFT_ARM)))
    ap.add_argument("--right-arm", default=",".join(map(str, DEFAULT_RIGHT_ARM)))
    args = ap.parse_args()

    left_arm = [int(x) for x in args.left_arm.split(",") if x != ""]
    right_arm = [int(x) for x in args.right_arm.split(",") if x != ""]

    ChannelFactoryInitialize(0, args.iface)

    t0 = time.time()
    fh = open(args.out, "w")
    # guarda primeiro/último frame por tópico p/ resumo
    first = {}
    last = {}
    counts = defaultdict(int)

    def log(topic, motors):
        counts[topic] += 1
        rec = {"t": round(time.time() - t0, 4), "topic": topic, "motor_cmd": motors}
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        if topic not in first:
            first[topic] = rec
        last[topic] = rec

    def cb_low(msg):
        log("rt/lowcmd", _motors(msg, 35))

    def cb_left(msg):
        log("rt/dex3/left/cmd", _motors(msg, 7))

    def cb_right(msg):
        log("rt/dex3/right/cmd", _motors(msg, 7))

    sub_low = ChannelSubscriber("rt/lowcmd", LowCmd_)
    sub_low.Init(cb_low, 10)
    sub_left = ChannelSubscriber("rt/dex3/left/cmd", HandCmd_)
    sub_left.Init(cb_left, 10)
    sub_right = ChannelSubscriber("rt/dex3/right/cmd", HandCmd_)
    sub_right.Init(cb_right, 10)

    print(f"[sniffer] iface={args.iface} ouvindo {args.seconds}s -> {args.out}", flush=True)
    end = t0 + args.seconds
    while time.time() < end:
        time.sleep(0.1)

    print("\n========== RESUMO DO SNIFFER ==========", flush=True)
    for topic in ("rt/lowcmd", "rt/dex3/left/cmd", "rt/dex3/right/cmd"):
        print(f"\n[{topic}] msgs={counts[topic]}")
        if topic not in first:
            print("  (nenhuma mensagem recebida)")
            continue
        if "dex3" in topic:
            kps = [m["kp"] for m in last[topic]["motor_cmd"]]
            qs = [m["q"] for m in last[topic]["motor_cmd"]]
            taus = [m["tau"] for m in last[topic]["motor_cmd"]]
            print(f"  ULTIMO frame: kp={kps}")
            print(f"               q ={[round(x,3) for x in qs]}")
            print(f"               tau={taus}")
            fk = [m["kp"] for m in first[topic]["motor_cmd"]]
            print(f"  PRIMEIRO frame kp={fk}")
        else:
            arm = left_arm + right_arm
            f_arm = {m["i"]: m for m in first[topic]["motor_cmd"] if m["i"] in arm}
            l_arm = {m["i"]: m for m in last[topic]["motor_cmd"] if m["i"] in arm}
            print("  idx | kp(1o)  q(1o)   | kp(fim) q(fim)  | dq(salto)")
            for i in arm:
                fa = f_arm.get(i, {})
                la = l_arm.get(i, {})
                dq = (la.get("q", 0.0) - fa.get("q", 0.0))
                print(f"  {i:3d} | {fa.get('kp',0):6.2f} {fa.get('q',0):7.3f} | "
                      f"{la.get('kp',0):6.2f} {la.get('q',0):7.3f} | {dq:+7.3f}")
    print("\n=======================================", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
