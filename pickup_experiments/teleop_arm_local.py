#!/usr/bin/env python3
"""
Teleop interativo do braco direito do G1 no SIM REAL (run_sim_visible.py), localhost.

Controla as 7 juntas do braco (22-28) por comandos de texto e ABRE/FECHA a mao
com UM comando so (nao precisa mexer em cada junta da mao). Uma thread fica
streamando o estado atual a 100Hz pra segurar o robo de pe.

Uso (com run_sim_visible.py rodando):
    conda activate g1
    python pickup_experiments/teleop_arm_local.py

Comandos (digite e Enter):
    25 0.5       -> seta a junta 25 (cotovelo) em 0.5 rad
    25 +0.1      -> incrementa a junta 25 em +0.1   (tambem aceita -0.1)
    close | c    -> FECHA a mao (comando unico)
    open  | o    -> ABRE a mao
    start        -> vai pra pose 'start' do ik_poses.json
    reach        -> vai pra pose 'reach' do ik_poses.json
    home         -> zera o braco
    p            -> imprime as juntas atuais
    q            -> sair

Juntas: 22=ombro_pitch 23=ombro_roll 24=ombro_yaw 25=cotovelo
        26=punho_roll 27=punho_pitch 28=punho_yaw
"""
import zmq
import json
import time
import threading
from pathlib import Path

HERE = Path(__file__).parent
ADDR = "tcp://127.0.0.1:6001"
ARM_IDS = [22, 23, 24, 25, 26, 27, 28]

HAND_OPEN = {i: 0.0 for i in range(7)}
HAND_CLOSE = {0: 0.0, 1: -1.5, 2: -1.5, 3: 1.5, 4: 1.5, 5: 1.5, 6: 1.5}

state = {
    "arm": {i: 0.0 for i in ARM_IDS},
    "hand": dict(HAND_OPEN),
    "hand_kp": 20.0,
    "running": True,
}
lock = threading.Lock()


def build_body(arm, leg_kp=120.0, leg_kd=2.5, arm_kp=200.0, arm_kd=4.0):
    out = []
    for i in range(29):
        if i in arm:
            out.append({"idx": i, "q": arm[i], "kp": arm_kp, "kd": arm_kd})
        else:
            out.append({"idx": i, "q": 0.0, "kp": leg_kp, "kd": leg_kd})
    return out


def streamer(sock):
    while state["running"]:
        with lock:
            bm = build_body(state["arm"])
            hm = [{"idx": i, "q": state["hand"].get(i, 0.0),
                   "kp": state["hand_kp"], "kd": 2.0} for i in range(7)]
        sock.send_string(json.dumps({"body_motors": bm, "right_hand": hm}))
        time.sleep(0.01)


def main():
    poses = {}
    pf = HERE / "ik_poses.json"
    if pf.exists():
        poses = json.loads(pf.read_text())

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(ADDR)
    print(f"[teleop] conectado em {ADDR}")
    time.sleep(0.3)

    t = threading.Thread(target=streamer, args=(sock,), daemon=True)
    t.start()
    print(__doc__.split("Comandos")[1] if "Comandos" in __doc__ else "")
    print(">> robo segurando standing. Comece a comandar.\n")

    def set_pose(name):
        if name not in poses:
            print(f"  (pose '{name}' nao existe no ik_poses.json)"); return
        with lock:
            for k, v in poses[name].items():
                state["arm"][int(k)] = float(v)
        print(f"  -> pose {name} aplicada")

    while True:
        try:
            cmd = input("teleop> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd:
            continue
        if cmd in ("q", "quit", "exit"):
            break
        elif cmd in ("c", "close"):
            with lock:
                state["hand"] = dict(HAND_CLOSE); state["hand_kp"] = 25.0
            print("  -> MAO FECHADA")
        elif cmd in ("o", "open"):
            with lock:
                state["hand"] = dict(HAND_OPEN); state["hand_kp"] = 20.0
            print("  -> mao aberta")
        elif cmd == "home":
            with lock:
                state["arm"] = {i: 0.0 for i in ARM_IDS}
            print("  -> braco zerado")
        elif cmd in ("start", "reach", "lift"):
            set_pose(cmd)
        elif cmd == "p":
            with lock:
                print("  juntas:", {i: round(state["arm"][i], 3) for i in ARM_IDS},
                      "| mao:", "fechada" if state["hand"] == HAND_CLOSE else "aberta")
        else:
            parts = cmd.split()
            if len(parts) == 2 and parts[0].isdigit():
                j = int(parts[0]); val = parts[1]
                if j not in ARM_IDS:
                    print("  junta invalida (use 22-28)"); continue
                try:
                    f = float(val)
                except ValueError:
                    print("  valor invalido"); continue
                with lock:
                    if val[0] in "+-":          # incremento relativo
                        state["arm"][j] += f
                    else:                        # valor absoluto
                        state["arm"][j] = f
                    print(f"  junta {j} = {state['arm'][j]:.3f}")
            else:
                print("  comando nao reconhecido (ex: '25 +0.1', 'close', 'p', 'q')")

    state["running"] = False
    time.sleep(0.1)
    sock.close(); ctx.term()
    print("[teleop] encerrado")


if __name__ == "__main__":
    main()
