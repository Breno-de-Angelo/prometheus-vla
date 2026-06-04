#!/usr/bin/env python3
"""
Envia a trajetória calculada com IK via ZMQ.

Lê os waypoints de trajectory_ik.json e envia cada um ao simulator.
"""

import zmq
import json
import time
import sys
from pathlib import Path


def main():
    # Carrega a trajetória
    traj_file = Path(__file__).parent / "trajectory_ik.json"

    if not traj_file.exists():
        print(f"❌ Arquivo não encontrado: {traj_file}")
        print("Execute primeiro: python generate_trajectory_ik.py")
        return

    with open(traj_file) as f:
        data = json.load(f)

    waypoints = data["trajectory"]
    print(f"[Traj] ✅ Carregados {len(waypoints)} waypoints")

    # Conecta ao simulator
    print(f"[Traj] Conectando a 192.168.15.111:6001...")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect("tcp://192.168.15.111:6001")
    time.sleep(0.5)

    print("[Traj] ✅ Conectado!")
    print("[Traj] Enviando trajetória...\n")

    # Envia cada waypoint
    for wp_idx, waypoint in enumerate(waypoints):
        qpos = waypoint["qpos"]
        name = waypoint["name"]

        # Cria o payload ZMQ
        # body_motors tem índices 0-28 (29 motores)
        payload = {
            "body_motors": [
                {"idx": i, "q": float(qpos[i]), "kp": 50.0, "kd": 1.0}
                for i in range(min(29, len(qpos)))
            ]
        }

        sock.send_string(json.dumps(payload))

        if wp_idx % 2 == 0:
            print(f"[Waypoint {wp_idx}] {name}: q[22]={qpos[22]:.4f}, q[25]={qpos[25]:.4f}")

        # Envia o mesmo waypoint 5 vezes para dar tempo de se mover
        for _ in range(4):
            time.sleep(0.02)  # 50Hz
            sock.send_string(json.dumps(payload))

    print("\n[Traj] ✅ Trajetória completa enviada!")
    print("[Traj] Observando o braço no simulator...\n")

    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
