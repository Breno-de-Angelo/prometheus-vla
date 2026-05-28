#!/usr/bin/env python3
"""
Trajetória COMPLETA para pegar o copo:
1. Abaixa o braço (shoulder pitch)
2. Abre o cotovelo (estende)
3. Fecha a mão (dedos)
4. Levanta (pega o copo)
5. Volta para posição inicial
"""

import zmq
import json
import time
import sys


def send_action(sock, body_motors, hand_motors=None):
    """Envia ação completa (corpo + mão)."""
    payload = {"body_motors": body_motors}

    if hand_motors:
        payload["right_hand"] = hand_motors

    sock.send_string(json.dumps(payload))


def main():
    print("[Pick] Conectando a 192.168.15.111:6001...")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect("tcp://192.168.15.111:6001")
    time.sleep(0.5)

    print("[Pick] ✅ Conectado!\n")

    # Índices do braço direito
    R_SHOULDER_PITCH = 22
    R_SHOULDER_ROLL = 23
    R_ELBOW = 25

    # Índices da mão direita (Dex3)
    R_THUMB_0 = 0
    R_THUMB_1 = 1
    R_THUMB_2 = 2
    R_INDEX_0 = 3
    R_INDEX_1 = 4
    R_MIDDLE_0 = 5
    R_MIDDLE_1 = 6

    # Trajetória: 6 waypoints
    waypoints = [
        {
            "name": "Posição inicial",
            "body": {
                R_SHOULDER_PITCH: 0.0,
                R_SHOULDER_ROLL: 0.2,
                R_ELBOW: 0.3,
            },
            "hand": {
                R_THUMB_0: 0.0,
                R_THUMB_1: 0.0,
                R_THUMB_2: 0.0,
                R_INDEX_0: 0.0,
                R_INDEX_1: 0.0,
                R_MIDDLE_0: 0.0,
                R_MIDDLE_1: 0.0,
            }
        },
        {
            "name": "Braço para frente e abaixado",
            "body": {
                R_SHOULDER_PITCH: 0.6,
                R_SHOULDER_ROLL: 0.2,
                R_ELBOW: 0.8,
            },
            "hand": {
                R_THUMB_0: 0.0,
                R_THUMB_1: 0.0,
                R_THUMB_2: 0.0,
                R_INDEX_0: 0.0,
                R_INDEX_1: 0.0,
                R_MIDDLE_0: 0.0,
                R_MIDDLE_1: 0.0,
            }
        },
        {
            "name": "Descer mais (perto do copo)",
            "body": {
                R_SHOULDER_PITCH: 1.0,
                R_SHOULDER_ROLL: 0.2,
                R_ELBOW: 0.8,
            },
            "hand": {
                R_THUMB_0: 0.0,
                R_THUMB_1: 0.0,
                R_THUMB_2: 0.0,
                R_INDEX_0: 0.0,
                R_INDEX_1: 0.0,
                R_MIDDLE_0: 0.0,
                R_MIDDLE_1: 0.0,
            }
        },
        {
            "name": "Fechar a mão (pegar o copo)",
            "body": {
                R_SHOULDER_PITCH: 1.0,
                R_SHOULDER_ROLL: 0.2,
                R_ELBOW: 0.8,
            },
            "hand": {
                R_THUMB_0: 0.5,
                R_THUMB_1: 0.5,
                R_THUMB_2: 0.5,
                R_INDEX_0: 0.8,
                R_INDEX_1: 0.8,
                R_MIDDLE_0: 0.8,
                R_MIDDLE_1: 0.8,
            }
        },
        {
            "name": "Levantar o copo",
            "body": {
                R_SHOULDER_PITCH: 0.3,
                R_SHOULDER_ROLL: 0.2,
                R_ELBOW: 0.8,
            },
            "hand": {
                R_THUMB_0: 0.5,
                R_THUMB_1: 0.5,
                R_THUMB_2: 0.5,
                R_INDEX_0: 0.8,
                R_INDEX_1: 0.8,
                R_MIDDLE_0: 0.8,
                R_MIDDLE_1: 0.8,
            }
        },
        {
            "name": "Volta para posição inicial",
            "body": {
                R_SHOULDER_PITCH: 0.0,
                R_SHOULDER_ROLL: 0.2,
                R_ELBOW: 0.3,
            },
            "hand": {
                R_THUMB_0: 0.0,
                R_THUMB_1: 0.0,
                R_THUMB_2: 0.0,
                R_INDEX_0: 0.0,
                R_INDEX_1: 0.0,
                R_MIDDLE_0: 0.0,
                R_MIDDLE_1: 0.0,
            }
        }
    ]

    # Enviar cada waypoint
    for wp_idx, waypoint in enumerate(waypoints):
        print(f"\n[Waypoint {wp_idx}] {waypoint['name']}")
        print(f"  Shoulder Pitch: {waypoint['body'][R_SHOULDER_PITCH]:.2f}")
        print(f"  Elbow: {waypoint['body'][R_ELBOW]:.2f}")

        # Criar arrays de 29 motores (tudo em zero, exceto os do braço)
        body_motors = [
            {"idx": i, "q": waypoint['body'].get(i, 0.0), "kp": 50.0, "kd": 1.0}
            for i in range(29)
        ]

        # Criar arrays de 7 motores da mão
        hand_motors = [
            {"idx": i, "q": waypoint['hand'].get(i, 0.0), "kp": 20.0, "kd": 1.0}
            for i in range(7)
        ]

        # Enviar o waypoint 15 vezes (0.3 segundos a 50Hz)
        for _ in range(15):
            send_action(sock, body_motors, hand_motors)
            time.sleep(0.02)

        print(f"  ✓ Enviado (0.3s)")
        time.sleep(0.5)

    print("\n[Pick] ✅ Trajetória completa enviada!")
    print("[Pick] Observando o robô pegar o copo...\n")

    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
