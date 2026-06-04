#!/usr/bin/env python3
"""
Trajetória MUITO AGRESSIVA para pegar o copo:
- Shoulder_pitch vai até 1.5 (bem abaixado)
- Dedos fecham com força máxima (1.0)
- Mais tempo em cada waypoint (0.5s em vez de 0.3s)
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
    print("[Aggressive] Conectando a 192.168.15.111:6001...")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect("tcp://192.168.15.111:6001")
    time.sleep(0.5)

    print("[Aggressive] ✅ Conectado!\n")

    # Índices do braço direito
    R_SHOULDER_PITCH = 22
    R_SHOULDER_ROLL = 23
    R_ELBOW = 25

    # Índices da mão direita
    R_THUMB_0 = 0
    R_THUMB_1 = 1
    R_THUMB_2 = 2
    R_INDEX_0 = 3
    R_INDEX_1 = 4
    R_MIDDLE_0 = 5
    R_MIDDLE_1 = 6

    # Trajetória AGRESSIVA: 5 waypoints
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
            "name": "Abaixar braço MUITO (shoulder_pitch=1.5)",
            "body": {
                R_SHOULDER_PITCH: 1.5,  # MUITO abaixado
                R_SHOULDER_ROLL: 0.2,
                R_ELBOW: 1.2,  # Estender bem
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
            "name": "Fechar mão COM FORÇA (todos os dedos=1.0)",
            "body": {
                R_SHOULDER_PITCH: 1.5,  # Manter abaixado
                R_SHOULDER_ROLL: 0.2,
                R_ELBOW: 1.2,
            },
            "hand": {
                R_THUMB_0: 1.0,  # Máximo!
                R_THUMB_1: 1.0,
                R_THUMB_2: 1.0,
                R_INDEX_0: 1.0,
                R_INDEX_1: 1.0,
                R_MIDDLE_0: 1.0,
                R_MIDDLE_1: 1.0,
            }
        },
        {
            "name": "Levantar braço rapidamente (shoulder_pitch=0.0)",
            "body": {
                R_SHOULDER_PITCH: 0.0,  # Levantar rapidamente
                R_SHOULDER_ROLL: 0.2,
                R_ELBOW: 0.8,
            },
            "hand": {
                R_THUMB_0: 1.0,  # Manter fechado
                R_THUMB_1: 1.0,
                R_THUMB_2: 1.0,
                R_INDEX_0: 1.0,
                R_INDEX_1: 1.0,
                R_MIDDLE_0: 1.0,
                R_MIDDLE_1: 1.0,
            }
        },
        {
            "name": "Abrir mão",
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

        # Criar arrays de 29 motores
        body_motors = [
            {"idx": i, "q": waypoint['body'].get(i, 0.0), "kp": 50.0, "kd": 1.0}
            for i in range(29)
        ]

        # Criar arrays de 7 motores da mão
        hand_motors = [
            {"idx": i, "q": waypoint['hand'].get(i, 0.0), "kp": 20.0, "kd": 1.0}
            for i in range(7)
        ]

        # Enviar por MAIS tempo (0.5s = 25 frames a 50Hz)
        for _ in range(25):
            send_action(sock, body_motors, hand_motors)
            time.sleep(0.02)

        print(f"  ✓ Enviado (0.5s)")
        time.sleep(0.5)

    print("\n[Aggressive] ✅ Trajetória agressiva completa!")
    print("[Aggressive] Verificando se pegou e levantou o copo...\n")

    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
