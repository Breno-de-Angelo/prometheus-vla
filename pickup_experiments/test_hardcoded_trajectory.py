#!/usr/bin/env python3
"""
Trajetória hardcoded para testar se o braço se move no simulator.

Se ESSA trajetória predefinida não funcionar, então o problema
está no action_receiver ou na simulação, NÃO na VLA.

Uso:
    python test_hardcoded_trajectory.py --robot-ip=192.168.15.111
"""

import zmq
import json
import time
import sys
import argparse


def send_action(sock, action_dict):
    """Envia ação via ZMQ (igual ao ActionSenderZMQ)."""
    payload = {
        "body_motors": [
            {"idx": i, "q": action_dict.get(i, 0.0), "kp": 50.0, "kd": 1.0}
            for i in range(29)
        ]
    }
    sock.send_string(json.dumps(payload))
    print(f"[send_action] Enviado: motor[22]={action_dict.get(22, 0.0):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Test hardcoded trajectory")
    parser.add_argument("--robot-ip", default="192.168.15.111", help="IP do Mori")
    parser.add_argument("--port", default=6001, type=int, help="Porta ZMQ")
    args = parser.parse_args()

    print(f"[Test] Conectando a {args.robot_ip}:{args.port}...")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect(f"tcp://{args.robot_ip}:{args.port}")
    time.sleep(0.5)  # Deixa conectar

    print("[Test] ✅ Conectado!")
    print("[Test] Enviando trajetória hardcoded para pegar o copo...\n")

    # Índices do braço direito
    R_SHOULDER_PITCH = 22
    R_SHOULDER_ROLL = 23
    R_SHOULDER_YAW = 24
    R_ELBOW = 25
    R_WRIST_ROLL = 26
    R_WRIST_PITCH = 27
    R_WRIST_YAW = 28

    # Trajetória hardcoded: 5 waypoints
    waypoints = [
        # Waypoint 0: Posição inicial (braço levantado)
        {
            R_SHOULDER_PITCH: 0.0,   # Ombro horizontal
            R_SHOULDER_ROLL: 0.3,    # Um pouco para fora
            R_SHOULDER_YAW: 0.0,     # Neutro
            R_ELBOW: 0.5,            # Cotovelo levemente dobrado
            R_WRIST_PITCH: 0.0,      # Pulso reto
            R_WRIST_ROLL: 0.0,       # Pulso reto
            R_WRIST_YAW: 0.0,        # Pulso reto
        },
        # Waypoint 1: Braço para frente e abaixado (em direção ao copo)
        {
            R_SHOULDER_PITCH: 0.8,   # Abaixar o ombro
            R_SHOULDER_ROLL: 0.3,    # Manter para fora
            R_SHOULDER_YAW: 0.0,     # Neutro
            R_ELBOW: 1.2,            # Estender o cotovelo
            R_WRIST_PITCH: 0.0,      # Pulso reto
            R_WRIST_ROLL: 0.0,       # Pulso reto
            R_WRIST_YAW: 0.0,        # Pulso reto
        },
        # Waypoint 2: Descer mais (pegar o copo)
        {
            R_SHOULDER_PITCH: 1.2,   # Abaixar mais
            R_SHOULDER_ROLL: 0.3,    # Manter para fora
            R_SHOULDER_YAW: 0.0,     # Neutro
            R_ELBOW: 1.2,            # Estender
            R_WRIST_PITCH: 0.0,      # Pulso reto
            R_WRIST_ROLL: 0.0,       # Pulso reto
            R_WRIST_YAW: 0.0,        # Pulso reto
        },
        # Waypoint 3: Levantar o copo
        {
            R_SHOULDER_PITCH: 0.3,   # Levantar ombro
            R_SHOULDER_ROLL: 0.3,    # Manter para fora
            R_SHOULDER_YAW: 0.0,     # Neutro
            R_ELBOW: 1.0,            # Manter estendido
            R_WRIST_PITCH: 0.0,      # Pulso reto
            R_WRIST_ROLL: 0.0,       # Pulso reto
            R_WRIST_YAW: 0.0,        # Pulso reto
        },
        # Waypoint 4: Volta para posição inicial
        {
            R_SHOULDER_PITCH: 0.0,   # Ombro horizontal
            R_SHOULDER_ROLL: 0.3,    # Um pouco para fora
            R_SHOULDER_YAW: 0.0,     # Neutro
            R_ELBOW: 0.5,            # Cotovelo levemente dobrado
            R_WRIST_PITCH: 0.0,      # Pulso reto
            R_WRIST_ROLL: 0.0,       # Pulso reto
            R_WRIST_YAW: 0.0,        # Pulso reto
        },
    ]

    waypoint_names = [
        "Posição inicial",
        "Braço para frente e abaixado",
        "Descer para pegar",
        "Levantar o copo",
        "Volta para inicial",
    ]

    # Enviar cada waypoint
    for wp_idx, (waypoint, name) in enumerate(zip(waypoints, waypoint_names)):
        print(f"\n[Waypoint {wp_idx}] {name}")
        print(f"  ShoulderPitch: {waypoint[R_SHOULDER_PITCH]:.2f}")
        print(f"  Elbow: {waypoint[R_ELBOW]:.2f}")

        # Enviar esse waypoint 10 vezes (50Hz × 0.2s = 10 frames)
        # para dar tempo do braço se mover
        for i in range(10):
            send_action(sock, waypoint)
            time.sleep(0.02)  # 50Hz

        print(f"  ✅ Waypoint {wp_idx} enviado (0.2s)")
        time.sleep(0.3)  # Pausa extra antes do próximo

    print("\n[Test] ✅ Trajetória completa enviada!")
    print("[Test] Se o braço não se mexeu, há um problema no simulator/receiver.")
    print("[Test] Se o braço se mexeu, então a VLA está funcional!\n")

    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
