"""
Enviador de ações via ZMQ para simulador remoto.
Usado por scripts de inferência em modo distribuído.
"""
import zmq
import json
import numpy as np


class ActionSenderZMQ:
    """Envia ações para simulador remoto via ZMQ."""

    def __init__(self, remote_ip: str, port: int = 6001, verbose: bool = False):
        """
        Args:
            remote_ip: IP do simulador remoto
            port: porta ZMQ (default: 6001)
            verbose: se True, printa debug logs
        """
        self.remote_ip = remote_ip
        self.port = port
        self.verbose = verbose

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.connect(f"tcp://{remote_ip}:{port}")

        print(f"[ActionSenderZMQ] Conectado a {remote_ip}:{port}")

    def send_action(self, action_dict: dict, body_motor_indices=None, left_hand_indices=None, right_hand_indices=None,
                    body_kp: float = 50.0, body_kd: float = 1.0):
        """
        Envia ação para simulador remoto.

        Args:
            action_dict: dicionário com ações (ex: {"shoulder_pitch.q": 0.5, ...})
            body_motor_indices: dict mapeando nome do motor → índice (ex: {"shoulder_pitch": 0, ...})
            left_hand_indices: dict mapeando nome do motor da mão esquerda → índice
            right_hand_indices: dict mapeando nome do motor da mão direita → índice
        """
        try:
            payload = {}

            # Body motors
            if body_motor_indices:
                body_motors = []
                for name, idx in body_motor_indices.items():
                    key = f"{name}.q"
                    if key in action_dict:
                        body_motors.append({
                            "idx": int(idx),
                            "q": float(action_dict[key]),
                            "kp": float(action_dict.get(f"{name}.kp", body_kp)),
                            "kd": float(action_dict.get(f"{name}.kd", body_kd)),
                            "tau": float(action_dict.get(f"{name}.tau", 0.0)),
                        })
                if body_motors:
                    payload["body_motors"] = body_motors

            # Left hand
            if left_hand_indices:
                left_hand = []
                for name, idx in left_hand_indices.items():
                    key = f"{name}.q"
                    if key in action_dict:
                        left_hand.append({
                            "idx": int(idx),
                            "q": float(action_dict[key]),
                            "kp": float(action_dict.get(f"{name}.kp", 20.0)),
                            "kd": float(action_dict.get(f"{name}.kd", 1.0)),
                            "tau": float(action_dict.get(f"{name}.tau", 0.0)),
                        })
                if left_hand:
                    payload["left_hand"] = left_hand

            # Right hand
            if right_hand_indices:
                right_hand = []
                for name, idx in right_hand_indices.items():
                    key = f"{name}.q"
                    if key in action_dict:
                        right_hand.append({
                            "idx": int(idx),
                            "q": float(action_dict[key]),
                            "kp": float(action_dict.get(f"{name}.kp", 20.0)),
                            "kd": float(action_dict.get(f"{name}.kd", 1.0)),
                            "tau": float(action_dict.get(f"{name}.tau", 0.0)),
                        })
                if right_hand:
                    payload["right_hand"] = right_hand

            if payload:
                self.sock.send_string(json.dumps(payload))
                if self.verbose:
                    print(f"[ActionSenderZMQ] ✓ Ação enviada")

        except Exception as e:
            print(f"[ActionSenderZMQ] Erro ao enviar ação: {e}")

    def close(self):
        """Fecha conexão ZMQ."""
        self.sock.close()
        self.ctx.term()
