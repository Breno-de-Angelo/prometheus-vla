"""
Receptor de ações (body motors) do Atenas via ZMQ.
Roda no Mori e injeta ações no simulator.
"""
import zmq
import json
import threading
import time


class ActionReceiver:
    """Recebe comandos de ações do Atenas (corpo + mãos) via ZMQ."""

    def __init__(self, simulator, port=6001, verbose=False):
        """
        Args:
            simulator: instância de BaseSim
            port: porta ZMQ para receber ações (default: 6001)
            verbose: se True, printa debug logs
        """
        self.simulator = simulator
        self.port = port
        self.verbose = verbose

        # ZMQ setup
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PULL)
        self.sock.bind(f"tcp://127.0.0.1:{port}")

        # Thread control
        self.running = False
        self.thread = None

        print(f"[ActionReceiver] Aguardando ações em tcp://127.0.0.1:{port}")

    def start(self):
        """Inicia thread de recepção de ações."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        if self.verbose:
            print("[ActionReceiver] Thread iniciada")

    def stop(self):
        """Para thread de recepção."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.sock.close()
        self.ctx.term()

    def _loop(self):
        """Loop de recepção de ações."""
        while self.running:
            try:
                msg = self.sock.recv_string(zmq.NOBLOCK)
                payload = json.loads(msg)
                self._inject_action(payload)
            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                if self.running:
                    print(f"[ActionReceiver] Erro: {e}")

    def _inject_action(self, payload: dict):
        """
        Injeta ação no simulator.

        Formato esperado:
        {
            "body_motors": [{"idx": 0, "q": 0.5, "kp": 50, "kd": 1}, ...],
            "left_hand": [{"idx": 0, "q": 0.1, "kp": 20, "kd": 1}, ...],
            "right_hand": [{"idx": 0, "q": 0.1, "kp": 20, "kd": 1}, ...],
        }
        """
        try:
            # Body motors
            if "body_motors" in payload:
                for motor_cmd in payload["body_motors"]:
                    idx = motor_cmd["idx"]
                    if idx < len(self.simulator.unitree_bridge.low_cmd.motor_cmd):
                        self.simulator.unitree_bridge.low_cmd.motor_cmd[idx].q = motor_cmd["q"]
                        self.simulator.unitree_bridge.low_cmd.motor_cmd[idx].kp = motor_cmd.get("kp", 50.0)
                        self.simulator.unitree_bridge.low_cmd.motor_cmd[idx].kd = motor_cmd.get("kd", 1.0)
                        self.simulator.unitree_bridge.low_cmd.motor_cmd[idx].tau = motor_cmd.get("tau", 0.0)

            # Left hand
            if "left_hand" in payload:
                for motor_cmd in payload["left_hand"]:
                    idx = motor_cmd["idx"]
                    if idx < len(self.simulator.unitree_bridge.left_hand_cmd.motor_cmd):
                        self.simulator.unitree_bridge.left_hand_cmd.motor_cmd[idx].q = motor_cmd["q"]
                        self.simulator.unitree_bridge.left_hand_cmd.motor_cmd[idx].kp = motor_cmd.get("kp", 20.0)
                        self.simulator.unitree_bridge.left_hand_cmd.motor_cmd[idx].kd = motor_cmd.get("kd", 1.0)
                        self.simulator.unitree_bridge.left_hand_cmd.motor_cmd[idx].tau = motor_cmd.get("tau", 0.0)

            # Right hand
            if "right_hand" in payload:
                for motor_cmd in payload["right_hand"]:
                    idx = motor_cmd["idx"]
                    if idx < len(self.simulator.unitree_bridge.right_hand_cmd.motor_cmd):
                        self.simulator.unitree_bridge.right_hand_cmd.motor_cmd[idx].q = motor_cmd["q"]
                        self.simulator.unitree_bridge.right_hand_cmd.motor_cmd[idx].kp = motor_cmd.get("kp", 20.0)
                        self.simulator.unitree_bridge.right_hand_cmd.motor_cmd[idx].kd = motor_cmd.get("kd", 1.0)
                        self.simulator.unitree_bridge.right_hand_cmd.motor_cmd[idx].tau = motor_cmd.get("tau", 0.0)

            if self.verbose:
                print(f"[ActionReceiver] ✓ Ação injetada")

        except Exception as e:
            print(f"[ActionReceiver] Erro ao injetar ação: {e}")
