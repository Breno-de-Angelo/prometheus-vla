#!/usr/bin/env python3
"""
Teste SUPER simples: apenas abre o cotovelo direito progressivamente.

Se até isso não funcionar, há algo MUITO errado no simulator.
"""

import zmq
import json
import time
import sys


def main():
    print("[Simple] Conectando a 192.168.15.111:6001...")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.connect("tcp://192.168.15.111:6001")
    time.sleep(0.5)

    print("[Simple] ✅ Conectado!\n")
    print("[Simple] Testando: abrindo apenas o COTOVELO DIREITO\n")

    # Cotovelo direito = índice 25
    R_ELBOW = 25

    # Valores de cotovelo: 0 = fechado, 1.5 = aberto
    elbow_values = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
    elbow_names = ["Fechado (0.0)", "Pouco aberto (0.3)", "Meio aberto (0.6)",
                   "Mais aberto (0.9)", "Muito aberto (1.2)", "Totalmente aberto (1.5)"]

    for elbow_val, elbow_name in zip(elbow_values, elbow_names):
        print(f"[Teste] {elbow_name}")

        # Criar payload com TODOS os motores em zero, menos o cotovelo
        payload = {
            "body_motors": [
                {"idx": i, "q": elbow_val if i == R_ELBOW else 0.0, "kp": 50.0, "kd": 1.0}
                for i in range(29)
            ]
        }

        # Enviar 10 vezes (0.2 segundos a 50Hz)
        for _ in range(10):
            sock.send_string(json.dumps(payload))
            time.sleep(0.02)

        print(f"  ✓ Enviado\n")
        time.sleep(0.5)

    print("[Simple] ✅ Teste completo!")
    print("[Simple] Se o cotovelo NÃO mexeu, há problema no simulator.\n")

    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
