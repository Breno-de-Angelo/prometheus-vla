#!/usr/bin/env python3
"""
Replay da trajetoria de pega (gravada por `pick_from_grasp.py --record`) pelo MESMO
canal que a VLA usa: ActionSenderZMQ -> ZMQ 6001 -> ActionReceiver -> sim (Mori).

E uma "VLA falsa": em vez da saida do modelo pi05d, stream-a a nossa trajetoria
conhecida-boa. Como ja conhecemos o resultado dessas acoes (o copo e pego e devolvido),
se o resultado no sim DIFERIR do pick direto, o transporte/formato do sender tem bug.

IMPORTANTE - escopo das acoes (nosso pick NAO e so braco):
  - braco dir 22-28 com kp=200      (no pi05d hoje vai kp~50)
  - mao direita fechando (grip)      (pi05d hoje manda "sem maos")
  - pernas 0-14 travadas kp=120      (pi05d hoje nao manda perna)
  - reset_cup (pin do copo)          (sim-only; a VLA real nunca manda)
  Por padrao manda TUDO (reproduz o pick). --arm-only imita o pi05d de hoje
  (so braco) -> o braco se move igual, mas o copo NAO e pego (esperado, nao e bug).

Uso:
  # LOCAL (smoke test, sim na mesma maquina):
  python pickup_experiments/replay_via_sender.py --traj pickup_experiments/pick_trajectory.json --remote 127.0.0.1
  # de ATENAS (rede real) apontando pro Mori:
  python pickup_experiments/replay_via_sender.py --traj pick_trajectory.json --remote 192.168.15.111
"""
import argparse
import json
import sys
import time
from pathlib import Path

# usa EXATAMENTE o mesmo sender que o init_lerobot_inference_pi05d_v2.py usa
sys.path.insert(0, str(Path(__file__).parent.parent / "lerobot-ext"))
from action_sender_zmq import ActionSenderZMQ


def frame_to_action_dict(frame):
    """Reconstroi (action_dict + mapas de indice) a partir do payload gravado, pra
    ActionSenderZMQ REMONTAR o payload (assim testamos a construcao dele, nao so o envio)."""
    ad = {}
    body_idx, rh_idx, lh_idx = {}, {}, {}
    for m in frame.get("body_motors", []):
        n = f"m{m['idx']}"
        body_idx[n] = int(m["idx"])
        ad[f"{n}.q"] = m["q"]; ad[f"{n}.kp"] = m.get("kp", 50.0); ad[f"{n}.kd"] = m.get("kd", 1.0)
    for m in frame.get("right_hand", []):
        n = f"rh{m['idx']}"
        rh_idx[n] = int(m["idx"])
        ad[f"{n}.q"] = m["q"]; ad[f"{n}.kp"] = m.get("kp", 20.0); ad[f"{n}.kd"] = m.get("kd", 1.0)
    for m in frame.get("left_hand", []):
        n = f"lh{m['idx']}"
        lh_idx[n] = int(m["idx"])
        ad[f"{n}.q"] = m["q"]; ad[f"{n}.kp"] = m.get("kp", 20.0); ad[f"{n}.kd"] = m.get("kd", 1.0)
    return ad, body_idx, rh_idx, lh_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="JSON gravado por pick_from_grasp.py --record")
    ap.add_argument("--remote", default="127.0.0.1",
                    help="IP do sim (Mori). Local=127.0.0.1; de Atenas->Mori=192.168.15.111")
    ap.add_argument("--port", type=int, default=6001)
    ap.add_argument("--arm-only", action="store_true",
                    help="manda SO o braco (15-28), como o pi05d de hoje -> copo NAO sera pego")
    ap.add_argument("--no-hand", action="store_true", help="nao manda as maos")
    ap.add_argument("--no-cup-reset", action="store_true", help="nao manda reset_cup (tira o pin)")
    args = ap.parse_args()

    data = json.loads(Path(args.traj).read_text())
    hz = data.get("hz", 100)
    frames = data["frames"]
    mode = "ARM-ONLY (imita pi05d de hoje)" if args.arm_only else "COMPLETO (braco+mao+pernas+pin)"
    print(f"[replay] {len(frames)} frames @ {hz}Hz ({len(frames)/hz:.1f}s) -> {args.remote}:{args.port} | modo: {mode}")

    sender = ActionSenderZMQ(args.remote, port=args.port, verbose=False)
    time.sleep(0.5)
    dt = 1.0 / hz
    sent = 0
    try:
        for k, fr in enumerate(frames):
            ad, body_idx, rh_idx, lh_idx = frame_to_action_dict(fr)
            if args.arm_only:
                body_idx = {n: i for n, i in body_idx.items() if 15 <= i <= 28}  # so juntas do braco
                rh_idx = lh_idx = {}
            if args.no_hand:
                rh_idx = lh_idx = {}
            reset = bool(fr.get("reset_cup")) and not args.no_cup_reset and not args.arm_only
            sender.send_action(ad, body_motor_indices=body_idx,
                               left_hand_indices=lh_idx or None,
                               right_hand_indices=rh_idx or None,
                               reset_cup=reset)
            sent += 1
            time.sleep(dt)
            if k % 200 == 0:
                print(f"[replay] frame {k}/{len(frames)}")
    finally:
        sender.close()
        print(f"[replay] fim ({sent} frames enviados)")


if __name__ == "__main__":
    main()
