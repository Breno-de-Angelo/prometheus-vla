#!/usr/bin/env python3
"""
Pick AUTORADO a partir da pose de grasp capturada (tecla T -> /tmp/grasp_state.json).

Ideia (acao reversa, do usuario): a pose de grasp (braco onde a mao segura o copo) ja foi
capturada manualmente. A aproximacao e o levantamento sao o REVERSO de erguer o ombro:
  pre-grasp (ombro erguido, mao aberta) -> DESCE ate a pose de grasp -> FECHA a mao -> LEVANTA
  (volta ao pre-grasp, agora segurando).
Repetivel: reseta o copo (R no sim) e roda de novo.

Uso (com run_sim_visible.py rodando, BAND_Z=0.847):
    conda activate g1
    python pickup_experiments/pick_from_grasp.py [--lift 0.45] [--grip 0.6] [--reset]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import zmq
import mujoco

ADDR = "tcp://127.0.0.1:6001"
XML = Path(__file__).parent.parent / "unitree-g1-mujoco/assets/scene_43dof.xml"
RA = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
      "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
      "right_wrist_yaw_joint"]
CLOSE = np.array([0.0, -1.5, -1.5, 1.5, 1.5, 1.5, 1.5])


def compute_pregrasp(q_grasp, dz=0.12):
    """IK: pose do braco com a MAO ~dz ACIMA do grasp (mesmo XY) -> descida vertical."""
    m = mujoco.MjModel.from_xml_path(str(XML)); d = mujoco.MjData(m)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    qadr = [m.jnt_qposadr[jid(n)] for n in RA]
    dofadr = [m.jnt_dofadr[jid(n)] for n in RA]
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_hand_middle_1_link")

    def setq(q):
        mujoco.mj_resetData(m, d); d.qpos[2] = 0.79
        for k, a in enumerate(qadr):
            d.qpos[a] = q[k]
        mujoco.mj_forward(m, d)

    setq(q_grasp); target = d.xpos[hand].copy(); target[2] += dz
    q = q_grasp.copy()
    for _ in range(300):
        setq(q); err = target - d.xpos[hand]
        if np.linalg.norm(err) < 0.004:
            break
        jacp = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, d, jacp, None, hand)
        J = jacp[:, dofadr]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.04 * np.eye(3), err * 4.0) * 0.02
        q = np.clip(q + dq, -3.0, 3.0)
    setq(q); e = np.linalg.norm(target - d.xpos[hand])
    print(f"[pick] pre-grasp IK: mao {dz*100:.0f}cm acima do grasp, err={e*100:.1f}cm")
    return q


def build_body(arm7, leg_kp=120.0, leg_kd=2.5, arm_kp=200.0, arm_kd=4.0):
    out = []
    for i in range(29):
        if 22 <= i <= 28:
            q = float(arm7[i - 22]); kp, kd = arm_kp, arm_kd
        else:
            q = 0.0; kp, kd = leg_kp, leg_kd
        out.append({"idx": i, "q": q, "kp": kp, "kd": kd})
    return out


def build_hand(vals7, kp, kd=1.2):
    return [{"idx": i, "q": float(vals7[i]), "kp": kp, "kd": kd} for i in range(7)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lift-z", type=float, default=0.14, help="altura (m) da mao acima do grasp no pre-grasp/lift")
    ap.add_argument("--grip", type=float, default=0.6, help="fracao de fechamento da mao (0..1)")
    ap.add_argument("--hand-kp", type=float, default=35.0)
    ap.add_argument("--reset", action="store_true", help="reseta o copo no inicio")
    ap.add_argument("--reverse", action="store_true", help="apos pegar, faz o reverso: desce, solta, sobe vazio")
    ap.add_argument("--record", type=str, default=None,
                    help="GRAVA a trajetoria (frames de payload) num JSON em vez de enviar via ZMQ")
    args = ap.parse_args()

    st = json.loads(Path("/tmp/grasp_state.json").read_text())
    q_grasp = np.array([st["arm"][n] for n in RA], float)
    # pre-grasp = pose com a MAO ~lift_z ACIMA do grasp (descida VERTICAL, nao arco) via IK
    pre = compute_pregrasp(q_grasp, dz=args.lift_z)
    # parada ALTA (mao bem acima do copo) p/ a retirada limpa no reverso: nenhum dedo encosta no copo
    park = compute_pregrasp(q_grasp, dz=max(args.lift_z + 0.18, 0.30)) if args.reverse else pre
    open_h = np.zeros(7)
    close_h = CLOSE * args.grip
    print(f"[pick] grasp arm={np.round(q_grasp,3)}")
    print(f"[pick] pre-grasp (mao {args.lift_z*100:.0f}cm acima) ={np.round(pre,3)}")

    HZ = 100
    rec = [] if args.record else None     # modo gravacao: acumula frames em vez de enviar
    ctx = sock = None
    if rec is None:
        ctx = zmq.Context(); sock = ctx.socket(zmq.PUSH); sock.connect(ADDR)
        time.sleep(0.5)

    def _frame(arm7, hand7, reset):
        d_ = {"body_motors": build_body(arm7), "right_hand": build_hand(hand7, args.hand_kp),
              "left_hand": build_hand(np.zeros(7), 20)}
        if reset:
            d_["reset_cup"] = True
        return d_

    def send(arm7, hand7, secs, hz=HZ, reset=False):
        d_ = _frame(arm7, hand7, reset); msg = json.dumps(d_)
        for _ in range(int(secs * hz)):
            if rec is not None:
                rec.append(d_)
            else:
                sock.send_string(msg); time.sleep(1.0 / hz)

    def lerp(a, b, hand7, secs, hz=HZ, reset=False):
        n = int(secs * hz)
        for i in range(1, n + 1):
            send_once(a + (b - a) * (i / n), hand7, reset)
            if rec is None:
                time.sleep(1.0 / hz)

    def send_once(arm7, hand7, reset=False):
        d_ = _frame(arm7, hand7, reset)
        if rec is not None:
            rec.append(d_)
        else:
            sock.send_string(json.dumps(d_))

    # ciclo: AFASTA o braco -> reseta o copo (mao longe) -> aproxima -> fecha -> levanta
    print(f"[pick] 1) vai pra posicao afastada (mao {args.lift_z*100:.0f}cm ACIMA do copo, aberta)")
    send(pre, open_h, 2.5)
    # reset do copo com o braco JA AFASTADO (senao a mao derruba o copo recem-colocado)
    print("[pick] 1b) reseta o copo pro home (braco afastado) - posicao inicial garantida")
    send(pre, open_h, 1.0, reset=True)
    send(pre, open_h, 0.5)
    print("[pick] 2) DESCE VERTICAL ate a pose de grasp (mao aberta)")
    lerp(pre, q_grasp, open_h, 2.5)
    # PINA o copo na mao: reset_cup em TODO frame por 1.2s (forca o cilindro pro home na mao)
    print("[pick] 3) PINA o copo na mao (reset por frame) + assenta")
    send(q_grasp, open_h, 1.2, reset=True)
    send(q_grasp, open_h, 0.4)   # solta o pin, deixa assentar 1 instante
    print("[pick] 4) FECHA a mao no copo (grasp)")
    send(q_grasp, close_h, 2.0)
    print("[pick] 5) LEVANTA VERTICAL (mao fechada)")
    lerp(q_grasp, pre, close_h, 2.5)
    print("[pick] 6) segura no alto 1.5s")
    send(pre, close_h, 1.5)

    if args.reverse:
        # REWIND: desce o copo de volta -> solta -> sobe a mao VAZIA, deixando o copo no HOME exato.
        # CHAVE: o copo fica PINADO no home durante toda a soltura+subida (senao a mao aberta subindo
        # "cata" o copo e o ARREMESSA pra longe). Termina com a mao bem no alto (longe), copo no home.
        print("[pick] 7) REVERSO: desce o copo de volta ate a pose de grasp (mao fechada)")
        lerp(pre, q_grasp, close_h, 2.5)
        print("[pick] 8) REVERSO: abre a mao com o copo PINADO no home")
        send(q_grasp, open_h, 1.2, reset=True)
        print("[pick] 9) REVERSO: sobe a mao VAZIA ate a parada ALTA (copo PINADO no home)")
        lerp(q_grasp, park, open_h, 2.8, reset=True)   # pina ate a mao estar bem longe do copo
        print("[pick] 10) REVERSO: assenta o copo no home com a mao no alto (longe) + confirma")
        send(park, open_h, 1.0, reset=True)            # mao a ~30cm -> copo fixa no home exato
        send(park, open_h, 1.0)                        # solta o pin: mao longe, copo fica no lugar
        print("[pick] fim (copo EXATO no home, mao levantada vazia)")
    else:
        send(pre, close_h, 1.5)
        print("[pick] fim")

    if rec is not None:
        out = {"hz": HZ, "frames": rec}
        Path(args.record).write_text(json.dumps(out))
        print(f"[pick] trajetoria GRAVADA: {len(rec)} frames @ {HZ}Hz -> {args.record}")
    else:
        sock.close(); ctx.term()


if __name__ == "__main__":
    main()
