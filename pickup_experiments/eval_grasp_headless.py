#!/usr/bin/env python3
"""
Avaliador HEADLESS do grasp (sem viewer/ZMQ) — replica o control law REAL do
base_sim pra rodar a trajetoria do dataset (ep18, mao direita) e medir se pega o
copo. Parametrizado p/ varrer configs em PARALELO (10 agentes) sem colidir, pois
cada processo tem seu proprio MjModel/MjData.

Control law (igual base_sim.compute_body_torques): torque = kp*(q_alvo-qpos) - kd*qvel.
Free base seguro pela faixa elastica (xfrc no torso_link, ponto [0,0,1], kp_pos=10000).

Config via flags: posicao/escala do copo, altura da mesa, inclinacao do tronco,
kp da mao, ponto de fechamento. Saida: JSON com contatos e elevacao do copo.

Uso:
    python pickup_experiments/eval_grasp_headless.py --cup-x 0.305 --cup-y 0.022 \
        --cup-scale 0.0016 --table-top 0.90 --torso 0 --hand-kp 20 --json
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import mujoco
import pandas as pd

HERE = Path(__file__).parent
XML = HERE.parent / "unitree-g1-mujoco/assets/scene_43dof.xml"
PARQUET = HERE.parent / "datasets/pick_up_the_cup_right_ref/data.parquet"

DS_RARM = list(range(7, 14))       # action -> motores corpo 22..28
DS_RHAND_TO_SIM = [21, 22, 23, 26, 27, 24, 25]   # -> right_hand 0..6 (idx/mid trocados)
HAND_OPEN = np.zeros(7)
HAND_CLOSE = np.array([0.0, -1.5, -1.5, 1.5, 1.5, 1.5, 1.5])

BASE_Z = 0.910
TORSO_BID = None


def make_model(cup_x, cup_y, cup_scale, table_top):
    """Carrega o XML editando escala do copo + posicoes (mesa/copo) num temp XML."""
    txt = XML.read_text()
    # escala do mesh do copo
    import re
    txt = re.sub(r'(<mesh name="cup" file="../cup.stl" scale=")[^"]+(")',
                 rf'\g<1>{cup_scale} {cup_scale} {cup_scale}\g<2>', txt)
    # posicao da mesa: topo = mesa_pos_z + 0.86 (tampo local em 0.835+0.025). topo alvo:
    mesa_z = table_top - 0.86
    txt = re.sub(r'(<body name="mesa" pos="0.45 0 )[^"]+(">)', rf'\g<1>{mesa_z:.3f}\g<2>', txt)
    # posicao do copo
    txt = re.sub(r'(<body name="objeto_customizado" pos=")[^"]+(">)',
                 rf'\g<1>{cup_x:.3f} {cup_y:.3f} {table_top+0.04:.3f}\g<2>', txt)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=str(XML.parent), delete=False) as f:
        f.write(txt); tmp = f.name
    m = mujoco.MjModel.from_xml_path(tmp)
    Path(tmp).unlink()
    return m


def run(cfg):
    m = make_model(cfg["cup_x"], cfg["cup_y"], cfg["cup_scale"], cfg["table_top"])
    d = mujoco.MjData(m)
    m.opt.timestep = 0.004
    torso_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    cup_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "objeto_customizado")
    waist_q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")]

    # estado inicial: free base em Z, upright, tronco inclinado cfg
    d.qpos[2] = BASE_Z
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[waist_q] = np.radians(cfg["torso"])
    mujoco.mj_forward(m, d)

    # trajetoria ep18
    df = pd.read_parquet(PARQUET)
    A = np.stack(df[df["episode_index"] == cfg["ep"]].reset_index(drop=True)["action"].values)
    fc = int(np.argmax((A[:, 24] + A[:, 26]) > (A[:, 24] + A[:, 26]).max() * 0.5))  # frame fechamento

    # alvos: corpo (29) kp/kd, mao dir (7)
    arm_kp, arm_kd, leg_kp, leg_kd = 200.0, 4.0, 120.0, 2.5
    hand_kp, hand_kd = cfg["hand_kp"], 1.0

    def body_targets(a):
        q = np.zeros(29)
        for k, mi in enumerate(range(22, 29)):
            q[mi] = a[DS_RARM[k]]
        q[14] = np.radians(cfg["torso"])  # waist pitch
        return q

    # indices reais de qpos/dof por atuador (robusto)
    act_q = np.array([m.jnt_qposadr[m.actuator_trnid[ai, 0]] for ai in range(m.nu)])
    act_v = np.array([m.jnt_dofadr[m.actuator_trnid[ai, 0]] for ai in range(m.nu)])

    def step_to(q_body, q_rhand, n_steps):
        for _ in range(n_steps):
            tau = np.zeros(m.nu)
            # corpo: atuadores 0..28
            for i in range(29):
                kp, kd = (arm_kp, arm_kd) if (15 <= i <= 28 or i == 14) else (leg_kp, leg_kd)
                tau[i] = kp * (q_body[i] - d.qpos[act_q[i]]) - kd * d.qvel[act_v[i]]
            # mao direita: atuadores 36..42
            for j in range(7):
                ai = 36 + j
                tau[ai] = hand_kp * (q_rhand[j] - d.qpos[act_q[ai]]) - hand_kd * d.qvel[act_v[ai]]
            d.ctrl[:] = tau
            # faixa elastica no torso
            pos = d.xpos[torso_bid]; quat = d.xquat[torso_bid].copy()
            vel = d.cvel[torso_bid]
            f = 10000.0 * (np.array([0, 0, 1.0]) - pos) + 1000.0 * (0 - vel[3:6])
            # torque angular simples (mantem upright)
            import scipy.spatial.transform as sst
            r = sst.Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
            torque = -1000.0 * r.as_rotvec() - 10.0 * vel[0:3]
            d.xfrc_applied[torso_bid] = np.concatenate([f, torque])
            mujoco.mj_step(m, d)

    steps_per = max(1, int(round(250 / 30 / cfg["speed"])))
    cup_z0 = d.xpos[cup_bid][2]
    # settle inicial 1.5s na pose frame0
    qb0 = body_targets(A[0])
    step_to(qb0, HAND_OPEN, int(1.5 * 250))
    cup_z0 = d.xpos[cup_bid][2]
    zmax = cup_z0; contacts = 0; mind = 9.9
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_hand_middle_1_link")
    for fi in range(len(A)):
        # mao: aberta ate o frame de fechamento, depois fecha
        rh = HAND_CLOSE if fi >= fc else np.array([A[fi][k] for k in DS_RHAND_TO_SIM])
        step_to(body_targets(A[fi]), rh, steps_per)
        cz = d.xpos[cup_bid]
        zmax = max(zmax, cz[2])
        contacts = max(contacts, d.ncon)
        mind = min(mind, float(np.linalg.norm(d.xpos[mid] - cz)))
    # hold final
    step_to(body_targets(A[-1]), HAND_CLOSE, int(2 * 250))
    cup_zf = d.xpos[cup_bid][2]
    return {
        "cup_z0": round(cup_z0, 3), "cup_zmax": round(zmax, 3), "cup_zf": round(cup_zf, 3),
        "lift_peak_cm": round((zmax - cup_z0) * 100, 1),
        "lift_net_cm": round((cup_zf - cup_z0) * 100, 1),
        "min_finger_dist_cm": round(mind * 100, 1),
        "grasped": bool((cup_zf - cup_z0) > 0.04 and cup_zf > 0.5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cup-x", type=float, default=0.305)
    ap.add_argument("--cup-y", type=float, default=0.022)
    ap.add_argument("--cup-scale", type=float, default=0.0016)
    ap.add_argument("--table-top", type=float, default=0.90)
    ap.add_argument("--torso", type=float, default=0.0)
    ap.add_argument("--hand-kp", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--ep", type=int, default=18)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cfg = vars(args)
    res = run(cfg)
    res["cfg"] = {k: cfg[k] for k in ("cup_x", "cup_y", "cup_scale", "table_top", "torso", "hand_kp")}
    if args.json:
        print(json.dumps(res))
    else:
        for k, v in res.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
