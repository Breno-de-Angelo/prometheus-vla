#!/usr/bin/env python3
"""
Desenha por pontos a trajetoria CARTESIANA da mao direita do G1.

Reconstroi a mesma sequencia do test_pick_cup_optimized.py
(HOME -> REACH -> LIFT, com os mesmos sub-passos), roda forward
kinematics em cada ponto pra obter a posicao (X,Y,Z) da mao, e
plota os pontos em 3D junto com a posicao do copo.

Uso:
    conda activate g1
    python pickup_experiments/plot_hand_trajectory.py            # salva PNG
    python pickup_experiments/plot_hand_trajectory.py --show     # abre janela
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import mujoco
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = Path(__file__).parent
XML_PATH = HERE.parent / "unitree-g1-mujoco/assets/scene_43dof.xml"
POSES_PATH = HERE / "ik_poses.json"
CUP_POS = np.array([0.35, -0.05, 0.80])

ARM_IDS = [22, 23, 24, 25, 26, 27, 28]  # ombro/cotovelo/punho direito (idx de motor)

# Motor idx -> nome da junta (pra resolver o endereco correto em qpos)
MOTOR_TO_JOINT = {
    22: "right_shoulder_pitch_joint",
    23: "right_shoulder_roll_joint",
    24: "right_shoulder_yaw_joint",
    25: "right_elbow_joint",
    26: "right_wrist_roll_joint",
    27: "right_wrist_pitch_joint",
    28: "right_wrist_yaw_joint",
}

# Cadeia de corpos do braco direito (ombro -> ... -> ponta do dedo)
ARM_CHAIN = [
    "right_shoulder_pitch_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_yaw_link",
    "right_hand_middle_1_link",
]


def lerp_arm(q_from, q_to, t):
    keys = set(q_from) | set(q_to)
    return {k: q_from.get(k, 0.0) + (q_to.get(k, 0.0) - q_from.get(k, 0.0)) * t
            for k in keys}


def build_phase_points():
    """Mesma sequencia de poses do test_pick_cup_optimized.py."""
    poses = json.loads(POSES_PATH.read_text())
    REACH = {int(k): v for k, v in poses["reach"].items()}
    LIFT = {int(k): v for k, v in poses["lift"].items()}
    HOME = {i: 0.0 for i in range(22, 29)}

    pts = [("1-Home", HOME)]
    for i in range(1, 6):                       # approach gradual
        t = i / 5.0
        pts.append((f"approach {t:.1f}", lerp_arm(HOME, REACH, t)))
    pts.append(("3-Reach", REACH))              # grasp/tighten ficam na REACH
    for i in range(1, 6):                       # lift gradual
        t = i / 5.0
        pts.append((f"lift {t:.1f}", lerp_arm(REACH, LIFT, t)))
    return pts


def motor_qpos_adr(model):
    """Mapa motor_idx -> endereco em qpos, resolvido pelo nome da junta."""
    adr = {}
    for midx, jname in MOTOR_TO_JOINT.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        adr[midx] = model.jnt_qposadr[jid]
    return adr


def arm_chain_positions(model, data, arm_dict, chain_ids, qadr):
    """Aplica a pose do braco e retorna a posicao (X,Y,Z) de cada link da cadeia."""
    mujoco.mj_resetData(model, data)
    for midx, q in arm_dict.items():
        data.qpos[qadr[midx]] = q
    mujoco.mj_forward(model, data)
    return np.array([data.xpos[bid].copy() for bid in chain_ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="abre janela interativa")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    chain_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                 for n in ARM_CHAIN]
    qadr = motor_qpos_adr(model)

    phases = build_phase_points()
    labels = [name for name, _ in phases]
    # chains[i] = posicoes dos links do braco no waypoint i
    chains = np.array([arm_chain_positions(model, data, arm, chain_ids, qadr)
                       for _, arm in phases])
    tip = chains[:, -1, :]  # ponta da mao em cada waypoint

    print("Ponta da mao (X, Y, Z) por waypoint:")
    for name, p in zip(labels, tip):
        print(f"  {name:14s} -> [{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}]  "
              f"dist_copo={np.linalg.norm(p - CUP_POS)*100:.1f}cm")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    n = len(chains)
    cmap = plt.cm.viridis
    # Desenha o braco inteiro (ombro->mao) em cada waypoint
    for i, chain in enumerate(chains):
        color = cmap(i / max(n - 1, 1))
        ax.plot(chain[:, 0], chain[:, 1], chain[:, 2],
                "-o", color=color, ms=4, lw=1.5, alpha=0.8)

    # Trajetoria da ponta da mao em destaque
    ax.plot(tip[:, 0], tip[:, 1], tip[:, 2], "--", color="black", lw=1, alpha=0.6)
    for name, p in zip(labels, tip):
        ax.text(p[0], p[1], p[2], name, fontsize=6)

    ax.scatter(*CUP_POS, c="red", marker="*", s=350, label="copo", depthshade=False)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Trajetoria do braco direito do G1 (ombro->mao, por pontos)\n"
                 "cor = ordem do waypoint (roxo=inicio, amarelo=fim)")
    ax.legend()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n - 1))
    fig.colorbar(sm, ax=ax, label="ordem do waypoint", shrink=0.6)

    if args.show:
        plt.show()
    else:
        out = HERE / "hand_trajectory.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    main()
