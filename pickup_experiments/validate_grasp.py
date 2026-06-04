#!/usr/bin/env python3
"""
Validacao de FISICA (sem ZMQ) da trajetoria de pickup.

Replica a lei de controle PD por torque do base_sim.py, roda as mesmas fases
do test_pick_cup_optimized.py (home->reach, fecha mao, aperta, levanta) e mede:
  - altura do copo antes do grasp e no fim do lift
  - numero de contatos mao-direita <-> copo

Uso:
    conda activate g1
    python pickup_experiments/validate_grasp.py            # headless
    python pickup_experiments/validate_grasp.py --view      # com viewer
"""
import json
import argparse
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).parent
XML = HERE.parent / "unitree-g1-mujoco/assets/scene_43dof.xml"

MOTOR_TO_JOINT = {
    22: "right_shoulder_pitch_joint", 23: "right_shoulder_roll_joint",
    24: "right_shoulder_yaw_joint",   25: "right_elbow_joint",
    26: "right_wrist_roll_joint",     27: "right_wrist_pitch_joint",
    28: "right_wrist_yaw_joint",
}
HAND_OPEN = {i: 0.0 for i in range(7)}
HAND_CLOSE = {0: 0.7, 1: 0.7, 2: 0.7, 3: 0.9, 4: 0.9, 5: 0.9, 6: 0.9}


def build_indices(model):
    body, rhand = [], []
    parts = ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
    for i in range(model.njnt):
        name = model.joint(i).name
        if any(p in name for p in parts):
            body.append(i)
        elif "right_hand" in name:
            rhand.append(i)
    return np.array(body), np.array(rhand)


def lerp(a, b, t):
    return {k: a.get(k, 0.0) + (b.get(k, 0.0) - a.get(k, 0.0)) * t for k in set(a) | set(b)}


class Sim:
    def __init__(self, view=False):
        self.m = mujoco.MjModel.from_xml_path(str(XML))
        self.d = mujoco.MjData(self.m)
        self.d.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)
        self.body_idx, self.rhand_idx = build_indices(self.m)
        # motor idx (22-28) -> posicao dentro de body_idx
        self.motor_slot = {midx: int(np.where(self.body_idx ==
                           mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j))[0][0])
                           for midx, j in MOTOR_TO_JOINT.items()}
        self.cup_gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "geometria_bloco")
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d) if view else None

    def apply_and_step(self, arm, hand, arm_kp, arm_kd, hand_kp, hand_kd,
                       leg_kp=100.0, leg_kd=2.5, n=1):
        torques = np.zeros(self.m.nu)
        for _ in range(n):
            # corpo: pernas/cintura/braco-esq seguram standing q=0; braco dir = arm
            for slot, jid in enumerate(self.body_idx):
                q_cur = self.d.qpos[jid + 7 - 1]
                dq_cur = self.d.qvel[jid + 6 - 1]
                midx = next((mi for mi, s in self.motor_slot.items() if s == slot), None)
                if midx is not None:
                    q_t, kp, kd = arm.get(midx, 0.0), arm_kp, arm_kd
                else:
                    q_t, kp, kd = 0.0, leg_kp, leg_kd
                torques[jid - 1] = kp * (q_t - q_cur) - kd * dq_cur
            # mao direita
            for i, jid in enumerate(self.rhand_idx):
                q_cur = self.d.qpos[jid + 7 - 1]
                dq_cur = self.d.qvel[jid + 6 - 1]
                torques[jid - 1] = hand_kp * (hand.get(i, 0.0) - q_cur) - hand_kd * dq_cur
            self.d.ctrl[:] = torques
            mujoco.mj_step(self.m, self.d)
            if self.viewer:
                self.viewer.sync()

    def cup_z(self):
        return float(self.d.geom_xpos[self.cup_gid][2])

    def hand_cup_contacts(self):
        c = 0
        for k in range(self.d.ncon):
            con = self.d.contact[k]
            n1 = self.m.geom(con.geom1).name or ""
            n2 = self.m.geom(con.geom2).name or ""
            names = n1 + n2
            if "geometria_bloco" in names and ("hand" in n1 or "hand" in n2):
                c += 1
        return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    poses = json.loads((HERE / "ik_poses.json").read_text())
    REACH = {int(k): v for k, v in poses["reach"].items()}
    LIFT = {int(k): v for k, v in poses["lift"].items()}
    HOME = {m: 0.0 for m in MOTOR_TO_JOINT}

    sim = Sim(view=args.view)

    # assenta o copo
    for _ in range(1500):
        sim.apply_and_step(HOME, HAND_OPEN, 150, 3, 20, 1, n=1)
    z0 = sim.cup_z()
    print(f"[fase 0] copo assentado em Z={z0:.3f}")

    # approach gradual HOME->REACH
    for i in range(1, 6):
        sim.apply_and_step(lerp(HOME, REACH, i/5.0), HAND_OPEN, 150, 3, 20, 1, n=80)
    sim.apply_and_step(REACH, HAND_OPEN, 150, 3, 20, 1, n=300)  # settle
    print(f"[fase reach] copo Z={sim.cup_z():.3f}, contatos mao-copo={sim.hand_cup_contacts()}")

    # grasp complacente + aperta
    sim.apply_and_step(REACH, HAND_CLOSE, 150, 3, 5, 3, n=400)
    sim.apply_and_step(REACH, HAND_CLOSE, 150, 3, 18, 2, n=200)
    z_grasp = sim.cup_z()
    print(f"[fase grasp] copo Z={z_grasp:.3f}, contatos mao-copo={sim.hand_cup_contacts()}")

    # lift gradual REACH->LIFT
    for i in range(1, 6):
        sim.apply_and_step(lerp(REACH, LIFT, i/5.0), HAND_CLOSE, 150, 3, 20, 2, n=100)
    sim.apply_and_step(LIFT, HAND_CLOSE, 150, 3, 20, 2, n=500)  # hold
    z1 = sim.cup_z()
    print(f"[fase lift] copo Z={z1:.3f}, contatos mao-copo={sim.hand_cup_contacts()}")

    print("\n=== RESULTADO ===")
    print(f"  Z copo assentado:      {z0:.3f} m")
    print(f"  Z copo apos lift:      {z1:.3f} m")
    print(f"  Elevacao liquida:      {(z1 - z0)*100:+.1f} cm")
    ok = (z1 - z0) > 0.04 and sim.hand_cup_contacts() > 0
    print(f"  >>> {'PEGOU E LEVANTOU ✓' if ok else 'NAO LEVANTOU ✗'}")

    if args.view:
        input("Enter pra fechar...")


if __name__ == "__main__":
    main()
