#!/usr/bin/env python3
"""
Pipeline integrado: assenta o copo -> le posicao real -> resolve IK pro
CENTRO DE PEGA (entre polegar e indicador) -> fecha a mao -> valida fisica.

Tudo no mesmo processo, entao a IK sempre mira o copo onde ele REALMENTE esta.
Salva ik_poses.json no fim (pras outras ferramentas / ZMQ).

Uso:
    conda activate g1
    python pickup_experiments/pickup_and_validate.py            # headless
    python pickup_experiments/pickup_and_validate.py --view      # com viewer
"""
import json
import time
import argparse
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

HERE = Path(__file__).parent
XML = HERE.parent / "unitree-g1-mujoco/assets/scene_43dof.xml"

MOTOR_TO_JOINT = {
    22: "right_shoulder_pitch_joint", 23: "right_shoulder_roll_joint",
    24: "right_shoulder_yaw_joint",   25: "right_elbow_joint",
    26: "right_wrist_roll_joint",     27: "right_wrist_pitch_joint",
    28: "right_wrist_yaw_joint",
}
HAND_OPEN = {i: 0.0 for i in range(7)}
HAND_CLOSE = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
LIFT_DZ = 0.12
GRASP_BODIES = ["right_hand_thumb_2_link", "right_hand_index_1_link"]


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
        self.motor_slot = {midx: int(np.where(self.body_idx ==
                           mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j))[0][0])
                           for midx, j in MOTOR_TO_JOINT.items()}
        self.qadr = {m: self.mj_jntadr(j) for m, j in MOTOR_TO_JOINT.items()}
        self.dofadr = {m: self.mj_dofadr(j) for m, j in MOTOR_TO_JOINT.items()}
        self.grasp_bids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
                           for b in GRASP_BODIES]
        self.cup_gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "geometria_bloco")
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d) if view else None

    def mj_jntadr(self, j):
        return self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)]

    def mj_dofadr(self, j):
        return self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)]

    # ---- controle PD por torque (igual base_sim) ----
    def apply_and_step(self, arm, hand, arm_kp, arm_kd, hand_kp, hand_kd,
                       leg_kp=120.0, leg_kd=3.0, n=1):
        torques = np.zeros(self.m.nu)
        for _ in range(n):
            bias = self.d.qfrc_bias  # gravidade + coriolis (feedforward)
            for slot, jid in enumerate(self.body_idx):
                dof = self.m.jnt_dofadr[jid]
                q_cur = self.d.qpos[jid + 6]
                dq_cur = self.d.qvel[dof]
                midx = next((mi for mi, s in self.motor_slot.items() if s == slot), None)
                if midx is not None:
                    q_t, kp, kd = arm.get(midx, 0.0), arm_kp, arm_kd
                else:
                    q_t, kp, kd = 0.0, leg_kp, leg_kd
                torques[jid - 1] = bias[dof] + kp * (q_t - q_cur) - kd * dq_cur
            for i, jid in enumerate(self.rhand_idx):
                dof = self.m.jnt_dofadr[jid]
                q_cur = self.d.qpos[jid + 6]
                dq_cur = self.d.qvel[dof]
                torques[jid - 1] = bias[dof] + hand_kp * (hand.get(i, 0.0) - q_cur) - hand_kd * dq_cur
            self.d.ctrl[:] = torques
            mujoco.mj_step(self.m, self.d)
            if self.viewer:
                self.viewer.sync()
                time.sleep(self.m.opt.timestep)  # ritmo ~tempo-real pra dar pra ver

    def grasp_center(self):
        return np.mean([self.d.xpos[b] for b in self.grasp_bids], axis=0)

    def cup_pos(self):
        return self.d.geom_xpos[self.cup_gid].copy()

    def cup_z(self):
        return float(self.d.geom_xpos[self.cup_gid][2])

    def hand_cup_contacts(self):
        c = 0
        for k in range(self.d.ncon):
            con = self.d.contact[k]
            n1 = self.m.geom(con.geom1).name or ""
            n2 = self.m.geom(con.geom2).name or ""
            if "geometria_bloco" in (n1 + n2) and ("hand" in n1 or "hand" in n2):
                c += 1
        return c

    # ---- IK puramente cinematica (snapshot do estado atual) ----
    def ik_to(self, q_start, target, iters=500, gain=6.0, dt=0.01, damping=0.03, tol=0.008):
        q = dict(q_start)
        snap = self.d.qpos.copy()
        for _ in range(iters):
            self.d.qpos[:] = snap
            for midx, v in q.items():
                self.d.qpos[self.qadr[midx]] = v
            mujoco.mj_forward(self.m, self.d)
            pos = self.grasp_center()
            err = target - pos
            if np.linalg.norm(err) < tol:
                break
            J = np.zeros((3, self.m.nv))
            for b in self.grasp_bids:
                jb = np.zeros((3, self.m.nv))
                mujoco.mj_jacBody(self.m, self.d, jb, None, b)
                J += jb
            J /= len(self.grasp_bids)
            cols = [self.dofadr[m] for m in MOTOR_TO_JOINT]
            Ja = J[:, cols]
            dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + damping * np.eye(3), err * gain)
            for k, midx in enumerate(MOTOR_TO_JOINT):
                q[midx] = float(np.clip(q[midx] + dq[k] * dt, -3.0, 3.0))
        self.d.qpos[:] = snap
        mujoco.mj_forward(self.m, self.d)
        return q, float(np.linalg.norm(err))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()
    sim = Sim(view=args.view)

    HOME = {m: 0.0 for m in MOTOR_TO_JOINT}
    TUCK = {m: 0.0 for m in MOTOR_TO_JOINT}
    TUCK[22] = -1.5   # ombro erguido: tira a mao de cima do copo durante o settle

    # 1) assenta o copo com o braco erguido (nao empurra o copo)
    for _ in range(1500):
        sim.apply_and_step(TUCK, HAND_OPEN, 150, 3, 20, 1)
    cup = sim.cup_pos()
    z0 = cup[2]
    print(f"[1] copo assentado em {np.round(cup,3)}")

    # 2) IK pro centro de pega chegar ao centro do copo, com pre-compensacao do
    #    droop de regime: alcanca, mede erro Cartesiano, re-mira (alvo += erro).
    ARM_KP, ARM_KD = 250, 5
    target = cup.copy()
    REACH = None
    for it in range(3):
        sim.d.qvel[:] = 0
        REACH, e1 = sim.ik_to(TUCK, target)
        for i in range(1, 6):
            sim.apply_and_step(lerp(TUCK, REACH, i/5.0), HAND_OPEN, ARM_KP, ARM_KD, 20, 1, n=90)
        sim.apply_and_step(REACH, HAND_OPEN, ARM_KP, ARM_KD, 20, 1, n=350)
        gc = sim.grasp_center()
        resid = sim.cup_pos() - gc
        print(f"[2.{it}] reach fisico dist_copo={np.linalg.norm(resid)*100:.1f}cm -> recompensando")
        target = target + resid  # pre-compensa o offset de regime
    LIFT, e2 = sim.ik_to(REACH, sim.cup_pos() + np.array([0, 0, LIFT_DZ]))
    gc = sim.grasp_center()
    print(f"[3] reach final. centro-pega={np.round(gc,3)} dist_copo={np.linalg.norm(gc-sim.cup_pos())*100:.1f}cm")

    # 4) fecha (complacente) e aperta
    sim.apply_and_step(REACH, HAND_CLOSE, 180, 4, 6, 3, n=400)
    sim.apply_and_step(REACH, HAND_CLOSE, 180, 4, 20, 2, n=200)
    print(f"[4] grasp: contatos mao-copo={sim.hand_cup_contacts()}  copo Z={sim.cup_z():.3f}")

    # 5) levanta + segura
    for i in range(1, 6):
        sim.apply_and_step(lerp(REACH, LIFT, i/5.0), HAND_CLOSE, 200, 4, 22, 2, n=110)
    sim.apply_and_step(LIFT, HAND_CLOSE, 200, 4, 22, 2, n=500)
    z1 = sim.cup_z()
    print(f"[5] lift: contatos={sim.hand_cup_contacts()}  copo Z={z1:.3f}")

    print("\n=== RESULTADO ===")
    print(f"  Z assentado={z0:.3f}  Z final={z1:.3f}  elevacao={(z1-z0)*100:+.1f}cm")
    ok = (z1 - z0) > 0.04 and sim.hand_cup_contacts() > 0
    print(f"  >>> {'PEGOU E LEVANTOU ✓' if ok else 'NAO LEVANTOU ✗'}")

    # salva poses pra reuso
    (HERE / "ik_poses.json").write_text(json.dumps(
        {"reach": {str(m): REACH[m] for m in MOTOR_TO_JOINT},
         "lift":  {str(m): LIFT[m] for m in MOTOR_TO_JOINT}}, indent=2))

    if args.view:
        print("Janela aberta. Feche a janela do MuJoCo pra encerrar.")
        while sim.viewer.is_running():
            sim.viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
