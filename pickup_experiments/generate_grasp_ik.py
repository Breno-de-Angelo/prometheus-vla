#!/usr/bin/env python3
"""
Gera ik_poses.json (reach + lift) resolvendo IK pro centro da mao (middle_1)
ate o centro REAL do copo (apos assentar a fisica), e uma pose de lift que
sobe o alvo ~12cm.

Saida: pickup_experiments/ik_poses.json  (motores 22-28)
"""
import json
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
HAND_BODY = "right_hand_middle_1_link"   # ponto de controle (centro da palma/dedo)
LIFT_DZ = 0.12                            # quanto levantar (m)
BASE_Z = 0.910                            # altura da pelvis no SIM REAL (faixa elastica)
CUP_REAL = np.array([0.343, -0.048, 0.80])  # copo observado no sim real (physics_state)


def settle_cup(model, data, steps=2000):
    mujoco.mj_resetData(model, data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "geometria_bloco")
    return data.geom_xpos[gid].copy()


def ik_solve(model, data, q_arm, target, body_id, qadr, dofadr,
             iters=400, gain=8.0, dt=0.01, damping=0.02, tol=0.01):
    """Resolve IK (DLS) movendo so as juntas do braco ate body_id ~ target."""
    q = dict(q_arm)
    for _ in range(iters):
        mujoco.mj_resetData(model, data)
        data.qpos[2] = BASE_Z  # pelvis na altura do sim real (faixa elastica)
        for midx, val in q.items():
            data.qpos[qadr[midx]] = val
        mujoco.mj_forward(model, data)

        pos = data.xpos[body_id].copy()
        err = target - pos
        if np.linalg.norm(err) < tol:
            break

        jac = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jac, None, body_id)
        cols = [dofadr[m] for m in MOTOR_TO_JOINT]
        J = jac[:, cols]
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + damping * np.eye(3), err * gain)
        for k, midx in enumerate(MOTOR_TO_JOINT):
            q[midx] = float(np.clip(q[midx] + dq[k] * dt, -3.0, 3.0))
    return q, np.linalg.norm(err)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)

    qadr = {m: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for m, j in MOTOR_TO_JOINT.items()}
    dofadr = {m: model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
              for m, j in MOTOR_TO_JOINT.items()}
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)

    cup = CUP_REAL  # posicao observada no sim real (mais confiavel que o settle offline)
    print(f"[IK] Copo alvo (sim real): {np.round(cup,3)}  | base pelvis Z={BASE_Z}")

    # Alvos: START (recuado p/ aproximacao FRONTAL), GRASP (no copo), LIFT
    grasp = cup + np.array([-0.01, 0.0, 0.0])
    start_target = grasp + np.array([-0.12, 0.0, 0.03])  # 12cm atras, 3cm acima
    lift_target = grasp + np.array([0.0, 0.0, LIFT_DZ])

    home = {m: 0.0 for m in MOTOR_TO_JOINT}

    start, e0 = ik_solve(model, data, home, start_target, body_id, qadr, dofadr)
    print(f"[IK] START resolvido, erro residual = {e0*100:.1f}cm")

    reach, e1 = ik_solve(model, data, start, grasp, body_id, qadr, dofadr)
    print(f"[IK] REACH resolvido, erro residual = {e1*100:.1f}cm")

    lift, e2 = ik_solve(model, data, reach, lift_target, body_id, qadr, dofadr)
    print(f"[IK] LIFT  resolvido, erro residual = {e2*100:.1f}cm")

    out = {
        "start": {str(m): start[m] for m in MOTOR_TO_JOINT},
        "reach": {str(m): reach[m] for m in MOTOR_TO_JOINT},
        "lift":  {str(m): lift[m]  for m in MOTOR_TO_JOINT},
    }
    path = HERE / "ik_poses.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[IK] Salvo em {path}")


if __name__ == "__main__":
    main()
