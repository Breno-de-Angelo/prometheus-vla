#!/usr/bin/env python3
"""
Gera ik_poses_parallel.json com IK 6-DOF (posicao + ORIENTACAO).

Estrategia (variante "paralela"): a orientacao do punho fica TRAVADA durante toda
a aproximacao (mao paralela ao copo), so a posicao translada START -> REACH -> LIFT.
A orientacao alvo = a orientacao natural do punho na pose de reach posicional.

Base da pelvis fixada em Z=0.910 (sim real, faixa elastica).
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
HAND_BODY = "right_hand_middle_1_link"
BASE_Z = 0.910
CUP_REAL = np.array([0.343, -0.048, 0.80])
LIFT_DZ = 0.12


def reset_with_arm(model, data, qadr, q):
    mujoco.mj_resetData(model, data)
    data.qpos[2] = BASE_Z
    for midx, v in q.items():
        data.qpos[qadr[midx]] = v
    mujoco.mj_forward(model, data)


def quat_of_body(model, data, bid):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, data.xmat[bid])
    return q


def ik6(model, data, q_arm, target_pos, target_quat, body_id, qadr, dofadr,
        cols, w_rot=1.0, iters=600, gain=4.0, dt=0.01, damping=0.04, tol=0.008):
    """IK 6-DOF: posicao + orientacao (se target_quat=None, so posicao)."""
    q = dict(q_arm)
    for _ in range(iters):
        reset_with_arm(model, data, qadr, q)
        pos = data.xpos[body_id].copy()
        perr = target_pos - pos

        if target_quat is not None:
            qc = quat_of_body(model, data, body_id)
            dq = np.zeros(4)
            mujoco.mju_subQuat(dq[:3], target_quat, qc)  # erro rot (3-vec, frame local)
            rerr = dq[:3] * w_rot
            err = np.concatenate([perr, rerr])
        else:
            err = perr

        if np.linalg.norm(perr) < tol and (target_quat is None or np.linalg.norm(err[3:]) < 0.05):
            break

        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        if target_quat is not None:
            J = np.vstack([jacp[:, cols], jacr[:, cols]])
            JJt = J @ J.T
            dq_arm = J.T @ np.linalg.solve(JJt + damping * np.eye(6), err * gain)
        else:
            J = jacp[:, cols]
            dq_arm = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err * gain)
        for k, midx in enumerate(MOTOR_TO_JOINT):
            q[midx] = float(np.clip(q[midx] + dq_arm[k] * dt, -3.0, 3.0))
    return q, np.linalg.norm(perr)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    qadr = {m: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
            for m, j in MOTOR_TO_JOINT.items()}
    dofadr = {m: model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
              for m, j in MOTOR_TO_JOINT.items()}
    cols = [dofadr[m] for m in MOTOR_TO_JOINT]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)

    cup = CUP_REAL
    grasp = cup + np.array([-0.01, 0.0, 0.0])
    start_pos = grasp + np.array([-0.05, 0.0, 0.0])   # mao ja perto, so 5cm atras
    lift_pos = grasp + np.array([0.0, 0.0, LIFT_DZ])
    home = {m: 0.0 for m in MOTOR_TO_JOINT}

    # 1) orientacao alvo = orientacao natural do punho na pose posicional de reach
    ref, _ = ik6(model, data, home, grasp, None, body_id, qadr, dofadr, cols)
    reset_with_arm(model, data, qadr, ref)
    target_quat = quat_of_body(model, data, body_id)
    print(f"[IK||] orientacao travada (quat) = {np.round(target_quat,3)}")

    # 2) START: pose inicial (mao paralela, 5cm atras, acima da mesa)
    start, es = ik6(model, data, home, start_pos, target_quat, body_id, qadr, dofadr, cols)
    print(f"[IK||] START err={es*100:.1f}cm")

    def cart_path(q0, p_from, p_to, step=0.01):
        """Integra passos cartesianos retos de p_from->p_to (orientacao travada).
        Devolve lista de waypoints (movimento minimo, caminho reto)."""
        n = max(2, int(np.linalg.norm(p_to - p_from) / step))
        wps, q = [], dict(q0)
        for i in range(1, n + 1):
            tgt = p_from + (p_to - p_from) * (i / n)
            q, _ = ik6(model, data, q, tgt, target_quat, body_id, qadr, dofadr, cols,
                       iters=80, tol=0.003)
            wps.append({str(m): q[m] for m in MOTOR_TO_JOINT})
        return wps, q

    # 3) APPROACH: vai reto pra frente START -> grasp (mao aberta)
    approach, q_reach = cart_path(start, start_pos, grasp, step=0.008)
    # 4) LIFT: sobe reto grasp -> lift (mao fechada)
    lift_wps, _ = cart_path(q_reach, grasp, lift_pos, step=0.008)
    print(f"[IK||] approach={len(approach)} wps, lift={len(lift_wps)} wps")

    out = {
        "start": {str(m): start[m] for m in MOTOR_TO_JOINT},
        "approach": approach,
        "lift": lift_wps,
    }
    (HERE / "ik_poses_parallel.json").write_text(json.dumps(out, indent=2))
    print(f"[IK||] Salvo em {HERE/'ik_poses_parallel.json'}")


if __name__ == "__main__":
    main()
