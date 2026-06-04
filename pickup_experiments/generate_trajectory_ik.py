#!/usr/bin/env python3
"""
Gera trajetória com cinemática inversa usando MuJoCo.

Calcula uma trajetória suave do estado inicial até pegar o copo
usando a Jacobiana do braço direito.
"""

import mujoco
import mujoco.viewer
import numpy as np
import json
import time
from pathlib import Path


def main():
    # Carrega o modelo do simulator
    xml_path = Path(__file__).parent.parent / "unitree-g1-mujoco/assets/scene_43dof.xml"
    print(f"[IK] Carregando modelo: {xml_path}")

    if not xml_path.exists():
        print(f"❌ Arquivo não encontrado: {xml_path}")
        return

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print("[IK] ✅ Modelo carregado!")

    # Tenta encontrar o site da mão direita
    try:
        hand_site_names = ["right_hand_marker", "right_palm_site", "hand_effector"]
        hand_site_id = -1
        hand_site_name = None
        hand_body_id = -1

        for name in hand_site_names:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                hand_site_id = site_id
                hand_site_name = name
                break

        if hand_site_id < 0:
            print("[IK] ⚠️ Nenhum site da mão encontrado, usando corpo da ponta do dedo")
            # Usa o corpo do dedo polegar (thumb_2) que é a ponta mais frontal da mão
            hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_2_link")
            if hand_body_id < 0:
                print("[IK] ❌ Não encontrou right_hand_thumb_2_link")
                return
        else:
            print(f"[IK] ✅ Site da mão encontrado: {hand_site_name} (id={hand_site_id})")

        # Posição do copo (baseado no diagnóstico de geometry)
        cup_pos = np.array([0.35, -0.05, 0.80])  # X, Y, Z (verificado via diagnóstico)
        print(f"[IK] ✅ Posição alvo do copo: {cup_pos}")

    except Exception as e:
        print(f"[IK] ❌ Erro ao encontrar objetos: {e}")
        return

    # Índices das juntas do braço direito
    R_SHOULDER_PITCH = 22
    R_SHOULDER_ROLL = 23
    R_SHOULDER_YAW = 24
    R_ELBOW = 25
    R_WRIST_ROLL = 26
    R_WRIST_PITCH = 27
    R_WRIST_YAW = 28

    arm_joint_ids = [R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW,
                     R_ELBOW, R_WRIST_ROLL, R_WRIST_PITCH, R_WRIST_YAW]

    print(f"[IK] Braço direito: {len(arm_joint_ids)} juntas")

    # Posição inicial (relax)
    print("[IK] Simulando trajetória com IK...")

    waypoints = []
    q_current = data.qpos.copy()

    # Waypoint 0: Posição inicial
    waypoints.append({
        "name": "inicial",
        "qpos": q_current.copy()
    })

    # Simula o movimento com IK
    dt = 0.01
    max_steps = 300
    reach_threshold = 0.05  # 5cm de tolerância
    gain = 10.0  # Ganho de velocidade (aumentado 10x)

    for step in range(max_steps):
        # Posição atual da mão
        if hand_site_id >= 0:
            hand_pos = data.site_xpos[hand_site_id].copy()
        else:
            # Usar o corpo do dedo polegar (ponta mais frontal)
            hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_2_link")
            hand_pos = data.xpos[hand_body_id].copy()

        # Erro de posição (apenas X, Y, Z)
        error = cup_pos - hand_pos
        distance = np.linalg.norm(error)

        if step % 50 == 0:
            print(f"[IK] Step {step}: distância ao copo = {distance:.4f}m")

        if distance < reach_threshold:
            print(f"[IK] ✅ Alcançou o copo em {step} passos!")
            waypoints.append({
                "name": "pegando_copo",
                "qpos": q_current.copy()
            })
            break

        # Calcula a Jacobiana para a posição da mão
        jac = np.zeros((3, model.nv))
        if hand_site_id >= 0:
            mujoco.mj_jacSite(model, data, jac, None, hand_site_id)
        else:
            hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_2_link")
            mujoco.mj_jacBody(model, data, jac, None, hand_body_id)

        # Filtra apenas as juntas do braço
        jac_arm = jac[:, arm_joint_ids]

        # Calcula a velocidade necessária com Levenberg-Marquardt (Damped Least Squares)
        try:
            damping = 0.01  # Fator de amortecimento para evitar singularidades

            # dq = J^T * inv(J * J^T + lambda^2 * I) * error
            jjt = jac_arm @ jac_arm.T
            damped_inv = np.linalg.inv(jjt + damping * np.eye(3))
            dq = jac_arm.T @ damped_inv @ (error * gain)
        except Exception as e:
            print(f"[IK] ❌ Erro no cálculo: {e}")
            break

        # Atualiza as posições das juntas
        q_current[arm_joint_ids] += dq * dt

        # Clipa nos limites (estimados)
        q_current[arm_joint_ids] = np.clip(q_current[arm_joint_ids], -2.0, 2.0)

        # Atualiza o simulador
        data.qpos[:] = q_current
        mujoco.mj_forward(model, data)

        # A cada 20 passos, salva um waypoint
        if (step + 1) % 20 == 0:
            waypoints.append({
                "name": f"step_{step}",
                "qpos": q_current.copy()
            })

    # Waypoint final: volta para inicial
    waypoints.append({
        "name": "volta",
        "qpos": waypoints[0]["qpos"].copy()
    })

    print(f"[IK] Total de waypoints: {len(waypoints)}")

    # Salva os waypoints
    output = {
        "trajectory": [
            {
                "name": wp["name"],
                "qpos": wp["qpos"].tolist()
            }
            for wp in waypoints
        ]
    }

    output_file = Path(__file__).parent / "trajectory_ik.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[IK] ✅ Trajetória salva em: {output_file}")


if __name__ == "__main__":
    main()
