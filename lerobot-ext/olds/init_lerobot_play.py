#!/usr/bin/env python
"""
Script para testar a IA treinada no robô físico ou no Simulador MuJoCo.
Usa o pré-processador do checkpoint e modo full_body (com pernas soltas).
"""
import os
import sys
import time
import argparse
import torch
import numpy as np

# Registra os módulos do G1
try:
    import robot.unitree_g1
    from robot.unitree_g1.g1_utils import G1_29_JointIndex
except ImportError as e:
    print(f"Erro ao carregar módulos: {e}")
    sys.exit(1)

from lerobot.policies.act.modeling_act import ACTPolicy
from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
from lerobot.policies.factory import make_pre_post_processors

def parse_args():
    parser = argparse.ArgumentParser(description="Inicia a inferência da Rede Neural no Unitree G1.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Caminho exato para a pasta 'pretrained_model' do seu treinamento.")
    parser.add_argument("--sim", action="store_true", help="Se ativado, roda no simulador MuJoCo. Se omitido, conecta no robô FÍSICO.")
    parser.add_argument("--robot_ip", type=str, default="127.0.0.1", help="Endereço IP do robô")
    return parser.parse_args()

def main():
    args = parse_args()
    CHECKPOINT_PATH = args.checkpoint_path

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n[ERRO] Checkpoint não encontrado: {CHECKPOINT_PATH}")
        sys.exit(1)

    modo_texto = "Simulação (MuJoCo)" if args.sim else "ROBÔ FÍSICO REAL"
    print(f"\n[1] Iniciando o sistema no modo: {modo_texto}")
    
    # Usa full_body para observar todas as juntas (43 posições)
    config = UnitreeG1Dex3Config(
        is_simulation=args.sim,
        robot_ip=args.robot_ip,
        control_mode="full_body"
    )
    robot = UnitreeG1Dex3(config)
    robot.connect()

    # 🛑 Desliga o torque das pernas e cintura
    print("[CONFIG] Desligando torque das pernas e cintura...")
    for joint_id in G1_29_JointIndex:
        name = joint_id.name.lower()
        if 'leg' in name or 'waist' in name:
            robot.msg.motor_cmd[joint_id.value].mode = 0
            robot.msg.motor_cmd[joint_id.value].kp = 0.0
            robot.msg.motor_cmd[joint_id.value].kd = 0.0
            robot.msg.motor_cmd[joint_id.value].q = 0.0
    robot.msg.crc = robot.crc.Crc(robot.msg)
    robot.lowcmd_publisher.Write(robot.msg)

    print("[2] Carregando a Rede Neural ACT e pré-processadores...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Carrega a política
        policy = ACTPolicy.from_pretrained(CHECKPOINT_PATH)
        policy.to(device)
        policy.eval()
        policy.reset()

        # Carrega os pré-processadores salvos no checkpoint
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=CHECKPOINT_PATH
        )
        
        print("Movendo para pose inicial...")
        robot.reset()    # move braços e mãos para posição padrão
        time.sleep(1.0)
        
    except Exception as e:
        print(f"\n[ERRO] Falha ao carregar os pesos da IA: {e}")
        import traceback
        traceback.print_exc()
        robot.disconnect()
        sys.exit(1)

    print("\n" + "="*60)
    print("🚨 ATENÇÃO: PREPARE O ROBÔ 🚨")
    print("A IA assume o controle em:")
    for i in range(3, 0, -1):
        print(f" {i}...")
        time.sleep(1)
    print(">>> IA NO CONTROLE! <<< (Ctrl+C para parar)")
    print("="*60 + "\n")
    
    # Ordem das features do estado (43 posições + 66 pressões)
    # Ordem das juntas (Motores 0-42) - Mantenha como está
    state_names = [
        "kLeftHipPitch.q", "kLeftHipRoll.q", "kLeftHipYaw.q", "kLeftKnee.q", "kLeftAnklePitch.q", "kLeftAnkleRoll.q",
        "kRightHipPitch.q", "kRightHipRoll.q", "kRightHipYaw.q", "kRightKnee.q", "kRightAnklePitch.q", "kRightAnkleRoll.q",
        "kWaistYaw.q", "kWaistRoll.q", "kWaistPitch.q",
        "kLeftShoulderPitch.q", "kLeftShoulderRoll.q", "kLeftShoulderYaw.q", "kLeftElbow.q", "kLeftWristRoll.q", "kLeftWristPitch.q", "kLeftWristyaw.q",
        "kRightShoulderPitch.q", "kRightShoulderRoll.q", "kRightShoulderYaw.q", "kRightElbow.q", "kRightWristRoll.q", "kRightWristPitch.q", "kRightWristYaw.q",
        "left_hand_thumb_0_joint.q", "left_hand_thumb_1_joint.q", "left_hand_thumb_2_joint.q",
        "left_hand_middle_0_joint.q", "left_hand_middle_1_joint.q",
        "left_hand_index_0_joint.q", "left_hand_index_1_joint.q",
        "right_hand_thumb_0_joint.q", "right_hand_thumb_1_joint.q", "right_hand_thumb_2_joint.q",
        "right_hand_index_0_joint.q", "right_hand_index_1_joint.q",
        "right_hand_middle_0_joint.q", "right_hand_middle_1_joint.q"
    ]

    # ✅ CORREÇÃO: Pressões intercaladas (Índices 43-108)
    # No seu dataset elas seguem a ordem: left_0, right_0, left_1, right_1...
    pressure_names = []
    for i in range(33):
        pressure_names.append(f"left_hand_pressure_{i}")
        pressure_names.append(f"right_hand_pressure_{i}")
    
    all_state_names = state_names + pressure_names

    try:
        while True:
            start_time = time.time()
            
            raw_obs = robot.get_observation()
            batch = {}
            
            # Imagens
            for cam_name in robot.cameras.keys():
                if cam_name in raw_obs:
                    img = torch.from_numpy(raw_obs[cam_name]).permute(2, 0, 1).float() / 255.0
                    batch[f"observation.images.{cam_name}"] = img.unsqueeze(0).to(device)
            
            # Estado
            state_values = []
            for name in all_state_names:
                val = raw_obs.get(name, 0.0)
                if isinstance(val, (list, np.ndarray)):
                    state_values.extend([float(v) for v in val])
                else:
                    state_values.append(float(val))
            
            # Garante tamanho 109
            if len(state_values) != 109:
                print(f"Aviso: estado com {len(state_values)} dimensões (esperado 109). Ajustando.")
                if len(state_values) < 109:
                    state_values.extend([0.0] * (109 - len(state_values)))
                else:
                    state_values = state_values[:109]
            
            batch["observation.state"] = torch.tensor(state_values, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Pré-processa
            batch = preprocessor(batch)
            
            # Inferência
            with torch.inference_mode():
                action_output = policy.select_action(batch)
                
                # Com Temporal Ensembling, o retorno costuma ser um dicionário ou tensor direto
                if isinstance(action_output, dict):
                    action_tensor = action_output["action"]
                else:
                    action_tensor = action_output

                # Removendo dimensões extras com squeeze()
                # Se for [1, 43] vira [43]. Se for [1, 1, 43] vira [43].
                action_array = action_tensor.squeeze().cpu().numpy()
                
                # Caso o squeeze resulte em algo vazio por erro de dimensão
                if action_array.ndim == 0:
                    action_array = action_tensor.cpu().numpy()
            
            # Converte para dicionário de ações
            action_keys = list(robot.action_features.keys())
            action_dict = {}
            for i, joint_name in enumerate(action_keys):
                if i < len(action_array):
                    action_dict[joint_name] = float(action_array[i])
            
            robot.send_action(action_dict)
            
            elapsed = time.time() - start_time
            time.sleep(max(0, 0.033 - elapsed))
            
    except KeyboardInterrupt:
        print("\n[INFO] Parada solicitada. Desligando...")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()