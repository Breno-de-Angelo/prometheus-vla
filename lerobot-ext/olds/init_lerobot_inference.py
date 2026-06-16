import os
import sys
import time
import torch
import numpy as np
from safetensors.torch import load_file
import types

# 1. IMPORTAÇÃO DOS SEUS MÓDULOS DE OPERAÇÃO
from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig
from train.act_d_injector import inject_act_d

def load_mutant_policy(checkpoint_dir, device):
    print(f"⏳ Carregando ACT-D (G1 Dex3 Multi-Modal) de: {checkpoint_dir}")
    config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    policy = ACTPolicy(config)
    
    # Aplica a injeção validada no treino (PointNet + Pressão)
    inject_act_d(policy, device)
    
    model_file = os.path.join(checkpoint_dir, "model.safetensors")
    state_dict = load_file(model_file)
    policy.load_state_dict(state_dict, strict=False)
    
    policy.eval()
    policy.to(device)
    print("✅ Cérebro Multi-Modal carregado!")
    return policy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "train_output/pick_up_the_cup3/checkpoints/001000/pretrained_model" 
    
    policy = load_mutant_policy(checkpoint_dir, device)
    
    # === CORREÇÃO: INSTANCIAÇÃO VIA CONFIG ===
    print("⏳ Conectando ao Unitree G1...")
    # Criamos o objeto de configuração primeiro
    g1_config = UnitreeG1Dex3Config(
        robot_ip="192.168.123.164", 
        control_mode="upper_body",
        is_simulation=False
    )
    # Passamos o objeto config para o robô
    robot = UnitreeG1Dex3(g1_config)
    robot.connect()
    print("✅ Robô Conectado!")
    
    # Lista de nomes de juntas na ordem EXATA do info.json (28 juntas)
    joint_names = [
        "kLeftShoulderPitch.q", "kLeftShoulderRoll.q", "kLeftShoulderYaw.q", "kLeftElbow.q", 
        "kLeftWristRoll.q", "kLeftWristPitch.q", "kLeftWristyaw.q",
        "kRightShoulderPitch.q", "kRightShoulderRoll.q", "kRightShoulderYaw.q", "kRightElbow.q", 
        "kRightWristRoll.q", "kRightWristPitch.q", "kRightWristYaw.q",
        "left_hand_thumb_0_joint.q", "left_hand_thumb_1_joint.q", "left_hand_thumb_2_joint.q",
        "left_hand_middle_0_joint.q", "left_hand_middle_1_joint.q", "left_hand_index_0_joint.q",
        "left_hand_index_1_joint.q",
        "right_hand_thumb_0_joint.q", "right_hand_thumb_1_joint.q", "right_hand_thumb_2_joint.q",
        "right_hand_index_0_joint.q", "right_hand_index_1_joint.q", "right_hand_middle_0_joint.q",
        "right_hand_middle_1_joint.q"
    ]

    print("\n🚀 INFERÊNCIA ATIVA: O Robô irá se mover sozinho!")
    
    try:
        while True:
            start_t = time.perf_counter()
            obs = robot.get_observation()
            if not obs: continue

            batch = {}
            
            # 1. Agrupa as juntas individuais no vetor 'observation.state' de 28 posições
            state_vector = []
            for name in joint_names:
                state_vector.append(obs.get(name, 0.0))
            batch["observation.state"] = torch.tensor(state_vector).float().to(device).unsqueeze(0)

            # 2. Processa as imagens (Normaliza para 0-1 e C,H,W)
            for cam_name in ["head_camera", "head_camera_depth"]:
                img = obs[cam_name] # H, W, C
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0
                batch[f"observation.images.{cam_name}"] = img_tensor.unsqueeze(0)

            # 3. Processa a Pressão (Chaves separadas como no record_v2)
            batch["observation.left_hand_pressure"] = torch.from_numpy(obs["left_hand_pressure"]).float().to(device).unsqueeze(0)
            batch["observation.right_hand_pressure"] = torch.from_numpy(obs["right_hand_pressure"]).float().to(device).unsqueeze(0)

            # 4. PENSAMENTO DA IA
            with torch.inference_mode(), torch.autocast(device_type=device.type if "cuda" in device.type else "cpu"):
                action = policy.select_action(batch)
            
            # 5. EXECUÇÃO
            action_numpy = action.squeeze(0).cpu().numpy()
            
            # Converte o vetor de volta para o dicionário que o send_action espera
            action_dict = {}
            for i, name in enumerate(joint_names):
                action_dict[name] = float(action_numpy[i])
            
            robot.send_action(action_dict)
            
            # Frequência de 50Hz
            elapsed = time.perf_counter() - start_t
            time.sleep(max(0, 0.02 - elapsed))

    except KeyboardInterrupt:
        print("\n🛑 Parando...")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()