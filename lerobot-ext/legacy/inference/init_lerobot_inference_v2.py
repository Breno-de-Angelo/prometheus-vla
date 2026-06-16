#!/usr/bin/env python

import os
import sys
import time
import torch
import cv2
import numpy as np
from safetensors.torch import load_file

# =====================================================================
# 1. ATIVAÇÃO DO REGISTRO NATIVO ('actdepth')
# Garante que o Python reconheça o seu custom registry antes de carregar
# a configuração do Hugging Face / LeRobot.
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import policies  # O __init__.py desta pasta registra o 'actdepth'
except ImportError as e:
    print(f"[ERRO]: Falha ao carregar o registry 'policies': {e}")
    sys.exit(1)

# =====================================================================
# 2. IMPORTAÇÃO DOS MÓDULOS
# =====================================================================
from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig

def load_native_policy(checkpoint_dir, device):
    print(f"⏳ Carregando ACT-D (Nativo) de: {checkpoint_dir}")
    config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    policy = ACTPolicy(config)
    model_file = os.path.join(checkpoint_dir, "model.safetensors")
    state_dict = load_file(model_file)
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    policy.to(device)
    print("✅ Cérebro Multi-Modal Nativo carregado com sucesso!")
    return policy

def main():
    # =================================================================
    # CHECAGEM DE ARGUMENTOS CLI
    # =================================================================
    if any(flag in sys.argv for flag in ["-h", "--help", "-help"]):
        print("\n" + "="*60)
        print("UNITREE G1 - INFERÊNCIA ATIVA")
        print("="*60)
        print("USO: python init_lerobot_inference_v2.py [OPÇÕES]")
        print("\nOPÇÕES:")
        print("  --sim, --simulation=true   Força o modo de simulação.")
        print("  --fake-video=<CAMINHO>     Injeta imagem/vídeo na head_camera.")
        print("  --cam-robot=<IP>           Usa stream de câmera externa (Ex: 192.168.123.164)")
        print("  --port-cam=<PORTA>         Porta do stream da câmera (Padrão: 5555)")
        print("  -h, --help                 Mostra esta mensagem de ajuda.\n")
        sys.exit(0)

    is_sim = False
    fake_video_path = None
    cam_robot_ip = None
    debug_mode = False
    cam_port = "5555" # Porta padrão caso o usuário não passe o --port-cam

    for arg in sys.argv:
        if arg in ["--sim", "--simulation=true"]:
            is_sim = True
            print("[INFO]: Modo SIMULAÇÃO ativado (--sim)")
        elif arg.startswith("--fake-video="):
            fake_video_path = arg.split("=")[1]
            print(f"[INFO]: Modo FAKE VIDEO ativado. Alvo: {fake_video_path}")
        elif arg.startswith("--cam-robot="):
            cam_robot_ip = arg.split("=")[1]
            print(f"[INFO]: Stream de Câmera Externa IP configurado: {cam_robot_ip}")
        elif arg.startswith("--port-cam="):
            cam_port = arg.split("=")[1]
        elif arg.startswith("--debug"):
            debug_mode = True
            print("[INFO]: Modo DEBUG ativado. Logs adicionais serão exibidos.")
            

    if not is_sim:
        print("[INFO]: Modo ROBÔ REAL ativado")

    # =================================================================
    # SETUP DAS CÂMERAS (Fake Video ou Stream Real do Robô via Rede)
    # =================================================================
    fake_img_rgb = None
    fake_cap = None
    
    stream_cap_rgb = None
    stream_cap_depth = None

    if cam_robot_ip:
        # ATENÇÃO: Se o seu stream usar RTSP, basta trocar de http:// para rtsp://
        # Aqui assumo rotas genéricas /rgb e /depth. Ajuste conforme sua API no Unitree.
        rgb_url = f"http://{cam_robot_ip}:{cam_port}/rgb"
        depth_url = f"http://{cam_robot_ip}:{cam_port}/depth"
        
        stream_cap_rgb = cv2.VideoCapture(rgb_url)
        stream_cap_depth = cv2.VideoCapture(depth_url)
        print(f"📡 Conectando ao stream de Câmeras em {cam_robot_ip}:{cam_port}...")
        
    elif fake_video_path:
        if not os.path.exists(fake_video_path):
            print(f"❌ ERRO: Arquivo fake não encontrado: {fake_video_path}")
            sys.exit(1)
            
        if fake_video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            fake_cap = cv2.VideoCapture(fake_video_path)
            print("✅ Vídeo fake carregado com sucesso! (Modo Loop Ativado)")
        else:
            img_bgr = cv2.imread(fake_video_path)
            fake_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) 
            print("✅ Imagem fake carregada com sucesso!")

    # =================================================================
    # INICIALIZAÇÃO DO MODELO E ROBÔ
    # =================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "train_output/pick_up_the_cup_nodepth-260511/checkpoints/010000/pretrained_model" 
    policy = load_native_policy(checkpoint_dir, device)
    
    print(f"⏳ Conectando ao Unitree G1 (Simulação: {is_sim})...")
    g1_config = UnitreeG1Dex3Config(
        robot_ip="192.168.123.164", 
        control_mode="upper_body",
        is_simulation=is_sim
    )
    robot = UnitreeG1Dex3(g1_config)
    robot.connect()
    print("✅ Robô Conectado!")
    
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
    print("📺 Uma janela de vídeo será aberta para você acompanhar a visão da IA.")
    
    try:
        while True:
            start_t = time.perf_counter()
            obs = robot.get_observation()
            if not obs: continue

            batch = {}
            
            # 1. Agrupa as juntas
            state_vector = []
            for name in joint_names:
                state_vector.append(obs.get(name, 0.0))
            batch["observation.state"] = torch.tensor(state_vector).float().to(device).unsqueeze(0)

            # =========================================================
            # 2. INJEÇÃO DE CÂMERAS (Stream de Rede, Fake ou Nativa)
            # =========================================================
            if stream_cap_rgb is not None and stream_cap_depth is not None:
                # Substitui tudo pelo que vem da rede
                ret_rgb, frame_rgb = stream_cap_rgb.read()
                ret_depth, frame_depth = stream_cap_depth.read()
                
                if ret_rgb:
                    obs["head_camera"] = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
                
                if ret_depth:
                    # Se vier como RGB (3 canais), converte para tons de cinza (1 canal)
                    if len(frame_depth.shape) == 3:
                        frame_depth = cv2.cvtColor(frame_depth, cv2.COLOR_BGR2GRAY)
                    
                    # Garante que tenha 3 dimensões (H, W, 1) para o permute não quebrar depois
                    if len(frame_depth.shape) == 2:
                        frame_depth = np.expand_dims(frame_depth, axis=-1)
                        
                    obs["head_camera_depth"] = frame_depth

            elif fake_cap is not None:
                # Modo de vídeo simulado
                ret, frame = fake_cap.read()
                if not ret:
                    fake_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = fake_cap.read()
                if ret:
                    fake_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # =========================================================
            # 3. PROCESSAMENTO FINAL DAS IMAGENS PRO TENSOR
            # =========================================================
            for cam_name in ["head_camera", "head_camera_depth"]:
                if cam_name == "head_camera" and fake_img_rgb is not None:
                    h_real, w_real = obs[cam_name].shape[:2]
                    img = cv2.resize(fake_img_rgb, (w_real, h_real))
                else:
                    img = obs[cam_name]

                # 📺 Exibição da janela (Apenas RGB para facilitar)
                if cam_name == "head_camera":
                    img_bgr_display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Visao da IA - Head Camera", img_bgr_display)
                    cv2.waitKey(1)

                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0
                batch[f"observation.images.{cam_name}"] = img_tensor.unsqueeze(0)

            # 4. Processa a Pressão
            batch["observation.left_hand_pressure"] = torch.from_numpy(obs["left_hand_pressure"]).float().to(device).unsqueeze(0)
            batch["observation.right_hand_pressure"] = torch.from_numpy(obs["right_hand_pressure"]).float().to(device).unsqueeze(0)

            # 5. PENSAMENTO DA IA
            with torch.inference_mode(), torch.autocast(device_type=device.type if "cuda" in device.type else "cpu"):
                action = policy.select_action(batch)
            
            # 6. EXECUÇÃO
            action_numpy = action.squeeze(0).cpu().numpy()
            
            action_dict = {}
            for i, name in enumerate(joint_names):
                action_dict[name] = float(action_numpy[i])

            if debug_mode:
                # Print Visual no Terminal
                valores_formatados = " | ".join([f"{v:.2f}" for v in action_numpy])
                print(f"\r🤖 IA -> [{valores_formatados}]", end="", flush=True)
            
            robot.send_action(action_dict)
            
            # Frequência de 50Hz
            elapsed = time.perf_counter() - start_t
            time.sleep(max(0, 0.02 - elapsed))

    except KeyboardInterrupt:
        print("\n🛑 Parando...")
    finally:
        if fake_cap is not None:
            fake_cap.release()
        if stream_cap_rgb is not None:
            stream_cap_rgb.release()
        if stream_cap_depth is not None:
            stream_cap_depth.release()
            
        robot.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()