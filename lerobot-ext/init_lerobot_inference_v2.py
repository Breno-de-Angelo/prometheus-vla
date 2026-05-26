#!/usr/bin/env python

import os
import sys
import time
import torch
import cv2
import zmq
import numpy as np
from safetensors.torch import load_file
from Scripts_Prometheus_int.sim.sensor_utils import SensorClient, ImageUtils

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
    # SETUP DAS CÂMERAS (Fake Video ou Stream Real do Robô via ZMQ)
    # =================================================================
    fake_img_rgb = None
    fake_cap = None
    stream_client = None

    if cam_robot_ip:
        # Inicializa a conexão ZMQ APENAS UMA VEZ aqui no setup
        stream_client = SensorClient()
        # Assumindo que o método seja connect. Ajuste se for start_client()
        stream_client.start_client(server_ip=cam_robot_ip, port=int(cam_port))
        print(f"📡 Conectando ao ZMQ SensorServer em tcp://{cam_robot_ip}:{cam_port}...")
        
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
    checkpoint_dir = "train_output/pick_up_the_cup_nodepth-260524/best_val_checkpoint/pretrained_model" 
    policy = load_native_policy(checkpoint_dir, device)

    # ─────────────────────────────────────────────────────────
    # ATIVA O SCENE UNCERTAINTY GATE
    # ─────────────────────────────────────────────────────────
    threshold = 0.5
    policy.config.scene_uncertainty_threshold = threshold
    print(f"✅ Uncertainty Gate ativado com threshold = {threshold}")

    # ─────────────────────────────────────────────────────────
    # GARANTE QUE O BUFFER neutral_position EXISTA
    # ─────────────────────────────────────────────────────────
    if not hasattr(policy, 'neutral_position'):
        action_dim = list(policy.config.output_features.values())[0].shape[0]
        policy.register_buffer('neutral_position', torch.zeros(action_dim, device=device))
        print("⚠️ Buffer 'neutral_position' não existia no checkpoint. Criado com zeros.")
    else:
        print("✅ Buffer 'neutral_position' já existe no modelo.")

    # ─────────────────────────────────────────────────────────
    # TENTA RECALCULAR A POSIÇÃO NEUTRA (se houver estatísticas)
    # ─────────────────────────────────────────────────────────
    stats_path = os.path.join(checkpoint_dir, "dataset_stats.pt")
    if os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location=device)
        if "action" in stats:
            try:
                from policies.act_depth.neutral_position import compute_neutral_position
                from robot.unitree_g1.config_unitree_g1 import UnitreeG1Config

                robot_cfg = UnitreeG1Config()
                action_stats = stats["action"]
                neutral_norm = compute_neutral_position(
                    robot_config=robot_cfg,
                    action_stats=action_stats,
                    action_dim=action_dim,
                    device=device
                )
                policy.neutral_position.copy_(neutral_norm)
                print("✅ Posição neutra recalculada com base nas estatísticas do dataset.")
            except Exception as e:
                print(f"⚠️ Falha ao recalcular posição neutra: {e}. Usando valor atual.")
        else:
            print("⚠️ dataset_stats.pt não contém 'action'. Mantendo neutral_position atual.")
    else:
        print("ℹ️ dataset_stats.pt não encontrado. Mantendo neutral_position atual (possivelmente zeros).")
        # Opcional: define um valor padrão conhecido (exemplo: todas as juntas em 0.0 rad normalizado)
        # policy.neutral_position.fill_(0.0)

    # Agora você pode acessar policy.neutral_position com segurança
    print(f"   Neutral_position (primeiros 5 valores): {policy.neutral_position[:5].cpu().numpy()}")
    
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
            # 2. INJEÇÃO DE CÂMERAS (Stream de Rede via ZMQ ou Fake)
            # =========================================================
            if stream_client is not None:
                # Recebe o pacote JSON via ZMQ em tempo real
                msg = stream_client.receive_message() 
                
                if msg and "images" in msg:
                    # Desempacota as imagens usando seu Utils
                    frame_rgb = ImageUtils.decode_image(msg["images"]["head_camera"])
                    frame_depth = ImageUtils.decode_image(msg["images"]["head_camera_depth"])
                    
                    # O servidor já enviou o RGB convertido, então não precisa converter de novo
                    obs["head_camera"] = frame_rgb
                        
                    obs["head_camera_depth"] = frame_depth
                else:
                    # Opcional: lidar com falha de frame do ZMQ se necessário
                    pass

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
                if cam_name not in obs and cam_name != "head_camera":
                    continue
                
                if cam_name == "head_camera" and fake_img_rgb is not None:
                    h_real, w_real = obs.get(cam_name, fake_img_rgb).shape[:2]
                    img = cv2.resize(fake_img_rgb, (w_real, h_real))
                else:
                    img = obs.get(cam_name)
                    if img is None:
                        continue

                # --- FIX DEFINITIVO DOS CANAIS DO DEPTH ---
                if cam_name == "head_camera_depth":
                    # Se tiver 2 dimensões (ex: 480x640), adiciona o eixo do canal (480x640x1)
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=-1)
                    # Se tiver 1 canal, triplica para 3 canais (480x640x3) para a ResNet aceitar
                    if img.shape[2] == 1:
                        img = np.repeat(img, 3, axis=-1)
                # ------------------------------------------

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
            
        if stream_client is not None:
            # Encerra o cliente ZMQ corretamente usando o método nativo da classe
            stream_client.stop_client()
            
        robot.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()