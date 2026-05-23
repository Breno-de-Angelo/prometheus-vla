#!/usr/bin/env python
"""
Inferência PI05-D em MuJoCo (adaptado de init_lerobot_inference_v2.py)
Roda policy pi05-D com depth + pressure no simulator.

Uso:
    python init_lerobot_inference_pi05d_v2.py --sim \
        --checkpoint=train_output/.../pretrained_model
"""

import os
import sys
import time
import torch
import cv2
import numpy as np
from pathlib import Path

# Adicionar repo ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Imports lerobot
from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
from lerobot.policies.factory import make_pre_post_processors

# Import pi05-D loader
from train.inference_pi05_d import load_pi05_d

# Import action sender para modo distribuído
from action_sender_zmq import ActionSenderZMQ


def main():
    # =================================================================
    # CHECAGEM DE ARGUMENTOS CLI
    # =================================================================
    if any(flag in sys.argv for flag in ["-h", "--help", "-help"]):
        print("\n" + "="*60)
        print("UNITREE G1 - INFERÊNCIA PI05-D")
        print("="*60)
        print("USO: python init_lerobot_inference_pi05d_v2.py [OPÇÕES]")
        print("\nOPÇÕES:")
        print("  --sim, --simulation=true        Força o modo de simulação.")
        print("  --checkpoint=<CAMINHO>          Caminho do checkpoint pi05-D.")
        print("  --task=<DESCRICAO>              Task description (default: 'Pick up the cup')")
        print("  --cam-robot=<IP>                Usa stream de câmera externa (Ex: 192.168.123.164)")
        print("  --port-cam=<PORTA>              Porta do stream da câmera (Padrão: 5555)")
        print("  --debug                         Ativa modo DEBUG com logs adicionais.\n")
        sys.exit(0)

    is_sim = False
    cam_robot_ip = None
    cam_port = "5555"
    debug_mode = False
    task_description = "Pick up the cup"
    checkpoint_dir = None

    for arg in sys.argv:
        if arg in ["--sim", "--simulation=true"]:
            is_sim = True
            print("[INFO]: Modo SIMULAÇÃO ativado (--sim)")
        elif arg.startswith("--checkpoint="):
            checkpoint_dir = arg.split("=")[1]
            print(f"[INFO]: Checkpoint: {checkpoint_dir}")
        elif arg.startswith("--task="):
            task_description = arg.split("=", 1)[1]
            print(f"[INFO]: Task: {task_description}")
        elif arg.startswith("--cam-robot="):
            cam_robot_ip = arg.split("=")[1]
            print(f"[INFO]: Stream de Câmera Externa IP: {cam_robot_ip}")
        elif arg.startswith("--port-cam="):
            cam_port = arg.split("=")[1]
        elif arg.startswith("--debug"):
            debug_mode = True
            print("[INFO]: Modo DEBUG ativado.")

    if not checkpoint_dir:
        print("[ERRO]: --checkpoint é obrigatório!")
        sys.exit(1)

    if not is_sim:
        print("[INFO]: Modo ROBÔ REAL ativado")

    # =================================================================
    # SETUP DAS CÂMERAS
    # =================================================================
    stream_cap_rgb = None
    stream_cap_depth = None

    if cam_robot_ip:
        rgb_url = f"http://{cam_robot_ip}:{cam_port}/rgb"
        depth_url = f"http://{cam_robot_ip}:{cam_port}/depth"
        stream_cap_rgb = cv2.VideoCapture(rgb_url)
        stream_cap_depth = cv2.VideoCapture(depth_url)
        print(f"📡 Conectando ao stream em {cam_robot_ip}:{cam_port}...")

        # Testa conexão
        ret_rgb, _ = stream_cap_rgb.read()
        ret_depth, _ = stream_cap_depth.read()
        if ret_rgb and ret_depth:
            print(f"✅ CONECTADO! Recebendo streams de {cam_robot_ip}:{cam_port}")
        else:
            print(f"⚠️ Tentando conectar... (timeout esperado até Mori publicar)")

    # =================================================================
    # LOAD PI05-D POLICY
    # =================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⏳ Carregando PI05-D de: {checkpoint_dir}")
    policy = load_pi05_d(checkpoint_dir, device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=checkpoint_dir
    )
    print("✅ PI05-D carregado com sucesso!")

    # =================================================================
    # CONECTAR AO ROBÔ (OU INICIALIZAR ACTION SENDER PARA MODO DISTRIBUÍDO)
    # =================================================================
    action_sender = None
    if cam_robot_ip and is_sim:
        # Modo distribuído: enviar ações via ZMQ para o simulador remoto
        print(f"⏳ Inicializando action sender para {cam_robot_ip}:6001...")
        action_sender = ActionSenderZMQ(cam_robot_ip, port=6001, verbose=False)
        print("✅ Action sender inicializado!")
        # Não precisa conectar ao robô local
        robot = None
    else:
        # Modo local ou robô real: conectar normalmente
        print(f"⏳ Conectando ao Unitree G1 (Simulação: {is_sim})...")
        g1_config = UnitreeG1Dex3Config(
            robot_ip="192.168.123.164",
            control_mode="upper_body",
            is_simulation=is_sim
        )
        robot = UnitreeG1Dex3(g1_config)
        robot.connect()
        print("✅ Robô Conectado!")

    # Preparar mapeamento de índices de motores (para modo distribuído)
    from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex
    body_motor_indices = {}
    for idx, motor_enum in enumerate(G1_29_JointArmIndex):
        body_motor_indices[motor_enum.name] = idx

    print("\n🚀 INFERÊNCIA PI05-D ATIVA: O Robô irá se mover sozinho!")
    print("📺 Uma janela será aberta para você acompanhar a visão da IA.\n")

    first_frame = True
    try:
        while True:
            start_t = time.perf_counter()
            if robot:
                obs = robot.get_observation()
                if not obs:
                    continue
            else:
                # Modo distribuído: criar obs vazio
                obs = {}

            batch = {}

            # 1. Estado (joints)
            state_vector = []
            for name in robot.observation_features.keys():
                if robot.observation_features[name] is float:
                    state_vector.append(float(obs.get(name, 0.0)))
            batch["observation.state"] = torch.tensor(state_vector).float().to(device).unsqueeze(0)

            # 2. IMAGENS (RGB + DEPTH via ZMQ ou stream)
            if stream_cap_rgb is not None and stream_cap_depth is not None:
                ret_rgb, frame_rgb = stream_cap_rgb.read()
                ret_depth, frame_depth = stream_cap_depth.read()

                if first_frame and (ret_rgb or ret_depth):
                    print(f"🎬 PRIMEIRA IMAGEM RECEBIDA! Frames chegando de {cam_robot_ip}:{cam_port}")
                    first_frame = False

                if ret_rgb:
                    obs["head_camera"] = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

                if ret_depth:
                    if len(frame_depth.shape) == 3:
                        frame_depth = cv2.cvtColor(frame_depth, cv2.COLOR_BGR2GRAY)
                    if len(frame_depth.shape) == 2:
                        frame_depth = np.expand_dims(frame_depth, axis=-1)
                    obs["head_camera_depth"] = frame_depth

            # 3. PROCESSAMENTO DE IMAGENS
            for cam_name in ["head_camera", "head_camera_depth"]:
                img = obs.get(cam_name)
                if img is None:
                    if cam_name == "head_camera":
                        img = np.zeros((480, 640, 3), dtype=np.uint8)
                    else:
                        img = np.zeros((480, 640, 3), dtype=np.uint8)

                # Exibição (RGB só) - skip se sem display
                if cam_name == "head_camera":
                    try:
                        img_bgr_display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Visao da IA - Head Camera", img_bgr_display)
                        cv2.waitKey(1)
                    except cv2.error:
                        pass  # Sem display (servidor)

                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0
                batch[f"observation.images.{cam_name}"] = img_tensor.unsqueeze(0)

            # 4. PRESSÃO (dois tensores de 33)
            batch["observation.left_hand_pressure"] = torch.zeros(1, 33, device=device)
            batch["observation.right_hand_pressure"] = torch.zeros(1, 33, device=device)

            # Se tiver dados de pressão, usar
            if "left_hand_pressure" in obs:
                batch["observation.left_hand_pressure"] = torch.from_numpy(
                    np.asarray(obs["left_hand_pressure"], dtype=np.float32)
                ).to(device).unsqueeze(0)
            if "right_hand_pressure" in obs:
                batch["observation.right_hand_pressure"] = torch.from_numpy(
                    np.asarray(obs["right_hand_pressure"], dtype=np.float32)
                ).to(device).unsqueeze(0)

            # 5. TASK DESCRIPTION
            batch["task"] = task_description

            # 6. INFERÊNCIA
            batch = preprocessor(batch)
            with torch.inference_mode():
                action_chunk = policy.predict_action_chunk(batch)  # shape: (1, chunk_size, action_dim)

            # 7. EXECUTAR AÇÕES (primeiro passo do chunk)
            action_norm = action_chunk[:, 0, :]  # shape: (1, action_dim)
            action_post = postprocessor(action_norm)  # desnormalizar
            action_numpy = action_post.squeeze(0).cpu().numpy()

            action_dict = {}
            action_list = action_numpy.tolist() if isinstance(action_numpy, np.ndarray) else list(action_numpy)
            for name, _ in robot.action_features.items():
                if not action_list:
                    break
                action_dict[name] = float(action_list.pop(0))

            if debug_mode:
                valores_formatados = " | ".join([f"{v:.2f}" for v in action_dict.values()])
                print(f"\r🤖 IA -> [{valores_formatados}]", end="", flush=True)

            if action_sender:
                # Modo distribuído: enviar via ZMQ (apenas corpo, sem mãos por enquanto)
                action_sender.send_action(action_dict, body_motor_indices=body_motor_indices)
            elif robot:
                # Modo local: enviar ao robô local
                robot.send_action(action_dict)

            # 50Hz
            elapsed = time.perf_counter() - start_t
            time.sleep(max(0, 0.02 - elapsed))

    except KeyboardInterrupt:
        print("\n🛑 Parando...")
    finally:
        if stream_cap_rgb is not None:
            stream_cap_rgb.release()
        if stream_cap_depth is not None:
            stream_cap_depth.release()
        if action_sender:
            action_sender.close()
        if robot:
            robot.disconnect()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
