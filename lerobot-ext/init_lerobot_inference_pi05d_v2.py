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
import numpy as np
from pathlib import Path

# Adicionar repo ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Imports lerobot
from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
from lerobot.cameras.zmq.camera_zmq import ZMQCamera
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig
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
    # SETUP DAS CÂMERAS (ZMQ — mesmo protocolo do inference_realtime_pi05d)
    # =================================================================
    zmq_cam_rgb = None
    zmq_cam_depth = None

    if cam_robot_ip:
        print(f"📡 Conectando ZMQ em {cam_robot_ip}:{cam_port}...")
        zmq_cam_rgb = ZMQCamera(ZMQCameraConfig(
            server_address=cam_robot_ip, port=int(cam_port),
            camera_name="head_camera", width=640, height=480,
        ))
        zmq_cam_depth = ZMQCamera(ZMQCameraConfig(
            server_address=cam_robot_ip, port=int(cam_port),
            camera_name="head_camera_depth", width=640, height=480,
        ))
        zmq_cam_rgb.connect()
        zmq_cam_depth.connect()
        print(f"⏳ Aguardando primeiro frame do simulador ({cam_robot_ip}:{cam_port})...")
        deadline = time.time() + 60
        got_frame = False
        while time.time() < deadline:
            try:
                # timeout curto por chamada: async_read() LEVANTA TimeoutError, não retorna None
                frame = zmq_cam_rgb.async_read(timeout_ms=2000)
                if frame is not None:
                    print("✅ ZMQ conectado — primeiro frame recebido!")
                    got_frame = True
                    break
            except TimeoutError:
                print(f"   ...sem frame ainda ({int(deadline - time.time())}s restantes) — o run_sim.py do Mori está publicando a câmera em :{cam_port}?")
            except Exception as e:
                print(f"   ...erro lendo frame: {e}")
                time.sleep(0.5)
        if not got_frame:
            print("[ABORT] Simulador não enviou frame. No Mori: confira que o run_sim.py está rodando, a janela do MuJoCo está ativa/avançando, e que imprimiu 'Cameras ... → ZMQ port 5555'.")
            zmq_cam_rgb.disconnect()
            zmq_cam_depth.disconnect()
            sys.exit(1)

    # =================================================================
    # LOAD PI05-D POLICY (só depois de confirmar câmera)
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
    from robot.unitree_g1.g1_utils import G1_29_JointArmIndex
    # CORREÇÃO (Mori, commit 104a9e3): usar o VALOR do enum (índice real do motor:
    # ombro dir = 22), não a posição do enumerate (que daria 0-13 e mandaria o
    # braço pros motores da perna/quadril — por isso o braço não se mexia).
    body_motor_indices = {m.name: int(m.value) for m in G1_29_JointArmIndex}

    # Índices das mãos Dex3 (0-6 por mão): habilita o fechamento dos dedos no sim.
    # A ordem bate com _SIM_JOINTS (esq: thumb→middle→index; dir: thumb→index→middle).
    from robot.unitree_g1.g1_utils import LEFT_HAND_JOINT_NAMES, RIGHT_HAND_JOINT_NAMES
    left_hand_indices = {n: i for i, n in enumerate(LEFT_HAND_JOINT_NAMES)}
    right_hand_indices = {n: i for i, n in enumerate(RIGHT_HAND_JOINT_NAMES)}

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

            # 1. Estado (joints) — zeros em modo sim (sem robot conectado)
            if robot is not None:
                state_vector = [float(obs.get(n, 0.0)) for n, t in robot.observation_features.items() if t is float]
            else:
                state_vector = [0.0] * 28  # arm14 + hand14
            batch["observation.state"] = torch.tensor(state_vector).float().to(device).unsqueeze(0)

            # 2. RGB via ZMQ
            if zmq_cam_rgb is not None:
                frame_rgb = zmq_cam_rgb.async_read()
                if frame_rgb is not None:
                    if first_frame:
                        print(f"🎬 PRIMEIRA IMAGEM RECEBIDA de {cam_robot_ip}:{cam_port}")
                        first_frame = False
                    obs["head_camera"] = frame_rgb  # (H,W,3) uint8 RGB

            # 3. DEPTH via ZMQ — uint16 mm → uint8 (clip×38) × 3ch
            if zmq_cam_depth is not None:
                frame_depth = zmq_cam_depth.async_read()
                if frame_depth is not None:
                    if frame_depth.ndim == 3:
                        frame_depth = frame_depth[:, :, 0]
                    depth_m = frame_depth.astype(np.float32) / 1000.0
                    depth_u8 = np.clip(depth_m * 127.5, 0, 255).astype(np.uint8)  # cup3 encoding
                    obs["head_camera_depth"] = np.stack([depth_u8, depth_u8, depth_u8], axis=-1)

            # 4. MONTAR BATCH DE IMAGENS
            for cam_name in ["head_camera", "head_camera_depth"]:
                img = obs.get(cam_name, np.zeros((480, 640, 3), dtype=np.uint8))
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

            _SIM_JOINTS = [
                "kLeftShoulderPitch.q","kLeftShoulderRoll.q","kLeftShoulderYaw.q","kLeftElbow.q",
                "kLeftWristRoll.q","kLeftWristPitch.q","kLeftWristyaw.q",
                "kRightShoulderPitch.q","kRightShoulderRoll.q","kRightShoulderYaw.q","kRightElbow.q",
                "kRightWristRoll.q","kRightWristPitch.q","kRightWristYaw.q",
                "left_hand_thumb_0_joint.q","left_hand_thumb_1_joint.q","left_hand_thumb_2_joint.q",
                "left_hand_middle_0_joint.q","left_hand_middle_1_joint.q",
                "left_hand_index_0_joint.q","left_hand_index_1_joint.q",
                "right_hand_thumb_0_joint.q","right_hand_thumb_1_joint.q","right_hand_thumb_2_joint.q",
                "right_hand_index_0_joint.q","right_hand_index_1_joint.q",
                "right_hand_middle_0_joint.q","right_hand_middle_1_joint.q",
            ]
            feature_names = list(robot.action_features.keys()) if robot is not None else _SIM_JOINTS
            action_dict = {}
            action_list = action_numpy.tolist() if isinstance(action_numpy, np.ndarray) else list(action_numpy)
            for name in feature_names:
                if not action_list:
                    break
                action_dict[name] = float(action_list.pop(0))

            if debug_mode:
                valores_formatados = " | ".join([f"{v:.2f}" for v in action_dict.values()])
                print(f"\r🤖 IA -> [{valores_formatados}]", end="", flush=True)

            if action_sender:
                # Modo distribuído: braços (kp=150/kd=5 p/ vencer a gravidade do MuJoCo) + mãos Dex3
                action_sender.send_action(
                    action_dict,
                    body_motor_indices=body_motor_indices,
                    left_hand_indices=left_hand_indices,
                    right_hand_indices=right_hand_indices,
                    body_kp=150.0, body_kd=5.0,
                )
            elif robot:
                # Modo local: enviar ao robô local
                robot.send_action(action_dict)

            # 50Hz
            elapsed = time.perf_counter() - start_t
            time.sleep(max(0, 0.02 - elapsed))

    except KeyboardInterrupt:
        print("\n🛑 Parando...")
    finally:
        if zmq_cam_rgb is not None:
            zmq_cam_rgb.disconnect()
        if zmq_cam_depth is not None:
            zmq_cam_depth.disconnect()
        if action_sender:
            action_sender.close()
        if robot:
            robot.disconnect()


if __name__ == "__main__":
    main()
