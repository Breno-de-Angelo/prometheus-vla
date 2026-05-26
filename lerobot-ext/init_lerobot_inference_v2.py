#!/usr/bin/env python
"""
Inference Entry Point V3 — Universal
Suporta: actdepth, pi05depth (e qualquer outro registrado em policies/)

Uso:
  python init_lerobot_inference_v3.py --checkpoint=<CAMINHO> [OPÇÕES]

Opções:
  --checkpoint=<PATH>    (obrigatório) Caminho para o pretrained_model
  --sim                  Modo simulação (sem robô real)
  --cam-robot=<IP>       Stream ZMQ de câmera externa
  --port-cam=<PORTA>     Porta do stream (padrão: 5555)
  --fake-video=<PATH>    Injeta imagem ou vídeo na câmera
  --uncertainty=<FLOAT>  Ativa o uncertainty gate (ex: 0.1)
  --v                    Abre janela de visualização da câmera
  --debug                Loga ações no terminal em tempo real
  -h, --help             Mostra esta mensagem

Exemplos:
  # ACT-D sem vídeo:
  python init_lerobot_inference_v3.py \
      --checkpoint=train_output/pick_up_the_cup_nodepth/best_val_checkpoint/pretrained_model

  # PI05-Depth com câmera e janela de vídeo:
  python init_lerobot_inference_v3.py \
      --checkpoint=train_output/pick_up_the_cup_pi05_depth/best_val_checkpoint/pretrained_model \
      --cam-robot=192.168.123.164 --v
"""

import os
import sys
import time
import torch
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# 1. REGISTRO DAS POLÍTICAS CUSTOMIZADAS
# ─────────────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import policies  # registra actdepth, pi05depth, etc.
except ImportError as e:
    print(f"[ERRO]: Falha ao carregar o registry 'policies': {e}")
    sys.exit(1)

from lerobot.configs.policies import PreTrainedConfig

# ─────────────────────────────────────────────────────────────────────
# 2. LOADER UNIVERSAL — detecta o tipo pelo config.json do checkpoint
# ─────────────────────────────────────────────────────────────────────
def load_policy(checkpoint_dir: str, device: torch.device):
    """
    Carrega qualquer política registrada a partir do checkpoint_dir.
    Detecta o tipo automaticamente via PreTrainedConfig.from_pretrained().
    Funciona com actdepth, pi05depth, e qualquer outro tipo registrado.
    """
    print(f"⏳ Carregando política de: {checkpoint_dir}")

    # Lê o config.json e instancia a subclasse correta automaticamente
    config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    policy_type = getattr(config, "type", "desconhecido")
    print(f"   Tipo detectado: {policy_type}")

    # Instancia a política correta via factory do LeRobot
    from lerobot.policies.factory import make_policy
    policy = make_policy(cfg=config, pretrained_path=checkpoint_dir)

    policy.eval()
    policy.to(device)
    print(f"✅ Política '{policy_type}' carregada com sucesso!")
    return policy, policy_type


# ─────────────────────────────────────────────────────────────────────
# 3. PREPROCESSOR UNIVERSAL — monta o batch correto para cada tipo
# ─────────────────────────────────────────────────────────────────────
def make_batch_for_policy(
    policy_type: str,
    obs: dict,
    joint_names: list[str],
    device: torch.device,
    has_depth: bool = False,
    has_pressure: bool = False,
) -> dict:
    """
    Monta o batch de entrada no formato que cada política espera.
    - actdepth : usa preprocessor próprio (normalização via dataset_stats)
    - pi05depth: usa preprocessor próprio (normaliza + tokeniza)
    Ambos retornam um dict com as chaves de observação já no formato correto.
    """
    batch = {}

    # ── Estado das juntas ────────────────────────────────────────────
    state_vector = [obs.get(name, 0.0) for name in joint_names]
    batch["observation.state"] = (
        torch.tensor(state_vector, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )

    # ── Câmera RGB ───────────────────────────────────────────────────
    rgb = obs.get("head_camera")
    if rgb is not None:
        batch["observation.images.head_camera"] = _img_to_tensor(rgb, device)

    # ── Câmera Depth (só se o modelo usa) ───────────────────────────
    if has_depth:
        depth = obs.get("head_camera_depth")
        if depth is not None:
            # Garante 3 canais (a ResNet/PointNet espera [B, 3, H, W])
            if len(depth.shape) == 2:
                depth = np.stack([depth] * 3, axis=-1)
            elif depth.shape[2] == 1:
                depth = np.repeat(depth, 3, axis=-1)
            batch["observation.images.head_camera_depth"] = _img_to_tensor(depth, device)

    # ── Pressão (só se o modelo usa) ─────────────────────────────────
    if has_pressure:
        for side in ["left", "right"]:
            key = f"{side}_hand_pressure"
            val = obs.get(key)
            if val is not None:
                batch[f"observation.{key}"] = (
                    torch.from_numpy(np.array(val, dtype=np.float32))
                    .unsqueeze(0)
                    .to(device)
                )

    # ── Task string (obrigatório para PI05) ──────────────────────────
    if policy_type == "pi05depth":
        batch["task"] = ["pick up the cup"]

    return batch


def _img_to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Converte HxWxC uint8 → [1, C, H, W] float32 em [0,1]."""
    return (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )


# ─────────────────────────────────────────────────────────────────────
# 4. SETUP DE CÂMERAS
# ─────────────────────────────────────────────────────────────────────
def setup_cameras(cam_robot_ip, cam_port, fake_video_path):
    """Inicializa stream ZMQ ou vídeo fake. Retorna (stream_client, fake_cap, fake_img_rgb)."""
    from Scripts_Prometheus_int.sim.sensor_utils import SensorClient, ImageUtils

    stream_client = None
    fake_cap = None
    fake_img_rgb = None

    if cam_robot_ip:
        stream_client = SensorClient()
        stream_client.start_client(server_ip=cam_robot_ip, port=int(cam_port))
        print(f"📡 Conectando ao ZMQ SensorServer em tcp://{cam_robot_ip}:{cam_port}...")

    elif fake_video_path:
        if not os.path.exists(fake_video_path):
            print(f"❌ ERRO: Arquivo fake não encontrado: {fake_video_path}")
            sys.exit(1)
        if fake_video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            fake_cap = cv2.VideoCapture(fake_video_path)
            print("✅ Vídeo fake carregado! (Modo Loop)")
        else:
            img_bgr = cv2.imread(fake_video_path)
            fake_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            print("✅ Imagem fake carregada!")

    return stream_client, fake_cap, fake_img_rgb


# ─────────────────────────────────────────────────────────────────────
# 5. LEITURA DE FRAME (ZMQ ou Fake)
# ─────────────────────────────────────────────────────────────────────
def get_camera_frames(obs, stream_client, fake_cap, fake_img_rgb):
    """
    Preenche obs["head_camera"] e obs["head_camera_depth"] com os frames atuais.
    Retorna o fake_img_rgb atualizado (pode mudar se for vídeo).
    """
    if stream_client is not None:
        from Scripts_Prometheus_int.sim.sensor_utils import ImageUtils
        msg = stream_client.receive_message()
        if msg and "images" in msg:
            obs["head_camera"] = ImageUtils.decode_image(msg["images"]["head_camera"])
            obs["head_camera_depth"] = ImageUtils.decode_image(msg["images"]["head_camera_depth"])

    elif fake_cap is not None:
        ret, frame = fake_cap.read()
        if not ret:
            fake_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = fake_cap.read()
        if ret:
            fake_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            obs["head_camera"] = fake_img_rgb

    elif fake_img_rgb is not None:
        obs["head_camera"] = fake_img_rgb

    return obs, fake_img_rgb


# ─────────────────────────────────────────────────────────────────────
# 6. PREPROCESSOR PI05 — aplica normalização e tokenização
# ─────────────────────────────────────────────────────────────────────
def load_pi05_preprocessor(checkpoint_dir: str, policy):
    """
    Carrega o preprocessor salvo junto com o checkpoint do PI05.
    Se não existir, tenta reconstruir a partir da config.
    """
    from lerobot.policies.factory import make_pre_post_processors
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=checkpoint_dir,
        )
        print("✅ Preprocessor PI05 carregado do checkpoint.")
        return preprocessor, postprocessor
    except Exception as e:
        print(f"⚠️ Não consegui carregar preprocessor salvo ({e}). "
              "O batch será passado diretamente sem normalização extra.")
        return None, None


# ─────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    # ── CLI ──────────────────────────────────────────────────────────
    if any(f in sys.argv for f in ["-h", "--help"]):
        print(__doc__)
        sys.exit(0)

    checkpoint_dir = None
    is_sim = False
    fake_video_path = None
    cam_robot_ip = None
    cam_port = "5555"
    debug_mode = False
    show_video = False            # ← desligado por padrão, ativa com --v
    uncertainty_threshold = 0.0

    for arg in sys.argv[1:]:
        if arg.startswith("--checkpoint="):
            checkpoint_dir = arg.split("=", 1)[1]
        elif arg in ["--sim", "--simulation=true"]:
            is_sim = True
        elif arg.startswith("--fake-video="):
            fake_video_path = arg.split("=", 1)[1]
        elif arg.startswith("--cam-robot="):
            cam_robot_ip = arg.split("=", 1)[1]
        elif arg.startswith("--port-cam="):
            cam_port = arg.split("=", 1)[1]
        elif arg.startswith("--uncertainty="):
            uncertainty_threshold = float(arg.split("=", 1)[1])
        elif arg == "--debug":
            debug_mode = True
        elif arg == "--v":
            show_video = True
            print("[INFO]: Visualização de câmera ativada (--v)")

    if checkpoint_dir is None:
        print("❌ ERRO: --checkpoint obrigatório.")
        print("   Uso: python init_lerobot_inference_v3.py --checkpoint=<CAMINHO>")
        sys.exit(1)

    # ── Dispositivo ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Usando device: {device}")

    # ── Carrega a política (universal) ────────────────────────────────
    policy, policy_type = load_policy(checkpoint_dir, device)

    # ── Detecta capacidades da política ──────────────────────────────
    has_depth = getattr(policy.config, "use_depth_3d", False)
    has_pressure = getattr(policy.config, "use_pressure", False)
    print(f"   Depth 3D: {has_depth} | Pressão: {has_pressure}")

    # ── Uncertainty Gate (PI05 e ACT-D suportam) ─────────────────────
    if uncertainty_threshold > 0:
        policy.config.scene_uncertainty_threshold = uncertainty_threshold
        print(f"✅ Uncertainty Gate ativado: threshold={uncertainty_threshold}")

    # ── Preprocessor (PI05 precisa de normalização + tokenização) ────
    preprocessor = None
    postprocessor = None
    if policy_type == "pi05depth":
        preprocessor, postprocessor = load_pi05_preprocessor(checkpoint_dir, policy)

    # ── Câmeras ───────────────────────────────────────────────────────
    stream_client, fake_cap, fake_img_rgb = setup_cameras(
        cam_robot_ip, cam_port, fake_video_path
    )

    # ── Robô ─────────────────────────────────────────────────────────
    from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
    print(f"⏳ Conectando ao Unitree G1 (Simulação: {is_sim})...")
    g1_config = UnitreeG1Dex3Config(
        robot_ip="192.168.123.164",
        control_mode="upper_body",
        is_simulation=is_sim,
    )
    robot = UnitreeG1Dex3(g1_config)
    robot.connect()
    print("✅ Robô conectado!")

    # Nomes das juntas na mesma ordem do dataset
    joint_names = [
        "kLeftShoulderPitch.q",  "kLeftShoulderRoll.q",  "kLeftShoulderYaw.q",
        "kLeftElbow.q",          "kLeftWristRoll.q",      "kLeftWristPitch.q",
        "kLeftWristyaw.q",
        "kRightShoulderPitch.q", "kRightShoulderRoll.q", "kRightShoulderYaw.q",
        "kRightElbow.q",         "kRightWristRoll.q",     "kRightWristPitch.q",
        "kRightWristYaw.q",
        "left_hand_thumb_0_joint.q",  "left_hand_thumb_1_joint.q",
        "left_hand_thumb_2_joint.q",  "left_hand_middle_0_joint.q",
        "left_hand_middle_1_joint.q", "left_hand_index_0_joint.q",
        "left_hand_index_1_joint.q",
        "right_hand_thumb_0_joint.q", "right_hand_thumb_1_joint.q",
        "right_hand_thumb_2_joint.q", "right_hand_index_0_joint.q",
        "right_hand_index_1_joint.q", "right_hand_middle_0_joint.q",
        "right_hand_middle_1_joint.q",
    ]

    print(f"\n🚀 INFERÊNCIA ATIVA [{policy_type.upper()}] — O robô vai se mover!")
    if show_video:
        print("   📺 Janela de câmera ativa.")
    print("   Ctrl+C para parar.\n")

    # ─────────────────────────────────────────────────────────────────
    # LOOP PRINCIPAL
    # ─────────────────────────────────────────────────────────────────
    try:
        while True:
            start_t = time.perf_counter()

            # 1. Observação do robô
            obs = robot.get_observation()
            if not obs:
                continue

            # 2. Câmeras
            obs, fake_img_rgb = get_camera_frames(
                obs, stream_client, fake_cap, fake_img_rgb
            )

            # 3. Exibe RGB na janela (só se --v foi passado)
            rgb = obs.get("head_camera")
            if show_video and rgb is not None:
                cv2.imshow("Visão da IA", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            # 4. Monta batch
            batch = make_batch_for_policy(
                policy_type=policy_type,
                obs=obs,
                joint_names=joint_names,
                device=device,
                has_depth=has_depth,
                has_pressure=has_pressure,
            )

            # 5. Preprocessor do PI05 (normalização + tokenização)
            #    O ACT-D já tem normalização interna no select_action
            if preprocessor is not None:
                batch = preprocessor(batch)

            # 6. Inferência
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu"
                ):
                    action = policy.select_action(batch)

            # 7. Pós-processamento PI05 (desnormaliza)
            if postprocessor is not None:
                action = postprocessor(action)

            # 8. Executa no robô
            action_numpy = action.squeeze(0).cpu().numpy()
            action_dict = {name: float(action_numpy[i]) for i, name in enumerate(joint_names)}

            if debug_mode:
                valores = " | ".join([f"{v:.2f}" for v in action_numpy])
                print(f"\r🤖 [{policy_type}] -> [{valores}]", end="", flush=True)

            robot.send_action(action_dict)

            # 9. 50 Hz
            elapsed = time.perf_counter() - start_t
            sleep_time = max(0.0, 0.02 - elapsed)
            if debug_mode and elapsed > 0.02:
                print(f"\n⚠️  Loop lento: {elapsed*1000:.1f}ms (limite: 20ms)")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 Parando inferência...")

    finally:
        if fake_cap is not None:
            fake_cap.release()
        if stream_client is not None:
            stream_client.stop_client()
        robot.disconnect()
        if show_video:
            cv2.destroyAllWindows()
        print("✅ Encerrado com segurança.")


if __name__ == "__main__":
    main()