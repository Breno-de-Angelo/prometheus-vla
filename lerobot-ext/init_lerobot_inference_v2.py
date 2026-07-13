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
    print(f"⏳ Carregando política de: {checkpoint_dir}")

    config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    policy_type = getattr(config, "type", "desconhecido")
    print(f"   Tipo detectado: {policy_type}")

    from safetensors.torch import load_file
    import importlib

    # LeRobot 0.4.4: make_policy() exige ds_meta — não serve para inferência.
    # Instanciamos a classe diretamente a partir da config.
    _POLICY_CLASS_MAP = {
        "actdepth":  ("policies.act_depth.modeling_act_depth", "ACTDepthPolicy"),
        "pi05depth": ("policies.pi0_depth.modeling_pi05",      "PI05DEPTHPolicy"),
    }

    if policy_type in _POLICY_CLASS_MAP:
        module_path, class_name = _POLICY_CLASS_MAP[policy_type]
        module = importlib.import_module(module_path)
        PolicyClass = getattr(module, class_name)
        policy = PolicyClass(config)
        print(f"   Instanciado: {module_path}.{class_name}")
    else:
        raise ValueError(f"Tipo '{policy_type}' não mapeado. Adicione em _POLICY_CLASS_MAP.")

    # Carrega os pesos
    model_file = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"model.safetensors não encontrado em {checkpoint_dir}")

    state_dict = load_file(model_file)
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"   ⚠️  {len(missing)} pesos ausentes (esperado com train_expert_only=True)")
    if unexpected:
        print(f"   ⚠️  {len(unexpected)} pesos inesperados")

    policy.eval()
    policy.to(device)
    print(f"✅ Política '{policy_type}' carregada!")
    return policy, policy_type


def resolve_checkpoint_path(checkpoint_dir: str | None) -> str | None:
    """Tenta resolver um checkpoint local válido se o usuário não informou um caminho real."""
    candidates = [
        "/home/breno/manipulation_policies/last/pretrained_model",
        os.path.join(current_dir, "..", "train_output", "pick_up_the_cup_nodepth", "best_val_checkpoint", "pretrained_model"),
        os.path.join(current_dir, "..", "train_output", "pick_up_the_cup_pi05_depth", "best_val_checkpoint", "pretrained_model"),
        os.path.join(current_dir, "..", "train_output", "actdepth", "best_val_checkpoint", "pretrained_model"),
        os.path.join(current_dir, "..", "train", "output", "pi05", "checkpoints", "best", "pretrained_model"),
        os.path.join(current_dir, "..", "train", "output", "checkpoints", "last", "pretrained_model"),
        os.path.expanduser("~/manipulation_policies/last/pretrained_model"),
    ]

    if checkpoint_dir:
        if checkpoint_dir.startswith("/caminho") or "seu/checkpoint" in checkpoint_dir:
            checkpoint_dir = None
        elif os.path.exists(checkpoint_dir):
            return os.path.abspath(checkpoint_dir)

    env_checkpoint = os.getenv("PROMETHEUS_CHECKPOINT")
    if env_checkpoint and os.path.exists(env_checkpoint):
        print(f"🔎 Usando checkpoint de PROMETHEUS_CHECKPOINT: {env_checkpoint}")
        return os.path.abspath(env_checkpoint)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            print(f"🔎 Checkpoint encontrado automaticamente: {candidate}")
            return os.path.abspath(candidate)

    return None


# ─────────────────────────────────────────────────────────────────────
# 3. HELPERS DE IMAGEM
# ─────────────────────────────────────────────────────────────────────
def _img_to_tensor_single(img: np.ndarray) -> torch.Tensor:
    """Converte HxWxC uint8 → [C, H, W] float32 em [0,1]. SEM batch dim.
    Usado como entrada para o preprocessor (to_batch_processor cuida do batch dim).
    """
    return (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
    )


def _img_to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Converte HxWxC uint8 → [1, C, H, W] float32 em [0,1]. COM batch dim.
    Usado para o actdepth (que não usa o preprocessor do LeRobot).
    """
    return (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )


# ─────────────────────────────────────────────────────────────────────
# 4. MONTA OBSERVAÇÃO BRUTA PARA O PREPROCESSOR (PI05)
# ─────────────────────────────────────────────────────────────────────
def make_raw_obs_for_preprocessor(
    obs: dict,
    joint_names: list[str],
    has_depth: bool = False,
    has_pressure: bool = False,
    task: str = "pick up the cup",
) -> dict:
    """
    Monta o dict de observação no formato que o preprocessor do LeRobot espera.

    Tensors SEM batch dimension — o step to_batch_processor do pipeline
    cuida de adicionar o batch dim. O preprocessor então aplica:
      normalização do estado → discretização → tokenização → move para device.
    O postprocessor aplica a desnormalização da ação.
    """
    raw = {}

    # Estado das juntas [28] — sem batch dim
    state_vector = [obs.get(name, 0.0) for name in joint_names]
    raw["observation.state"] = torch.tensor(state_vector, dtype=torch.float32)

    # RGB [C, H, W] — sem batch dim
    rgb = obs.get("head_camera")
    if rgb is not None:
        raw["observation.images.head_camera"] = _img_to_tensor_single(rgb)

    # Depth [C, H, W] — sem batch dim
    if has_depth:
        depth = obs.get("head_camera_depth")
        if depth is not None:
            if len(depth.shape) == 2:
                depth = np.stack([depth] * 3, axis=-1)
            elif depth.shape[2] == 1:
                depth = np.repeat(depth, 3, axis=-1)
            raw["observation.images.head_camera_depth"] = _img_to_tensor_single(depth)

    # Pressão [33] — sem batch dim
    if has_pressure:
        for side in ["left", "right"]:
            val = obs.get(f"{side}_hand_pressure")
            if val is not None:
                raw[f"observation.{side}_hand_pressure"] = torch.from_numpy(
                    np.array(val, dtype=np.float32)
                )

    # Task — string simples; o to_batch_processor envolve em lista
    raw["task"] = task

    return raw


# ─────────────────────────────────────────────────────────────────────
# 5. MONTA BATCH PARA ACTDEPTH (sem preprocessor do LeRobot)
# ─────────────────────────────────────────────────────────────────────
def make_batch_for_actdepth(
    obs: dict,
    joint_names: list[str],
    device: torch.device,
    has_depth: bool = False,
    has_pressure: bool = False,
) -> dict:
    """
    Monta o batch de entrada para o ACT-Depth.
    Aqui o batch dim já é adicionado pois não há preprocessor intermediário.
    """
    batch = {}

    state_vector = [obs.get(name, 0.0) for name in joint_names]
    batch["observation.state"] = (
        torch.tensor(state_vector, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )

    rgb = obs.get("head_camera")
    if rgb is not None:
        batch["observation.images.head_camera"] = _img_to_tensor(rgb, device)

    if has_depth:
        depth = obs.get("head_camera_depth")
        if depth is not None:
            if len(depth.shape) == 2:
                depth = np.stack([depth] * 3, axis=-1)
            elif depth.shape[2] == 1:
                depth = np.repeat(depth, 3, axis=-1)
            batch["observation.images.head_camera_depth"] = _img_to_tensor(depth, device)

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

    return batch


# ─────────────────────────────────────────────────────────────────────
# 6. SETUP DE CÂMERAS
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
# 7. LEITURA DE FRAME (ZMQ ou Fake)
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
            if "head_camera_depth" in msg["images"]:
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
# 8. CARREGA PREPROCESSOR E POSTPROCESSOR DO CHECKPOINT (PI05)
# ─────────────────────────────────────────────────────────────────────
def load_pi05_preprocessors(checkpoint_dir: str, policy):
    """
    Carrega o preprocessor e o postprocessor salvos junto com o checkpoint do PI05.

    O preprocessor aplica (na ordem):
      rename → to_batch → normalize (state/images) → discretize state → tokenize → to device

    O postprocessor aplica:
      unnormalize action (QUANTILES⁻¹) → to CPU

    Os pesos de normalização (q01, q99) vêm dos safetensors salvos no checkpoint:
      policy_preprocessor_step_2_normalizer_processor.safetensors
      policy_postprocessor_step_0_unnormalizer_processor.safetensors
    """
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_dir,
    )
    print("✅ Preprocessor e postprocessor PI05 carregados do checkpoint.")
    return preprocessor, postprocessor


# ─────────────────────────────────────────────────────────────────────
# 9. MAIN
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
    show_video = False
    uncertainty_threshold = 0.0
    remote_sim_ip = None

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
        elif arg.startswith("--remote-sim="):
            remote_sim_ip = arg.split("=", 1)[1]

    checkpoint_dir = resolve_checkpoint_path(checkpoint_dir)
    if checkpoint_dir is None:
        print("❌ ERRO: não encontrei um checkpoint válido localmente.")
        print("   Passe --checkpoint=<CAMINHO> para a pasta pretrained_model ou defina PROMETHEUS_CHECKPOINT.")
        print("   Exemplos de caminhos buscados:")
        print("     train_output/pick_up_the_cup_nodepth/best_val_checkpoint/pretrained_model")
        print("     train_output/pick_up_the_cup_pi05_depth/best_val_checkpoint/pretrained_model")
        print("     train_output/actdepth/best_val_checkpoint/pretrained_model")
        print("     train/output/pi05/checkpoints/best/pretrained_model")
        print("     ~/manipulation_policies/last/pretrained_model")
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

    # ── Uncertainty Gate ──────────────────────────────────────────────
    if uncertainty_threshold > 0:
        policy.config.scene_uncertainty_threshold = uncertainty_threshold
        print(f"✅ Uncertainty Gate ativado: threshold={uncertainty_threshold}")

    # ── Preprocessor e Postprocessor (somente PI05) ───────────────────
    # O preprocessor normaliza o estado, discretiza e tokeniza.
    # O postprocessor desnormaliza a ação de volta para radianos reais.
    # Ambos carregam seus pesos dos safetensors salvos no checkpoint.
    preprocessor = None
    postprocessor = None

    if policy_type == "pi05depth":
        preprocessor, postprocessor = load_pi05_preprocessors(checkpoint_dir, policy)

    # ── Câmeras ───────────────────────────────────────────────────────
    stream_client, fake_cap, fake_img_rgb = setup_cameras(
        cam_robot_ip, cam_port, fake_video_path
    )

    # ── Robô ─────────────────────────────────────────────────────────
    from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
    robot_ip = cam_robot_ip or "10.9.8.73"
    print(f"⏳ Conectando ao Unitree G1 (Simulação: {is_sim}) em {robot_ip}...")
    g1_config = UnitreeG1Dex3Config(
        robot_ip=robot_ip,
        control_mode="upper_body",
        is_simulation=is_sim,
        remote_sim_ip=remote_sim_ip,
    )
    robot = UnitreeG1Dex3(g1_config)
    robot.connect()
    print("✅ Robô conectado!")

    for cam in robot.cameras.values():
        if hasattr(cam, 'timeout_ms'):
            cam.timeout_ms = 800

    # Nomes das juntas na mesma ordem do dataset (info.json)
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
            # 1. Observação do robô
            try:
                obs = robot.get_observation()
            except TimeoutError as e:
                print(f"⚠️  Timeout de câmera: {e}. Pulando frame...")
                continue         # Aborta este ciclo e tenta de novo no topo do loop
            if not obs:
                continue

            # 2. Câmeras
            obs, fake_img_rgb = get_camera_frames(
                obs, stream_client, fake_cap, fake_img_rgb
            )

            # 3. Exibe RGB na janela (só se --v foi passado)
            rgb = obs.get("head_camera")
            if show_video and rgb is not None:
                cv2.imshow("Visao da IA", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            # ── PI05-Depth ────────────────────────────────────────────
            # O preprocessor faz tudo: normaliza estado (QUANTILES),
            # discretiza, tokeniza e move para GPU.
            # O postprocessor desnormaliza a ação (QUANTILES⁻¹).
            if policy_type == "pi05depth":

                # 4. Monta dict bruto (sem batch dim — o preprocessor cuida disso)
                raw_obs = make_raw_obs_for_preprocessor(
                    obs=obs,
                    joint_names=joint_names,
                    has_depth=has_depth,
                    has_pressure=has_pressure,
                    task="pick up the cup",
                )

                # 5. Preprocessor: normalize → discretize → tokenize → to device
                batch = preprocessor(raw_obs)

                # 👇 ADICIONE ESTE BLOCO (Garante o batch dim na pressão) 👇
                for side in ["left", "right"]:
                    k = f"observation.{side}_hand_pressure"
                    if k in batch and batch[k].dim() == 1:
                        batch[k] = batch[k].unsqueeze(0)
                # 👆 ==================================================== 👆

                # 6. Inferência
                with torch.inference_mode():
                    action = policy.select_action(batch)

                # 7. Postprocessor: desnormaliza ação → CPU
                # select_action retorna tensor [action_dim] normalizado em [-1, 1].
                # O postprocessor aplica a transformação QUANTILES⁻¹ usando
                # os q01/q99 do safetensors salvo no checkpoint.
                action = postprocessor(action)

                # 8. Converte para numpy e monta action_dict
                if isinstance(action, dict):
                    action_numpy = action["action"].squeeze(0).cpu().numpy()
                else:
                    action_numpy = action.squeeze(0).cpu().numpy()

            # ── ACT-Depth ─────────────────────────────────────────────
            else:

                # 4. Monta batch com batch dim (ACT-D não usa preprocessor LeRobot)
                batch = make_batch_for_actdepth(
                    obs=obs,
                    joint_names=joint_names,
                    device=device,
                    has_depth=has_depth,
                    has_pressure=has_pressure,
                )

                # 5. Inferência
                with torch.inference_mode():
                    with torch.autocast(
                        device_type="cuda" if device.type == "cuda" else "cpu"
                    ):
                        action = policy.select_action(batch)

                # 6. Converte para numpy
                action_numpy = action.squeeze(0).cpu().numpy()

            # 9. Debug — mostra valores reais em radianos
            if debug_mode:
                arm_vals = " | ".join([f"{v:.3f}" for v in action_numpy[:7]])
                print(f"\r🤖 [{policy_type}] braço E: [{arm_vals}]", end="", flush=True)

            # 10. Envia ao robô
            action_dict = {name: float(action_numpy[i]) for i, name in enumerate(joint_names)}
            robot.send_action(action_dict)

            # 11. Mantém ~50 Hz
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