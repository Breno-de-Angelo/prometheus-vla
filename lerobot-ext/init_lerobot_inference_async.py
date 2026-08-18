#!/usr/bin/env python
"""
Inference Entry Point — Async Local (Threading)
Mesmo suporte do v2 (actdepth, pi05depth), mas com inferência desacoplada do loop de controle.

Arquitetura:
  Thread A (inference_worker): obs_queue → política → action_queue  [GPU ~constante]
  Thread B (main/control loop): action_queue → robot.send_action()   [dt real da câmera]

Como dimensionar --chunk e --lead:
  - O log mostra "[inference] Xms" e "Loop dt: Xms" — meça esses valores reais.
  - chunk  >= ceil(inferencia_ms / loop_dt_ms)  →  nunca zera o buffer durante inferência
  - lead   ~= chunk                              →  pede nova inferência com 1 ciclo de antecedência

  Exemplo típico G1 + PI05: inferencia=1600ms, loop_dt=33ms
    chunk = ceil(1600/33) = 49  →  use --chunk=60  (margem de segurança)
    lead  = 50                  →  use --lead=50

Uso:
  python init_lerobot_inference_async.py --checkpoint=<CAMINHO> [OPÇÕES]

Opções:
  --checkpoint=<PATH>    (obrigatório) Caminho para o pretrained_model
  --sim                  Modo simulação (sem robô real)
  --cam-robot=<IP>       Stream ZMQ de câmera externa
  --port-cam=<PORTA>     Porta do stream (padrão: 5555)
  --fake-video=<PATH>    Injeta imagem ou vídeo na câmera
  --uncertainty=<FLOAT>  Ativa o uncertainty gate (ex: 0.1)
  --chunk=<INT>          Ações por chunk (padrão: 60). Deve cobrir 1 inferência inteira.
  --lead=<INT>           Pede nova inferência quando restam N ações no buffer (padrão: 50).
                         Deve ser ~= chunk para evitar gap. Máximo = chunk.
  --v                    Abre janela de visualização da câmera
  --debug                Loga ações e tempos no terminal em tempo real
  -h, --help             Mostra esta mensagem

Exemplos:
  # PI05-Depth (inferência ~1600ms, loop ~33ms → chunk=60, lead=50):
  python init_lerobot_inference_async.py \
      --checkpoint=train_output/pi05/best_val_checkpoint/pretrained_model \
      --chunk=60 --lead=50 --debug

  # ACT-Depth mais rápido (inferência ~200ms, loop ~33ms → chunk=10, lead=8):
  python init_lerobot_inference_async.py \
      --checkpoint=train_output/pick_up_the_cup_nodepth/best_val_checkpoint/pretrained_model \
      --chunk=10 --lead=8

  # Com câmera ZMQ e visualização:
  python init_lerobot_inference_async.py \
      --checkpoint=train_output/pi05/best_val_checkpoint/pretrained_model \
      --cam-robot=192.168.123.164 --v --chunk=60 --lead=50
"""

import os
import sys
import time
import threading
from queue import Queue, Empty, Full

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
# 2. LOADER UNIVERSAL
# ─────────────────────────────────────────────────────────────────────
def load_policy(checkpoint_dir: str, device: torch.device):
    print(f"⏳ Carregando política de: {checkpoint_dir}")

    config = PreTrainedConfig.from_pretrained(checkpoint_dir)
    policy_type = getattr(config, "type", "desconhecido")
    print(f"   Tipo detectado: {policy_type}")

    from safetensors.torch import load_file
    import importlib

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


# ─────────────────────────────────────────────────────────────────────
# 3. HELPERS DE IMAGEM
# ─────────────────────────────────────────────────────────────────────
def _depth_to_tensor(depth: "np.ndarray", device=None) -> "torch.Tensor":
    """Mapa de profundidade → `[1, H, W]` float32 em MILÍMETROS.

    Formato nativo da 0.6.1: 1 canal, valor métrico. O caminho antigo replicava
    em 3 canais e dividia por 255, porque a profundidade era gravada como
    imagem de 8 bits (0–2000 mm espremidos em 0–255). Fazer isso hoje não
    quebra nada visivelmente — só entrega milímetros divididos por 255 à
    política, que espera milímetros. Erro silencioso, o pior tipo.

    A política converte mm → metros na projeção 3D
    (`policies/act_depth/depth_encoder.py::depth_to_pointcloud`, `depth_unit`).
    """
    import numpy as _np

    depth = _np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(
            f"Profundidade deveria ser um mapa de 1 canal [H, W], veio {depth.shape}. "
            "Se o servidor ainda publica cinza de 3 canais, atualize-o "
            "(Scripts_Prometheus_int/full_realsenser_server.py)."
        )
    tensor = torch.from_numpy(_np.ascontiguousarray(depth)).float().unsqueeze(0)
    return tensor if device is None else tensor.to(device)


def _img_to_tensor_single(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)


def _img_to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────────────
# 4. MONTA OBS BRUTA PARA O PREPROCESSOR (PI05)
# ─────────────────────────────────────────────────────────────────────
def make_raw_obs_for_preprocessor(
    obs, joint_names, has_depth=False, has_pressure=False, task="pick up the cup"
):
    raw = {}
    state_vector = [obs.get(name, 0.0) for name in joint_names]
    raw["observation.state"] = torch.tensor(state_vector, dtype=torch.float32)

    rgb = obs.get("head_camera")
    if rgb is not None:
        raw["observation.images.head_camera"] = _img_to_tensor_single(rgb)

    if has_depth:
        depth = obs.get("head_camera_depth")
        if depth is not None:
            raw["observation.images.head_camera_depth"] = _depth_to_tensor(depth)

    if has_pressure:
        for side in ["left", "right"]:
            val = obs.get(f"{side}_hand_pressure")
            if val is not None:
                raw[f"observation.{side}_hand_pressure"] = torch.from_numpy(
                    np.array(val, dtype=np.float32)
                )

    raw["task"] = task
    return raw


# ─────────────────────────────────────────────────────────────────────
# 5. MONTA BATCH PARA ACTDEPTH
# ─────────────────────────────────────────────────────────────────────
def make_batch_for_actdepth(obs, joint_names, device, has_depth=False, has_pressure=False):
    batch = {}
    state_vector = [obs.get(name, 0.0) for name in joint_names]
    batch["observation.state"] = (
        torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(device)
    )

    rgb = obs.get("head_camera")
    if rgb is not None:
        batch["observation.images.head_camera"] = _img_to_tensor(rgb, device)

    if has_depth:
        depth = obs.get("head_camera_depth")
        if depth is not None:
            batch["observation.images.head_camera_depth"] = _depth_to_tensor(depth, device).unsqueeze(0)

    if has_pressure:
        for side in ["left", "right"]:
            key = f"{side}_hand_pressure"
            val = obs.get(key)
            if val is not None:
                batch[f"observation.{key}"] = (
                    torch.from_numpy(np.array(val, dtype=np.float32)).unsqueeze(0).to(device)
                )
    return batch


# ─────────────────────────────────────────────────────────────────────
# 6. SETUP DE CÂMERAS / LEITURA DE FRAME
# ─────────────────────────────────────────────────────────────────────
def setup_cameras(cam_robot_ip, cam_port, fake_video_path):
    from Scripts_Prometheus_int.sim.sensor_utils import SensorClient, ImageUtils  # noqa: F401

    stream_client = fake_cap = fake_img_rgb = None

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


def get_camera_frames(obs, stream_client, fake_cap, fake_img_rgb):
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
# 7. CARREGA PREPROCESSOR / POSTPROCESSOR (PI05)
# ─────────────────────────────────────────────────────────────────────
def load_pi05_preprocessors(checkpoint_dir, policy):
    from lerobot.policies.factory import make_pre_post_processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_dir,
    )
    print("✅ Preprocessor e postprocessor PI05 carregados do checkpoint.")
    return preprocessor, postprocessor


# ─────────────────────────────────────────────────────────────────────
# 8. THREAD DE INFERÊNCIA
# ─────────────────────────────────────────────────────────────────────
def inference_worker(
    *,
    obs_queue: Queue,
    action_queue: Queue,
    stop_event: threading.Event,
    policy,
    policy_type: str,
    preprocessor,
    postprocessor,
    joint_names: list,
    device: torch.device,
    has_depth: bool,
    has_pressure: bool,
    actions_per_chunk: int,
    debug: bool,
):
    """
    Roda em thread separada.
    Consome obs_queue → inferência → empurra chunk na action_queue.
    Nunca dorme entre inferências: a próxima começa assim que a obs_queue tiver uma nova obs.
    """
    print("🧠 [inference_worker] Iniciada.")

    while not stop_event.is_set():
        # Bloqueia até chegar nova obs (timeout curto para checar stop_event)
        try:
            obs = obs_queue.get(timeout=0.5)
        except Empty:
            continue

        t0 = time.perf_counter()

        try:
            # ── PI05-Depth ────────────────────────────────────────────
            if policy_type == "pi05depth":
                raw_obs = make_raw_obs_for_preprocessor(
                    obs, joint_names, has_depth=has_depth,
                    has_pressure=has_pressure, task="pick up the cup",
                )
                batch = preprocessor(raw_obs)

                for side in ["left", "right"]:
                    k = f"observation.{side}_hand_pressure"
                    if k in batch and batch[k].dim() == 1:
                        batch[k] = batch[k].unsqueeze(0)

                with torch.inference_mode():
                    if hasattr(policy, "predict_action_chunk"):
                        raw_chunk = policy.predict_action_chunk(batch)        # (1, T, D)
                        raw_chunk = raw_chunk[0, :actions_per_chunk, :]       # (T, D)
                        chunk_np = []
                        for i in range(raw_chunk.shape[0]):
                            a = postprocessor(raw_chunk[i].unsqueeze(0))
                            if isinstance(a, dict):
                                a = a["action"]
                            chunk_np.append(a.squeeze(0).cpu().numpy())
                    else:
                        action = policy.select_action(batch)
                        action = postprocessor(action)
                        if isinstance(action, dict):
                            action = action["action"]
                        chunk_np = [action.squeeze(0).cpu().numpy()]

            # ── ACT-Depth ─────────────────────────────────────────────
            else:
                batch = make_batch_for_actdepth(
                    obs, joint_names, device, has_depth=has_depth, has_pressure=has_pressure
                )

                with torch.inference_mode():
                    with torch.autocast(
                        device_type="cuda" if device.type == "cuda" else "cpu"
                    ):
                        if hasattr(policy, "predict_action_chunk"):
                            raw_chunk = policy.predict_action_chunk(batch)    # (1, T, D)
                            raw_chunk = raw_chunk[0, :actions_per_chunk, :]   # (T, D)
                            chunk_np = [raw_chunk[i].cpu().numpy() for i in range(raw_chunk.shape[0])]
                        else:
                            action = policy.select_action(batch)
                            chunk_np = [action.squeeze(0).cpu().numpy()]

            t_inf_ms = (time.perf_counter() - t0) * 1000

            if debug:
                print(f"\n🧠 [inference] {t_inf_ms:.1f}ms | chunk={len(chunk_np)} ações")

            # Descarta chunk antigo se a fila estiver cheia, insere o novo
            try:
                action_queue.put_nowait(chunk_np)
            except Full:
                try:
                    action_queue.get_nowait()
                except Empty:
                    pass
                action_queue.put_nowait(chunk_np)

        except Exception as e:
            print(f"\n❌ [inference_worker] Erro: {e}")

    print("🧠 [inference_worker] Encerrada.")


# ─────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
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
    actions_per_chunk = 60       # deve cobrir 1 inferência inteira: ceil(inf_ms / loop_dt_ms)
    lead_actions = 50            # pede nova inferência quando restam N ações no buffer

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
        elif arg.startswith("--chunk="):
            actions_per_chunk = int(arg.split("=", 1)[1])
        elif arg.startswith("--lead="):
            lead_actions = int(arg.split("=", 1)[1])
        elif arg == "--debug":
            debug_mode = True
        elif arg == "--v":
            show_video = True
        elif arg.startswith("--remote-sim="):
            remote_sim_ip = arg.split("=", 1)[1]

    # lead não pode ser maior que chunk
    lead_actions = min(lead_actions, actions_per_chunk)

    if checkpoint_dir is None:
        print("❌ ERRO: --checkpoint obrigatório.")
        sys.exit(1)

    # ── Dispositivo ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Usando device: {device}")

    # ── Carrega política ──────────────────────────────────────────────
    policy, policy_type = load_policy(checkpoint_dir, device)

    has_depth = getattr(policy.config, "use_depth_3d", False)
    has_pressure = getattr(policy.config, "use_pressure", False)
    print(f"   Depth 3D: {has_depth} | Pressão: {has_pressure}")

    if uncertainty_threshold > 0:
        policy.config.scene_uncertainty_threshold = uncertainty_threshold
        print(f"✅ Uncertainty Gate ativado: threshold={uncertainty_threshold}")

    # ── Preprocessors (PI05 only) ─────────────────────────────────────
    preprocessor = postprocessor = None
    if policy_type == "pi05depth":
        preprocessor, postprocessor = load_pi05_preprocessors(checkpoint_dir, policy)

    # ── Câmeras ───────────────────────────────────────────────────────
    stream_client, fake_cap, fake_img_rgb = setup_cameras(cam_robot_ip, cam_port, fake_video_path)

    # ── Robô ─────────────────────────────────────────────────────────
    from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
    print(f"⏳ Conectando ao Unitree G1 (Simulação: {is_sim})...")
    g1_config = UnitreeG1Dex3Config(
        robot_ip="10.9.8.73",
        control_mode="upper_body",
        is_simulation=is_sim,
        remote_sim_ip=remote_sim_ip,
    )
    robot = UnitreeG1Dex3(g1_config)
    robot.connect()
    print("✅ Robô conectado!")

    for cam in robot.cameras.values():
        if hasattr(cam, "timeout_ms"):
            cam.timeout_ms = 800

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

    # ── Filas de comunicação entre threads ───────────────────────────
    #   obs_queue:    maxsize=1  → inferência sempre usa obs mais recente
    #   action_queue: maxsize=2  → no máximo 2 chunks em fila (evita ações obsoletas)
    obs_queue    = Queue(maxsize=1)
    action_queue = Queue(maxsize=2)

    stop_event = threading.Event()

    # ── Inicia thread de inferência ───────────────────────────────────
    inf_thread = threading.Thread(
        target=inference_worker,
        kwargs=dict(
            obs_queue=obs_queue,
            action_queue=action_queue,
            stop_event=stop_event,
            policy=policy,
            policy_type=policy_type,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            joint_names=joint_names,
            device=device,
            has_depth=has_depth,
            has_pressure=has_pressure,
            actions_per_chunk=actions_per_chunk,
            debug=debug_mode,
        ),
        daemon=True,
        name="inference_worker",
    )
    inf_thread.start()

    print(f"\n🚀 INFERÊNCIA ASYNC ATIVA [{policy_type.upper()}]")
    print(f"   chunk={actions_per_chunk} ações | pede nova inferência quando buf <= {lead_actions}")
    if show_video:
        print("   📺 Janela de câmera ativa.")
    print("   Ctrl+C para parar.\n")

    # Buffer local de ações do chunk atual (lista de np.ndarray)
    current_chunk: list = []

    # Contadores de diagnóstico (resetados a cada 100 ciclos)
    _diag_loops = 0
    _diag_elapsed_sum = 0.0

    # ─────────────────────────────────────────────────────────────────
    # LOOP PRINCIPAL (thread principal — dt determinado pela câmera)
    # ─────────────────────────────────────────────────────────────────
    try:
        while True:
            start_t = time.perf_counter()

            # 1. Lê observação do robô (bloqueia até câmera retornar)
            obs_valid = True
            try:
                obs = robot.get_observation()
            except TimeoutError as e:
                print(f"\n⚠️  Timeout de câmera: {e}. Mantendo ação do buffer.")
                obs_valid = False
                obs = None
            if obs is not None and not obs:
                obs_valid = False

            # 2. Câmeras externas (ZMQ / fake) — só se obs é válida
            if obs_valid and obs is not None:
                obs, fake_img_rgb = get_camera_frames(obs, stream_client, fake_cap, fake_img_rgb)

            # 3. Exibe RGB na janela (só com obs válida)
            if obs_valid and show_video:
                rgb = obs.get("head_camera")
                if rgb is not None:
                    cv2.imshow("Visao da IA", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

            # 4. Pede nova inferência se o buffer estiver baixo E a obs for válida
            #    Em timeout: não alimenta a política com câmera ruim
            if obs_valid and len(current_chunk) <= lead_actions:
                try:
                    obs_queue.put_nowait(obs)
                except Full:
                    try:
                        obs_queue.get_nowait()
                    except Empty:
                        pass
                    obs_queue.put_nowait(obs)

            # 5. Pega chunk novo da action_queue se disponível
            #    Limita o buffer a actions_per_chunk para não acumular ações velhas
            #    (evita buf=700 que acontece quando timeouts pausam o consumo)
            if not action_queue.empty():
                try:
                    new_chunk = action_queue.get_nowait()
                    # Concatena preservando continuidade, mas cap no máximo do chunk
                    combined = current_chunk + new_chunk
                    current_chunk = combined[:actions_per_chunk]
                except Empty:
                    pass

            # 6. Executa próxima ação do buffer local
            if current_chunk:
                action_numpy = current_chunk.pop(0)

                if debug_mode:
                    arm_vals = " | ".join([f"{v:.3f}" for v in action_numpy[:7]])
                    print(f"\r🤖 [{policy_type}] E: [{arm_vals}] buf={len(current_chunk)}", end="", flush=True)

                action_dict = {name: float(action_numpy[i]) for i, name in enumerate(joint_names)}
                robot.send_action(action_dict)
            else:
                if debug_mode:
                    print("\r⏳ Aguardando 1º chunk de inferência...", end="", flush=True)

            # 7. Diagnóstico periódico do dt real do loop (sem sleep — câmera já regula o ritmo)
            elapsed = time.perf_counter() - start_t
            _diag_elapsed_sum += elapsed
            _diag_loops += 1
            if _diag_loops >= 100:
                avg_dt_ms = (_diag_elapsed_sum / _diag_loops) * 1000
                if debug_mode:
                    print(f"\n📊 Loop dt médio (100 ciclos): {avg_dt_ms:.1f}ms  ({1000/avg_dt_ms:.1f} Hz)")
                    print(f"   Sugestão: --chunk >= {int(1600/avg_dt_ms)+5}  --lead >= {int(1600/avg_dt_ms)}")
                _diag_loops = 0
                _diag_elapsed_sum = 0.0

    except KeyboardInterrupt:
        print("\n🛑 Parando inferência...")

    finally:
        stop_event.set()
        inf_thread.join(timeout=3.0)

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