#!/usr/bin/env python
"""
Inference Entry Point V3 — Universal
Suporta: actdepth, pi05depth, openvladepth

Políticas condicionadas por linguagem (`pi05depth`, `openvladepth`) EXIGEM
`--task`: é o texto desse argumento que vira o prompt e decide qual tarefa o
robô executa. `actdepth` não usa linguagem e ignora o argumento.

Uso:
  python init_lerobot_inference_v3.py --checkpoint=<CAMINHO> [OPÇÕES]

Opções:
  --checkpoint=<PATH>    (obrigatório) Caminho para o pretrained_model
  --task="<TEXTO>"       Comando em linguagem natural (obrigatório para VLAs)
  --interactive          Permite trocar o comando digitando durante a execução
  --sim                  Modo simulação (sem robô real)
  --cam-robot=<IP>       Stream ZMQ de câmera externa
  --port-cam=<PORTA>     Porta do stream (padrão: 5555)
  --fake-video=<PATH>    Injeta imagem ou vídeo na câmera
  --uncertainty=<FLOAT>  Ativa o uncertainty gate (ex: 0.1)
  --v                    Abre janela de visualização da câmera
  --debug                Loga ações no terminal em tempo real
  -h, --help             Mostra esta mensagem

Exemplos:
  # ACT-D (sem linguagem):
  python init_lerobot_inference_v3.py \
      --checkpoint=train_output/pick_up_the_cup_nodepth/best_val_checkpoint/pretrained_model

  # OpenVLA-Depth por comando de texto:
  python init_lerobot_inference_v3.py \
      --checkpoint=train/output/openvla_depth_cup_2026-06-09/best_val_checkpoint/pretrained_model \
      --task="pick up the white mug and place it to the right" \
      --cam-robot=192.168.123.164 --v

  # Trocando o comando durante a execução:
  python init_lerobot_inference_v3.py --checkpoint=<PATH> \
      --task="pick up the white mug and place it to the right" --interactive

O texto precisa casar com o que foi gravado no campo `task` do dataset. Ver
docs/INFERENCIA_COMANDO_TEXTO.md.
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
        "actdepth":     ("policies.act_depth.modeling_act", "ACTPolicy"),
        "pi05depth":    ("policies.pi0_depth.modeling_pi05", "PI05DEPTHPolicy"),
        "openvladepth": ("policies.openvla_depth.modeling_openvla", "OPENVLADEPTHPolicy"),
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
        # Pesos do VAE encoder são esperados ausentes em inferência (não são usados)
        vae_prefixes = (
            "model.vae_encoder", "model.vae_encoder_cls_embed",
            "model.vae_encoder_robot_state_input_proj",
            "model.vae_encoder_action_input_proj",
            "model.vae_encoder_latent_output_proj",
        )
        real_missing = [k for k in missing if not any(k.startswith(p) for p in vae_prefixes)]
        if real_missing:
            print(f"   ⚠️  {len(real_missing)} pesos ausentes inesperados:")
            for k in real_missing[:10]:
                print(f"      - {k}")
        else:
            print(f"   ✅ {len(missing)} ausentes = VAE encoder (normal em inferência)")
    if unexpected:
        print(f"   ⚠️  {len(unexpected)} pesos inesperados")

    policy.eval()
    policy.to(device)
    print(f"✅ Política '{policy_type}' carregada!")
    return policy, policy_type


# ─────────────────────────────────────────────────────────────────────
# 3. CARREGA PREPROCESSOR E POSTPROCESSOR DO CHECKPOINT
# ─────────────────────────────────────────────────────────────────────
def load_pre_post_processors(checkpoint_dir: str, policy):
    """
    Carrega preprocessor e postprocessor salvos junto com o checkpoint.

    Ambas as políticas usam arquivos externos de normalização:

    ACT  (MEAN_STD):
      preprocessor → rename, to_batch, device, normalize(images+state com mean/std)
      postprocessor → unnormalize(action com mean/std inverso), cpu

    PI05 (QUANTILES):
      preprocessor → rename, to_batch, normalize(state com q01/q99), discretize, tokenize, device
      postprocessor → unnormalize(action com q01/q99 inverso), cpu

    Os pesos vêm dos safetensors no checkpoint:
      policy_preprocessor_step_N_normalizer_processor.safetensors
      policy_postprocessor_step_0_unnormalizer_processor.safetensors
    """
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_dir,
    )
    print("✅ Preprocessor e postprocessor carregados do checkpoint.")
    return preprocessor, postprocessor


# ─────────────────────────────────────────────────────────────────────
# 4. MONTA OBSERVAÇÃO BRUTA (ACT e PI05 usam a mesma função)
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


def make_raw_obs(
    obs: dict,
    joint_names: list[str],
    has_depth: bool = False,
    has_pressure: bool = False,
    task: str | None = None,
) -> dict:
    """
    Monta o dict de observação SEM batch dim e SEM normalização.

    O preprocessor cuida de tudo:
      - Batch dim (to_batch_processor)
      - ACT:  normaliza imagens com ImageNet mean/std e estado com MEAN_STD do dataset
      - PI05: normaliza estado com QUANTILES, discretiza e tokeniza

    Parâmetros:
      task: None para ACT (não usa linguagem), "pick up the cup" para PI05
    """
    raw = {}

    # Estado das juntas [28] — radianos brutos, sem normalização
    state_vector = [obs.get(name, 0.0) for name in joint_names]
    raw["observation.state"] = torch.tensor(state_vector, dtype=torch.float32)

    # RGB [C, H, W] em [0, 1] — preprocessor aplica ImageNet mean/std (ACT) ou nada (PI05 IDENTITY)
    rgb = obs.get("head_camera")
    if rgb is not None:
        raw["observation.images.head_camera"] = (
            torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        )

    # Depth [C, H, W] em [0, 1]
    if has_depth:
        depth = obs.get("head_camera_depth")
        if depth is not None:
            raw["observation.images.head_camera_depth"] = _depth_to_tensor(depth)

    # Pressão [33] — sem batch dim
    if has_pressure:
        for side in ["left", "right"]:
            val = obs.get(f"{side}_hand_pressure")
            if val is not None:
                raw[f"observation.{side}_hand_pressure"] = torch.from_numpy(
                    np.array(val, dtype=np.float32)
                )

    # Task string — só PI05 usa; o preprocessor tokeniza internamente
    if task is not None:
        raw["task"] = task

    return raw


# ─────────────────────────────────────────────────────────────────────
# 5. SETUP DE CÂMERAS
# ─────────────────────────────────────────────────────────────────────
def setup_cameras(cam_robot_ip, cam_port, fake_video_path):
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
# 6. LEITURA DE FRAME
# ─────────────────────────────────────────────────────────────────────
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
# 7. MAIN
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
    fps = 30
    task_cli = None
    interactive = False

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
        elif arg.startswith("--fps="):
            fps = int(arg.split("=", 1)[1])
        elif arg.startswith("--task="):
            task_cli = arg.split("=", 1)[1]
        elif arg == "--interactive":
            interactive = True

    if checkpoint_dir is None:
        print("❌ ERRO: --checkpoint obrigatório.")
        print("   Uso: python init_lerobot_inference_v3.py --checkpoint=<CAMINHO>")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Usando device: {device}")

    # ── Carrega política ──────────────────────────────────────────────
    policy, policy_type = load_policy(checkpoint_dir, device)

    has_depth    = getattr(policy.config, "use_depth_3d", False)
    has_pressure = getattr(policy.config, "use_pressure", False)
    print(f"   Depth 3D: {has_depth} | Pressão: {has_pressure}")

    if uncertainty_threshold > 0:
        policy.config.scene_uncertainty_threshold = uncertainty_threshold
        print(f"✅ Uncertainty Gate ativado: threshold={uncertainty_threshold}")

    # ── Preprocessor e Postprocessor ─────────────────────────────────
    # Ambas as políticas (ACT e PI05) têm preprocessor/postprocessor
    # salvos no checkpoint com seus pesos de normalização.
    # O preprocessor normaliza entradas; o postprocessor desnormaliza a saída.
    preprocessor, postprocessor = load_pre_post_processors(checkpoint_dir, policy)

    # ── Comando em texto ──────────────────────────────────────────────
    # `pi05depth` e `openvladepth` são condicionados por linguagem: o texto aqui
    # vira o prompt do modelo e é o que decide qual tarefa ele executa. `actdepth`
    # não usa linguagem nenhuma e ignora isto.
    #
    # O comando fica numa lista de um elemento para poder ser trocado em tempo de
    # execução pela thread de --interactive (ver command_listener).
    LANGUAGE_POLICIES = {"pi05depth", "openvladepth"}
    task_box = [None]

    if policy_type in LANGUAGE_POLICIES:
        if task_cli is None:
            print(
                f"\n❌ ERRO: a política '{policy_type}' é condicionada por linguagem e "
                f"precisa de um comando.\n"
                f'   Use: --task="pick up the white mug and place it to the right"\n'
                f"   O texto deve casar com o que foi gravado no campo `task` do dataset.\n"
            )
            sys.exit(1)
        task_box[0] = task_cli
        print(f'🗣️  Comando: "{task_cli}"')
    elif task_cli is not None:
        print(f"⚠️  --task ignorado: '{policy_type}' não usa linguagem.")

    # Troca de comando em tempo de execução: cada linha digitada vira o novo
    # prompt. `policy.reset()` é essencial — sem ele o robô termina o chunk
    # antigo (até 50 passos, ~1,7 s a 30 fps) antes de obedecer ao comando novo.
    if interactive and policy_type in LANGUAGE_POLICIES:
        import threading

        def command_listener():
            print("\n💬 Modo interativo: digite um comando e Enter para trocar a tarefa.")
            print("   (Ctrl+C encerra a inferência)\n")
            for line in sys.stdin:
                novo = line.strip()
                if not novo:
                    continue
                task_box[0] = novo
                policy.reset()          # descarta o chunk em execução
                print(f'🗣️  Comando trocado para: "{novo}"')

        threading.Thread(target=command_listener, daemon=True, name="CommandListener").start()

    # ── Câmeras ───────────────────────────────────────────────────────
    stream_client, fake_cap, fake_img_rgb = setup_cameras(
        cam_robot_ip, cam_port, fake_video_path
    )

    # ── Robô ─────────────────────────────────────────────────────────
    from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
    print(f"⏳ Conectando ao Unitree G1 (Simulação: {is_sim})...")
    g1_config = UnitreeG1Dex3Config(
        robot_ip="10.9.8.73",
        #robot_ip="192.168.123.164",
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
    print(f"   ⏱️  FPS: {fps} ({1000/fps:.1f}ms por ciclo)")
    print("   Ctrl+C para parar.\n")

    # ─────────────────────────────────────────────────────────────────
    # LOOP PRINCIPAL
    # ─────────────────────────────────────────────────────────────────
    try:
        while True:
            start_t = time.perf_counter()

            # 1. Observação do robô
            try:
                obs = robot.get_observation()
            except TimeoutError as e:
                print(f"⚠️  Timeout de câmera: {e}. Pulando frame...")
                continue
            if not obs:
                continue

            # 2. Câmeras externas
            obs, fake_img_rgb = get_camera_frames(
                obs, stream_client, fake_cap, fake_img_rgb
            )

            # 3. Visualização
            rgb = obs.get("head_camera")
            if show_video and rgb is not None:
                cv2.imshow("Visao da IA", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            # 4. Monta observação bruta (sem batch dim, sem normalização)
            raw_obs = make_raw_obs(
                obs=obs,
                joint_names=joint_names,
                has_depth=has_depth,
                has_pressure=has_pressure,
                task=task_box[0],   # lido a cada passo: --interactive pode ter trocado
            )

            # 5. Preprocessor: normaliza + batch dim + device
            #    ACT:  mean/std nas imagens (ImageNet) e estado (dataset stats)
            #    PI05: quantiles no estado, discretiza, tokeniza
            batch = preprocessor(raw_obs)

            # Remove "action" do batch — o preprocessor define essa chave nos features
            # para normalizar targets durante treino, mas em inferência ela não existe
            # e fica None. Com ela no batch o ACT entra no VAE encoder e crasha.
            batch.pop("action", None)

            # Filtra o batch: mantém só tensors.
            # O preprocessor pode deixar valores escalares (floats, ints, strings)
            # de metadata interna no dict. O modelo faz next(iter(batch.values())).shape[0]
            # para inferir B, então qualquer não-tensor causa AttributeError.
            batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Workaround: garante batch dim na pressão se o preprocessor não adicionar
            if has_pressure:
                for side in ["left", "right"]:
                    k = f"observation.{side}_hand_pressure"
                    if k in batch and batch[k].dim() == 1:
                        batch[k] = batch[k].unsqueeze(0)

            # 6. Inferência
            with torch.inference_mode():
                action = policy.select_action(batch)

            # 7. Postprocessor: desnormaliza ação → radianos reais → CPU
            #    ACT:  action * std + mean   (MEAN_STD inverso)
            #    PI05: (action + 1) / 2 * (q99 - q01) + q01   (QUANTILES inverso)
            action = postprocessor(action)

            # 8. Converte para numpy
            if isinstance(action, dict):
                action_numpy = action["action"].squeeze(0).cpu().numpy()
            else:
                action_numpy = action.squeeze(0).cpu().numpy()

            # 9. Debug — valores devem estar em radianos reais, não em [-1, 1]
            if debug_mode:
                arm = " | ".join([f"{v:.3f}" for v in action_numpy[:7]])
                print(f"\r🤖 [{policy_type}] braço E: [{arm}]", end="", flush=True)

            # 10. Envia ao robô
            action_dict = {name: float(action_numpy[i]) for i, name in enumerate(joint_names)}
            robot.send_action(action_dict)

            # 11. Limita ao fps configurado (padrão: 30Hz)
            elapsed = time.perf_counter() - start_t
            sleep_time = max(0.0, (1.0 / fps) - elapsed)
            if debug_mode and elapsed > (1.0 / fps):
                print(f"\n⚠️  Loop lento: {elapsed*1000:.1f}ms (limite: {1000/fps:.1f}ms)")
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