#!/usr/bin/env python
"""
Inference Entry Point — Async Local (Threading)
Mesmo suporte do v3 (actdepth, pi05depth), mas com inferência desacoplada do loop de controle.

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
  --v-control            Abre janela de visualização com controles de player:
                         pause/resume, velocidade (0.25x–4x), seek, e painéis
                         laterais de contraste e brilho para simular
                         degradação de vídeo.
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
# 0. PLAYER COM CONTROLES DE VÍDEO (--v-control)
# ─────────────────────────────────────────────────────────────────────

class VideoControlWindow:
    """
    Janela OpenCV com controles de player e ajuste de imagem.

    Todos os controles são via TECLADO — zero APIs de janela Qt
    (createTrackbar / setMouseCallback crasham com NULL handler no Qt backend).

    Painel inferior-esquerdo — Player:
      [SPACE]   Pause / Resume
      [← →]     Seek  -5 / +5 frames  (só com --fake-video)
      [1-5]     Velocidade  0.25× 0.5× 1× 2× 4×
      [Q / ESC] Fechar

    Painel inferior-direito — Imagem:
      [C] / [c]  Contraste  +5 / -5   (range 0-200, neutro=100)
      [B] / [b]  Brilho     +5 / -5   (range 0-200, neutro=100)
      [R]        Reset contraste e brilho para 100

    Como usar:
      vc = VideoControlWindow("Visao da IA")
      vc.create()                        # namedWindow apenas
      frame_out = vc.process(rgb_frame)  # aplica contraste/brilho
      alive     = vc.show(frame_out)     # exibe + lê teclado
      paused, delay_ms, seek = vc.state()
    """

    SPEEDS     = [0.25, 0.5, 1.0, 2.0, 4.0]
    SPEED_KEYS = {ord(str(i + 1)): i for i in range(5)}
    SEEK_STEP  = 5
    IMG_STEP   = 5    # passo de contraste/brilho por tecla

    PANEL_H  = 120
    RIGHT_W  = 280
    BAR_H    = 14

    def __init__(self, window_name: str = "Visao da IA"):
        self.win         = window_name
        self._paused     = False
        self._speed_idx  = 2
        self._seek_delta = 0
        self._frame_idx  = 0
        self._contrast   = 100   # 0-200, int  (100 = neutro)
        self._brightness = 100   # 0-200, int  (100 = neutro)
        self._lock       = threading.Lock()

    # ── Criação da janela ─────────────────────────────────────────────
    # Só namedWindow — nenhuma outra API de janela Qt aqui.
    def create(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 960, 600)

    # ── Aplica contraste e brilho ao frame ────────────────────────────
    def process(self, frame_rgb: np.ndarray) -> np.ndarray:
        with self._lock:
            contrast   = self._contrast / 100.0
            brightness = self._brightness - 100
        out = frame_rgb.astype(np.float32) * contrast + brightness
        return np.clip(out, 0, 255).astype(np.uint8)

    # ── Desenha painel de controles sobre o frame ─────────────────────
    def _draw_panel(self, canvas: np.ndarray) -> np.ndarray:
        h, w = canvas.shape[:2]

        # Fundo semitransparente na faixa inferior
        ph  = self.PANEL_H
        roi = canvas[h - ph:h, :]
        cv2.addWeighted(roi, 0.25, np.zeros_like(roi), 0.75, 0, roi)
        canvas[h - ph:h, :] = roi

        with self._lock:
            speed      = self.SPEEDS[self._speed_idx]
            paused     = self._paused
            contrast   = self._contrast
            brightness = self._brightness
            fidx       = self._frame_idx

        status = "|| PAUSADO" if paused else f"> {speed:.2f}x"
        sc     = (80, 180, 255) if paused else (80, 255, 130)

        # ── Coluna esquerda: player ───────────────────────────────────
        x0, y0, dy = 10, h - ph + 18, 18
        for i, (txt, col, fs) in enumerate([
            ("PLAYER",               (210, 210, 210), 0.42),
            (status,                 sc,              0.50),
            ("[SPACE] Pause/Resume", (160, 160, 160), 0.35),
            ("[< >]  Seek +/-5 fr.", (160, 160, 160), 0.35),
            ("[1-5]  Velocidade",    (160, 160, 160), 0.35),
            (f"Frame {fidx}",        (120, 120, 120), 0.32),
            ("[Q/ESC] Fechar",       (160, 80,  80),  0.35),
        ]):
            cv2.putText(canvas, txt, (x0, y0 + i * dy),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)

        # ── Coluna direita: contraste e brilho ────────────────────────
        rx0 = w - self.RIGHT_W + 10
        rx1 = w - 10
        bar_w = rx1 - rx0

        def draw_bar(label, key_up, key_dn, value, ry, color):
            txt = f"{label} [{key_up}/{key_dn}]: {value - 100:+d}%"
            cv2.putText(canvas, txt, (rx0, ry - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (rx0, ry), (rx1, ry + self.BAR_H), (50, 50, 50), -1)
            fill = rx0 + int(value / 200 * bar_w)
            cv2.rectangle(canvas, (rx0, ry), (fill, ry + self.BAR_H), color, -1)
            mid = rx0 + bar_w // 2
            cv2.line(canvas, (mid, ry), (mid, ry + self.BAR_H), (255, 255, 255), 1)
            cv2.circle(canvas, (fill, ry + self.BAR_H // 2), 6, (255, 255, 255), -1)

        ry_c = h - ph + 18
        ry_b = ry_c + self.BAR_H + 30
        draw_bar("Contraste", "=", "-", contrast,   ry_c, (100, 200, 255))
        draw_bar("Brilho",    "]", "[", brightness,  ry_b, (255, 180,  80))
        cv2.putText(canvas, "[R] Reset imagem", (rx0, ry_b + self.BAR_H + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1, cv2.LINE_AA)

        return canvas

    # ── Exibe frame e lê teclado; retorna False para fechar ───────────
    def show(self, frame_rgb: np.ndarray) -> bool:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        bgr = self._draw_panel(bgr)
        cv2.imshow(self.win, bgr)

        with self._lock:
            self._frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):                       # Q / ESC → fechar
            return False
        elif key == ord(' '):
            with self._lock:
                self._paused = not self._paused
        elif key in self.SPEED_KEYS:
            with self._lock:
                self._speed_idx = self.SPEED_KEYS[key]
        elif key in (81, 2):                            # seta ←
            with self._lock:
                self._seek_delta -= self.SEEK_STEP
        elif key in (83, 3):                            # seta →
            with self._lock:
                self._seek_delta += self.SEEK_STEP
        elif key == ord('='):                           # Contraste +  (tecla =)
            with self._lock:
                self._contrast = min(200, self._contrast + self.IMG_STEP)
        elif key == ord('-'):                           # Contraste -  (tecla -)
            with self._lock:
                self._contrast = max(0, self._contrast - self.IMG_STEP)
        elif key == ord(']'):                           # Brilho +     (tecla ])
            with self._lock:
                self._brightness = min(200, self._brightness + self.IMG_STEP)
        elif key == ord('['):                           # Brilho -     (tecla [)
            with self._lock:
                self._brightness = max(0, self._brightness - self.IMG_STEP)
        elif key == ord('r'):                           # Reset imagem
            with self._lock:
                self._contrast   = 100
                self._brightness = 100
        return True

    # ── Estado para o loop principal ──────────────────────────────────
    def state(self):
        with self._lock:
            paused    = self._paused
            speed     = self.SPEEDS[self._speed_idx]
            seek      = self._seek_delta
            self._seek_delta = 0
        delay = int(33 / speed) if speed > 0 else 33
        return paused, delay, seek

    def destroy(self):
        try:
            cv2.destroyWindow(self.win)
        except Exception:
            pass


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
        "actdepth":  ("policies.act_depth.modeling_act",  "ACTPolicy"),
        "pi05depth": ("policies.pi0_depth.modeling_pi05", "PI05DEPTHPolicy"),
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
    Funciona para ACT (MEAN_STD) e PI05 (QUANTILES).

    ACT:
      preprocessor → rename, to_batch, device, normalize(images+state com mean/std)
      postprocessor → unnormalize(action), cpu

    PI05:
      preprocessor → rename, to_batch, normalize(state com quantiles), discretize, tokenize, device
      postprocessor → unnormalize(action com quantiles), cpu
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
def make_raw_obs(
    obs: dict,
    joint_names: list,
    has_depth: bool = False,
    has_pressure: bool = False,
    task: str | None = None,
) -> dict:
    """
    Monta o dict de observação SEM batch dim e SEM normalização.
    O preprocessor cuida de: batch dim, normalização, tokenização, device.

      ACT:  task=None  (não usa linguagem)
      PI05: task="pick up the cup"
    """
    raw = {}

    # Estado das juntas — radianos brutos
    state_vector = [obs.get(name, 0.0) for name in joint_names]
    raw["observation.state"] = torch.tensor(state_vector, dtype=torch.float32)

    # RGB [C, H, W] em [0, 1]
    rgb = obs.get("head_camera")
    if rgb is not None:
        raw["observation.images.head_camera"] = (
            torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        )

    # Depth [C, H, W] em [0, 1]
    if has_depth:
        depth = obs.get("head_camera_depth")
        if depth is not None:
            if len(depth.shape) == 2:
                depth = np.stack([depth] * 3, axis=-1)
            elif depth.shape[2] == 1:
                depth = np.repeat(depth, 3, axis=-1)
            raw["observation.images.head_camera_depth"] = (
                torch.from_numpy(depth).permute(2, 0, 1).float().div(255.0)
            )

    # Pressão [33]
    if has_pressure:
        for side in ["left", "right"]:
            val = obs.get(f"{side}_hand_pressure")
            if val is not None:
                raw[f"observation.{side}_hand_pressure"] = torch.from_numpy(
                    np.array(val, dtype=np.float32)
                )

    # Task string — só PI05; preprocessor tokeniza internamente
    if task is not None:
        raw["task"] = task

    return raw


# ─────────────────────────────────────────────────────────────────────
# 5. SETUP DE CÂMERAS / LEITURA DE FRAME
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
# 6. THREAD DE INFERÊNCIA
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
    task_str: str | None,
    actions_per_chunk: int,
    debug: bool,
):
    """
    Roda em thread separada.
    Consome obs_queue → preprocessor → inferência → postprocessor → action_queue.
    """
    print("🧠 [inference_worker] Iniciada.")

    while not stop_event.is_set():
        try:
            obs = obs_queue.get(timeout=0.5)
        except Empty:
            continue

        t0 = time.perf_counter()

        try:
            # 1. Monta obs bruta (sem batch dim, sem normalização)
            raw_obs = make_raw_obs(
                obs, joint_names,
                has_depth=has_depth,
                has_pressure=has_pressure,
                task=task_str,
            )

            # 2. Preprocessor: normaliza + batch dim + device
            #    ACT:  mean/std em imagens (ImageNet) e estado (dataset stats)
            #    PI05: quantiles no estado, discretiza, tokeniza
            batch = preprocessor(raw_obs)

            # Remove "action" — o preprocessor inclui essa chave para treino,
            # em inferência fica None e faz o VAE encoder crashar.
            batch.pop("action", None)

            # Filtra não-tensors — o preprocessor pode deixar scalars de metadata
            # que causam AttributeError em next(iter(batch.values())).shape[0]
            batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Workaround: garante batch dim na pressão
            if has_pressure:
                for side in ["left", "right"]:
                    k = f"observation.{side}_hand_pressure"
                    if k in batch and batch[k].dim() == 1:
                        batch[k] = batch[k].unsqueeze(0)

            # 3. Inferência + postprocessor
            with torch.inference_mode():
                if hasattr(policy, "predict_action_chunk"):
                    # Gera chunk completo de uma vez (mais eficiente)
                    raw_chunk = policy.predict_action_chunk(batch)  # (1, T, D)
                    raw_chunk = raw_chunk[0, :actions_per_chunk, :]  # (T, D)

                    # Postprocessor: desnormaliza cada ação do chunk
                    #   ACT:  action * std + mean
                    #   PI05: (action + 1) / 2 * (q99 - q01) + q01
                    chunk_np = []
                    for i in range(raw_chunk.shape[0]):
                        a = postprocessor(raw_chunk[i].unsqueeze(0))
                        if isinstance(a, dict):
                            a = a["action"]
                        chunk_np.append(a.squeeze(0).cpu().numpy())
                else:
                    # Fallback: select_action (gerencia fila interna)
                    action = policy.select_action(batch)
                    action = postprocessor(action)
                    if isinstance(action, dict):
                        action = action["action"]
                    chunk_np = [action.squeeze(0).cpu().numpy()]

            t_inf_ms = (time.perf_counter() - t0) * 1000

            if debug:
                print(f"\n🧠 [inference] {t_inf_ms:.1f}ms | chunk={len(chunk_np)} ações")

            # 4. Envia chunk para o loop de controle
            #    Se a fila estiver cheia, descarta o chunk antigo e insere o novo
            try:
                action_queue.put_nowait(chunk_np)
            except Full:
                try:
                    action_queue.get_nowait()
                except Empty:
                    pass
                action_queue.put_nowait(chunk_np)

        except Exception as e:
            import traceback
            print(f"\n❌ [inference_worker] Erro: {e}")
            traceback.print_exc()

    print("🧠 [inference_worker] Encerrada.")


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
    show_video_control = False
    uncertainty_threshold = 0.0
    remote_sim_ip = None
    actions_per_chunk = 60
    lead_actions = 50

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
        elif arg == "--v-control":
            show_video_control = True
            show_video = True   # --v-control implica exibição
        elif arg.startswith("--remote-sim="):
            remote_sim_ip = arg.split("=", 1)[1]

    lead_actions = min(lead_actions, actions_per_chunk)

    if checkpoint_dir is None:
        print("❌ ERRO: --checkpoint obrigatório.")
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

    # ── Preprocessor e Postprocessor (ACT e PI05) ─────────────────────
    preprocessor, postprocessor = load_pre_post_processors(checkpoint_dir, policy)

    # Task string: PI05 usa linguagem, ACT não
    task_str = "pick up the cup" if policy_type == "pi05depth" else None

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

    # ── Filas entre threads ───────────────────────────────────────────
    obs_queue    = Queue(maxsize=1)   # sempre a obs mais recente
    action_queue = Queue(maxsize=2)   # no máximo 2 chunks em fila

    stop_event = threading.Event()

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
            task_str=task_str,
            actions_per_chunk=actions_per_chunk,
            debug=debug_mode,
        ),
        daemon=True,
        name="inference_worker",
    )
    inf_thread.start()

    print(f"\n🚀 INFERÊNCIA ASYNC ATIVA [{policy_type.upper()}]")
    print(f"   chunk={actions_per_chunk} ações | pede nova inferência quando buf <= {lead_actions}")
    if show_video_control:
        print("   🎛️  Janela de câmera com controles de player ativa.")
        print("      [SPACE] Pause/Resume | [1-5] Velocidade | [←][→] Seek | [Q] Fechar")
        print("      Trackbars: Contraste e Brilho para simular degradação de vídeo.")
    elif show_video:
        print("   📺 Janela de câmera ativa.")
    print("   Ctrl+C para parar.\n")

    # Instancia o player com controles (se pedido)
    vc = None
    if show_video_control:
        vc = VideoControlWindow("Visao da IA — Controles")
        vc.create()

    current_chunk: list = []
    vc_last_rgb = None   # último frame RGB válido para o modo pause

    _diag_loops = 0
    _diag_elapsed_sum = 0.0

    # ─────────────────────────────────────────────────────────────────
    # LOOP PRINCIPAL
    # ─────────────────────────────────────────────────────────────────
    try:
        while True:
            start_t = time.perf_counter()

            # ── Lê estado do player (antes de qualquer get_observation) ──
            # Precisa ser no topo para que o pause bloqueie a câmera fake
            # antes de avançar o frame, não depois.
            if vc is not None:
                paused, delay_ms, seek_delta = vc.state()

                # Seek: só faz sentido com vídeo fake
                if seek_delta != 0 and fake_cap is not None:
                    cur_pos = int(fake_cap.get(cv2.CAP_PROP_POS_FRAMES))
                    new_pos = max(0, cur_pos + seek_delta)
                    fake_cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)

                # Pausado: exibe último frame congelado e volta ao topo
                # SEM chamar get_observation nem get_camera_frames,
                # então fake_cap não avança e o robô não recebe ações.
                if paused:
                    if vc_last_rgb is not None:
                        alive = vc.show(vc.process(vc_last_rgb))
                        if not alive:
                            raise KeyboardInterrupt
                    else:
                        cv2.waitKey(30)
                    continue
            else:
                paused    = False
                delay_ms  = 0
                seek_delta = 0

            # 1. Lê observação
            obs_valid = True
            try:
                obs = robot.get_observation()
            except TimeoutError as e:
                print(f"\n⚠️  Timeout de câmera: {e}. Mantendo ação do buffer.")
                obs_valid = False
                obs = None
            if obs is not None and not obs:
                obs_valid = False

            # 2. Câmeras externas
            if obs_valid and obs is not None:
                obs, fake_img_rgb = get_camera_frames(obs, stream_client, fake_cap, fake_img_rgb)

            # 3. Visualização + aplicação de contraste/brilho à observação
            if obs_valid and obs is not None and show_video:
                rgb = obs.get("head_camera")
                if rgb is not None:
                    if vc is not None:
                        # Salva frame original para exibição pausada
                        vc_last_rgb = rgb

                        # Aplica contraste/brilho ao frame que VAI para a
                        # obs_queue — assim a inferência vê a imagem degradada
                        processed = vc.process(rgb)
                        obs["head_camera"] = processed   # <── afeta inferência

                        # Exibe o frame processado
                        alive = vc.show(processed)
                        if not alive:
                            raise KeyboardInterrupt

                        # Velocidade: sleep extra (waitKey(1) já foi feito em show)
                        if delay_ms > 1:
                            time.sleep(max(0, delay_ms - 1) / 1000.0)
                    else:
                        # ── Modo --v simples ──────────────────────────────────
                        cv2.imshow("Visao da IA", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

            # 4. Alimenta a fila de inferência quando o buffer estiver baixo
            if obs_valid and len(current_chunk) <= lead_actions:
                try:
                    obs_queue.put_nowait(obs)
                except Full:
                    try:
                        obs_queue.get_nowait()
                    except Empty:
                        pass
                    obs_queue.put_nowait(obs)

            # 5. Pega chunk novo — SUBSTITUI o buffer atual
            #    (não concatena: ações velhas na frente causam movimentos de recuo)
            if not action_queue.empty():
                try:
                    new_chunk = action_queue.get_nowait()
                    current_chunk = new_chunk  # substitui, não concatena
                except Empty:
                    pass

            # 6. Executa próxima ação
            if current_chunk:
                action_numpy = current_chunk.pop(0)

                if debug_mode:
                    arm = " | ".join([f"{v:.3f}" for v in action_numpy[:7]])
                    print(f"\r🤖 [{policy_type}] E: [{arm}] buf={len(current_chunk)}", end="", flush=True)

                action_dict = {name: float(action_numpy[i]) for i, name in enumerate(joint_names)}
                robot.send_action(action_dict)
            else:
                if debug_mode:
                    print("\r⏳ Aguardando 1º chunk de inferência...", end="", flush=True)

            # 7. Diagnóstico periódico do dt do loop
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
        if vc is not None:
            vc.destroy()
        if show_video:
            cv2.destroyAllWindows()
        print("✅ Encerrado com segurança.")


if __name__ == "__main__":
    main()