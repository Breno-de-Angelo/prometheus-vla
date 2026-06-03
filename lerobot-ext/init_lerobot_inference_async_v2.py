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
  --fake-video=<PATH>    Injeta imagem ou vídeo na câmera RGB
  --fake-depth=<PATH>    Injeta vídeo ou imagem de depth sincronizado com --fake-video.
                         O frame de depth avança em lock-step com o RGB:
                         mesmo índice, mesmo loop, mesma lógica de seek/pause.
                         Útil para testar modelos ACT-D (use_depth_3d=True) sem
                         câmera real. O canal depth deve ser gravado com o mesmo
                         hack ZMQ (0–255 → 0–1 → metros reais em depth_to_pointcloud).
  --uncertainty=<FLOAT>  Ativa o uncertainty gate (ex: 0.1)
  --chunk=<INT>          Ações por chunk (padrão: 60). Deve cobrir 1 inferência inteira.
  --lead=<INT>           Pede nova inferência quando restam N ações no buffer (padrão: 50).
                         Deve ser ~= chunk para evitar gap. Máximo = chunk.
  --v                    Abre janela de visualização da câmera
  --v-control            Abre janela de visualização com controles de player:
                         pause/resume, velocidade (0.25x–4x), seek, e painéis
                         laterais de contraste e brilho para simular
                         degradação de vídeo.
  --v-attn               Abre janela de attention map ao lado da câmera.
                         Mostra heatmap de onde o decoder presta atenção
                         nos tokens de imagem RGB a cada inferência.
                         Compatível com --v e --v-control.
  --v-depth              Abre terceira janela com a Point Cloud 3D gerada
                         pelo mesmo depth_to_pointcloud() que alimenta a
                         PointNet — vista Top-Down (XZ) + lateral (YZ)
                         lado a lado. Requer use_depth_3d=True ou --fake-depth.
                         [Q/ESC] na janela para fechar.
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

  # ACT-D com vídeo fake RGB + depth sincronizado do dataset:
  python init_lerobot_inference_async.py \
      --checkpoint=train_output/actdepth/best_val_checkpoint/pretrained_model \
      --fake-video=dataset/episode_rgb.mp4 \
      --fake-depth=dataset/episode_depth.mp4 \
      --v --v-attn --chunk=10 --lead=8 --debug
"""

import os
import sys
import time
import threading
import multiprocessing as mp
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


# ─────────────────────────────────────────────────────────────────────
# 1b. JANELA DE ATTENTION MAP (--v-attn)
# ─────────────────────────────────────────────────────────────────────

class AttnMapWindow:
    """
    Janela OpenCV que exibe o attention map do decoder ACT-D sobreposto
    na imagem RGB ao vivo.

    Como funciona:
      - Após cada inferência, `policy.last_attn_weights` contém
        [B, chunk_size, num_encoder_tokens].
      - Os primeiros tokens correspondem aos patches de imagem (h*w por câmera).
      - Fazemos a média dos pesos sobre o chunk, redimensionamos para a
        resolução da imagem e sobrepõemos como heatmap JET.

    Uso:
      aw = AttnMapWindow("Attention Map — ACT-D")
      aw.create()
      aw.update(policy, rgb_frame)   # chame após cada inferência
      alive = aw.show()              # False → fechar
    """

    # Tokens de imagem no encoder: h*w do backbone (feature map final do ResNet18)
    # Para entrada 480×640 com ResNet18: feature map = 15×20 = 300 patches por câmera
    # Calculamos dinamicamente no primeiro update.
    _img_token_h: int | None = None
    _img_token_w: int | None = None

    def __init__(self, window_name: str = "Attention Map — ACT-D"):
        self.win = window_name
        self._last_display: np.ndarray | None = None
        self._lock = threading.Lock()

    def create(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 640, 480)

    def update(self, policy, rgb_frame: np.ndarray | None):
        """
        Processa `policy.last_attn_weights` e prepara o frame para exibição.
        Pode ser chamado da thread de inferência (tem lock interno).

        Args:
            policy: ACTPolicy com `last_attn_weights` populado.
            rgb_frame: imagem RGB [H, W, 3] uint8 — pode ser None se não houver câmera.
        """
        attn = getattr(policy, "last_attn_weights", None)
        if attn is None or rgb_frame is None:
            return

        try:
            # attn: [B, chunk_size, num_encoder_tokens]  (pode ser Tensor ou ndarray)
            if isinstance(attn, torch.Tensor):
                attn_np = attn.detach().cpu().float().numpy()
            else:
                attn_np = np.array(attn, dtype=np.float32)

            # Média sobre batch e chunk → [num_encoder_tokens]
            weights = attn_np[0].mean(axis=0)  # [num_encoder_tokens]

            # ── Descobre a geometria do feature map de imagem ────────────
            # Tokens de imagem ficam no início do encoder.
            # Heurística: raiz quadrada do número de tokens por câmera.
            # Para ResNet18 com entrada 480×640: backbone layer4 → 15×20 = 300.
            # Para entradas diferentes, tentamos o split mais quadrado possível.
            n_img_tokens = weights.shape[0]
            # Pega só tokens de imagem (descarta state/latent/depth extras no final)
            # Estimativa conservadora: descartamos os últimos 3 tokens
            # (state, latent, latent_pos). Ajuste se tiver mais modalidades.
            n_extra = 3  # state token + 2 latent pos tokens
            n_img = max(n_img_tokens - n_extra, 1)

            # Tenta encontrar dimensões h×w para o feature map
            if self._img_token_h is None or self._img_token_w is None:
                # Tenta resolução conhecida primeiro, senão usa sqrt
                if n_img == 300:        # ResNet18, 480×640 → 15×20
                    self._img_token_h, self._img_token_w = 15, 20
                elif n_img == 225:      # ResNet18, 360×480 → 15×15 (approx)
                    self._img_token_h, self._img_token_w = 15, 15
                else:
                    side = int(n_img ** 0.5)
                    self._img_token_h = side
                    self._img_token_w = side

            h_feat, w_feat = self._img_token_h, self._img_token_w
            n_use = h_feat * w_feat

            img_weights = weights[:n_use]  # [h_feat * w_feat]

            # Normaliza para [0, 1]
            w_min, w_max = img_weights.min(), img_weights.max()
            if w_max - w_min > 1e-6:
                img_weights = (img_weights - w_min) / (w_max - w_min)
            else:
                img_weights = np.zeros_like(img_weights)

            # Reshape → [h_feat, w_feat]
            attn_map = img_weights.reshape(h_feat, w_feat)

            # Upscale para resolução da imagem original
            H, W = rgb_frame.shape[:2]
            attn_resized = cv2.resize(attn_map, (W, H), interpolation=cv2.INTER_LINEAR)

            # Converte para heatmap colorido (JET)
            attn_uint8 = (attn_resized * 255).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

            # Sobrepõe na imagem RGB original
            alpha = 0.55
            blended = (alpha * heatmap_rgb + (1 - alpha) * rgb_frame).astype(np.uint8)

            # Legenda
            cv2.putText(
                blended, "Decoder Cross-Attention (media chunk)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                blended, "QUENTE = maior atencao",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
            )

            with self._lock:
                self._last_display = blended

        except Exception as e:
            # Nunca deixa a thread de inferência crashar por causa do visual
            print(f"[AttnMapWindow] Erro ao processar attention: {e}")

    def show(self) -> bool:
        """Exibe o último frame processado. Retorna False se janela fechada."""
        with self._lock:
            frame = self._last_display

        if frame is None:
            # Ainda sem dados — exibe tela preta com mensagem
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, "Aguardando 1a inferencia...",
                (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1,
            )
            cv2.imshow(self.win, placeholder)
        else:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.win, bgr)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            return False
        return True

    def destroy(self):
        try:
            cv2.destroyWindow(self.win)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
# 1c. JANELA DE POINT CLOUD 3D — Open3D em processo separado (--v-depth)
# ─────────────────────────────────────────────────────────────────────

def _pointcloud_process(queue: mp.Queue, intrinsics: dict, stop_evt: mp.Event):
    """
    Roda em processo separado (não thread) porque o Open3D precisa da
    thread principal do seu próprio processo para desenhar.

    Recebe dicts {"depth": uint8[H,W,3], "rgb": uint8[H,W,3]} via queue.

    Cor de cada ponto: back-projection do RGB.
    Para cada ponto XYZ gerado pelo depth_to_pointcloud(), reprojetamos
    de volta para o plano da imagem (u, v) e lemos o pixel RGB ali.
    Isso é exatamente como um sensor RGBD funde as duas câmeras — você
    vê a nuvem colorida com a textura real da cena.

    Se o RGB não estiver disponível, cai no colormap por distância Z.
    """
    import torch
    import numpy as np
    import open3d as o3d
    from policies.act_depth.depth_encoder import depth_to_pointcloud

    fx = intrinsics["fx"]; fy = intrinsics["fy"]
    cx = intrinsics["cx"]; cy = intrinsics["cy"]

    # ── Helpers locais ────────────────────────────────────────────────
    def criar_grade(tamanho=2.0, espacamento=0.1, altura=-0.2):
        pontos, linhas, idx = [], [], 0
        for z in np.arange(-tamanho / 2, tamanho / 2 + espacamento, espacamento):
            pontos += [[-tamanho / 2, altura, z], [tamanho / 2, altura, z]]
            linhas.append([idx, idx + 1]); idx += 2
        for x in np.arange(-tamanho / 2, tamanho / 2 + espacamento, espacamento):
            pontos += [[x, altura, -tamanho / 2], [x, altura, tamanho / 2]]
            linhas.append([idx, idx + 1]); idx += 2
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pontos)
        ls.lines  = o3d.utility.Vector2iVector(linhas)
        ls.colors = o3d.utility.Vector3dVector([[0.25, 0.25, 0.25]] * len(linhas))
        return ls

    def backproject_rgb(pontos_xyz_original: np.ndarray,
                        rgb_hwc: np.ndarray) -> np.ndarray:
        """
        Para cada ponto [X, Y, Z] (antes da inversão de eixos Open3D),
        recalcula o pixel (u, v) na imagem RGB usando a projeção pinhole
        inversa — e lê a cor RGB ali.

        pontos_xyz_original: [N, 3] float32, eixos câmera (Y↓, Z→frente)
        rgb_hwc:             [H, W, 3] uint8 RGB

        Retorna cores [N, 3] float64 em [0, 1] para o Open3D.
        """
        H, W = rgb_hwc.shape[:2]
        X, Y, Z = pontos_xyz_original[:, 0], pontos_xyz_original[:, 1], pontos_xyz_original[:, 2]

        # Projeção pinhole: (X, Y, Z) → (u, v) em pixels
        # u = fx * X/Z + cx  |  v = fy * Y/Z + cy
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(Z > 0, fx * X / Z + cx, -1).astype(np.int32)
            v = np.where(Z > 0, fy * Y / Z + cy, -1).astype(np.int32)

        # Clipa para dentro da imagem
        u = np.clip(u, 0, W - 1)
        v = np.clip(v, 0, H - 1)

        # Lê pixel RGB e normaliza para [0, 1]
        cores = rgb_hwc[v, u].astype(np.float64) / 255.0   # [N, 3]
        return cores

    # ── Janela Open3D ─────────────────────────────────────────────────
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Point Cloud RGB — ACT-D  [Q para fechar]",
        width=1280, height=720,
    )

    pcd   = o3d.geometry.PointCloud()
    eixos = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.15, origin=[0, 0, 0])
    grade = criar_grade()

    vis.add_geometry(pcd)
    vis.add_geometry(eixos)
    vis.add_geometry(grade)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.08])
    opt.point_size = 3.0

    primeiro_frame = True

    while not stop_evt.is_set():
        # Drena a fila pegando sempre o par mais recente
        payload = None
        while not queue.empty():
            try:
                payload = queue.get_nowait()
            except Exception:
                break

        if payload is not None:
            try:
                depth_hwc = payload["depth"]   # uint8 [H, W, 3]
                rgb_hwc   = payload.get("rgb")  # uint8 [H, W, 3] RGB ou None

                # ── Depth → Point Cloud (igual ao modelo) ─────────────
                canal_r = depth_hwc[:, :, 0]
                depth_tensor = (
                    torch.from_numpy(canal_r).float()
                    .div(255.0)
                    .unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
                )

                with torch.no_grad():
                    pc = depth_to_pointcloud(
                        depth_tensor, intrinsics, num_points=5000
                    )  # [1, 3, 5000]

                # [5000, 3] — X, Y, Z em metros, eixos câmera (Y↓, Z→frente)
                pontos_cam = pc[0].T.numpy()

                # Filtra padding de zeros
                validos    = pontos_cam[:, 2] > 0.05
                pontos_cam = pontos_cam[validos]

                if pontos_cam.shape[0] > 0:
                    # ── Cor: back-projection RGB ou fallback por Z ─────
                    if rgb_hwc is not None:
                        # Usa os eixos ORIGINAIS da câmera (antes de inverter)
                        # para reprojetar corretamente no plano da imagem.
                        cores = backproject_rgb(pontos_cam, rgb_hwc)
                    else:
                        # Fallback: colormap por distância Z
                        z_n = (pontos_cam[:, 2] - pontos_cam[:, 2].min()) / \
                              (pontos_cam[:, 2].max() - pontos_cam[:, 2].min() + 1e-5)
                        cores = np.zeros((pontos_cam.shape[0], 3))
                        cores[:, 0] = 1.0 - z_n
                        cores[:, 1] = z_n * 0.5
                        cores[:, 2] = z_n

                    # ── Inverte eixos para convenção Open3D ────────────
                    pontos_o3d = pontos_cam.copy()
                    pontos_o3d[:, 1] *= -1
                    pontos_o3d[:, 2] *= -1

                    pcd.points = o3d.utility.Vector3dVector(pontos_o3d)
                    pcd.colors = o3d.utility.Vector3dVector(cores)
                    vis.update_geometry(pcd)

                    if primeiro_frame:
                        vis.reset_view_point(True)
                        ctr = vis.get_view_control()
                        ctr.set_front([0.0, -0.4, 1.0])
                        ctr.set_up([0.0, 1.0, 0.0])
                        ctr.set_zoom(0.5)
                        primeiro_frame = False

            except Exception as e:
                print(f"[PointCloudProc] Erro: {e}")

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()


class PointCloudWindow:
    """
    Wrapper que lança _pointcloud_process num processo separado e
    expõe a mesma API das outras janelas (update / show / destroy).

    O Open3D PRECISA da thread principal do seu processo — por isso
    usamos multiprocessing em vez de threading.
    """

    def __init__(self, intrinsics: dict | None = None):
        self._intrinsics = intrinsics or {
            "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0
        }
        self._queue: mp.Queue | None = None
        self._stop:  mp.Event | None = None
        self._proc:  mp.Process | None = None

    def create(self):
        self._queue = mp.Queue(maxsize=2)
        self._stop  = mp.Event()
        self._proc  = mp.Process(
            target=_pointcloud_process,
            args=(self._queue, self._intrinsics, self._stop),
            daemon=True,
        )
        self._proc.start()

    def update(self, depth_frame: np.ndarray | None, rgb_frame: np.ndarray | None = None):
        """
        Envia o par (depth, rgb) para o processo Open3D.

        depth_frame: uint8 [H, W, 3] — canal R = depth quantizado.
        rgb_frame:   uint8 [H, W, 3] RGB — mesma resolução do depth.
                     Se None, o processo usa colormap por distância como fallback.
        """
        if depth_frame is None or self._queue is None:
            return
        payload = {
            "depth": depth_frame.copy(),
            "rgb":   rgb_frame.copy() if rgb_frame is not None else None,
        }
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except Exception:
                pass
        try:
            self._queue.put_nowait(payload)
        except Exception:
            pass

    def show(self) -> bool:
        """Retorna False se o processo Open3D foi fechado."""
        if self._proc is None:
            return False
        return self._proc.is_alive()

    def destroy(self):
        if self._stop is not None:
            self._stop.set()
        if self._proc is not None and self._proc.is_alive():
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.terminate()


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
def setup_cameras(cam_robot_ip, cam_port, fake_video_path, fake_depth_path=None):
    """
    Inicializa câmeras reais (ZMQ) ou fontes fake (vídeo/imagem).

    fake_depth_path: caminho para vídeo ou imagem de depth a injetar em
                     obs["head_camera_depth"]. Avança frame-a-frame em
                     lock-step com fake_cap (mesmo índice, mesmo seek).
                     Ignorado se cam_robot_ip estiver definido (ZMQ já
                     entrega depth nativo).
    """
    from Scripts_Prometheus_int.sim.sensor_utils import SensorClient, ImageUtils  # noqa: F401

    stream_client = fake_cap = fake_img_rgb = None
    fake_depth_cap = fake_depth_img = None   # ← novas variáveis de depth

    if cam_robot_ip:
        stream_client = SensorClient()
        stream_client.start_client(server_ip=cam_robot_ip, port=int(cam_port))
        print(f"📡 Conectando ao ZMQ SensorServer em tcp://{cam_robot_ip}:{cam_port}...")
    elif fake_video_path:
        if not os.path.exists(fake_video_path):
            print(f"❌ ERRO: Arquivo fake RGB não encontrado: {fake_video_path}")
            sys.exit(1)
        if fake_video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            fake_cap = cv2.VideoCapture(fake_video_path)
            print("✅ Vídeo fake RGB carregado! (Modo Loop)")
        else:
            img_bgr = cv2.imread(fake_video_path)
            fake_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            print("✅ Imagem fake RGB carregada!")

    # ── Depth fake — só faz sentido sem ZMQ (ZMQ já entrega depth) ────
    if fake_depth_path and not cam_robot_ip:
        if not os.path.exists(fake_depth_path):
            print(f"❌ ERRO: Arquivo fake depth não encontrado: {fake_depth_path}")
            sys.exit(1)
        if fake_depth_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            fake_depth_cap = cv2.VideoCapture(fake_depth_path)
            print("✅ Vídeo fake Depth carregado! (sincronizado frame-a-frame com RGB)")
        else:
            depth_bgr = cv2.imread(fake_depth_path, cv2.IMREAD_GRAYSCALE)
            if depth_bgr is None:
                depth_bgr = cv2.imread(fake_depth_path)
            if depth_bgr is not None:
                if len(depth_bgr.shape) == 2:
                    # Grayscale → replica para 3 canais (make_raw_obs espera H×W×3)
                    fake_depth_img = np.stack([depth_bgr] * 3, axis=-1)
                else:
                    fake_depth_img = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
            print("✅ Imagem fake Depth carregada!")
    elif fake_depth_path and cam_robot_ip:
        print("⚠️  --fake-depth ignorado: ZMQ já fornece depth nativo.")

    return stream_client, fake_cap, fake_img_rgb, fake_depth_cap, fake_depth_img


def get_camera_frames(obs, stream_client, fake_cap, fake_img_rgb,
                      fake_depth_cap=None, fake_depth_img=None):
    """
    Lê o próximo frame de câmera e o injeta em `obs`.

    Depth fake (fake_depth_cap / fake_depth_img):
      - Avança em lock-step com o RGB fake: mesmo read(), mesmo loop,
        mesmo seek pelo chamador.
      - O seek de --v-control já reposiciona fake_cap ANTES desta função,
        então basta espelhar CAP_PROP_POS_FRAMES do fake_cap no fake_depth_cap.
    """
    if stream_client is not None:
        from Scripts_Prometheus_int.sim.sensor_utils import ImageUtils
        msg = stream_client.receive_message()
        if msg and "images" in msg:
            obs["head_camera"]       = ImageUtils.decode_image(msg["images"]["head_camera"])
            obs["head_camera_depth"] = ImageUtils.decode_image(msg["images"]["head_camera_depth"])

    elif fake_cap is not None:
        # ── RGB ──────────────────────────────────────────────────────
        ret, frame = fake_cap.read()
        if not ret:
            fake_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = fake_cap.read()
        if ret:
            fake_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            obs["head_camera"] = fake_img_rgb

        # ── Depth fake em lock-step com RGB ──────────────────────────
        if fake_depth_cap is not None:
            # Espelha o índice de frame do RGB para garantir sincronia
            # mesmo depois de loops ou seeks externos.
            rgb_pos = int(fake_cap.get(cv2.CAP_PROP_POS_FRAMES))
            fake_depth_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, rgb_pos - 1))
            ret_d, depth_frame = fake_depth_cap.read()
            if not ret_d:
                fake_depth_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_d, depth_frame = fake_depth_cap.read()
            if ret_d:
                # Converte para RGB de 3 canais (make_raw_obs → div(255) → PointNet)
                if len(depth_frame.shape) == 2:
                    depth_rgb = np.stack([depth_frame] * 3, axis=-1)
                else:
                    depth_rgb = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2RGB)
                obs["head_camera_depth"] = depth_rgb

        elif fake_depth_img is not None:
            obs["head_camera_depth"] = fake_depth_img

    elif fake_img_rgb is not None:
        obs["head_camera"] = fake_img_rgb
        if fake_depth_img is not None:
            obs["head_camera_depth"] = fake_depth_img

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
    attn_window=None,        # AttnMapWindow ou None
):
    """
    Roda em thread separada.
    Consome obs_queue → preprocessor → inferência → postprocessor → action_queue.
    """
    print("🧠 [inference_worker] Iniciada.")

    # Frame RGB mais recente — usado para gerar o attention overlay
    _last_rgb: np.ndarray | None = None

    while not stop_event.is_set():
        try:
            obs = obs_queue.get(timeout=0.5)
        except Empty:
            continue

        # Guarda o frame RGB para o overlay de atenção
        rgb = obs.get("head_camera") if obs else None
        if rgb is not None:
            _last_rgb = rgb

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

            # 3b. Atualiza o attention map (não bloqueia — tem lock interno)
            if attn_window is not None and _last_rgb is not None:
                attn_window.update(policy, _last_rgb)

            t_inf_ms = (time.perf_counter() - t0) * 1000

            if debug:
                has_attn = getattr(policy, "last_attn_weights", None) is not None
                print(f"\n🧠 [inference] {t_inf_ms:.1f}ms | chunk={len(chunk_np)} ações | attn={'✓' if has_attn else '✗'}")

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
    fake_depth_path = None          # ← novo: --fake-depth
    cam_robot_ip = None
    cam_port = "5555"
    debug_mode = False
    show_video = False
    show_video_control = False
    show_attn = False
    show_depth = False
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
        elif arg.startswith("--fake-depth="):             # ← novo
            fake_depth_path = arg.split("=", 1)[1]
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
        elif arg == "--v-attn":
            show_attn = True
            show_video = True   # --v-attn também precisa do frame RGB
        elif arg == "--v-depth":
            show_depth = True
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
    stream_client, fake_cap, fake_img_rgb, fake_depth_cap, fake_depth_img = setup_cameras(
        cam_robot_ip, cam_port, fake_video_path, fake_depth_path
    )

    # ── Robô ─────────────────────────────────────────────────────────
    from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
    print(f"⏳ Conectando ao Unitree G1 (Simulação: {is_sim})...")
    g1_config = UnitreeG1Dex3Config(
        #robot_ip="10.9.8.73",
        robot_ip="192.168.123.164",
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

    # ── Janelas OpenCV — criadas ANTES do inf_thread (aw precisa existir) ─
    vc = None
    if show_video_control:
        vc = VideoControlWindow("Visao da IA — Controles")
        vc.create()

    aw = None
    if show_attn:
        aw = AttnMapWindow("Attention Map — ACT-D")
        aw.create()

    dw = None
    if show_depth:
        # Pega intrinsics do config da política (setadas em configuration_act.py)
        # Fallback para os valores do pointcloud.py original que funcionava
        pc_intrinsics = getattr(policy, "camera_intrinsics", None) or {
            "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0
        }
        dw = PointCloudWindow(intrinsics=pc_intrinsics)
        dw.create()

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
            attn_window=aw,          # None se --v-attn não foi passado
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
    if show_attn:
        print("   🔥 Janela de Attention Map ativa (--v-attn).")
        print("      Atualiza a cada inferência. [Q/ESC] na janela de attn para fechar.")
    if show_depth:
        print("   🟣 Janela de Point Cloud ativa (--v-depth). Vista Top-Down + Lateral.")
        print("      Replica depth_to_pointcloud() — você vê o que a PointNet vê.")
        print("      [Q/ESC] na janela de PC para fechar.")
    if fake_depth_cap is not None:
        print("   📐 Depth fake (vídeo) ativo — sincronizado frame-a-frame com RGB.")
    elif fake_depth_img is not None:
        print("   📐 Depth fake (imagem estática) ativa.")
    print("   Ctrl+C para parar.\n")

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
                    # Mantém depth em lock-step com RGB no seek
                    if fake_depth_cap is not None:
                        fake_depth_cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)

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
                obs, fake_img_rgb = get_camera_frames(
                    obs, stream_client, fake_cap, fake_img_rgb,
                    fake_depth_cap=fake_depth_cap,
                    fake_depth_img=fake_depth_img,
                )

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

            # 3b. Janela de Point Cloud Open3D (--v-depth)
            if dw is not None and obs_valid and obs is not None:
                depth_raw = obs.get("head_camera_depth")
                rgb_raw = obs.get("head_camera")  # Pega o frame RGB colorido
                
                # Envia OS DOIS para a janela do Open3D
                dw.update(depth_raw, rgb_raw)  
                
                alive = dw.show()
                if not alive:
                    raise KeyboardInterrupt

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

            # 8. Exibe janela de attention map (atualiza a cada inferência,
            #    o loop principal só chama show() para processar eventos OpenCV)
            if aw is not None:
                alive = aw.show()
                if not alive:
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n🛑 Parando inferência...")

    finally:
        stop_event.set()
        inf_thread.join(timeout=3.0)

        if fake_cap is not None:
            fake_cap.release()
        if fake_depth_cap is not None:       # ← novo
            fake_depth_cap.release()
        if stream_client is not None:
            stream_client.stop_client()
        robot.disconnect()
        if vc is not None:
            vc.destroy()
        if aw is not None:
            aw.destroy()
        if dw is not None:
            dw.destroy()
        if show_video:
            cv2.destroyAllWindows()
        print("✅ Encerrado com segurança.")


if __name__ == "__main__":
    # "spawn" evita problemas de CUDA + fork no Linux quando --v-depth está ativo.
    # É ignorado se o processo já foi iniciado com outro método.
    mp.set_start_method("spawn", force=False)
    main()