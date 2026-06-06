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
    # WINDOW_NORMAL sem flags extras: o Qt backend escala o frame para
    # preencher a janela inteira quando ela é redimensionada/maximizada.
    def create(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 640, 480)

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
#
# Abre DUAS janelas separadas:
#
#   "Attention Map — RGB"
#     Heatmap INFERNO sobreposto na câmera, normalizado SOMENTE dentro
#     dos tokens de imagem (min/max dos 300 patches RGB). Isso garante
#     que regiões quentes correspondem a patches que o decoder realmente
#     olhou — sem contaminação dos valores de state/latente, que são
#     escalares muito maiores e antes deixavam tudo "flat".
#
#   "Attention Map — Analise"
#     Painel dividido em 3 seções ocupando 100% da janela:
#
#     ① Comparação de tokens (barras horizontais — mesma escala entre si)
#        RGB médio | Depth 3D | State (1 token) | Latente z
#        → responde: "o modelo usa a câmera ou decorou o movimento?"
#        → escala local: min/max dos 4 escalares (não contamina o heatmap)
#
#     ② Ação predita por junta (chunk completo)
#        28 linhas coloridas, eixo X = tempo (steps do chunk)
#        Cada grupo de juntas em cor diferente:
#          braço esq (0-6)  | braço dir (7-13) | mão esq (14-20) | mão dir (21-27)
#        → responde: "o modelo está abrindo a mão? fazendo o gesto certo?"
#
#     ③ Histórico de atenção ao longo das últimas inferências
#        Linhas: RGB | State | Depth — na mesma escala relativa
#        → responde: "a atenção muda quando a cena muda?"
#
# Sem nenhuma modificação no modelo.
# ─────────────────────────────────────────────────────────────────────

class AttnMapWindow:
    """
    Duas janelas OpenCV para análise de atenção e ação do ACT-D.
    Sem modificação no modelo — lê apenas last_attn_weights e _last_action_chunk_np.

    Ordem dos tokens no encoder (ACT-D):
      [0 .. n_rgb-1]   → patches RGB ResNet18 (15×20 = 300 para 480×640)
      [n_rgb]          → depth token   (se use_depth_3d)
      [n_rgb + d]      → state token   (sempre — 1 único token para todas as juntas)
      [n_rgb + d + 1]  → latente z     (se use_vae)

    Grupos de juntas para o gráfico de ação (28 juntas, G1 upper body):
      0-6   braço esquerdo  (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
      7-13  braço direito
      14-20 mão esquerda    (thumb 0/1/2, middle 0/1, index 0/1)
      21-27 mão direita
    """

    _img_token_h: int | None = None
    _img_token_w: int | None = None
    _HISTORY_LEN = 120   # pontos no gráfico de histórico

    # Grupos e cores BGR para o gráfico de ação
    _JOINT_GROUPS = [
        ("Braco Esq",  range(0,  7),  (60,  200,  60)),   # verde
        ("Braco Dir",  range(7,  14), (60,  160, 240)),   # azul
        ("Mao Esq",    range(14, 21), (220, 120,  60)),   # laranja
        ("Mao Dir",    range(21, 28), (180,  60, 220)),   # roxo
    ]

    def __init__(
        self,
        window_name: str = "Attention Map",
        use_depth: bool = False,
        use_vae: bool = True,
        action_dim: int = 28,
    ):
        self.win_rgb   = window_name + " — RGB"
        self.win_panel = window_name + " — Analise"
        self.use_depth  = use_depth
        self.use_vae    = use_vae
        self.action_dim = action_dim

        self._last_rgb_display:   np.ndarray | None = None
        self._last_panel_display: np.ndarray | None = None
        self._lock = threading.Lock()

        self._state_history = []
        self._depth_history = []
        self._rgb_history   = []

    def create(self):
        # WINDOW_NORMAL: o Qt backend escala o frame para preencher a janela
        # inteira ao redimensionar — sem bordas brancas, sem distorção.
        cv2.namedWindow(self.win_rgb, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_rgb, 640, 545)

        cv2.namedWindow(self.win_panel, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_panel, 1200, 900)

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _hline(canvas, y, color=(70, 70, 70)):
        cv2.line(canvas, (0, y), (canvas.shape[1], y), color, 1)

    @staticmethod
    def _label(canvas, txt, x, y, scale=0.42, color=(180, 180, 180), bold=False):
        th = 2 if bold else 1
        cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, th, cv2.LINE_AA)

    @staticmethod
    def _draw_bar(canvas, x0, y, bar_w, fill_frac, label, raw_val, color_bgr, bar_h=16):
        """Barra horizontal: Agora o texto fica ACIMA da barra para não cortar no layout."""
        cv2.putText(canvas, f"{label}: {raw_val:.5f}",
                    (x0, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y), (x0 + bar_w, y + bar_h), (45, 45, 45), -1)
        fw = max(1, int(fill_frac * bar_w))
        cv2.rectangle(canvas, (x0, y), (x0 + fw, y + bar_h), color_bgr, -1)

    @staticmethod
    def _draw_action_group(canvas, chunk, joint_range, x0, y0, w, h, color_bgr, label):
        """Modo dinâmico: Se T > 1, divide o espaço e cria um mini-gráfico (sparkline) separado para cada junta."""
        chunk = np.array(chunk)
        T = chunk.shape[0]

        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (22, 22, 22), -1)
        cv2.putText(canvas, label, (x0 + 4, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr, 1, cv2.LINE_AA)

        sub = chunk[:, list(joint_range)]
        n_joints = sub.shape[1]

        if T == 1:
            # ── Modo estático (chunk=1): Barras horizontais instantâneas
            bar_area_y0 = y0 + 25
            avail_h = h - 30
            bar_h = max(4, avail_h // n_joints - 2)
            vmin, vmax = sub.min(), sub.max()
            rng = vmax - vmin if abs(vmax - vmin) > 1e-6 else 1.0

            for j in range(n_joints):
                alpha = 0.45 + 0.55 * (j / max(n_joints - 1, 1))
                col   = tuple(int(c * alpha) for c in color_bgr)
                by    = bar_area_y0 + j * (bar_h + 2)
                val   = float(sub[0, j])
                frac  = (val - vmin) / rng

                cv2.rectangle(canvas, (x0 + 4, by), (x0 + w - 4, by + bar_h), (40, 40, 40), -1)
                fw = max(1, int(frac * (w - 8)))
                cv2.rectangle(canvas, (x0 + 4, by), (x0 + 4 + fw, by + bar_h), col, -1)
                cv2.putText(canvas, f"{val:.3f}", (x0 + w - 42, by + bar_h - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1, cv2.LINE_AA)
            return

        # ── Modo temporal (chunk > 1): Sparklines empilhadas
        row_h = (h - 25) / n_joints
        start_y = y0 + 22

        for j in range(n_joints):
            ry = start_y + j * row_h
            # Linha divisória suave entre as juntas
            cv2.line(canvas, (x0 + 2, int(ry + row_h)), (x0 + w - 2, int(ry + row_h)), (35, 35, 35), 1)

            j_vals = sub[:, j]
            # Escala local: cada junta normaliza contra seu próprio min/max daquele chunk
            vmin, vmax = j_vals.min(), j_vals.max()
            rng = vmax - vmin if abs(vmax - vmin) > 1e-6 else 1.0

            pts = []
            for t in range(T):
                px = x0 + int(t / (T - 1) * (w - 4)) + 2
                py = ry + row_h - ((j_vals[t] - vmin) / rng * (row_h - 4)) - 2
                pts.append((px, int(py)))

            alpha = 0.4 + 0.6 * (j / max(n_joints - 1, 1))
            col = tuple(int(c * alpha) for c in color_bgr)
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], col, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_history(canvas, histories_and_colors, x0, y0, w, h):
        """Desenha históricos com escalas Y completamente independentes."""
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (22, 22, 22), -1)
        
        for history, color_bgr in histories_and_colors:
            if len(history) < 2:
                continue
            
            # NORMALIZAÇÃO INDIVIDUAL: O State e Depth agora vão oscilar e 
            # preencher toda a altura do quadro baseados nos seus próprios valores.
            vmin, vmax = min(history), max(history)
            rng = vmax - vmin if abs(vmax - vmin) > 1e-8 else 1.0
            
            pts = []
            n = len(history)
            for i, v in enumerate(history):
                px = x0 + int(i / (n - 1) * (w - 2)) + 1
                py = y0 + h - int((v - vmin) / rng * (h - 4)) - 2
                py = max(y0 + 1, min(y0 + h - 1, py))
                pts.append((px, py))
                
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], color_bgr, 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────
    # update() — thread de inferência
    # ─────────────────────────────────────────────────────────────────

    def update(self, policy, rgb_frame: np.ndarray | None):
        attn = getattr(policy, "last_attn_weights", None)
        if attn is None or rgb_frame is None:
            return

        action_chunk = getattr(policy, "_last_action_chunk_np", None)

        try:
            if isinstance(attn, torch.Tensor):
                attn_np = attn.detach().cpu().float().numpy()
            else:
                attn_np = np.array(attn, dtype=np.float32)

            # max sobre as queries do decoder → [N_tokens]
            # Usamos max (não mean) para preservar os picos de atenção.
            # A mean de 100 queries suaviza tudo — o range encolhe para ~0.001
            # e o heatmap fica flat. O max mostra qual token alguma query
            # realmente priorizou, que é o sinal visual que queremos.
            weights = attn_np[0].max(axis=0) if attn_np.ndim == 3 else attn_np.max(axis=0)
            N = weights.shape[0]

            # ── Descobre geometria ────────────────────────────────────
            n_extra = int(self.use_depth) + 1 + int(self.use_vae)
            n_img   = max(N - n_extra, 1)

            if self._img_token_h is None:
                if n_img == 300:
                    self._img_token_h, self._img_token_w = 15, 20
                elif n_img == 225:
                    self._img_token_h, self._img_token_w = 15, 15
                else:
                    side = int(n_img ** 0.5)
                    self._img_token_h = self._img_token_w = side

            h_feat, w_feat = self._img_token_h, self._img_token_w
            n_rgb_tokens   = h_feat * w_feat

            # ── Fatia tokens não-visuais ──────────────────────────────
            off = n_rgb_tokens
            attn_depth  = float(weights[off])     if self.use_depth else 0.0
            off        += int(self.use_depth)
            attn_state  = float(weights[off])
            off        += 1
            attn_latent = float(weights[off])     if self.use_vae   else 0.0
            attn_rgb_mean = float(weights[:n_rgb_tokens].mean())
            attn_rgb_max  = float(weights[:n_rgb_tokens].max())
            # Para o histórico e comparação de tokens usamos attn_rgb_max:
            # a média de 300 tokens após softmax é ~1/N e varia pouquíssimo
            # entre inferências — o máximo preserva o pico de atenção real.
            attn_rgb_scalar = attn_rgb_max

            # ── Histórico ─────────────────────────────────────────────
            self._rgb_history.append(attn_rgb_scalar)
            self._state_history.append(attn_state)
            self._depth_history.append(attn_depth)
            for lst in (self._rgb_history, self._state_history, self._depth_history):
                if len(lst) > self._HISTORY_LEN:
                    lst.pop(0)

            # ══════════════════════════════════════════════════════════
            # JANELA 1 — Heatmap RGB
            # Normaliza APENAS dentro dos tokens de imagem (não contamina
            # com escalares de state/latente que são muito maiores).
            # ══════════════════════════════════════════════════════════
            img_w = weights[:n_rgb_tokens].copy()
            i_min, i_max = img_w.min(), img_w.max()
            if i_max - i_min > 1e-8:
                img_w = (img_w - i_min) / (i_max - i_min)
            else:
                img_w = np.zeros_like(img_w)

            attn_map     = img_w.reshape(h_feat, w_feat)
            H, W         = rgb_frame.shape[:2]
            attn_resized = cv2.resize(attn_map.astype(np.float32), (W, H),
                                      interpolation=cv2.INTER_LINEAR)
            attn_uint8   = (attn_resized * 255).astype(np.uint8)
            heat_bgr     = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_INFERNO)
            heat_rgb     = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


            blended      = (0.55 * heat_rgb + 0.45 * rgb_frame).astype(np.uint8)

            self._label(blended, "Cross-Attention decoder (tokens RGB)", 8, 22,
                        0.5, (255, 255, 255), bold=True)
            self._label(blended,
                        f"RGB max: {attn_rgb_max:.5f}  mean: {attn_rgb_mean:.5f}"
                        f"  |  State: {attn_state:.5f}"
                        + (f"  |  Depth: {attn_depth:.5f}" if self.use_depth else ""),
                        8, 44, 0.38, (200, 200, 200))

            # ══════════════════════════════════════════════════════════
            # NOVO: Barra inferior com quadrados de State e Depth
            # Compartilhando a MESMA escala de cor e normalização do RGB
            # ══════════════════════════════════════════════════════════
            BAR_H = 65
            new_H = H + BAR_H
            canvas_rgb = np.zeros((new_H, W, 3), dtype=np.uint8)
            
            # Copia a imagem sobreposta para o topo do novo canvas
            canvas_rgb[:H, :W] = blended
            
            # Desenha fundo da barra inferior
            cv2.rectangle(canvas_rgb, (0, H), (W, new_H), (20, 20, 20), -1)
            cv2.line(canvas_rgb, (0, H), (W, H), (50, 50, 50), 1)

            def get_color_from_val(val):
                """Aplica a MESMA normalização (min/max) da matriz RGB e o colormap."""
                if i_max - i_min > 1e-8:
                    norm = (val - i_min) / (i_max - i_min)
                else:
                    norm = 0.0
                
                # Se o token de state/depth for muito maior que o max do RGB, ele trava em 1.0 (cor mais quente)
                norm = np.clip(norm, 0.0, 1.0)
                
                # Gera 1 pixel e passa pelo colormap OpenCV
                arr = np.array([[int(norm * 255)]], dtype=np.uint8)
                color_bgr = cv2.applyColorMap(arr, cv2.COLORMAP_INFERNO)[0, 0]
                
                # Converte BGR para RGB (pois o canvas_rgb está em formato RGB)
                return (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))

            sq_size = 28
            y_pad = H + (BAR_H - sq_size) // 2

            # Quadrado 1: State
            x1 = 20
            color_state = get_color_from_val(attn_state)
            cv2.rectangle(canvas_rgb, (x1, y_pad), (x1 + sq_size, y_pad + sq_size), color_state, -1)
            cv2.rectangle(canvas_rgb, (x1, y_pad), (x1 + sq_size, y_pad + sq_size), (150, 150, 150), 1) # Borda
            self._label(canvas_rgb, f"State Token", x1 + sq_size + 12, y_pad + 20, 0.45, (220, 220, 220))

            # Quadrado 2: Depth (só desenha se o modelo usar depth)
            if self.use_depth:
                x2 = x1 + sq_size + 150
                color_depth = get_color_from_val(attn_depth)
                cv2.rectangle(canvas_rgb, (x2, y_pad), (x2 + sq_size, y_pad + sq_size), color_depth, -1)
                cv2.rectangle(canvas_rgb, (x2, y_pad), (x2 + sq_size, y_pad + sq_size), (150, 150, 150), 1)
                self._label(canvas_rgb, f"Depth Token", x2 + sq_size + 12, y_pad + 20, 0.45, (220, 220, 220))

            # Redireciona a variável blended para o nosso novo canvas expandido
            blended = canvas_rgb

            # ══════════════════════════════════════════════════════════
            # JANELA 2 — Painel de análise
            # Aumentamos o canvas nativo para 1200x900 para caber o texto com folga.
            # O cv2.WINDOW_NORMAL fará o downscale sem cortar nada.
            # ══════════════════════════════════════════════════════════
            PW, PH = 1200, 900
            panel = np.zeros((PH, PW, 3), dtype=np.uint8)

            PAD  = 15
            INNER_W = PW - 2 * PAD
            y = 0

            # ── SEÇÃO 1: Comparação de tokens (25% da altura) ─────────
            sec1_h = PH * 25 // 100
            cv2.rectangle(panel, (0, 0), (PW, sec1_h), (18, 18, 18), -1)
            self._label(panel, "Atencao por tipo de token (escala relativa ao proprio pico recente)",
                        PAD, 25, 0.48, (220, 220, 220), bold=True)

            bar_w   = INNER_W // 2 - 40
            bar_h   = 16
            bar_gap = 30  # Mais espaço vertical para o texto que agora fica em cima
            by      = 65

            # Atrelamos cada métrica ao seu próprio histórico para calcular quão cheia a barra fica
            token_specs = [
                ("RGB (max patch)",       attn_rgb_scalar, self._rgb_history,   (60,  200,  60)),
                ("State (1 tok)",         attn_state,      self._state_history, (220, 180,  60)),
                ("Latente z",             attn_latent,     [attn_latent],       (60,  160, 220)),
            ]
            if self.use_depth:
                token_specs.insert(2, ("Depth 3D", attn_depth, self._depth_history, (180, 80, 220)))

            col_xs = [PAD, PAD + bar_w + 60]
            for idx, (lbl, raw, hist, col) in enumerate(token_specs):
                cx = col_xs[idx % 2]
                cy = by + (idx // 2) * (bar_h + bar_gap)
                
                # A barra enche em relação ao maior valor recente DAQUELA métrica específica
                h_max = max(hist) if hist else raw
                fill_frac = raw / h_max if h_max > 1e-8 else 0.0
                
                self._draw_bar(panel, cx, cy, bar_w, fill_frac, lbl, raw, col, bar_h)

            y = sec1_h
            self._hline(panel, y)

            # ── SEÇÃO 2: Ação predita por grupo de juntas (50% da altura)
            sec2_h = PH * 50 // 100
            cv2.rectangle(panel, (0, y), (PW, y + sec2_h), (15, 15, 15), -1)

            chunk_t = len(action_chunk) if action_chunk is not None else 0
            sec2_title = (
                f"Acao predita — barras estaticas (chunk=1, 1 timestep)"
                if chunk_t == 1 else
                f"Acao predita — chunk completo por grupo de juntas ({chunk_t} steps)"
            )
            self._label(panel, sec2_title, PAD, y + 18, 0.42, (220, 220, 220), bold=True)

            if action_chunk is not None and len(action_chunk) >= 1:
                n_groups  = len(self._JOINT_GROUPS)
                grp_w     = (INNER_W - (n_groups - 1) * 6) // n_groups
                grp_h     = sec2_h - 30
                grp_y     = y + 26

                for gi, (grp_label, jrange, gcol) in enumerate(self._JOINT_GROUPS):
                    grp_x = PAD + gi * (grp_w + 6)
                    self._draw_action_group(
                        panel, action_chunk, jrange,
                        grp_x, grp_y, grp_w, grp_h, gcol, grp_label
                    )

                # Legenda simples
                lx = PAD
                for grp_label, _, gcol in self._JOINT_GROUPS:
                    cv2.rectangle(panel, (lx, y + sec2_h - 12),
                                  (lx + 10, y + sec2_h - 2), gcol, -1)
                    self._label(panel, grp_label, lx + 14, y + sec2_h - 3,
                                0.33, (160, 160, 160))
                    lx += 140
            else:
                self._label(panel, "Aguardando chunk...", PAD + 10, y + sec2_h // 2,
                            0.5, (80, 80, 80))

            y += sec2_h
            self._hline(panel, y)

            # ── SEÇÃO 3: Histórico de atenção (25% restante) ──────────
            sec3_h = PH - y
            cv2.rectangle(panel, (0, y), (PW, PH), (18, 18, 18), -1)
            self._label(panel, "Historico de atencao ao longo das inferencias",
                        PAD, y + 18, 0.45, (220, 220, 220), bold=True)

            hist_series = [
                (self._rgb_history,   (60,  200,  60)),
                (self._state_history, (220, 180,  60)),
            ]
            if self.use_depth:
                hist_series.append((self._depth_history, (180, 80, 220)))

            self._draw_history(
                panel, hist_series,
                PAD, y + 24, INNER_W, sec3_h - 40
            )

            # Legenda do histórico
            lx = PAD
            for lbl, col in [("RGB", (60, 200, 60)), ("State", (220, 180, 60)),
                              ("Depth", (180, 80, 220))]:
                if lbl == "Depth" and not self.use_depth:
                    continue
                cv2.rectangle(panel, (lx, PH - 14), (lx + 10, PH - 4), col, -1)
                self._label(panel, lbl, lx + 14, PH - 5, 0.33, (160, 160, 160))
                lx += 90

            self._label(panel, "[Q/ESC] Fechar", PW - 90, PH - 5,
                        0.33, (80, 80, 80))

            with self._lock:
                self._last_rgb_display   = blended
                self._last_panel_display = panel

        except Exception as e:
            print(f"[AttnMapWindow] Erro: {e}")
            import traceback; traceback.print_exc()

    # ─────────────────────────────────────────────────────────────────
    # show() — loop principal
    # ─────────────────────────────────────────────────────────────────

    def show(self) -> bool:
        with self._lock:
            frame_rgb   = self._last_rgb_display
            frame_panel = self._last_panel_display

        # ── placeholder enquanto não há dados ─────────────────────────
        ph_rgb = np.zeros((545, 640, 3), dtype=np.uint8)
        cv2.putText(ph_rgb, "Aguardando 1a inferencia...",
                    (110, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)

        ph_panel = np.zeros((900, 1200, 3), dtype=np.uint8)
        cv2.putText(ph_panel, "Aguardando 1a inferencia...",
                    (350, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 1)

        src_rgb   = frame_rgb   if frame_rgb   is not None else ph_rgb
        src_panel = frame_panel if frame_panel is not None else ph_panel

        # WINDOW_NORMAL faz o Qt backend escalar o frame para preencher
        # a janela automaticamente — não precisamos de resize manual.
        cv2.imshow(self.win_rgb,   cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow(self.win_panel, src_panel)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            return False
        return True

    def destroy(self):
        for win in (self.win_rgb, self.win_panel):
            try:
                cv2.destroyWindow(win)
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

            # 3b. Salva o chunk de ação na policy para o AttnMapWindow ler
            #     (lista de arrays numpy — o painel de ação mostra o chunk completo)
            if attn_window is not None:
                policy._last_action_chunk_np = chunk_np   # sem torch, só numpy

            # 3c. Atualiza o attention map (não bloqueia — tem lock interno)
            if attn_window is not None and _last_rgb is not None:
                attn_window.update(policy, _last_rgb)

            t_inf_ms = (time.perf_counter() - t0) * 1000

            if debug:
                has_attn = getattr(policy, "last_attn_weights", None) is not None
                print(f"\n🧠 [inference] {t_inf_ms:.1f}ms | chunk={len(chunk_np)} ações | attn={'✓' if has_attn else '✗'}")

                # debug temporário — imprime primeiro frame do chunk
            if debug:
                arr = np.array(chunk_np)   # converte lista para array aqui
                print(f"\n[DEBUG] chunk shape={arr.shape} "
                    f"min={arr.min():.4f} max={arr.max():.4f} "
                    f"mean={arr.mean():.4f}")
                print(f"[DEBUG] frame 0 bracos: {arr[0, :14].round(3).tolist()}")
                print(f"[DEBUG] frame 0 maos:   {arr[0, 14:].round(3).tolist()}")

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

    # ── Scene Uncertainty Gate ───────────────────────────────────────────
    # O config.json do checkpoint pode ter scene_uncertainty_threshold > 0
    # salvo do treino. Forçamos 0.0 por padrão na inferência — o gate só
    # ativa se o usuário passar --uncertainty= explicitamente.
    # Motivo: a neutral_position não é salva no checkpoint (é um buffer
    # calculado em runtime no run_train.py). Se o gate estiver ativo mas
    # neutral_position = zeros, o robô é puxado para posição zeros (inválida).
    policy.config.scene_uncertainty_threshold = 0.0

    if uncertainty_threshold > 0:
        policy.config.scene_uncertainty_threshold = uncertainty_threshold
        print(f"✅ Uncertainty Gate ativado: threshold={uncertainty_threshold}")
    else:
        saved_threshold = getattr(policy.config, "scene_uncertainty_threshold", 0.0)
        print(f"   Uncertainty Gate: DESATIVADO (use --uncertainty=X para ativar)")
        print(f"   (valor salvo no checkpoint era {saved_threshold:.2f} — ignorado sem --uncertainty)")

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
        _action_dim = list(policy.config.output_features.values())[0].shape[0]
        aw = AttnMapWindow(
            "Attention Map",
            use_depth=has_depth,
            use_vae=getattr(policy.config, "use_vae", True),
            action_dim=_action_dim,
        )
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
        cv2.namedWindow("Visao da IA", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Visao da IA", 640, 480)
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