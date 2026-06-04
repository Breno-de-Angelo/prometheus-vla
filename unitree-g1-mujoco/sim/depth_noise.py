"""Modelo procedural de ruído pra aproximar o depth ground-truth do MuJoCo
da Intel RealSense D435/D435i.

O MuJoCo entrega depth geométrico perfeito (a distância exata de cada pixel,
sem ruído). A D435 real, sendo um sensor de stereo ativo por IR, tem:

  - erro axial (eixo Z) que cresce ~quadraticamente com a distância
    (vem da quantização de disparidade: z = f*b/disp),
  - buracos/halo nas bordas de objetos (flying pixels),
  - dropout em regiões sem textura / oblíquas,
  - alcance útil limitado (~0.2 m a ~3 m), fora disso = inválido (0).

Este módulo injeta esses efeitos sobre o depth em METROS, antes de virar mm.
É um modelo procedural (nível 1 — barato) na linha do simkinect / Barron-Malik
2013 / Bohg 2014, parametrizado com defaults da D435. NÃO simula ray-tracing do
padrão IR (isso seria o DREDS, em Blender) nem ruído aprendido por difusão.

Referências:
  - Ahn et al., "Analysis and Noise Modeling of the Intel RealSense D435 for
    Mobile Robots", IEEE 2019.
  - Handa et al. (simkinect), Barron & Malik 2013, Bohg et al. 2014.

Pixels inválidos são representados por 0.0 (mesma convenção da D435/librealsense).
"""

import os
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class DepthNoiseModel:
    """Aplica ruído estilo-D435 sobre um mapa de profundidade em metros.

    Uso:
        model = DepthNoiseModel.from_config(config)
        if model.enabled:
            depth_m = model.apply(depth_m)   # (H, W) float32, 0 = inválido
    """

    def __init__(
        self,
        enabled: bool = True,
        z_min: float = 0.2,            # alcance mínimo útil da D435 (m)
        z_max: float = 3.0,            # alcance máximo confiável p/ manipulação (m)
        baseline_m: float = 0.05,      # baseline do par stereo da D435 (~50 mm)
        focal_px: float | None = None, # focal do sensor de depth em px (auto se None)
        subpix_sigma: float = 0.2,     # desvio-padrão do ruído de disparidade (px)
        quantize: bool = True,         # quantiza disparidade (resolução cresce com z²)
        subpix_levels: int = 8,        # passos de sub-pixel de disparidade (1/8 px)
        edge_thresh_m: float = 0.02,   # salto de profundidade que conta como borda (m)
        edge_dilate: int = 2,          # dilata o halo de buracos nas bordas (px)
        p_dropout: float = 0.005,      # fração de pixels válidos zerados (speckle)
        lateral_sigma_px: float = 0.0, # jitter lateral (0 = desligado; custa caro)
        seed: int = 0,
    ):
        self.enabled = enabled and _HAS_CV2
        if enabled and not _HAS_CV2:
            print("[DepthNoise] ⚠️ cv2 indisponível — modelo de ruído DESLIGADO.")
        self.z_min = z_min
        self.z_max = z_max
        self.baseline_m = baseline_m
        self.focal_px = focal_px
        self.subpix_sigma = subpix_sigma
        self.quantize = quantize
        self.subpix_levels = subpix_levels
        self.edge_thresh_m = edge_thresh_m
        self.edge_dilate = edge_dilate
        self.p_dropout = p_dropout
        self.lateral_sigma_px = lateral_sigma_px
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_config(cls, config: dict) -> "DepthNoiseModel":
        """Lê a seção DEPTH_NOISE do config.yaml. Override por env var
        G1_DEPTH_NOISE (1/0/true/false) tem precedência sobre o config."""
        cfg = dict(config.get("DEPTH_NOISE", {}) or {})
        enabled = bool(cfg.pop("enabled", True))

        env = os.environ.get("G1_DEPTH_NOISE")
        if env is not None:
            enabled = env.strip().lower() in ("1", "true", "yes", "on")

        # remove chaves desconhecidas pra não quebrar o __init__
        valid = {
            "z_min", "z_max", "baseline_m", "focal_px", "subpix_sigma",
            "quantize", "subpix_levels", "edge_thresh_m", "edge_dilate",
            "p_dropout", "lateral_sigma_px", "seed",
        }
        kwargs = {k: v for k, v in cfg.items() if k in valid}
        model = cls(enabled=enabled, **kwargs)
        if model.enabled:
            print(
                f"[DepthNoise] ✅ ativo (z∈[{model.z_min},{model.z_max}]m, "
                f"subpix_sigma={model.subpix_sigma}px, dropout={model.p_dropout}, "
                f"edges={model.edge_thresh_m}m). Desligue com G1_DEPTH_NOISE=0."
            )
        return model

    def _focal(self, height: int) -> float:
        if self.focal_px is not None:
            return float(self.focal_px)
        # D435 a 480 linhas, VFOV de depth ~58° -> fy = (H/2)/tan(VFOV/2)
        vfov = np.deg2rad(58.0)
        return (height / 2.0) / np.tan(vfov / 2.0)

    def apply(self, depth_m: np.ndarray) -> np.ndarray:
        """depth_m: (H, W) ou (H, W, 1) float, em metros. Retorna (mesma forma)
        com ruído estilo-D435 e inválidos = 0."""
        if not self.enabled:
            return depth_m

        squeeze_back = False
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]
            squeeze_back = True

        H, W = depth_m.shape
        z = depth_m.astype(np.float32)
        fx = self._focal(H)
        fb = fx * self.baseline_m

        # --- máscara de validade a partir do depth LIMPO ---
        valid = np.isfinite(z) & (z >= self.z_min) & (z <= self.z_max)

        # --- buracos/halo nas bordas (descontinuidade de profundidade) ---
        # gradiente em metros; bordas viram inválidas e são dilatadas.
        # scale=1/8 normaliza o Sobel ksize=3 -> gradiente real (m por pixel),
        # senão o operador devolve ~8x e qualquer superfície inclinada vira "borda".
        gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
        gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
        grad = cv2.magnitude(gx, gy)
        edges = grad > self.edge_thresh_m
        if self.edge_dilate > 0:
            k = np.ones((self.edge_dilate * 2 + 1,) * 2, np.uint8)
            edges = cv2.dilate(edges.astype(np.uint8), k).astype(bool)
        valid &= ~edges

        # --- ruído axial via disparidade (quadrático em z) + quantização ---
        zc = np.clip(z, self.z_min, self.z_max)
        disp = fb / zc
        disp = disp + self.rng.normal(0.0, self.subpix_sigma, size=disp.shape).astype(np.float32)
        if self.quantize and self.subpix_levels > 0:
            disp = np.round(disp * self.subpix_levels) / self.subpix_levels
        disp = np.maximum(disp, 1e-3)
        z_noisy = fb / disp

        # --- dropout aleatório (speckle em superfícies sem textura) ---
        if self.p_dropout > 0:
            drop = self.rng.random(z.shape) < self.p_dropout
            valid &= ~drop

        # --- jitter lateral opcional (caro; off por padrão) ---
        if self.lateral_sigma_px > 0:
            ys, xs = np.indices((H, W), dtype=np.float32)
            xs += self.rng.normal(0.0, self.lateral_sigma_px, size=(H, W)).astype(np.float32)
            ys += self.rng.normal(0.0, self.lateral_sigma_px, size=(H, W)).astype(np.float32)
            z_noisy = cv2.remap(
                z_noisy, xs, ys, interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

        out = np.where(valid, z_noisy, 0.0).astype(np.float32)
        if squeeze_back:
            out = out[..., np.newaxis]
        return out
