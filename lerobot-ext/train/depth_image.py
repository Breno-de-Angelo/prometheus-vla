"""Conversão DEPTH -> IMAGEM (colormap) — util ÚNICO compartilhado entre treino,
sim e inferência (run5, opção B / depth-as-image).

Convenção IDÊNTICA ao viewer `_colorize_depth` de `tools/live_omniview.py`
(o que o Luiz vê na câmera de depth durante a inferência):
  - colormap TURBO
  - VERMELHO = perto, AZUL = longe   (usa 1 - norm)
  - pixels inválidos (profundidade <= 0) -> PRETO
  - faixa métrica FIXA [vis_min_m, vis_max_m] (default 0.2..1.5 m) — preserva
    escala absoluta (mesma cor = mesma distância sempre), o que importa pra
    reach/grasp. NÃO normaliza por-frame.

Saída em [0,1] (B,3,H,W) DE PROPÓSITO: no pi05 (LeRobot) a RGB chega no batch em
[0,1] e o `prepare_images` aplica `img*2-1` a TODA imagem presente
(modeling_pi05.py:1176). Emitindo o depth em [0,1] ele passa pelo MESMO `*2-1`
e termina em [-1,1] EXATAMENTE como a RGB — casando as estatísticas de canal
(o erro clássico seria emitir [-1,1] aqui, que viraria [-3,1] após o *2-1).

Referências: cVLA (arXiv 2507.02190) coloriza depth e usa a MESMA torre RGB.
"""
import os
import numpy as np
import torch

# --- LUT do TURBO (256x3, RGB, [0,1]) gerada uma vez via cv2 (mesmo colormap do viewer) ---
try:
    import cv2
    _lut_bgr = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_TURBO)
    _TURBO_RGB = np.ascontiguousarray(_lut_bgr[:, 0, ::-1])  # BGR -> RGB, (256,3) uint8
except Exception as _ex:  # pragma: no cover
    _TURBO_RGB = None
    _TURBO_IMPORT_ERR = _ex

_TURBO_LUT = None  # cache do tensor


def _turbo_lut(device):
    global _TURBO_LUT
    if _TURBO_LUT is None:
        if _TURBO_RGB is None:
            raise RuntimeError(f"cv2 indisponível para a LUT TURBO: {_TURBO_IMPORT_ERR!r}")
        _TURBO_LUT = torch.from_numpy(_TURBO_RGB).float() / 255.0  # (256,3) [0,1]
    return _TURBO_LUT.to(device=device)


DEFAULT_VIS_MIN_M = float(os.environ.get("DEPTH_VIS_MIN_M", "0.2"))
DEFAULT_VIS_MAX_M = float(os.environ.get("DEPTH_VIS_MAX_M", "1.5"))


def _to_bchw(d: torch.Tensor) -> torch.Tensor:
    """Normaliza shape arbitrário de depth para (B,1,H,W). Treino entrega (B,1,H,W)."""
    if not torch.is_tensor(d):
        d = torch.as_tensor(d)
    if d.dim() == 2:                      # (H,W)
        d = d[None, None]
    elif d.dim() == 3:
        if d.shape[-1] == 1:              # (H,W,1)
            d = d.permute(2, 0, 1)[None]
        elif d.shape[0] == 1:             # (1,H,W)
            d = d[None]
        else:                            # (B,H,W)
            d = d[:, None]
    elif d.dim() == 4:
        if d.shape[1] == 1:              # (B,1,H,W)  <- caso do treino
            pass
        elif d.shape[-1] == 1:           # (B,H,W,1)
            d = d.permute(0, 3, 1, 2)
        else:
            d = d[:, :1]                 # fallback: 1º canal
    else:
        raise ValueError(f"shape de depth não suportado: {tuple(d.shape)}")
    return d


def depth_to_colormap01(depth, vis_min_m=None, vis_max_m=None, depth_scale=0.001) -> torch.Tensor:
    """Profundidade crua -> RGB colormap TURBO em [0,1], shape (B,3,H,W).

    depth: tensor de profundidade crua (mm quando depth_scale=0.001). Aceita
        (B,1,H,W) | (B,H,W,1) | (B,H,W) | (1,H,W) | (H,W).
    vis_min_m/vis_max_m: faixa métrica fixa (default env/0.2..1.5 m).
    depth_scale: fator p/ metros (PNG16 mm -> 0.001).
    """
    vis_min = DEFAULT_VIS_MIN_M if vis_min_m is None else float(vis_min_m)
    vis_max = DEFAULT_VIS_MAX_M if vis_max_m is None else float(vis_max_m)

    d = _to_bchw(depth).float()                         # (B,1,H,W) cru
    d_m = d * float(depth_scale)                        # metros
    valid = d > 0                                       # inválidos = profundidade 0
    norm = ((d_m - vis_min) / (vis_max - vis_min)).clamp(0.0, 1.0)
    idx = ((1.0 - norm) * 255.0).round().long().clamp(0, 255)[:, 0]   # perto->255->vermelho, (B,H,W)
    rgb = _turbo_lut(d.device)[idx]                     # (B,H,W,3) [0,1]
    rgb = rgb.permute(0, 3, 1, 2).contiguous()         # (B,3,H,W)
    rgb = rgb * valid.float()                           # inválidos -> preto
    return rgb
