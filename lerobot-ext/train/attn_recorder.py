"""Captura o mapa de atenção do pi05 durante a inferência (visualização OmniView).

Como funciona: o forward custom do pi05 (lerobot modeling_pi05) computa a atenção
chamando `modeling_gemma.eager_attention_forward(...)` e DESCARTA os pesos
retornados. Este módulo faz monkey-patch nessa função (no módulo do transformers,
que é o mesmo objeto referenciado pelo call site) e, dentro de uma janela de
gravação (context manager), acumula a atenção das queries do action expert sobre
as KV dos TOKENS DE IMAGEM (posições 0..n_img-1 do prefixo).

Filtro importante: o forward do PREFIXO (imagem+língua, Q≈305) também pode passar
por aqui — só interessam as chamadas dos denoise steps (Q = tokens de ação, ~50),
então só registramos quando Q <= max_q_len. O resultado agregado (média sobre
denoise steps × camadas × heads × queries) vira um heatmap 16×16 sobre o RGB.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers.models.gemma import modeling_gemma


class AttnRecorder:
    """Context manager: grava atenção→imagem durante policy.predict_action_chunk."""

    def __init__(self, n_img_tokens: int = 256, max_q_len: int = 64):
        self.n_img = n_img_tokens
        self.max_q_len = max_q_len
        self._orig = None
        self._sum = None
        self._n = 0
        self._active = False

    def install(self) -> "AttnRecorder":
        if self._orig is not None:
            return self
        self._orig = modeling_gemma.eager_attention_forward
        rec = self

        def wrapper(module, query, key, value, attention_mask, *args, **kwargs):
            out, w = rec._orig(module, query, key, value, attention_mask, *args, **kwargs)
            try:
                # w: [B, heads, Q, KV]; só os denoise steps (Q pequeno) e só se a
                # janela KV alcança os tokens de imagem.
                if (
                    rec._active
                    and w is not None
                    and w.dim() == 4
                    and w.shape[2] <= rec.max_q_len
                    and w.shape[3] >= rec.n_img
                ):
                    img_w = w[..., : rec.n_img].float().mean(dim=(0, 1, 2))  # [n_img]
                    rec._sum = img_w if rec._sum is None else rec._sum + img_w
                    rec._n += 1
            except Exception:
                pass  # visualização nunca pode derrubar a inferência
            return out, w

        modeling_gemma.eager_attention_forward = wrapper
        return self

    def __enter__(self):
        self._sum, self._n, self._active = None, 0, True
        return self

    def __exit__(self, *exc):
        self._active = False
        return False

    def heatmap(self) -> np.ndarray | None:
        """Mapa [g, g] normalizado 0-1 (g = sqrt(n_img); 256 tokens → 16×16)."""
        if not self._n or self._sum is None:
            return None
        g = int(round(self.n_img ** 0.5))
        h = (self._sum / self._n).reshape(g, g).cpu().numpy()
        h = h - h.min()
        mx = h.max()
        return h / mx if mx > 0 else h


def overlay_heatmap(rgb: np.ndarray, heat01: np.ndarray, alpha: float = 0.45,
                    out_width: int = 424) -> np.ndarray:
    """RGB uint8 (H,W,3) + heatmap 0-1 → BGR uint8 com JET, redimensionado p/ web."""
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if out_width and bgr.shape[1] > out_width:
        s = out_width / bgr.shape[1]
        bgr = cv2.resize(bgr, (out_width, int(bgr.shape[0] * s)))
    hm = cv2.resize((heat01 * 255).astype(np.uint8), (bgr.shape[1], bgr.shape[0]),
                    interpolation=cv2.INTER_CUBIC)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1.0 - alpha, hm, alpha, 0)
