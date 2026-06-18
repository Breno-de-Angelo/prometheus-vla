"""Injeta DEPTH COMO IMAGEM (2ª câmera) na política PI05 — opção B (run5).

Em vez de PointNet (token único, `inject_pi05_d`), o mapa de profundidade é
colorizado (TURBO, vermelho=perto) e tratado como uma SEGUNDA imagem pela MESMA
torre SigLIP do pi05 (~256 tokens próprios, posição preservada). NÃO adiciona
NENHUM parâmetro novo: reusa a torre visual existente, então o checkpoint é
estruturalmente idêntico ao run4b (EMA/resume/save inalterados).

Mecânica (espelha `inject_pi05_d`, ao contrário): a cada forward/predict,
converte `batch[depth_key]` (profundidade crua, mm) -> imagem [0,1] (B,3,H,W) e
ADICIONA temporariamente `depth_key` a `config.input_features` (type VISUAL) —
que vira automaticamente uma entrada de `config.image_features` (property que
filtra VISUAL) e é processada pelo loop de imagens nativo do pi05
(modeling_pi05.py:1143). Restaura tudo no `finally`.
"""
import os
import types
import torch

from lerobot.configs.types import PolicyFeature, FeatureType
from train.depth_image import depth_to_colormap01


def inject_pi05_depth_image(policy,
                            depth_key: str = "observation.images.head_camera_depth",
                            vis_min_m: float = 0.2, vis_max_m: float = 1.5,
                            depth_scale: float = 0.001, debug_dir: str | None = None):
    print(f"\n[INJECAO PI05 DEPTH-IMAGE]: depth como 2a imagem (TURBO, vermelho=perto, "
          f"faixa [{vis_min_m},{vis_max_m}]m, scale={depth_scale}, key={depth_key})")

    policy._depth_img_sanity_done = False

    def _to_image(self, batch):
        depth = batch.get(depth_key, None)
        if depth is None:
            return None  # degrade: sem depth nesse step -> passada pi05 normal (1 câmera)
        img = depth_to_colormap01(depth, vis_min_m=vis_min_m, vis_max_m=vis_max_m,
                                  depth_scale=depth_scale)  # (B,3,H,W) [0,1]
        if torch.is_tensor(depth):
            img = img.to(device=depth.device)
        if not self._depth_img_sanity_done:
            self._depth_img_sanity_done = True
            _sanity(depth, img, debug_dir)
        return img

    def _run_with_depth_image(self, runner, batch, *args, **kwargs):
        img = _to_image(self, batch)
        if img is None:
            return runner(batch, *args, **kwargs)
        saved = batch.get(depth_key)
        batch[depth_key] = img
        added = depth_key not in self.config.input_features
        if added:
            _, c, h, w = img.shape
            self.config.input_features[depth_key] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))
        try:
            out = runner(batch, *args, **kwargs)
        finally:
            if added:
                self.config.input_features.pop(depth_key, None)
            batch[depth_key] = saved
        return out

    original_forward = policy.forward
    original_predict = policy.predict_action_chunk

    def patched_forward(self, batch, *a, **k):
        return _run_with_depth_image(self, original_forward, batch, *a, **k)

    def patched_predict(self, batch, *a, **k):
        return _run_with_depth_image(self, original_predict, batch, *a, **k)

    policy.forward = types.MethodType(patched_forward, policy)
    policy.predict_action_chunk = types.MethodType(patched_predict, policy)
    print("[INJECAO PI05 DEPTH-IMAGE]: Concluida. Depth entra como 2a camera SigLIP.\n")


def _sanity(depth, img, debug_dir):
    d = depth.float()
    print(f"[depth-image sanity] depth_in shape={tuple(depth.shape)} dtype={depth.dtype} "
          f"min={float(d.min()):.1f} max={float(d.max()):.1f} mean={float(d.mean()):.1f} | "
          f"img_out shape={tuple(img.shape)} dtype={img.dtype} "
          f"min={float(img.min()):.3f} max={float(img.max()):.3f} mean={float(img.mean()):.3f}")
    if debug_dir:
        try:
            import numpy as np, cv2
            os.makedirs(debug_dir, exist_ok=True)
            rgb = (img[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            p = os.path.join(debug_dir, "depth_image_sample.png")
            cv2.imwrite(p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"[depth-image sanity] amostra salva em {p}")
        except Exception as ex:
            print("[depth-image sanity] falha ao salvar amostra:", repr(ex)[:120])
