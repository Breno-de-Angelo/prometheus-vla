"""Injeção de profundidade 3D (PointNet) + tato (pressão Dex3) na política PI05.

Paralelo ao `act_d_injector`, mas adaptado à arquitetura PaliGemma + Gemma-Expert do PI05:
em vez de somar features ao "state token" do ACT, acoplamos tokens extras ao final da
sequência de prefixo (logo após os tokens de linguagem), antes do `torch.cat` em
`PI05Pytorch.embed_prefix`. A largura dos tokens extras é ajustada ao `hidden_size` do VLM.
"""
import logging
import types
from pathlib import Path

import safetensors.torch as st
import torch
import torch.nn as nn

from train.depth_encoder import PointNetEncoder, depth_to_pointcloud


DEPTH_KEY = "observation.images.head_camera_depth"
LEFT_PRESSURE_KEY = "observation.left_hand_pressure"
RIGHT_PRESSURE_KEY = "observation.right_hand_pressure"


def _vlm_hidden_size(policy) -> int:
    embed = policy.model.paligemma_with_expert.paligemma.language_model.embed_tokens
    return embed.weight.shape[1]


def _load_injected_weights(policy, checkpoint_dir):
    """Se um checkpoint com `pointnet.*` / `pressure_proj.*` for passado, carrega-os
    na policy recém-injetada (caso contrário eles ficariam com init aleatório após
    o make_policy silenciosamente descartar esses keys como 'unexpected')."""
    sd_path = Path(checkpoint_dir) / "model.safetensors"
    if not sd_path.exists():
        return 0
    sd = st.load_file(str(sd_path))
    injected = {k: v for k, v in sd.items() if "pointnet" in k or "pressure_proj" in k}
    if not injected:
        return 0
    missing, unexpected = policy.load_state_dict(injected, strict=False)
    return len(injected)


def inject_pi05_d(policy, device, camera_intrinsics=None, pressure_dim=66, load_injected_from=None):
    """Acopla PointNet + encoder de pressão ao PI05 e injeta tokens extras no prefixo.

    - Dois tokens são acrescentados ao final da sequência de prefixo:
          [IMG_TOK..., LANG_TOK..., DEPTH_TOK, PRESSURE_TOK]
      Com `att_masks=0` (mesmo bloco de prefixo dos demais) e `pad_masks=True`.
    - `observation.images.head_camera_depth` é escondido de `config.input_features`
      durante o forward, evitando que o pipeline SigLIP tente processá-lo como RGB.
    - Depth / pressões são retirados do batch durante o forward e restaurados ao fim.
    - Se o batch não trouxer depth/pressão, o injector degrada para uma passada pi05
      nativa (nenhum token extra é adicionado naquele step).
    """
    print("\n[INJECAO PI05-D]: Ativando fusao 3D (PointNet) + tato (Dex3) para PI05...")

    hidden_size = _vlm_hidden_size(policy)

    policy.pointnet = PointNetEncoder(output_dim=hidden_size).to(device)
    policy.pressure_proj = nn.Sequential(
        nn.Linear(pressure_dim, 256),
        nn.ReLU(),
        nn.Linear(256, hidden_size),
    ).to(device)
    policy.camera_intrinsics = camera_intrinsics or {
        "fx": 600.0,
        "fy": 600.0,
        "cx": 320.0,
        "cy": 240.0,
    }

    model = policy.model
    model._extra_prefix_embs = None

    original_embed_prefix = model.embed_prefix

    def patched_embed_prefix(self, images, img_masks, tokens, masks):
        embs, pad_masks, att_masks = original_embed_prefix(images, img_masks, tokens, masks)

        extra = getattr(self, "_extra_prefix_embs", None)
        if extra is None:
            return embs, pad_masks, att_masks

        extra_embs, extra_pad_masks = extra
        extra_embs = extra_embs.to(dtype=embs.dtype, device=embs.device)
        extra_pad_masks = extra_pad_masks.to(device=pad_masks.device)

        bsize, k, _ = extra_embs.shape
        embs = torch.cat([embs, extra_embs], dim=1)
        pad_masks = torch.cat([pad_masks, extra_pad_masks], dim=1)
        extra_att = torch.zeros(bsize, k, dtype=att_masks.dtype, device=att_masks.device)
        att_masks = torch.cat([att_masks, extra_att], dim=1)

        return embs, pad_masks, att_masks

    model.embed_prefix = types.MethodType(patched_embed_prefix, model)

    original_forward = policy.forward
    original_predict = policy.predict_action_chunk

    def _compute_extra_tokens(self, batch):
        depth = batch.pop(DEPTH_KEY, None)
        left = batch.pop(LEFT_PRESSURE_KEY, None)
        right = batch.pop(RIGHT_PRESSURE_KEY, None)

        tokens = []
        if depth is not None:
            # depth chega em mm; converte pra metros na back-projection
            pc = depth_to_pointcloud(depth.float(), self.camera_intrinsics, depth_scale=0.001)
            depth_tok = self.pointnet(pc).unsqueeze(1)
            tokens.append(depth_tok)
        if left is not None and right is not None:
            full_pressure = torch.cat([left.float(), right.float()], dim=1)
            press_tok = self.pressure_proj(full_pressure).unsqueeze(1)
            tokens.append(press_tok)

        saved = (depth, left, right)
        if not tokens:
            return None, saved

        extra_embs = torch.cat(tokens, dim=1)
        bsize, k, _ = extra_embs.shape
        extra_pad = torch.ones(bsize, k, dtype=torch.bool, device=extra_embs.device)
        return (extra_embs, extra_pad), saved

    def _restore_batch(batch, saved):
        depth, left, right = saved
        if depth is not None:
            batch[DEPTH_KEY] = depth
        if left is not None:
            batch[LEFT_PRESSURE_KEY] = left
        if right is not None:
            batch[RIGHT_PRESSURE_KEY] = right

    def _run_with_extras(self, runner, batch, *args, **kwargs):
        extras, saved = _compute_extra_tokens(self, batch)
        removed_feat = self.config.input_features.pop(DEPTH_KEY, None)
        try:
            self.model._extra_prefix_embs = extras
            out = runner(batch, *args, **kwargs)
        finally:
            self.model._extra_prefix_embs = None
            if removed_feat is not None:
                self.config.input_features[DEPTH_KEY] = removed_feat
            _restore_batch(batch, saved)
        return out

    def patched_forward(self, batch, *args, **kwargs):
        return _run_with_extras(self, original_forward, batch, *args, **kwargs)

    def patched_predict(self, batch, *args, **kwargs):
        return _run_with_extras(self, original_predict, batch, *args, **kwargs)

    policy.forward = types.MethodType(patched_forward, policy)
    policy.predict_action_chunk = types.MethodType(patched_predict, policy)

    if load_injected_from is not None:
        n = _load_injected_weights(policy, load_injected_from)
        if n > 0:
            print(f"[INJECAO PI05-D]: carregados {n} pesos treinados de {load_injected_from}")

    print("[INJECAO PI05-D]: Concluida. Profundidade e tato inseridos como tokens de prefixo.\n")
