#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""
PI05-Depth Policy — Improved
Melhorias sobre a versão original:
  1. Scene Uncertainty Gate: o Flow Matching não tem VAE, então a incerteza é
     estimada pela VARIÂNCIA entre múltiplas amostras de ruído inicial (Monte Carlo).
     Quando alta, a ação é suavemente misturada com a posição neutra.
  2. neutral_position: register_buffer (salvo no checkpoint), inicializado pelos
     default_positions do UnitreeG1Config via neutral_position.py.
  3. pressure_tensor duplicação removida: o forward() e predict_action_chunk()
     tinham código duplicado para extrair pressão — unificado em _extract_pressure().
  4. _preprocess_images movida para PI05DEPTHPolicy (estava em model mas chamada
     inconsistentemente via self.model._preprocess_images no predict_action_chunk).
  5. Logging de incerteza: incerteza média exposta no output_dict para o WandB.
"""

import builtins
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.utils.import_utils import _transformers_available

from .depth_encoder import PointNetEncoder, depth_to_pointcloud

if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    GemmaForCausalLM = None
    PaliGemmaForConditionalGeneration = None

from lerobot.configs.policies import PreTrainedConfig
from .configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05DEPTHConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def _get_lm(paligemma):
    """Compatibilidade entre versões do transformers.
    4.46.x: PaliGemmaForConditionalGeneration
              .model (PaliGemmaModel)
                .language_model (GemmaModel)   ← aqui
    4.49+ : PaliGemmaForConditionalGeneration
              .language_model (GemmaModel)     ← direto
    """
    # Tenta direto no objeto (4.49+)
    lm = getattr(paligemma, "language_model", None)
    if lm is not None:
        return lm
    # Tenta via .model (4.46.x)
    inner = getattr(paligemma, "model", None)
    if inner is not None:
        lm = getattr(inner, "language_model", None)
        if lm is not None:
            return lm
    raise RuntimeError(
        f"_get_lm: não encontrei language_model em {type(paligemma)}. "
        f"Sub-módulos: {[n for n, _ in paligemma.named_children()]}"
    )


def _get_vision_tower(paligemma):
    """Retorna o vision encoder independente da versão do transformers.
    4.46.x: paligemma.model.vision_tower
    4.49+ : paligemma.vision_tower
    """
    vt = getattr(paligemma, "vision_tower", None)
    if vt is not None:
        return vt
    inner = getattr(paligemma, "model", None)
    if inner is not None:
        vt = getattr(inner, "vision_tower", None)
        if vt is not None:
            return vt
    raise RuntimeError(
        f"_get_vision_tower: não encontrei vision_tower em {type(paligemma)}. "
        f"Sub-módulos: {[n for n, _ in paligemma.named_children()]}"
    )


def _get_projector(paligemma):
    """Retorna o multi_modal_projector independente da versão do transformers.
    4.46.x: paligemma.model.multi_modal_projector
    4.49+ : paligemma.multi_modal_projector
    """
    proj = getattr(paligemma, "multi_modal_projector", None)
    if proj is not None:
        return proj
    inner = getattr(paligemma, "model", None)
    if inner is not None:
        proj = getattr(inner, "multi_modal_projector", None)
        if proj is not None:
            return proj
    raise RuntimeError(
        f"_get_projector: não encontrei multi_modal_projector em {type(paligemma)}. "
        f"Sub-módulos: {[n for n, _ in paligemma.named_children()]}"
    )


def get_safe_dtype(target_dtype, device_type):
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    batch_size, channels, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=constant_value,
    )

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)

    return padded_images


def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    models = [_get_lm(paligemma), gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []

    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        if adarms_cond[i] is not None:
            hidden_states = hidden_states + adarms_cond[i].unsqueeze(1)
        hidden_states = layer.input_layernorm(hidden_states)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)

    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )

    cos, sin = _get_lm(paligemma).rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    batch_size = query_states.shape[0]
    scaling = _get_lm(paligemma).layers[layer_idx].self_attn.scaling

    att_output, _ = modeling_gemma.eager_attention_forward(
        _get_lm(paligemma).layers[layer_idx].self_attn,
        query_states, key_states, value_states,
        attention_mask, scaling,
    )

    head_dim = _get_lm(paligemma).layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        out_emb = hidden_states + out_emb
        after_first_residual = out_emb.clone()
        if adarms_cond[i] is not None:
            out_emb = out_emb + adarms_cond[i].unsqueeze(1)
        out_emb = layer.post_attention_layernorm(out_emb)
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = after_first_residual + out_emb
        outputs_embeds.append(out_emb)
        start_pos = end_pos

    return outputs_embeds


class GemmaConfig:
    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:
    if variant == "gemma_300m":
        return GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
    elif variant == "gemma_2b":
        return GemmaConfig(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256)
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(nn.Module):
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None

        self.paligemma = PaliGemmaForConditionalGeneration(vlm_config_hf)

        expert_config_hf = CONFIG_MAPPING["gemma"]()
        expert_config_hf.hidden_size = action_expert_config.width
        expert_config_hf.intermediate_size = action_expert_config.mlp_dim
        expert_config_hf.num_attention_heads = action_expert_config.num_heads
        expert_config_hf.head_dim = action_expert_config.head_dim
        expert_config_hf.num_hidden_layers = action_expert_config.depth
        expert_config_hf.num_key_value_heads = action_expert_config.num_kv_heads
        expert_config_hf.hidden_activation = "gelu_pytorch_tanh"
        expert_config_hf.torch_dtype = "float32"
        expert_config_hf.vocab_size = 257152
        expert_config_hf.use_adarms = use_adarms[1]
        expert_config_hf.adarms_cond_dim = action_expert_config.width if use_adarms[1] else None

        self.gemma_expert = GemmaForCausalLM(expert_config_hf)

        if freeze_vision_encoder:
            for param in _get_vision_tower(self.paligemma).parameters():
                param.requires_grad = False

        if train_expert_only:
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def embed_image(self, image):
        # Passo 1: vision encoder → [B, num_patches, 1152]
        vision_out = _get_vision_tower(self.paligemma)(pixel_values=image, output_hidden_states=False)
        if isinstance(vision_out, torch.Tensor):
            vision_feats = vision_out
        elif hasattr(vision_out, "last_hidden_state"):
            vision_feats = vision_out.last_hidden_state
        elif hasattr(vision_out, "pooler_output"):
            vision_feats = vision_out.pooler_output
        else:
            raise RuntimeError(f"embed_image: tipo inesperado do vision encoder: {type(vision_out)}")

        # Passo 2: projeta para o espaço do LM → [B, num_patches, lm_hidden_size]
        vision_feats = _get_projector(self.paligemma)(vision_feats)
        return vision_feats

    def embed_language_tokens(self, tokens):
        return _get_lm(self.paligemma).get_input_embeddings()(tokens)

    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=False,
        adarms_cond=None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = _get_lm(self.paligemma).forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [_get_lm(self.paligemma), self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond,
                        use_reentrant=False, preserve_rng_state=False,
                        paligemma=self.paligemma, gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond,
                        paligemma=self.paligemma, gemma_expert=self.gemma_expert,
                    )

            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    if adarms_cond[i] is not None:
                        hidden_states = hidden_states + adarms_cond[i].unsqueeze(1)
                    out_emb = models[i].norm(hidden_states)
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):
    """Core PI05 PyTorch model."""

    def __init__(self, config: PI05DEPTHConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config, action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        if self.config.use_pressure:
            self.pressure_proj = nn.Sequential(
                nn.Linear(self.config.pressure_feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, paligemma_config.width),
            )

        if self.config.use_depth_3d:
            self.pointnet = PointNetEncoder(output_dim=paligemma_config.width)

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.gradient_checkpointing_enabled = False

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        _get_lm(self.paligemma_with_expert.paligemma).gradient_checkpointing = True
        _get_vision_tower(self.paligemma_with_expert.paligemma).gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        _get_lm(self.paligemma_with_expert.paligemma).gradient_checkpointing = False
        _get_vision_tower(self.paligemma_with_expert.paligemma).gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        return (time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset).to(
            dtype=torch.float32, device=device
        )

    def embed_prefix(
        self, images, img_masks, tokens, masks, depth_images=None, pressure_tensor=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._apply_checkpoint(self.paligemma_with_expert.embed_image, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        # ── ACT-D: Nuvem de Pontos 3D ────────────────────────────────────────
        if self.config.use_depth_3d and depth_images is not None and len(depth_images) > 0:
            depth_tensor = depth_images[0]
            pc = depth_to_pointcloud(
                depth_tensor,
                self.config.camera_intrinsics,
                num_points=self.config.pointnet_num_points,
            )
            f3d = self._apply_checkpoint(self.pointnet, pc)   # [B, hidden_dim]
            f3d = f3d.unsqueeze(1)                             # [B, 1, hidden_dim]
            bsize = f3d.shape[0]
            embs.append(f3d)
            pad_masks.append(torch.ones((bsize, 1), dtype=torch.bool, device=masks.device))
            att_masks += [0]

        # ── ACT-D: Tato / Pressão ────────────────────────────────────────────
        if self.config.use_pressure and pressure_tensor is not None:
            fpress = self._apply_checkpoint(self.pressure_proj, pressure_tensor)
            fpress = fpress.unsqueeze(1)                       # [B, 1, hidden_dim]
            bsize = fpress.shape[0]
            embs.append(fpress)
            pad_masks.append(torch.ones((bsize, 1), dtype=torch.bool, device=masks.device))
            att_masks += [0]

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        embs = []
        pad_masks = []
        att_masks = []

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        ).type(dtype=timestep.dtype)

        action_emb = self._apply_checkpoint(self.action_in_proj, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        pad_masks.append(torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device))
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self, images, img_masks, tokens, masks, actions,
        noise=None, time=None, depth_images=None, pressure_tensor=None
    ) -> Tensor:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks,
            depth_images=depth_images, pressure_tensor=pressure_tensor,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            _get_lm(self.paligemma_with_expert.paligemma).layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size:].to(dtype=torch.float32)
        v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(
        self,
        images, img_masks, tokens, masks,
        depth_images=None, pressure_tensor=None,
        noise=None, num_steps=None,
        n_samples_for_uncertainty: int = 1,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> tuple[Tensor, float]:
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device
        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)

        # ── Prefixo: computado EM CADA DENOISE STEP (sem cache) ────────────────────
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks,
            depth_images=depth_images, pressure_tensor=pressure_tensor,
        )
        
        dt = -1.0 / num_steps

        # ── Denoising (potencialmente múltiplas amostras) ─────────────────────
        all_results = []
        n_runs = max(1, n_samples_for_uncertainty)

        for run_idx in range(n_runs):
            if noise is not None and run_idx == 0:
                x_t = noise
            else:
                x_t = self.sample_noise(actions_shape, device)

            for step in range(num_steps):
                time = 1.0 + step * dt
                time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

                # ✅ FIX: Passe os prefix embs diretamente (sem cache)
                v_t = self.denoise_step(
                    prefix_embs=prefix_embs,
                    prefix_pad_masks=prefix_pad_masks,
                    prefix_att_masks=prefix_att_masks,
                    x_t=x_t,
                    timestep=time_tensor,
                )

                x_t = x_t + dt * v_t

            all_results.append(x_t)

        # ── Incerteza estimada pela variância entre amostras ──────────────────
        if n_runs > 1:
            stacked = torch.stack(all_results, dim=0)
            uncertainty = stacked.std(dim=0).mean().item()
            actions = stacked.mean(dim=0)
        else:
            uncertainty = 0.0
            actions = all_results[0]

        return actions, uncertainty

    def denoise_step(self, prefix_embs, prefix_pad_masks, prefix_att_masks, x_t, timestep):
        """Denoise sem cache — recomputa prefix cada vez (mais simples, mais seguro)."""
        
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # ✅ SIMPLES: combina prefix + suffix diretamente
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        _get_lm(self.paligemma_with_expert.paligemma).config._attn_implementation = "eager"
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        # ✅ Recomputa prefix + suffix em um só forward (sem cache)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,  # ← SEM CACHE
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1][:, -self.config.chunk_size:].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

# ══════════════════════════════════════════════════════════════════════════════
# PI05DEPTHPolicy — com neutral position gate e best-val support
# ══════════════════════════════════════════════════════════════════════════════
class PI05DEPTHPolicy(PreTrainedPolicy):
    """PI05 Policy — ACT-D Improved (Depth + Pressure + Uncertainty Gate)."""

    config_class = PI05DEPTHConfig
    name = "pi05depth"

    def __init__(self, config: PI05DEPTHConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        # ── Neutral position buffer ───────────────────────────────────────────
        # Salvo com o checkpoint. Injetado pelo run_train via neutral_position.py.
        # O uncertainty gate usa este buffer durante inferência para misturar a
        # predição com a posição segura quando a incerteza (variância Monte Carlo)
        # superar o limiar configurado em scene_uncertainty_threshold.
        original_action_dim = list(config.output_features.values())[0].shape[0]
        self._original_action_dim = original_action_dim
        self.register_buffer("neutral_position", torch.zeros(original_action_dim))

        self.reset()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list, list, list]:
        """Separa imagens RGB (→ SigLIP) das imagens de depth (→ PointNet)."""
        images = []
        img_masks = []
        depth_images = []

        device = next(self.parameters()).device
        present_img_keys = [key for key in self.config.image_features if key in batch]

        for key in present_img_keys:
            img = batch[key]

            if "depth" in key.lower():
                if img.device != device:
                    img = img.to(device)
                if img.dtype != torch.float32:
                    img = img.to(torch.float32)
                depth_images.append(img)
                continue  # Não passa pelo SigLIP

            if img.device != device:
                img = img.to(device)
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            is_channels_first = img.shape[1] == 3
            if is_channels_first:
                img = img.permute(0, 2, 3, 1)
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)
            img = img * 2.0 - 1.0
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)

            images.append(img)
            bsize = img.shape[0]
            img_masks.append(torch.ones(bsize, dtype=torch.bool, device=device))

        missing_img_keys = [key for key in self.config.image_features if key not in batch]
        for _ in range(len(missing_img_keys)):
            if images:
                img = torch.ones_like(images[-1]) * -1
                mask = torch.zeros(img.shape[0], dtype=torch.bool, device=device)
            else:
                img = torch.ones((1, 3, *self.config.image_resolution), device=device) * -1
                mask = torch.zeros((1,), dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks, depth_images

    def _extract_pressure(self, batch: dict[str, Tensor]) -> Tensor | None:
        """Extrai e concatena os tensores de pressão se presentes."""
        left = batch.get("observation.left_hand_pressure")
        right = batch.get("observation.right_hand_pressure")
        if left is not None and right is not None:
            return torch.cat([left, right], dim=1)
        return None

    def _apply_uncertainty_gate(
        self,
        actions: Tensor,           # [B, chunk_size, action_dim]
        uncertainty: float,
        threshold: float,
    ) -> Tensor:
        """
        Mistura as ações preditas com a posição neutra quando a incerteza
        (variância Monte Carlo entre amostras de ruído) supera o limiar.

        Flow Matching não tem log_sigma como o VAE do ACT.
        Aqui a incerteza é o std médio entre n_samples_for_uncertainty runs.

        blend_alpha = clamp((uncertainty - threshold) / threshold, 0, 1)
        actions_safe = (1 - alpha) * actions + alpha * neutral
        """
        if threshold <= 0 or uncertainty <= 0:
            return actions

        excess = (uncertainty - threshold) / (threshold + 1e-6)
        blend_alpha = max(0.0, min(1.0, excess))

        if blend_alpha < 0.01:
            return actions

        neutral = (
            self.neutral_position
            .to(actions.device)
            .unsqueeze(0).unsqueeze(0)    # [1, 1, action_dim]
            .expand_as(actions)
        )
        actions_safe = (1.0 - blend_alpha) * actions + blend_alpha * neutral

        logging.debug(
            f"[UncertaintyGate/PI05] uncertainty={uncertainty:.4f} > threshold={threshold:.4f} "
            f"→ blend_alpha={blend_alpha:.3f}"
        )
        return actions_safe

    def prepare_action(self, batch):
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    # ── Inferência ────────────────────────────────────────────────────────────

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        self._last_uncertainty = 0.0

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        assert not self._rtc_enabled(), "RTC não suportado em select_action"
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        images, img_masks, depth_images = self._preprocess_images(batch)
        pressure_tensor = self._extract_pressure(batch)
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # ── Incerteza via Monte Carlo (n > 1 amostra) ─────────────────────────
        threshold = getattr(self.config, "scene_uncertainty_threshold", 0.0)
        n_samples = getattr(self.config, "n_samples_uncertainty", 1)
        # Se threshold > 0, ativa automaticamente 3 amostras (configurável)
        if threshold > 0 and n_samples <= 1:
            n_samples = 3

        actions, uncertainty = self.model.sample_actions(
            images, img_masks, tokens, masks,
            depth_images=depth_images,
            pressure_tensor=pressure_tensor,
            n_samples_for_uncertainty=n_samples,
            **kwargs,
        )
        self._last_uncertainty = uncertainty

        # ── Scene Uncertainty Gate ────────────────────────────────────────────
        if threshold > 0:
            actions = self._apply_uncertainty_gate(actions, uncertainty, threshold)

        # Unpad para a dimensão real da ação
        actions = actions[:, :, : self._original_action_dim]

        return actions

    # ── Treinamento ───────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        images, img_masks, depth_images = self._preprocess_images(batch)
        pressure_tensor = self._extract_pressure(batch)
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)

        losses = self.model.forward(
            images, img_masks, tokens, masks, actions,
            depth_images=depth_images, pressure_tensor=pressure_tensor,
        )

        # Trunca para a dimensão real da ação
        losses = losses[:, :, : self._original_action_dim]

        loss_per_dim = losses.mean(dim=[0, 1]).detach().cpu()  # [action_dim]
        loss_dict = {
            # Cada dimensão vira uma chave escalar: loss_per_dim/0, /1, ...
            # O WandBLogger só aceita scalars — listas causam o WARNING que você viu.
            **{f"loss_per_dim/{i}": v.item() for i, v in enumerate(loss_per_dim)},
            # Resumo estatístico útil para monitorar evolução geral
            "loss_per_dim_mean": loss_per_dim.mean().item(),
            "loss_per_dim_max": loss_per_dim.max().item(),
            "loss_per_dim_min": loss_per_dim.min().item(),
        }

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    # ── Carregamento de pesos pré-treinados ───────────────────────────────────

    def get_optim_params(self) -> dict:
        return self.parameters()

    def init_rtc_processor(self):
        self.rtc_processor = None
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        print(
            "The PI05 model is a direct port of the OpenPI implementation.\n"
            "Original: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download, resume_download=resume_download,
                proxies=proxies, token=token, cache_dir=cache_dir,
                local_files_only=local_files_only, revision=revision, **kwargs,
            )

        model = cls(config, **kwargs)

        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file
                resolved_file = cached_file(
                    pretrained_name_or_path, "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    use_auth_token=kwargs.get("use_auth_token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file
                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict: {e}")
                return model

            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            remapped_state_dict = {}
            remap_count = 0
            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=strict)

            if missing_keys:
                # neutral_position é um buffer novo — não existirá em checkpoints antigos
                neutral_keys = [k for k in missing_keys if "neutral_position" in k]
                real_missing = [k for k in missing_keys if "neutral_position" not in k]
                if neutral_keys:
                    print(f"ℹ️  neutral_position não encontrado no checkpoint (normal em checkpoints antigos) — usando zeros.")
                if real_missing:
                    print(f"Missing keys: {len(real_missing)}")
                    for k in real_missing[:5]:
                        print(f"  - {k}")

            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
                for k in unexpected_keys[:5]:
                    print(f"  - {k}")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load weights: {e}")

        return model

    def _fix_pytorch_state_dict_keys(self, state_dict, model_config):
        import re
        fixed_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue
            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key: {key}")
                continue
            if "patch_embedding" in key:
                logging.warning(f"Vision embedding key might need handling: {key}")
            fixed_state_dict[new_key] = value
        return fixed_state_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {"target_modules": target_modules, "modules_to_save": []}