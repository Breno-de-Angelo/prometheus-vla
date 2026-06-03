#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""
ACT-D Policy — Improved
Melhorias:
  - Scene Uncertainty Gate: durante inferência, se o VAE estiver muito incerto
    (log_sigma alto), mistura a predição com a posição neutra, evitando
    movimentos explosivos quando o cenário muda.
  - Depth como token próprio no Encoder (não mais somado ao state_token).
    O Transformer pode fazer cross-attention entre a nuvem de pontos e a ação.
  - Pressão como token próprio no VAE encoder (melhora o prior latente).
  - Pressão somada ao state_token (propriocepção tátil, faz sentido semântico).
  - Temporal Ensembling mantido.
"""

import math
from collections import deque
from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from .configuration_act import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .depth_encoder import PointNetEncoder, depth_to_pointcloud


# ══════════════════════════════════════════════════════════════
# ACT POLICY (wrapper de alto nível)
# ══════════════════════════════════════════════════════════════
class ACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy — ACT-D Improved.

    Mudanças em relação à versão anterior:
      - Depth vira um token próprio no ACTEncoder (não mais adição ao state_token).
        Isso permite ao Transformer aprender atenção cruzada entre geometria 3D e ação.
      - Pressão tátil vai como token próprio para o VAE encoder, e como adição
        ao state_token para o encoder principal (é propriocepção, faz sentido semântico).
      - _inject_extra_features() foi substituído por passagem direta de tensors
        no batch com chaves bem definidas: '_act_d_depth_feat' e '_act_d_pressure_feat'.
      - scene_uncertainty_threshold: quando a incerteza VAE supera esse limiar,
        a ação predita é "mixada" em direção à posição neutra.

    Controlado 100% pelo YAML:
      use_depth_3d: true/false   → liga/desliga PointNet + token de depth
      use_pressure: true/false   → liga/desliga pressão tátil
    """

    config_class = ACTConfig
    name = "act"

    def __init__(self, config: ACTConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )

        # Buffer da posição neutra para o Scene Uncertainty Gate.
        # Preenchido pelo run_train.py via compute_neutral_position().
        action_dim = list(config.output_features.values())[0].shape[0]
        self._action_dim = action_dim
        self.register_buffer("neutral_position", torch.zeros(action_dim))

        # ── Módulos ACT-D (só instanciados se ativados no config) ────────────
        if config.use_depth_3d:
            self.pointnet = PointNetEncoder(output_dim=config.dim_model)
            self.camera_intrinsics = config.camera_intrinsics

        if config.use_pressure:
            self.pressure_proj = nn.Sequential(
                nn.Linear(config.pressure_feature_dim, config.pressure_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.pressure_hidden_dim, config.dim_model),
            )

        self.reset()
        self.last_attn_weights = None

    # ── Otimizador ────────────────────────────────────────────
    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """Deve ser chamado sempre que o ambiente for resetado."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._uncertainty_history = deque([], maxlen=10)

    # ──────────────────────────────────────────────────────────
    # SCENE UNCERTAINTY GATE
    # ──────────────────────────────────────────────────────────
    def _apply_uncertainty_gate(
        self,
        actions: Tensor,
        log_sigma: Tensor | None,
        threshold: float,
    ) -> Tensor:
        if log_sigma is None or threshold <= 0:
            return actions

        sigma = torch.exp(log_sigma / 2.0)
        uncertainty = sigma.mean(dim=-1, keepdim=True)
        self._uncertainty_history.append(uncertainty.mean().item())

        excess = (uncertainty - threshold) / (threshold + 1e-6)
        blend_alpha = excess.clamp(0.0, 1.0)

        neutral = (
            self.neutral_position
            .to(actions.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand_as(actions)
        )
        blend_alpha = blend_alpha.unsqueeze(-1)
        actions_safe = (1.0 - blend_alpha) * actions + blend_alpha * neutral

        if blend_alpha.max().item() > 0.05:
            import logging
            logging.debug(
                f"[UncertaintyGate] uncertainty={uncertainty.mean():.3f}, "
                f"blend_alpha={blend_alpha.mean():.3f} — aplicando retorno ao neutro"
            )

        return actions_safe

    # ──────────────────────────────────────────────────────────
    # EXTRAÇÃO DE FEATURES ACT-D
    # ──────────────────────────────────────────────────────────
    def _extract_depth_features(self, batch: dict[str, Tensor]) -> Tensor | None:
        """
        Remove o depth do batch e retorna o feature [B, dim_model] da PointNet.
        Retorna None se use_depth_3d=false ou se a chave não existir no batch.
        """
        if not self.config.use_depth_3d:
            return None
        depth_key = "observation.images.head_camera_depth"
        depth_tensor = batch.pop(depth_key, None)
        if depth_tensor is None:
            return None
        pc = depth_to_pointcloud(
            depth_tensor, self.camera_intrinsics, self.config.pointnet_num_points
        )
        return self.pointnet(pc)  # [B, dim_model]

    def _extract_pressure_features(self, batch: dict[str, Tensor]) -> Tensor | None:
        """
        Remove pressão do batch e retorna o feature [B, dim_model] do MLP.
        Retorna None se use_pressure=false ou se as chaves não existirem.
        """
        if not self.config.use_pressure:
            return None
        left = batch.pop("observation.left_hand_pressure", None)
        right = batch.pop("observation.right_hand_pressure", None)
        if left is None or right is None:
            return None
        full = torch.cat([left, right], dim=1)  # [B, 66]
        return self.pressure_proj(full)          # [B, dim_model]

    def _pack_act_d_features(
        self,
        batch: dict[str, Tensor],
        depth_feat: Tensor | None,
        pressure_feat: Tensor | None,
    ) -> dict[str, Tensor]:
        """
        Injeta os tensors de depth e pressão no batch com chaves dedicadas.
        O ACT.forward() os consome diretamente — sem somas nem fusões aqui.
        """
        if depth_feat is not None:
            batch["_act_d_depth_feat"] = depth_feat         # [B, dim_model]
        if pressure_feat is not None:
            batch["_act_d_pressure_feat"] = pressure_feat   # [B, dim_model]
        return batch

    # ──────────────────────────────────────────────────────────
    # INFERÊNCIA
    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        batch = dict(batch)  # shallow copy

        depth_feat    = self._extract_depth_features(batch)
        pressure_feat = self._extract_pressure_features(batch)

        # Filtra imagens RGB (exclui depth da lista de câmeras)
        if self.config.image_features:
            rgb_keys = [k for k in self.config.image_features if "depth" not in k]
            batch[OBS_IMAGES] = [batch[key] for key in rgb_keys]

        batch = self._pack_act_d_features(batch, depth_feat, pressure_feat)

        actions, (mu, log_sigma), last_attn = self.model(batch)
        self.last_attn_weights = last_attn

        threshold = getattr(self.config, "scene_uncertainty_threshold", 0.0)
        actions = self._apply_uncertainty_gate(actions, log_sigma, threshold)

        return actions

    # ──────────────────────────────────────────────────────────
    # TREINAMENTO
    # ──────────────────────────────────────────────────────────
    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        batch = dict(batch)  # shallow copy

        depth_feat    = self._extract_depth_features(batch)
        pressure_feat = self._extract_pressure_features(batch)

        if self.config.image_features:
            rgb_keys = [k for k in self.config.image_features if "depth" not in k]
            batch[OBS_IMAGES] = [batch[key] for key in rgb_keys]

        batch = self._pack_act_d_features(batch, depth_feat, pressure_feat)

        actions_hat, (mu_hat, log_sigma_x2_hat), _ = self.model(batch)

        l1_loss_unreduced = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            * ~batch["action_is_pad"].unsqueeze(-1)
        )

        if reduction == "none":
            l1_loss = l1_loss_unreduced.mean(dim=(-1, -2))
        else:
            l1_loss = l1_loss_unreduced.mean()

        loss_dict = {
            "l1_loss": l1_loss.mean().item() if reduction == "none" else l1_loss.item()
        }

        if self.config.use_vae and mu_hat is not None and log_sigma_x2_hat is not None:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()))
                .sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


# ══════════════════════════════════════════════════════════════
# ACT TEMPORAL ENSEMBLER (inalterado)
# ══════════════════════════════════════════════════════════════
class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)

        next_action = self.ensembled_actions[:, 0]
        self.ensembled_actions = self.ensembled_actions[:, 1:]
        self.ensembled_actions_count = self.ensembled_actions_count[1:]
        return next_action


# ══════════════════════════════════════════════════════════════
# ACT BACKBONE (ResNet)
# ══════════════════════════════════════════════════════════════
class ACTSinusoidalPositionEmbedding2D(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, scale: float = 2 * math.pi):
        super().__init__()
        assert num_pos_feats % 2 == 0, "num_pos_feats must be even"
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, h, w = x.shape
        not_mask = torch.ones((batch_size, h, w), dtype=torch.bool, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class ACTBackbone(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        backbone = getattr(torchvision.models, config.vision_backbone)(
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            weights=config.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.body = IntermediateLayerGetter(backbone, return_layers={"layer4": "0"})
        self.num_channels = 512 if config.vision_backbone in ("resnet18", "resnet34") else 2048
        self.position_embedding = ACTSinusoidalPositionEmbedding2D(num_pos_feats=config.dim_model // 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.body(x)["0"]
        pos = self.position_embedding(features)
        return features, pos


# ══════════════════════════════════════════════════════════════
# ACT ENCODER
# ══════════════════════════════════════════════════════════════
class ACTEncoder(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            ACTEncoderLayer(config) for _ in range(config.n_encoder_layers)
        ])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None):
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        return self.norm(x)


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout_layer = nn.Dropout(config.dropout)
        self.activation = _get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed=None, key_padding_mask=None):
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout_layer(x)
        if not self.pre_norm:
            x = self.norm1(x)

        skip = x
        if self.pre_norm:
            x = self.norm2(x)
        x = self.linear2(self.dropout_layer(self.activation(self.linear1(x))))
        x = skip + self.dropout_layer(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


# ══════════════════════════════════════════════════════════════
# ACT DECODER
# ══════════════════════════════════════════════════════════════
class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)
        ])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(self, x, encoder_out, pos_embed=None, query_pos_embed=None):
        last_attn = None
        for layer in self.layers:
            x = layer(x, encoder_out, pos_embed=pos_embed, query_pos_embed=query_pos_embed)
            last_attn = getattr(layer, "last_cross_attn_weights", None)
        if self.norm is not None:
            x = self.norm(x)
        return x.unsqueeze(0), last_attn


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout_layer = nn.Dropout(config.dropout)
        self.activation = _get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, encoder_out, pos_embed=None, query_pos_embed=None):
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if query_pos_embed is None else x + query_pos_embed
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout_layer(x)
        if not self.pre_norm:
            x = self.norm1(x)

        skip = x
        if self.pre_norm:
            x = self.norm2(x)
        q = x if query_pos_embed is None else x + query_pos_embed
        k = encoder_out if pos_embed is None else encoder_out + pos_embed
        x, self.last_cross_attn_weights = self.multihead_attn(
            q, k, value=encoder_out,
            need_weights=True, average_attn_weights=True,
        )
        x = skip + self.dropout_layer(x)
        if not self.pre_norm:
            x = self.norm2(x)

        skip = x
        if self.pre_norm:
            x = self.norm3(x)
        x = self.linear2(self.dropout_layer(self.activation(self.linear1(x))))
        x = skip + self.dropout_layer(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


# ══════════════════════════════════════════════════════════════
# ACT MAIN MODEL
# ══════════════════════════════════════════════════════════════
class ACT(nn.Module):
    """
    ACT model principal — ACT-D Improved.

    Tokens do Encoder (ordem de concatenação):
      1. Tokens RGB da câmera(s)         — [B, h*w * num_cams, dim_model]
      2. Token de depth (PointNet)        — [B, 1, dim_model]  ← NOVO (se use_depth_3d)
      3. Token de estado do robô          — [B, 1, dim_model]
         └─ pressão somada como residual ao state_token (se use_pressure)
      4. Token de estado do ambiente      — [B, 1, dim_model]  (se existir)
      5. Token latente VAE                — [B, 1, dim_model]  (se use_vae)

    VAE Encoder recebe:
      cls + state + (depth_token) + (pressure_token) + action_sequence
      Pressão e depth entram no VAE para que o prior latente aprenda a geometria
      e o tato, não só a trajetória das juntas.
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config

        # ── Backbone de visão RGB ─────────────────────────────
        if config.image_features:
            num_cameras = len([k for k in config.image_features if "depth" not in k])
            self.backbone = ACTBackbone(config)
            self.encoder_img_feat_input_proj = nn.Conv2d(
                self.backbone.num_channels, config.dim_model, kernel_size=1
            )
            self.encoder_cam_feat_pos_embed = nn.Embedding(num_cameras, config.dim_model)

        # ── Projeção do depth para o Encoder ─────────────────
        # Token próprio: o Transformer pode fazer self-attention
        # entre a nuvem de pontos e os tokens RGB / state / action.
        if config.use_depth_3d:
            self.encoder_depth_input_proj = nn.Linear(config.dim_model, config.dim_model)
            self.encoder_depth_pos_embed  = nn.Embedding(1, config.dim_model)

        # ── Estado do robô ────────────────────────────────────
        if config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0], config.dim_model
            )

        # ── Estado do ambiente ────────────────────────────────
        if config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                config.env_state_feature.shape[0], config.dim_model
            )

        # ── VAE ───────────────────────────────────────────────
        if config.use_vae:
            self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
            self.latent_4d_pos_embed = nn.Embedding(2, config.dim_model)

            self.vae_encoder = ACTEncoder(config)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)

            self.vae_encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0] if config.robot_state_feature else config.dim_model,
                config.dim_model,
            )
            self.vae_encoder_action_input_proj = nn.Linear(
                list(config.output_features.values())[0].shape[0], config.dim_model
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)

            # Depth e pressão no VAE (para o prior latente aprender geometria/tato)
            if config.use_depth_3d:
                self.vae_encoder_depth_input_proj = nn.Linear(config.dim_model, config.dim_model)
            if config.use_pressure:
                self.vae_encoder_pressure_input_proj = nn.Linear(config.dim_model, config.dim_model)

        # ── Encoder / Decoder / Head ──────────────────────────
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(
            config.dim_model, list(config.output_features.values())[0].shape[0]
        )

    def forward(self, batch: dict[str, Tensor]):
        # ── Consome as features ACT-D do batch ───────────────
        # Chaves colocadas por ACTPolicy._pack_act_d_features()
        depth_feat    = batch.pop("_act_d_depth_feat", None)     # [B, dim_model] ou None
        pressure_feat = batch.pop("_act_d_pressure_feat", None)  # [B, dim_model] ou None

        # ── Montagem dos tokens do Encoder ───────────────────
        encoder_in_tokens   = []
        encoder_in_pos_embed = []

        # 1. Tokens RGB
        if self.config.image_features:
            all_images = batch[OBS_IMAGES]  # lista de tensors [B, C, H, W]
            all_feats, all_pos = [], []
            for img in all_images:
                feat, pos = self.backbone(img)
                feat = self.encoder_img_feat_input_proj(feat)
                B, C, h, w = feat.shape
                feat = feat.flatten(2).permute(0, 2, 1)  # [B, h*w, dim_model]
                pos  = pos.flatten(2).permute(0, 2, 1)
                all_feats.append(feat)
                all_pos.append(pos)
            cam_features = torch.cat(all_feats, dim=1)
            cam_pos      = torch.cat(all_pos, dim=1)
            encoder_in_tokens.append(cam_features)
            encoder_in_pos_embed.append(cam_pos)

        # 2. Token de depth — token próprio no Encoder
        #    O Transformer pode fazer cross-attention entre geometria 3D e ação.
        if depth_feat is not None and self.config.use_depth_3d:
            depth_token = self.encoder_depth_input_proj(depth_feat)          # [B, dim_model]
            depth_token = depth_token.unsqueeze(1)                            # [B, 1, dim_model]
            B_d = depth_token.shape[0]
            depth_pos = (
                self.encoder_depth_pos_embed.weight                           # [1, dim_model]
                .unsqueeze(0)                                                 # [1, 1, dim_model]
                .expand(B_d, -1, -1)                                         # [B, 1, dim_model]
            )
            encoder_in_tokens.append(depth_token)
            encoder_in_pos_embed.append(depth_pos)

        # 3. Token de estado do robô
        #    Pressão é adicionada como residual (propriocepção tátil).
        if self.config.robot_state_feature and OBS_STATE in batch:
            robot_state = batch[OBS_STATE]
            state_token = self.encoder_robot_state_input_proj(robot_state)   # [B, dim_model]

            # Pressão somada ao state (não ao depth — erros semânticos diferentes)
            if pressure_feat is not None and self.config.use_pressure:
                state_token = state_token + pressure_feat

            encoder_in_tokens.append(state_token.unsqueeze(1))
            encoder_in_pos_embed.append(torch.zeros_like(state_token.unsqueeze(1)))

        # 4. Token de estado do ambiente (opcional)
        if self.config.env_state_feature and OBS_ENV_STATE in batch:
            env_state = batch[OBS_ENV_STATE]
            env_token = self.encoder_env_state_input_proj(env_state)
            encoder_in_tokens.append(env_token.unsqueeze(1))
            encoder_in_pos_embed.append(torch.zeros_like(env_token.unsqueeze(1)))

        # 5. Token latente VAE
        mu = log_sigma_x2 = None
        if self.config.use_vae:
            if ACTION in batch:
                # Modo treino: codifica ação (+ depth/pressão) no espaço latente
                actions = batch[ACTION]
                B = actions.shape[0]

                cls_embed = self.vae_encoder_cls_embed.weight.unsqueeze(0).expand(B, -1, -1)

                if self.config.robot_state_feature and OBS_STATE in batch:
                    robot_state_embed = self.vae_encoder_robot_state_input_proj(
                        batch[OBS_STATE]
                    ).unsqueeze(1)
                else:
                    robot_state_embed = torch.zeros(
                        B, 1, self.config.dim_model, device=actions.device
                    )

                action_embed = self.vae_encoder_action_input_proj(actions)  # [B, chunk, dim]

                # Sequência base do VAE
                vae_in_list = [cls_embed, robot_state_embed, action_embed]

                # Depth entra no VAE para que o prior latente aprenda geometria
                if depth_feat is not None and self.config.use_depth_3d:
                    d_tok = self.vae_encoder_depth_input_proj(depth_feat).unsqueeze(1)
                    vae_in_list.insert(1, d_tok)  # logo após cls

                # Pressão entra no VAE para que o prior aprenda tato
                if pressure_feat is not None and self.config.use_pressure:
                    p_tok = self.vae_encoder_pressure_input_proj(pressure_feat).unsqueeze(1)
                    vae_in_list.insert(1, p_tok)  # logo após cls (e depth se existir)

                vae_in  = torch.cat(vae_in_list, dim=1)
                vae_out = self.vae_encoder(vae_in)

                latent_params = self.vae_encoder_latent_output_proj(vae_out[:, 0])
                mu, log_sigma_x2 = latent_params.chunk(2, dim=-1)

                if self.training:
                    latent = mu + (log_sigma_x2 / 2).exp() * torch.randn_like(mu)
                else:
                    latent = mu
            else:
                # Modo inferência: latente zero (mu=0, sem ruído)
                B = next(iter(batch.values())).shape[0]
                latent = torch.zeros(
                    B, self.config.latent_dim,
                    device=next(iter(batch.values())).device
                )

            latent_token = self.encoder_latent_input_proj(latent)
            encoder_in_tokens.append(latent_token.unsqueeze(1))
            pos = (
                self.latent_4d_pos_embed.weight[0]
                .unsqueeze(0).unsqueeze(0)
                .expand(latent_token.shape[0], -1, -1)
            )
            encoder_in_pos_embed.append(pos)

        # ── Encoder ──────────────────────────────────────────
        encoder_in  = torch.cat(encoder_in_tokens, dim=1)
        encoder_pos = torch.cat(encoder_in_pos_embed, dim=1)
        encoder_out = self.encoder(encoder_in, pos_embed=encoder_pos)

        # ── Decoder ──────────────────────────────────────────
        B = encoder_out.shape[0]
        decoder_in = torch.zeros(
            B, self.config.chunk_size, self.config.dim_model,
            device=encoder_out.device
        )
        query_pos = self.decoder_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)
        decoder_out, last_attn = self.decoder(decoder_in, encoder_out, query_pos_embed=query_pos)

        actions = self.action_head(decoder_out[0])  # [B, chunk_size, action_dim]
        return actions, (mu, log_sigma_x2), last_attn


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def _get_activation_fn(activation: str) -> Callable:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"Activation '{activation}' não reconhecida")