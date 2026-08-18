#!/usr/bin/env python
"""
OpenVLA-Depth — Política.

Uma única rede que recebe **texto + RGB + profundidade + tato + propriocepção**
e devolve um chunk de ações. O texto vem do campo `task` do dataset LeRobot, o
que torna a mesma rede capaz de executar comandos diferentes:

    task = "pick up the cup"                      → ação de preensão
    task = "place the cup on the coffee stand"    → ação de colocação

sem trocar de checkpoint e sem cabeça de saída por tarefa.

## Layout da sequência

    [BOS] [patches RGB ×256] [depth] [pressão] [estado] [prompt ×L] [queries ×C]
     └──────────────────── prefixo (contexto) ──────────────────┘   └── saída ──┘

As `C = chunk_size` *action queries* são embeddings aprendidos. Após um único
forward do LLM, os hidden states nessas posições passam por um MLP e viram
`[B, chunk_size, action_dim]`. É a decodificação paralela do OpenVLA-OFT — o
OpenVLA original geraria 28 × 50 = 1400 tokens autoregressivos para o mesmo
resultado.

## Atenção

As action queries usam atenção **bidirecional entre si** (`bidirectional_action_attn`),
mas continuam causais em relação ao prefixo: a query do passo 3 pode olhar a do
passo 47. Isso é o que dá coerência temporal ao chunk inteiro num forward só.
O prefixo nunca vê as queries.
"""

from __future__ import annotations

import logging
from collections import deque
from types import SimpleNamespace

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from ..act_depth.depth_encoder import build_depth_encoder, depth_to_pointcloud
from .backbone import build_fused_pixel_values, build_openvla_backbone
from .configuration_openvla import OPENVLADEPTHConfig

# Estatísticas ImageNet — usadas para desfazer a normalização que o
# `lerobot/datasets/factory.py` aplica indiscriminadamente a TODA feature VISUAL,
# incluindo o mapa de profundidade. Mesmo tratamento do act_depth/pi0_depth.


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Zero-pad a última dimensão até `new_dim`."""
    if vector.shape[-1] == new_dim:
        return vector
    if vector.shape[-1] > new_dim:
        raise ValueError(f"Vetor com {vector.shape[-1]} dims não cabe em {new_dim}.")
    shape = list(vector.shape)
    shape[-1] = new_dim
    out = torch.zeros(shape, dtype=vector.dtype, device=vector.device)
    out[..., : vector.shape[-1]] = vector
    return out


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, dropout: float = 0.0) -> nn.Sequential:
    """MLP com LayerNorm na entrada — estabiliza o head sobre hidden states do LLM."""
    layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
    dim = in_dim
    for _ in range(max(0, n_layers - 1)):
        layers += [nn.Linear(dim, hidden_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


# ══════════════════════════════════════════════════════════════════════════════
# Núcleo
# ══════════════════════════════════════════════════════════════════════════════


class OpenVLADepthModel(nn.Module):
    """Backbone OpenVLA + tokens multimodais extras + head de ação OFT."""

    def __init__(self, config: OPENVLADEPTHConfig):
        super().__init__()
        self.config = config

        self.backbone = build_openvla_backbone(config)
        llm_dim = self.backbone.llm_dim

        # ── Token de geometria 3D ─────────────────────────────────────────────
        # Reaproveita o encoder do act_depth (PointNet ou Point Transformer).
        # `build_depth_encoder` lê `dim_model` do config — passamos um shim para
        # não poluir o OPENVLADEPTHConfig com um campo que só existe no ACT.
        if config.use_depth_3d:
            self.depth_encoder = build_depth_encoder(
                SimpleNamespace(
                    dim_model=llm_dim,
                    depth_encoder_type=config.depth_encoder_type,
                    point_transformer_k=config.point_transformer_k,
                    point_transformer_layers=config.point_transformer_layers,
                    point_transformer_dim=config.point_transformer_dim,
                    depth_pretrained_weights=config.depth_pretrained_weights,
                    depth_pretrained_cache_dir=config.depth_pretrained_cache_dir,
                    depth_pretrained_prefix_remap=None,
                )
            )

        # ── Token tátil ───────────────────────────────────────────────────────
        if config.use_pressure:
            self.pressure_proj = nn.Sequential(
                nn.Linear(config.pressure_feature_dim, config.pressure_hidden_dim),
                nn.GELU(),
                nn.Linear(config.pressure_hidden_dim, llm_dim),
            )

        # ── Token de propriocepção ────────────────────────────────────────────
        if config.state_as_token:
            self.state_proj = nn.Sequential(
                nn.Linear(config.max_state_dim, config.pressure_hidden_dim),
                nn.GELU(),
                nn.Linear(config.pressure_hidden_dim, llm_dim),
            )

        # ── Action queries + head (OFT) ───────────────────────────────────────
        self.action_queries = nn.Parameter(torch.randn(config.chunk_size, llm_dim) * 0.02)
        self.action_head = _mlp(
            llm_dim,
            config.action_head_hidden_dim,
            config.max_action_dim,
            config.action_head_n_layers,
            dropout=config.action_head_dropout,
        )

        self.gradient_checkpointing_enabled = False

    # ── Congelamento / LoRA ───────────────────────────────────────────────────

    def apply_freezing_and_lora(self) -> None:
        cfg = self.config

        if cfg.freeze_vision_encoder:
            for p in self.backbone.vision.parameters():
                p.requires_grad = False
            logging.info("[OpenVLA-D] Torre visual congelada.")

        if cfg.train_new_modules_only:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logging.info(
                "[OpenVLA-D] train_new_modules_only=True — backbone inteiro congelado. "
                "Só pointnet/pressão/estado/queries/head treinam."
            )
            return

        if cfg.use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as e:
                raise ImportError(
                    "use_lora=True precisa do `peft`. Instale com: pip install 'peft>=0.13.0'"
                ) from e

            lora_cfg = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.backbone.language_model = get_peft_model(self.backbone.language_model, lora_cfg)
            trainable = sum(
                p.numel() for p in self.backbone.language_model.parameters() if p.requires_grad
            )
            total = sum(p.numel() for p in self.backbone.language_model.parameters())
            logging.info(
                f"[OpenVLA-D] LoRA r={cfg.lora_r} aplicado ao LLM — "
                f"{trainable / 1e6:.1f}M treináveis de {total / 1e6:.0f}M ({100 * trainable / total:.2f}%)."
            )

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True
        lm = self.backbone.language_model
        target = getattr(lm, "base_model", lm)  # desembrulha o PeftModel
        if hasattr(target, "gradient_checkpointing_enable"):
            target.gradient_checkpointing_enable()
        else:
            target.gradient_checkpointing = True
        logging.info("[OpenVLA-D] Gradient checkpointing ligado no LLM.")

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing_enabled = False
        lm = self.backbone.language_model
        target = getattr(lm, "base_model", lm)
        if hasattr(target, "gradient_checkpointing_disable"):
            target.gradient_checkpointing_disable()
        else:
            target.gradient_checkpointing = False

    # ── Montagem da sequência multimodal ──────────────────────────────────────

    def _extra_tokens(
        self,
        depth_images: list[Tensor] | None,
        pressure: Tensor | None,
        state: Tensor | None,
        dtype: torch.dtype,
    ) -> tuple[list[Tensor], list[str]]:
        """Tokens que entram entre os patches visuais e o prompt."""
        tokens: list[Tensor] = []
        names: list[str] = []

        if self.config.use_depth_3d and depth_images:
            pc = depth_to_pointcloud(
                depth_images[0], self.config.camera_intrinsics, self.config.pointnet_num_points
            )
            tokens.append(self.depth_encoder(pc).unsqueeze(1).to(dtype))
            names.append("depth")

        if self.config.use_pressure and pressure is not None:
            tokens.append(self.pressure_proj(pressure).unsqueeze(1).to(dtype))
            names.append("pressure")

        if self.config.state_as_token and state is not None:
            tokens.append(self.state_proj(state).unsqueeze(1).to(dtype))
            names.append("state")

        return tokens, names

    def _build_attention_mask_4d(
        self, pad_mask: Tensor, num_queries: int, dtype: torch.dtype
    ) -> Tensor:
        """
        Máscara causal com um bloco bidirecional no final (as action queries).

        `pad_mask` é [B, L] booleano indicando tokens reais. O transformers 5.x
        aceita uma máscara 4D já pronta e a repassa intacta ao attention
        (masking_utils.create_causal_mask: "if the mask is already 4D, return as-is").

        Convenção: 0.0 = pode atender, min(dtype) = bloqueado.
        """
        bsize, total_len = pad_mask.shape
        device = pad_mask.device

        allowed = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool, device=device))
        if self.config.bidirectional_action_attn and num_queries > 0:
            # Bloco quadrado final totalmente conectado: query i vê query j para todo j.
            allowed[-num_queries:, -num_queries:] = True

        allowed = allowed[None, None, :, :].expand(bsize, 1, total_len, total_len)
        # Um token padded nunca pode ser atendido (dimensão de chave).
        allowed = allowed & pad_mask[:, None, None, :]

        return torch.where(
            allowed, torch.zeros((), dtype=dtype, device=device), torch.finfo(dtype).min
        )

    def forward(
        self,
        rgb_images: list[Tensor],
        input_ids: Tensor,
        lang_mask: Tensor,
        depth_images: list[Tensor] | None = None,
        pressure: Tensor | None = None,
        state: Tensor | None = None,
    ) -> Tensor:
        """Retorna as ações normalizadas `[B, chunk_size, max_action_dim]`."""
        bsize = input_ids.shape[0]
        device = input_ids.device

        tok_emb = self.backbone.embed_tokens(input_ids)
        dtype = tok_emb.dtype

        def ones_mask(n: int) -> Tensor:
            return torch.ones(bsize, n, dtype=torch.bool, device=device)

        lang_mask = lang_mask.to(torch.bool)

        # O BOS do Llama tem que continuar na posição 0 — os patches entram logo depois.
        parts: list[Tensor] = [tok_emb[:, :1]]
        masks: list[Tensor] = [lang_mask[:, :1]]

        n_img_tokens = []
        for pixel_values in rgb_images:
            # `dtype` vem do embedding de tokens, ou seja, é o dtype do backbone
            # (bfloat16 em treino). As imagens chegam em float32 do preprocessor,
            # e a torre visual é bf16 — sem este cast o timm reclama de
            # "Input type (float) and bias type (c10::BFloat16) should be the same"
            # já na primeira Conv2d do patch_embed.
            img_emb = self.backbone.embed_images(pixel_values.to(dtype)).to(dtype)
            parts.append(img_emb)
            masks.append(ones_mask(img_emb.shape[1]))
            n_img_tokens.append(img_emb.shape[1])

        extra_tokens, extra_names = self._extra_tokens(depth_images, pressure, state, dtype)
        for tok in extra_tokens:
            parts.append(tok)
            masks.append(ones_mask(1))

        parts.append(tok_emb[:, 1:])
        masks.append(lang_mask[:, 1:])

        num_queries = self.config.chunk_size
        parts.append(self.action_queries[None].expand(bsize, -1, -1).to(dtype))
        masks.append(ones_mask(num_queries))

        embs = torch.cat(parts, dim=1)
        pad_mask = torch.cat(masks, dim=1)

        # Posições contíguas pulando o padding do prompt (mesma lógica do pi05depth).
        position_ids = (torch.cumsum(pad_mask.long(), dim=1) - 1).clamp(min=0)
        attn_4d = self._build_attention_mask_4d(pad_mask, num_queries, embs.dtype)

        outputs = self.backbone.language_model(
            inputs_embeds=embs,
            attention_mask=attn_4d,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state

        self._last_token_geometry = {
            "n_img_tokens": n_img_tokens,
            "extra_tokens": extra_names,
            "n_lang_tokens": int(lang_mask.shape[1]),
            "num_queries": num_queries,
            "total_len": int(embs.shape[1]),
        }

        action_hidden = hidden[:, -num_queries:].to(torch.float32)
        return self.action_head(action_hidden)


# ══════════════════════════════════════════════════════════════════════════════
# Política
# ══════════════════════════════════════════════════════════════════════════════


class OPENVLADEPTHPolicy(PreTrainedPolicy):
    """OpenVLA condicionado por linguagem, com depth 3D e tato, para o G1 + Dex3."""

    config_class = OPENVLADEPTHConfig
    name = "openvladepth"

    def __init__(self, config: OPENVLADEPTHConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = OpenVLADepthModel(config)
        self.model.apply_freezing_and_lora()

        if config.dtype != "float32":
            # Backbone em baixa precisão; o head fica em fp32 (a regressão de
            # ação é sensível e custa quase nada em memória).
            target_dtype = getattr(torch, config.dtype)
            self.model.backbone.to(target_dtype)
            self.model.action_head.to(torch.float32)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if config.device:
            self.model.to(config.device)

        # Posição neutra usada pelo uncertainty gate. Injetada pelo run_train
        # (policies/act_depth/neutral_position.py) e salva junto do checkpoint.
        original_action_dim = list(config.output_features.values())[0].shape[0]
        self._original_action_dim = original_action_dim
        self.register_buffer("neutral_position", torch.zeros(original_action_dim))

        self._last_uncertainty = 0.0
        self.reset()

    # ── Preparo de entradas ───────────────────────────────────────────────────

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """
        Separa RGB (→ torre DINOv2+SigLIP) de profundidade (→ encoder de nuvem de pontos).

        RGB sai como `pixel_values` de 6 canais em 224×224; depth sai cru em [0, 1].
        """
        device = next(self.parameters()).device
        rgb_pixel_values: list[Tensor] = []
        depth_images: list[Tensor] = []

        for key in self.config.image_features:
            if key not in batch:
                continue
            img = batch[key].to(device=device, dtype=torch.float32)

            if "depth" in key.lower():
                # Crua, em MILÍMETROS: a 0.6.1 pula as câmeras de profundidade
                # ao carimbar stats do ImageNet (`datasets/factory.py`), então
                # não há normalização a desfazer — a reversão que existia aqui
                # virou dead code e foi removida.
                depth_images.append(img)
                continue

            if img.shape[1] != 3:  # channels-last → channels-first
                img = img.permute(0, 3, 1, 2)
            if img.shape[-2:] != tuple(self.config.image_resolution):
                img = F.interpolate(
                    img,
                    size=tuple(self.config.image_resolution),
                    mode="bicubic",
                    align_corners=False,
                ).clamp(0.0, 1.0)

            rgb_pixel_values.append(build_fused_pixel_values(img))

        if not rgb_pixel_values:
            raise ValueError(
                f"Nenhuma câmera RGB no batch. Esperava alguma de {self.config.rgb_keys}."
            )
        return rgb_pixel_values, depth_images

    def _extract_pressure(self, batch: dict[str, Tensor]) -> Tensor | None:
        if not self.config.use_pressure:
            return None
        left = batch.get("observation.left_hand_pressure")
        right = batch.get("observation.right_hand_pressure")
        if left is None or right is None:
            return None
        device = next(self.parameters()).device
        return torch.cat([left, right], dim=-1).to(device=device, dtype=torch.float32)

    def _extract_state(self, batch: dict[str, Tensor]) -> Tensor | None:
        if not self.config.state_as_token:
            return None
        state = batch.get(OBS_STATE)
        if state is None:
            return None
        device = next(self.parameters()).device
        return pad_vector(state.to(device=device, dtype=torch.float32), self.config.max_state_dim)

    def _model_inputs(self, batch: dict[str, Tensor]) -> dict:
        rgb, depth = self._preprocess_images(batch)
        return {
            "rgb_images": rgb,
            "input_ids": batch[OBS_LANGUAGE_TOKENS],
            "lang_mask": batch[OBS_LANGUAGE_ATTENTION_MASK],
            "depth_images": depth,
            "pressure": self._extract_pressure(batch),
            "state": self._extract_state(batch),
        }

    # ── Treinamento ───────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        actions_gt = pad_vector(batch[ACTION], self.config.max_action_dim)
        actions_pred = self.model(**self._model_inputs(batch))

        if self.config.loss_type == "l1":
            losses = F.l1_loss(actions_pred, actions_gt, reduction="none")
        elif self.config.loss_type == "smooth_l1":
            losses = F.smooth_l1_loss(actions_pred, actions_gt, reduction="none")
        else:
            losses = F.mse_loss(actions_pred, actions_gt, reduction="none")

        # Só as dimensões reais entram no loss — o padding até max_action_dim é zero
        # em ambos os lados e só diluiria a métrica.
        losses = losses[..., : self._original_action_dim]

        loss = losses.mean() if reduction == "mean" else losses

        loss_per_dim = losses.mean(dim=[0, 1]).detach().cpu()
        loss_dict = {f"loss_per_dim/{i}": v.item() for i, v in enumerate(loss_per_dim)}
        loss_dict["l1_loss"] = losses.mean().item()

        return loss, loss_dict

    def get_optim_params(self) -> list[dict]:
        """
        Dois grupos: o LLM (LoRA, LR baixo) e os módulos novos (LR maior).

        Os módulos novos — encoder de profundidade, projeções de tato/estado,
        action queries e head — começam do zero e precisam andar mais rápido que
        adaptadores sobre um LLM já treinado.
        """
        new_module_names = (
            "depth_encoder",
            "pressure_proj",
            "state_proj",
            "action_queries",
            "action_head",
        )
        backbone_params, new_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(f".{m}" in f".{name}" for m in new_module_names):
                new_params.append(param)
            else:
                backbone_params.append(param)

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.config.optimizer_lr})
        if new_params:
            groups.append({"params": new_params, "lr": self.config.optimizer_lr_new_modules})
        return groups

    # ── Inferência ────────────────────────────────────────────────────────────

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._last_uncertainty = 0.0

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        inputs = self._model_inputs(batch)

        threshold = self.config.scene_uncertainty_threshold
        n_samples = self.config.n_samples_uncertainty if threshold > 0 else 1

        if n_samples > 1:
            actions, uncertainty = self._predict_with_uncertainty(inputs, n_samples)
        else:
            actions, uncertainty = self.model(**inputs), 0.0

        self._last_uncertainty = uncertainty
        self._last_token_geometry = getattr(self.model, "_last_token_geometry", None)

        if threshold > 0:
            actions = self._apply_uncertainty_gate(actions, uncertainty, threshold)

        return actions[:, :, : self._original_action_dim]

    def _predict_with_uncertainty(self, inputs: dict, n_samples: int) -> tuple[Tensor, float]:
        """
        Incerteza por MC-dropout no head.

        O LLM roda **uma vez** — é ele que domina o custo. Só o MLP do head roda
        `n_samples` vezes, com dropout ativo, sobre o mesmo hidden state. A
        dispersão entre as amostras é o proxy de incerteza.

        Exige `action_head_dropout > 0`, que o config liga automaticamente quando
        `scene_uncertainty_threshold > 0`.
        """
        if self.config.action_head_dropout <= 0:
            logging.warning(
                "[OpenVLA-D] Uncertainty gate pedido mas action_head_dropout=0 — "
                "as amostras seriam idênticas. Gate ignorado."
            )
            return self.model(**inputs), 0.0

        # Monta a sequência e roda o LLM uma vez, replicando forward() até o head.
        was_training = self.model.action_head.training
        self.model.action_head.train()  # reativa só o dropout do head
        try:
            hidden = self._forward_until_head(inputs)
            samples = torch.stack([self.model.action_head(hidden) for _ in range(n_samples)], dim=0)
        finally:
            self.model.action_head.train(was_training)

        return samples.mean(dim=0), samples.std(dim=0).mean().item()

    def _forward_until_head(self, inputs: dict) -> Tensor:
        """Hidden states nas posições das action queries, sem aplicar o head."""
        model = self.model
        head, model.action_head = model.action_head, nn.Identity()
        try:
            return model(**inputs)
        finally:
            model.action_head = head

    def _apply_uncertainty_gate(self, actions: Tensor, uncertainty: float, threshold: float) -> Tensor:
        """
        Mistura a predição com a posição neutra proporcionalmente ao excesso de
        incerteza. Mesma fórmula do act_depth/pi0_depth, para as ablations
        continuarem comparáveis.
        """
        if threshold <= 0 or uncertainty <= 0:
            return actions

        blend_alpha = max(0.0, min(1.0, (uncertainty - threshold) / (threshold + 1e-6)))
        if blend_alpha < 0.01:
            return actions

        neutral = pad_vector(
            self.neutral_position.to(actions.device), self.config.max_action_dim
        ).view(1, 1, -1).expand_as(actions)

        logging.debug(
            f"[UncertaintyGate/OpenVLA-D] uncertainty={uncertainty:.4f} > "
            f"threshold={threshold:.4f} → blend_alpha={blend_alpha:.3f}"
        )
        return (1.0 - blend_alpha) * actions + blend_alpha * neutral
