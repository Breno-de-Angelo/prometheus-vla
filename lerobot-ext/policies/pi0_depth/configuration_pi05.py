#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05depth")
@dataclass
class PI05DEPTHConfig(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    empty_cameras: int = 0
    tokenizer_max_length: int = 200

    # ── ACT-D: Geometria 3D ──────────────────────────────────────────────────
    use_depth_3d: bool = True
    pointnet_num_points: int = 1024
    camera_intrinsics: dict = field(
        default_factory=lambda: {'fx': 600.0, 'fy': 600.0, 'cx': 320.0, 'cy': 240.0}
    )

    # ── ACT-D: Tato / Pressão ────────────────────────────────────────────────
    use_pressure: bool = True
    pressure_feature_dim: int = 66

    # ── Scene Uncertainty Gate ───────────────────────────────────────────────
    # O PI05 usa Flow Matching (não VAE), então não há log_sigma.
    # A incerteza é estimada rodando n_samples_uncertainty denoising passes com
    # ruídos iniciais diferentes e medindo a variância entre os resultados.
    # O prefix (VLM + SigLIP) é computado UMA vez com KV-cache; só o suffix
    # (Gemma expert) roda n vezes — custo razoável.
    #
    # scene_uncertainty_threshold: limiar do std médio das ações.
    #   0.0  → gate desligado (padrão — sem custo extra)
    #   0.05 → ativa com incerteza baixa (mais conservador)
    #   0.10 → boa partida para o G1
    #   0.20 → só em cenários muito incertos
    #
    # n_samples_uncertainty: quantas amostras usar para estimar incerteza.
    #   1  → sem estimativa (gate desligado mesmo que threshold > 0)
    #   3  → bom custo-benefício (ativado automaticamente se threshold > 0)
    #   5  → mais preciso, ~5× mais lento no suffix
    scene_uncertainty_threshold: float = 0.0
    n_samples_uncertainty: int = 1  # auto-ajustado para 3 se threshold > 0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")
        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        # ── Validação de consistência depth/pressure ──────────────────────────
        # (Mesma lógica do ACTConfig — detecta YAML inconsistente antes de treinar)
        has_depth = any(
            "depth" in k.lower() for k in self.input_features
        )
        if self.use_depth_3d and not has_depth:
            raise ValueError(
                "use_depth_3d=True mas nenhuma feature com 'depth' no nome está em "
                "input_features. Adicione a feature ou coloque use_depth_3d=False."
            )
        if not self.use_depth_3d and has_depth:
            import warnings
            warnings.warn(
                "use_depth_3d=False mas uma feature de depth está em input_features. "
                "A câmera será carregada mas ignorada — considere remover do YAML.",
                stacklevel=2,
            )

        has_pressure = (
            "observation.left_hand_pressure" in self.input_features
            or "observation.right_hand_pressure" in self.input_features
        )
        if self.use_pressure and not has_pressure:
            raise ValueError(
                "use_pressure=True mas as features de pressão não estão em input_features."
            )

        # Auto-ajusta n_samples se threshold foi setado mas amostras esquecidas
        if self.scene_uncertainty_threshold > 0 and self.n_samples_uncertainty <= 1:
            self.n_samples_uncertainty = 3

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            self.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None