#!/usr/bin/env python
"""
OpenVLA-Depth — Configuração.

Política VLA multi-tarefa condicionada por linguagem, com fusão de profundidade,
para o Unitree G1 + Dex3 (28 DoF).

Diferenças estruturais em relação ao `pi05depth` / `actdepth`:

  ┌──────────────┬─────────────────┬──────────────────┬─────────────────────┐
  │              │ actdepth        │ pi05depth        │ openvladepth        │
  ├──────────────┼─────────────────┼──────────────────┼─────────────────────┤
  │ Backbone     │ ResNet18        │ PaliGemma 3B     │ Prismatic/OpenVLA 7B│
  │ Visão        │ RGB             │ SigLIP           │ DINOv2 + SigLIP     │
  │ Linguagem    │ NENHUMA         │ prompt PaliGemma │ prompt Llama-2      │
  │ Decodificação│ transformer dec │ flow matching    │ OFT paralelo (L1)   │
  │ Depth        │ token PointNet  │ token PointNet   │ token PointNet      │
  │ Multi-tarefa │ ✗               │ ✓ (via `task`)   │ ✓ (via `task`)      │
  └──────────────┴─────────────────┴──────────────────┴─────────────────────┘

O ponto central: **uma única rede** recebe o texto da tarefa junto com RGB, depth
e propriocepção, e produz o chunk de ações correspondente àquele comando. Trocar
"pick up the cup" por "place the cup on the coffee stand" muda o comportamento
sem trocar de checkpoint.

Decodificação OFT (Optimized Fine-Tuning): em vez de gerar tokens de ação
autoregressivamente (o OpenVLA original faz 1 token por dimensão, o que daria
28 × 50 = 1400 passos de decodificação por chunk — inviável no robô real), a
política anexa `chunk_size` *action queries* aprendidas ao final da sequência e
lê os hidden states dessas posições num único forward. Custo: 1 forward.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224

# Prompt canônico do OpenVLA. `{task}` é substituído pelo campo `task` do
# dataset LeRobot — é ele que torna a política multi-tarefa.
OPENVLA_PROMPT_TEMPLATE = "In: What action should the robot take to {task}?\nOut:"


@PreTrainedConfig.register_subclass("openvladepth")
@dataclass
class OPENVLADEPTHConfig(PreTrainedConfig):
    # ── Backbone ──────────────────────────────────────────────────────────────
    # Checkpoint base no Hub. `openvla/openvla-7b` é o modelo pré-treinado em
    # 970k episódios do Open X-Embodiment.
    pretrained_backbone: str = "openvla/openvla-7b"

    # Como carregar o backbone:
    #   "native"      → reconstrói Prismatic com timm + LlamaForCausalLM e carrega
    #                   os pesos do safetensors. Funciona com transformers >= 5.x
    #                   (é o que está instalado aqui). RECOMENDADO.
    #   "remote_code" → AutoModelForVision2Seq(trust_remote_code=True). Só funciona
    #                   num venv com transformers==4.40.1 / timm==0.9.10, que
    #                   conflita com o resto do LeRobot. Ver docs/OPENVLA_DEPTH.md.
    load_mode: str = "native"

    dtype: str = "bfloat16"

    # Torres visuais do Prismatic. As dimensões (visão 2176 → LLM 4096 no
    # openvla-7b) NÃO são declaradas aqui: `backbone.py` as infere das formas do
    # próprio checkpoint, então trocar de backbone não exige mexer no YAML.
    dinov2_model: str = "vit_large_patch14_reg4_dinov2.lvd142m"
    siglip_model: str = "vit_so400m_patch14_siglip_224"

    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    empty_cameras: int = 0
    tokenizer_max_length: int = 64

    # Por padrão usa o tokenizer do próprio checkpoint (LlamaTokenizer com o
    # `<PAD>` extra, vocab 32001). O repo do openvla-7b carrega código customizado,
    # então o transformers imprime um prompt de `trust_remote_code` ao resolver o
    # tokenizer — barulhento, mas inofensivo: o fallback devolve o tokenizer certo.
    # Aponte para outro repo aqui se quiser evitar o prompt.
    tokenizer_name: str | None = None

    # ── Horizonte de ação ─────────────────────────────────────────────────────
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    # ── Head de ação (OFT) ────────────────────────────────────────────────────
    action_head_hidden_dim: int = 1024
    action_head_n_layers: int = 3
    # Atenção bidirecional entre as action queries (comportamento do OFT). Com
    # False, as queries usam atenção causal entre si — mais próximo do decoder
    # original do Llama, geralmente um pouco pior.
    bidirectional_action_attn: bool = True
    # "l1" é o do paper OFT; "smooth_l1" é mais tolerante a outliers nos dedos.
    loss_type: str = "l1"

    # ── Tokens extras injetados no prefix ─────────────────────────────────────
    # Propriocepção como token contínuo (OFT) em vez de texto discretizado
    # (o que o pi05depth faz). Evita gastar ~60 tokens de prompt com números.
    state_as_token: bool = True

    # ── Geometria 3D (depth) ──────────────────────────────────────────────────
    use_depth_3d: bool = True
    depth_key: str = "observation.images.head_camera_depth"
    # Reaproveita policies/act_depth/depth_encoder.py:
    #   "pointnet"          → PointNetEncoder (leve)
    #   "point_transformer" → PointTransformerEncoder (mais robusto)
    depth_encoder_type: str = "point_transformer"
    pointnet_num_points: int = 1024
    point_transformer_k: int = 16
    point_transformer_layers: int = 3
    point_transformer_dim: int = 256
    depth_pretrained_weights: str | None = None
    depth_pretrained_cache_dir: str | None = None
    camera_intrinsics: dict = field(
        default_factory=lambda: {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}
    )

    # ── Tato / Pressão (Dex3) ─────────────────────────────────────────────────
    use_pressure: bool = True
    pressure_feature_dim: int = 66      # 33 sensores por mão
    pressure_hidden_dim: int = 256

    # ── Linguagem / multi-tarefa ──────────────────────────────────────────────
    prompt_template: str = OPENVLA_PROMPT_TEMPLATE
    # Força um texto fixo, ignorando o `task` do dataset. Use SOMENTE para
    # depurar uma tarefa isolada — deixar preenchido anula o multi-tarefa, que
    # é justamente o motivo de existir desta política.
    override_task: str | None = None
    # Aborta o treino se o dataset tiver uma única string de `task` distinta.
    # Um VLA multi-tarefa treinado com um só comando aprende a ignorar o texto.
    require_multitask: bool = False

    # ── LoRA / congelamento ───────────────────────────────────────────────────
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    # "all-linear" cobre q/k/v/o + gate/up/down do Llama (setup do paper OpenVLA).
    lora_target_modules: str = "all-linear"
    freeze_vision_encoder: bool = True
    # Treina só os módulos novos (pointnet, pressão, state, queries, head) e
    # mantém o LLM 100% congelado. Útil para um smoke test barato.
    train_new_modules_only: bool = False

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    gradient_checkpointing: bool = True
    device: str | None = None

    # ── Otimização ────────────────────────────────────────────────────────────
    # 5e-4 é o LR de LoRA do paper OpenVLA. Os módulos novos (treinados do zero)
    # aprendem melhor com LR maior — daí o grupo separado.
    optimizer_lr: float = 5e-4
    optimizer_lr_new_modules: float = 1e-3
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 5e-5

    # ── Scene Uncertainty Gate ────────────────────────────────────────────────
    # Ao contrário do pi05depth (que estima incerteza rodando N denoisings com
    # ruídos diferentes), o head OFT é determinístico. A incerteza aqui vem de
    # MC-dropout no head: N forwards do head com dropout ativo sobre o MESMO
    # hidden state do LLM — o custo extra é desprezível (só o MLP roda N vezes).
    #   0.0  → gate desligado (padrão)
    #   0.10 → bom ponto de partida no G1
    scene_uncertainty_threshold: float = 0.0
    n_samples_uncertainty: int = 1      # auto-ajustado para 3 se threshold > 0
    action_head_dropout: float = 0.0    # auto-ajustado para 0.1 se threshold > 0

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) não pode ser maior que "
                f"chunk_size ({self.chunk_size})"
            )
        if self.dtype not in ["bfloat16", "float32", "float16"]:
            raise ValueError(f"dtype inválido: {self.dtype}")
        if self.load_mode not in ["native", "remote_code"]:
            raise ValueError(f"load_mode inválido: {self.load_mode} (use 'native' ou 'remote_code')")
        if self.loss_type not in ["l1", "smooth_l1", "mse"]:
            raise ValueError(f"loss_type inválido: {self.loss_type}")
        if self.depth_encoder_type not in ["pointnet", "point_transformer"]:
            raise ValueError(
                f"depth_encoder_type deve ser 'pointnet' ou 'point_transformer', "
                f"recebeu '{self.depth_encoder_type}'"
            )

        # NOTA: a consistência entre use_depth_3d/use_pressure e as features do
        # dataset é checada em `validate_features()`, não aqui. Quando o draccus
        # constrói este config a partir do YAML, `input_features` ainda está
        # VAZIO — quem preenche é o `make_policy`, a partir dos metadados do
        # dataset, logo antes de instanciar a política. Validar no __post_init__
        # rejeitaria qualquer YAML com use_depth_3d=true.

        if self.override_task is not None:
            import warnings

            warnings.warn(
                f"override_task='{self.override_task}' está setado: o campo `task` do "
                "dataset será IGNORADO e todos os episódios verão o mesmo prompt. "
                "Isso desliga o condicionamento por linguagem — use apenas para debug.",
                stacklevel=2,
            )

        if self.scene_uncertainty_threshold > 0:
            if self.n_samples_uncertainty <= 1:
                self.n_samples_uncertainty = 3
            if self.action_head_dropout <= 0:
                self.action_head_dropout = 0.1

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

        self._validate_multimodal_features()

    def _validate_multimodal_features(self) -> None:
        """
        Confere que o YAML e o dataset concordam sobre depth e tato.

        Roda a partir de `validate_features()` — chamado pela política, depois de
        o `make_policy` ter preenchido `input_features` com os metadados do
        dataset. No `__post_init__` isso não funcionaria: naquele momento o
        draccus acabou de ler o YAML e `input_features` ainda está vazio.

        O objetivo é pegar YAML inconsistente na primeira dezena de segundos, em
        vez de depois de carregar 15 GB de pesos.
        """
        import warnings

        has_depth = any("depth" in k.lower() for k in self.input_features)
        if self.use_depth_3d and not has_depth:
            raise ValueError(
                "use_depth_3d=True mas nenhuma feature com 'depth' no nome está em "
                f"input_features. Presentes: {sorted(self.input_features)}. "
                "Adicione a câmera de profundidade ao dataset ou use use_depth_3d=False."
            )
        if not self.use_depth_3d and has_depth:
            warnings.warn(
                "use_depth_3d=False mas há uma feature de depth em input_features. "
                "A câmera será carregada e ignorada — considere removê-la do YAML.",
                stacklevel=2,
            )

        has_pressure = (
            "observation.left_hand_pressure" in self.input_features
            or "observation.right_hand_pressure" in self.input_features
        )
        if self.use_pressure and not has_pressure:
            raise ValueError(
                "use_pressure=True mas as features de pressão não estão em input_features. "
                f"Presentes: {sorted(self.input_features)}."
            )

    # ── Chaves RGB (tudo que é VISUAL e não é depth) ──────────────────────────
    @property
    def rgb_keys(self) -> list[str]:
        return [k for k in self.image_features if "depth" not in k.lower()]

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
