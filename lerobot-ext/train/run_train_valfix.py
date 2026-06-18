#!/usr/bin/env python

import dataclasses
import logging
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any
import sys
import os

import torch
import torch.nn.functional as F  # noqa: N812
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

# Add current directory to path to ensure we can import train.utils
sys.path.append(os.getcwd())
# Add this script's directory so we can import sibling modules (pi05_depth_injector, etc.)
# without relative-import errors when executed as a standalone script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add lerobot-ext/ (parent of this script's dir) BEFORE cwd so that
# `from train.depth_encoder import ...` resolves to lerobot-ext/train/depth_encoder.py
# (the right file with depth_scale support) instead of prometheus-vla/train/ (older, no depth_encoder.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- MONKEY PATCHES START ---
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# guarda o original
_original_getitem = LeRobotDataset.__getitem__

def patched_getitem(self, idx):
    # garante que o dataset está carregado (opcional, mas seguro)
    self._ensure_hf_dataset_loaded()

    # --- SEU PATCH: mapear índice global → relativo ---
    if getattr(self, "_absolute_to_relative_idx", None) is not None:
        if idx in self._absolute_to_relative_idx:
            idx = self._absolute_to_relative_idx[idx]
    # --------------------------------------------------

    # delega TODO o resto para o método original
    return _original_getitem(self, idx)

# aplica o patch
LeRobotDataset.__getitem__ = patched_getitem

# Image-transform patch: aplica image_transforms APENAS em RGB, nunca em depth.
# Color jitter / affine em mapas de profundidade destrói a geometria, então
# removemos os transforms antes de chamar o original e re-aplicamos só nos RGB.
def patched_getitem_rgb_only_transforms(self, idx):
    orig_transforms = self.image_transforms
    self.image_transforms = None
    try:
        item = patched_getitem(self, idx)
    finally:
        self.image_transforms = orig_transforms

    if orig_transforms is not None:
        for key in list(item.keys()):
            if key.startswith("observation.images.") and "depth" not in key and torch.is_tensor(item[key]):
                item[key] = orig_transforms(item[key])

    return item

LeRobotDataset.__getitem__ = patched_getitem_rgb_only_transforms

# --- MONKEY PATCHES END ---


from lerobot.configs.train import TrainPipelineConfig, DatasetConfig
@dataclasses.dataclass
class CustomTrainPipelineConfig(TrainPipelineConfig):
    val_dataset: DatasetConfig | None = None
    depth_fusion: bool = True
    # Fusion mode when depth_fusion=True and policy.type="pi05":
    #   "full"       — PointNet (depth) + pressure_proj (tactile)  [pi05-D, default]
    #   "depth_only" — PointNet only                               [pi05-depth ablation]
    # Ignored when depth_fusion=False or policy is not pi05.
    fusion_mode: str = "full"
    # Feature key in the LeRobot batch that carries the depth map. Default
    # matches our cup3 setup; override per dataset:
    #   CALVIN          -> "observation.depths.static"
    #   LIBERO+depth    -> "observation.images.image_depth"
    depth_key: str = "observation.images.head_camera_depth"
    # Multiplier applied to depth values to obtain meters — OBRIGATÓRIO quando
    # depth_fusion=true (auditoria FASE 3: o default antigo de 2.0, hack do cup3,
    # treinaria com escala 2000x errada em silêncio se esquecido no YAML).
    # PNG16 em milímetros → 0.001; CALVIN/LIBERO+depth já em metros → 1.0.
    depth_scale: float | None = None
    # Intrínsecos {fx, fy, cx, cy} do stream de DEPTH na resolução gravada —
    # OBRIGATÓRIO quando depth_fusion=true. Ler do sensor com
    # lerobot-ext/tools/dump_realsense_intrinsics.py (o default antigo era
    # nominal de 640x480 e distorcia a nuvem do stream 848x480 em silêncio).
    depth_intrinsics: dict[str, float] | None = None
    # Crop de workspace no frame da câmera, em metros (auditoria FASE 4) —
    # {"z": [min,max], "x": [min,max], "y": [min,max]}; eixos faltantes não são
    # cropados; None = sem crop. Sem ele, fundo até ~32 m entra na nuvem e domina
    # o sampling. Valores ficam no YAML, não hardcoded.
    depth_workspace: dict[str, list[float]] | None = None
    # true = grava SÓ 2 checkpoints em disco: `best` (cópia real, atualizada quando
    # o val melhora) e `last` (rolling, sobrescrito a cada save_freq) — sem
    # acumular checkpoints numerados (9.1G cada). Pico de disco: ~3x um checkpoint
    # (best + last + tmp do swap atômico).
    keep_only_best_and_last: bool = False
    # EMA (Exponential Moving Average) dos pesos — run 3 do A/B. Default OFF:
    # com ema_enabled=False o script é no-op estrito (nenhum objeto EMA, nenhuma
    # eval/save extra, caminho raw de validação intacto). Quando true, mantém
    # uma cópia fp32 dos pesos com média móvel e seleciona o best por ela.
    ema_enabled: bool = False
    ema_decay: float = 0.999
    ema_warmup: bool = True   # decay efetivo cresce com o step no começo
    ema_start_step: int = 0   # só começa a atualizar a EMA a partir deste step
    # Run 4a — regularização do `state` (texto) contra causal confusion / atalho
    # proprioceptivo (o braço ficou open-loop nas 3 runs do A/B: braço→imagem ~0.07).
    # SÓ no treino: o forward de validação e o processor salvo p/ deploy ficam intactos
    # (state completo). Default 0/0 = no-op estrito (nem toca o processor). Ver
    # train/state_regularizer.py.
    #   state_dropout_prob: com prob p, monta o prompt SEM "State: …" → força a imagem.
    #   state_noise_bins:   perturba os bins discretizados do state real ±k (clip 0..255).
    state_dropout_prob: float = 0.0
    state_noise_bins: int = 0
    # Validação (valfix): explícitos na dataclass p/ o YAML poder setá-los sem quebrar o
    # parse do draccus (antes eram só getattr com default).
    val_action_mse_batches: int = 16   # nº de batches no val_action_mse (best menos ruidoso)
    val_flow_samples: int = 1          # K amostras de flow (t/ruído) por batch no val_loss (robustez)
    # Run 5 — DEPTH COMO IMAGEM (opção B). Independente de depth_fusion (que é o
    # caminho PointNet/tátil). Quando true, o head_camera_depth (mm) é colorizado
    # (TURBO, vermelho=perto, faixa fixa) e injetado como uma 2ª câmera SigLIP via
    # train/pi05_depth_image_injector.py — sem parâmetro novo. Default false = no-op.
    depth_as_image: bool = False
    depth_image_vis_min_m: float = 0.2   # faixa métrica FIXA p/ o colormap (metros)
    depth_image_vis_max_m: float = 1.5

QUANTILE_MIN_RANGE = 1e-3  # rad; abaixo disso a dim é considerada congelada


def _guard_zero_range_quantile_stats(dataset) -> None:
    """Safety net (auditoria FASE 2): dims congeladas fazem a quantile norm dividir
    por um denominador minúsculo e amplificar ruído de sensor para a escala dos
    sinais reais (medido no right14: state dim 7 tem range 7.2e-6 rad de LSB do
    encoder → normalizava para ±3; `== 0` não pega porque o range não é exatamente
    zero). Critério: range < QUANTILE_MIN_RANGE → q99 = q01 + 1.0, em memória,
    antes do normalizer ser construído. O mesmo guard existe no
    slice_right_arm_only.py para datasets novos."""
    import numpy as _np

    for feat_key in ("action", "observation.state"):
        st = dataset.meta.stats.get(feat_key) if dataset.meta.stats else None
        if not st or "q01" not in st or "q99" not in st:
            continue
        q01 = _np.asarray(st["q01"], dtype=_np.float64)
        q99 = _np.asarray(st["q99"], dtype=_np.float64)
        frozen = _np.nonzero(q99 - q01 < QUANTILE_MIN_RANGE)[0]
        if len(frozen):
            logging.warning(
                f"[stats-guard] {feat_key}: dims {frozen.tolist()} com range ~zero "
                f"(q99-q01 < {QUANTILE_MIN_RANGE}) — forçando q99 = q01 + 1.0 pra "
                f"evitar divisão por ~eps na quantile norm"
            )
            q99[frozen] = q01[frozen] + 1.0
            st["q99"] = q99.astype(_np.asarray(st["q99"]).dtype, copy=False)


from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import MetricsTracker
from lerobot.utils.logging_utils import AverageMeter
from train.utils import VarianceMeter
from train.state_regularizer import install_state_regularizer, state_regularizer_active

from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
from lerobot.utils.constants import ACTION

class EMA:
    """Média móvel exponencial dos pesos treináveis (run 3 do A/B).

    Guarda uma sombra em fp32 (no device do param) de cada
    `p for p in policy.parameters() if p.requires_grad`, indexada pelo nome.
    `apply_to` troca temporariamente os pesos vivos pelos da EMA pra avaliar/
    exportar e restaura os pesos raw no `finally` (mesmo sob exceção).
    """

    @staticmethod
    def _key(name: str) -> str:
        # tolera o prefixo "module." que o DDP adiciona quando a policy vem wrapped
        return name[len("module."):] if name.startswith("module.") else name

    def __init__(self, policy, decay: float = 0.999, warmup: bool = True):
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in policy.named_parameters():
            if p.requires_grad:
                self.shadow[self._key(name)] = p.detach().clone().float()

    @torch.no_grad()
    def update(self, policy, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step)) if self.warmup else self.decay
        for name, p in policy.named_parameters():
            if not p.requires_grad:
                continue
            key = self._key(name)
            ema = self.shadow.get(key)
            if ema is None:
                # param novo (ex.: injeção tardia) — inicia a sombra
                self.shadow[key] = p.detach().clone().float()
                continue
            ema.mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @contextmanager
    def apply_to(self, policy):
        # (a) guarda os pesos raw, (b) copia os pesos EMA pros params vivos,
        # (c) yield, (d) restaura os raw no finally.
        backup: list[tuple[Any, torch.Tensor]] = []
        try:
            with torch.no_grad():
                for name, p in policy.named_parameters():
                    ema = self.shadow.get(self._key(name))
                    if ema is None:
                        continue
                    backup.append((p, p.detach().clone()))
                    p.copy_(ema.to(dtype=p.dtype, device=p.device))
            yield
        finally:
            with torch.no_grad():
                for p, raw in backup:
                    p.copy_(raw)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device).float())
            else:
                self.shadow[k] = v.float()


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.
    """
    start_time = time.perf_counter()
    policy.train()

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        # Use per-sample loss when RA-BC is enabled for proper weighting
        if rabc_batch_weights is not None:
            # Get per-sample losses
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Apply RA-BC weights: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
            # rabc_batch_weights is already normalized to sum to batch_size
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            # Log raw mean weight (before normalization) - this is the meaningful metric
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def export_ema_checkpoint(policy, ema, out_dir, cfg, preprocessor=None, postprocessor=None) -> None:
    """Exporta um pretrained_model/ com os pesos EMA já embutidos (carregável por
    um script de inferência padrão). Dentro de `ema.apply_to`, salva o policy +
    cfg/processors do mesmo jeito que o save_checkpoint faz."""
    out_dir = Path(out_dir)
    with ema.apply_to(policy):
        policy.save_pretrained(out_dir)
        cfg.save_pretrained(out_dir)
        if cfg.peft is not None:
            policy.config.save_pretrained(out_dir)
        if preprocessor is not None:
            preprocessor.save_pretrained(out_dir)
        if postprocessor is not None:
            postprocessor.save_pretrained(out_dir)


@parser.wrap()
def train(cfg: CustomTrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.
    """
    cfg.validate()

    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset creation with Val support
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
        _guard_zero_range_quantile_stats(dataset)
        if hasattr(cfg, 'val_dataset') and cfg.val_dataset:
            logging.info("Creating validation dataset")
            # Temporarily swap dataset config to create val dataset
            train_ds_cfg = cfg.dataset
            cfg.dataset = cfg.val_dataset
            val_dataset = make_dataset(cfg)
            cfg.dataset = train_ds_cfg
        else:
            val_dataset = None

    accelerator.wait_for_everyone()

    if not is_main_process:
        dataset = make_dataset(cfg)
        _guard_zero_range_quantile_stats(dataset)
        if hasattr(cfg, 'val_dataset') and cfg.val_dataset:
            train_ds_cfg = cfg.dataset
            cfg.dataset = cfg.val_dataset
            val_dataset = make_dataset(cfg)
            cfg.dataset = train_ds_cfg
        else:
            val_dataset = None

    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    # =========================================================
    # --- MONKEY PATCH DE FUSÃO GEOMÉTRICA 3D + TÁTIL ---
    # =========================================================
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.depth_fusion and cfg.policy.type == "act":
        from act_d_injector import inject_act_d

        inject_act_d(policy, device=_device)
    elif cfg.depth_fusion and cfg.policy.type == "pi05":
        # In resume (or when pretrained_path points at a local checkpoint that
        # already carries injected weights), reload pointnet/pressure_proj after
        # injection so we don't discard what was already learned.
        injected_ckpt = None
        if cfg.policy.pretrained_path:
            candidate = Path(str(cfg.policy.pretrained_path))
            if candidate.exists() and (candidate / "model.safetensors").exists():
                injected_ckpt = candidate

        if cfg.depth_scale is None:
            raise ValueError(
                "depth_fusion=true exige `depth_scale` explícito no YAML "
                "(PNG16 em milímetros → 0.001). Sem default — ver auditoria FASE 3."
            )

        if cfg.fusion_mode == "depth_only":
            from pi05_depth_injector import inject_pi05_depth

            inject_pi05_depth(
                policy,
                device=_device,
                camera_intrinsics=cfg.depth_intrinsics,
                load_injected_from=injected_ckpt,
                depth_key=cfg.depth_key,
                depth_scale=cfg.depth_scale,
                workspace=cfg.depth_workspace,
            )
        elif cfg.fusion_mode == "full":
            from pi05_d_injector import inject_pi05_d

            inject_pi05_d(
                policy,
                device=_device,
                camera_intrinsics=cfg.depth_intrinsics,
                load_injected_from=injected_ckpt,
                depth_scale=cfg.depth_scale,
                workspace=cfg.depth_workspace,
            )
        else:
            raise ValueError(
                f"Unknown fusion_mode={cfg.fusion_mode!r}; expected 'full' or 'depth_only'."
            )
    else:
        logging.info(
            "Skipping depth-fusion injection (policy type='%s', depth_fusion=%s).",
            cfg.policy.type,
            cfg.depth_fusion,
        )
    # =========================================================

    # --- RUN 5: DEPTH COMO IMAGEM (opção B) — 2ª câmera SigLIP, sem PointNet ---
    # Independente do bloco depth_fusion acima. Patcha forward/predict pra
    # colorizar o depth (TURBO, vermelho=perto) e tratá-lo como imagem nativa.
    if getattr(cfg, "depth_as_image", False):
        if cfg.policy.type != "pi05":
            raise ValueError(f"depth_as_image só suportado em pi05 (policy.type={cfg.policy.type!r}).")
        from pi05_depth_image_injector import inject_pi05_depth_image

        inject_pi05_depth_image(
            policy,
            depth_key=cfg.depth_key,
            vis_min_m=cfg.depth_image_vis_min_m,
            vis_max_m=cfg.depth_image_vis_max_m,
            depth_scale=cfg.depth_scale if cfg.depth_scale is not None else 0.001,
            debug_dir=str(cfg.output_dir),
        )
    # =========================================================

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    accelerator.wait_for_everyone()

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        if cfg.policy.type == "pi05" and hasattr(cfg.policy, "tokenizer_name"):
            processor_kwargs["preprocessor_overrides"]["tokenizer_processor"] = {
                "tokenizer_name": cfg.policy.tokenizer_name,
            }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    # Run 4a — state-dropout/ruído (treino-only). No-op se ambas as flags forem 0.
    # Lê o nº de dims reais do state (sem padding) p/ mirar só elas com o ruído.
    _state_dim = None
    try:
        _sf = (cfg.policy.input_features or {}).get("observation.state")
        _state_dim = int(_sf.shape[0]) if _sf is not None else None
    except Exception:
        _state_dim = None
    state_reg = install_state_regularizer(
        dropout_prob=cfg.state_dropout_prob,
        noise_bins=cfg.state_noise_bins,
        state_dim=_state_dim,
        seed=cfg.seed,
    )
    if state_reg is not None and is_main_process:
        logging.info(
            f"[state-reg] ATIVO (treino-only): dropout_prob={cfg.state_dropout_prob} "
            f"noise_bins={cfg.state_noise_bins} state_dim={_state_dim}"
        )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    # EMA dos pesos (run 3) — criada DEPOIS do load_training_state pra os pesos já
    # estarem carregados e requires_grad setado. No resume, carrega a sombra de
    # disco se existir; senão inicia dos pesos atuais. Tudo guardado por ema_enabled.
    ema = None
    if cfg.ema_enabled:
        ema = EMA(policy, decay=cfg.ema_decay, warmup=cfg.ema_warmup)
        if cfg.resume:
            from safetensors.torch import load_file as _st_load

            ema_path = Path(cfg.checkpoint_path) / "training_state" / "ema_state.safetensors"
            if ema_path.exists():
                ema.load_state_dict(_st_load(str(ema_path)))
                if is_main_process:
                    logging.info(f"EMA restaurada de {ema_path}")
            elif is_main_process:
                logging.info("EMA sem checkpoint prévio; iniciada dos pesos atuais")

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        train_sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        train_sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=0,
            shuffle=True,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False if train_sampler else (shuffle and not cfg.dataset.streaming),
        sampler=train_sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Create Validation DataLoader (if val_dataset provided)
    val_dataloader = None
    if val_dataset:
        val_sampler = EpisodeAwareSampler(
            val_dataset.meta.episodes["dataset_from_index"],
            val_dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=val_dataset.episodes,
            drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
            shuffle=False, # No need to shuffle val
        )
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size, 
            sampler=val_sampler,
            pin_memory=device.type == "cuda",
            drop_last=False
        )
        val_dataloader = accelerator.prepare(val_dataloader)

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": VarianceMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    best_val_loss = float("inf")
    best_val_mse = None
    best_checkpoint_step = None
    # No RESUME, restaura o estado do best salvo em disco — senão a 1ª eval
    # pós-restart vira best incondicional e sobrescreve um best melhor.
    if cfg.resume:
        import json as _json

        _bm = Path(cfg.output_dir) / "checkpoints" / "best" / "best_meta.json"
        if _bm.exists():
            try:
                _meta = _json.loads(_bm.read_text())
                best_val_loss = float(_meta.get("val_loss", float("inf")))
                best_val_mse = _meta.get("val_action_mse")
                best_checkpoint_step = int(_meta["step"])
                if is_main_process:
                    logging.info(
                        f"best restaurado do disco: step {best_checkpoint_step} "
                        f"val_action_mse={best_val_mse} val_loss={best_val_loss:.4f}"
                    )
            except Exception as e:
                logging.warning(f"best_meta.json ilegível ({e}); estado do best zerado")

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        # state-dropout/ruído só aqui (forward de TREINO); val e deploy ficam com state completo.
        with state_regularizer_active(state_reg):
            batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )

        step += 1

        # EMA: atualiza a sombra logo após o optimizer.step() (feito dentro de
        # update_policy), só quando os grads foram sincronizados (sem accum). Guardado
        # por ema_enabled e pelo ema_start_step. Usa a policy unwrapped.
        if ema is not None and accelerator.sync_gradients and step >= cfg.ema_start_step:
            ema.update(accelerator.unwrap_model(policy), step)

        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = (cfg.save_freq > 0 and step % cfg.save_freq == 0) or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # loss_per_dim vem do forward como LISTA. O per-dim vai pra uma seção
                # SEPARADA "dim_train/" (continua logado, mas sem poluir "train/").
                per_dim = wandb_log_dict.pop("loss_per_dim", None)
                dim_train_dict = {}
                if isinstance(per_dim, (list, tuple)):
                    for i, v in enumerate(per_dim):
                        dim_train_dict[f"loss_dim_{i:02d}"] = float(v)
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )

                if isinstance(train_tracker.metrics["loss"], VarianceMeter):
                    wandb_log_dict["loss_std"] = train_tracker.metrics["loss"].std

                wandb_logger.log_dict(wandb_log_dict, step)
                # per-dim na seção separada dim_train/ (mantém logado, fora da principal)
                if dim_train_dict and getattr(wandb_logger, "_wandb", None) is not None:
                    wandb_logger._wandb.log({f"dim_train/{k}": v for k, v in dim_train_dict.items()}, step=step)
            train_tracker.reset_averages()

        # Validation Loop
        if is_eval_step and val_dataloader is not None:
            if is_main_process:
                logging.info(f"Validating at step {step}...")
            
            # modo de avaliação; policy.train() é restaurado ao fim do loop
            policy.eval()

            # action_mse é caro (num_inference_steps forwards por batch), limite em
            # até N batches por eval pra não dominar o wall-clock.
            max_action_mse_batches = getattr(cfg, "val_action_mse_batches", 16)  # mais batches no action_mse pra um best menos ruidoso
            policy_for_predict = accelerator.unwrap_model(policy)

            _EVAL_SEED = 1234   # ruído e timestep fixos no eval: val_loss comparável entre checkpoints
            _val_flow_k = max(1, int(cfg.val_flow_samples))  # K amostras de flow no val_loss (robustez; revisão do prof)
            # fork_rng no device atual (antes era hardcoded devices=[0]). O index pode
            # vir None quando o device é só "cuda" (single-GPU mascarada) → cai pro
            # torch.cuda.current_device(), que é sempre um inteiro válido.
            if accelerator.device.type == "cuda":
                _idx = accelerator.device.index
                _dev = [_idx if _idx is not None else torch.cuda.current_device()]
            else:
                _dev = []

            def _run_val_pass():
                """Roda um passe completo no val_dataloader e devolve os medidores/
                acumuladores. Chamado com os pesos raw e (se ema_enabled) também sob
                ema.apply_to — mesmo seeding _EVAL_SEED pra comparabilidade."""
                val_loss_meter = VarianceMeter("val_loss", ":.3f")
                val_action_mse_meter = VarianceMeter("val_action_mse", ":.4f")
                val_metrics = {}
                # acumuladores per-dim (action MSE e flow loss) — viram val_*_dim_XX
                val_mse_dim_sum, val_mse_dim_n = None, 0
                val_lpd_sum, val_lpd_n = None, 0

                _max_action_mse_batches = max_action_mse_batches
                action_mse_batches_done = 0
                vb_idx = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_batch = preprocessor(val_batch)
                        vb_idx += 1
                        # K amostras de flow (t/ruído distintos) por batch -> val_loss menos ruidoso
                        # (revisão do prof). _EVAL_SEED mantém comparabilidade entre checkpoints; K=1 = de sempre.
                        with torch.random.fork_rng(devices=_dev):
                            _vl_acc, val_output_dict = None, None
                            for _ks in range(_val_flow_k):
                                torch.manual_seed(_EVAL_SEED + vb_idx + _ks * 7919)
                                with accelerator.autocast():
                                    _vl, _vod = policy.forward(val_batch)
                                _vl_acc = _vl if _vl_acc is None else _vl_acc + _vl
                                if val_output_dict is None:
                                    val_output_dict = _vod
                            val_loss = _vl_acc / _val_flow_k

                        if action_mse_batches_done < _max_action_mse_batches:
                            try:
                                # ruído inicial fixo no predict_action_chunk: val_action_mse comparável
                                with torch.random.fork_rng(devices=_dev):
                                    torch.manual_seed(_EVAL_SEED + 100000 + vb_idx)
                                    with accelerator.autocast():
                                        pred_actions = policy_for_predict.predict_action_chunk(val_batch)
                                gt_actions = val_batch[ACTION].to(pred_actions.device)
                                # gt pode vir com shape [B, chunk, dim] ou [B, dim]
                                if gt_actions.dim() == pred_actions.dim():
                                    dim = min(pred_actions.shape[-1], gt_actions.shape[-1])
                                    sq = (pred_actions[..., :dim] - gt_actions[..., :dim]) ** 2
                                    val_action_mse_meter.update(sq.mean().item())
                                    mse_dim = sq.mean(dim=tuple(range(sq.dim() - 1))).float().cpu()
                                    val_mse_dim_sum = mse_dim if val_mse_dim_sum is None else val_mse_dim_sum + mse_dim
                                    val_mse_dim_n += 1
                                action_mse_batches_done += 1
                            except Exception as e:
                                logging.warning("action_mse eval falhou: %s", e)
                                _max_action_mse_batches = 0

                        val_loss_gathered = accelerator.gather(val_loss)

                        if val_output_dict:
                            # loss_per_dim é LISTA (filtrada do loop de meters abaixo);
                            # acumula aqui pra virar val_loss_dim_XX.
                            _lpd = val_output_dict.get("loss_per_dim")
                            if isinstance(_lpd, (list, tuple)):
                                _lpd_t = torch.tensor(_lpd, dtype=torch.float32)
                                val_lpd_sum = _lpd_t if val_lpd_sum is None else val_lpd_sum + _lpd_t
                                val_lpd_n += 1
                            for k, v in val_output_dict.items():
                                if isinstance(v, (int, float, torch.Tensor)):
                                    if k not in val_metrics:
                                        val_metrics[k] = AverageMeter(f"val_{k}", ":.3f")

                                    val_k_gathered = accelerator.gather(torch.tensor(v, device=val_loss.device) if not isinstance(v, torch.Tensor) else v)
                                    if accelerator.num_processes > 1:
                                        for l in val_k_gathered:
                                             if isinstance(l, torch.Tensor): l = l.item()
                                             val_metrics[k].update(l)
                                    else:
                                        if isinstance(v, torch.Tensor):
                                            val = v.mean().item() if v.numel() > 1 else v.item()
                                        else:
                                            val = v
                                        val_metrics[k].update(val)

                        if accelerator.num_processes > 1:
                            for l in val_loss_gathered:
                                val_loss_meter.update(l.item())
                        else:
                            val_loss_meter.update(val_loss.item())

                return {
                    "val_loss_meter": val_loss_meter,
                    "val_action_mse_meter": val_action_mse_meter,
                    "val_metrics": val_metrics,
                    "val_mse_dim_sum": val_mse_dim_sum,
                    "val_mse_dim_n": val_mse_dim_n,
                    "val_lpd_sum": val_lpd_sum,
                    "val_lpd_n": val_lpd_n,
                }

            # caminho RAW (idêntico ao de sempre)
            raw = _run_val_pass()
            val_loss_meter = raw["val_loss_meter"]
            val_action_mse_meter = raw["val_action_mse_meter"]
            val_metrics = raw["val_metrics"]
            val_mse_dim_sum = raw["val_mse_dim_sum"]
            val_mse_dim_n = raw["val_mse_dim_n"]
            val_lpd_sum = raw["val_lpd_sum"]
            val_lpd_n = raw["val_lpd_n"]

            # caminho EMA (só quando ema_enabled): mesmo seeding, pesos da média móvel
            ema_res = None
            if ema is not None:
                with ema.apply_to(policy):
                    ema_res = _run_val_pass()

            policy.train()

            if is_main_process:
                logging.info(f"Validation Results: {val_loss_meter}")
                # eval/ = PRINCIPAL (só o que importa: val_action_mse + split braço/grasp).
                # dim_eval/ = seção separada: per-dim + a flow-loss (val_loss, proxy frouxa),
                # mantida logada mas fora da visão principal.
                val_log_dict = {}
                ema_log_dict = {}   # seção ema/ no wandb (métricas dos pesos EMA, separadas de eval/)
                dim_eval_dict = {
                    "val_loss": val_loss_meter.avg,
                    "val_loss_std": val_loss_meter.std,
                }
                if val_action_mse_meter.count > 0:
                    logging.info(f"  {val_action_mse_meter}")
                    val_log_dict["val_action_mse"] = val_action_mse_meter.avg
                    val_log_dict["val_action_mse_std"] = val_action_mse_meter.std
                mse_dim_avg = None
                if val_mse_dim_n:
                    mse_dim_avg = val_mse_dim_sum / val_mse_dim_n
                    _md = mse_dim_avg.tolist()
                    for i, v in enumerate(_md):
                        dim_eval_dict[f"val_action_mse_dim_{i:02d}"] = v
                    # split p/ eval/ principal: braço = dims 0-6, grasp = dims 7+ (squeeze/dedos)
                    if len(_md) >= 8:
                        val_log_dict["val_action_mse_arm"] = float(sum(_md[:7]) / 7)
                        _grasp = _md[7:]
                        val_log_dict["val_action_mse_grasp"] = float(sum(_grasp) / len(_grasp))
                if val_lpd_n:
                    for i, v in enumerate((val_lpd_sum / val_lpd_n).tolist()):
                        dim_eval_dict[f"val_loss_dim_{i:02d}"] = v
                for k, meter in val_metrics.items():
                    val_log_dict[f"val_{k}"] = meter.avg
                    logging.info(f"  {k}: {meter.avg:.3f}")

                # --- métricas EMA + gaps (só quando ema_enabled) ---
                # Espelha raw/ema lado a lado: action_mse (+ arm/grasp), val_loss e os
                # gaps ema_minus_raw. cur_mse_ema alimenta a seleção do best abaixo.
                cur_mse_ema = None
                if ema_res is not None:
                    def _arm_grasp(mse_dim_sum, mse_dim_n):
                        if not mse_dim_n:
                            return None, None
                        _md = (mse_dim_sum / mse_dim_n).tolist()
                        if len(_md) < 8:
                            return None, None
                        return float(sum(_md[:7]) / 7), float(sum(_md[7:]) / len(_md[7:]))

                    raw_mse = val_action_mse_meter.avg if val_action_mse_meter.count > 0 else None
                    ema_mse_meter = ema_res["val_action_mse_meter"]
                    cur_mse_ema = ema_mse_meter.avg if ema_mse_meter.count > 0 else None
                    raw_arm, raw_grasp = _arm_grasp(val_mse_dim_sum, val_mse_dim_n)
                    ema_arm, ema_grasp = _arm_grasp(ema_res["val_mse_dim_sum"], ema_res["val_mse_dim_n"])
                    raw_loss = val_loss_meter.avg
                    ema_loss = ema_res["val_loss_meter"].avg

                    # raw continua em eval/; EMA vai pra seção própria ema/ (pedido do user)
                    if raw_mse is not None:
                        val_log_dict["val_action_mse_raw"] = raw_mse
                    if cur_mse_ema is not None:
                        ema_log_dict["val_action_mse"] = cur_mse_ema
                    if raw_mse is not None and cur_mse_ema is not None:
                        ema_log_dict["val_action_mse_minus_raw"] = cur_mse_ema - raw_mse
                    if raw_arm is not None:
                        val_log_dict["val_action_mse_arm_raw"] = raw_arm
                    if ema_arm is not None:
                        ema_log_dict["val_action_mse_arm"] = ema_arm
                    if raw_arm is not None and ema_arm is not None:
                        ema_log_dict["val_action_mse_arm_minus_raw"] = ema_arm - raw_arm
                    if raw_grasp is not None:
                        val_log_dict["val_action_mse_grasp_raw"] = raw_grasp
                    if ema_grasp is not None:
                        ema_log_dict["val_action_mse_grasp"] = ema_grasp
                    if raw_grasp is not None and ema_grasp is not None:
                        ema_log_dict["val_action_mse_grasp_minus_raw"] = ema_grasp - raw_grasp
                    val_log_dict["val_loss_raw"] = raw_loss
                    ema_log_dict["val_loss"] = ema_loss
                    ema_log_dict["val_loss_minus_raw"] = ema_loss - raw_loss
                    logging.info(
                        f"  [EMA] val_action_mse raw={raw_mse} ema={cur_mse_ema} "
                        f"val_loss raw={raw_loss:.4f} ema={ema_loss:.4f}"
                    )

                if wandb_logger:
                    wandb_logger.log_dict(val_log_dict, step, mode="eval")
                    if getattr(wandb_logger, "_wandb", None) is not None:
                        # per-dim + flow-loss na seção dim_eval/; métricas dos pesos EMA na seção ema/
                        wandb_logger._wandb.log({f"dim_eval/{k}": v for k, v in dim_eval_dict.items()}, step=step)
                        if ema_log_dict:
                            wandb_logger._wandb.log({f"ema/{k}": v for k, v in ema_log_dict.items()}, step=step)

                # Critério do BEST: SÓ val_action_mse (erro do chunk de ação
                # gerado no held-out) — é o
                # proxy mais direto de comportamento. O critério anterior (dominância
                # em val_loss E val_action_mse) congelou o best no run right8: as
                # duas métricas andaram em direções OPOSTAS (flow-loss subiu 0.098→0.18
                # enquanto o mse caiu 0.093→0.050) e o empate nunca atualizava.
                # Fallback: sem action_mse disponível, compara só val_loss.
                # Com ema_enabled, o best vira o val_action_mse dos pesos EMA
                # (o que será exportado/deployado); senão, o raw de sempre.
                cur_loss = val_loss_meter.avg
                cur_mse = val_action_mse_meter.avg if val_action_mse_meter.count > 0 else None
                sel_mse = cur_mse_ema if cfg.ema_enabled else cur_mse
                if best_checkpoint_step is None:
                    is_new_best = True
                elif sel_mse is not None and best_val_mse is not None:
                    is_new_best = sel_mse <= best_val_mse
                else:
                    is_new_best = cur_loss <= best_val_loss
                if is_new_best:
                    best_val_loss = cur_loss
                    # best_val_mse rastreia a métrica que decide o best (ema mse com
                    # ema_enabled, senão raw mse) — pra o resume comparar igual.
                    best_val_mse = sel_mse
                    best_checkpoint_step = step
                    logging.info(
                        f"  ↑ NEW BEST (val_action_mse{'_ema' if cfg.ema_enabled else ''}) "
                        f"val_action_mse={sel_mse if sel_mse is None else round(sel_mse, 4)} "
                        f"(val_loss={cur_loss:.4f}) at step {step}"
                    )
                    # Salva CÓPIA REAL do best no momento da melhora (granularidade
                    # do eval_freq) — antes o best era só symlink e o mínimo de val
                    # entre saves periódicos nunca era materializado em disco
                    # (ex.: rgb238 teve mínimo no step ~2000 com save_freq 5000).
                    # Escrita em dir temporário + rename atômico: um crash no meio
                    # não corrompe o best anterior.
                    if cfg.save_checkpoint:
                        import shutil as _sh

                        best_dir = Path(cfg.output_dir) / "checkpoints" / "best"
                        tmp_dir = best_dir.parent / ".best_tmp"
                        if tmp_dir.exists():
                            _sh.rmtree(tmp_dir)
                        save_checkpoint(
                            checkpoint_dir=tmp_dir,
                            step=step,
                            cfg=cfg,
                            policy=accelerator.unwrap_model(policy),
                            optimizer=optimizer,
                            scheduler=lr_scheduler,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                        )
                        # métricas do best persistidas junto (lidas no resume).
                        # val_action_mse = métrica que decidiu o best (sel_mse); raw/ema
                        # guardados à parte quando ema_enabled.
                        import json as _json

                        _meta = {
                            "step": step,
                            "val_loss": cur_loss,
                            "val_action_mse": sel_mse,
                        }
                        if cfg.ema_enabled:
                            _meta["val_action_mse_raw"] = cur_mse
                            _meta["val_action_mse_ema"] = cur_mse_ema
                        (tmp_dir / "best_meta.json").write_text(_json.dumps(_meta))
                        # EMA: salva a sombra no training_state do best e exporta um
                        # pretrained_model_ema/ com os pesos EMA embutidos (deploy).
                        if cfg.ema_enabled and ema is not None:
                            from safetensors.torch import save_file as _st_save

                            _st_save(
                                ema.state_dict(),
                                str(tmp_dir / "training_state" / "ema_state.safetensors"),
                            )
                            export_ema_checkpoint(
                                accelerator.unwrap_model(policy),
                                ema,
                                tmp_dir / "pretrained_model_ema",
                                cfg,
                                preprocessor=preprocessor,
                                postprocessor=postprocessor,
                            )
                        if best_dir.is_symlink() or best_dir.is_file():
                            best_dir.unlink()
                        elif best_dir.is_dir():
                            _sh.rmtree(best_dir)
                        tmp_dir.rename(best_dir)
                        logging.info(f"  best salvo (cópia real) em {best_dir} [step {step}]")

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                if getattr(cfg, "keep_only_best_and_last", False):
                    # rolling `last`: sobrescreve via tmp + rename atômico, sem
                    # acumular numerados (só best+last reais em disco).
                    import shutil as _sh

                    last_dir = Path(cfg.output_dir) / "checkpoints" / "last"
                    tmp_dir = last_dir.parent / ".last_tmp"
                    if tmp_dir.exists():
                        _sh.rmtree(tmp_dir)
                    save_checkpoint(
                        checkpoint_dir=tmp_dir,
                        step=step,
                        cfg=cfg,
                        policy=accelerator.unwrap_model(policy),
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                    )
                    if last_dir.is_symlink() or last_dir.is_file():
                        last_dir.unlink()
                    elif last_dir.is_dir():
                        _sh.rmtree(last_dir)
                    tmp_dir.rename(last_dir)
                    logging.info(f"  last (rolling) salvo em {last_dir} [step {step}]")
                    checkpoint_dir = last_dir
                else:
                    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=step,
                        cfg=cfg,
                        policy=accelerator.unwrap_model(policy),
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                    )
                    update_last_checkpoint(checkpoint_dir)
                # EMA: salva a sombra junto do checkpoint periódico (last ou numerado),
                # no mesmo training_state/ que o save_checkpoint criou. Guardado por ema_enabled.
                if cfg.ema_enabled and ema is not None:
                    from safetensors.torch import save_file as _st_save

                    _st_save(
                        ema.state_dict(),
                        str(Path(checkpoint_dir) / "training_state" / "ema_state.safetensors"),
                    )
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                aggregated = eval_info["overall"]

                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
