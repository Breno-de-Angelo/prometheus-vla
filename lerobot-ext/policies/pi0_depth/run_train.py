#!/usr/bin/env python
"""
ACT-D Training Engine — Improved
Melhorias nesta versão:
  1. Best-Val Checkpoint automático: salva sempre que val_loss bate um novo recorde
  2. Ctrl+C graceful: intercepta SIGINT e salva o estado atual antes de encerrar
  3. Treinamento de posição neutra: curriculum que ensina o robô a retornar ao neutro
     quando não reconhece o cenário (scene uncertainty gate).
"""

import dataclasses
import logging
import signal
import time
from contextlib import nullcontext
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

# Add current directory to path
sys.path.append(os.getcwd())

# ─────────────────────────────────────────────────────────────
# MONKEY PATCHES
# ─────────────────────────────────────────────────────────────
# Os dois patches que viviam aqui — remapear índice absoluto→relativo quando o
# YAML seleciona um subconjunto de `episodes`, e aplicar o ColorJitter só nas
# câmeras RGB — foram removidos na migração para a 0.6.1: o LeRobot passou a
# fazer os dois nativamente, no `datasets/dataset_reader.py`:
#
#   - o índice é resolvido pelo próprio reader (`_absolute_to_relative_idx`);
#   - as transformações pulam a profundidade
#     (`for cam in camera_keys: if cam in depth_keys: continue`).
#
# Manter o patch quebrava o treino de saída: ele chamava
# `self._ensure_hf_dataset_loaded()`, método que não existe mais no dataset.
# ─────────────────────────────────────────────────────────────

from .processor_pi05 import make_pi05depth_pre_post_processors as make_pi05_pre_post_processors

from .configuration_pi05 import PI05DEPTHConfig
from .modeling_pi05 import PI05DEPTHPolicy

from lerobot.configs.train import TrainPipelineConfig, DatasetConfig

@dataclasses.dataclass
class CustomTrainPipelineConfig(TrainPipelineConfig):
    val_dataset: DatasetConfig | None = None
    # De quantos em quantos steps rodar a validação no `val_dataset`.
    # Na 0.6.1 o `eval_freq` do LeRobot foi partido em dois — `env_eval_freq`
    # (rollout no simulador) e `eval_steps` (loss em episódios separados, via
    # `eval_split`) — e sumiu com esse nome. O laço de validação aqui é nosso e
    # usa o `val_dataset`, não o `eval_split`, então o knob volta como campo
    # nosso, com a semântica de sempre: 0 desliga.
    eval_freq: int = 0
    # ── Novas flags de treinamento ──────────────────────────
    # Salva um checkpoint separado sempre que val_loss melhora
    save_best_checkpoint: bool = True
    # Pesos para o loss de posição neutra no curriculum
    neutral_position_loss_weight: float = 0.3
    # Limiar de incerteza para acionar posição neutra (0 = desligado)
    scene_uncertainty_threshold: float = 0.0

from lerobot.configs import parser
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
# 0.6.1: `cycle` saiu de `lerobot.datasets.utils` (que virou só coisa de dataset).
from lerobot.utils.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.common.wandb_utils import WandBLogger  # 0.6.1: era lerobot.rl.wandb_utils
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import MetricsTracker, AverageMeter
from policies.act_depth.utils import VarianceMeter
from lerobot.utils.random_utils import set_seed
from lerobot.common.train_utils import (  # 0.6.1: era lerobot.utils.train_utils
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, has_method, init_logging


# ═════════════════════════════════════════════════════════════
# DECODER DE VÍDEO: cada worker precisa dos seus próprios
# ═════════════════════════════════════════════════════════════
# O LeRobot mantém um cache global de decoders em
# `lerobot/datasets/video_utils.py` (VideoDecoderCache), e cada entrada guarda
# um FILE HANDLE aberto:
#
#     file_handle = fsspec.open(video_path).__enter__()
#     decoder = VideoDecoder(file_handle, seek_mode="approximate")
#
# Quando o DataLoader forka os workers, os filhos herdam esse handle — e um
# fork compartilha a *file description*, ou seja, o mesmo offset de arquivo
# entre o pai e todos os workers. Quatro processos fazendo seek no mesmo offset
# se atropelam, o decoder lê bytes da posição errada e estoura com
# "Could not push packet to decoder: Invalid data found when processing input".
#
# O sintoma é intermitente (depende de qual índice cai em qual worker), some em
# leitura sequencial e reaparece com shuffle — o que dificulta o diagnóstico.
# Medido: com o cache populado no pai antes do fork, 3 de 3 tentativas falham
# dentro de 41 batches; com este worker_init_fn, 3 de 3 passam.
#
# Vale para as três políticas (actdepth, pi05depth, openvladepth), porque todas
# usam este mesmo loop de treino.
from lerobot.datasets import video_utils


def _reset_video_decoder_cache(_worker_id: int) -> None:
    """Descarta os decoders herdados do pai; cada worker abre os seus."""
    video_utils._default_decoder_cache.clear()


# ═════════════════════════════════════════════════════════════
# GRACEFUL SHUTDOWN: Ctrl+C salva o estado e encerra limpo
# ═════════════════════════════════════════════════════════════
_SHUTDOWN_REQUESTED = False

def _handle_sigint(signum, frame):
    global _SHUTDOWN_REQUESTED
    if not _SHUTDOWN_REQUESTED:
        _SHUTDOWN_REQUESTED = True
        # Não levanta exceção: o loop de treino vai checar a flag e salvar
        print(colored(
            "\n[SIGINT] Ctrl+C detectado — o treinamento vai salvar o estado atual e encerrar limpo.",
            "yellow", attrs=["bold"]
        ))
    else:
        # Segunda vez: encerra na força
        sys.exit(1)


# ═════════════════════════════════════════════════════════════
# BEST-VAL CHECKPOINT TRACKER
# ═════════════════════════════════════════════════════════════
class BestValTracker:
    """Rastreia o melhor val_loss e dispara checkpoint quando bate recorde."""

    def __init__(self, output_dir: Path, total_steps: int):
        self.output_dir = output_dir
        self.total_steps = total_steps
        self.best_val_loss = float("inf")
        self.best_step = -1

    def update(self, val_loss: float, step: int, *, cfg, policy, optimizer,
               lr_scheduler, preprocessor, postprocessor, accelerator) -> bool:
        """Retorna True se um novo recorde foi salvo."""
        if val_loss >= self.best_val_loss:
            return False

        self.best_val_loss = val_loss
        self.best_step = step

        checkpoint_dir = self.output_dir / "best_val_checkpoint"
        logging.info(
            colored(
                f"🏆 NOVO RECORDE de val_loss={val_loss:.5f} no step {step}! "
                f"Salvando best checkpoint em {checkpoint_dir}",
                "green", attrs=["bold"]
            )
        )
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
        # Marca também como last para compatibilidade
        update_last_checkpoint(checkpoint_dir)

        # Salva metadados legíveis
        meta_path = checkpoint_dir / "best_val_meta.txt"
        meta_path.write_text(
            f"best_val_loss={val_loss:.6f}\nbest_step={step}\ntotal_steps={self.total_steps}\n"
        )
        return True


# ═════════════════════════════════════════════════════════════
# NEUTRAL POSITION CURRICULUM
# ═════════════════════════════════════════════════════════════
class NeutralPositionCurriculum:
    """
    Ensina o robô a voltar para uma posição neutra quando está incerto.

    Estratégia:
      - A cada N steps, injeta no batch um exemplo "sintético" onde a ação
        correta é a posição neutra (zeros ou posição configurável).
      - O loss desse exemplo é somado ao loss real com peso `loss_weight`.
      - Durante inferência (select_action), o modelo pode usar a entropia
        das predições VAE (log_sigma) para detectar incerteza de cenário.
    """

    def __init__(
        self,
        neutral_position: torch.Tensor | None = None,
        loss_weight: float = 0.3,
        inject_every_n_steps: int = 50,
        action_dim: int = 28,
        chunk_size: int = 100,
    ):
        self.loss_weight = loss_weight
        self.inject_every_n_steps = inject_every_n_steps
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        # Posição neutra: zeros se não configurada
        if neutral_position is None:
            self.neutral_position = torch.zeros(action_dim)
        else:
            self.neutral_position = neutral_position

    def should_inject(self, step: int) -> bool:
        return self.inject_every_n_steps > 0 and (step % self.inject_every_n_steps == 0)

    def compute_neutral_loss(
        self,
        policy: PreTrainedPolicy,
        batch: dict[str, torch.Tensor],
        accelerator: Accelerator,
    ) -> torch.Tensor:
        """
        Clona o batch, substitui as ações pela posição neutra e computa o loss.
        Isso força a política a aprender que "quando incerta → volte ao neutro".

        IMPORTANTE — gestão de VRAM:
        O forward do PaliGemma aloca ~60 GB de ativações para o grafo de gradiente.
        Ter dois grafos simultâneos (main loss + neutral loss) estoura a VRAM mesmo
        em GPUs A100-80 GB. Por isso este método NÃO retorna um tensor com grafo —
        ele faz o backward() aqui mesmo, dentro de um bloco isolado, e devolve apenas
        o valor escalar (.detach()) para logging. O backward do loss principal é feito
        separadamente em update_policy(), garantindo que apenas UM grafo gigante
        existe na VRAM por vez.

        Fluxo em update_policy():
          1. neutral forward → backward → grafo liberado   ← aqui
          2. main    forward → backward → grafo liberado   ← update_policy
          3. optimizer.step() acumula os dois gradientes
        """
        neutral_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        B = neutral_batch["action"].shape[0]
        half_B = max(1, B // 2)  # metade do batch para reduzir pico de VRAM
        device = neutral_batch["action"].device

        # Trunca o batch para half_B (menos VRAM, sinal de gradiente ainda útil)
        neutral_batch = {
            k: v[:half_B] if isinstance(v, torch.Tensor) else v
            for k, v in neutral_batch.items()
        }

        # Preenche todas as ações do chunk com a posição neutra
        neutral_actions = (
            self.neutral_position
            .unsqueeze(0)              # [1, action_dim]
            .unsqueeze(0)              # [1, 1, action_dim]
            .expand(half_B, self.chunk_size, self.action_dim)
            .to(device)
        )
        neutral_batch["action"] = neutral_actions
        # Remove o pad mask para que todos os steps sejam usados
        neutral_batch["action_is_pad"] = torch.zeros(half_B, self.chunk_size, dtype=torch.bool, device=device)

        with accelerator.autocast():
            neutral_loss, _ = policy.forward(neutral_batch)

        scaled_loss = neutral_loss * self.loss_weight

        # Backward imediato: libera o grafo antes do forward principal
        accelerator.backward(scaled_loss)

        # Retorna apenas o valor escalar para logging — sem grafo residual na VRAM
        return scaled_loss.detach()


# ═════════════════════════════════════════════════════════════
# UPDATE POLICY (com suporte ao curriculum neutro)
# ═════════════════════════════════════════════════════════════
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
    neutral_curriculum: NeutralPositionCurriculum | None = None,
    step: int = 0,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()

    # ── Zero grad antes de qualquer backward ─────────────────────────────────
    # Movido para cá pois o neutral curriculum faz seu próprio backward()
    # internamente — precisamos do zero_grad antes dele, não depois.
    optimizer.zero_grad()

    output_dict = {}

    # ── Passo 1: Curriculum de posição neutra (backward separado) ────────────
    # compute_neutral_loss() faz forward + backward + libera o grafo por conta
    # própria. Isso garante que apenas UM grafo gigante (PaliGemma) existe na
    # VRAM por vez, evitando OOM em GPUs com ~79 GB.
    if neutral_curriculum is not None and neutral_curriculum.should_inject(step):
        neutral_loss_val = neutral_curriculum.compute_neutral_loss(policy, batch, accelerator)
        output_dict["neutral_curriculum_loss"] = neutral_loss_val.item()

    # ── Passo 2: Loss principal (backward separado) ──────────────────────────
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    with accelerator.autocast():
        if rabc_batch_weights is not None:
            per_sample_loss, main_output_dict = policy.forward(batch, reduction="none")
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            main_output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            main_output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            main_output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, main_output_dict = policy.forward(batch)

    output_dict.update(main_output_dict)
    accelerator.backward(loss)

    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    with lock if lock is not None else nullcontext():
        optimizer.step()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


# ═════════════════════════════════════════════════════════════
# MAIN TRAIN FUNCTION
# ═════════════════════════════════════════════════════════════
@parser.wrap()
def train(cfg: CustomTrainPipelineConfig, accelerator: Accelerator | None = None):
    cfg.validate()

    # Registra o handler de Ctrl+C
    signal.signal(signal.SIGINT, _handle_sigint)

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
            logging.info(colored("Logs serão salvos localmente.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ── Dataset ──────────────────────────────────────────────
    if is_main_process:
        logging.info("Criando dataset de treino")
        dataset = make_dataset(cfg)
        if hasattr(cfg, "val_dataset") and cfg.val_dataset:
            logging.info("Criando dataset de validação")
            train_ds_cfg = cfg.dataset
            cfg.dataset = cfg.val_dataset
            val_dataset = make_dataset(cfg)
            cfg.dataset = train_ds_cfg
        else:
            val_dataset = None

    accelerator.wait_for_everyone()

    if not is_main_process:
        dataset = make_dataset(cfg)
        if hasattr(cfg, "val_dataset") and cfg.val_dataset:
            train_ds_cfg = cfg.dataset
            cfg.dataset = cfg.val_dataset
            val_dataset = make_dataset(cfg)
            cfg.dataset = train_ds_cfg
        else:
            val_dataset = None

    # A correção que ficava aqui — sobrescrever as stats do depth por
    # identidade (mean=0, std=1) — saiu na migração para a 0.6.1. Ela existia
    # porque o `make_dataset` carimbava stats do ImageNet em TODAS as
    # camera_keys, depth incluída. A 0.6.1 pula as câmeras de profundidade
    # nesse ponto (`datasets/factory.py`: `if key in depth_keys: continue`).
    #
    # Além de dead code, ela agora estaria ERRADA: escrevia tensores de 3
    # canais em stats de um mapa de 1 canal, e apagava as stats reais (em mm).
    # ─────────────────────────────────────────────────────────────────────────

    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Criando ambiente de avaliação")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Criando política")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    if cfg.peft is not None:
        logging.info("Usando PEFT! Envolvendo o modelo.")
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

    if is_main_process:
        logging.info("Criando otimizador e scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    rabc_weights = None
    if getattr(cfg, "use_rabc", False):
        # 0.6.1: o RA-BC virou parte do subsistema de recompensa (SARM).
        from lerobot.rewards.sarm.rabc import RABCWeights
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("chunk_size não encontrado na configuração da política")
        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Carregando progresso SARM para RA-BC: {cfg.rabc_progress_path}")
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

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Criando processadores de ambiente")
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

    # ── DataLoaders ──────────────────────────────────────────
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        train_sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
            # 0.6.1: o sampler numera os quadros pelo índice ABSOLUTO do dataset
            # inteiro, mas o `__getitem__` recebe índice RELATIVO ao subconjunto de
            # `episodes`. Este mapa é a ponte — sem ele o treino estoura com
            # "Invalid key: N is out of bounds" assim que o YAML seleciona episódios.
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
        )
    else:
        shuffle = True
        train_sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=0,
            shuffle=True,
            # 0.6.1: o sampler numera os quadros pelo índice ABSOLUTO do dataset
            # inteiro, mas o `__getitem__` recebe índice RELATIVO ao subconjunto de
            # `episodes`. Este mapa é a ponte — sem ele o treino estoura com
            # "Invalid key: N is out of bounds" assim que o YAML seleciona episódios.
            absolute_to_relative_idx=dataset.absolute_to_relative_idx,
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
        worker_init_fn=_reset_video_decoder_cache,
    )

    val_dataloader = None
    if val_dataset:
        val_sampler = EpisodeAwareSampler(
            val_dataset.meta.episodes["dataset_from_index"],
            val_dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=val_dataset.episodes,
            drop_n_last_frames=getattr(cfg.policy, "drop_n_last_frames", 0),
            shuffle=False,
            # 0.6.1: o sampler numera os quadros pelo índice ABSOLUTO do dataset
            # inteiro, mas o `__getitem__` recebe índice RELATIVO ao subconjunto de
            # `episodes`. Este mapa é a ponte — sem ele o treino estoura com
            # "Invalid key: N is out of bounds" assim que o YAML seleciona episódios.
            absolute_to_relative_idx=val_dataset.absolute_to_relative_idx,
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            sampler=val_sampler,
            pin_memory=device.type == "cuda",
            drop_last=False,
            worker_init_fn=_reset_video_decoder_cache,
        )
        val_dataloader = accelerator.prepare(val_dataloader)

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)
    policy.train()

    # ── Métricas ─────────────────────────────────────────────
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_metrics = {
        "loss": VarianceMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    # ── Best-Val Tracker ──────────────────────────────────────
    best_val_tracker = BestValTracker(
        output_dir=Path(cfg.output_dir),
        total_steps=cfg.steps,
    )

    # ── Neutral Position Curriculum ───────────────────────────
    neutral_curriculum = None
    if getattr(cfg, "neutral_position_loss_weight", 0.0) > 0:
        action_dim = list(cfg.policy.output_features.values())[0].shape[0]
        chunk_size = getattr(cfg.policy, "chunk_size", 100)

        # ── Calcula a posição neutra REAL a partir do robô ───
        # Prioridade 1: UnitreeG1Config.default_positions (fonte da verdade física)
        # Prioridade 2: média dos primeiros frames do dataset
        # Prioridade 3: zeros (último recurso)
        neutral_position = None
        if is_main_process:
            from policies.act_depth.neutral_position import compute_neutral_position, compute_neutral_from_dataset
            try:
                from robot.unitree_g1.config_unitree_g1 import UnitreeG1Config
                robot_cfg = UnitreeG1Config()  # usa os defaults — mesmo que o robô real
                action_stats = {
                    k: torch.tensor(v) for k, v in dataset.meta.stats["action"].items()
                }
                neutral_position = compute_neutral_position(
                    robot_config=robot_cfg,
                    action_stats=action_stats,
                    action_dim=action_dim,
                    device="cpu",
                )
                logging.info(
                    colored("[NeutralPosition] Usando default_positions do UnitreeG1Config.", "cyan")
                )
            except Exception as e:
                logging.warning(
                    f"[NeutralPosition] Não consegui carregar UnitreeG1Config ({e}). "
                    "Tentando calcular do dataset..."
                )
                try:
                    neutral_position = compute_neutral_from_dataset(
                        dataset=dataset,
                        action_dim=action_dim,
                        device="cpu",
                    )
                    logging.info(
                        colored("[NeutralPosition] Usando média dos primeiros frames do dataset.", "cyan")
                    )
                except Exception as e2:
                    logging.warning(
                        f"[NeutralPosition] Fallback também falhou ({e2}). Usando zeros."
                    )
                    neutral_position = None  # NeutralPositionCurriculum usa zeros

        neutral_curriculum = NeutralPositionCurriculum(
            neutral_position=neutral_position,
            loss_weight=cfg.neutral_position_loss_weight,
            inject_every_n_steps=50,
            action_dim=action_dim,
            chunk_size=chunk_size,
        )

        # Passa a posição neutra também para o modelo (uncertainty gate)
        if neutral_position is not None:
            unwrapped = accelerator.unwrap_model(policy)
            if hasattr(unwrapped, "neutral_position"):
                unwrapped.neutral_position.copy_(
                    neutral_position.to(unwrapped.neutral_position.device)
                )
                if is_main_process:
                    logging.info(
                        colored("[NeutralPosition] Uncertainty gate do modelo atualizado.", "cyan")
                    )

        if is_main_process:
            logging.info(
                colored(
                    f"[Curriculum Neutro] ATIVO — loss_weight={cfg.neutral_position_loss_weight}, "
                    f"inject_every=50 steps",
                    "cyan"
                )
            )

    if is_main_process:
        logging.info(
            f"Iniciando treinamento offline com effective batch size: {effective_batch_size}"
        )

    # ═══════════════════════════════════════════════════════
    # LOOP PRINCIPAL DE TREINAMENTO
    # ═══════════════════════════════════════════════════════
    try:
        for _ in range(step, cfg.steps):
            # ── Checar Ctrl+C ────────────────────────────────
            if _SHUTDOWN_REQUESTED:
                break

            start_time = time.perf_counter()
            batch = next(dl_iter)
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
                neutral_curriculum=neutral_curriculum,
                step=step,
            )

            step += 1
            train_tracker.step()
            is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
            is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
            is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

            # ── Logging ──────────────────────────────────────
            if is_log_step:
                logging.info(train_tracker)
                if wandb_logger:
                    wandb_log_dict = train_tracker.to_dict()
                    if output_dict:
                        flat_output = {}
                        for k, v in output_dict.items():
                            if isinstance(v, (list, tuple)):
                                # Expande listas em chaves indexadas: loss_per_dim/0, loss_per_dim/1, ...
                                for i, val in enumerate(v):
                                    flat_output[f"{k}/{i}"] = val
                            else:
                                flat_output[k] = v
                        wandb_log_dict.update(flat_output)
                    if rabc_weights is not None:
                        rabc_stats = rabc_weights.get_stats()
                        wandb_log_dict.update({
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        })
                    if isinstance(train_tracker.metrics["loss"], VarianceMeter):
                        wandb_log_dict["loss_std"] = train_tracker.metrics["loss"].std
                    wandb_logger.log_dict(wandb_log_dict, step)
                train_tracker.reset_averages()

            # ── Validação + Best-Val Checkpoint ──────────────
            if is_eval_step and val_dataloader is not None:
                if is_main_process:
                    logging.info(f"Validando no step {step}...")

                val_loss_meter = VarianceMeter("val_loss", ":.3f")
                val_metrics = {}

                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_batch = preprocessor(val_batch)
                        with accelerator.autocast():
                            val_loss, val_output_dict = policy.forward(val_batch)

                        val_loss_gathered = accelerator.gather(val_loss)

                        if val_output_dict:
                            for k, v in val_output_dict.items():
                                if isinstance(v, (int, float, torch.Tensor)):
                                    if k not in val_metrics:
                                        val_metrics[k] = AverageMeter(f"val_{k}", ":.3f")
                                    val_k_gathered = accelerator.gather(
                                        torch.tensor(v, device=val_loss.device)
                                        if not isinstance(v, torch.Tensor) else v
                                    )
                                    if accelerator.num_processes > 1:
                                        for l in val_k_gathered:
                                            if isinstance(l, torch.Tensor):
                                                l = l.item()
                                            val_metrics[k].update(l)
                                    else:
                                        val = v.mean().item() if isinstance(v, torch.Tensor) and v.numel() > 1 else (v.item() if isinstance(v, torch.Tensor) else v)
                                        val_metrics[k].update(val)

                        if accelerator.num_processes > 1:
                            for l in val_loss_gathered:
                                val_loss_meter.update(l.item())
                        else:
                            val_loss_meter.update(val_loss.item())

                policy.train()

                if is_main_process:
                    logging.info(f"Validação: {val_loss_meter}")
                    val_log_dict = {
                        "val_loss": val_loss_meter.avg,
                        "val_loss_std": val_loss_meter.std,
                    }
                    for k, meter in val_metrics.items():
                        val_log_dict[f"val_{k}"] = meter.avg
                        logging.info(f"  {k}: {meter.avg:.3f}")

                    if wandb_logger:
                        wandb_logger.log_dict(val_log_dict, step, mode="eval")

                    # ── Salva best-val checkpoint se melhorou ──
                    if getattr(cfg, "save_best_checkpoint", True):
                        is_new_record = best_val_tracker.update(
                            val_loss=val_loss_meter.avg,
                            step=step,
                            cfg=cfg,
                            policy=policy,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            accelerator=accelerator,
                        )
                        if is_new_record and wandb_logger:
                            wandb_logger.log_dict(
                                {"best_val_loss": val_loss_meter.avg, "best_val_step": step},
                                step, mode="eval"
                            )

            # ── Checkpoint periódico normal ───────────────────
            if cfg.save_checkpoint and is_saving_step:
                if is_main_process:
                    logging.info(f"Salvando checkpoint no step {step}")
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
                    if wandb_logger:
                        wandb_logger.log_policy(checkpoint_dir)
                accelerator.wait_for_everyone()

            # ── Avaliação em ambiente ─────────────────────────
            if cfg.env and is_eval_step:
                if is_main_process:
                    step_id = get_step_identifier(step, cfg.steps)
                    logging.info(f"Avaliando política no step {step}")
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
                        logging.info("Suite %s: %s", suite, suite_info)
                    eval_metrics = {
                        "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                        "pc_success": AverageMeter("success", ":.1f"),
                        "eval_s": AverageMeter("eval_s", ":.3f"),
                    }
                    eval_tracker = MetricsTracker(
                        cfg.batch_size, dataset.num_frames, dataset.num_episodes,
                        eval_metrics, initial_step=step, accelerator=accelerator,
                    )
                    eval_tracker.eval_s = aggregated.pop("eval_s")
                    eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                    eval_tracker.pc_success = aggregated.pop("pc_success")
                    if wandb_logger:
                        wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                        wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                        wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")
                accelerator.wait_for_everyone()

    # ══════════════════════════════════════════════════════════
    # GRACEFUL SHUTDOWN: salva o estado atual ao receber Ctrl+C
    # ══════════════════════════════════════════════════════════
    except Exception as exc:
        logging.exception(f"Erro não tratado durante o treinamento: {exc}")
        raise
    finally:
        if _SHUTDOWN_REQUESTED or True:  # always runs
            if is_main_process and _SHUTDOWN_REQUESTED:
                logging.info(colored(
                    f"[SHUTDOWN] Salvando checkpoint de emergência no step {step}...",
                    "yellow", attrs=["bold"]
                ))
                emergency_dir = Path(cfg.output_dir) / "emergency_checkpoint"
                save_checkpoint(
                    checkpoint_dir=emergency_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(emergency_dir)
                logging.info(colored(
                    f"✅ [SHUTDOWN] Checkpoint salvo em {emergency_dir}. Encerrando.",
                    "green", attrs=["bold"]
                ))

        if eval_env:
            close_envs(eval_env)

        if is_main_process:
            logging.info("Fim do treinamento")
            if not _SHUTDOWN_REQUESTED and cfg.policy.push_to_hub:
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