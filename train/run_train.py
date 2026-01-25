#!/usr/bin/env python

import dataclasses
import logging
import time
from contextlib import nullcontext
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

# --- MONKEY PATCHES START ---
import lerobot.datasets.lerobot_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Patch LeRobotDataset.__getitem__ to fix global indexing
def custom_getitem(self, idx):
    # Ensure dataset is loaded when we actually need to read from it
    self._ensure_hf_dataset_loaded()
    
    # --- PATCH: Map global index to local index if needed ---
    if self._absolute_to_relative_idx is not None:
        if idx in self._absolute_to_relative_idx:
            idx = self._absolute_to_relative_idx[idx]
    # --------------------------------------------------------

    item = self.hf_dataset[idx]
    ep_idx = item["episode_index"].item()
    # Use the absolute index from the dataset for delta timestamp calculations
    abs_idx = item["index"].item()

    query_indices = None
    if self.delta_indices is not None:
        query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
        query_result = self._query_hf_dataset(query_indices)
        item = {**item, **padding}

        # For video frames, we need to decode them from the video file
        for vid_key, timestamps in self._get_query_timestamps(
            item["timestamp"].item(), query_indices
        ).items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            # We need to shift the timestamps to be relative to the beginning of the video file
            # (video file starts at `video_from_timestamp`)
            # Correct retrieval of timestamp from metadata
            video_from_timestamp = self.meta.episodes[ep_idx][f"videos/{vid_key}/from_timestamp"]
            # Note: query_timestamps are relative to the episode start.
            # video_from_timestamp is the start of the episode in the video file.
            # wait, confirm logic in original source:
            # shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            # My current implementation was:
            # shifted_query_ts = [ts - video_from_timestamp for ts in timestamps]
            # Wait, let's re-read original _query_videos logic.
            # item["timestamp"] is relative to episode start?
            # query_timestamps = self._get_query_timestamps(...) -> returns timestamps relative to episode start (if using hf_dataset which has relative timestamps?)
            # hf_dataset['timestamp'] are usually relative to episode start in lerobot?
            # Let's check _get_query_timestamps.
            # If they are relative to episode start, then to get position in video file:
            # video_file_pos = from_timestamp_of_episode_in_video + timestamp_in_episode
            # So it should be ADDED.
            shifted_query_ts = [val + video_from_timestamp for val in timestamps]
            frames = lerobot.datasets.lerobot_dataset.decode_video_frames(video_path, shifted_query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        # For other keys (e.g. state, action), we can just retrieve them from the dataset directly
        for key, val in query_result.items():
            item[key] = val.squeeze(0)

    if self.image_transforms is not None:
        item = self.image_transforms(item)

    return item

LeRobotDataset.__getitem__ = custom_getitem

# Patch ACTPolicy.forward to fix None KLD issue
import lerobot.policies.act.modeling_act
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES

# def custom_act_forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
#     """Run the batch through the model and compute the loss for training or validation."""
#     if self.config.image_features:
#         batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
#         batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

#     actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

#     l1_loss = (
#         F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
#     ).mean()

#     loss_dict = {"l1_loss": l1_loss.item()}
    
#     # --- PATCH: Check for None (validation mode) ---
#     if self.config.use_vae and log_sigma_x2_hat is not None:
#     # -----------------------------------------------
#         # Calculate Dₖₗ(latent_pdf || standard_normal). Note: After computing the KL-divergence for
#         # each dimension independently, we sum over the latent dimension to get the total
#         # KL-divergence per batch element, then take the mean over the batch.
#         # (See App. B of https://huggingface.co/papers/1312.6114 for more details).
#         mean_kld = (
#             (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
#         )
#         loss_dict["kld_loss"] = mean_kld.item()
#         loss = l1_loss + mean_kld * self.config.kl_weight
#     else:
#         loss = l1_loss

#     return loss, loss_dict

# ACTPolicy.forward = custom_act_forward

from lerobot.configs.train import TrainPipelineConfig, DatasetConfig
@dataclasses.dataclass
class CustomTrainPipelineConfig(TrainPipelineConfig):
    val_dataset: DatasetConfig | None = None

# --- MONKEY PATCHES END ---

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

    for _ in range(step, cfg.steps):
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
        )

        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
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
            train_tracker.reset_averages()

        # Validation Loop
        if is_eval_step and val_dataloader is not None:
            if is_main_process:
                logging.info(f"Validating at step {step}...")
            
            # Have to keep the policy in training mode because of style prediction
            # policy.eval()
            policy.train()

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
                                
                                val_k_gathered = accelerator.gather(torch.tensor(v, device=val_loss.device) if not isinstance(v, torch.Tensor) else v)
                                if accelerator.num_processes > 1:
                                    for l in val_k_gathered:
                                         if isinstance(l, torch.Tensor): l = l.item()
                                         val_metrics[k].update(l)
                                else:
                                    val = v.item() if isinstance(v, torch.Tensor) else v
                                    val_metrics[k].update(val)

                    if accelerator.num_processes > 1:
                        for l in val_loss_gathered:
                            val_loss_meter.update(l.item())
                    else:
                        val_loss_meter.update(val_loss.item())

            policy.train()

            if is_main_process:
                logging.info(f"Validation Results: {val_loss_meter}")
                val_log_dict = {
                    "val_loss": val_loss_meter.avg,
                    "val_loss_std": val_loss_meter.std
                }
                for k, meter in val_metrics.items():
                    val_log_dict[f"val_{k}"] = meter.avg
                    logging.info(f"  {k}: {meter.avg:.3f}")

                if wandb_logger:
                    wandb_logger.log_dict(val_log_dict, step, mode="eval")

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
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
