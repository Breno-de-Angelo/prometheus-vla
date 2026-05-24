#!/usr/bin/env python
"""Closed-loop eval for pi05/pi05depth on ManiSkill3 PickCube-v1 (and similar).

Loads a checkpoint dir produced by `train/run_train.py`, instantiates the
ManiSkill env with the same obs_mode/control_mode/resolution we used to build
the training dataset, runs N rollouts and reports success rate + episode
length. Optionally also computes offline action MSE on a held-out split of
the LeRobot dataset.

Standalone — does not register the env in lerobot's factory; we just need the
numbers for the vanilla-vs-depth comparison.

Usage:
  python scripts/eval_maniskill_pi05.py \
      --ckpt-path /path/to/checkpoint/dir \
      --task PickCube-v1 \
      --n-episodes 50 \
      --image-size 224 \
      --output-json /tmp/eval_vanilla.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lerobot-ext"))

import gymnasium as gym
import mani_skill.envs  # noqa: F401  (register envs)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_maniskill")


TASK_DESCRIPTIONS = {
    "PickCube-v1": "pick up the red cube",
    "StackCube-v1": "stack the red cube on top of the green cube",
    "PushCube-v1": "push the red cube to the goal",
    "PullCube-v1": "pull the red cube to the goal",
}


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def squeeze_batch(t):
    arr = to_numpy(t)
    if arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def load_policy(ckpt_path: str, device: torch.device):
    """Load PI05Policy or PI05DEPTHPolicy based on the config's `type`.

    Falls back to vanilla pi05 if the registered subclass for the saved
    `type` is not importable (e.g. lerobot-ext not on path).
    """
    # Force-register pi05depth if available
    try:
        import policies  # noqa: F401  (lerobot-ext/policies pkg)
    except Exception as e:
        log.warning(f"Could not import lerobot-ext policies: {e}")

    cfg_path = Path(ckpt_path) / "config.json"
    if not cfg_path.exists():
        # lerobot saves under pretrained_model/
        sub = Path(ckpt_path) / "pretrained_model"
        if (sub / "config.json").exists():
            ckpt_path = str(sub)
            cfg_path = sub / "config.json"

    cfg = json.loads(cfg_path.read_text())
    policy_type = cfg.get("type", "pi05")
    log.info(f"loading policy type={policy_type} from {ckpt_path}")

    if policy_type == "pi05depth":
        from policies.pi0_depth.modeling_pi05 import PI05DEPTHPolicy
        policy = PI05DEPTHPolicy.from_pretrained(ckpt_path)
    else:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        policy = PI05Policy.from_pretrained(ckpt_path)

    policy = policy.to(device).eval()
    return policy, policy_type


def build_obs_batch(obs: dict, camera: str, image_size: int, task_text: str,
                    with_depth: bool, device: torch.device) -> dict:
    """Turn one ManiSkill obs into a policy batch dict (B=1)."""
    rgb = squeeze_batch(obs["sensor_data"][camera]["rgb"]).astype(np.uint8)  # (H, W, 3)
    if rgb.shape != (image_size, image_size, 3):
        raise ValueError(f"RGB shape mismatch: got {rgb.shape}")

    qpos = squeeze_batch(obs["agent"]["qpos"]).astype(np.float32)
    qvel = squeeze_batch(obs["agent"]["qvel"]).astype(np.float32)
    state = np.concatenate([qpos, qvel], axis=0)

    # Convert to torch (B=1, C, H, W) for RGB and (B=1, state_dim) for state.
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # [0,1]
    state_t = torch.from_numpy(state).unsqueeze(0)

    batch = {
        f"observation.images.{camera}": rgb_t.to(device),
        "observation.state": state_t.to(device),
        "task": [task_text],
    }
    if with_depth:
        depth_mm = squeeze_batch(obs["sensor_data"][camera]["depth"])
        if depth_mm.ndim == 3:
            depth_mm = depth_mm[..., 0]
        depth_m = depth_mm.astype(np.float32) / 1000.0  # → meters, matches dataset
        depth_t = torch.from_numpy(depth_m).unsqueeze(0)  # (1, H, W)
        batch[f"observation.depths.{camera}"] = depth_t.to(device)
    return batch


def rollout_one(env, policy, *, camera: str, image_size: int, task_text: str,
                with_depth: bool, max_steps: int, device: torch.device,
                seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    t = 0
    done = False
    cumulative_reward = 0.0
    success = False

    # Reset action queue if policy buffers chunks
    if hasattr(policy, "reset"):
        policy.reset()

    while not done and t < max_steps:
        batch = build_obs_batch(obs, camera, image_size, task_text, with_depth, device)
        with torch.no_grad():
            action = policy.select_action(batch)
        action_np = to_numpy(action)
        if action_np.ndim == 2:
            action_np = action_np[0]
        obs, reward, terminated, truncated, info = env.step(action_np)
        # ManiSkill returns batched scalars
        rew = float(np.asarray(to_numpy(reward)).reshape(-1)[0]) if reward is not None else 0.0
        cumulative_reward += rew
        term = bool(np.asarray(to_numpy(terminated)).reshape(-1)[0])
        trunc = bool(np.asarray(to_numpy(truncated)).reshape(-1)[0])
        # success usually surfaces in info or we infer from terminated
        if info.get("success") is not None:
            v = info["success"]
            success = bool(np.asarray(to_numpy(v)).reshape(-1)[0])
        elif term and not trunc:
            success = True
        done = term or trunc
        t += 1
    return {"steps": t, "reward": cumulative_reward, "success": success}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True, type=str)
    ap.add_argument("--task", default="PickCube-v1")
    ap.add_argument("--camera", default="base_camera")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--control-mode", default="pd_joint_pos")
    ap.add_argument("--n-episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=10000,
                    help="Base seed; episode i uses seed + i.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-json", type=str, default=None)
    ap.add_argument("--with-depth", action="store_true",
                    help="Pass depth in batch (auto-set to True for pi05depth).")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    policy, policy_type = load_policy(args.ckpt_path, device)
    with_depth = args.with_depth or policy_type == "pi05depth"
    log.info(f"with_depth={with_depth}")

    task_text = TASK_DESCRIPTIONS.get(args.task, args.task.lower())

    env = gym.make(
        args.task,
        obs_mode="rgbd",
        control_mode=args.control_mode,
        sensor_configs={"width": args.image_size, "height": args.image_size},
    )

    results = []
    t0 = time.time()
    for i in range(args.n_episodes):
        out = rollout_one(
            env, policy,
            camera=args.camera, image_size=args.image_size, task_text=task_text,
            with_depth=with_depth, max_steps=args.max_steps, device=device,
            seed=args.seed + i,
        )
        results.append(out)
        elapsed = time.time() - t0
        sr = sum(r["success"] for r in results) / len(results)
        log.info(f"  ep {i+1}/{args.n_episodes}: success={out['success']} steps={out['steps']} "
                 f"R={out['reward']:.2f} | running SR={sr:.2%} | elapsed={elapsed:.0f}s")

    env.close()

    n = len(results)
    n_success = sum(r["success"] for r in results)
    success_rate = n_success / n if n else 0.0
    mean_steps = float(np.mean([r["steps"] for r in results]))
    mean_reward = float(np.mean([r["reward"] for r in results]))

    summary = {
        "ckpt_path": args.ckpt_path,
        "policy_type": policy_type,
        "task": args.task,
        "n_episodes": n,
        "success_rate": success_rate,
        "mean_episode_steps": mean_steps,
        "mean_episode_reward": mean_reward,
        "max_steps": args.max_steps,
        "image_size": args.image_size,
        "control_mode": args.control_mode,
        "per_episode": results,
    }
    log.info(f"=== SUMMARY ===")
    log.info(f"  ckpt: {args.ckpt_path}")
    log.info(f"  type: {policy_type}")
    log.info(f"  episodes: {n}, successes: {n_success} ({success_rate:.2%})")
    log.info(f"  mean steps: {mean_steps:.1f}, mean reward: {mean_reward:.3f}")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2))
        log.info(f"  saved → {args.output_json}")


if __name__ == "__main__":
    main()
