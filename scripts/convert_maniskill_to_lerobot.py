"""Convert ManiSkill3 motion-planning demos to LeRobotDataset v3.0.

ManiSkill3 demos store only `actions` and `env_states` in trajectory.h5
(recorded with `obs_mode='none'`). To get RGB + depth + agent observations,
each trajectory has to be replayed in the env with `obs_mode='rgbd'`.

Depth handling:
  - ManiSkill returns depth as int16 millimeters in (H, W, 1).
  - We convert to float32 meters in (H, W) and store under
    `observation.depths.<camera>` (CALVIN-style key, avoids LeRobot's
    "image" stat-validator constraints). Pair this with `depth_scale=1.0`
    in pi05_depth YAML — the values are already in meters.

Usage:
  python scripts/convert_maniskill_to_lerobot.py \
      --task PickCube-v1 \
      --demo-h5 ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
      --repo-id local/maniskill_pickcube_v1 \
      --image-size 224 \
      --max-episodes 200
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401  (registers envs)
import numpy as np
import torch

from lerobot.datasets import compute_stats as _compute_stats_mod
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _patch_depth_stats_fixed():
    """Override `compute_episode_stats` so multi-dim float32 features (depth maps)
    get DETERMINISTIC fixed-shape `(1,)` scalar stats, regardless of episode-length
    or value variance. This sidesteps a LeRobot v0.4.4 issue where running these
    features through `RunningQuantileStats` plus an image feature in the same
    dataset produces parquet schemas that drift across episodes (`np.stack` of
    aggregated stats then fails with `all input arrays must have the same shape`).

    The pi05-depth policy reads depth pixels raw (in meters) and sends them
    through `depth_to_pointcloud` → PointNet. It does NOT consume per-pixel mean/std,
    so faking these stats with constants is safe for the use case.
    """
    import numpy as np

    _orig = _compute_stats_mod.compute_episode_stats

    def _fixed_stat(count_value: int, key: str) -> np.ndarray:
        if key == "count":
            return np.array([int(count_value)], dtype=np.int64)
        # mean=0.0, std=1.0, min/max/quantiles=0.0  → identity-ish normalization
        if key == "std":
            return np.array([1.0], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)

    def patched(episode_data, features, quantile_list=None):
        # Separate features: depth-like (float32, ndim>=2 shape) → fake stats;
        # everything else → stock compute_episode_stats.
        depth_keys = {
            k for k, ft in features.items()
            if ft["dtype"] == "float32" and len(ft.get("shape", ())) >= 2
        }
        non_depth_data = {k: v for k, v in episode_data.items() if k not in depth_keys}
        non_depth_features = {k: ft for k, ft in features.items() if k not in depth_keys}
        result = _orig(non_depth_data, non_depth_features, quantile_list)

        for k in depth_keys:
            arr = episode_data[k]
            count = arr.shape[0] if hasattr(arr, "shape") and arr.ndim > 0 else 1
            keys = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]
            result[k] = {sk: _fixed_stat(count, sk) for sk in keys}
        return result

    _compute_stats_mod.compute_episode_stats = patched
    # Also patch the captured-by-import-from reference inside lerobot_dataset
    import lerobot.datasets.lerobot_dataset as _ds_mod
    _ds_mod.compute_episode_stats = patched


_patch_depth_stats_fixed()

FPS = 20  # ManiSkill control_freq for these tasks

TASK_DESCRIPTIONS = {
    "PickCube-v1": "pick up the red cube",
    "StackCube-v1": "stack the red cube on top of the green cube",
    "PushCube-v1": "push the red cube to the goal",
    "PullCube-v1": "pull the red cube to the goal",
}


def make_features(image_size: int, state_dim: int, action_dim: int, camera_name: str,
                  use_videos: bool, with_depth: bool):
    # Honor the --no-videos flag: with use_videos=False we emit dtype="image"
    # (PNG frames) which lets training read without torchcodec — important
    # because torchcodec ships against PyTorch>=2.5 and ms3 has 2.4.1.
    feats = {
        f"observation.images.{camera_name}": {
            "dtype": "video" if use_videos else "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": {"axes": [f"state_{i}" for i in range(state_dim)]},
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": {"axes": [f"action_{i}" for i in range(action_dim)]},
        },
    }
    if with_depth:
        feats[f"observation.depths.{camera_name}"] = {
            "dtype": "float32",
            "shape": (image_size, image_size),
            "names": ["height", "width"],
        }
    return feats


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def squeeze_batch(t):
    arr = to_numpy(t)
    if arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def build_state(obs: dict) -> np.ndarray:
    qpos = squeeze_batch(obs["agent"]["qpos"]).astype(np.float32)
    qvel = squeeze_batch(obs["agent"]["qvel"]).astype(np.float32)
    return np.concatenate([qpos, qvel], axis=0)


def build_frame(obs: dict, action: np.ndarray, camera: str, image_size: int, task: str,
                with_depth: bool) -> dict:
    rgb = squeeze_batch(obs["sensor_data"][camera]["rgb"])
    if rgb.shape != (image_size, image_size, 3):
        raise ValueError(f"RGB shape mismatch: got {rgb.shape}, expected ({image_size},{image_size},3)")
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)

    frame = {
        f"observation.images.{camera}": rgb,
        "observation.state": build_state(obs),
        "action": np.asarray(action, dtype=np.float32),
        "task": task,
    }
    if with_depth:
        depth_mm = squeeze_batch(obs["sensor_data"][camera]["depth"])
        if depth_mm.ndim == 3:
            depth_mm = depth_mm[..., 0]
        depth_m = depth_mm.astype(np.float32) / 1000.0
        if depth_m.shape != (image_size, image_size):
            raise ValueError(f"Depth shape mismatch: got {depth_m.shape}, expected ({image_size},{image_size})")
        frame[f"observation.depths.{camera}"] = depth_m
    return frame


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="ManiSkill env id (e.g. PickCube-v1).")
    ap.add_argument("--demo-h5", required=True, type=Path)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--camera", default="base_camera")
    ap.add_argument("--control-mode", default="pd_joint_pos")
    ap.add_argument("--keep-failures", action="store_true",
                    help="Keep episodes that did not end in success (default: filter out).")
    ap.add_argument("--no-videos", action="store_true",
                    help="Store RGB as PNG frames instead of mp4. Avoids torchcodec dep.")
    ap.add_argument("--no-depth", action="store_true",
                    help="Skip depth feature entirely (workaround for a LeRobot stats-schema bug "
                         "when depth + image features coexist; train pi05 vanilla first, fix later).")
    return ap.parse_args()


def main():
    args = parse_args()
    args.demo_h5 = args.demo_h5.expanduser()

    task_desc = TASK_DESCRIPTIONS.get(args.task)
    if task_desc is None:
        task_desc = args.task.replace("-v1", "").replace("_", " ").lower()
        logging.warning(f"No description for {args.task}, using '{task_desc}'")

    meta_path = args.demo_h5.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    episodes = meta["episodes"]
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    logging.info(f"Replaying {len(episodes)} episodes from {args.demo_h5.name}")

    env = gym.make(
        args.task,
        obs_mode="rgbd",
        control_mode=args.control_mode,
        sensor_configs={"width": args.image_size, "height": args.image_size},
    )

    obs0, _ = env.reset(seed=0)
    state_dim = build_state(obs0).shape[0]
    action_dim = env.action_space.shape[0]
    logging.info(f"state_dim={state_dim} action_dim={action_dim} image_size={args.image_size}")

    use_videos = not args.no_videos
    with_depth = not args.no_depth
    features = make_features(args.image_size, state_dim, action_dim, args.camera,
                             use_videos, with_depth)
    robot_type = type(env.unwrapped.agent).__name__.lower()

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=features,
        robot_type=robot_type,
        root=args.root,
        use_videos=use_videos,
        image_writer_processes=2,
        image_writer_threads=8,
    )

    saved = 0
    skipped = 0
    with h5py.File(args.demo_h5, "r") as h5:
        for ep_meta in episodes:
            ep_id = ep_meta["episode_id"]
            seed = ep_meta["episode_seed"]

            if not args.keep_failures and not ep_meta.get("success", True):
                skipped += 1
                continue

            traj_key = f"traj_{ep_id}"
            if traj_key not in h5:
                logging.warning(f"{traj_key} missing in h5, skipping")
                skipped += 1
                continue

            actions = h5[traj_key]["actions"][:]
            obs, _ = env.reset(seed=seed, options=ep_meta.get("reset_kwargs", {}).get("options", {}))

            for action in actions:
                frame = build_frame(obs, action, args.camera, args.image_size, task_desc, with_depth)
                dataset.add_frame(frame)
                obs, _, terminated, truncated, _ = env.step(action)
                done_t = bool(to_numpy(terminated).reshape(-1)[0]) if hasattr(terminated, "shape") else bool(terminated)
                done_r = bool(to_numpy(truncated).reshape(-1)[0]) if hasattr(truncated, "shape") else bool(truncated)
                if done_t or done_r:
                    break

            dataset.save_episode()
            saved += 1
            if saved % 10 == 0:
                logging.info(f"  saved {saved}/{len(episodes)} (skipped {skipped})")

    env.close()
    dataset.finalize()
    logging.info(f"Done. saved={saved} skipped={skipped}. Dataset at: {dataset.root}")


if __name__ == "__main__":
    main()
