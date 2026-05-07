"""Convert CALVIN dataset (NPZ frames + lang_annotations boundaries) to LeRobot format.

CALVIN ships as one NPZ per frame, with language-task boundaries stored separately.
LeRobot expects one episode = one task instance. We treat each lang_annotation
range (info.indx[i] = (start, end)) as one LeRobot episode.

Features kept:
  - observation.images.rgb_static   (200, 200, 3) uint8 → video
  - observation.images.rgb_gripper  (84, 84, 3)   uint8 → video
  - observation.depths.static       (200, 200)    float32 → array (full precision)
  - observation.depths.gripper      (84, 84)      float32 → array
  - observation.state               (15,)         float32 (robot_obs)
  - observation.scene_state         (24,)         float32 (scene_obs)
  - action                          (7,)          float32 (absolute actions)
  - action.rel                      (7,)          float32 (relative actions)
  - task                            (string)      from lang_annotations.language.ann

NOTE on depth key naming: LeRobot's stat validator (compute_stats.py:546) treats
any feature key containing the substring "image" as an image and enforces shape
(3,1,1). To keep depth as a plain float32 array (which it is — it's not an RGB
image), we use the key prefix `observation.depths.*` (note: NOT "depth_images").

Dropped (intentionally):
  - rgb_tactile (160,120,6) and depth_tactile (160,120,2) — non-canonical tactile,
    not used for pi05_d arch validation (cup3 has no tactile equivalent).

Usage:
  python convert_calvin_to_lerobot.py \\
      --raw-dir /home/hercules/Prometheus/calvin/dataset/calvin_debug_dataset \\
      --repo-id local/calvin_debug_lerobot_depth \\
      --root /home/hercules/.cache/huggingface/lerobot/local/calvin_debug_lerobot_depth
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CALVIN_FPS = 30  # CALVIN teleop runs ~30 Hz
ROBOT_TYPE = "franka_panda"

CALVIN_FEATURES = {
    "observation.images.rgb_static": {
        "dtype": "video",
        "shape": (200, 200, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.rgb_gripper": {
        "dtype": "video",
        "shape": (84, 84, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.depths.static": {
        "dtype": "float32",
        "shape": (200, 200),
        "names": ["height", "width"],
    },
    "observation.depths.gripper": {
        "dtype": "float32",
        "shape": (84, 84),
        "names": ["height", "width"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (15,),
        "names": {"axes": [f"robot_obs_{i}" for i in range(15)]},
    },
    "observation.scene_state": {
        "dtype": "float32",
        "shape": (24,),
        "names": {"axes": [f"scene_obs_{i}" for i in range(24)]},
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"axes": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
    },
    "action.rel": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"axes": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
    },
}


def load_lang_annotations(split_dir: Path) -> dict:
    """Returns dict with keys: ann (list[str]), task (list[str]), indx (list[(start,end)])."""
    ann_path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(ann_path, allow_pickle=True).item()
    return {
        "ann": ann["language"]["ann"],
        "task": ann["language"]["task"],
        "indx": ann["info"]["indx"],
    }


def episode_iterator(split_dir: Path, lang: dict):
    """Yield (episode_idx, lang_instruction, frames) for each lang annotation range."""
    for ep_idx, ((start_id, end_id), instruction) in enumerate(zip(lang["indx"], lang["ann"], strict=True)):
        frames = []
        for frame_id in range(start_id, end_id + 1):
            npz_path = split_dir / f"episode_{frame_id:07d}.npz"
            if not npz_path.exists():
                logging.warning(f"Missing NPZ {npz_path}, skipping frame")
                continue
            data = np.load(npz_path, allow_pickle=True)
            frames.append({k: data[k] for k in data.files})
        if not frames:
            logging.warning(f"Episode {ep_idx} has no frames, skipping")
            continue
        yield ep_idx, instruction, frames


def npz_frame_to_lerobot(npz: dict, instruction: str) -> dict:
    """Map one NPZ frame to LeRobot frame dict."""
    return {
        "observation.images.rgb_static": npz["rgb_static"].astype(np.uint8),
        "observation.images.rgb_gripper": npz["rgb_gripper"].astype(np.uint8),
        "observation.depths.static": npz["depth_static"].astype(np.float32),
        "observation.depths.gripper": npz["depth_gripper"].astype(np.float32),
        "observation.state": npz["robot_obs"].astype(np.float32),
        "observation.scene_state": npz["scene_obs"].astype(np.float32),
        "action": npz["actions"].astype(np.float32),
        "action.rel": npz["rel_actions"].astype(np.float32),
        "task": instruction,
    }


def convert(raw_dir: Path, repo_id: str, root: Path | None, splits: list[str], force: bool = False):
    """Convert CALVIN raw dir → LeRobotDataset.

    raw_dir/training/* and raw_dir/validation/* are concatenated as one dataset
    (LeRobot doesn't have train/val splits per-se; episode_index orders them).
    """
    if root is None:
        root = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    root = Path(root)

    if root.exists() and any(root.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output dir {root} already exists and is non-empty. Delete it first, "
                f"pick a new --root, or pass --force to remove it automatically."
            )
        import shutil
        logging.warning(f"--force given: removing existing {root}")
        shutil.rmtree(root)

    logging.info(f"Creating LeRobotDataset at {root}")
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=ROBOT_TYPE,
        fps=CALVIN_FPS,
        features=CALVIN_FEATURES,
        root=root,
        use_videos=True,
        # Paraleliza video encoding sem usar shard+aggregate (zero risco de race
        # nas metadatas globais). Em ABC com 1.79M frames, isso é o gargalo
        # principal. processes=2 + threads=8 dá ~2-3x speedup empiricamente
        # (ffmpeg interno usa multiplos threads também).
        image_writer_processes=2,
        image_writer_threads=8,
        # batch_encoding_size=1 (default) — encoda imediatamente após cada
        # save_episode. Tentei batch=10 e o flush no debug com 9 episodes
        # ficou pendente, falhando o sanity check. Mantém=1 por segurança.
    )

    total_episodes = 0
    total_frames = 0
    for split in splits:
        split_dir = raw_dir / split
        if not split_dir.exists():
            logging.warning(f"Split {split} not found at {split_dir}, skipping")
            continue
        lang = load_lang_annotations(split_dir)
        n_eps = len(lang["indx"])
        logging.info(f"[{split}] {n_eps} episodes (lang_annotations)")

        for ep_idx, instruction, frames in episode_iterator(split_dir, lang):
            for npz in frames:
                ds.add_frame(npz_frame_to_lerobot(npz, instruction))
            ds.save_episode()
            total_episodes += 1
            total_frames += len(frames)
            if ep_idx % 50 == 0:
                logging.info(f"[{split}] ep {ep_idx}/{n_eps} done ({len(frames)} frames)")

    ds.finalize()
    logging.info(f"Done: {total_episodes} episodes, {total_frames} frames written to {root}")

    # Sanity: verify each video_key has exactly one MP4 at the canonical path.
    # Catches the path-nesting bug seen when an early conversion attempt failed
    # mid-encoding and a later run picked up stale temp files.
    video_root = root / "videos"
    expected = []
    for vid_key in ("observation.images.rgb_static", "observation.images.rgb_gripper"):
        target = video_root / vid_key / "chunk-000" / "file-000.mp4"
        if not target.exists():
            raise FileNotFoundError(f"Expected video missing: {target}")
        expected.append(target)
    actual = list(video_root.rglob("*.mp4"))
    extras = set(actual) - set(expected)
    if extras:
        raise RuntimeError(
            f"Unexpected MP4 files outside canonical paths (likely path-nesting bug):\n  "
            + "\n  ".join(str(p) for p in extras)
        )
    logging.info("Post-convert sanity check passed: video paths are clean.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["training"],
        help="Which raw subdirs to include (default: training only). Use 'training validation' to merge both.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If --root already exists and is non-empty, remove it before converting.",
    )
    args = parser.parse_args()
    convert(args.raw_dir, args.repo_id, args.root, args.splits, force=args.force)


if __name__ == "__main__":
    main()
