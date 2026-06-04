#!/usr/bin/env python3
# Closed-loop inference of pi05-D (and ablations) against unitree_sim_isaaclab.
#
# Differences vs inference_realtime_pi05d.py:
#   - cameras come from sim shared memory (isaac_<name>_image_shm + isaac_<name>_depth_shm)
#     instead of ZMQ (which only the real robot publishes).
#   - pressure comes from rt/dex3/pressure (JSON String_) published by pressure_dds.py.
#   - body+hand state and action commands still go over DDS via UnitreeG1Dex3
#     (same protocol the sim already speaks).
#   - episode termination + success counting via rt/rewards_state.

from __future__ import annotations
import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "lerobot-ext"))
sys.path.insert(0, str(Path.home() / "Prometheus" / "shared" / "unitree_sim_isaaclab"))

from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402
import safetensors.torch as st  # noqa: E402

def _load_policy(checkpoint_dir, fusion_mode, device):
    """Load a pi05 checkpoint with optional depth/pressure injection."""
    policy = PI05Policy.from_pretrained(checkpoint_dir, strict=False).to(device).eval()
    if fusion_mode == "none":
        return policy
    if fusion_mode == "depth_only":
        from train.pi05_depth_injector import inject_pi05_depth
        inject_pi05_depth(policy, device=device)
    elif fusion_mode == "full":
        from train.pi05_d_injector import inject_pi05_d
        inject_pi05_d(policy, device=device)
    else:
        raise ValueError(f"unknown fusion_mode={fusion_mode}")
    # reload safetensors so injected weights (pointnet/pressure_proj) come from disk
    from pathlib import Path
    sd_path = Path(checkpoint_dir) / "model.safetensors"
    sd = st.load_file(str(sd_path), device=str(device))
    policy.load_state_dict(sd, strict=False)
    return policy

# Usa a classe de lerobot-ext (tátil via get_observation + heartbeat anti-watchdog),
# igual ao inference_realtime_pi05d.py e ao pipeline de gravação. (lerobot-ext já está
# no sys.path acima, linha 26.) A cópia de lerobot/src NÃO expõe tátil nem heartbeat.
# CAVEAT Isaac: este script mira o unitree_sim_isaaclab; p/ ext conectar na ponte Isaac
# externa (e NÃO subir um MuJoCo in-process), o robot_cfg deve usar sim_backend="isaac".
from robot.unitree_g1.unitree_g1_dex3 import (  # noqa: E402
    UnitreeG1Dex3,
    UnitreeG1Dex3Config,
)
from tools.shared_memory_utils import MultiImageReader  # noqa: E402
from tools.depth_shm_utils import DepthImageReader  # noqa: E402

# DDS imports for pressure subscriber + reward subscriber
from unitree_sdk2py.core.channel import (  # noqa: E402
    ChannelFactoryInitialize, ChannelSubscriber,
)
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_  # noqa: E402

logger = logging.getLogger("pi05d_sim")


class _LatestMsg:
    """Thread-safe holder for last DDS String_ payload."""
    def __init__(self):
        self._data = None
        self._ts = 0.0

    def set(self, data: str):
        self._data = data
        self._ts = time.time()

    def get_json(self):
        return json.loads(self._data) if self._data else None

    def stale(self, max_age_sec: float) -> bool:
        return self._ts == 0 or (time.time() - self._ts) > max_age_sec


class SimBridge:
    """Reads sim observations (cameras, depth, pressure, rewards) over SHM+DDS."""

    def __init__(self, image_size=(480, 640)):
        self.image_size = image_size
        self.rgb_reader = MultiImageReader()
        self.depth_reader = DepthImageReader(image_names=("head", "left", "right"))
        self.pressure_msg = _LatestMsg()
        self.rewards_msg = _LatestMsg()
        self._pressure_sub = None
        self._rewards_sub = None

    def connect(self):
        # subscribers
        self._pressure_sub = ChannelSubscriber("rt/dex3/pressure", String_)
        self._pressure_sub.Init(lambda m: self.pressure_msg.set(m.data), 1)
        self._rewards_sub = ChannelSubscriber("rt/rewards_state", String_)
        self._rewards_sub.Init(lambda m: self.rewards_msg.set(m.data), 1)
        logger.info("DDS subscribers up: rt/dex3/pressure, rt/rewards_state")

    def get_rgb_head(self) -> np.ndarray:
        imgs = self.rgb_reader.read_images() or {}
        img = imgs.get("head")
        if img is None:
            return np.zeros((*self.image_size, 3), dtype=np.uint8)
        H, W = self.image_size
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return img

    def get_depth_head(self) -> np.ndarray:
        ds = self.depth_reader.read() or {}
        d = ds.get("head")
        if d is None:
            return np.zeros(self.image_size, dtype=np.float32)
        H, W = self.image_size
        if d.shape != (H, W):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
        # Pi05-D depth_to_pointcloud expects depth normalized [0, 1] meaning [0, 2 m]
        # Sim distance_to_image_plane is in meters → divide by 2 and clamp.
        d = np.clip(d.astype(np.float32) / 2.0, 0.0, 1.0)
        return d

    def get_pressure(self):
        msg = self.pressure_msg.get_json()
        if not msg:
            return np.zeros(33, dtype=np.float32), np.zeros(33, dtype=np.float32)
        left = np.asarray(msg.get("left", [0.0] * 33), dtype=np.float32)
        right = np.asarray(msg.get("right", [0.0] * 33), dtype=np.float32)
        return left, right

    def get_reward(self) -> float:
        msg = self.rewards_msg.get_json()
        if not msg:
            return 0.0
        r = msg.get("rewards", [0.0])
        if isinstance(r, list) and r:
            return float(r[0])
        return float(r)


def build_observation_batch(
    robot: UnitreeG1Dex3,
    bridge: SimBridge,
    task: str,
    device: torch.device,
    image_shape_hw=(480, 640),
) -> dict:
    obs = robot.get_observation()
    state_vec: list[float] = []
    for name, kind in robot.observation_features.items():
        if kind is float:
            state_vec.append(float(obs[name]))
    state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)

    H, W = image_shape_hw

    def rgb_to_tensor(img):
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img).to(device).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)

    def depth_to_tensor(d):
        # to 3-ch by tiling to match training convention (model treated depth as 3-ch image)
        d3 = np.stack([d, d, d], axis=2)
        return rgb_to_tensor(d3)

    rgb = bridge.get_rgb_head()
    depth = bridge.get_depth_head()
    left_p, right_p = bridge.get_pressure()

    return {
        "observation.state": state_tensor,
        "observation.images.head_camera": rgb_to_tensor(rgb),
        "observation.images.head_camera_depth": depth_to_tensor(depth),
        "observation.left_hand_pressure": torch.from_numpy(left_p).to(device).unsqueeze(0),
        "observation.right_hand_pressure": torch.from_numpy(right_p).to(device).unsqueeze(0),
        "task": task,
    }


ABLATIONS = (
    "none",
    "drop-rgb",
    "drop-depth",
    "drop-proprio",
    "drop-pressure",
    "occlude-rgb",
    "occlude-depth",
    "swap-depth-noise",
)


def _random_pixel_mask(hw, fraction, generator, device):
    H, W = hw
    n = H * W
    keep = torch.ones(n, device=device)
    n_drop = int(round(fraction * n))
    if n_drop > 0:
        idx = torch.randperm(n, generator=generator, device=device)[:n_drop]
        keep[idx] = 0.0
    return keep.view(1, 1, H, W)


def apply_pre_ablation(batch, kind, fraction, generator):
    """Mutates the raw (pre-preproc) batch. Used for sensor-level ablations."""
    if kind == "occlude-rgb":
        t = batch["observation.images.head_camera"]
        mask = _random_pixel_mask(t.shape[-2:], fraction, generator, t.device)
        batch["observation.images.head_camera"] = t * mask
    elif kind == "occlude-depth":
        t = batch["observation.images.head_camera_depth"]
        mask = _random_pixel_mask(t.shape[-2:], fraction, generator, t.device)
        batch["observation.images.head_camera_depth"] = t * mask
    elif kind == "swap-depth-noise":
        t = batch["observation.images.head_camera_depth"]
        batch["observation.images.head_camera_depth"] = torch.rand(
            t.shape, generator=generator, device=t.device, dtype=t.dtype
        )
    return batch


def apply_post_ablation(batch, kind):
    """Mutates the post-preproc (normalized) batch. Used for modality dropout —
    zeroing the normalized tensor corresponds to feeding the dataset mean,
    i.e. 'no information' relative to training distribution."""
    if kind == "drop-rgb":
        t = batch["observation.images.head_camera"]
        batch["observation.images.head_camera"] = torch.zeros_like(t)
    elif kind == "drop-depth":
        t = batch["observation.images.head_camera_depth"]
        batch["observation.images.head_camera_depth"] = torch.zeros_like(t)
    elif kind == "drop-proprio":
        t = batch["observation.state"]
        batch["observation.state"] = torch.zeros_like(t)
    elif kind == "drop-pressure":
        for k in ("observation.left_hand_pressure", "observation.right_hand_pressure"):
            if k in batch:
                batch[k] = torch.zeros_like(batch[k])
    return batch


def action_tensor_to_robot_action(action_vec: torch.Tensor, robot: UnitreeG1Dex3) -> dict:
    flat = action_vec.detach().cpu().numpy().astype(float).tolist()
    out: dict = {}
    for name, _ in robot.action_features.items():
        if not flat:
            break
        out[name] = flat.pop(0)
    return out


class GracefulKiller:
    def __init__(self):
        self.kill = False
        signal.signal(signal.SIGINT, self._h)
        signal.signal(signal.SIGTERM, self._h)
    def _h(self, *_): self.kill = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--fusion-mode", choices=["none","depth_only","full"], default="none")
    p.add_argument("--task", default="Pick up the cup")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--actions-per-chunk", type=int, default=50)
    p.add_argument("--episode-timeout", type=float, default=30.0)
    p.add_argument("--reward-success-threshold", type=float, default=0.8)
    p.add_argument("--reward-success-frames", type=int, default=5)
    p.add_argument("--max-episodes", type=int, default=20)
    p.add_argument("--dds-domain-id", type=int, default=1)
    p.add_argument("--results-out", type=str, default=None,
                   help="Path to write sim_eval JSON {success_rate, ...}")
    p.add_argument("--ablation", choices=ABLATIONS, default="none",
                   help="Inference-time input ablation. drop-* zeros the normalized "
                        "modality (post-preproc); occlude-* and swap-depth-noise "
                        "act on the raw sensor (pre-preproc).")
    p.add_argument("--occlude-fraction", type=float, default=0.5,
                   help="Fraction of pixels to mask for occlude-rgb / occlude-depth.")
    p.add_argument("--ablation-seed", type=int, default=0,
                   help="Seed for occlusion masks and depth-noise (reproducibility).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"loading pi05-D from {args.checkpoint}")
    policy = _load_policy(args.checkpoint, args.fusion_mode, device)
    preproc, postproc = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.checkpoint)
    logger.info("policy ready")

    # Init DDS factory once for both robot and bridge
    ChannelFactoryInitialize(args.dds_domain_id)

    robot_cfg = UnitreeG1Dex3Config(robot_ip="127.0.0.1", control_mode="upper_body", is_simulation=True)
    robot = UnitreeG1Dex3(robot_cfg)
    robot.connect()

    bridge = SimBridge()
    bridge.connect()

    killer = GracefulKiller()
    period = 1.0 / args.fps
    abl_gen = torch.Generator(device=device).manual_seed(args.ablation_seed)
    if args.ablation != "none":
        logger.info(f"ablation={args.ablation} fraction={args.occlude_fraction} seed={args.ablation_seed}")

    successes = 0
    total = 0
    ep_lengths = []
    ep_rewards = []

    for ep in range(args.max_episodes):
        if killer.kill:
            break
        logger.info(f"=== episode {ep} ===")
        ep_start = time.perf_counter()
        consecutive_above = 0
        success = False
        max_r = -1e9

        while not killer.kill and (time.perf_counter() - ep_start) < args.episode_timeout:
            t0 = time.perf_counter()
            batch = build_observation_batch(robot, bridge, args.task, device)
            batch = apply_pre_ablation(batch, args.ablation, args.occlude_fraction, abl_gen)
            batch = preproc(batch)
            batch = apply_post_ablation(batch, args.ablation)
            with torch.no_grad():
                chunk = policy.predict_action_chunk(batch)
            steps = min(args.actions_per_chunk, chunk.shape[1])
            for i in range(steps):
                if killer.kill: break
                a_norm = chunk[:, i, :]
                a_out = postproc(a_norm).squeeze(0)
                robot.send_action(action_tensor_to_robot_action(a_out, robot))
                # success check via reward
                r = bridge.get_reward()
                max_r = max(max_r, r)
                if r > args.reward_success_threshold:
                    consecutive_above += 1
                    if consecutive_above >= args.reward_success_frames:
                        success = True
                        break
                else:
                    consecutive_above = 0
                # FPS pacing
                rest = period - (time.perf_counter() - t0) / max(1, i + 1)
                if rest > 0: time.sleep(rest)
            if success: break

        ep_lengths.append(time.perf_counter() - ep_start)
        ep_rewards.append(max_r)
        total += 1
        if success: successes += 1
        logger.info(f"episode {ep}: success={success} max_reward={max_r:.3f} duration={ep_lengths[-1]:.1f}s")

    rate = successes / total if total else 0.0
    summary = {
        "checkpoint": args.checkpoint,
        "fusion_mode": args.fusion_mode,
        "ablation": args.ablation,
        "occlude_fraction": args.occlude_fraction if args.ablation.startswith("occlude") else None,
        "ablation_seed": args.ablation_seed if args.ablation != "none" else None,
        "n_episodes": total,
        "successes": successes,
        "success_rate": rate,
        "avg_episode_length_sec": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
        "avg_max_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
        "reward_threshold": args.reward_success_threshold,
        "reward_success_frames": args.reward_success_frames,
    }
    logger.info(f"FINAL: {summary}")
    if args.results_out:
        Path(args.results_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results_out).write_text(json.dumps(summary, indent=2))
        logger.info(f"wrote {args.results_out}")

    try: robot.disconnect()
    except Exception: pass


if __name__ == "__main__":
    main()
