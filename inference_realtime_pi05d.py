#!/usr/bin/env python3
"""Real-time inference loop for the pi05-D policy on the Unitree G1 Dex3.

Replaces the policy_server + robot_client async-gRPC pair, which cannot run a
pi05-D model because:
  1. policy_server instantiates a vanilla PI05Policy and never calls
     inject_pi05_d(), so the pointnet/pressure_proj weights load but stay
     disconnected from the forward pass.
  2. the vanilla pipeline has no slot for the custom depth/pressure tokens the
     model was trained with.

What this script does:
  - Connects to the robot through the ZMQ bridge (run_g1_server.py +
    realsense_server.py running on the robot).
  - Loads the checkpoint via load_pi05_d() (from lerobot-ext), which runs
    inject_pi05_d() so the extra prefix tokens are active.
  - Runs a synchronous loop: observation → preprocess → predict_action_chunk →
    postprocess → execute chunk at the specified FPS → repeat.

Usage (after the robot-side services are up):
    python inference_realtime_pi05d.py \
        --checkpoint /home/hercules/prometheus-vla/train/output/pi05/checkpoints/best/pretrained_model \
        --robot-ip 10.9.8.73 \
        --task "Pick up the cup" \
        --fps 30 \
        --actions-per-chunk 50
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime

faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)
from pathlib import Path

import h5py
import numpy as np
import torch
import cv2

# Import lerobot-ext modules for the pi05-D loader. lerobot-ext lives beside the
# main prometheus-vla repo but isn't a package, so extend sys.path by convention.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "lerobot-ext"))

from train.inference_pi05_d import load_pi05_d  # noqa: E402  (needs sys.path first)
from train.pi05_depth_injector import inject_pi05_depth  # noqa: E402  (depth-only variant)
import safetensors.torch as st  # noqa: E402  (for raw weight load on vanilla/droid)
from lerobot.cameras.zmq.camera_zmq import ZMQCamera  # noqa: E402
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402
from lerobot.robots.unitree_g1.unitree_g1_dex3 import (  # noqa: E402
    UnitreeG1Dex3,
    UnitreeG1Dex3Config,
)

logger = logging.getLogger("pi05d_runtime")


# Fusion-mode controls which extra-prefix injection (if any) is applied to the
# pi05 policy. The trained checkpoint must match the mode chosen here:
#   - "none"        → vanilla / droid checkpoints (no PointNet, no pressure_proj)
#   - "depth_only"  → checkpoints trained with inject_pi05_depth (PointNet only)
#   - "full"        → checkpoints trained with inject_pi05_d  (PointNet + tactile)
FUSION_MODES = ("none", "depth_only", "full")


class RunLogger:
    """Per-chunk HDF5 + JPEG/PNG dump for offline analysis of a robot run.

    Writes one row per chunk into ``run.h5`` plus aligned RGB/depth frames under
    ``frames/``. All datasets are appendable (chunked) so files stay small if the
    run is killed early.
    """

    def __init__(
        self,
        out_dir: Path,
        chunk_size: int,
        action_dim: int,
        state_dim: int,
        pressure_dim: int = 33,
        save_frames: bool = True,
        model_chunk_size: int | None = None,
    ):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.dir / "frames"
        if save_frames:
            self.frames_dir.mkdir(exist_ok=True)
        self.save_frames = save_frames

        # `model_chunk_size` is the full chunk size predicted by the policy
        # (usually 50). `chunk_size` (= --actions-per-chunk CLI) is how many of
        # those we actually execute before re-planning, which can be smaller.
        # The HDF5 datasets keep both at their natural shapes so a user lowering
        # --actions-per-chunk to 10 doesn't break logging.
        self.model_chunk_size = model_chunk_size or chunk_size
        self.exec_chunk_size = chunk_size

        self.h5 = h5py.File(self.dir / "run.h5", "w")
        self._mk_ds("t_obs", (0,), (None,), "f8")
        self._mk_ds("t_predict", (0,), (None,), "f8")
        self._mk_ds("t_send_start", (0,), (None,), "f8")
        self._mk_ds("t_send_end", (0,), (None,), "f8")
        self._mk_ds("obs_state", (0, state_dim), (None, state_dim), "f4")
        self._mk_ds("obs_left_pressure", (0, pressure_dim), (None, pressure_dim), "f4")
        self._mk_ds("obs_right_pressure", (0, pressure_dim), (None, pressure_dim), "f4")
        self._mk_ds("action_chunk", (0, self.model_chunk_size, action_dim), (None, self.model_chunk_size, action_dim), "f4")
        self._mk_ds("sent_actions", (0, self.exec_chunk_size, action_dim), (None, self.exec_chunk_size, action_dim), "f4")
        self._mk_ds("obs_state_post", (0, state_dim), (None, state_dim), "f4")
        self.idx = 0

    def _mk_ds(self, name, shape, maxshape, dtype):
        self.h5.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype, chunks=True, compression="gzip", compression_opts=4)

    @staticmethod
    def _append(ds, value):
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = value

    def log_chunk(
        self,
        t_obs: float,
        t_predict: float,
        t_send_start: float,
        t_send_end: float,
        obs_state: np.ndarray,
        obs_left_pressure: np.ndarray,
        obs_right_pressure: np.ndarray,
        action_chunk: np.ndarray,
        sent_actions: np.ndarray,
        obs_state_post: np.ndarray,
        rgb: np.ndarray | None = None,
        depth: np.ndarray | None = None,
    ):
        i = self.idx
        self._append(self.h5["t_obs"], t_obs)
        self._append(self.h5["t_predict"], t_predict)
        self._append(self.h5["t_send_start"], t_send_start)
        self._append(self.h5["t_send_end"], t_send_end)
        self._append(self.h5["obs_state"], obs_state)
        self._append(self.h5["obs_left_pressure"], obs_left_pressure)
        self._append(self.h5["obs_right_pressure"], obs_right_pressure)
        self._append(self.h5["action_chunk"], action_chunk)
        self._append(self.h5["sent_actions"], sent_actions)
        self._append(self.h5["obs_state_post"], obs_state_post)
        if self.save_frames:
            if rgb is not None:
                cv2.imwrite(str(self.frames_dir / f"rgb_{i:04d}.jpg"),
                            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
            if depth is not None:
                cv2.imwrite(str(self.frames_dir / f"depth_{i:04d}.png"), depth)
        self.idx += 1
        if i % 5 == 0:
            self.h5.flush()

    def close(self):
        self.h5.flush()
        self.h5.close()


class HighFreqStateLogger:
    """Continuous high-frequency state + last-action logger.

    Runs in a daemon thread that polls ``robot.get_observation()`` at a fixed
    rate (default 50 Hz) and writes every sample into ``state.h5``. Unlike
    RunLogger, which records one row per inference chunk, this captures
    *between* chunks and *during* the per-action sleeps — so the user can see
    real state vs commanded action at fine granularity end-to-end.

    Datasets in ``state.h5``:
      - t                  (N,)             float64  wall timestamp
      - state              (N, state_dim)   float32  joint positions (the same
                                                     vector RunLogger logs as
                                                     ``obs_state``)
      - left_pressure      (N, 33)          float32
      - right_pressure     (N, 33)          float32
      - last_action_sent   (N, action_dim)  float32  snapshot of the most recent
                                                     action handed to
                                                     robot.send_action()
    """

    def __init__(
        self,
        out_dir: Path,
        state_dim: int,
        action_dim: int,
        hz: float = 50.0,
        pressure_dim: int = 33,
    ):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pressure_dim = pressure_dim
        self.period = 1.0 / float(hz)
        self.hz = float(hz)

        self.h5 = h5py.File(self.dir / "state.h5", "w")
        self._mk_ds("t", (0,), (None,), "f8")
        self._mk_ds("state", (0, state_dim), (None, state_dim), "f4")
        self._mk_ds("left_pressure", (0, pressure_dim), (None, pressure_dim), "f4")
        self._mk_ds("right_pressure", (0, pressure_dim), (None, pressure_dim), "f4")
        self._mk_ds("last_action_sent", (0, action_dim), (None, action_dim), "f4")

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_action = np.zeros((action_dim,), dtype=np.float32)
        self._thread: threading.Thread | None = None
        self._robot: UnitreeG1Dex3 | None = None
        self._sample_count = 0

    def _mk_ds(self, name, shape, maxshape, dtype):
        self.h5.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=dtype,
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )

    @staticmethod
    def _append(ds, value):
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = value

    def update_last_action(self, action_np: np.ndarray) -> None:
        """Thread-safe snapshot of the most recent action sent to the robot."""
        with self._lock:
            self._last_action = np.asarray(action_np, dtype=np.float32).copy()

    def start(self, robot: UnitreeG1Dex3) -> None:
        if self._thread is not None:
            return
        self._robot = robot
        self._thread = threading.Thread(
            target=self._loop, name="HighFreqStateLogger", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        assert self._robot is not None
        robot = self._robot
        next_deadline = time.perf_counter()
        while not self._stop.is_set():
            t_wall = time.time()
            try:
                obs = robot.get_observation()
                state_vec = np.array(
                    [
                        float(obs[name])
                        for name, kind in robot.observation_features.items()
                        if kind is float
                    ],
                    dtype=np.float32,
                )
                left_p = np.asarray(
                    obs.get("left_hand_pressure", np.zeros(self.pressure_dim, dtype=np.float32)),
                    dtype=np.float32,
                )
                right_p = np.asarray(
                    obs.get("right_hand_pressure", np.zeros(self.pressure_dim, dtype=np.float32)),
                    dtype=np.float32,
                )
            except Exception as e:
                # If a poll fails (e.g. transient ZMQ hiccup), skip this tick but
                # keep the thread alive so we don't lose the rest of the run.
                logger.warning(f"hf_logger sample failed: {e}")
                next_deadline += self.period
                sleep_for = next_deadline - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                continue

            with self._lock:
                last_action = self._last_action.copy()

            try:
                self._append(self.h5["t"], t_wall)
                self._append(self.h5["state"], state_vec)
                self._append(self.h5["left_pressure"], left_p)
                self._append(self.h5["right_pressure"], right_p)
                self._append(self.h5["last_action_sent"], last_action)
                self._sample_count += 1
                # Flush occasionally so a hard kill doesn't lose everything.
                if self._sample_count % 250 == 0:
                    self.h5.flush()
            except Exception as e:
                logger.warning(f"hf_logger h5 append failed: {e}")

            # Pace the loop to ``hz`` using a monotonic deadline so jitter does
            # not accumulate. If we fall behind, catch up by skipping the sleep.
            next_deadline += self.period
            sleep_for = next_deadline - time.perf_counter()
            if sleep_for > 0:
                # Use stop.wait so shutdown is responsive even between samples.
                self._stop.wait(timeout=sleep_for)
            else:
                # Behind schedule: reset the deadline to "now" to avoid bursts.
                next_deadline = time.perf_counter()

    def stop_and_close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self.h5.flush()
            self.h5.close()
        except Exception as e:
            logger.warning(f"hf_logger h5 close raised: {e}")


def load_policy_for_mode(
    checkpoint_dir: str,
    device: torch.device,
    fusion_mode: str,
    depth_key: str,
    depth_scale: float,
):
    """Load a pi05 policy and run the right injection step (or none) for the
    requested ``fusion_mode``. Returns the prepared policy in eval() mode."""
    if fusion_mode not in FUSION_MODES:
        raise ValueError(f"fusion_mode must be one of {FUSION_MODES}, got {fusion_mode!r}")

    if fusion_mode == "full":
        # pi05-D: same path used by the offline smoke (load_pi05_d in lerobot-ext).
        return load_pi05_d(checkpoint_dir, device)

    if fusion_mode == "depth_only":
        # PointNet-only injection (no pressure). Loader pattern mirrors load_pi05_d
        # to make sure the injected weights are pulled from the safetensors file
        # AFTER inject_pi05_depth has created the matching submodules.
        policy = PI05Policy.from_pretrained(checkpoint_dir, strict=False)
        policy.to(device).eval()
        inject_pi05_depth(
            policy,
            device=device,
            depth_key=depth_key,
            depth_scale=depth_scale,
        )
        sd_path = Path(checkpoint_dir) / "model.safetensors"
        sd = st.load_file(str(sd_path), device=str(device))
        missing, unexpected = policy.load_state_dict(sd, strict=False)
        loaded_injected = [k for k in sd.keys() if "pointnet" in k]
        logger.info(
            f"depth_only: loaded {len(loaded_injected)} injected tensors; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        return policy

    # fusion_mode == "none": plain pi05 (vanilla, droid). No injection, no extra
    # tokens, no depth/pressure consumed from the batch.
    policy = PI05Policy.from_pretrained(checkpoint_dir)
    policy.to(device).eval()
    logger.info("fusion_mode=none: vanilla PI05Policy loaded (no depth/pressure injection)")
    return policy


def build_observation_batch(
    robot: UnitreeG1Dex3,
    depth_camera: ZMQCamera | None,
    task: str,
    device: torch.device,
    fusion_mode: str = "full",
    depth_key: str = "observation.images.head_camera_depth",
    image_shape_hw: tuple[int, int] = (480, 640),
) -> dict:
    """Convert a raw robot observation into the batched dict the policy expects.

    Matches the exact feature names in the trained checkpoint's config.json:
      - observation.state                 (28,)
      - observation.images.head_camera    (3, 480, 640) float32 in [0, 1]
      - observation.images.head_camera_depth (3, 480, 640)  (depth_only / full)
      - observation.left_hand_pressure    (33,)             (full only)
      - observation.right_hand_pressure   (33,)             (full only)
      - task                              (str, later tokenized by preprocessor)

    For ``fusion_mode="none"`` (vanilla / droid) the depth & pressure entries are
    skipped entirely; for ``fusion_mode="depth_only"`` only the depth entry is
    populated, and pressure keys are omitted.

    The driver emits the RGB camera under key ``cam_rgb_high`` (historic name
    used by the ACT policy). The depth stream is fetched out-of-band via
    ``depth_camera`` because the shared driver config can't be extended without
    breaking ACT.
    """
    obs = robot.get_observation()

    # State: body joints followed by hand joints, in the order used during training.
    # observation_features dict ordering is stable on Python 3.7+. pressure features
    # use tuple specs and are filtered out here (they go to their own top-level keys).
    state_vec: list[float] = []
    for name, kind in robot.observation_features.items():
        if kind is float:
            state_vec.append(float(obs[name]))
    state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)

    H, W = image_shape_hw

    def to_tensor(img: np.ndarray) -> torch.Tensor:
        if img is None:
            img = np.zeros((H, W, 3), dtype=np.uint8)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        # HWC uint8 -> CHW float [0,1]
        return torch.from_numpy(img).to(device).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)

    rgb_img = obs["cam_rgb_high"]

    batch: dict = {
        "observation.state": state_tensor,
        "observation.images.head_camera": to_tensor(rgb_img),
        "task": task,
    }

    if fusion_mode in ("depth_only", "full"):
        if depth_camera is None:
            raise RuntimeError(
                f"fusion_mode={fusion_mode!r} requires a connected depth camera; got None."
            )
        depth_img = depth_camera.async_read()
        batch[depth_key] = to_tensor(depth_img)

    if fusion_mode == "full":
        batch["observation.left_hand_pressure"] = torch.from_numpy(
            np.asarray(obs["left_hand_pressure"], dtype=np.float32)
        ).to(device).unsqueeze(0)
        batch["observation.right_hand_pressure"] = torch.from_numpy(
            np.asarray(obs["right_hand_pressure"], dtype=np.float32)
        ).to(device).unsqueeze(0)

    return batch


def _build_batch_from_raw(
    raw_obs: dict,
    depth_raw: np.ndarray | None,
    state_vec_np: np.ndarray,
    task: str,
    device: torch.device,
    fusion_mode: str,
    depth_key: str,
    image_shape_hw: tuple[int, int] = (480, 640),
) -> dict:
    """Same shape as build_observation_batch but reuses an already-fetched obs/depth.

    Avoids a second robot.get_observation()/depth_camera.async_read() so the
    timestamps logged for the chunk match the data actually fed to the policy.
    """
    H, W = image_shape_hw

    def to_tensor(img: np.ndarray | None) -> torch.Tensor:
        if img is None:
            img = np.zeros((H, W, 3), dtype=np.uint8)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img).to(device).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)

    state_tensor = torch.from_numpy(state_vec_np).to(device).unsqueeze(0)

    rgb_img = raw_obs.get("cam_rgb_high")
    batch: dict = {
        "observation.state": state_tensor,
        "observation.images.head_camera": to_tensor(rgb_img),
        "task": task,
    }
    if fusion_mode in ("depth_only", "full"):
        batch[depth_key] = to_tensor(depth_raw)
    if fusion_mode == "full":
        batch["observation.left_hand_pressure"] = torch.from_numpy(
            np.asarray(raw_obs.get("left_hand_pressure", np.zeros(33, dtype=np.float32)), dtype=np.float32)
        ).to(device).unsqueeze(0)
        batch["observation.right_hand_pressure"] = torch.from_numpy(
            np.asarray(raw_obs.get("right_hand_pressure", np.zeros(33, dtype=np.float32)), dtype=np.float32)
        ).to(device).unsqueeze(0)
    return batch


def action_tensor_to_robot_action(action_vec: torch.Tensor, robot: UnitreeG1Dex3) -> dict:
    """Convert a 28-dim action tensor into the dict send_action() expects."""
    action = action_vec.detach().cpu().numpy().astype(float).tolist()
    out: dict = {}
    for name, _ in robot.action_features.items():
        if not action:
            break
        out[name] = action.pop(0)
    return out


class GracefulKiller:
    """Ctrl+C handler that sets a flag so the main loop can exit cleanly."""

    def __init__(self):
        self.kill = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, *_):
        logger.warning("shutdown requested; finishing current chunk and stopping...")
        self.kill = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="pi05 / pi05-D pretrained_model directory")
    parser.add_argument("--robot-ip", default="10.9.8.73")
    parser.add_argument("--task", default="Pick up the cup")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--actions-per-chunk", type=int, default=50,
                        help="Number of actions to execute from each predicted chunk. "
                             "Training chunk_size is 50; <=50 is safe, >50 is invalid.")
    parser.add_argument("--control-mode", default="upper_body")
    parser.add_argument("--arm", default="G1_29")
    parser.add_argument(
        "--fusion-mode",
        choices=FUSION_MODES,
        default="full",
        help=(
            "Which extra-prefix injection to apply when loading the checkpoint. "
            "'none' = vanilla / droid pi05 (no PointNet, no pressure); "
            "'depth_only' = PointNet only (pi05-depth); "
            "'full' = PointNet + pressure (pi05-D, default)."
        ),
    )
    parser.add_argument(
        "--depth-key",
        default="observation.images.head_camera_depth",
        help="Batch key for the depth tensor (must match the trained checkpoint).",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=2.0,
        help=(
            "Multiplier applied inside the PointNet to obtain meters from the depth "
            "tensor. Matches the cup3 ZMQ hack (2.0). Ignored for fusion_mode=none."
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Log predicted actions but don't send to the robot.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-dir", default=None,
                        help="Directory to dump run.h5 + frames/. Default: auto-generated under results/robot_runs/.")
    parser.add_argument("--no-frames", action="store_true",
                        help="Skip saving RGB/depth frames (HDF5 still written).")
    parser.add_argument("--skip-reset-hands", action="store_true",
        help="Skip the pre-loop reset_hands() that opens both hands to q=0.")
    parser.add_argument("--reset-hands-duration", type=float, default=2.0,
        help="Seconds to interpolate from current pose to open pose at startup (default 2.0).")
    parser.add_argument("--soft-start-frames", type=int, default=30,
        help="Linearly blend the first N actions of chunk 0 from current state to model plan, "
             "to avoid jerk on initial pose transition (default 30 = 1 s at 30 FPS, 0 = disable).")
    parser.add_argument("--auto-analyze", action=argparse.BooleanOptionalAction, default=True,
                        help="Run tools/analyze_run.py on the run dir at shutdown.")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    logger.info(
        f"loading pi05 policy (fusion_mode={args.fusion_mode}); this loads ~7GB of weights..."
    )
    policy = load_policy_for_mode(
        checkpoint_dir=args.checkpoint,
        device=device,
        fusion_mode=args.fusion_mode,
        depth_key=args.depth_key,
        depth_scale=args.depth_scale,
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint
    )
    logger.info("policy ready")

    # Sanity check: a dead unnormalizer (std~0) silently kills entire action
    # channels. We can't recover at runtime — surface a loud warning so the
    # operator knows the run will be biased even before the robot moves.
    unnorm_path = Path(args.checkpoint) / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if unnorm_path.exists():
        try:
            unnorm_tensors = st.load_file(str(unnorm_path))
            std_key = next(
                (k for k in unnorm_tensors if "std" in k.lower() and "action" in k.lower()),
                None,
            )
            if std_key is not None:
                std_vec = unnorm_tensors[std_key].numpy()
                dead_idxs = [int(i) for i, v in enumerate(std_vec) if v < 1e-4]
                if dead_idxs:
                    logger.warning(
                        "######################################################"
                    )
                    logger.warning(
                        f"!! UNNORMALIZER has near-zero std in dims {dead_idxs} !!"
                    )
                    logger.warning(
                        "!! These channels will produce DEAD commands (always   !!"
                    )
                    logger.warning(
                        "!! equal to action.mean). Re-train or fix the postproc.!!"
                    )
                    logger.warning(
                        "######################################################"
                    )
        except Exception as e:
            logger.warning(f"unnormalizer sanity check failed: {e}")

    robot_cfg = UnitreeG1Dex3Config(robot_ip=args.robot_ip, control_mode=args.control_mode, is_simulation=False)
    robot = UnitreeG1Dex3(robot_cfg)
    robot.connect()
    logger.info(f"robot connected at {args.robot_ip}")

    depth_camera: ZMQCamera | None = None
    if args.fusion_mode in ("depth_only", "full"):
        depth_camera = ZMQCamera(
            ZMQCameraConfig(
                server_address=args.robot_ip,
                port=5555,
                camera_name="head_camera_depth",
                width=640,
                height=480,
            )
        )
        depth_camera.connect()
        logger.info(
            "depth camera connected at {}:5555/head_camera_depth".format(args.robot_ip)
        )
    else:
        logger.info("fusion_mode=none: skipping depth camera connection")

    killer = GracefulKiller()
    step_period = 1.0 / args.fps
    chunk_counter = 0

    # Resolve log dir (auto-generate if not provided).
    if args.log_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ckpt_name = Path(args.checkpoint).parts[-3] if "checkpoints" in args.checkpoint else Path(args.checkpoint).name
        log_dir = Path("/home/hercules/prometheus-vla-calvin/results/robot_runs") / f"run_{ts}_{ckpt_name}_{args.fusion_mode}"
    else:
        log_dir = Path(args.log_dir)
    logger.info(f"run log dir: {log_dir}")

    # Inspect feature dims off the policy config (28 for upper_body G1+Dex3).
    state_dim = sum(1 for _, k in robot.observation_features.items() if k is float)
    action_dim = len(robot.action_features)
    # The policy predicts `n_action_steps` (usually 50) per call regardless of
    # how many we actually execute (--actions-per-chunk). RunLogger needs both.
    model_chunk_size = getattr(policy.config, "n_action_steps", args.actions_per_chunk) or args.actions_per_chunk
    run_logger = RunLogger(
        out_dir=log_dir,
        chunk_size=args.actions_per_chunk,
        action_dim=action_dim,
        state_dim=state_dim,
        save_frames=not args.no_frames,
        model_chunk_size=model_chunk_size,
    )
    # Continuous 50 Hz state + last-action logger (separate state.h5).
    hf_logger = HighFreqStateLogger(
        out_dir=log_dir,
        state_dim=state_dim,
        action_dim=action_dim,
        hz=50,
    )
    hf_logger.start(robot)
    # Open both hands to q≈0 BEFORE the main loop. Aligns the physical hand pose
    # with the dataset's typical initial state and gives the policy's first chunk
    # a smooth ramp instead of a 1-frame jump from current pose to ±1.5 rad.
    if not args.dry_run and not args.skip_reset_hands:
        try:
            logger.info("resetting hands to q=0 (open) before policy starts...")
            robot.reset_hands(duration_s=args.reset_hands_duration)
        except Exception as e:
            logger.warning(f"reset_hands failed (continuing anyway): {e}")
    # Dump run metadata.
    with open(log_dir / "meta.json", "w") as f:
        json.dump({
            "started_at": datetime.now().isoformat(),
            "checkpoint": str(args.checkpoint),
            "fusion_mode": args.fusion_mode,
            "task": args.task,
            "robot_ip": args.robot_ip,
            "fps": args.fps,
            "actions_per_chunk": args.actions_per_chunk,
            "dry_run": bool(args.dry_run),
            "control_mode": args.control_mode,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }, f, indent=2)

    try:
        while not killer.kill:
            # 1) Fetch a fresh observation and predict a chunk.
            t_obs_start = time.perf_counter()
            t_obs_wall = time.time()
            raw_obs = robot.get_observation()
            depth_raw = depth_camera.async_read() if depth_camera is not None else None
            rgb_raw = raw_obs.get("cam_rgb_high")
            state_vec_np = np.array(
                [float(raw_obs[name]) for name, k in robot.observation_features.items() if k is float],
                dtype=np.float32,
            )
            left_pressure_np = np.asarray(
                raw_obs.get("left_hand_pressure", np.zeros(33, dtype=np.float32)),
                dtype=np.float32,
            )
            right_pressure_np = np.asarray(
                raw_obs.get("right_hand_pressure", np.zeros(33, dtype=np.float32)),
                dtype=np.float32,
            )

            # Reuse the captures to assemble the policy batch (avoids second robot read).
            batch = _build_batch_from_raw(
                raw_obs=raw_obs,
                depth_raw=depth_raw,
                state_vec_np=state_vec_np,
                task=args.task,
                device=device,
                fusion_mode=args.fusion_mode,
                depth_key=args.depth_key,
            )
            batch = preprocessor(batch)
            with torch.no_grad():
                action_chunk = policy.predict_action_chunk(batch)  # (1, chunk_size, action_dim)
            t_predict = time.perf_counter()
            t_inf = t_predict - t_obs_start
            logger.info(
                f"chunk {chunk_counter}: predicted in {t_inf*1000:.0f}ms "
                f"(shape {tuple(action_chunk.shape)})"
            )

            # 2) Execute the chunk at the policy FPS, accumulating sent actions.
            # SOFT-START: on the very first chunk only, the robot is at rest pose
            # but the model's first action is whatever the policy predicts —
            # potentially 1+ rad away. Sending it raw causes the arm to jerk. So
            # for the first ``soft_start_frames`` actions of chunk 0, linearly
            # blend from the captured initial state into the model's plan. This
            # does NOT modify the policy, the checkpoint, or the saved chunk —
            # it only affects what reaches the motor controller. After chunk 0,
            # the model's predictions already start near the current state
            # (because we just executed last chunk), so no blending is needed.
            steps_to_run = min(args.actions_per_chunk, action_chunk.shape[1])
            sent_actions_np = np.zeros((args.actions_per_chunk, action_dim), dtype=np.float32)
            do_soft_start = (chunk_counter == 0 and args.soft_start_frames > 0)
            soft_start_n = min(args.soft_start_frames, steps_to_run) if do_soft_start else 0
            if do_soft_start:
                logger.info(f"soft-start: blending first {soft_start_n} action(s) from current state to model plan")
            t_send_start = time.perf_counter()
            for i in range(steps_to_run):
                if killer.kill:
                    break
                loop_start = time.perf_counter()
                action_norm = action_chunk[:, i, :]
                action_out = postprocessor(action_norm).squeeze(0)
                action_out_np = action_out.detach().cpu().numpy().astype(np.float32)
                if soft_start_n > 0 and i < soft_start_n:
                    alpha = float(i + 1) / float(soft_start_n)
                    action_out_np = (1.0 - alpha) * state_vec_np + alpha * action_out_np
                    action_out = torch.from_numpy(action_out_np).to(action_out.device)
                sent_actions_np[i] = action_out_np
                hf_logger.update_last_action(action_out_np)
                if args.dry_run:
                    if i == 0:
                        logger.info(f"dry-run first action: {action_out_np.round(3).tolist()}")
                else:
                    robot.send_action(action_tensor_to_robot_action(action_out, robot))
                elapsed = time.perf_counter() - loop_start
                sleep_for = step_period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
            t_send_end = time.perf_counter()

            # 3) Capture post-chunk state (was the cmd executed?).
            raw_obs_post = robot.get_observation()
            state_vec_post = np.array(
                [float(raw_obs_post[name]) for name, k in robot.observation_features.items() if k is float],
                dtype=np.float32,
            )

            # 4) Log + diagnostic print for right-hand actions.
            run_logger.log_chunk(
                t_obs=t_obs_wall,
                t_predict=t_obs_wall + (t_predict - t_obs_start),
                t_send_start=t_obs_wall + (t_send_start - t_obs_start),
                t_send_end=t_obs_wall + (t_send_end - t_obs_start),
                obs_state=state_vec_np,
                obs_left_pressure=left_pressure_np,
                obs_right_pressure=right_pressure_np,
                action_chunk=action_chunk[0].detach().cpu().numpy().astype(np.float32),
                sent_actions=sent_actions_np,
                obs_state_post=state_vec_post,
                rgb=rgb_raw,
                depth=depth_raw,
            )
            # Right hand spans indices 21..27 (upper_body: 14 arm + 7 left + 7 right).
            rh = sent_actions_np[:, 21:28]
            print(
                f"chunk {chunk_counter:04d} | predict {t_inf*1000:5.0f}ms | "
                f"send {(t_send_end-t_send_start)*1000:6.0f}ms | "
                f"R-hand sent mean={rh.mean():+.3f} min={rh.min():+.3f} max={rh.max():+.3f} | "
                f"R-hand state pre={state_vec_np[21:28].mean():+.3f} post={state_vec_post[21:28].mean():+.3f}",
                flush=True,
            )
            chunk_counter += 1
    finally:
        logger.info("disconnecting robot")
        try:
            hf_logger.stop_and_close()
        except Exception as e:
            logger.warning(f"hf_logger close raised: {e}")
        try:
            run_logger.close()
        except Exception as e:
            logger.warning(f"run_logger close raised: {e}")
        if depth_camera is not None:
            try:
                depth_camera.disconnect()
            except Exception as e:
                logger.warning(f"depth disconnect raised: {e}")
        try:
            robot.disconnect()
        except Exception as e:
            logger.warning(f"disconnect raised: {e}")
        if args.auto_analyze:
            try:
                sys.path.insert(0, str(Path(__file__).parent / "tools"))
                from analyze_run import analyze  # noqa: E402
                a = analyze(log_dir)
                logger.info(
                    f"analysis written to {log_dir}/analysis.json + report.md  "
                    f"(status={a['diagnosis']['status']}, issues={a['diagnosis']['issues']})"
                )
            except Exception as e:
                logger.warning(f"auto-analyze failed: {e}")


if __name__ == "__main__":
    main()
