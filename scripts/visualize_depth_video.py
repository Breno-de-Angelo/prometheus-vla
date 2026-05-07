"""Render depth feature of a converted CALVIN LeRobotDataset as MP4 for inspection.

Reads observation.depths.static (or .gripper) frame-by-frame from the local parquet,
normalizes to uint8, applies a colormap, writes an MP4. Doesn't go through
LeRobotDataset so it doesn't try to hit the HF Hub.

Usage:
  python scripts/visualize_depth_video.py \\
      --root ~/.cache/huggingface/lerobot/local/calvin_debug_lerobot_depth \\
      --feature observation.depths.static \\
      --out /tmp/calvin_debug_depth_static.mp4 \\
      --colormap turbo
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CV2_COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "magma": cv2.COLORMAP_MAGMA,
    "gray": None,
}


def normalize_depth(depth: np.ndarray, vmin: float, vmax: float, invert: bool = True) -> np.ndarray:
    """Map depth(m) → uint8 [0,255].

    invert=True (default) follows the RealSense/Kinect/RViz convention: near = hot
    color (high uint8), far = cold color (low uint8). With cv2.COLORMAP_TURBO this
    yields red/orange for near and blue for far. Set invert=False to keep the raw
    "far = high value" mapping.
    """
    clipped = np.clip(depth, vmin, vmax)
    norm = (clipped - vmin) / max(vmax - vmin, 1e-6)
    if invert:
        norm = 1.0 - norm
    return (norm * 255.0).astype(np.uint8)


def load_depth_array(parquet_path: Path, feature: str) -> np.ndarray:
    """Load depth column as (N, H, W) float32 array from parquet.

    LeRobot stores 2D arrays as list-of-list-of-float in parquet, so each row
    comes back as ndarray(dtype=object) of length H, with each element a
    1D float32 array of length W. We restack them.
    """
    df = pd.read_parquet(parquet_path, columns=[feature])
    return np.stack(
        [np.stack([np.asarray(row_v, dtype=np.float32) for row_v in row]) for row in df[feature]]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--feature", type=str, default="observation.depths.static")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--colormap", choices=list(CV2_COLORMAPS), default="turbo")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--vmin", type=float, default=None, help="Min depth (m). Default: 1st-percentile.")
    parser.add_argument("--vmax", type=float, default=None, help="Max depth (m). Default: 99th-percentile.")
    parser.add_argument(
        "--no-invert",
        dest="invert",
        action="store_false",
        help="Disable color inversion. Default: invert (near=hot, far=cold à la RealSense).",
    )
    parser.set_defaults(invert=True)
    args = parser.parse_args()

    parquet_paths = sorted((args.root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files under {args.root / 'data'}")

    print(f"Loading depth from {len(parquet_paths)} parquet file(s)…")
    arrays = [load_depth_array(p, args.feature) for p in parquet_paths]
    depth = np.concatenate(arrays, axis=0)
    n_frames, h, w = depth.shape
    print(f"  shape: ({n_frames}, {h}, {w}), dtype={depth.dtype}")

    if args.vmin is None:
        args.vmin = float(np.percentile(depth, 1))
    if args.vmax is None:
        args.vmax = float(np.percentile(depth, 99))
    print(f"  depth range used for colormap: vmin={args.vmin:.3f} m, vmax={args.vmax:.3f} m")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, args.fps, (w, h), isColor=True)

    for i in range(n_frames):
        gray = normalize_depth(depth[i], args.vmin, args.vmax, invert=args.invert)
        if args.colormap == "gray":
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            frame = cv2.applyColorMap(gray, CV2_COLORMAPS[args.colormap])
        writer.write(frame)
        if i % 100 == 0:
            print(f"  frame {i}/{n_frames}")

    writer.release()
    print(f"Done: {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
