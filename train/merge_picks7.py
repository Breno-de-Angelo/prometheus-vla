#!/usr/bin/env python
"""Merge the 7 HWC-compatible Unitree G1_Dex3 single-pick datasets into one LeRobot
dataset for the pi05_depth_uni_lf cotraining pretrain (Option A).

Downloads each source fully (with retries), then aggregates into one dataset via
lerobot's aggregate_datasets(). Idempotent: skips if the merged dataset already exists.
Exit code 0 = success; non-zero = failure (orchestrator aborts and does NOT train).
"""
import os, sys, time, traceback
os.environ.setdefault("HF_HOME", "/data/huggingface-models")
os.environ.setdefault("HF_LEROBOT_HOME", "/data/huggingface-models/lerobot")
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.aggregate import aggregate_datasets

PICKS = ["PickBottle", "PickApple", "PickSnack", "PickGum", "PickTissue", "PickCharger", "PickDoll"]
REPO_IDS = [f"unitreerobotics/G1_Dex3_{p}_Dataset" for p in PICKS]
AGGR_REPO = "lewislf/G1_Dex3_picks7_uni"
AGGR_ROOT = Path("/data/huggingface-models/lerobot/lewislf/G1_Dex3_picks7_uni")


def ensure_quantiles() -> None:
    """aggregate_datasets() only writes min/max/mean/std, but pi05 normalizes STATE and ACTION
    with QUANTILES — without q01/q10/q50/q90/q99 the training dies in the normalizer. Compute
    them from the parquet for state+action only (VISUAL is IDENTITY for pi05, so no video decode
    needed). Idempotent. Note: torchcodec is broken in this env, so force video_backend=pyav."""
    import glob
    import numpy as np
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import lerobot.datasets.v30.augment_dataset_quantile_stats as A

    ds = LeRobotDataset(AGGR_REPO, root=str(AGGR_ROOT), video_backend="pyav")
    if A.has_quantile_stats(ds.meta.stats):
        print("[merge] quantile stats already present", flush=True)
        return
    print("[merge] adding state/action quantile stats (pi05 needs them)", flush=True)
    stats = ds.meta.stats
    files = sorted(glob.glob(str(AGGR_ROOT) + "/data/**/*.parquet", recursive=True))
    Q = [0.01, 0.10, 0.50, 0.90, 0.99]
    NAMES = ["q01", "q10", "q50", "q90", "q99"]
    for col in ("observation.state", "action"):
        data = np.concatenate(
            [np.stack(pd.read_parquet(f, columns=[col])[col].to_numpy()).astype(np.float64) for f in files], 0
        )
        ref = np.asarray(stats[col]["min"])
        qv = np.quantile(data, Q, axis=0)
        for i, nm in enumerate(NAMES):
            stats[col][nm] = qv[i].reshape(ref.shape).astype(ref.dtype)
    A.write_stats(stats, ds.meta.root)
    print("[merge] quantile stats written", flush=True)


def main() -> int:
    if (AGGR_ROOT / "meta" / "info.json").exists():
        print(f"[merge] already merged at {AGGR_ROOT} -> skipping aggregate", flush=True)
        ensure_quantiles()
        return 0

    # 1) Full download of each source dataset (retried), so aggregate has local files.
    for r in REPO_IDS:
        for attempt in (1, 2, 3):
            try:
                print(f"[merge] downloading {r} (try {attempt}/3)", flush=True)
                LeRobotDataset(r)
                break
            except Exception as e:  # network blips etc.
                print(f"[merge] download {r} failed (try {attempt}): {e}", flush=True)
                if attempt == 3:
                    traceback.print_exc()
                    return 2
                time.sleep(20)

    # 2) Aggregate into a single dataset.
    try:
        print(f"[merge] aggregating {len(REPO_IDS)} datasets -> {AGGR_REPO}", flush=True)
        aggregate_datasets(repo_ids=REPO_IDS, aggr_repo_id=AGGR_REPO, aggr_root=AGGR_ROOT)
    except Exception:
        print("[merge] aggregate_datasets FAILED", flush=True)
        traceback.print_exc()
        return 3

    ensure_quantiles()
    ok = (AGGR_ROOT / "meta" / "info.json").exists()
    print(f"[merge] done. merged info.json present={ok}", flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
