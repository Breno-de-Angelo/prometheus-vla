#!/usr/bin/env python3
"""Quick offline action-MSE eval for the pi05_depth_uni_lf pretrain checkpoint.

Reuses the proven loaders from eval_calvin_pi05_arch.py (load_policy with
fusion_mode='none', build_preproc, predict_chunk_physical). For a set of
held-out frames (sampled from the LAST K episodes -- very likely unseen at
<1 epoch), it predicts the action chunk and compares to ground truth in
PHYSICAL space (post-postprocessor, i.e. radians), against a
predict-the-dataset-mean-action baseline. If model MSE << baseline MSE, the
policy is genuinely using the inputs, not just regressing to the mean.
"""
from __future__ import annotations
import os, sys, argparse, json, time
os.environ.setdefault("HF_HOME", "/data/huggingface-models")
os.environ.setdefault("HF_LEROBOT_HOME", "/data/huggingface-models/lerobot")
import numpy as np
import torch
from pathlib import Path

REPO = Path("/home/hercules/prometheus-vla")
sys.path.insert(0, str(REPO))                 # eval_calvin_pi05_arch
sys.path.insert(0, str(REPO / "lerobot-ext"))  # train.* (not needed for fusion none)
from eval_calvin_pi05_arch import load_policy, build_preproc, predict_chunk_physical


def make_batch_local(sample: dict, policy, device) -> dict:
    """Single-frame obs batch, filtered to exactly policy.config.input_features."""
    out = {}
    for name in policy.config.input_features.keys():
        if name not in sample:
            raise KeyError(f"sample missing {name!r}; have {list(sample)[:8]}")
        v = sample[name]
        v = v if torch.is_tensor(v) else torch.as_tensor(v)
        out[name] = v.to(device).unsqueeze(0).float()
    t = sample.get("task", "")
    out["task"] = t if isinstance(t, str) else ""
    return out


def gt_chunk_local(ds, idx: int, H: int, A: int) -> torch.Tensor:
    """GT action chunk of length H from frame idx, clamped at episode end."""
    ep = int(ds[idx]["episode_index"])
    out, last = [], None
    for h in range(H):
        try:
            s = ds[idx + h]
            same = int(s["episode_index"]) == ep
        except Exception:
            s, same = None, False
        if s is None or not same:
            out.append(last.clone() if last is not None else torch.zeros(A))
        else:
            a = s["action"]
            a = (a if torch.is_tensor(a) else torch.as_tensor(a)).float().cpu()
            out.append(a)
            last = a
    return torch.stack(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo-id", default="lewislf/G1_Dex3_picks7_uni")
    ap.add_argument("--root", default="/data/huggingface-models/lerobot/lewislf/G1_Dex3_picks7_uni")
    ap.add_argument("--n-frames", type=int, default=120)
    ap.add_argument("--holdout-last-eps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={args.ckpt}", flush=True)
    policy = load_policy(args.ckpt, "none", device)
    preproc, postproc = build_preproc(policy, args.ckpt)
    infeat = list(policy.config.input_features.keys())
    print("policy.input_features:", infeat, flush=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav")
    n_eps = ds.num_episodes
    print(f"dataset: {n_eps} eps, {ds.num_frames} frames", flush=True)

    # held-out = frames from the last K episodes (unseen-ish at <1 epoch)
    ep_col = np.asarray(ds.hf_dataset["episode_index"])
    holdout = set(range(max(0, n_eps - args.holdout_last_eps), n_eps))
    eligible = np.where(np.isin(ep_col, list(holdout)))[0]
    rng = np.random.default_rng(args.seed)
    pick = sorted(int(x) for x in rng.choice(eligible, size=min(args.n_frames, len(eligible)), replace=False))
    print(f"held-out frames: {len(pick)} from last {args.holdout_last_eps} eps", flush=True)

    # rename cam_left_high -> head_camera iff the policy expects head_camera
    need_head = "observation.images.head_camera" in infeat
    def fix(s):
        if need_head and "observation.images.head_camera" not in s and "observation.images.cam_left_high" in s:
            s["observation.images.head_camera"] = s["observation.images.cam_left_high"]
        return s

    # horizon from one prediction
    pred0 = predict_chunk_physical(policy, preproc, postproc, make_batch_local(fix(ds[pick[0]]), policy, device))
    H, A = pred0.shape[1], pred0.shape[2]
    print(f"action chunk: horizon H={H}, action_dim A={A}", flush=True)

    mean_act = torch.as_tensor(np.asarray(ds.meta.stats["action"]["mean"]), dtype=torch.float32).reshape(-1)[:A]

    se = {k: [] for k in ("m0", "mc", "m0_arm", "m0_hand", "b0", "bc")}
    t0 = time.time()
    for j, idx in enumerate(pick):
        pred = predict_chunk_physical(policy, preproc, postproc, make_batch_local(fix(ds[idx]), policy, device))[0]  # (H,A)
        gt = gt_chunk_local(ds, idx, H, A)
        d0 = (pred[0] - gt[0]) ** 2
        se["m0"].append(d0.mean().item())
        se["mc"].append(((pred - gt) ** 2).mean().item())
        se["m0_arm"].append(d0[:14].mean().item())
        se["m0_hand"].append(d0[14:].mean().item())
        bpred = mean_act.unsqueeze(0).repeat(H, 1)
        se["b0"].append(((bpred[0] - gt[0]) ** 2).mean().item())
        se["bc"].append(((bpred - gt) ** 2).mean().item())
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{len(pick)}  ({time.time()-t0:.0f}s)", flush=True)

    m = lambda x: float(np.mean(x))
    res = {
        "ckpt": args.ckpt, "n_frames": len(pick), "horizon": H, "action_dim": A,
        "model_mse_action0": m(se["m0"]),
        "model_mse_chunk": m(se["mc"]),
        "model_mse_action0_arm(0-13)": m(se["m0_arm"]),
        "model_mse_action0_hand(14-27)": m(se["m0_hand"]),
        "baseline_mean_action_mse_action0": m(se["b0"]),
        "baseline_mean_action_mse_chunk": m(se["bc"]),
        "reduction_vs_baseline_action0_pct": (1 - m(se["m0"]) / m(se["b0"])) * 100,
        "reduction_vs_baseline_chunk_pct": (1 - m(se["mc"]) / m(se["bc"])) * 100,
    }
    print("\n=== RESULT (physical / radian^2) ===")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
