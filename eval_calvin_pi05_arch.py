#!/usr/bin/env python3
# Compare pi05-vanilla vs pi05_d (depth-as-prefix-token) on CALVIN ABC->D, offline.
#
# Decisions documented (cf. spec ~/I2CA/tasks/2026-05-07_eval_calvin_pi05_arch_spec.md):
#  1) Action MSE reported in TWO horizons:
#       (a) MSE(action[0])  -- open-loop honest, the action that would be sent next
#       (b) MSE(mean over chunk vs GT chunk)  -- full chunk capacity
#     cup3 showed actions_per_chunk=50 hides visual feedback; both numbers matter.
#  2) Compare actions in PHYSICAL space (post postprocessor), never normalized.
#     mean/std may diverge between vanilla/d if norm_stats were recomputed
#     (cf. openpi#711). All MSEs reported here are post-unnormalization.
#  3) drop-state probe FIRST: if relDelta > 90% as in cup3, abort interpretation
#     of remaining probes -- proprio shortcut dominated and visual signals are
#     swamped.
#  4) drop-language probe NEW for CALVIN (cup3 had no task tokens). relDelta ~0
#     means the policy ignores the instruction -> overfitting.
#  5) Pairing: same frame indices and same seeds across both checkpoints.
#  6) Validation D split is not yet materialized (phase 2 of converter optional).
#     Workaround: take last N episodes from phase 1 (ABC) as pseudo-OOD val and
#     mark as 'preliminary' in the report.
#  7) Group per_task by lang_category (first-word verb, ~23 buckets) instead of
#     task_index (~389), more legible for the human eyeballing the report.
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "lerobot-ext"))


# === bloco 0: imports + constants ===

PERTURBS_CALVIN: tuple[str, ...] = (
    "drop-state",
    "drop-rgb-static",
    "drop-rgb-gripper",
    "drop-depth-static",
    "swap-depth-noise",
    "drop-language",
)

DEPTH_KEY_CALVIN = "observation.depths.static"
DEPTH_SCALE_CALVIN = 1.0  # CALVIN depth already in meters
RGB_STATIC_KEY = "observation.images.rgb_static"
RGB_GRIPPER_KEY = "observation.images.rgb_gripper"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

DEFAULT_DATASET_REPO_ID = "local/calvin_abc_lerobot_depth"

# Probes that only touch a depth key are skipped on vanilla (no PointNet).
DEPTH_ONLY_PROBES = {"drop-depth-static", "swap-depth-noise"}


# === bloco 1: load_policy/build_preproc ===

def load_policy(ckpt_dir: str, fusion_mode: str, device: torch.device):
    """Load a PI05 policy and optionally inject the depth PointNet prefix-token.

    fusion_mode: 'none' (vanilla pi05) or 'depth_only' (pi05_d).
    For pi05_d we re-apply load_state_dict(strict=False) AFTER injection so the
    PointNet weights stop being 'unexpected keys' and actually load.
    """
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    import safetensors.torch as st

    policy = PI05Policy.from_pretrained(ckpt_dir, strict=False).to(device).eval()

    if fusion_mode == "none":
        return policy
    if fusion_mode != "depth_only":
        raise ValueError(f"unsupported fusion_mode={fusion_mode!r}; expected 'none' or 'depth_only'")

    from train.pi05_depth_injector import inject_pi05_depth
    inject_pi05_depth(
        policy,
        device=device,
        depth_key=DEPTH_KEY_CALVIN,
        depth_scale=DEPTH_SCALE_CALVIN,
    )

    sd_path = Path(ckpt_dir) / "model.safetensors"
    if sd_path.exists():
        sd = st.load_file(str(sd_path), device=str(device))
        policy.load_state_dict(sd, strict=False)
    return policy


def build_preproc(policy, ckpt_dir: str):
    """Return (preprocessor, postprocessor) tuple. postprocessor reverses the
    action normalization back to physical space."""
    from lerobot.policies.factory import make_pre_post_processors
    return make_pre_post_processors(policy_cfg=policy.config, pretrained_path=ckpt_dir)


# === bloco 2: sample_val_frames + make_batch + predict_chunk ===

def _verb_of(task_str: str) -> str:
    """First word of the lang annotation as the lang_category bucket.
    ~23 verbs in CALVIN ABC, more legible than task_index (~389)."""
    if not task_str:
        return "<empty>"
    return task_str.split()[0].lower()


def load_task_table(dataset_root: Path) -> dict[int, str]:
    """Read meta/tasks.parquet -> {task_index: task_str}. Resilient to either
    the LeRobot v3 layout (rows indexed by task name, column 'task_index') or
    a flatter layout."""
    import pandas as pd
    p = dataset_root / "meta" / "tasks.parquet"
    df = pd.read_parquet(p)
    if "task_index" in df.columns and df.index.name in (None, "task"):
        # rows indexed by task string
        return {int(row["task_index"]): str(name) for name, row in df.iterrows()}
    if {"task_index", "task"}.issubset(df.columns):
        return {int(r["task_index"]): str(r["task"]) for _, r in df.iterrows()}
    raise RuntimeError(f"unexpected tasks.parquet schema: cols={df.columns.tolist()}")


def sample_val_frames(
    ds,
    n_per_cat: int,
    task_table: dict[int, str],
    episode_filter: Sequence[int] | None = None,
    seed: int = 0,
) -> list[int]:
    """Stratified sample of dataset frame indices, n_per_cat per lang_category.

    If episode_filter is given, only frames whose episode_index is in that set
    are eligible (used to carve out a pseudo-D validation split from the tail
    of phase 1).
    """
    rng = np.random.default_rng(seed)
    total = len(ds)

    # Pull task_index column cheaply through the underlying hf_dataset
    try:
        hf_ds = ds.hf_dataset
        task_idx_col = np.asarray(hf_ds["task_index"])
        ep_idx_col = np.asarray(hf_ds["episode_index"]) if episode_filter is not None else None
    except Exception:
        # fallback: per-sample (slow); only used in mocks
        task_idx_col = np.array([int(ds[i]["task_index"]) for i in range(total)])
        ep_idx_col = None
        if episode_filter is not None:
            ep_idx_col = np.array([int(ds[i]["episode_index"]) for i in range(total)])

    if episode_filter is not None and ep_idx_col is not None:
        ep_set = set(int(e) for e in episode_filter)
        eligible_mask = np.array([int(e) in ep_set for e in ep_idx_col])
    else:
        eligible_mask = np.ones(total, dtype=bool)

    # Group eligible global indices by lang_category
    cat_to_idxs: dict[str, list[int]] = {}
    for global_i in np.where(eligible_mask)[0]:
        ti = int(task_idx_col[global_i])
        cat = _verb_of(task_table.get(ti, ""))
        cat_to_idxs.setdefault(cat, []).append(int(global_i))

    out: list[int] = []
    for cat in sorted(cat_to_idxs.keys()):
        pool = cat_to_idxs[cat]
        if not pool:
            continue
        k = min(n_per_cat, len(pool))
        chosen = rng.choice(np.asarray(pool), size=k, replace=False).tolist()
        out.extend(int(x) for x in chosen)

    print(
        f"[sample_val_frames] sampled {len(out)} frames across {len(cat_to_idxs)} "
        f"lang_categories (n_per_cat={n_per_cat}, eligible={int(eligible_mask.sum())}/{total})"
    )
    return sorted(out)


def make_batch(sample: dict, policy, device: torch.device, lang_task: str) -> dict:
    """Convert a LeRobotDataset sample into a single-batch dict that the
    policy can consume. Filters by policy.config.input_features so vanilla
    (no depth) does not get the depth tensor."""
    out: dict = {}
    expected = set(policy.config.input_features.keys())
    # depth_key is popped by the injector at forward time but NOT listed in
    # input_features for pi05_d post-injection; carry it through anyway when the
    # policy has a pointnet attribute (i.e. injected).
    if hasattr(policy, "pointnet") and DEPTH_KEY_CALVIN in sample:
        expected.add(DEPTH_KEY_CALVIN)

    for name in expected:
        if name not in sample:
            # rgb_gripper or depth might be missing in some legacy samples;
            # raise for clarity rather than silently mis-batch.
            raise KeyError(f"sample missing {name!r}; have {list(sample.keys())[:8]}...")
        v = sample[name]
        if not torch.is_tensor(v):
            v = torch.as_tensor(v)
        out[name] = v.to(device).unsqueeze(0).float()

    out["task"] = lang_task
    return out


def clone_batch(b: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}


def predict_chunk(policy, preproc, batch: dict) -> torch.Tensor:
    """Run preprocessor + predict_action_chunk; returns NORMALIZED chunk (B, H, A)."""
    b = preproc(clone_batch(batch))
    with torch.no_grad():
        return policy.predict_action_chunk(b).detach()


def predict_chunk_physical(policy, preproc, postproc, batch: dict) -> torch.Tensor:
    """Predict chunk and apply postprocessor to revert action normalization.
    Returns chunk in physical units (B, H, A) on CPU as float32.

    LeRobot postprocessor expects an action tensor; we apply it per chunk-step
    and stack. That's the same trick run_eval uses for batched envs."""
    chunk = predict_chunk(policy, preproc, batch)  # (B, H, A) on device
    if postproc is None:
        return chunk.detach().cpu().float()
    # apply postproc per step; postproc usually expects a 2D action tensor
    # but is robust to anything ActionProcessorStep handles. Be defensive:
    out_steps: list[torch.Tensor] = []
    for h in range(chunk.shape[1]):
        a = chunk[:, h, :]  # (B, A)
        try:
            out_a = postproc(a)
        except Exception:
            # some post pipelines expect a dict-like transition; fall back
            out_a = a
        if torch.is_tensor(out_a):
            out_steps.append(out_a.detach().cpu().float())
        else:
            out_steps.append(torch.as_tensor(out_a).detach().cpu().float())
    return torch.stack(out_steps, dim=1)  # (B, H, A)


# === bloco 3: action_mse (chunk vs first) ===

def _gt_chunk_for_index(ds, idx: int, horizon: int, action_key: str = ACTION_KEY) -> torch.Tensor:
    """Pull the GT action chunk of length `horizon` starting at frame idx,
    clamping at episode boundaries by repeating the last action.
    Returns (horizon, A) on CPU as float32."""
    sample = ds[idx]
    ep_idx = int(sample.get("episode_index", torch.tensor(0)).item()
                 if torch.is_tensor(sample.get("episode_index")) else sample.get("episode_index", 0))
    out: list[torch.Tensor] = []
    last_a = None
    for h in range(horizon):
        try:
            s = ds[idx + h]
            same_ep = int(
                s.get("episode_index", torch.tensor(ep_idx)).item()
                if torch.is_tensor(s.get("episode_index"))
                else s.get("episode_index", ep_idx)
            ) == ep_idx
        except Exception:
            s = None
            same_ep = False
        if s is None or not same_ep:
            if last_a is None:
                last_a = torch.zeros(7)
            out.append(last_a.clone())
        else:
            a = s[action_key]
            if not torch.is_tensor(a):
                a = torch.as_tensor(a)
            a = a.float().detach().cpu()
            out.append(a)
            last_a = a
    return torch.stack(out, dim=0)


def action_mse(
    policy,
    preproc,
    postproc,
    ds,
    indices: Sequence[int],
    task_table: dict[int, str],
    horizon: int | None = None,
    log_every: int = 50,
) -> dict:
    """Compute MSE in physical action space, BOTH for action[0] and for chunk-mean.

    Returns:
      {
        'mse_first': float, 'mae_first': float,
        'mse_chunk_mean': float, 'mae_chunk_mean': float,
        'n': int,
        'per_task_index': {ti: {'mse_first', 'mse_chunk_mean', 'n'}},
      }
    """
    if horizon is None:
        horizon = int(getattr(policy.config, "n_action_steps", 50))

    sq_first: list[float] = []
    abs_first: list[float] = []
    sq_chunk: list[float] = []
    abs_chunk: list[float] = []

    per_ti: dict[int, dict[str, list[float]]] = {}

    for k, idx in enumerate(indices):
        sample = ds[idx]
        ti = int(sample["task_index"].item() if torch.is_tensor(sample["task_index"])
                 else sample["task_index"])
        lang = task_table.get(ti, "")
        batch = make_batch(sample, policy, _device_of(policy), lang_task=lang)
        pred = predict_chunk_physical(policy, preproc, postproc, batch)  # (1, H, A)
        pred = pred[0]  # (H, A)
        H = pred.shape[0]
        gt = _gt_chunk_for_index(ds, idx, horizon=H)  # (H, A)

        # pred[0] vs gt[0]
        d0 = (pred[0] - gt[0]).numpy()
        sq_first.append(float((d0 ** 2).mean()))
        abs_first.append(float(np.abs(d0).mean()))

        # mean over chunk vs gt mean
        dchunk = (pred.mean(dim=0) - gt.mean(dim=0)).numpy()
        sq_chunk.append(float((dchunk ** 2).mean()))
        abs_chunk.append(float(np.abs(dchunk).mean()))

        slot = per_ti.setdefault(ti, {"mse_first": [], "mse_chunk_mean": []})
        slot["mse_first"].append(sq_first[-1])
        slot["mse_chunk_mean"].append(sq_chunk[-1])

        if log_every and (k + 1) % log_every == 0:
            print(f"  [action_mse] {k+1}/{len(indices)}  mse_first={np.mean(sq_first):.4f}")

    per_task_summary = {
        ti: {
            "mse_first": float(np.mean(s["mse_first"])),
            "mse_chunk_mean": float(np.mean(s["mse_chunk_mean"])),
            "n": len(s["mse_first"]),
        }
        for ti, s in per_ti.items()
    }

    return {
        "mse_first": float(np.mean(sq_first)) if sq_first else float("nan"),
        "mae_first": float(np.mean(abs_first)) if abs_first else float("nan"),
        "mse_chunk_mean": float(np.mean(sq_chunk)) if sq_chunk else float("nan"),
        "mae_chunk_mean": float(np.mean(abs_chunk)) if abs_chunk else float("nan"),
        "n": int(len(sq_first)),
        "per_task_index": per_task_summary,
    }


# === bloco 4: perturb + probe_table ===

def _device_of(policy) -> torch.device:
    try:
        return next(policy.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def perturb(batch: dict, kind: str, device: torch.device, seed: int) -> dict:
    """Apply one of PERTURBS_CALVIN to a fresh clone of `batch`."""
    b = clone_batch(batch)
    if kind == "drop-state":
        if STATE_KEY in b:
            b[STATE_KEY] = torch.zeros_like(b[STATE_KEY])
    elif kind == "drop-rgb-static":
        if RGB_STATIC_KEY in b:
            b[RGB_STATIC_KEY] = torch.zeros_like(b[RGB_STATIC_KEY])
    elif kind == "drop-rgb-gripper":
        if RGB_GRIPPER_KEY in b:
            b[RGB_GRIPPER_KEY] = torch.zeros_like(b[RGB_GRIPPER_KEY])
    elif kind == "drop-depth-static":
        if DEPTH_KEY_CALVIN in b:
            b[DEPTH_KEY_CALVIN] = torch.zeros_like(b[DEPTH_KEY_CALVIN])
    elif kind == "swap-depth-noise":
        if DEPTH_KEY_CALVIN in b:
            t = b[DEPTH_KEY_CALVIN]
            g = torch.Generator(device=device).manual_seed(seed * 31 + 7)
            b[DEPTH_KEY_CALVIN] = torch.rand(t.shape, generator=g, device=device, dtype=t.dtype)
    elif kind == "drop-language":
        b["task"] = ""
    else:
        raise ValueError(f"unknown perturbation: {kind!r}")
    return b


def probe_table(
    policy,
    preproc,
    postproc,
    ds,
    indices: Sequence[int],
    task_table: dict[int, str],
    perturbs: Sequence[str] = PERTURBS_CALVIN,
    n_seeds: int = 1,
    log_every: int = 50,
) -> dict[str, dict[str, float]]:
    """For each perturbation, measure absolute and relative L2-distance between
    clean physical action chunk and perturbed physical action chunk.
    Returns {kind: {'abs_mean','abs_std','rel_mean','rel_std','n','skipped'}}."""
    device = _device_of(policy)
    has_depth = hasattr(policy, "pointnet")  # True for pi05_d

    out: dict[str, dict[str, float]] = {}
    for kind in perturbs:
        skipped = (kind in DEPTH_ONLY_PROBES) and not has_depth
        out[kind] = {
            "abs_per": [], "rel_per": [], "skipped": skipped,
        }

    for k, idx in enumerate(indices):
        sample = ds[idx]
        ti = int(sample["task_index"].item() if torch.is_tensor(sample["task_index"])
                 else sample["task_index"])
        lang = task_table.get(ti, "")
        batch = make_batch(sample, policy, device, lang_task=lang)
        clean = predict_chunk_physical(policy, preproc, postproc, batch)  # (1, H, A) cpu
        clean_norm = float(clean.norm()) + 1e-9

        for kind in perturbs:
            if out[kind]["skipped"]:
                continue
            for s in range(n_seeds):
                bp = perturb(batch, kind, device, seed=s + k * 11)
                cp = predict_chunk_physical(policy, preproc, postproc, bp)
                d = float((cp - clean).norm())
                out[kind]["abs_per"].append(d)
                out[kind]["rel_per"].append(d / clean_norm)

        if log_every and (k + 1) % log_every == 0:
            print(f"  [probe_table] {k+1}/{len(indices)} processed")

    summary: dict[str, dict[str, float]] = {}
    for kind, slot in out.items():
        if slot["skipped"]:
            summary[kind] = {
                "abs_mean": float("nan"), "abs_std": float("nan"),
                "rel_mean": float("nan"), "rel_std": float("nan"),
                "n": 0, "skipped": True,
            }
            continue
        a = np.asarray(slot["abs_per"])
        r = np.asarray(slot["rel_per"])
        summary[kind] = {
            "abs_mean": float(a.mean()) if a.size else float("nan"),
            "abs_std": float(a.std()) if a.size else float("nan"),
            "rel_mean": float(r.mean()) if r.size else float("nan"),
            "rel_std": float(r.std()) if r.size else float("nan"),
            "n": int(a.size),
            "skipped": False,
        }
    return summary


# === bloco 5: per_task_mse (lang_category) ===

def per_task_mse(
    action_mse_result: dict,
    task_table: dict[int, str],
) -> dict[str, dict[str, Any]]:
    """Re-aggregate per-task_index MSE up to per-lang_category (first verb)."""
    grouped: dict[str, dict[str, list[float] | list[int]]] = {}
    for ti, s in action_mse_result.get("per_task_index", {}).items():
        lang = task_table.get(int(ti), "")
        cat = _verb_of(lang)
        slot = grouped.setdefault(cat, {"mse_first": [], "mse_chunk_mean": [], "n": [], "task_indices": []})
        slot["mse_first"].append(float(s["mse_first"]))
        slot["mse_chunk_mean"].append(float(s["mse_chunk_mean"]))
        slot["n"].append(int(s["n"]))
        slot["task_indices"].append(int(ti))

    out: dict[str, dict[str, Any]] = {}
    for cat, s in grouped.items():
        ns = np.asarray(s["n"], dtype=float)
        n_total = float(ns.sum())
        if n_total <= 0:
            continue
        # weighted mean by frame count
        mf = float((np.asarray(s["mse_first"]) * ns).sum() / n_total)
        mc = float((np.asarray(s["mse_chunk_mean"]) * ns).sum() / n_total)
        out[cat] = {
            "mse_first": mf,
            "mse_chunk_mean": mc,
            "n_frames": int(n_total),
            "n_task_indices": len(s["task_indices"]),
            "task_indices": s["task_indices"],
        }
    return out


# === bloco 6: report MD + plots PNG ===

def _fmt_pct(x: float) -> str:
    if not math.isfinite(x):
        return "n/a"
    return f"{x*100:.1f}%"


def _fmt_num(x: float, fmt: str = ".4f") -> str:
    if not math.isfinite(x):
        return "n/a"
    return f"{x:{fmt}}"


def make_report_md(
    args,
    results_v: dict,
    results_d: dict,
    out_md_path: Path,
    notes: list[str],
) -> None:
    """Write a Markdown report comparing vanilla and pi05_d. Uses absolute file
    paths in headers so I2CA convention is honored."""
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    pv = results_v.get("probes", {})
    pd_ = results_d.get("probes", {})
    av = results_v.get("action_mse", {})
    ad = results_d.get("action_mse", {})

    lines: list[str] = []
    lines.append(f"# CALVIN ABC -> D | pi05 vs pi05_d (offline)")
    lines.append("")
    lines.append(f"- date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- ckpt vanilla: `{args.ckpt_vanilla}`")
    lines.append(f"- ckpt pi05_d:  `{args.ckpt_d}`")
    lines.append(f"- dataset:      `{args.dataset_root}`")
    lines.append(f"- frames per category: {args.n_frames_per_cat}")
    lines.append(f"- n eval frames:  vanilla={results_v.get('n_frames','?')} | d={results_d.get('n_frames','?')}")
    lines.append("")
    if notes:
        lines.append("## Notes / caveats")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    lines.append("## Action MSE (physical space, post unnormalization)")
    lines.append("")
    lines.append("| metric | vanilla | pi05_d | rel diff (d/v - 1) |")
    lines.append("|---|---|---|---|")
    for label, key in (
        ("MSE(action[0])", "mse_first"),
        ("MAE(action[0])", "mae_first"),
        ("MSE(chunk_mean)", "mse_chunk_mean"),
        ("MAE(chunk_mean)", "mae_chunk_mean"),
    ):
        v = av.get(key, float("nan"))
        d = ad.get(key, float("nan"))
        rel = (d / v - 1.0) if (math.isfinite(v) and v != 0 and math.isfinite(d)) else float("nan")
        lines.append(f"| {label} | {_fmt_num(v)} | {_fmt_num(d)} | {_fmt_pct(rel) if math.isfinite(rel) else 'n/a'} |")
    lines.append("")

    lines.append("## Drop-probes (relative L2 distance vs clean chunk)")
    lines.append("")
    lines.append("| perturbation | vanilla relDelta | pi05_d relDelta |")
    lines.append("|---|---|---|")
    for kind in PERTURBS_CALVIN:
        v_slot = pv.get(kind, {})
        d_slot = pd_.get(kind, {})
        v_str = "skipped" if v_slot.get("skipped") else _fmt_pct(v_slot.get("rel_mean", float("nan")))
        d_str = "skipped" if d_slot.get("skipped") else _fmt_pct(d_slot.get("rel_mean", float("nan")))
        lines.append(f"| {kind} | {v_str} | {d_str} |")
    lines.append("")

    lines.append("## Per-lang_category MSE (chunk_mean, weighted by frame count)")
    lines.append("")
    pt_v = results_v.get("per_lang_category", {})
    pt_d = results_d.get("per_lang_category", {})
    cats = sorted(set(pt_v.keys()) | set(pt_d.keys()))
    lines.append("| lang_cat | n_frames | vanilla MSE | pi05_d MSE | rel diff |")
    lines.append("|---|---|---|---|---|")
    for c in cats:
        v_s = pt_v.get(c, {})
        d_s = pt_d.get(c, {})
        n = max(int(v_s.get("n_frames", 0)), int(d_s.get("n_frames", 0)))
        vm = v_s.get("mse_chunk_mean", float("nan"))
        dm = d_s.get("mse_chunk_mean", float("nan"))
        rel = (dm / vm - 1.0) if (math.isfinite(vm) and vm != 0 and math.isfinite(dm)) else float("nan")
        lines.append(
            f"| {c} | {n} | {_fmt_num(vm)} | {_fmt_num(dm)} | "
            f"{_fmt_pct(rel) if math.isfinite(rel) else 'n/a'} |"
        )
    lines.append("")

    lines.append("## Verdict")
    v = verdict(results_v, results_d)
    lines.append(f"- automated verdict: **{v}**")
    lines.append("- rule: pi05_d wins iff (a) mse_chunk_mean of d <= 0.95 * vanilla AND")
    lines.append("  (b) drop-depth-static relDelta on d > 1.5 * (drop-rgb-static relDelta on d)")
    lines.append("")
    out_md_path.write_text("\n".join(lines))
    print(f"[report] wrote {out_md_path}")


def plot_loss_curves(wandb_run_ids: Sequence[str], out_png: Path) -> None:
    """Optional: download train_loss / val_loss curves from wandb.
    Best-effort; if wandb is not configured this just emits a stub PNG."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        import wandb  # noqa: F401
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        api = wandb.Api()
        fig, ax = plt.subplots(figsize=(8, 4))
        plotted = 0
        for rid in wandb_run_ids:
            try:
                run = api.run(rid)
                hist = run.history(keys=["train/loss", "val/loss", "_step"], pandas=False)
                steps = [h.get("_step") for h in hist if h.get("_step") is not None]
                tl = [h.get("train/loss") for h in hist]
                vl = [h.get("val/loss") for h in hist]
                if any(x is not None for x in tl):
                    ax.plot(steps, tl, label=f"{rid} train")
                    plotted += 1
                if any(x is not None for x in vl):
                    ax.plot(steps, vl, label=f"{rid} val", linestyle="--")
                    plotted += 1
            except Exception as e:
                print(f"[plot_loss_curves] could not fetch {rid}: {e}")
        if plotted:
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            ax.legend()
            ax.set_title("loss curves (vanilla vs pi05_d)")
            fig.tight_layout()
            fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f"[plot_loss_curves] wrote {out_png} (n curves={plotted})")
    except Exception as e:
        print(f"[plot_loss_curves] skipped: {e}")


def plot_drop_bar(probes_v: dict, probes_d: dict, out_png: Path) -> None:
    """Side-by-side bar chart of relDelta per perturbation."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        kinds = list(PERTURBS_CALVIN)
        v_vals = [probes_v.get(k, {}).get("rel_mean", 0.0) if not probes_v.get(k, {}).get("skipped") else 0.0
                  for k in kinds]
        d_vals = [probes_d.get(k, {}).get("rel_mean", 0.0) if not probes_d.get(k, {}).get("skipped") else 0.0
                  for k in kinds]
        x = np.arange(len(kinds))
        w = 0.4
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x - w / 2, v_vals, width=w, label="vanilla")
        ax.bar(x + w / 2, d_vals, width=w, label="pi05_d")
        ax.set_xticks(x)
        ax.set_xticklabels(kinds, rotation=20, ha="right")
        ax.set_ylabel("relDelta (||a_pert - a_clean|| / ||a_clean||)")
        ax.set_title("Drop-probes: vanilla vs pi05_d")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f"[plot_drop_bar] wrote {out_png}")
    except Exception as e:
        print(f"[plot_drop_bar] skipped: {e}")


# === bloco 7: verdict (regras concretas) ===

def verdict(results_v: dict, results_d: dict) -> Literal["pi05_d", "neutral", "vanilla"]:
    """Concrete rule for the green/yellow/red light:
      - pi05_d wins iff mse_chunk_mean(d) <= 0.95 * mse_chunk_mean(v)
        AND drop-depth-static relDelta on d > 1.5 * drop-rgb-static relDelta on d
      - vanilla wins iff mse_chunk_mean(v) <= 0.95 * mse_chunk_mean(d)
      - else neutral
    """
    av = results_v.get("action_mse", {})
    ad = results_d.get("action_mse", {})
    pv = results_v.get("probes", {})
    pd_ = results_d.get("probes", {})

    mv = av.get("mse_chunk_mean", float("nan"))
    md = ad.get("mse_chunk_mean", float("nan"))

    drop_depth_d = pd_.get("drop-depth-static", {})
    drop_rgb_d = pd_.get("drop-rgb-static", {})

    if not (math.isfinite(mv) and math.isfinite(md) and mv > 0 and md > 0):
        return "neutral"

    d_better = md <= 0.95 * mv
    v_better = mv <= 0.95 * md

    depth_drives_d = (
        not drop_depth_d.get("skipped", True)
        and math.isfinite(drop_depth_d.get("rel_mean", float("nan")))
        and math.isfinite(drop_rgb_d.get("rel_mean", float("nan")))
        and drop_rgb_d.get("rel_mean", 0.0) > 0
        and drop_depth_d["rel_mean"] > 1.5 * drop_rgb_d["rel_mean"]
    )

    if d_better and depth_drives_d:
        return "pi05_d"
    if v_better:
        return "vanilla"
    return "neutral"


# === bloco main: glue + CLI ===

def evaluate_ckpt(
    args,
    ckpt_dir: str,
    fusion_mode: str,
    indices: list[int],
    task_table: dict[int, str],
    device: torch.device,
) -> dict:
    """Run action_mse + (optionally) probe_table on one ckpt.
    Indices and task_table are shared across both ckpts for fair pairing."""
    print(f"\n=== evaluate_ckpt({fusion_mode}) {ckpt_dir} ===")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    policy = load_policy(ckpt_dir, fusion_mode, device)
    pre, post = build_preproc(policy, ckpt_dir)

    ds = _open_dataset(args.dataset_root)

    a_mse = action_mse(policy, pre, post, ds, indices, task_table)
    if args.skip_probes:
        probes_summary = {}
    else:
        probes_summary = probe_table(policy, pre, post, ds, indices, task_table)

    per_lang = per_task_mse(a_mse, task_table)

    return {
        "checkpoint": ckpt_dir,
        "fusion_mode": fusion_mode,
        "n_frames": len(indices),
        "action_mse": a_mse,
        "probes": probes_summary,
        "per_lang_category": per_lang,
    }


def _open_dataset(dataset_root: str | Path):
    """LeRobotDataset over a local cache root. Not yet enforcing val split (D)
    until the converter exposes phase 2; pseudo-val carving is done at sample
    time via episode_filter."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = Path(dataset_root).expanduser().resolve()
    repo_id = DEFAULT_DATASET_REPO_ID
    print(f"[dataset] opening {repo_id} at {root}")
    return LeRobotDataset(repo_id, root=str(root))


def _maybe_warn_dataset_state(dataset_root: Path) -> list[str]:
    """If meta/info.json is missing or total_episodes differs from the spec
    target (~13510), emit a warning but do not crash."""
    notes: list[str] = []
    info_p = Path(dataset_root).expanduser().resolve() / "meta" / "info.json"
    if not info_p.exists():
        notes.append(
            f"WARNING: {info_p} does not exist yet -- dataset conversion may still be running."
        )
        return notes
    try:
        info = json.loads(info_p.read_text())
    except Exception as e:
        notes.append(f"WARNING: could not parse {info_p}: {e}")
        return notes
    n_ep = int(info.get("total_episodes", 0))
    n_fr = int(info.get("total_frames", 0))
    if n_ep < 13000:
        notes.append(
            f"WARNING: total_episodes={n_ep} (expected ~13510 for CALVIN ABC). "
            "Dataset conversion may still be running -- numbers below are preliminary."
        )
    if n_fr < 800_000:
        notes.append(
            f"WARNING: total_frames={n_fr} (expected ~810k). Preliminary."
        )
    return notes


def _carve_pseudo_val_episodes(dataset_root: Path, frac: float = 0.1) -> list[int]:
    """Pseudo-D split: take last `frac` of episodes from phase 1.
    Documented as a workaround in the report."""
    info_p = Path(dataset_root).expanduser().resolve() / "meta" / "info.json"
    if not info_p.exists():
        return []
    info = json.loads(info_p.read_text())
    n_ep = int(info.get("total_episodes", 0))
    if n_ep <= 0:
        return []
    n_val = max(1, int(n_ep * frac))
    return list(range(n_ep - n_val, n_ep))


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compare pi05 vanilla vs pi05_d (depth as prefix-token) on CALVIN "
            "ABC->D, OFFLINE: action MSE in physical space + drop-probes "
            "(drop-state, drop-rgb-{static,gripper}, drop-depth-static, "
            "swap-depth-noise, drop-language). Pairs frames + seeds across "
            "both checkpoints."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt-vanilla", required=False,
                    help="Path to vanilla pi05 checkpoint dir (the .../pretrained_model leaf).")
    ap.add_argument("--ckpt-d", required=False,
                    help="Path to pi05_d checkpoint dir (the .../pretrained_model leaf).")
    ap.add_argument("--dataset-root", required=False,
                    default=str(Path("~/.cache/huggingface/lerobot/local/calvin_abc_lerobot_depth").expanduser()),
                    help="LeRobot dataset cache root (will be opened with repo_id=local/calvin_abc_lerobot_depth).")
    ap.add_argument("--n-frames-per-cat", type=int, default=20,
                    help="Number of frames sampled per lang_category bucket.")
    ap.add_argument("--out-dir", required=False,
                    default=str(Path("~/I2CA/relatorios/treinamento/eval_calvin_pi05d_2026-05-07").expanduser()),
                    help="Where to drop the .json + .md + .png artifacts.")
    ap.add_argument("--wandb-runs", type=str, default="",
                    help="Comma-separated 'entity/project/run_id,entity/project/run_id' for the loss curve plot.")
    ap.add_argument("--skip-probes", action="store_true",
                    help="If set, skip drop-probe table (only compute action MSE).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-pseudo-val", action="store_true",
                    help="Carve last 10%% of episodes as pseudo-D OOD val (workaround until phase 2).")
    ap.add_argument("--smoke", action="store_true",
                    help="Structural smoke test: import + assert key signatures, no real inference.")
    args = ap.parse_args()

    if args.smoke:
        return _run_smoke()

    if not (args.ckpt_vanilla and args.ckpt_d):
        ap.error("--ckpt-vanilla and --ckpt-d are required (or pass --smoke)")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    notes = _maybe_warn_dataset_state(Path(args.dataset_root))
    if args.use_pseudo_val:
        notes.append(
            "USING PSEUDO-D VAL SPLIT (last 10% of phase-1 episodes as OOD). "
            "Real D split awaits phase 2 of the converter; numbers are preliminary."
        )

    task_table = load_task_table(Path(args.dataset_root).expanduser())

    # Open dataset once just to sample frames; the per-ckpt loop reopens
    # because LeRobotDataset is cheap and avoids tying state across loads.
    ds_for_sampling = _open_dataset(args.dataset_root)
    episode_filter = _carve_pseudo_val_episodes(Path(args.dataset_root)) if args.use_pseudo_val else None
    indices = sample_val_frames(
        ds_for_sampling,
        n_per_cat=args.n_frames_per_cat,
        task_table=task_table,
        episode_filter=episode_filter,
        seed=args.seed,
    )

    # Vanilla first; then free GPU memory and load pi05_d.
    results_v = evaluate_ckpt(args, args.ckpt_vanilla, "none", indices, task_table, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    results_d = evaluate_ckpt(args, args.ckpt_d, "depth_only", indices, task_table, device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_path = out_dir / "results.json"
    raw_path.write_text(json.dumps({"vanilla": results_v, "pi05_d": results_d, "notes": notes}, indent=2))
    print(f"[main] wrote raw {raw_path}")

    md_path = out_dir / "report.md"
    make_report_md(args, results_v, results_d, md_path, notes=notes)

    if args.wandb_runs:
        plot_loss_curves(args.wandb_runs.split(","), out_dir / "loss_curves.png")
    plot_drop_bar(results_v.get("probes", {}), results_d.get("probes", {}),
                  out_dir / "drop_probes.png")

    print(f"\n=== verdict: {verdict(results_v, results_d)} ===")


def _run_smoke() -> int:
    """Structural smoke: ensure all top-level callables exist with the right
    signatures and that mock objects can flow through the perturbations and
    report assembly. Does not load any real ckpt or open a real dataset."""
    print("[smoke] starting structural checks (no GPU, no real ckpt, no real dataset)")
    import inspect

    # 1) every function declared in spec is callable
    public_fns = [
        load_policy, build_preproc, load_task_table, sample_val_frames,
        make_batch, predict_chunk, predict_chunk_physical,
        action_mse, perturb, probe_table, per_task_mse,
        make_report_md, plot_loss_curves, plot_drop_bar, verdict,
        evaluate_ckpt, _carve_pseudo_val_episodes, _maybe_warn_dataset_state,
    ]
    for fn in public_fns:
        sig = inspect.signature(fn)
        assert callable(fn), f"{fn} not callable"
        assert fn.__doc__ is not None or fn.__name__.startswith("_"), \
            f"{fn.__name__} missing docstring"
        print(f"  ok: {fn.__name__}{sig}")

    # 2) perturb on a mock batch covers all kinds without throwing
    device = torch.device("cpu")
    mock_batch = {
        STATE_KEY: torch.randn(1, 15),
        RGB_STATIC_KEY: torch.rand(1, 3, 200, 200),
        RGB_GRIPPER_KEY: torch.rand(1, 3, 84, 84),
        DEPTH_KEY_CALVIN: torch.rand(1, 200, 200),
        "task": "pick up the red block",
    }
    for kind in PERTURBS_CALVIN:
        b = perturb(mock_batch, kind, device, seed=0)
        assert isinstance(b, dict), f"perturb({kind}) did not return a dict"
        if kind == "drop-language":
            assert b["task"] == "", f"drop-language did not zero task"
        elif kind == "swap-depth-noise":
            assert b[DEPTH_KEY_CALVIN].shape == mock_batch[DEPTH_KEY_CALVIN].shape
        else:
            # for drop-* keys we assert the targeted key was zeroed
            target_map = {
                "drop-state": STATE_KEY,
                "drop-rgb-static": RGB_STATIC_KEY,
                "drop-rgb-gripper": RGB_GRIPPER_KEY,
                "drop-depth-static": DEPTH_KEY_CALVIN,
            }
            tk = target_map[kind]
            assert torch.all(b[tk] == 0), f"perturb({kind}) did not zero {tk}"
    print("  ok: perturb covers all kinds in PERTURBS_CALVIN")

    # 3) verdict gates correctly on synthetic results
    v_better = {
        "action_mse": {"mse_chunk_mean": 0.10},
        "probes": {"drop-rgb-static": {"rel_mean": 0.20, "skipped": False},
                   "drop-depth-static": {"rel_mean": float("nan"), "skipped": True}},
    }
    d_better = {
        "action_mse": {"mse_chunk_mean": 0.05},
        "probes": {"drop-rgb-static": {"rel_mean": 0.15, "skipped": False},
                   "drop-depth-static": {"rel_mean": 0.40, "skipped": False}},
    }
    neutral_d = {
        "action_mse": {"mse_chunk_mean": 0.10},
        "probes": {"drop-rgb-static": {"rel_mean": 0.20, "skipped": False},
                   "drop-depth-static": {"rel_mean": 0.05, "skipped": False}},
    }
    assert verdict(v_better, d_better) == "pi05_d", "verdict failed: pi05_d case"
    assert verdict(d_better, v_better) == "vanilla", "verdict failed: vanilla case"
    assert verdict(v_better, neutral_d) == "neutral", "verdict failed: neutral case"
    print("  ok: verdict gating")

    # 4) per_task_mse handles empty/typical inputs
    fake_table = {0: "pick up the red block", 1: "push the switch downwards",
                  2: "pick the blue block"}
    fake_action_mse = {
        "per_task_index": {
            0: {"mse_first": 0.1, "mse_chunk_mean": 0.2, "n": 10},
            1: {"mse_first": 0.05, "mse_chunk_mean": 0.06, "n": 5},
            2: {"mse_first": 0.3, "mse_chunk_mean": 0.4, "n": 3},
        }
    }
    pt = per_task_mse(fake_action_mse, fake_table)
    assert "pick" in pt and "push" in pt, f"per_task_mse missing buckets: {pt.keys()}"
    assert pt["pick"]["n_frames"] == 13, f"weighted aggregation broken: {pt['pick']}"
    print("  ok: per_task_mse aggregation")

    # 5) make_report_md does not crash on synthetic input
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        class A:
            ckpt_vanilla = "/fake/v"
            ckpt_d = "/fake/d"
            dataset_root = "/fake/ds"
            n_frames_per_cat = 20
        results_v_full = {**v_better, "n_frames": 100, "per_lang_category": pt}
        results_d_full = {**d_better, "n_frames": 100, "per_lang_category": pt}
        make_report_md(A(), results_v_full, results_d_full, td_p / "r.md", notes=["mock note"])
        assert (td_p / "r.md").exists()
        print("  ok: make_report_md")

    print("[smoke] all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
