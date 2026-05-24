#!/usr/bin/env python
"""Analyze a robot run produced by inference_realtime_pi05d.py.

Reads ``run.h5`` (+ optional ``state.h5``) and the run's checkpoint
unnormalizer, then writes three siblings into the run directory:

  - ``analysis.json``  structured metrics + assertions (consumable by LLMs/diff)
  - ``report.md``      human-readable pt-BR summary
  - ``analysis_plots/*.png``  visual evidence (norm stats, RH/body
                              evolution, scene, timing, etc.)

CLI:

    python tools/analyze_run.py <run_dir>

Library:

    from tools.analyze_run import analyze
    analysis = analyze(run_dir)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

# Matplotlib must use a non-interactive backend so headless runs don't try to
# open windows (this script is also invoked from the runtime's `finally`).
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402


# --- Joint layout (28 dims) -------------------------------------------------
# control_mode=upper_body order: body[0..13] + LH[14..20] + RH[21..27]
BODY_SLICE = slice(0, 14)
LH_SLICE = slice(14, 21)
RH_SLICE = slice(21, 28)

RH_NAMES = [
    "thumb_0", "thumb_1", "thumb_2",
    "index_0", "index_1",
    "middle_0", "middle_1",
]
LH_NAMES = list(RH_NAMES)  # same finger order, left side

JOINT_NAMES = (
    [f"body_{i}" for i in range(14)]
    + [f"lh_{n}" for n in LH_NAMES]
    + [f"rh_{n}" for n in RH_NAMES]
)

# Movement thresholds (rad). Anything smaller is "didn't move".
MOVE_THRESHOLD_HAND = 0.10
MOVE_THRESHOLD_BODY = 0.10
STD_DEAD_THRESHOLD = 1e-4  # unnormalizer std below this kills the channel


def _load_run(run_dir: Path) -> dict:
    """Load run.h5 + optional state.h5 + meta.json into a dict of arrays."""
    out: dict = {"run_dir": run_dir}
    run_h5 = run_dir / "run.h5"
    if not run_h5.exists():
        raise FileNotFoundError(f"missing run.h5 in {run_dir}")
    with h5py.File(run_h5, "r") as f:
        for k in [
            "t_obs", "t_predict", "t_send_start", "t_send_end",
            "obs_state", "obs_state_post",
            "obs_left_pressure", "obs_right_pressure",
            "action_chunk", "sent_actions",
        ]:
            out[k] = f[k][:] if k in f else None

    state_h5 = run_dir / "state.h5"
    if state_h5.exists():
        with h5py.File(state_h5, "r") as f:
            out["state_t"] = f["t"][:] if "t" in f else None
            out["state_state"] = f["state"][:] if "state" in f else None
            out["state_left_pressure"] = f["left_pressure"][:] if "left_pressure" in f else None
            out["state_right_pressure"] = f["right_pressure"][:] if "right_pressure" in f else None
            out["state_last_action_sent"] = f["last_action_sent"][:] if "last_action_sent" in f else None
    else:
        for k in ["state_t", "state_state", "state_left_pressure",
                  "state_right_pressure", "state_last_action_sent"]:
            out[k] = None

    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            out["meta"] = json.load(f)
    else:
        out["meta"] = {}
    return out


def _load_unnormalizer_stats(checkpoint_dir: Path) -> dict | None:
    """Read action mean/std from the postprocessor safetensors, if present."""
    fp = checkpoint_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if not fp.exists():
        return None
    try:
        import safetensors.torch as st
    except Exception as e:
        return {"error": f"safetensors import failed: {e}"}
    tensors = st.load_file(str(fp))
    out: dict = {}
    for k, v in tensors.items():
        if k.startswith("action."):
            stat = k.split(".", 1)[1]
            out[stat] = v.numpy().tolist()
    return out


def _group_amplitude(arr_2d_or_3d: np.ndarray, dim_slice: slice) -> float:
    """Max - min over time/chunks for the given dim slice, then max over dims."""
    sub = arr_2d_or_3d[..., dim_slice]
    if sub.ndim == 3:
        amp = sub.max(axis=(0, 1)) - sub.min(axis=(0, 1))
    elif sub.ndim == 2:
        amp = sub.max(axis=0) - sub.min(axis=0)
    else:
        amp = np.array([sub.max() - sub.min()])
    return float(np.max(amp))


def _percentile_p99(arr: np.ndarray) -> float:
    return float(np.percentile(arr, 99)) if arr.size else float("nan")


def _make_plots(data: dict, plot_dir: Path, unnorm: dict | None) -> list[str]:
    """Render all diagnostic PNGs and return their relative paths."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    action_chunk = data["action_chunk"]
    sent_actions = data["sent_actions"]
    obs_state = data["obs_state"]
    obs_state_post = data["obs_state_post"]
    lpress = data["obs_left_pressure"]
    rpress = data["obs_right_pressure"]
    t_obs = data["t_obs"]
    t_predict = data["t_predict"]
    n_chunks = action_chunk.shape[0]

    # ---- 1. Norm stats summary -------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    if unnorm and "std" in unnorm and "mean" in unnorm:
        std = np.array(unnorm["std"])
        mean = np.array(unnorm["mean"])
        x = np.arange(len(std))
        colors = ["tab:blue"] * 14 + ["tab:green"] * 7 + ["tab:red"] * 7
        axes[0].bar(x, std, color=colors)
        axes[0].axhline(STD_DEAD_THRESHOLD, color="black", ls=":", label=f"dead-channel threshold ({STD_DEAD_THRESHOLD})")
        axes[0].set_title("Unnormalizer action.std per dim (blue=body, green=LH, red=RH)")
        axes[0].set_yscale("symlog", linthresh=1e-4)
        axes[0].set_xticks(x); axes[0].set_xticklabels(JOINT_NAMES, rotation=70, fontsize=7)
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].bar(x, mean, color=colors)
        axes[1].set_title("Unnormalizer action.mean per dim")
        axes[1].set_xticks(x); axes[1].set_xticklabels(JOINT_NAMES, rotation=70, fontsize=7)
        axes[1].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "unnormalizer not found", ha="center", va="center")
        axes[1].axis("off")
    plt.tight_layout()
    p = plot_dir / "norm_stats_summary.png"
    plt.savefig(p, dpi=110); plt.close()
    saved.append(p.name)

    # ---- 2. Right hand evolution -----------------------------------------
    fig, axes = plt.subplots(7, 1, figsize=(14, 16), sharex=True)
    for k, j in enumerate(range(21, 28)):
        ax = axes[k]
        ax.plot(obs_state[:, j], "b-", label="state PRE", lw=2)
        ax.plot(obs_state_post[:, j], "c-", label="state POST", lw=1.2, alpha=0.7)
        ax.plot(sent_actions[:, -1, j], "r-", label="sent last", lw=1.5)
        ax.plot(sent_actions[:, 0, j], "r:", label="sent first", lw=1, alpha=0.5)
        ax.plot(action_chunk[:, -1, j], "m--", label="raw last", lw=1, alpha=0.6)
        ax.set_ylabel(f"rh_{RH_NAMES[k]}")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("chunk #")
    plt.suptitle("Right hand: state PRE/POST vs commanded (first/last of chunk)")
    plt.tight_layout()
    p = plot_dir / "right_hand_evolution.png"
    plt.savefig(p, dpi=110); plt.close()
    saved.append(p.name)

    # ---- 3. Body evolution (6 representative dims) -----------------------
    body_dims = [0, 1, 6, 7, 12, 13]
    fig, axes = plt.subplots(len(body_dims), 1, figsize=(14, 12), sharex=True)
    for k, j in enumerate(body_dims):
        ax = axes[k]
        ax.plot(obs_state[:, j], "b-", label="state PRE", lw=2)
        ax.plot(obs_state_post[:, j], "c-", label="state POST", lw=1.2, alpha=0.7)
        ax.plot(sent_actions[:, -1, j], "r-", label="sent last", lw=1.5)
        ax.set_ylabel(f"body_{j}")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("chunk #")
    plt.suptitle("Body joints: state vs commanded (representative dims)")
    plt.tight_layout()
    p = plot_dir / "body_evolution.png"
    plt.savefig(p, dpi=110); plt.close()
    saved.append(p.name)

    # ---- 4. Pressure evolution -------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    im0 = axes[0, 0].imshow(lpress.T, aspect="auto", origin="lower", cmap="viridis")
    axes[0, 0].set_title(f"Left pressure heatmap ({lpress.shape[1]} ch x {lpress.shape[0]} chunks)")
    axes[0, 0].set_xlabel("chunk #"); axes[0, 0].set_ylabel("channel")
    plt.colorbar(im0, ax=axes[0, 0])
    im1 = axes[0, 1].imshow(rpress.T, aspect="auto", origin="lower", cmap="viridis")
    axes[0, 1].set_title(f"Right pressure heatmap")
    axes[0, 1].set_xlabel("chunk #"); axes[0, 1].set_ylabel("channel")
    plt.colorbar(im1, ax=axes[0, 1])
    axes[1, 0].plot(lpress.mean(axis=1), "b-", label="left mean")
    axes[1, 0].plot(rpress.mean(axis=1), "r-", label="right mean")
    axes[1, 0].set_title("Pressure mean (over 33 channels) per chunk")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    axes[1, 1].plot(lpress.max(axis=1), "b-", label="left max")
    axes[1, 1].plot(rpress.max(axis=1), "r-", label="right max")
    axes[1, 1].set_title("Pressure max per chunk")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
    plt.tight_layout()
    p = plot_dir / "pressure_evolution.png"
    plt.savefig(p, dpi=110); plt.close()
    saved.append(p.name)

    # ---- 5. Scene evolution (RGB + depth at 4 chunks) --------------------
    try:
        import cv2  # local import; pillow fallback below if missing
    except Exception:
        cv2 = None
    frames_dir = Path(data["run_dir"]) / "frames"
    chunks_to_show = sorted(set([0, n_chunks // 3, (2 * n_chunks) // 3, n_chunks - 1]))
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, len(chunks_to_show), hspace=0.15, wspace=0.05)
    any_frame = False
    for col, c in enumerate(chunks_to_show):
        rgb_path = frames_dir / f"rgb_{c:04d}.jpg"
        dep_path = frames_dir / f"depth_{c:04d}.png"
        if rgb_path.exists() and cv2 is not None:
            rgb = cv2.imread(str(rgb_path))
            if rgb is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                ax = fig.add_subplot(gs[0, col])
                ax.imshow(rgb); ax.set_title(f"RGB chunk {c}"); ax.axis("off")
                any_frame = True
        if dep_path.exists() and cv2 is not None:
            dep = cv2.imread(str(dep_path), cv2.IMREAD_UNCHANGED)
            if dep is not None:
                ax = fig.add_subplot(gs[1, col])
                im = ax.imshow(dep, cmap="turbo")
                ax.set_title(f"Depth chunk {c}"); ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.04)
                any_frame = True
    plt.suptitle("Scene evolution (RGB top, depth bottom)")
    p = plot_dir / "scene_evolution.png"
    if any_frame:
        plt.savefig(p, dpi=110)
    plt.close()
    if any_frame:
        saved.append(p.name)

    # ---- 6. Timing distribution ------------------------------------------
    predict_ms = (t_predict - t_obs) * 1000.0
    chunk_period_ms = np.diff(t_obs) * 1000.0 if t_obs.shape[0] > 1 else np.array([0.0])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(predict_ms, bins=40, color="tab:blue")
    axes[0].axvline(200, color="red", ls=":", label="200ms target")
    axes[0].set_title(f"Predict latency (ms) | mean={predict_ms.mean():.0f} p99={_percentile_p99(predict_ms):.0f}")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].hist(chunk_period_ms, bins=40, color="tab:orange")
    axes[1].set_title(f"Chunk period (ms) | mean={chunk_period_ms.mean():.0f}")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    p = plot_dir / "timing_distribution.png"
    plt.savefig(p, dpi=110); plt.close()
    saved.append(p.name)

    # ---- 7. Chunk action trajectory (1 typical chunk) --------------------
    c_mid = n_chunks // 2
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    for k, j in enumerate(range(21, 28)):
        axes[0].plot(action_chunk[c_mid, :, j], label=f"rh_{RH_NAMES[k]}")
    axes[0].axhline(obs_state[c_mid, 21:28].mean(), color="gray", ls=":", label="state PRE mean")
    axes[0].set_title(f"Chunk {c_mid}: raw action_chunk over 50 timesteps (right hand)")
    axes[0].set_xlabel("timestep inside chunk")
    axes[0].legend(loc="upper right", fontsize=7, ncol=4); axes[0].grid(alpha=0.3)
    for j in [0, 1, 6, 12, 13]:
        axes[1].plot(action_chunk[c_mid, :, j], label=f"body_{j}")
    axes[1].set_title(f"Chunk {c_mid}: raw action_chunk over 50 timesteps (body)")
    axes[1].set_xlabel("timestep inside chunk")
    axes[1].legend(loc="upper right", fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    p = plot_dir / "chunk_action_traj.png"
    plt.savefig(p, dpi=110); plt.close()
    saved.append(p.name)

    return [f"analysis_plots/{n}" for n in saved]


def _build_per_joint(data: dict) -> dict:
    per_joint: dict[str, dict] = {}
    sent = data["sent_actions"]
    state = data["obs_state"]
    state_post = data["obs_state_post"]
    for j in range(28):
        name = JOINT_NAMES[j]
        sj = sent[:, :, j]
        st_pre = state[:, j]
        st_post = state_post[:, j]
        per_joint[f"j{j:02d}_{name}"] = {
            "sent_mean": float(sj.mean()),
            "sent_std": float(sj.std()),
            "sent_min": float(sj.min()),
            "sent_max": float(sj.max()),
            "state_mean": float(st_pre.mean()),
            "state_std": float(st_pre.std()),
            "state_post_mean": float(st_post.mean()),
            "tracking_error_mean": float((st_post - sj[:, -1]).mean()),
        }
    return per_joint


def _build_groups(data: dict) -> dict:
    """Per-group movement summary.

    Reports both the maximum per-joint amplitude (state_amplitude) AND the
    average per-joint amplitude (state_amplitude_mean). ``movement_observed``
    uses the mean — that catches "only 1 joint of the hand moved" cases (e.g.
    one finger flexed because of the body trajectory), which the max-based view
    would falsely classify as "the hand moved".
    """
    sent = data["sent_actions"]
    state = data["obs_state"]
    out: dict = {}
    for name, sl, thresh in [
        ("body", BODY_SLICE, MOVE_THRESHOLD_BODY),
        ("left_hand", LH_SLICE, MOVE_THRESHOLD_HAND),
        ("right_hand", RH_SLICE, MOVE_THRESHOLD_HAND),
    ]:
        sub_state = state[:, sl]
        sub_sent = sent[:, :, sl]
        per_joint_state_amp = sub_state.max(axis=0) - sub_state.min(axis=0)
        per_joint_sent_amp = sub_sent.max(axis=(0, 1)) - sub_sent.min(axis=(0, 1))
        # Hands: mean-per-joint is a fairer signal because >50% of fingers must
        # move to count as "the hand moved". Body: max is fine (any large arm
        # joint counts).
        if name == "body":
            move_metric = float(per_joint_state_amp.max())
        else:
            move_metric = float(per_joint_state_amp.mean())
        out[name] = {
            "sent_amplitude": float(per_joint_sent_amp.max()),
            "state_amplitude": float(per_joint_state_amp.max()),
            "state_amplitude_mean": float(per_joint_state_amp.mean()),
            "state_amplitude_per_joint": [float(x) for x in per_joint_state_amp],
            "sent_amplitude_per_joint": [float(x) for x in per_joint_sent_amp],
            "movement_observed": bool(move_metric > thresh),
            "movement_metric": move_metric,
            "threshold": thresh,
        }
    return out


def _build_pressure(data: dict) -> dict:
    lp = data["obs_left_pressure"]
    rp = data["obs_right_pressure"]
    lp_active = ((lp.max(0) - lp.min(0)) > 1.0).astype(bool)
    rp_active = ((rp.max(0) - rp.min(0)) > 1.0).astype(bool)
    return {
        "left_active_cells": [int(i) for i in np.where(lp_active)[0]],
        "left_active_count": int(lp_active.sum()),
        "right_active_cells": [int(i) for i in np.where(rp_active)[0]],
        "right_active_count": int(rp_active.sum()),
        "left_max": float(lp.max()),
        "right_max": float(rp.max()),
    }


def _build_checkpoint_health(unnorm: dict | None) -> dict:
    if not unnorm or "std" not in unnorm:
        return {"unnormalizer": None}
    std = np.array(unnorm["std"])
    zero_idxs = [int(i) for i, v in enumerate(std) if v < STD_DEAD_THRESHOLD]
    # "saturating" = std present but max/min indistinguishable for unnorm purposes
    saturating = []
    if "min" in unnorm and "max" in unnorm:
        mn = np.array(unnorm["min"]); mx = np.array(unnorm["max"])
        for i, (a, b) in enumerate(zip(mn, mx)):
            if abs(a) >= 1.499 and abs(b) >= 1.499:
                saturating.append(int(i))
    min_dim = int(np.argmin(std))
    return {
        "unnormalizer": {
            "min_std": float(std.min()),
            "max_std": float(std.max()),
            "zero_channels": zero_idxs,
            "zero_channel_names": [JOINT_NAMES[i] for i in zero_idxs],
            "saturating_channels": saturating,
            "saturating_channel_names": [JOINT_NAMES[i] for i in saturating],
            "min_movement_dim": JOINT_NAMES[min_dim],
        }
    }


def _build_timing(data: dict) -> dict:
    t_obs = data["t_obs"]
    t_predict = data["t_predict"]
    predict_ms = (t_predict - t_obs) * 1000.0
    chunk_period_ms = np.diff(t_obs) * 1000.0 if t_obs.shape[0] > 1 else np.array([0.0])
    send_ms = (data["t_send_end"] - data["t_send_start"]) * 1000.0
    return {
        "predict_mean": float(predict_ms.mean()),
        "predict_max": float(predict_ms.max()),
        "predict_min": float(predict_ms.min()),
        "predict_p99": _percentile_p99(predict_ms),
        "send_mean": float(send_ms.mean()),
        "chunk_period_mean": float(chunk_period_ms.mean()) if chunk_period_ms.size else 0.0,
    }


def _build_assertions(
    checkpoint_health: dict,
    groups: dict,
    timing: dict,
) -> dict:
    unn = checkpoint_health.get("unnormalizer") if checkpoint_health else None
    min_std = unn.get("min_std") if unn else None
    ckpt_pass = bool(min_std is not None and min_std > STD_DEAD_THRESHOLD)
    return {
        "ckpt_min_std_above_threshold": {
            "passed": ckpt_pass,
            "value": float(min_std) if min_std is not None else None,
            "threshold": STD_DEAD_THRESHOLD,
        },
        "right_hand_moved": {
            "passed": bool(groups["right_hand"]["movement_observed"]),
            "value": float(groups["right_hand"]["movement_metric"]),
            "threshold": MOVE_THRESHOLD_HAND,
        },
        "left_hand_moved": {
            "passed": bool(groups["left_hand"]["movement_observed"]),
            "value": float(groups["left_hand"]["movement_metric"]),
            "threshold": MOVE_THRESHOLD_HAND,
        },
        "body_moved": {
            "passed": bool(groups["body"]["movement_observed"]),
            "value": float(groups["body"]["movement_metric"]),
            "threshold": MOVE_THRESHOLD_BODY,
        },
        "predict_latency_p99_under_200ms": {
            "passed": bool(timing["predict_p99"] < 200.0),
            "value": float(timing["predict_p99"]),
            "threshold": 200.0,
        },
    }


def _diagnose(
    assertions: dict,
    groups: dict,
    checkpoint_health: dict,
) -> dict:
    issues: list[str] = []
    if not assertions["ckpt_min_std_above_threshold"]["passed"]:
        issues.append("UNNORMALIZER_STD_ZERO")
    if not assertions["right_hand_moved"]["passed"]:
        # distinguish "cmd dead" from "motor disobeyed"
        if groups["right_hand"]["sent_amplitude"] > MOVE_THRESHOLD_HAND:
            issues.append("RIGHT_HAND_CMD_BUT_NO_MOTION")
        else:
            issues.append("RIGHT_HAND_NO_CMD")
    if not assertions["left_hand_moved"]["passed"]:
        if groups["left_hand"]["sent_amplitude"] > MOVE_THRESHOLD_HAND:
            issues.append("LEFT_HAND_CMD_BUT_NO_MOTION")
        else:
            issues.append("LEFT_HAND_NO_CMD")
    if not assertions["body_moved"]["passed"]:
        issues.append("BODY_STATIC")
    if not assertions["predict_latency_p99_under_200ms"]["passed"]:
        issues.append("HIGH_PREDICT_LATENCY")

    if "UNNORMALIZER_STD_ZERO" in issues or "RIGHT_HAND_CMD_BUT_NO_MOTION" in issues:
        status = "BUG"
    elif issues:
        status = "WARN"
    else:
        status = "OK"
    return {"status": status, "issues": issues}


def _render_report(analysis: dict) -> str:
    g = analysis["groups"]
    diag = analysis["diagnosis"]
    timing = analysis["timing_ms"]
    ck = analysis["checkpoint_health"].get("unnormalizer")

    def yn(b: bool) -> str:
        return "SIM" if b else "NAO"

    lines: list[str] = []
    lines.append(f"# Run {analysis['run_id']}")
    lines.append("")
    lines.append(f"**Status**: {diag['status']}")
    lines.append(
        f"**Duracao**: {analysis['duration_s']:.1f}s | "
        f"**Chunks**: {analysis['chunks_logged']} | "
        f"**State samples**: {analysis['state_samples_logged']}"
    )
    lines.append(f"**Checkpoint**: `{analysis.get('checkpoint', 'unknown')}`")
    lines.append(f"**Fusion mode**: {analysis.get('fusion_mode', 'unknown')}")
    lines.append("")

    # Diagnostico
    lines.append("## Diagnostico")
    issue_explanations = {
        "UNNORMALIZER_STD_ZERO": "Unnormalizer com std~0 em alguns canais -> comandos nesses dims sao mortos (sempre = mean).",
        "RIGHT_HAND_CMD_BUT_NO_MOTION": "Comandos da mao direita variam mas o estado nao se move -> motor/bridge nao executa.",
        "RIGHT_HAND_NO_CMD": "Mao direita sem comando significativo (politica nao manda nada).",
        "LEFT_HAND_CMD_BUT_NO_MOTION": "Comandos da mao esquerda existem mas estado nao se move.",
        "LEFT_HAND_NO_CMD": "Mao esquerda sem comando (politica produz cmd ~0 ou unnormalizer mata).",
        "BODY_STATIC": "Corpo nao se move.",
        "HIGH_PREDICT_LATENCY": "Predict latency p99 acima de 200ms -> torch.compile pode estar desligado.",
    }
    if not diag["issues"]:
        lines.append("- Nenhuma issue detectada.")
    else:
        for iss in diag["issues"]:
            lines.append(f"- **{iss}**: {issue_explanations.get(iss, '(sem descricao)')}")
    lines.append("")

    # Health do checkpoint
    lines.append("## Health do checkpoint (unnormalizer)")
    if ck is None:
        lines.append("- (unnormalizer.safetensors nao encontrado)")
    else:
        lines.append(f"- min(std) = {ck['min_std']:.6f} | max(std) = {ck['max_std']:.6f}")
        if ck["zero_channels"]:
            lines.append(f"- **Canais mortos** (std<{STD_DEAD_THRESHOLD}): {ck['zero_channels']}")
            lines.append(f"  - nomes: {ck['zero_channel_names']}")
        if ck["saturating_channels"]:
            lines.append(f"- Canais saturando (|min|,|max|>=1.5): {ck['saturating_channels']}")
            lines.append(f"  - nomes: {ck['saturating_channel_names']}")
    lines.append("")

    # Movimento
    lines.append("## Movimento")
    lines.append("| grupo | cmd max | state max | state mean per-joint | mexeu? |")
    lines.append("|---|---|---|---|---|")
    for k in ["body", "left_hand", "right_hand"]:
        lines.append(
            f"| {k} | {g[k]['sent_amplitude']:.3f} | {g[k]['state_amplitude']:.3f} | "
            f"{g[k]['state_amplitude_mean']:.3f} | {yn(g[k]['movement_observed'])} |"
        )
    lines.append("")
    # Hand per-joint detail
    for k in ("left_hand", "right_hand"):
        amps = g[k]["state_amplitude_per_joint"]
        sent = g[k]["sent_amplitude_per_joint"]
        lines.append(f"### {k} per-joint (state amp | sent amp)")
        for i, name in enumerate(LH_NAMES if k == "left_hand" else RH_NAMES):
            lines.append(f"- {name}: state={amps[i]:.4f} | sent={sent[i]:.4f}")
    lines.append("")

    # Timing
    lines.append("## Timing")
    lines.append(
        f"- Predict latency: mean={timing['predict_mean']:.0f}ms "
        f"p99={timing['predict_p99']:.0f}ms "
        f"max={timing['predict_max']:.0f}ms"
    )
    lines.append(
        f"- Send latency: mean={timing['send_mean']:.0f}ms | "
        f"Chunk period: mean={timing['chunk_period_mean']:.0f}ms"
    )
    lines.append("")

    # Pressao
    pr = analysis["pressure"]
    lines.append("## Pressao")
    lines.append(
        f"- LH: {pr['left_active_count']}/33 celulas ativas (max={pr['left_max']:.0f}) | "
        f"RH: {pr['right_active_count']}/33 celulas ativas (max={pr['right_max']:.0f})"
    )
    lines.append("")

    # Assertions
    lines.append("## Assertions")
    lines.append("| nome | passou | valor | threshold |")
    lines.append("|---|---|---|---|")
    for name, a in analysis["assertions"].items():
        passed = yn(a["passed"])
        val = "-" if a["value"] is None else f"{a['value']:.4f}"
        thresh = f"{a['threshold']:.4f}"
        lines.append(f"| {name} | {passed} | {val} | {thresh} |")
    lines.append("")

    # Proximos passos
    lines.append("## Proximos passos sugeridos")
    if "UNNORMALIZER_STD_ZERO" in diag["issues"]:
        lines.append("- Re-treinar/regenerar o unnormalizer: o checkpoint tem std=0 em canais criticos.")
        lines.append("- Conferir se o dataset usado tinha movimento real nesses canais (talvez todos os ep foram travados).")
    if "RIGHT_HAND_CMD_BUT_NO_MOTION" in diag["issues"]:
        lines.append("- Diagnosticar o Dex3 bridge (hand_kp/hand_kd, motor enabled, DDS).")
    if "HIGH_PREDICT_LATENCY" in diag["issues"]:
        lines.append("- Habilitar torch.compile e medir aquecimento; conferir se warmup foi executado.")
    if not diag["issues"]:
        lines.append("- Run aparenta saudavel. Considerar rodar com seeds e tasks variadas.")
    lines.append("")

    # Plots
    lines.append("## Plots gerados")
    for p in analysis.get("plots", []):
        lines.append(f"- `{p}`")
    return "\n".join(lines)


def analyze(run_dir: Path, save_plots: bool = True) -> dict:
    """Analyze a robot run; writes analysis.json, report.md, analysis_plots/*.png; returns the analysis dict."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {run_dir}")

    data = _load_run(run_dir)
    meta = data["meta"]
    ckpt_dir = Path(meta["checkpoint"]) if meta.get("checkpoint") else None
    unnorm = _load_unnormalizer_stats(ckpt_dir) if ckpt_dir else None

    t_obs = data["t_obs"]
    t_send_end = data["t_send_end"]
    duration_s = float(t_send_end[-1] - t_obs[0]) if t_obs.shape[0] else 0.0
    state_samples = int(data["state_state"].shape[0]) if data.get("state_state") is not None else 0

    per_joint = _build_per_joint(data)
    groups = _build_groups(data)
    pressure = _build_pressure(data)
    checkpoint_health = _build_checkpoint_health(unnorm)
    timing = _build_timing(data)
    assertions = _build_assertions(checkpoint_health, groups, timing)
    diagnosis = _diagnose(assertions, groups, checkpoint_health)

    plots: list[str] = []
    if save_plots:
        try:
            plots = _make_plots(data, run_dir / "analysis_plots", unnorm)
        except Exception as e:
            plots = [f"<plot generation failed: {e}>"]

    analysis = {
        "run_id": run_dir.name,
        "started_at": meta.get("started_at"),
        "checkpoint": meta.get("checkpoint"),
        "fusion_mode": meta.get("fusion_mode"),
        "task": meta.get("task"),
        "chunks_logged": int(t_obs.shape[0]),
        "state_samples_logged": state_samples,
        "duration_s": duration_s,
        "diagnosis": diagnosis,
        "checkpoint_health": checkpoint_health,
        "per_joint": per_joint,
        "groups": groups,
        "pressure": pressure,
        "timing_ms": timing,
        "assertions": assertions,
        "plots": plots,
    }

    # Write JSON + Markdown report.
    with open(run_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
    with open(run_dir / "report.md", "w") as f:
        f.write(_render_report(analysis))
    return analysis


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: analyze_run.py <run_dir>", file=sys.stderr)
        return 1
    run_dir = Path(sys.argv[1])
    a = analyze(run_dir)
    print(f"analysis written to {run_dir}/analysis.json + report.md")
    print(f"status: {a['diagnosis']['status']} | issues: {a['diagnosis']['issues']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
