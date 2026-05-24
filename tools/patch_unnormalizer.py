#!/usr/bin/env python
"""Patch the action unnormalizer's std vector in a pi05 checkpoint.

When the training dataset has joints that were idle across all demos,
``compute_dataset_stats`` produces ``action.std == 0`` for those channels.
At inference time the unnormalizer applies ``sent = std * raw + mean``, so a
zero std collapses the entire channel onto ``action.mean`` (dead command).

This tool clamps near-zero entries of ``action.std`` to a small positive
value (default ``1e-3``), restoring controllability on those joints. It
operates **only** on the ``action.std`` tensor inside
``policy_postprocessor_step_0_unnormalizer_processor.safetensors``; every
other tensor (mean, min, max, quantiles, observation.* stats, model
weights, etc.) is left untouched.

Behavior:
- Reads ``<ckpt>/policy_postprocessor_step_0_unnormalizer_processor.safetensors``
- For ``action.std``, replaces values below ``--threshold`` with ``--new-std``
- Saves a ``.bak`` of the original alongside the file before overwriting
- Re-loads the file and asserts ``min(action.std) >= --new-std``

CLI:

    python tools/patch_unnormalizer.py --ckpt <path> \
        [--threshold 1e-4] [--new-std 1e-3] [--dry-run]

``<path>`` may be either the checkpoint root (``.../best/pretrained_model``)
or a direct path to the safetensors file itself.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import safetensors.torch as st
import torch


SAFETENSORS_NAME = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
TARGET_TENSOR = "action.std"


def _resolve_safetensors_path(ckpt_arg: Path) -> Path:
    """Accept either the pretrained_model dir or a direct file path."""
    if ckpt_arg.is_file():
        return ckpt_arg
    candidate = ckpt_arg / SAFETENSORS_NAME
    if not candidate.exists():
        raise FileNotFoundError(
            f"could not find {SAFETENSORS_NAME} under {ckpt_arg}"
        )
    return candidate


def patch_unnormalizer(
    ckpt: Path,
    threshold: float = 1e-4,
    new_std: float = 1e-3,
    dry_run: bool = False,
) -> dict:
    """Patch ``action.std`` of one checkpoint.

    Returns a dict describing what changed (or would change in dry-run).
    """
    fp = _resolve_safetensors_path(ckpt)
    # CRITICAL: clone every tensor right after load_file. safetensors uses mmap
    # and returns tensor views into the file; saving back to the same path
    # invalidates those views and zeros every tensor we didn't touch.
    raw = st.load_file(str(fp))
    tensors = {k: v.clone() for k, v in raw.items()}
    del raw
    if TARGET_TENSOR not in tensors:
        raise KeyError(
            f"{TARGET_TENSOR!r} not in {fp} (keys: {sorted(tensors.keys())[:5]}...)"
        )

    std = tensors[TARGET_TENSOR].clone()
    if std.dtype != torch.float32:
        std = std.to(torch.float32)

    before = std.clone()
    dead_mask = before < threshold
    dead_idxs = [int(i) for i in dead_mask.nonzero(as_tuple=False).flatten().tolist()]
    before_vals = [float(before[i].item()) for i in dead_idxs]

    if dead_idxs:
        std[dead_mask] = float(new_std)
    after_vals = [float(std[i].item()) for i in dead_idxs]

    print(f"[{fp.parent.name}/{fp.parent.parent.name}]")
    print(f"  file: {fp}")
    print(f"  tensor: {TARGET_TENSOR} (shape={tuple(std.shape)}, dtype={std.dtype})")
    print(f"  min(std) before = {float(before.min()):.6g}")
    print(f"  min(std) after  = {float(std.min()):.6g}")
    print(f"  patched indices ({len(dead_idxs)}): {dead_idxs}")
    for i, b, a in zip(dead_idxs, before_vals, after_vals):
        print(f"    idx {i:2d}: {b:.6g}  ->  {a:.6g}")

    result = {
        "file": str(fp),
        "patched_indices": dead_idxs,
        "min_std_before": float(before.min()),
        "min_std_after": float(std.min()),
        "before_values": before_vals,
        "after_values": after_vals,
        "threshold": float(threshold),
        "new_std": float(new_std),
        "backup": None,
        "dry_run": dry_run,
    }

    if dry_run:
        print("  (dry-run: not writing)")
        return result

    if not dead_idxs:
        print("  no dead channels; nothing to write")
        return result

    # 1) backup
    bak = fp.with_suffix(fp.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(fp, bak)
        print(f"  backup -> {bak}")
    else:
        print(f"  backup already exists at {bak} (not overwriting)")
    result["backup"] = str(bak)

    # 2) write patched tensors back
    tensors[TARGET_TENSOR] = std.contiguous()
    st.save_file(tensors, str(fp))
    print(f"  wrote patched file -> {fp}")

    # 3) verify
    reloaded = st.load_file(str(fp))
    reloaded_std = reloaded[TARGET_TENSOR]
    rmin = float(reloaded_std.min())
    rmax = float(reloaded_std.max())
    print(f"  verify: min={rmin:.6g} max={rmax:.6g}")
    assert rmin >= float(new_std) - 1e-9, (
        f"verification failed: min(std)={rmin} < new_std={new_std}"
    )
    result["verified_min_std"] = rmin
    result["verified_max_std"] = rmax
    return result


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Clamp near-zero entries of action.std in a pi05 checkpoint."
    )
    p.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help=(
            "Path to the checkpoint's pretrained_model dir "
            f"(containing {SAFETENSORS_NAME}) or the safetensors file itself."
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=1e-4,
        help="Values of action.std below this are considered dead (default: 1e-4).",
    )
    p.add_argument(
        "--new-std",
        type=float,
        default=1e-3,
        help="Replacement value for dead channels (default: 1e-3).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change but do not write or backup.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        patch_unnormalizer(
            ckpt=args.ckpt,
            threshold=args.threshold,
            new_std=args.new_std,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
