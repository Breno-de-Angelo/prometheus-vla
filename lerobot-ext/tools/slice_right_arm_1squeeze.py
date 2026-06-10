#!/usr/bin/env python3
"""Cria variante 8-dim (braço direito + right_squeeze) de um dataset G1_Dex3 v3_grasp.

LAYOUT DO DATASET DE ORIGEM (v3_grasp, 32-dim action / 28-dim state)
----------------------------------------------------------------------
action (32 dims):
  [0-6]   braço esquerdo
  [7-13]  braço direito        <- ACTION_INDICES[0:7]
  [14-20] mão esquerda (7 dedos)
  [21-27] mão direita  (7 dedos)
  [28]    left_grasp_squeeze
  [29]    right_grasp_squeeze  <- ACTION_INDICES[7]
  [30]    left_grasp_trigger
  [31]    right_grasp_trigger

state (28 dims):
  [0-6]   braço esquerdo
  [7-13]  braço direito        <- STATE_INDICES[0:7]
  [14-20] mão esquerda (7 dedos)
  [21-27] mão direita  (7 dedos)  <- STATE_INDICES[7:14]

SAÍDA
-----
  action  (8 dims):  [braço_direito(7), right_squeeze(1)]
  state  (14 dims):  [braço_direito(7), mão_direita_medida(7)]

No DEPLOY, action[7] (right_squeeze) é convertido pelo controlador:
    hand_q = squeeze × RIGHT_TARGET
    RIGHT_TARGET = [0.0, -0.920, -1.74, 1.57, 1.74, 1.57, 1.74]

USO
---
    python slice_right_arm_1squeeze.py <dataset_v3_grasp_dir> [--out <out_dir>]

    Se --out não for passado, cria:
        <mesmo_pai>/<nome_original>_right8_1squeeze
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Índices no ACTION (32-dim): braço direito (7) + right_grasp_squeeze (1)
ACTION_INDICES = [7, 8, 9, 10, 11, 12, 13, 29]
N_ACTION = len(ACTION_INDICES)  # 8

# Índices no STATE (28-dim): braço direito (7) + mão direita medida (7)
STATE_INDICES = [7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25, 26, 27]
N_STATE = len(STATE_INDICES)  # 14

QUANTILE_MIN_RANGE = 1e-3  # rad; dims congeladas causam divisão por ~eps na normalização


def _slice_col(series: pd.Series, indices: list) -> pd.Series:
    return series.apply(lambda a: np.array(a, dtype=np.float32)[indices])


def _slice_stat_list(vals: list, indices: list) -> list:
    return np.array(vals, dtype=np.float32)[indices].tolist()


def _guard_zero_range_quantiles(feat_stats: dict, feat_key: str) -> None:
    q01, q99 = feat_stats.get("q01"), feat_stats.get("q99")
    if not (isinstance(q01, list) and isinstance(q99, list)):
        return
    for d, (lo, hi) in enumerate(zip(q01, q99)):
        if hi - lo < QUANTILE_MIN_RANGE:
            print(f"  [AVISO] {feat_key} dim {d}: range ~zero (q99-q01={hi-lo:.3e}) "
                  f"— forçando q99 = q01+1.0 pra evitar divisão por eps")
            feat_stats["q99"][d] = lo + 1.0


def process_data_parquets(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for pf in sorted(src.glob("**/*.parquet")):
        rel = pf.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(pf)
        if "action" in df.columns:
            df["action"] = _slice_col(df["action"], ACTION_INDICES)
        if "observation.state" in df.columns:
            df["observation.state"] = _slice_col(df["observation.state"], STATE_INDICES)
        df.to_parquet(out, index=False)
        print(f"  data {rel}: action→{N_ACTION} state→{N_STATE}")


def process_meta_info(src: Path, dst: Path, action_names: list, state_names: list) -> None:
    info = json.loads((src / "meta/info.json").read_text())
    if "action" in info.get("features", {}):
        info["features"]["action"]["shape"] = [N_ACTION]
        info["features"]["action"]["names"] = action_names
    if "observation.state" in info.get("features", {}):
        info["features"]["observation.state"]["shape"] = [N_STATE]
        info["features"]["observation.state"]["names"] = state_names
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "meta/info.json").write_text(json.dumps(info, indent=2))
    print(f"  meta/info.json: action→{N_ACTION} state→{N_STATE}")


def process_meta_stats(src: Path, dst: Path) -> None:
    stats = json.loads((src / "meta/stats.json").read_text())
    if "action" in stats:
        for k, v in stats["action"].items():
            if isinstance(v, list) and len(v) == 32:
                stats["action"][k] = _slice_stat_list(v, ACTION_INDICES)
        _guard_zero_range_quantiles(stats["action"], "action")
    if "observation.state" in stats:
        for k, v in stats["observation.state"].items():
            if isinstance(v, list) and len(v) == 28:
                stats["observation.state"][k] = _slice_stat_list(v, STATE_INDICES)
        _guard_zero_range_quantiles(stats["observation.state"], "observation.state")
    (dst / "meta/stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  meta/stats.json atualizado")


def process_meta_episodes(src: Path, dst: Path) -> None:
    ep_dir = src / "meta/episodes"
    out_ep_dir = dst / "meta/episodes"
    out_ep_dir.mkdir(parents=True, exist_ok=True)
    for pf in sorted(ep_dir.glob("**/*.parquet")):
        rel = pf.relative_to(ep_dir)
        out = out_ep_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(pf)
        for col in df.columns:
            if "stats/action/" in col:
                df[col] = df[col].apply(
                    lambda v: np.array(v, dtype=np.float32)[ACTION_INDICES]
                    if isinstance(v, (list, np.ndarray)) and len(v) == 32 else v
                )
            elif "stats/observation.state/" in col:
                df[col] = df[col].apply(
                    lambda v: np.array(v, dtype=np.float32)[STATE_INDICES]
                    if isinstance(v, (list, np.ndarray)) and len(v) == 28 else v
                )
        df.to_parquet(out, index=False)
    print(f"  meta/episodes: stats fatiados")


def symlink_videos(src: Path, dst: Path) -> None:
    videos_src = src / "videos"
    if not videos_src.exists():
        return
    videos_dst = dst / "videos"
    if videos_dst.exists() or videos_dst.is_symlink():
        videos_dst.unlink() if videos_dst.is_symlink() else shutil.rmtree(videos_dst)
    videos_dst.symlink_to(videos_src.resolve())
    print(f"  videos → symlink para {videos_src.resolve()}")


def copy_tasks_parquet(src: Path, dst: Path) -> None:
    src_tasks = src / "meta/tasks.parquet"
    if src_tasks.exists():
        shutil.copy2(src_tasks, dst / "meta/tasks.parquet")
        print("  meta/tasks.parquet: copiado")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir",
                        help="Pasta raiz do dataset v3_grasp (action=32, state=28)")
    parser.add_argument("--out",
                        help="Pasta de saída (default: <mesmo_pai>/<nome>_right8_1squeeze)")
    args = parser.parse_args()

    src = Path(args.dataset_dir).expanduser().resolve()
    dst = Path(args.out).expanduser().resolve() if args.out else \
        src.parent / (src.name + "_right8_1squeeze")

    print(f"Origem : {src}")
    print(f"Destino: {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    info_src = json.loads((src / "meta/info.json").read_text())
    all_action_names = info_src["features"]["action"].get("names", [])
    all_state_names = info_src["features"]["observation.state"].get("names", [])

    action_names = ([all_action_names[i] for i in ACTION_INDICES]
                    if len(all_action_names) == 32 else [])
    state_names = ([all_state_names[i] for i in STATE_INDICES]
                   if len(all_state_names) == 28 else [])

    print(f"\nAction ({N_ACTION} dims): {action_names}")
    print(f"State  ({N_STATE} dims): {state_names}")

    print("\n[1/5] Parquets de dados...")
    process_data_parquets(src / "data", dst / "data")

    print("\n[2/5] meta/info.json...")
    process_meta_info(src, dst, action_names, state_names)

    print("\n[3/5] meta/stats.json...")
    process_meta_stats(src, dst)

    print("\n[4/5] meta/episodes...")
    process_meta_episodes(src, dst)

    print("\n[5/5] Vídeos e tasks...")
    symlink_videos(src, dst)
    copy_tasks_parquet(src, dst)

    print(f"\n✅ Dataset 8-dim criado em:\n  {dst}")
    print(f"\nDeploy: action[7] (right_squeeze) → hand_q = squeeze × [0.0, -0.920, -1.74, 1.57, 1.74, 1.57, 1.74]")


if __name__ == "__main__":
    main()
