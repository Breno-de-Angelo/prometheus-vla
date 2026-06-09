#!/usr/bin/env python3
"""Cria variante 14-dim (braço+mão direita) de um dataset LeRobot G1_Dex3 28-dim.

CONTEXTO
--------
O dataset gravado pelo pipeline de teleop do G1 tem 28 dimensões de action/state:
  [0-6]   braço esquerdo  (shoulder P/R/Y, elbow, wrist R/P/Y)
  [7-13]  braço direito   (shoulder P/R/Y, elbow, wrist R/P/Y)
  [14-20] mão esquerda Dex3 (thumb 0/1/2, middle 0/1, index 0/1)
  [21-27] mão direita Dex3  (thumb 0/1/2, index 0/1, middle 0/1)

Para treinar o pi05 só com o lado direito, as 14 dims [7-13, 21-27] são
suficientes. Manter as 14 dims do lado esquerdo (congeladas/zero) no
treinamento desperdiçaria capacidade do modelo e causaria divisão por zero
na normalização QUANTILES das dims zeradas.

O QUE O SCRIPT FAZ
------------------
Faz uma passagem única sobre todos os arquivos do dataset de origem e grava
um novo dataset em <out_dir> com:
  - data/chunk-*/file-*.parquet  → action e observation.state fatiados para 14 dims
  - meta/info.json               → shape e names atualizados para 14 dims
  - meta/stats.json              → estatísticas (min/max/mean/std/quantis) fatiadas
  - meta/episodes/               → stats por episódio fatiadas
  - videos/                      → symlink para o diretório original (sem duplicar dados)
  - meta/tasks.parquet           → copiado sem alteração

Sensores táteis (left/right_hand_pressure) e câmeras não são alterados.

USO
---
    python slice_right_arm_only.py <dataset_dir> [--out <out_dir>]

    Se --out não for passado, cria:
        datasets/G1_Dex3_right14_dataset/<nome_do_dataset_dir>
    (dois níveis acima de <dataset_dir>, mesmo nome de timestamp)

WORKFLOW TÍPICO
---------------
    # 1. Gravar episódios (gera dataset 28-dim em G1_Dex3_depth_tactil_dataset)
    python lerobot-ext/tools/init_lerobot_record_v2.py ...

    # 2. Fatiar para 14-dim (gera dataset em G1_Dex3_right14_dataset)
    python lerobot-ext/tools/slice_right_arm_only.py \\
        datasets/G1_Dex3_depth_tactil_dataset/<timestamp>

    # 3. Treinar apontando para o dataset 14-dim
    python -m train.run_train \\
        --config lerobot-ext/config/train/train_cup_pi05_right14.yaml
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Índices (0-based, espaço 28-dim) do braço direito + mão direita
RIGHT_INDICES = [7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25, 26, 27]
N_RIGHT = len(RIGHT_INDICES)  # 14


def _slice_array_col(series: pd.Series) -> pd.Series:
    """Fatia cada elemento (array 28-dim) para RIGHT_INDICES."""
    return series.apply(lambda a: np.array(a, dtype=np.float32)[RIGHT_INDICES])


def _slice_stat_list(vals: list) -> list:
    """Fatia lista de 28 floats para os 14 índices direitos."""
    arr = np.array(vals, dtype=np.float32)
    return arr[RIGHT_INDICES].tolist()


def process_data_parquets(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for pf in sorted(src.glob("**/*.parquet")):
        rel = pf.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(pf)
        for col in ("action", "observation.state"):
            if col in df.columns:
                df[col] = _slice_array_col(df[col])
        df.to_parquet(out, index=False)
        print(f"  data {rel}: {len(df)} linhas, action/state → 14 dims")


def process_meta_info(src: Path, dst: Path, right_names: list) -> None:
    info = json.loads((src / "meta/info.json").read_text())

    for feat_key in ("action", "observation.state"):
        if feat_key in info.get("features", {}):
            feat = info["features"][feat_key]
            if isinstance(feat.get("shape"), list):
                feat["shape"] = [N_RIGHT]
            if isinstance(feat.get("names"), list) and len(feat["names"]) == 28:
                feat["names"] = right_names

    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "meta/info.json").write_text(json.dumps(info, indent=2))
    print(f"  meta/info.json: action/state shape → {N_RIGHT}")


def process_meta_stats(src: Path, dst: Path) -> None:
    stats = json.loads((src / "meta/stats.json").read_text())

    for feat_key in ("action", "observation.state"):
        if feat_key not in stats:
            continue
        for stat_key, val in stats[feat_key].items():
            if isinstance(val, list) and len(val) == 28:
                stats[feat_key][stat_key] = _slice_stat_list(val)

    (dst / "meta/stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  meta/stats.json: action/state stats → 14 dims")


def process_meta_episodes(src: Path, dst: Path) -> None:
    ep_dir = src / "meta/episodes"
    out_ep_dir = dst / "meta/episodes"
    out_ep_dir.mkdir(parents=True, exist_ok=True)

    for pf in sorted(ep_dir.glob("**/*.parquet")):
        rel = pf.relative_to(ep_dir)
        out = out_ep_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(pf)
        stat_cols = [
            c for c in df.columns
            if ("stats/action/" in c or "stats/observation.state/" in c)
        ]
        for col in stat_cols:
            df[col] = df[col].apply(
                lambda v: np.array(v, dtype=np.float32)[RIGHT_INDICES]
                if isinstance(v, (list, np.ndarray)) and len(v) == 28
                else v
            )
        df.to_parquet(out, index=False)
    print(f"  meta/episodes: stats de action/state → 14 dims")


def symlink_videos(src: Path, dst: Path) -> None:
    """Cria symlinks para os diretórios de vídeo (evita duplicar GBs)."""
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
    parser.add_argument("dataset_dir", help="Pasta raiz do dataset LeRobot 28-dim")
    parser.add_argument("--out", help="Pasta de saída (default: ../G1_Dex3_right14_dataset/<nome>)")
    args = parser.parse_args()

    src = Path(args.dataset_dir).expanduser().resolve()
    if args.out:
        dst = Path(args.out).expanduser().resolve()
    else:
        dst = src.parent.parent / "G1_Dex3_right14_dataset" / src.name

    print(f"Origem : {src}")
    print(f"Destino: {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    # Nomes das juntas direitas (extrai do info.json de origem)
    info_src = json.loads((src / "meta/info.json").read_text())
    all_names = info_src["features"]["action"].get("names", [])
    right_names = [all_names[i] for i in RIGHT_INDICES] if len(all_names) == 28 else []

    print("\n[1/5] Parquets de dados...")
    process_data_parquets(src / "data", dst / "data")

    print("\n[2/5] meta/info.json...")
    process_meta_info(src, dst, right_names)

    print("\n[3/5] meta/stats.json...")
    process_meta_stats(src, dst)

    print("\n[4/5] meta/episodes...")
    process_meta_episodes(src, dst)

    print("\n[5/5] Vídeos e tasks...")
    symlink_videos(src, dst)
    copy_tasks_parquet(src, dst)

    print(f"\nDataset 14-dim criado em:\n  {dst}")
    print(f"\nDims: {right_names}")


if __name__ == "__main__":
    main()
