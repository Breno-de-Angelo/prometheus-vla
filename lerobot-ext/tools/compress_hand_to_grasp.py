#!/usr/bin/env python3
"""Comprime a representação da mão de 14 posições de dedo -> 4 escalares de grasp.

As 7 posições de dedo de cada mão são EXATAMENTE `squeeze·TARGET + trigger·W` (resíduo 0,
ver teleop/xr_g1_arm.py). Logo carregam só 2 graus de liberdade reais por mão. Este script
substitui no ACTION as 14 dims de dedo (7 esq + 7 dir) por 4 escalares (squeeze/trigger ×
2 mãos), mantendo as 14 dims de braço. action: 28 -> 18 dims.

    action[0:14]  = braço (7 esq + 7 dir)            <- idêntico ao original
    action[14]    = left_grasp_squeeze   (fecha a mão esq)
    action[15]    = right_grasp_squeeze  (fecha a mão dir)
    action[16]    = left_grasp_trigger   (pinça esq)
    action[17]    = right_grasp_trigger  (pinça dir)

O ACTION é LOSSLESS (comando = squeeze·TARGET + trigger·W exato, resíduo 0): na inferência
reconstrói os 7 dedos com `hand_q = squeeze·TARGET + trigger·W` (inclui thumb_0 = -0.5·trigger
→ pinça preservada).

O `observation.state` é MANTIDO em 28 dims (14 braço + 14 dedos medidos), pois o estado real
dos dedos NÃO segue a fórmula (resíduo ~0.22 rad: a mão para no objeto, tem dinâmica) — comprimi-lo
perderia propriocepção. pi05 aceita state (28) e action (18) de tamanhos diferentes.

Parte do dataset CRU de 28 dims (ex: o v2). Recomputa stats.json (com quantis) e meta/episodes/*.
Vídeos via symlink.

Uso:
    python tools/compress_hand_to_grasp.py IN_DIR OUT_DIR
"""
import sys, os, json, shutil, glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RIGHT_TARGET = np.array([0.0, -0.920, -1.74, 1.57, 1.74, 1.57, 1.74])
LEFT_TARGET  = np.array([0.0,  0.920,  1.74,-1.57,-1.74,-1.57,-1.74])
W_RIGHT = np.array([-0.5, -0.8, -0.8, 0.0, 0.0, 2.2, 2.1])
W_LEFT  = np.array([-0.5,  0.8,  0.8, 0.0, 0.0,-2.2,-2.1])

NEW_HAND_NAMES = ["left_grasp_squeeze.q", "right_grasp_squeeze.q",
                  "left_grasp_trigger.q", "right_grasp_trigger.q"]
STAT_KEYS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]


def recover(qhand, TARGET, W):
    A = np.stack([TARGET, W], axis=1)
    sol = qhand @ np.linalg.pinv(A).T
    resid = float(np.abs(sol @ A.T - qhand).max())
    return np.clip(sol[:, 0], 0, 1), np.clip(sol[:, 1], 0, 1), resid


def compress_vec(v28):
    """(N,28) [7 braço-e, 7 braço-d, 7 mão-e, 7 mão-d] -> (N,18) [14 braço, 4 grasp]."""
    arm = v28[:, 0:14]
    sqL, trL, rL = recover(v28[:, 14:21], LEFT_TARGET, W_LEFT)
    sqR, trR, rR = recover(v28[:, 21:28], RIGHT_TARGET, W_RIGHT)
    grasp = np.stack([sqL, sqR, trL, trR], axis=1)
    return np.concatenate([arm, grasp], axis=1).astype(np.float32), max(rL, rR)


def col_stats(x):
    return {"min": [float(x.min())], "max": [float(x.max())], "mean": [float(x.mean())],
            "std": [float(x.std())], "count": [int(x.shape[0])],
            "q01": [float(np.quantile(x, .01))], "q10": [float(np.quantile(x, .10))],
            "q50": [float(np.quantile(x, .50))], "q90": [float(np.quantile(x, .90))],
            "q99": [float(np.quantile(x, .99))]}


def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    shutil.copytree(os.path.join(in_dir, "meta"), os.path.join(out_dir, "meta"), dirs_exist_ok=True)
    vsrc = os.path.join(in_dir, "videos")
    if os.path.isdir(vsrc) and not os.path.exists(os.path.join(out_dir, "videos")):
        os.symlink(os.path.abspath(vsrc), os.path.join(out_dir, "videos"))

    acc = {"action": [], "observation.state": []}     # global p/ stats
    epi = {"action": {}, "observation.state": {}}       # por episódio
    files = sorted(glob.glob(os.path.join(in_dir, "data", "**", "*.parquet"), recursive=True))
    max_resid = 0.0
    for f in files:
        out_f = os.path.join(out_dir, os.path.relpath(f, in_dir))
        os.makedirs(os.path.dirname(out_f), exist_ok=True)
        df = pq.read_table(f).to_pandas()
        for col in ("action",):
            v = np.stack(df[col].to_numpy()).astype(np.float32)[:, :28]
            nv, r = compress_vec(v)
            max_resid = max(max_resid, r)
            df[col] = list(nv)
            acc[col].append(nv)
            eps = df["episode_index"].to_numpy()
            for ep in np.unique(eps):
                epi[col].setdefault(int(ep), []).append(nv[eps == ep])
        df.to_parquet(out_f, index=False)
    print(f"[ok] {len(files)} parquets | resíduo máx = {max_resid:.6f} (0 = lossless)")

    arm_names = json.load(open(os.path.join(in_dir, "meta/info.json")))["features"]["action"]["names"][:14]
    new_names = arm_names + NEW_HAND_NAMES

    # info.json
    info_p = os.path.join(out_dir, "meta/info.json"); info = json.load(open(info_p))
    for col in ("action",):
        info["features"][col]["names"] = list(new_names)
        info["features"][col]["shape"] = [18]
    json.dump(info, open(info_p, "w"), indent=4)

    # stats.json (recomputado inteiro p/ as 18 dims)
    stats_p = os.path.join(out_dir, "meta/stats.json"); stats = json.load(open(stats_p))
    for col in ("action",):
        data = np.concatenate(acc[col], axis=0)
        stats[col] = {k: [col_stats(data[:, j])[k][0] for j in range(18)] for k in STAT_KEYS}
    json.dump(stats, open(stats_p, "w"), indent=4)

    # meta/episodes/* (stats por episódio)
    for ef in glob.glob(os.path.join(out_dir, "meta/episodes", "**", "*.parquet"), recursive=True):
        edf = pq.read_table(ef).to_pandas()
        for col in ("action",):
            for k in STAT_KEYS:
                key = f"stats/{col}/{k}"
                if key not in edf.columns:
                    continue
                edf[key] = [np.array([col_stats(np.concatenate(epi[col][int(r['episode_index'])])[:, j])[k][0]
                                      for j in range(18)], dtype=np.float32) for _, r in edf.iterrows()]
        edf.to_parquet(ef, index=False)
    print(f"[ok] action -> 18 dims | observation.state mantido em 28 (fiel)")
    g = np.concatenate(acc["action"], axis=0)
    print(f"   action grasp means: sqL={g[:,14].mean():.3f} sqR={g[:,15].mean():.3f} "
          f"trL={g[:,16].mean():.3f} trR={g[:,17].mean():.3f}")
    print(f"✅ pronto: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
