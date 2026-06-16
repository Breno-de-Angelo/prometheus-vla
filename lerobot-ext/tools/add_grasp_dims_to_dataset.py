#!/usr/bin/env python3
"""Adiciona retroativamente as 4 dims de grasp do controle a um dataset LeRobot já gravado.

As dims de mão no dataset são, EXATAMENTE (resíduo 0), uma combinação linear de dois
sinais do controle VR (ver teleop/xr_g1_arm.py get_action):

    hand_q[i] = squeeze * TARGET[i] + trigger * W[i]

Como TARGET e W são conhecidos, recuperamos (squeeze, trigger) por least-squares por frame
(2 incógnitas, 7 equações) — exato pros datasets gravados com esse código. Anexamos 4 dims
ao vetor `action` (idx 28-31), batendo o schema das gravações novas:

    28 left_grasp_squeeze   29 right_grasp_squeeze   30 left_grasp_trigger   31 right_grasp_trigger

Só mexe em `action` (não em observation.state). Recomputa stats.json (global, com quantis) e
meta/episodes/*.parquet (stats por episódio). Vídeos são reaproveitados via symlink (não mudam).

Uso:
    python tools/add_grasp_dims_to_dataset.py IN_DIR OUT_DIR
"""
import sys, os, json, shutil, glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# --- pesos exatos de teleop/xr_g1_arm.py (modo controller) ---
RIGHT_TARGET = np.array([0.0, -0.920, -1.74, 1.57, 1.74, 1.57, 1.74])
LEFT_TARGET  = np.array([0.0,  0.920,  1.74,-1.57,-1.74,-1.57,-1.74])
W_RIGHT = np.array([-0.5, -0.8, -0.8, 0.0, 0.0, 2.2, 2.1])   # trigger: thumb0=-0.5,thumb1/2=-0.8,mid0=+2.2,mid1=+2.1
W_LEFT  = np.array([-0.5,  0.8,  0.8, 0.0, 0.0,-2.2,-2.1])

NEW_NAMES = ["left_grasp_squeeze.q", "right_grasp_squeeze.q",
             "left_grasp_trigger.q", "right_grasp_trigger.q"]
STAT_KEYS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]


def recover(qhand, TARGET, W):
    """Resolve [squeeze, trigger] por least-squares; retorna (N,) cada, clip [0,1]."""
    A = np.stack([TARGET, W], axis=1)        # (7,2)
    M = np.linalg.pinv(A)                      # (2,7)
    sol = qhand @ M.T                          # (N,2)
    recon = sol @ A.T
    resid = float(np.abs(recon - qhand).max())
    sq = np.clip(sol[:, 0], 0.0, 1.0)
    tr = np.clip(sol[:, 1], 0.0, 1.0)
    return sq, tr, resid


def grasp4(action):
    """action (N,28) -> 4 colunas (N,4) na ordem do schema novo."""
    lq, rq = action[:, 14:21], action[:, 21:28]
    sqL, trL, rL = recover(lq, LEFT_TARGET, W_LEFT)
    sqR, trR, rR = recover(rq, RIGHT_TARGET, W_RIGHT)
    return np.stack([sqL, sqR, trL, trR], axis=1), max(rL, rR)


def col_stats(x):
    """x (N,) -> dict de stats no formato LeRobot (listas de 1 elem)."""
    return {
        "min":  [float(x.min())], "max": [float(x.max())],
        "mean": [float(x.mean())], "std": [float(x.std())],
        "count": [int(x.shape[0])],
        "q01": [float(np.quantile(x, 0.01))], "q10": [float(np.quantile(x, 0.10))],
        "q50": [float(np.quantile(x, 0.50))], "q90": [float(np.quantile(x, 0.90))],
        "q99": [float(np.quantile(x, 0.99))],
    }


def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # --- meta: copia tudo, depois reescrevemos info/stats/episodes ---
    shutil.copytree(os.path.join(in_dir, "meta"), os.path.join(out_dir, "meta"), dirs_exist_ok=True)
    # --- videos: symlink (não mudam, evita duplicar GBs) ---
    vsrc = os.path.join(in_dir, "videos")
    if os.path.isdir(vsrc):
        vdst = os.path.join(out_dir, "videos")
        if not os.path.exists(vdst):
            os.symlink(os.path.abspath(vsrc), vdst)

    # --- data: reescreve cada parquet com action 28->32 ---
    grasp_all = []           # acumula 4 sinais de TODOS os frames (stats global)
    epi_grasp = {}           # episode_index -> lista de (N,4) p/ stats por episódio
    files = sorted(glob.glob(os.path.join(in_dir, "data", "**", "*.parquet"), recursive=True))
    max_resid = 0.0
    for f in files:
        rel = os.path.relpath(f, in_dir)
        out_f = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(out_f), exist_ok=True)
        df = pq.read_table(f).to_pandas()
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        g4, resid = grasp4(action)
        max_resid = max(max_resid, resid)
        new_action = np.concatenate([action, g4.astype(np.float32)], axis=1)  # (N,32)
        df["action"] = list(new_action)
        df.to_parquet(out_f, index=False)
        grasp_all.append(g4)
        for ep in np.unique(df["episode_index"].to_numpy()):
            m = df["episode_index"].to_numpy() == ep
            epi_grasp.setdefault(int(ep), []).append(g4[m])
    grasp_all = np.concatenate(grasp_all, axis=0)
    print(f"[ok] {len(files)} parquets | resíduo máx de reconstrução = {max_resid:.6f} (0 = exato)")

    # --- info.json: action shape 28->32 + names ---
    info_p = os.path.join(out_dir, "meta", "info.json")
    info = json.load(open(info_p))
    a = info["features"]["action"]
    if a["names"][-4:] != NEW_NAMES:
        a["names"] = list(a["names"]) + NEW_NAMES
        a["shape"] = [len(a["names"])]
    json.dump(info, open(info_p, "w"), indent=4)
    print(f"[ok] info.json: action shape -> {a['shape']}")

    # --- stats.json: estende os arrays de action de 28 -> 32 ---
    stats_p = os.path.join(out_dir, "meta", "stats.json")
    stats = json.load(open(stats_p))
    ast = stats["action"]
    for j, nm in enumerate(NEW_NAMES):
        cs = col_stats(grasp_all[:, j])
        for k in STAT_KEYS:
            ast[k] = list(ast[k]) + cs[k]
    json.dump(stats, open(stats_p, "w"), indent=4)
    print(f"[ok] stats.json: action min len -> {len(ast['min'])}")

    # --- meta/episodes/*.parquet: estende stats/action/* por episódio ---
    epi_files = sorted(glob.glob(os.path.join(out_dir, "meta", "episodes", "**", "*.parquet"), recursive=True))
    for ef in epi_files:
        edf = pq.read_table(ef).to_pandas()
        for k in STAT_KEYS:
            col = f"stats/action/{k}"
            if col not in edf.columns:
                continue
            newvals = []
            for _, row in edf.iterrows():
                ep = int(row["episode_index"])
                base = list(np.asarray(row[col], dtype=float))
                g = np.concatenate(epi_grasp[ep], axis=0)  # (Nep,4)
                ext = [col_stats(g[:, j])[k][0] for j in range(4)]
                newvals.append(np.array(base + ext, dtype=np.float32))
            edf[col] = newvals
        edf.to_parquet(ef, index=False)
    print(f"[ok] {len(epi_files)} meta/episodes parquet(s) estendidos")
    print(f"\n✅ pronto: {out_dir}")
    print(f"   squeeze_R: mean={grasp_all[:,1].mean():.3f}  trigger_R: mean={grasp_all[:,3].mean():.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
