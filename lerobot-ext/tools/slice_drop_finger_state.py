#!/usr/bin/env python3
"""Experimento "tirar os dedos do state": de um dataset right8 (state=14, action=8),
remove os 7 DEDOS MEDIDOS do observation.state -> state=7 (braço-only). A ACTION fica
INTACTA (7 braço + 1 squeeze). Recomputa stats/episodes pra a nova dim.

Motivo: o pi05 injeta o state como TEXTO discretizado no prompt; tirar os dedos do state
remove o atalho proprioceptivo da mão (ela fechava ecoando os dedos medidos). RGB-only.

Uso:  python slice_drop_finger_state.py <right8_dataset_dir> [--out <dir>]
      default out: <pai>/<nome>_armstate7
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np, pandas as pd

STATE_KEEP = [0, 1, 2, 3, 4, 5, 6]   # braço direito; descarta 7-13 (dedos medidos)
N_STATE = len(STATE_KEEP)            # 7
SRC_STATE_DIM = 14
QMIN = 1e-3


def _slice_state(series):
    return series.apply(lambda a: np.array(a, dtype=np.float32)[STATE_KEEP])

def _slice_list(v):
    return np.array(v, dtype=np.float32)[STATE_KEEP].tolist()

def _guard(stats, key):
    q01, q99 = stats.get("q01"), stats.get("q99")
    if isinstance(q01, list) and isinstance(q99, list):
        for d, (lo, hi) in enumerate(zip(q01, q99)):
            if hi - lo < QMIN:
                print(f"  [AVISO] {key} dim {d}: range~0 -> q99=q01+1")
                stats["q99"][d] = lo + 1.0

def proc_data(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for pf in sorted(src.glob("**/*.parquet")):
        out = dst / pf.relative_to(src); out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(pf)
        if "observation.state" in df.columns:
            df["observation.state"] = _slice_state(df["observation.state"])
        df.to_parquet(out, index=False)
    print("  data/: observation.state 14->7 (action intacta)")

def proc_info(src, dst, names):
    j = json.loads((src / "meta/info.json").read_text())
    st = j["features"]["observation.state"]
    st["shape"] = [N_STATE]; st["names"] = names
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "meta/info.json").write_text(json.dumps(j, indent=2))
    print(f"  meta/info.json: state shape->[{N_STATE}]")

def proc_stats(src, dst):
    s = json.loads((src / "meta/stats.json").read_text())
    if "observation.state" in s:
        for k, v in s["observation.state"].items():
            if isinstance(v, list) and len(v) == SRC_STATE_DIM:
                s["observation.state"][k] = _slice_list(v)
        _guard(s["observation.state"], "observation.state")
    (dst / "meta/stats.json").write_text(json.dumps(s, indent=2))
    print("  meta/stats.json: state stats 14->7 (action intacta)")

def proc_episodes(src, dst):
    ed = src / "meta/episodes"; od = dst / "meta/episodes"; od.mkdir(parents=True, exist_ok=True)
    for pf in sorted(ed.glob("**/*.parquet")):
        out = od / pf.relative_to(ed); out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(pf)
        for col in df.columns:
            if "stats/observation.state/" in col:
                df[col] = df[col].apply(
                    lambda v: np.array(v, dtype=np.float32)[STATE_KEEP]
                    if isinstance(v, (list, np.ndarray)) and len(v) == SRC_STATE_DIM else v)
        df.to_parquet(out, index=False)
    print("  meta/episodes: state stats 14->7")

def proc_videos(src, dst):
    vs = src / "videos"
    if not vs.exists(): return
    vd = dst / "videos"
    if vd.exists() or vd.is_symlink():
        vd.unlink() if vd.is_symlink() else shutil.rmtree(vd)
    vd.symlink_to(vs.resolve()); print(f"  videos -> symlink {vs.resolve()}")

def proc_tasks(src, dst):
    t = src / "meta/tasks.parquet"
    if t.exists(): shutil.copy2(t, dst / "meta/tasks.parquet"); print("  tasks.parquet copiado")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir"); ap.add_argument("--out")
    a = ap.parse_args()
    src = Path(a.dataset_dir).expanduser().resolve()
    dst = Path(a.out).expanduser().resolve() if a.out else src.parent / (src.name + "_armstate7")
    print(f"Origem : {src}\nDestino: {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    j = json.loads((src / "meta/info.json").read_text())
    sn = j["features"]["observation.state"].get("names", [])
    assert isinstance(sn, list) and len(sn) == SRC_STATE_DIM, f"names inesperado: {sn}"
    names = [sn[i] for i in STATE_KEEP]
    print(f"State ({N_STATE}, mantidos): {names}")
    print(f"Dedos removidos: {[sn[i] for i in range(7,14)]}")
    print("[1/5] data...");     proc_data(src / "data", dst / "data")
    print("[2/5] info...");     proc_info(src, dst, names)
    print("[3/5] stats...");    proc_stats(src, dst)
    print("[4/5] episodes...");  proc_episodes(src, dst)
    print("[5/5] videos+tasks..."); proc_videos(src, dst); proc_tasks(src, dst)
    print(f"\n✅ Dataset state[7] (braço-only) em:\n  {dst}")

if __name__ == "__main__":
    main()
