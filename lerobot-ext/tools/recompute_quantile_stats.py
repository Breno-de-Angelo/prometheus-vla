"""Recomputa stats exatos (min/max/mean/std/quantis) das features vetoriais de um
dataset LeRobot merged, lendo os parquets, e corrige o meta/stats.json (que o
merge_datasets agrega por média ponderada de quantis — aproximação)."""
import glob, json, shutil, sys
import numpy as np
import pyarrow.parquet as pq

root = sys.argv[1]
KEYS = ["action", "observation.state",
        "observation.left_hand_pressure", "observation.right_hand_pressure"]
QS = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}

files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
assert files, f"sem parquets em {root}/data"
cols = {k: [] for k in KEYS}
n = 0
for f in files:
    t = pq.read_table(f, columns=KEYS)
    n += t.num_rows
    for k in KEYS:
        cols[k].append(np.stack(t[k].to_numpy(zero_copy_only=False)))
print(f"[lidos] {len(files)} parquets, {n} frames")

stats_path = f"{root}/meta/stats.json"
shutil.copy(stats_path, stats_path + ".pre_exact_quantiles.bak")
stats = json.load(open(stats_path))

for k in KEYS:
    x = np.concatenate(cols[k], axis=0).astype(np.float64)  # (N, D)
    new = {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0),
           "count": [x.shape[0]]}
    for qn, qv in QS.items():
        new[qn] = np.quantile(x, qv, axis=0)
    old = stats.get(k, {})
    if "q01" in old:
        do, dn = np.array(old["q01"], float), new["q01"]
        i = int(np.argmax(np.abs(do - dn)))
        print(f"[{k}] maior delta q01: dim {i}: {do[i]:.4f} -> {dn[i]:.4f}")
        do, dn = np.array(old["q99"], float), new["q99"]
        i = int(np.argmax(np.abs(do - dn)))
        print(f"[{k}] maior delta q99: dim {i}: {do[i]:.4f} -> {dn[i]:.4f}")
    stats[k] = {s: np.asarray(v).tolist() for s, v in new.items()}

json.dump(stats, open(stats_path, "w"), indent=4)
print(f"[ok] stats.json atualizado ({stats_path}); backup .pre_exact_quantiles.bak")
