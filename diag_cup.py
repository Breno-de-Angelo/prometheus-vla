#!/usr/bin/env python3
# Diagnostico do detector de copo: carrega frames de reach e mostra escala/stats
# da imagem + contagem de pixels "brancos" em varios thresholds.
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent / "lerobot-ext"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset

RID = sys.argv[1]; ROOT = sys.argv[2]
IMG = "observation.images.head_camera"
ds = LeRobotDataset(RID, root=ROOT, episodes=list(range(214)), video_backend="pyav")
ep = np.asarray(ds.hf_dataset.select_columns(["episode_index"])["episode_index"])
acts = np.asarray(ds.hf_dataset.select_columns(["action"])["action"], dtype=np.float32)
sq = acts[:, 7]
sel = []
for e in range(214):
    fr = np.where(ep == e)[0]
    if len(fr) < 6: continue
    lo, hi = fr[int(0.25*len(fr))], fr[int(0.55*len(fr))]
    cand = [int(i) for i in range(lo, hi+1) if sq[i] < 0.1]
    if cand: sel.append(cand[len(cand)//2])
sel = sel[:8]
print(f"frames diag: {len(sel)}")
for i in sel:
    a = ds[i][IMG].detach().cpu().numpy()
    if a.ndim == 3 and a.shape[0] in (1, 3): a = np.transpose(a, (1, 2, 0))
    a = a.astype(np.float32)
    scale = "0-1" if a.max() <= 1.01 else "0-255"
    if a.max() <= 1.01: a = a*255
    mx = a.max(2); mn = a.min(2); sat = mx-mn
    print(f"\nframe {i}: shape={a.shape} scale={scale} min={a.min():.0f} max={a.max():.0f} mean={a.mean():.0f}")
    for tb in (120, 140, 160, 180):
        for ts in (40, 60, 90):
            m = (mx > tb) & (sat < ts)
            print(f"   bright>{tb} sat<{ts}: {int(m.sum())} px", end="")
        print()
