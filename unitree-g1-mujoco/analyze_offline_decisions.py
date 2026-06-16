#!/usr/bin/env python
"""Audita as decisões da VLA numa run OFFLINE (replay aberto) vs a ação REAL do dataset.

A VLA, no frame t, viu state[t]+image[t] do dataset e decidiu um chunk. Comparamos a 1ª
ação decidida (chunk.actions[0]) com a ação REAL do dataset naquele frame — quanto a VLA
"acerta" o que o humano fez. Foca nas 7 juntas do braço (diretas) e nos 7 dedos
(reconstruindo os dedos do dataset right8 a partir do squeeze, se for o caso).

Uso:
  python analyze_offline_decisions.py --run /caminho/run_<ts> \\
      --dataset-root ../datasets/_merged/G1_Dex3_pick_white_cup_right8_1squeeze --episode 0
"""
import argparse, json, glob
import numpy as np
import pandas as pd

ARM = ["ShPitch", "ShRoll", "ShYaw", "Elbow", "WrRoll", "WrPitch", "WrYaw"]
FINGERS = ["thumb0", "thumb1", "thumb2", "index0", "index1", "middle0", "middle1"]
RIGHT_TARGET = np.array([0.0, -0.920, -1.74, 1.57, 1.74, 1.57, 1.74])  # squeeze -> dedos (right8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir run_<ts> com chunks.jsonl")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--align", type=int, default=0, help="offset de frame VLA->dataset (latência)")
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(f"{args.run}/chunks.jsonl")]
    print(f"run: {len(chunks)} inferências")

    # dataset: ação real + state do episódio
    meta = pd.concat([pd.read_parquet(f) for f in
                      sorted(glob.glob(f"{args.dataset_root}/meta/episodes/chunk-*/*.parquet"))], ignore_index=True)
    row = meta[meta["episode_index"] == args.episode].iloc[0]
    dci, dfi = int(row["data/chunk_index"]), int(row["data/file_index"])
    df = pd.read_parquet(f"{args.dataset_root}/data/chunk-{dci:03d}/file-{dfi:03d}.parquet")
    df = df[df["episode_index"] == args.episode] if "episode_index" in df.columns else df
    real_act = np.vstack(df["action"].values).astype(np.float32)   # [n, 8 ou 14]
    n_ds = len(real_act)
    # reconstrói dedos do dataset (se right8: 7 braço + 1 squeeze -> 7 dedos)
    if real_act.shape[1] == 8:
        sq = np.clip(real_act[:, 7:8], 0, 1)
        real_full = np.concatenate([real_act[:, :7], sq * RIGHT_TARGET[None, :]], axis=1)  # [n,14]
    else:
        real_full = real_act[:, :14]

    # VLA: 1ª ação decidida por inferência. O state_raw guarda o frame que a VLA viu;
    # como o replay é frame-a-frame, casamos por ÍNDICE de inferência ~ frame (com --align).
    vla = np.array([c["actions"][0] for c in chunks])[:, :14]  # [m,14]
    m = min(len(vla), n_ds)
    err = []
    print(f"\n=== ERRO MÉDIO |VLA - dataset| por junta (rad), {m} frames ===")
    print(f"{'junta':>8} {'erro_med':>9} {'erro_p90':>9} {'real_range':>12} {'vla_range':>12}")
    for i, nm in enumerate(ARM + FINGERS):
        a = vla[:m, i]
        b = real_full[args.align:args.align + m, i] if args.align else real_full[:m, i]
        b = b[:len(a)]
        e = np.abs(a[:len(b)] - b)
        err.append(e.mean())
        print(f"{nm:>8} {e.mean():>9.3f} {np.percentile(e,90):>9.3f} "
              f"[{b.min():>5.2f},{b.max():>5.2f}] [{a.min():>5.2f},{a.max():>5.2f}]")
    print(f"\nerro médio braço (7 juntas): {np.mean(err[:7]):.3f} rad | dedos: {np.mean(err[7:]):.3f} rad")

    # quão bem o braço SEGUE o reach? correlação por junta
    print(f"\n=== correlação VLA vs dataset (1.0 = segue perfeito) ===")
    for i, nm in enumerate(ARM):
        a = vla[:m, i]; b = real_full[:m, i][:len(a)]
        if a.std() > 1e-6 and b.std() > 1e-6:
            c = np.corrcoef(a[:len(b)], b)[0, 1]
            print(f"  {nm}: {c:+.2f}")

    # a VLA fecha a mão na mesma FASE? (index0 cruza 0.7)
    vsq = vla[:m, 10]; rsq = real_full[:m, 10][:len(vsq)]
    def first_close(x):
        c = x > 0.7
        return int(np.argmax(c)) / len(x) if c.any() else None
    print(f"\n=== timing do fechamento (fração do episódio em que index0>0.7) ===")
    print(f"  VLA: {first_close(vsq)}  | dataset: {first_close(rsq)}")


if __name__ == "__main__":
    main()
