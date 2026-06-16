#!/usr/bin/env python
"""EVAL OFFLINE de um episódio (teacher forcing / loop ABERTO): compara a AÇÃO que a VLA
decidiu (vendo o state+imagem REAIS do dataset) com a AÇÃO REAL da demo, junta a junta.

Casa cada decisão da VLA (chunks.jsonl: state_raw + actions[0]) ao frame do dataset por
ESTADO mais próximo (o replay roda o ep em loop, então índice não serve), pega a ação real
daquele frame e mede o erro. Reporta por junta: MAE, RMSE (rad e graus), correlação; e
agregado: MSE/RMSE do braço, dos dedos e total, + NMSE (normalizado pelo std do dataset,
comparável em espírito ao val_action_mse do treino) + timing do fechamento da mão.

Uso: python eval_offline_episode.py --chunks /tmp/cl_run/chunks_replay_aberto.jsonl \
        --dataset-root ../datasets/_merged/G1_Dex3_pick_white_cup_right8_1squeeze --episode 0
"""
import argparse, glob, json
import numpy as np
import pandas as pd

ARM = ["ShPitch", "ShRoll", "ShYaw", "Elbow", "WrRoll", "WrPitch", "WrYaw"]
FING = ["thumb0", "thumb1", "thumb2", "index0", "index1", "middle0", "middle1"]
# squeeze (right8, 0..1) -> 7 dedos (right14), alvo do grasp fechado
RIGHT_TARGET = np.array([0.0, -0.920, -1.74, 1.57, 1.74, 1.57, 1.74], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--episode", type=int, default=0)
    args = ap.parse_args()

    ch = [json.loads(l) for l in open(args.chunks) if l.strip()]
    vla = np.array([c["actions"][0] for c in ch], np.float32)[:, :14]     # [M,14]
    seen = np.array([c.get("state_raw", c["state"]) for c in ch], np.float32)  # [M,14]

    meta = pd.concat([pd.read_parquet(f) for f in
                      sorted(glob.glob(f"{args.dataset_root}/meta/episodes/chunk-*/*.parquet"))], ignore_index=True)
    row = meta[meta["episode_index"] == args.episode].iloc[0]
    dci, dfi = int(row["data/chunk_index"]), int(row["data/file_index"])
    df = pd.read_parquet(f"{args.dataset_root}/data/chunk-{dci:03d}/file-{dfi:03d}.parquet")
    df = df[df["episode_index"] == args.episode]
    ds_state = np.vstack(df["observation.state"].values).astype(np.float32)  # [n,14]
    ds_act = np.vstack(df["action"].values).astype(np.float32)              # [n,8] (right8)

    # ação real -> 14 dims (braço 7 + dedos reconstruídos do squeeze)
    if ds_act.shape[1] == 8:
        sq = np.clip(ds_act[:, 7:8], 0, 1)
        real = np.concatenate([ds_act[:, :7], sq * RIGHT_TARGET[None, :]], axis=1)  # [n,14]
    else:
        real = ds_act[:, :14]

    # casa cada decisão da VLA ao frame do dataset por ESTADO mais próximo
    idx = np.array([int(np.argmin(np.linalg.norm(ds_state - seen[k], axis=1))) for k in range(len(seen))])
    match_err = np.mean([np.linalg.norm(ds_state[idx[k]] - seen[k]) for k in range(len(idx))])
    real_m = real[idx]                                                      # [M,14] ação real casada

    print(f"\n=== EVAL OFFLINE — episódio {args.episode} (teacher forcing / loop ABERTO) ===")
    print(f"decisões da VLA: {len(vla)} | casadas ao dataset por estado (erro médio de match: {match_err:.4f} rad)")

    err = vla - real_m
    std = ds_state.std(0) + 1e-6      # normalizador (z-score do dataset, por junta)
    print(f"\n{'junta':>8} | {'MAE(rad)':>8} {'RMSE(rad)':>9} {'RMSE(°)':>8} | {'corr':>5} | {'real_range':>13} {'vla_range':>13}")
    print("-" * 88)
    for i, nm in enumerate(ARM + FING):
        a, b = vla[:, i], real_m[:, i]
        mae = np.abs(a - b).mean()
        rmse = np.sqrt(((a - b) ** 2).mean())
        corr = np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else float("nan")
        tag = "  <braço" if i == 0 else ("  <dedos" if i == 7 else "")
        print(f"{nm:>8} | {mae:8.3f} {rmse:9.3f} {np.degrees(rmse):8.1f} | {corr:+5.2f} | "
              f"[{b.min():5.2f},{b.max():5.2f}] [{a.min():5.2f},{a.max():5.2f}]{tag}")

    def agg(sl, label):
        e = err[:, sl]
        mse = (e ** 2).mean()
        rmse = np.sqrt(mse)
        nmse = ((e / std[sl]) ** 2).mean()    # normalizado pelo std do dataset
        print(f"  {label:<14} MSE={mse:.4f} rad²  RMSE={rmse:.3f} rad ({np.degrees(rmse):.1f}°)  NMSE(norm)={nmse:.4f}")

    print("\n=== AGREGADO ===")
    agg(slice(0, 7), "braço (7)")
    agg(slice(7, 14), "dedos (7)")
    agg(slice(0, 14), "total (14)")

    # timing do fechamento da mão (index0 cruza 0.7 — fração do episódio)
    def first_cross(x):
        c = x > 0.7
        return float(np.argmax(c)) / len(x) if c.any() else None
    # ordena por frame casado pra ter a sequência temporal do episódio
    order = np.argsort(idx)
    vla_seq, real_seq = vla[order, 10], real_m[order, 10]
    fv, fr = first_cross(vla_seq), first_cross(real_seq)
    print("\n=== TIMING do fechamento (index0>0.7, fração do episódio) ===")
    print(f"  VLA: {('%.0f%%'%(fv*100)) if fv is not None else 'nunca'}   "
          f"| demo real: {('%.0f%%'%(fr*100)) if fr is not None else 'nunca'}")


if __name__ == "__main__":
    main()
