#!/usr/bin/env python
"""Audita a distribuição da MÃO (squeeze) no dataset de treino right14, treino vs validação,
pra testar a hipótese de DOMAIN SHIFT do split sequencial (eps 0-213 treino, 214-237 val).

Os 7 dedos são 1 DOF: dedos = squeeze(0..1) × RIGHT_TARGET. Recuperamos squeeze = action[:,10]/1.57
(index_0 / alvo fechado). Mede por episódio: quando fecha (fração do ep), magnitude, duração; e
testa: (a) drift do timing ao longo do índice do episódio, (b) treino vs val, (c) controle
adversarial — o gap treino↔val é explicado pelo split SEQUENCIAL (vs split aleatório)?
"""
import glob, json
import numpy as np
import pandas as pd

DS = "/home/luiz-aumo/I2CA/prometheus-vla/lerobot-ext/datasets/G1_Dex3_right14_dataset/v3_238ep"
RIGHT_TARGET = np.array([0.0, -0.920, -1.74, 1.57, 1.74, 1.57, 1.74])
TRAIN_EPS = set(range(0, 214))   # do config: 0-213
VAL_EPS = set(range(214, 238))   # do config: 214-237


def load_all():
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DS}/data/chunk-*/file-*.parquet"))],
                   ignore_index=True)
    out = {}
    for ep, g in df.groupby("episode_index"):
        a = np.vstack(g.sort_values("frame_index")["action"].values).astype(float)
        out[int(ep)] = a
    return out


def squeeze_of(a):
    # index_0 (dim 10) / alvo 1.74? não — usa dim 10 (index_1=1.74) ou dim 9? RIGHT_TARGET[3]=1.57 (index_0=dim10)
    # dims dos dedos = action[:,7:14]; RIGHT_TARGET ordena [thumb0,thumb1,thumb2,index0,index1,middle0,middle1]
    # usa o de maior alvo p/ recuperar squeeze de forma robusta (index_1 = 1.74 = dim 11)
    fin = a[:, 7:14]
    # mínimos quadrados: squeeze = (fin·target)/(target·target)
    tt = RIGHT_TARGET @ RIGHT_TARGET
    return np.clip(fin @ RIGHT_TARGET / tt, 0, 1)


def ep_stats(a):
    sq = squeeze_of(a)
    n = len(sq)
    closed = sq > 0.5
    close_frac = float(np.argmax(closed)) / n if closed.any() else None  # quando cruza 0.5
    return {
        "n": n,
        "close_frac": close_frac,                       # fração do ep em que fecha
        "max_sq": float(sq.max()),
        "dur_closed": float(closed.mean()),             # fração do ep com mão fechada
        "sq_curve": np.interp(np.linspace(0, 1, 50), np.linspace(0, 1, n), sq),  # perfil resampleado
    }


def main():
    eps = load_all()
    stats = {e: ep_stats(a) for e, a in eps.items()}
    idx = sorted(stats)
    cf = np.array([stats[e]["close_frac"] if stats[e]["close_frac"] is not None else np.nan for e in idx])
    maxsq = np.array([stats[e]["max_sq"] for e in idx])
    dur = np.array([stats[e]["dur_closed"] for e in idx])
    idxa = np.array(idx)

    tr = np.array([e in TRAIN_EPS for e in idx])
    va = np.array([e in VAL_EPS for e in idx])

    def mstd(x, m):
        x = x[m]; x = x[~np.isnan(x)]
        return x.mean(), x.std(), len(x)

    print("=== TIMING DE FECHAMENTO (close_frac = fração do ep em que a mão cruza squeeze>0.5) ===")
    for nm, m in [("TREINO (0-213)", tr), ("VAL (214-237)", va)]:
        mu, sd, k = mstd(cf, m)
        print(f"  {nm:<16} média={mu:.3f}  std={sd:.3f}  (n={k})")
    print("\n=== MAGNITUDE/DURAÇÃO ===")
    for nm, m in [("TREINO", tr), ("VAL", va)]:
        print(f"  {nm:<8} max_squeeze méd={maxsq[m].mean():.3f}  dur_fechada méd={dur[m].mean():.3f}")

    # DRIFT temporal: correlação close_frac vs índice do episódio
    ok = ~np.isnan(cf)
    r = np.corrcoef(idxa[ok], cf[ok])[0, 1]
    print(f"\n=== DRIFT: correlação (close_frac vs índice do episódio) = {r:+.3f} ===")
    print("  (>0 = fecha cada vez MAIS TARDE nos episódios finais; <0 = mais cedo)")
    # binned
    print("  por faixa de índice:")
    for lo in range(0, 238, 40):
        m = (idxa >= lo) & (idxa < lo + 40) & ok
        if m.any():
            print(f"    eps {lo:>3}-{lo+39:<3}: close_frac méd={cf[m].mean():.3f} (n={m.sum()})")

    # CONTROLE ADVERSARIAL: o gap treino↔val do PERFIL de squeeze é explicado pelo split sequencial?
    curves = np.array([stats[e]["sq_curve"] for e in idx])  # [238,50]
    def profile_gap(mask_a, mask_b):
        return float(np.linalg.norm(curves[mask_a].mean(0) - curves[mask_b].mean(0)))
    seq_gap = profile_gap(tr, va)
    rng = np.random.RandomState(0)
    rand_gaps = []
    nval = va.sum()
    for _ in range(2000):
        perm = rng.permutation(len(idx))
        mb = np.zeros(len(idx), bool); mb[perm[:nval]] = True
        rand_gaps.append(profile_gap(~mb, mb))
    rand_gaps = np.array(rand_gaps)
    pct = float((rand_gaps < seq_gap).mean() * 100)
    print(f"\n=== CONTROLE ADVERSARIAL (perfil de squeeze treino vs val) ===")
    print(f"  gap do split SEQUENCIAL (0-213 | 214-237): {seq_gap:.4f}")
    print(f"  gap de splits ALEATÓRIOS (2000x): média={rand_gaps.mean():.4f} p95={np.percentile(rand_gaps,95):.4f}")
    print(f"  -> o split sequencial é maior que {pct:.1f}% dos splits aleatórios")
    print(f"  (se ~50% = sem domain shift; se >95% = val É sistematicamente diferente = domain shift real)")

    # episódios SEM fechamento (ruído de label?)
    noclose = [e for e in idx if stats[e]["close_frac"] is None]
    print(f"\n=== episódios que NUNCA fecham a mão: {len(noclose)} -> {noclose[:20]} ===")


if __name__ == "__main__":
    main()
