#!/usr/bin/env python
"""Merge os dois datasets do copo (pick_up_the_cup_2026-04-30 + pick_up_the_cup3) num
dataset LeRobot ÚNICO, pra ter mais dados no finetune do pi05-D (~48 eps / ~10.5k frames).

NÃO-DESTRUTIVO: só LÊ os datasets de origem (o do ACT fica quietinho) e ESCREVE um
dataset NOVO em AGGR_ROOT. Não move nem altera os originais — não precisa copiar nada.

Mesma receita do merge_picks7.py: aggregate_datasets() + ensure_quantiles() (o pi05
normaliza state/action com QUANTIS; o aggregate só escreve min/max/mean/std).
Idempotente: pula se o merge já existir. Não dá push (fica local).
"""
import os, sys, traceback
os.environ.setdefault("HF_HOME", "/data/huggingface-models")
os.environ.setdefault("HF_LEROBOT_HOME", "/data/huggingface-models/lerobot")
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.aggregate import aggregate_datasets

# Fontes — SOMENTE LEITURA, não são tocadas
SRC_REPOS = ["Mrwlker/pick_up_the_cup_2026-04-30", "Mrwlker/pick_up_the_cup3"]
# Saída — dataset NOVO
AGGR_REPO = "lewislf/pick_up_the_cup_merged"
AGGR_ROOT = Path("/data/huggingface-models/lerobot/lewislf/pick_up_the_cup_merged")


def ensure_quantiles() -> None:
    """aggregate_datasets() só escreve min/max/mean/std, mas o pi05 normaliza STATE e ACTION
    com QUANTIS (q01/q10/q50/q90/q99) -- sem eles o treino morre no normalizer. Calcula do
    parquet (state+action; VISUAL é IDENTITY no pi05, sem decodificar vídeo). Idempotente.
    torchcodec quebra no ms3 -> força video_backend=pyav."""
    import glob
    import numpy as np
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import lerobot.datasets.v30.augment_dataset_quantile_stats as A

    ds = LeRobotDataset(AGGR_REPO, root=str(AGGR_ROOT), video_backend="pyav")
    if A.has_quantile_stats(ds.meta.stats):
        print("[merge] quantis já presentes", flush=True)
        return
    print("[merge] adicionando quantis de state/action (pi05 precisa)", flush=True)
    stats = ds.meta.stats
    files = sorted(glob.glob(str(AGGR_ROOT) + "/data/**/*.parquet", recursive=True))
    Q = [0.01, 0.10, 0.50, 0.90, 0.99]
    NAMES = ["q01", "q10", "q50", "q90", "q99"]
    for col in ("observation.state", "action"):
        data = np.concatenate(
            [np.stack(pd.read_parquet(f, columns=[col])[col].to_numpy()).astype(np.float64) for f in files], 0
        )
        ref = np.asarray(stats[col]["min"])
        qv = np.quantile(data, Q, axis=0)
        for i, nm in enumerate(NAMES):
            stats[col][nm] = qv[i].reshape(ref.shape).astype(ref.dtype)
    A.write_stats(stats, ds.meta.root)
    print("[merge] quantis escritos", flush=True)


def main() -> int:
    if (AGGR_ROOT / "meta" / "info.json").exists():
        print(f"[merge] já existe em {AGGR_ROOT} -> pulando aggregate", flush=True)
        ensure_quantiles()
        return 0

    # Garante que as fontes estão acessíveis (não re-baixa se já estão na lerobot home). SÓ LEITURA.
    for r in SRC_REPOS:
        try:
            LeRobotDataset(r)
            print(f"[merge] fonte ok (leitura): {r}", flush=True)
        except Exception:
            print(f"[merge] não consegui abrir {r}", flush=True)
            traceback.print_exc()
            return 2

    try:
        print(f"[merge] agregando {SRC_REPOS} -> {AGGR_REPO}", flush=True)
        aggregate_datasets(repo_ids=SRC_REPOS, aggr_repo_id=AGGR_REPO, aggr_root=AGGR_ROOT)
    except Exception:
        print("[merge] aggregate_datasets FALHOU", flush=True)
        traceback.print_exc()
        return 3

    ensure_quantiles()
    ok = (AGGR_ROOT / "meta" / "info.json").exists()
    print(f"[merge] pronto. info.json presente={ok} -> {AGGR_ROOT}", flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
