"""Repara data/chunk_index e data/file_index no meta/episodes de um dataset LeRobot v3,
reconstruindo o mapeamento real episodio->arquivo a partir dos parquets de dados.
Também valida refs de vídeo (existência do arquivo) e self-refs do meta.
Uso: python repair_refs.py ROOT [--dry-run]"""
import glob, os, re, sys
import pandas as pd
import pyarrow.parquet as pq

root = sys.argv[1]
dry = "--dry-run" in sys.argv

# 1. mapa real: episode_index -> (chunk, file) a partir dos dados
ep_to_loc = {}
for p in sorted(glob.glob(f"{root}/data/chunk-*/file-*.parquet")):
    m = re.search(r"chunk-(\d+)/file-(\d+)\.parquet", p)
    c, f = int(m.group(1)), int(m.group(2))
    eps = pq.read_table(p, columns=["episode_index"])["episode_index"].to_pandas().unique()
    for e in eps:
        assert e not in ep_to_loc, f"ep {e} em dois arquivos! {ep_to_loc[e]} e {(c,f)}"
        ep_to_loc[int(e)] = (c, f)
print(f"[data] {len(ep_to_loc)} episódios mapeados em {root}/data")

# 2. reescreve meta/episodes
n_fix = 0
for p in sorted(glob.glob(f"{root}/meta/episodes/chunk-*/file-*.parquet")):
    df = pd.read_parquet(p)
    before = df[["data/chunk_index", "data/file_index"]].copy()
    df["data/chunk_index"] = df["episode_index"].map(lambda e: ep_to_loc[int(e)][0])
    df["data/file_index"] = df["episode_index"].map(lambda e: ep_to_loc[int(e)][1])
    changed = (before != df[["data/chunk_index", "data/file_index"]]).any(axis=1).sum()
    n_fix += int(changed)
    # valida self-ref do meta
    m = re.search(r"chunk-(\d+)/file-(\d+)\.parquet", p)
    bad_self = ((df["meta/episodes/chunk_index"] != int(m.group(1))) |
                (df["meta/episodes/file_index"] != int(m.group(2)))).sum()
    if bad_self:
        print(f"  [meta-self] {p}: {bad_self} self-refs erradas — corrigindo")
        df["meta/episodes/chunk_index"] = int(m.group(1))
        df["meta/episodes/file_index"] = int(m.group(2))
    # valida refs de vídeo (existência)
    for col in [c for c in df.columns if c.startswith("videos/") and c.endswith("file_index")]:
        key = col.split("/")[1]
        ccol = f"videos/{key}/chunk_index"
        miss = 0
        for _, r in df.iterrows():
            vp = f"{root}/videos/{key}/chunk-{int(r[ccol]):03d}/file-{int(r[col]):03d}.mp4"
            if not os.path.exists(vp):
                miss += 1
        if miss:
            print(f"  [video] {p}: {miss} refs de {key} para mp4 INEXISTENTE — NÃO corrigido (investigar)")
    if not dry:
        df.to_parquet(p, index=False)
print(f"[ok] {n_fix} episódios com data refs corrigidas" + (" (dry-run, nada gravado)" if dry else ""))
