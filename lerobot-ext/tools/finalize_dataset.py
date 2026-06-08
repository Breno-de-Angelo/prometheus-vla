#!/usr/bin/env python3
"""Finaliza/repara um dataset LeRobot v3 gravado pelo init_lerobot_record_v2.py.

Por que existe: o fluxo de --resume + batch encoding do LeRobot v3 é frágil e, ao
crashar no encoding, deixa o dataset com:
  - meta/episodes incompleta (faltam stats da maioria dos episódios)
  - vídeos não encodados (frames ainda como PNG em images/)
  - episódios não-contíguos (ex: ep54 descartado deixa buraco)

Os DADOS BRUTOS, porém, ficam íntegros: os parquets em data/ têm action, state,
pressão e depth; os frames RGB ficam como PNG em images/. Este script reconstrói
TUDO de forma limpa e consistente a partir dessas fontes:

  1. Lê todos os parquets de data, renumera episódios para contíguo (0..N-1)
  2. Recalcula index global e frame_index, consolida num único data/chunk-000/file-000
  3. Encoda o vídeo RGB completo (1 mp4): usa o backup mp4 para os episódios que ele
     já cobre + os PNGs para os demais
  4. Recalcula stats por episódio (só features numéricas — imagens não entram na meta)
  5. Reescreve meta/episodes (consolidada), stats.json e info.json

Uso:
    python tools/finalize_dataset.py --root datasets/G1_Dex3_depth_tactil_dataset/20260605_173343
    # opcional: --backup-mp4 <caminho> --fps 30 --video-key observation.images.head_camera
"""
import argparse
import glob
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.utils import DEFAULT_EPISODES_PATH, flatten_dict


def _ffprobe_nframes(path: Path) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return -1


def _encode_pngs(png_dir: Path, n_frames: int, out_mp4: Path, fps: int):
    """Encoda os primeiros n_frames PNGs (frame-000000.png ...) num mp4 h264."""
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps), "-start_number", "0",
        "-i", str(png_dir / "frame-%06d.png"),
        "-vframes", str(n_frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast",
        str(out_mp4),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg falhou em {png_dir}: {r.stderr[-300:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="raiz do dataset (a pasta com timestamp)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--video-key", default="observation.images.head_camera")
    ap.add_argument("--backup-mp4", default=None,
                    help="mp4 já encodado que cobre os primeiros episódios (default: file-000_backup.mp4)")
    args = ap.parse_args()

    DSET = Path(args.root).resolve()
    vkey = args.video_key
    print(f"[finalize] dataset: {DSET}")

    info = json.loads((DSET / "meta/info.json").read_text())
    features = info["features"]

    # ---- 1. carrega todos os parquets de data ----------------------------------
    data_files = sorted(glob.glob(str(DSET / "data/chunk-000/file-*.parquet")))
    if not data_files:
        raise SystemExit("nenhum parquet de data encontrado")
    df = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    old_eps = sorted(int(e) for e in df["episode_index"].unique())
    remap = {old: new for new, old in enumerate(old_eps)}
    print(f"[finalize] {len(old_eps)} episódios: {old_eps[0]}..{old_eps[-1]} "
          f"(renumerando p/ 0..{len(old_eps)-1})")
    gaps = [e for e in range(old_eps[0], old_eps[-1] + 1) if e not in old_eps]
    if gaps:
        print(f"[finalize]   buracos preenchidos (descartados): {gaps}")

    # ---- 2. renumera ep/index/frame e consolida --------------------------------
    parts = []
    ep_lengths = {}
    for old in old_eps:
        sub = df[df["episode_index"] == old].sort_values("frame_index").reset_index(drop=True)
        new = remap[old]
        sub["episode_index"] = np.int64(new)
        sub["frame_index"] = np.arange(len(sub), dtype=np.int64)
        ep_lengths[new] = len(sub)
        parts.append(sub)
    df2 = pd.concat(parts, ignore_index=True)
    df2["index"] = np.arange(len(df2), dtype=np.int64)
    total_frames = len(df2)
    print(f"[finalize] total de frames: {total_frames}")

    # ---- 3. encoda o vídeo RGB completo ----------------------------------------
    vid_dir = DSET / f"videos/{vkey}/chunk-000"
    vid_dir.mkdir(parents=True, exist_ok=True)
    backup = Path(args.backup_mp4) if args.backup_mp4 else vid_dir / "file-000_backup.mp4"

    tmp = Path(tempfile.mkdtemp(prefix="finalize_vid_"))
    seg_list = tmp / "concat.txt"
    segments = []

    # Quantos episódios (renumerados, a partir do 0) o backup cobre?
    backup_eps_covered = 0
    if backup.exists():
        bk_frames = _ffprobe_nframes(backup)
        bk_acc = 0
        for new in range(len(old_eps)):
            if bk_acc + ep_lengths[new] <= bk_frames:
                bk_acc += ep_lengths[new]
                backup_eps_covered = new + 1
            else:
                break
        if bk_acc == bk_frames:
            print(f"[finalize] backup mp4 cobre eps 0..{backup_eps_covered-1} ({bk_frames} frames)")
            segments.append(backup)
        else:
            print(f"[finalize] backup mp4 ({bk_frames}f) não bate com nenhum corte de episódio "
                  f"→ ignorando, encodando tudo dos PNGs")
            backup_eps_covered = 0

    # Episódios restantes: encoda dos PNGs (primeiros N frames)
    for new in range(backup_eps_covered, len(old_eps)):
        old = old_eps[new]
        png_dir = DSET / f"images/{vkey}/episode-{old:06d}"
        n = ep_lengths[new]
        if not png_dir.is_dir():
            raise SystemExit(f"ep{new} (antigo {old}): PNG dir não existe: {png_dir}")
        n_png = len(list(png_dir.glob("frame-*.png")))
        if n_png < n:
            raise SystemExit(f"ep{new} (antigo {old}): faltam PNGs ({n_png} < {n})")
        seg = tmp / f"ep{new:04d}.mp4"
        _encode_pngs(png_dir, n, seg, args.fps)
        got = _ffprobe_nframes(seg)
        if got != n:
            raise SystemExit(f"ep{new}: vídeo com {got} frames, esperado {n}")
        segments.append(seg)
        if new % 10 == 0 or new == len(old_eps) - 1:
            print(f"[finalize]   encodado ep{new}/{len(old_eps)-1}")

    # Concatena todos os segmentos (stream copy)
    seg_list.write_text("".join(f"file '{s}'\n" for s in segments))
    out_vid = vid_dir / "file-000.mp4"
    tmp_out = tmp / "full.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(seg_list), "-c", "copy", str(tmp_out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"concat falhou: {r.stderr[-400:]}")
    vfull = _ffprobe_nframes(tmp_out)
    if vfull != total_frames:
        raise SystemExit(f"vídeo final com {vfull} frames, esperado {total_frames}")
    shutil.move(str(tmp_out), str(out_vid))
    print(f"[finalize] vídeo final: {out_vid} ({vfull} frames, {out_vid.stat().st_size/1e6:.1f} MB)")

    # ---- 4. salva o parquet de data consolidado --------------------------------
    # limpa os files antigos e escreve um único file-000
    for f in data_files:
        Path(f).unlink()
    df2.to_parquet(DSET / "data/chunk-000/file-000.parquet", index=False)
    print(f"[finalize] data consolidado em data/chunk-000/file-000.parquet")

    # ---- 5. stats por episódio + meta/episodes ---------------------------------
    # só as features numéricas entram na meta (imagens/vídeo não têm stats aqui)
    num_keys = [k for k, v in features.items() if v["dtype"] not in ("image", "video", "string")]
    ep_rows = []
    all_stats = []
    frames_acc = 0  # frames acumulados: serve p/ dataset_from/to_index E from/to_timestamp do vídeo
    for new in range(len(old_eps)):
        sub = df2[df2["episode_index"] == new]
        n = ep_lengths[new]
        ep_data = {}
        for k in num_keys:
            col = sub[k].to_numpy()
            if isinstance(col[0], np.ndarray):
                ep_data[k] = np.stack(col)            # (N, D)
            else:
                ep_data[k] = col.astype(np.float64)   # (N,) escalar
        stats = compute_episode_stats(ep_data, {k: features[k] for k in num_keys})
        all_stats.append(stats)

        row = {
            "episode_index": new,
            "tasks": [info.get("_task_str", "Pick up the white cup")],
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            f"videos/{vkey}/chunk_index": 0,
            f"videos/{vkey}/file_index": 0,
            # episódios ficam sequenciais num único mp4 → cada um ocupa um range de tempo
            f"videos/{vkey}/from_timestamp": frames_acc / args.fps,
            f"videos/{vkey}/to_timestamp": (frames_acc + n) / args.fps,
            "dataset_from_index": frames_acc,
            "dataset_to_index": frames_acc + n,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        row.update(flatten_dict({"stats": stats}))
        ep_rows.append(row)
        frames_acc += n

    # task real (da tasks.parquet, se houver)
    tasks_path = DSET / "meta/tasks.parquet"
    task_str = "Pick up the white cup"
    if tasks_path.exists():
        tdf = pd.read_parquet(tasks_path)
        if len(tdf.index) > 0:
            task_str = str(tdf.index[0])
    for row in ep_rows:
        row["tasks"] = [task_str]

    ep_df = pd.DataFrame(ep_rows)
    # limpa metas antigas e escreve consolidada
    old_meta = glob.glob(str(DSET / "meta/episodes/chunk-000/file-*.parquet"))
    for f in old_meta:
        Path(f).unlink()
    meta_out = DSET / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    ep_df.to_parquet(meta_out, index=False)
    print(f"[finalize] meta/episodes reconstruída ({len(ep_df)} episódios)")

    # ---- 6. stats.json agregado ------------------------------------------------
    global_stats = aggregate_stats(all_stats)
    stats_json = {k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()}
                  for k, v in global_stats.items()}
    (DSET / "meta/stats.json").write_text(json.dumps(stats_json, indent=4))
    print(f"[finalize] stats.json agregado ({len(stats_json)} features)")

    # ---- 7. info.json ----------------------------------------------------------
    info["total_episodes"] = len(old_eps)
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{len(old_eps)}"}
    (DSET / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(f"[finalize] info.json: total_episodes={len(old_eps)} total_frames={total_frames}")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[finalize] ✅ pronto — dataset consistente com {len(old_eps)} episódios.")


if __name__ == "__main__":
    main()
