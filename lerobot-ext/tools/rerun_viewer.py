#!/usr/bin/env python3
"""
Viewer Rerun para sessões de dataset — base genérica de visualização.

Não depende do formato LeRobot: varre a sessão atrás de vídeos (.mp4) e de
frames soltos (.png) e loga tudo numa timeline do Rerun. A ideia é começar
"só vendo os vídeos" e ir plugando estado/ação/tátil/3D por cima depois.

Uso:
    python lerobot-ext/tools/rerun_viewer.py <pasta_sessao>
    python lerobot-ext/tools/rerun_viewer.py <pasta_sessao> --save out.rrd   # salva em vez de abrir a GUI

Exemplos:
    # sessão completa (RGB em mp4)
    python lerobot-ext/tools/rerun_viewer.py datasets/G1_Dex3_depth_tactil_dataset/20260603_125956
    # sessão abortada (só PNGs de depth)
    python lerobot-ext/tools/rerun_viewer.py datasets/G1_Dex3_depth_tactil_dataset/20260604_204606
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import rerun as rr


def short_cam_name(path_part: str) -> str:
    """observation.images.head_camera_depth -> head_camera_depth"""
    return path_part.replace("observation.images.", "").replace("observation.", "")


def log_video(entity: str, mp4_path: Path) -> int:
    """Loga um .mp4 como AssetVideo (o viewer decodifica sob demanda — leve)."""
    video = rr.AssetVideo(path=str(mp4_path))
    rr.log(entity, video, static=True)
    ts_ns = video.read_frame_timestamps_nanos()
    rr.send_columns(
        entity,
        indexes=[rr.TimeColumn("video_time", duration=1e-9 * ts_ns)],
        columns=rr.VideoFrameReference.columns_nanos(ts_ns),
    )
    return len(ts_ns)


def log_png_sequence(entity: str, frames: list[Path], fps: float = 30.0) -> int:
    """Loga uma sequência de PNGs soltos. Single-channel uint16 -> depth."""
    n = 0
    for i, f in enumerate(frames):
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        rr.set_time("frame", sequence=i)
        rr.set_time("video_time", duration=i / fps)
        if img.ndim == 2:
            if img.dtype == np.uint16:
                rr.log(entity, rr.DepthImage(img, meter=1000.0))  # mm -> m
            else:
                rr.log(entity, rr.DepthImage(img))
        else:
            rr.log(entity, rr.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        n += 1
    return n


def frame_sort_key(p: Path):
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser(description="Viewer Rerun para sessões de dataset")
    ap.add_argument("session", type=Path, help="pasta da sessão")
    ap.add_argument("--save", type=Path, default=None, help="salva .rrd em vez de abrir a GUI")
    args = ap.parse_args()

    session = args.session
    if not session.is_dir():
        raise SystemExit(f"pasta não encontrada: {session}")

    rr.init(f"dataset:{session.name}", spawn=args.save is None)
    if args.save is not None:
        rr.save(str(args.save))

    found = False

    # 1) vídeos .mp4  (videos/<cam>/.../*.mp4)
    for mp4 in sorted((session / "videos").rglob("*.mp4")):
        cam = short_cam_name(mp4.parts[mp4.parts.index("videos") + 1])
        n = log_video(cam, mp4)
        print(f"[video] {cam}: {n} frames  ({mp4.name})")
        found = True

    # 2) frames soltos .png  (images/<cam>/episode-XXXX/*.png)
    img_root = session / "images"
    if img_root.is_dir():
        for cam_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
            cam = short_cam_name(cam_dir.name)
            for ep_dir in sorted(p for p in cam_dir.iterdir() if p.is_dir()):
                frames = sorted(ep_dir.glob("*.png"), key=frame_sort_key)
                if not frames:
                    continue
                ep = ep_dir.name
                n = log_png_sequence(f"{cam}/{ep}", frames)
                print(f"[png]   {cam}/{ep}: {n} frames")
                found = True

    if not found:
        print("nenhum vídeo (.mp4) nem frame (.png) encontrado nesta sessão.")
    elif args.save is not None:
        print(f"\nsalvo em {args.save} — abra com:  rerun {args.save}")


if __name__ == "__main__":
    main()
