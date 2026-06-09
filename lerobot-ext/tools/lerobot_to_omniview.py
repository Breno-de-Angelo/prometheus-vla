#!/usr/bin/env python3
"""Converte dataset LeRobot para o formato de arquivos do OmniView dataset viewer.

CONTEXTO
--------
O OmniView é um visualizador web estático de episódios do G1 (RGB, depth, trajetória
de juntas, pressão tátil). Ele espera, em <dataset_dir>/data/real/, um conjunto de
arquivos gerados por este script — não lê parquet diretamente.

O QUE O SCRIPT FAZ
------------------
Para cada episódio N do dataset, gera os seguintes arquivos em <out_dir>:

  epN_gzb64.js         Payload gzip+base64 com action, state, timestamps e pressão
                       tátil (left/right_hand_pressure) normalizados. Carregado pelo
                       OmniView como window.__EPN_GZB64.

  epN_depth_turbo.mp4  Vídeo colorido (colormap TURBO) dos frames de depth do parquet.
                       Sincronizado quadro-a-quadro com o vídeo RGB.

  epN_depth_raw.png    Frame representativo de depth em escala raw uint8
                       (mm = pixel × 4, range 0–1020 mm). Usado pelo OmniView para
                       amostragem de distância ao clicar na imagem.

  epN_video_rgb.mp4    Segmento RGB do episódio, fatiado do file-000.mp4 (ou chunk
                       equivalente) usando ffmpeg. O OmniView usa este vídeo como
                       relógio mestre — depth e trajetória são sincronizados a ele.

  all_episodes.js      Catálogo completo de todos os episódios (nframes, durSec,
                       payload b64). Carregado como window.__OV_ALL_EPISODES.

DEPENDÊNCIAS
------------
  - ffmpeg no PATH (para depth_turbo.mp4 e epN_video_rgb.mp4)
  - conda env g1 (pandas, numpy, pillow, opencv-python)

USO
---
    python lerobot_to_omniview.py <dataset_dir> [--out <out_dir>]

    Se --out não for passado, grava em <dataset_dir>/data/real/.

    Depois rode o viewer:
        omniview --data <dataset_dir>
        # abre http://127.0.0.1:8000/omniview.html no browser

NOTAS
-----
  - O vídeo RGB de origem (videos/observation.images.head_camera/chunk-000/file-000.mp4)
    contém TODOS os episódios concatenados. O script fatia via ffmpeg usando o índice
    global de frames do parquet. Múltiplos chunks ainda não são suportados.
  - Frames de depth corrompidos são substituídos pelo último frame bom (mantém contagem).
  - Pressão tátil é normalizada pelo max global entre todos os episódios (consistência
    visual ao navegar entre episódios).
"""
import argparse
import base64
import gzip
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ---- colormap depth: replica exatamente o live_omniview._colorize_depth ------
# Range fixo 0.2–1.5 m, INVERTIDO (perto=quente, longe=frio), cv2.COLORMAP_TURBO
_VIS_MIN_M = float(os.environ.get("DEPTH_VIS_MIN_M", "0.2"))
_VIS_MAX_M = float(os.environ.get("DEPTH_VIS_MAX_M", "1.5"))


def _colorize_depth(depth_u16: np.ndarray) -> np.ndarray:
    """uint16 (mm para robô real) → RGB uint8, idêntico ao live viewer."""
    import cv2
    depth = depth_u16.astype(np.float32)
    valid = depth > 0
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    # robô real: mm; sim: metros*38 (valores < 150)
    scale = 38.0 if float(depth[valid].max()) < 150.0 else 1000.0
    depth_m = depth / scale
    norm = np.clip((depth_m - _VIS_MIN_M) / (_VIS_MAX_M - _VIS_MIN_M), 0.0, 1.0)
    # invertido: perto → 255 (vermelho/quente), longe → 0 (azul/frio)
    d_vis = ((1.0 - norm) * 255).astype(np.uint8)
    d_bgr = cv2.applyColorMap(d_vis, cv2.COLORMAP_TURBO)
    d_bgr[~valid] = 0
    return cv2.cvtColor(d_bgr, cv2.COLOR_BGR2RGB)  # retorna RGB


def _depth_raw_uint8(depth_u16: np.ndarray) -> Image.Image:
    """Encode depth como uint8 para amostragem JS: mm = pixel * 4 (0-1020 mm)."""
    raw = np.clip(depth_u16 // 4, 0, 255).astype(np.uint8)
    raw[depth_u16 == 0] = 0
    return Image.fromarray(raw, "L")


def _decode_depth(v):
    """Decodifica 1 frame de depth do parquet -> ndarray, ou None se corrompido."""
    try:
        b = v["bytes"] if isinstance(v, dict) else bytes(v)
        return np.array(Image.open(io.BytesIO(b)))
    except Exception:
        return None


def extract_depth_images(df_ep: pd.DataFrame, out_dir: Path, ep_idx: int) -> bool:
    """Extrai frame representativo (raw para mm) e vídeo turbo de depth."""
    col = "observation.images.head_camera_depth"
    if col not in df_ep.columns:
        return False
    try:
        # frame raw representativo: pega o do meio; se corrompido, procura um bom
        mid = len(df_ep) // 2
        arr_mid = _decode_depth(df_ep[col].iloc[mid])
        if arr_mid is None:
            for v in df_ep[col]:
                arr_mid = _decode_depth(v)
                if arr_mid is not None:
                    break
        if arr_mid is not None:
            _depth_raw_uint8(arr_mid).save(out_dir / f"ep{ep_idx}_depth_raw.png")
        _encode_depth_video(df_ep, col, out_dir, ep_idx)
        return True
    except Exception as e:
        print(f"  [AVISO] depth ep{ep_idx}: {e}", file=sys.stderr)
        return False


def _encode_depth_video(df_ep: pd.DataFrame, col: str, out_dir: Path, ep_idx: int):
    """Codifica vídeo depth via ffmpeg pipe, idêntico ao live viewer."""
    import subprocess, shutil
    if not shutil.which("ffmpeg"):
        print("  [AVISO] ffmpeg não encontrado, pulando vídeo depth", file=sys.stderr)
        return

    fps = 30
    out_path = out_dir / f"ep{ep_idx}_depth_turbo.mp4"

    # acha o 1o frame bom p/ fixar H,W (frames corrompidos não abortam mais)
    arr0 = None
    for v in df_ep[col]:
        arr0 = _decode_depth(v)
        if arr0 is not None:
            break
    if arr0 is None:
        print(f"  [AVISO] depth ep{ep_idx}: nenhum frame decodificável", file=sys.stderr)
        return
    H, W = arr0.shape

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-preset", "fast",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # robusto: frame corrompido -> repete o último RGB bom (mantém contagem/alinhamento)
    last_rgb = _colorize_depth(arr0)
    n_bad = 0
    for row in df_ep[col]:
        arr = _decode_depth(row)
        if arr is None or arr.shape != (H, W):
            n_bad += 1
            rgb = last_rgb
        else:
            rgb = _colorize_depth(arr)
            last_rgb = rgb
        proc.stdin.write(rgb.tobytes())

    proc.stdin.close()
    proc.wait()
    if n_bad:
        print(f"  [info] ep{ep_idx}: {n_bad} frame(s) de depth corrompido(s) substituído(s) pelo anterior", file=sys.stderr)


# Índices amostrados dos 108 sensores Dex3 → 33 canais para o OmniView
_PRESSURE_SAMPLE_IDX = np.round(np.linspace(0, 107, 33)).astype(int)


# ---- normalização de pressão 108→33 -----------------------------------------
def _pressure_to_taxels(pressure_arr: np.ndarray, baseline: np.ndarray, delta_max: float) -> list:
    """Converte (N, 108) float → lista de N listas de 33 floats [0..1] para o OmniView.

    Aplica subtração de baseline por sensor (remove offset de repouso do Dex3) e
    normaliza pelo delta_max global (calculado em main() sobre os mesmos 33 índices,
    garantindo consistência visual entre episódios).

    Args:
        pressure_arr: (N, 108) raw do parquet.
        baseline:     (33,) mínimo por sensor calculado sobre todos os frames do dataset.
        delta_max:    escalar — max(p33 - baseline) global.
    """
    if delta_max < 1e-6:
        return [[0.0] * 33 for _ in range(len(pressure_arr))]
    p33 = pressure_arr[:, _PRESSURE_SAMPLE_IDX].astype(np.float32)
    p33 = p33 - baseline          # remove offset de repouso por sensor
    p33 = np.clip(p33 / delta_max, 0.0, 1.0)
    return p33.tolist()


# ---- payload principal -------------------------------------------------------
def episode_to_gzb64(
    df_ep: pd.DataFrame,
    pressure_baselines: dict,   # col_key → np.ndarray (33,)
    pressure_delta_max: float,
) -> str:
    n = len(df_ep)
    fps = 30

    action = df_ep["action"].tolist()
    state = df_ep["observation.state"].tolist()
    ts_raw = df_ep["timestamp"].tolist()

    t0 = ts_raw[0]
    ts = [int(round((t - t0) * 1000)) for t in ts_raw]

    action = [a.tolist() if hasattr(a, "tolist") else list(a) for a in action]
    state  = [s.tolist() if hasattr(s, "tolist") else list(s) for s in state]

    arr = np.array(action)
    ranges = [[float(arr[:, i].min()), float(arr[:, i].max())] for i in range(arr.shape[1])]

    payload = {
        "nframes": n,
        "fps": fps,
        "ts": ts,
        "action": action,
        "state": state,
        "ranges": ranges,
    }

    for col_key, out_key in [
        ("observation.left_hand_pressure",  "pressure_left"),
        ("observation.right_hand_pressure", "pressure_right"),
    ]:
        if col_key in df_ep.columns:
            raw = np.stack(df_ep[col_key].tolist()).astype(np.float32)
            baseline = pressure_baselines.get(col_key, np.zeros(33, dtype=np.float32))
            payload[out_key] = _pressure_to_taxels(raw, baseline, pressure_delta_max)

    compressed = gzip.compress(json.dumps(payload).encode())
    return base64.b64encode(compressed).decode()


def extract_rgb_videos(
    df: pd.DataFrame, episodes: list, dataset_dir: Path, out_dir: Path, fps: int = 30
) -> None:
    """Fatia o vídeo RGB do LeRobot (file-000.mp4) em um mp4 por episódio."""
    import subprocess, shutil
    if not shutil.which("ffmpeg"):
        print("  [AVISO] ffmpeg não encontrado, pulando vídeo RGB", file=sys.stderr)
        return

    video_glob = sorted(dataset_dir.glob("videos/observation.images.head_camera/**/*.mp4"))
    if not video_glob:
        print("  [AVISO] Vídeo RGB não encontrado em videos/observation.images.head_camera/", file=sys.stderr)
        return

    # Concatena todos os chunks numa lista ordenada
    all_chunks = video_glob  # já deve ser chunk-000/file-000.mp4, chunk-001/…

    # Verifica se há apenas um chunk (caso comum)
    if len(all_chunks) == 1:
        source = all_chunks[0]
        procs = []
        for ep_idx in episodes:
            df_ep = df[df["episode_index"] == ep_idx].sort_values("index")
            n = len(df_ep)
            start_frame = int(df_ep["index"].min())
            ss = start_frame / fps
            dur = n / fps
            out_path = out_dir / f"ep{ep_idx}_video_rgb.mp4"
            if out_path.exists():
                continue
            cmd = [
                "ffmpeg", "-y", "-ss", f"{ss:.6f}", "-i", str(source),
                "-t", f"{dur:.6f}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast",
                str(out_path), "-loglevel", "error",
            ]
            procs.append((ep_idx, subprocess.Popen(cmd)))
        for ep_idx, p in procs:
            p.wait()
            print(f"  ep{ep_idx}: rgb✓", end="  ")
        if procs:
            print()
    else:
        print(f"  [AVISO] Múltiplos chunks RGB não suportados ainda ({len(all_chunks)} chunks)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="Pasta raiz do dataset LeRobot")
    parser.add_argument("--out", help="Pasta de saída (default: <dataset_dir>/data/real)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else dataset_dir / "data" / "real"
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not parquet_files:
        print(f"Nenhum parquet encontrado em {dataset_dir}/data/", file=sys.stderr)
        sys.exit(1)

    dfs = []
    for pf in parquet_files:
        try:
            dfs.append(pd.read_parquet(pf))
        except Exception as e:
            print(f"[AVISO] Pulando {pf.name}: {e}", file=sys.stderr)

    if not dfs:
        print("Nenhum parquet válido.", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    episodes = sorted(df["episode_index"].unique())
    print(f"Encontrados {len(episodes)} episódios: {list(episodes)}")

    # Normalização de pressão: baseline por sensor (remove offset de repouso) +
    # cap no percentil 99 dos deltas (evita que um único pico outlier comprima
    # todos os outros episódios; valores acima do cap ficam clipados em 1.0).
    pressure_baselines: dict = {}
    pressure_delta_max = 0.0
    for col in ["observation.left_hand_pressure", "observation.right_hand_pressure"]:
        if col in df.columns:
            arr = np.stack(df[col].tolist()).astype(np.float32)
            p33 = arr[:, _PRESSURE_SAMPLE_IDX]          # (N_total, 33)
            baseline = p33.min(axis=0)                  # (33,) offset de repouso por sensor
            pressure_baselines[col] = baseline
            deltas = (p33 - baseline).ravel()
            cap = float(np.percentile(deltas[deltas > 0], 99)) if (deltas > 0).any() else 1.0
            pressure_delta_max = max(pressure_delta_max, cap)
    if pressure_delta_max > 0:
        print(f"  pressão: delta p99 = {pressure_delta_max:.1f}")

    # Vídeo RGB: fatia file-000.mp4 em ep{N}_video_rgb.mp4
    extract_rgb_videos(df, episodes, dataset_dir, out_dir)

    for ep_idx in episodes:
        df_ep = df[df["episode_index"] == ep_idx].sort_values("index")

        # Depth
        has_depth = extract_depth_images(df_ep, out_dir, ep_idx)

        # Payload gzb64
        b64 = episode_to_gzb64(df_ep, pressure_baselines, pressure_delta_max)
        out_file = out_dir / f"ep{ep_idx}_gzb64.js"
        header = (
            f"// ep{ep_idx} — {len(df_ep)} frames, gzip+base64\n"
            f"window.__EP{ep_idx}_GZB64 = \"{b64}\";\n"
        )
        out_file.write_text(header)
        depth_note = " depth✓" if has_depth else ""
        print(f"  ep{ep_idx}: {len(df_ep)} frames -> {out_file.name}{depth_note}")

    # all_episodes.js
    catalog = []
    for ep_idx in episodes:
        df_ep = df[df["episode_index"] == ep_idx].sort_values("index")
        b64 = episode_to_gzb64(df_ep, pressure_baselines, pressure_delta_max)
        catalog.append({
            "idx": int(ep_idx),
            "nframes": len(df_ep),
            "durSec": round(len(df_ep) / 30, 2),
            "fps": 30,
            "b64": b64,
        })
    all_ep_file = out_dir / "all_episodes.js"
    all_ep_file.write_text(
        "// All real episodes — generated by lerobot_to_omniview.py\n"
        f"window.__OV_ALL_EPISODES = {json.dumps(catalog)};\n"
    )
    print(f"  all_episodes.js: {len(catalog)} episódios")

    print(f"\nPronto! Rode:")
    print(f"  omniview --data {dataset_dir}")


if __name__ == "__main__":
    main()
