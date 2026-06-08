#!/usr/bin/env python3
"""Converte dataset LeRobot para arquivos epN_gzb64.js do OmniView.

Uso:
    python lerobot_to_omniview.py <dataset_dir> [--out <out_dir>]

Se --out não for passado, cria <dataset_dir>/data/real/ e coloca os arquivos lá.
Depois rode:
    omniview --data <dataset_dir>
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


# ---- normalização de pressão 108→33 -----------------------------------------
def _pressure_to_taxels(pressure_arr: np.ndarray, global_max: float) -> list:
    """
    pressure_arr: (N, 108) float
    Retorna lista de N listas de 33 floats [0..1] para o OmniView.
    Downsampling: amostra índices linearmente espaçados de 0 a 107.
    """
    N = pressure_arr.shape[0]
    if global_max < 1e-6:
        return [[0.0] * 33 for _ in range(N)]
    indices = np.round(np.linspace(0, 107, 33)).astype(int)
    p33 = pressure_arr[:, indices].astype(np.float32) / global_max
    p33 = np.clip(p33, 0.0, 1.0)
    return p33.tolist()


# ---- payload principal -------------------------------------------------------
def episode_to_gzb64(df_ep: pd.DataFrame, pressure_global_max: float = 0.0) -> str:
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

    # Pressão tátil real (quando disponível)
    gmax = pressure_global_max
    for col_key, out_key in [
        ("observation.left_hand_pressure",  "pressure_left"),
        ("observation.right_hand_pressure", "pressure_right"),
    ]:
        if col_key in df_ep.columns:
            raw = np.stack(df_ep[col_key].tolist()).astype(np.float32)
            if gmax < 1e-6:
                gmax = float(raw.max()) if raw.max() > 0 else 1.0
            payload[out_key] = _pressure_to_taxels(raw, gmax)

    compressed = gzip.compress(json.dumps(payload).encode())
    return base64.b64encode(compressed).decode()


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

    # Calcula max global de pressão (para normalização consistente entre episódios)
    pressure_global_max = 0.0
    for col in ["observation.left_hand_pressure", "observation.right_hand_pressure"]:
        if col in df.columns:
            arr = np.stack(df[col].tolist()).astype(np.float32)
            pressure_global_max = max(pressure_global_max, float(arr.max()))
    if pressure_global_max > 0:
        print(f"  pressão: max global = {pressure_global_max:.0f}")

    for ep_idx in episodes:
        df_ep = df[df["episode_index"] == ep_idx].sort_values("index")

        # Depth
        has_depth = extract_depth_images(df_ep, out_dir, ep_idx)

        # Payload gzb64
        b64 = episode_to_gzb64(df_ep, pressure_global_max)
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
        b64 = episode_to_gzb64(df_ep, pressure_global_max)
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
