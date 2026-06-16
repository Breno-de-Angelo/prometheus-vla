#!/usr/bin/env python3
"""
Grava um clipe RGB+depth do robô (ZMQ :5555) + todos os dados pro alinhamento
sim<->real via ArUco (mesa ids 0-3 / copo ids 4-7) e profundidade.

Salva em <out>/:
  - annotated.mp4    : RGB com ArUcos, eixos de pose e distâncias desenhados
  - rgb/000123.png   : frames RGB crus (BGR)
  - depth/000123.png : profundidade em mm (uint16, reconstruída do stream)
  - manifest.json    : intrínsecos + por-frame {marker: rvec,tvec, z_pnp, z_depth}
                       + agregados (pose mediana de cada marcador, dist câmera->mesa/copo)

LIMITAÇÃO do depth: o full_realsenser_server.py manda depth clipado em 2.0 m,
quantizado em 8 bits (~7.8 mm/passo) e via JPEG. dist_mm = pixel * (2000/255).
Pra precisão fina use a pose do ArUco (solvePnP); o depth é cross-check/escala.

Uso:
    conda activate g1
    python record_sync_data.py --host 192.168.68.71 --secs 6
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "unitree-g1-mujoco"))
from sim.sensor_utils import SensorClient, ImageUtils  # noqa: E402
import zmq  # noqa: E402

INTR_CACHE = HERE / "assets" / "realsense_color_intrinsics.json"
TABLE_IDS = {0, 1, 2, 3}
CUP_IDS = {4, 5, 6, 7}
KNOWN_IDS = TABLE_IDS | CUP_IDS
SIZE_M = {**{i: 0.060 for i in TABLE_IDS}, **{i: 0.025 for i in CUP_IDS}}
DEPTH_CLIP_MM = 2000.0  # full_realsenser_server.py: clip(0, 2000)


def load_K():
    c = json.loads(INTR_CACHE.read_text())
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], np.float64)
    dist = np.array(c.get("coeffs", [0, 0, 0, 0, 0]), np.float64)
    return K, dist, c


def decode_depth_mm(depth_any):
    """Stream manda depth 8-bit 3c (clip 2m/255). Devolve mm float + máscara válida."""
    d = depth_any
    if d.ndim == 3:
        d = d.mean(axis=2)
    d8 = d.astype(np.float32)
    depth_mm = d8 * (DEPTH_CLIP_MM / 255.0)
    valid = d8 > 0
    return depth_mm, valid


def depth_at(depth_mm, valid, uv, win=3):
    """Mediana do depth (mm) numa janela em torno do pixel (u,v). NaN se sem dado."""
    u, v = int(round(uv[0])), int(round(uv[1]))
    h, w = depth_mm.shape
    u0, u1 = max(0, u - win), min(w, u + win + 1)
    v0, v1 = max(0, v - win), min(h, v + win + 1)
    patch = depth_mm[v0:v1, u0:u1]
    m = valid[v0:v1, u0:u1]
    if m.sum() == 0:
        return float("nan")
    return float(np.median(patch[m]))


def detect(gray):
    D = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 35
    p.adaptiveThreshWinSizeStep = 4
    det = cv2.aruco.ArucoDetector(D, p)
    corners, ids, _ = det.detectMarkers(gray)
    out = []
    if ids is not None:
        for c, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            if mid in KNOWN_IDS:
                out.append((mid, c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.68.71")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--secs", type=float, default=6.0)
    ap.add_argument("--out", default=None, help="pasta de saída (default runs/sync_<ts>)")
    ap.add_argument("--save-frames", action="store_true", default=True,
                    help="salvar PNGs crus de rgb/ e depth/ (default on)")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else HERE / "runs" / f"sync_{ts}"
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    K, dist, intr = load_K()
    print(f"[intr] fx={intr['fx']:.1f} fy={intr['fy']:.1f} cx={intr['cx']:.1f} cy={intr['cy']:.1f}")

    client = SensorClient()
    client.start_client(server_ip=args.host, port=args.port)
    client.socket.setsockopt(zmq.RCVTIMEO, 4000)
    print(f"[rec] gravando ~{args.secs:.0f}s de tcp://{args.host}:{args.port} ...")

    writer = None
    frames = []           # registros por-frame pro manifest
    t0 = time.time()
    n = 0
    while time.time() - t0 < args.secs:
        try:
            data = client.receive_message()
        except Exception:
            continue
        imgs = data.get("images", data)
        rgb_s, dep_s = imgs.get("head_camera"), imgs.get("head_camera_depth")
        if rgb_s is None:
            continue
        bgr = ImageUtils.decode_image(rgb_s) if isinstance(rgb_s, str) else rgb_s
        depth_mm = valid = None
        if dep_s is not None:
            dep_raw = ImageUtils.decode_image(dep_s) if isinstance(dep_s, str) else dep_s
            depth_mm, valid = decode_depth_mm(dep_raw)
        h, w = bgr.shape[:2]

        if writer is None:
            writer = cv2.VideoWriter(str(out / "annotated.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))

        vis = bgr.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rec = {"t": data.get("timestamps", {}).get("head_camera"), "markers": {}}
        for mid, c in detect(gray):
            half = SIZE_M[mid] / 2.0
            objp = np.array([[-half, half, 0], [half, half, 0],
                             [half, -half, 0], [-half, -half, 0]], np.float32)
            ok, rvec, tvec = cv2.solvePnP(objp, c[0], K, dist,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            ctr = c[0].mean(axis=0)
            z_pnp = float(tvec[2][0]) if ok else None
            z_dep = depth_at(depth_mm, valid, ctr) / 1000.0 if depth_mm is not None else None
            grp = "table" if mid in TABLE_IDS else "cup"
            col = (0, 255, 0) if mid in TABLE_IDS else (255, 255, 0)
            cv2.aruco.drawDetectedMarkers(vis, [c], np.array([[mid]]), col)
            if ok:
                cv2.drawFrameAxes(vis, K, dist, rvec, tvec, SIZE_M[mid] * 0.7, 2)
                txt = f"id{mid} pnp={z_pnp:.2f}m"
                if z_dep == z_dep and z_dep:  # não-NaN
                    txt += f" dep={z_dep:.2f}m"
                cv2.putText(vis, txt, (int(ctr[0]) - 30, int(ctr[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            rec["markers"][mid] = {
                "group": grp,
                "corners": c[0].tolist(),
                "rvec": rvec.flatten().tolist() if ok else None,
                "tvec": tvec.flatten().tolist() if ok else None,
                "z_pnp_m": z_pnp,
                "z_depth_m": (None if z_dep is None or z_dep != z_dep else z_dep),
            }
        frames.append(rec)
        writer.write(vis)
        if args.save_frames:
            cv2.imwrite(str(out / "rgb" / f"{n:06d}.png"), bgr)
            if depth_mm is not None:
                cv2.imwrite(str(out / "depth" / f"{n:06d}.png"),
                            np.clip(depth_mm, 0, 65535).astype(np.uint16))
        n += 1
    client.stop_client()
    if writer:
        writer.release()

    # ---- agregados: pose mediana de cada marcador + distâncias ----
    agg = {}
    seen = {}
    for f in frames:
        for mid, m in f["markers"].items():
            seen.setdefault(int(mid), []).append(m)
    for mid, lst in sorted(seen.items()):
        zs_pnp = [m["z_pnp_m"] for m in lst if m["z_pnp_m"]]
        zs_dep = [m["z_depth_m"] for m in lst if m["z_depth_m"]]
        tvecs = np.array([m["tvec"] for m in lst if m["tvec"]])
        agg[mid] = {
            "group": lst[0]["group"],
            "n_obs": len(lst),
            "tvec_median": np.median(tvecs, axis=0).tolist() if len(tvecs) else None,
            "z_pnp_median_m": float(np.median(zs_pnp)) if zs_pnp else None,
            "z_depth_median_m": float(np.median(zs_dep)) if zs_dep else None,
        }

    manifest = {
        "created": ts,
        "host": args.host,
        "n_frames": n,
        "resolution": [848, 480],
        "intrinsics": intr,
        "depth_note": "8-bit, clip 2m, JPEG; dist_mm = pixel*2000/255; coarse",
        "markers_seen": sorted(seen.keys()),
        "aggregate": agg,
        "frames": frames,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n[ok] {n} frames -> {out}")
    print(f"     vídeo: {out/'annotated.mp4'}")
    print(f"     marcadores vistos: {sorted(seen.keys()) or 'NENHUM'}")
    for mid, a in agg.items():
        zp = a["z_pnp_median_m"]; zd = a["z_depth_median_m"]
        print(f"     id{mid} ({a['group']}): obs={a['n_obs']}  "
              f"z_pnp={zp:.3f}m  z_depth={zd if zd is None else round(zd,3)}m")


if __name__ == "__main__":
    main()
