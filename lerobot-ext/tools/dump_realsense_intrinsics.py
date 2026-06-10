#!/usr/bin/env python3
"""Lê os intrínsecos REAIS do stream de depth da RealSense e salva em JSON.

Rodar NO ROBÔ (onde a D435 está conectada), na resolução usada na gravação:

    python lerobot-ext/tools/dump_realsense_intrinsics.py --width 848 --height 480 \
        [--serial 327122071538] [--out depth_intrinsics.json]

Saída: JSON {fx, fy, cx, cy, width, height, model, coeffs, serial} + bloco YAML
pronto pra colar na config de treino (`depth_intrinsics:`).

Snippet equivalente para embutir no realsense_server_depth16.py (no robô), logo
após o pipeline.start(config) — loga e salva os intrínsecos junto da gravação:

    profile = pipeline.start(config)
    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    info = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
            "width": intr.width, "height": intr.height}
    print(f"[depth-intrinsics] {info}")
    with open("depth_intrinsics.json", "w") as f:
        json.dump(info, f, indent=2)
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--serial", default=None, help="serial da câmera (opcional)")
    ap.add_argument("--out", default="depth_intrinsics.json")
    args = ap.parse_args()

    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    try:
        stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = stream.get_intrinsics()
        dev = profile.get_device()
        info = {
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.ppx,
            "cy": intr.ppy,
            "width": intr.width,
            "height": intr.height,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
            "serial": dev.get_info(rs.camera_info.serial_number),
        }
    finally:
        pipeline.stop()

    with open(args.out, "w") as f:
        json.dump(info, f, indent=2)

    print(json.dumps(info, indent=2))
    print(f"\n[ok] salvo em {args.out}")
    print("\nBloco pronto pra config de treino:")
    print("depth_intrinsics:")
    for k in ("fx", "fy", "cx", "cy"):
        print(f"  {k}: {info[k]:.4f}")


if __name__ == "__main__":
    main()
