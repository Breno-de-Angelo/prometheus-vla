import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lerobot-ext" / "Scripts_Prometheus_int"))
from sim.sensor_utils import SensorServer, ImageUtils


def start_realsense_zmq(port: int, serial: str | None, fps: int, enable_depth: bool) -> None:
    pipeline = rs.pipeline()
    config = rs.config()

    if serial:
        config.enable_device(serial)

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
    if enable_depth:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)

    try:
        profile = pipeline.start(config)
        print(f"[RealSense D435i] Câmera iniciada em {fps} FPS.")
        if enable_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            print(f"[RealSense D435i] Depth ativo, depth_scale={depth_scale}")
        else:
            depth_scale = None
            print("[RealSense D435i] Depth desativado; apenas RGB será publicado.")
    except Exception as e:
        print(f"[Erro] {e}")
        return

    server = SensorServer()
    server.start_server(port=port)
    mode = "RGB + depth" if enable_depth else "RGB-only"
    print(f"[ZMQ] Servidor ativo na porta {port} ({mode})")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            rgb = np.asanyarray(color_frame.get_data())
            encoded_rgb = ImageUtils.encode_image(rgb)
            t = time.time()

            if enable_depth:
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue

                depth_raw = np.asanyarray(depth_frame.get_data())
                depth_m = depth_raw.astype(np.float32) * depth_scale
                depth_norm = np.clip(depth_m / 2.0, 0.0, 1.0)
                depth_u8 = (depth_norm * 255.0).astype(np.uint8)
                depth_3ch = cv2.merge([depth_u8, depth_u8, depth_u8])
                encoded_depth = ImageUtils.encode_image(depth_3ch)

                message = {
                    "images": {
                        "head_camera": encoded_rgb,
                        "head_camera_depth": encoded_depth,
                    },
                    "timestamps": {
                        "head_camera": t,
                        "head_camera_depth": t,
                    },
                }
            else:
                message = {
                    "images": {"head_camera": encoded_rgb},
                    "timestamps": {"head_camera": t},
                }

            server.send_message(message)

    except KeyboardInterrupt:
        print("Encerrando...")
    finally:
        pipeline.stop()
        server.stop_server()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start RealSense ZMQ server for the Unitree G1 Prometheus camera."
    )
    parser.add_argument("--port", type=int, default=5555, help="ZMQ server port")
    parser.add_argument("--serial", default=None, help="Realsense serial number if multiple devices are connected")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS")
    parser.add_argument("--enable-depth", action="store_true", help="Also publish depth frames under head_camera_depth")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth and publish only head_camera")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enable_depth = args.enable_depth and not args.no_depth
    start_realsense_zmq(port=args.port, serial=args.serial, fps=args.fps, enable_depth=enable_depth)
