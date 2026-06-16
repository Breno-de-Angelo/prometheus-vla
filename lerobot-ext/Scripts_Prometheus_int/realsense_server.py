import pyrealsense2 as rs
import numpy as np
import cv2
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sim.sensor_utils import SensorServer


def start_realsense_zmq():
    pipeline = rs.pipeline()
    cfg = rs.config()

    # RGB + Depth streams da D435i
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

    try:
        profile = pipeline.start(cfg)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()  # m/unit (D435i: ~0.001)
        print(f"[RealSense D435i] Camera iniciada. depth_scale={depth_scale}")
    except Exception as e:
        print(f"[Erro] {e}")
        return

    align = rs.align(rs.stream.color)
    server = SensorServer()
    server.start_server(port=5555)
    print("[ZMQ] Servidor ativo na porta 5555 (RGB + depth)")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())          # RealSense rs.format.bgr8 → BGR
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)           # FONTE DA VERDADE: o robô já manda RGB.
            #   Consumidores (record/deploy) NÃO reconvertem. Dataset/treino = RGB. (ver init_lerobot_record)
            depth_raw = np.asanyarray(depth_frame.get_data())  # uint16 (unidades do sensor)

            # Depth em mm como uint16 (1 canal), enviado como PNG no multipart ZMQ.
            depth_mm = np.clip(
                depth_raw.astype(np.float32) * depth_scale * 1000.0, 0.0, 32767.0
            ).astype(np.uint16)

            t = time.time()
            server.send_images(
                {"head_camera": rgb, "head_camera_depth": depth_mm},
                {"head_camera": t, "head_camera_depth": t},
            )

    except KeyboardInterrupt:
        print("Encerrando...")
    finally:
        pipeline.stop()
        server.stop_server()


if __name__ == "__main__":
    start_realsense_zmq()
