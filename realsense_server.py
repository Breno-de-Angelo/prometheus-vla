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
    config = rs.config()

    # Stream RGB da D435i
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)
        print("[RealSense D435i] Câmera iniciada.")
    except Exception as e:
        print(f"[Erro] {e}")
        return

    server = SensorServer()
    server.start_server(port=5555)

    print("[ZMQ] Servidor ativo na porta 5555")

    try:
        while True:

            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())

            server.send_images({"head_camera": img}, {"head_camera": time.time()})

    except KeyboardInterrupt:
        print("Encerrando...")

    finally:
        pipeline.stop()
        server.stop_server()


if __name__ == "__main__":
    start_realsense_zmq()
