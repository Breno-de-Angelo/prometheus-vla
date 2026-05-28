import pyrealsense2 as rs
import numpy as np
import cv2
import zmq
import time
import sys
from pathlib import Path

# Garante que o Python ache o utils do seu projeto
sys.path.insert(0, str(Path(__file__).parent))
from sim.sensor_utils import SensorServer, ImageUtils

def start_real_robot_cameras():
    # ==========================================================
    # CONFIGURAÇÕES DE RESOLUÇÃO
    # ==========================================================
    HEAD_WIDTH, HEAD_HEIGHT = 640, 480
    DEPTH_WIDTH, DEPTH_HEIGHT = 640, 480 
    FPS = 30

    # ==========================================================
    # 1. INICIALIZA A INTEL REALSENSE D435i
    # ==========================================================
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device('327122071538') # Serial do Prometheus

    config.enable_stream(rs.stream.color, HEAD_WIDTH, HEAD_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)

    try:
        profile = pipeline.start(config)
        
        # ==========================================================
        # 🚀 O PULO DO GATO: FORÇAR 30 FPS (DESATIVAR PRIORIDADE DE EXPOSIÇÃO)
        # ==========================================================
        # Pegamos o sensor de cor (geralmente índice 1)
        color_sensor = profile.get_device().query_sensors()[1]
        if color_sensor.supports(rs.option.auto_exposure_priority):
            # 0 desativa a prioridade, forçando a câmera a manter os 30 FPS constantes
            color_sensor.set_option(rs.option.auto_exposure_priority, 0)
        
        print(f"[RealSense D435i] Iniciada com sucesso a {FPS} FPS fixos!")
    except Exception as e:
        print(f"[Erro RealSense] {e}")
        return

    server = SensorServer()
    server.start_server(port=5555)

    align_to = rs.stream.color
    align = rs.align(align_to)

    print("[ZMQ] Servidor de Visão ativo na porta 5555. Aguardando LeRobot...")

    try:
        # Loop otimizado
        while True:
            # Reduzimos o timeout para 1000ms para o script não ficar "preso"
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # --- TRATAMENTO RGB ---
            img_bgr = np.asanyarray(color_frame.get_data())
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # --- TRATAMENTO DEPTH (CORRIGIDO PARA IA) ---
            depth_raw = np.asanyarray(depth_frame.get_data())
            
            # 1. Corta tudo acima de 2 metros (foca na manipulação)
            depth_clipped = np.clip(depth_raw, 0, 2000)
            
            # 2. Converte metricamente para 8-bits (escala de cinza linear)
            depth_8bit = (depth_clipped * (255.0 / 2000.0)).astype(np.uint8)
            
            # 3. Replica o canal cinza 3 vezes (R=Depth, G=Depth, B=Depth)
            # Sem Colormap! Apenas cinza triplicado para o codec MP4 aceitar.
            depth_3c = cv2.cvtColor(depth_8bit, cv2.COLOR_GRAY2RGB)

            # --- ENVIO ---
            current_time = time.time()
            message = {
                "images": {
                    "head_camera": ImageUtils.encode_image(img_rgb),
                    "head_camera_depth": ImageUtils.encode_image(depth_3c), # Envia a imagem cinza corrigida
                },
                "timestamps": {
                    "head_camera": current_time,
                    "head_camera_depth": current_time,
                }
            }
            server.send_message(message)

    except KeyboardInterrupt:
        print("\nEncerrando transmissão...")
    finally:
        pipeline.stop()
        server.stop_server()

if __name__ == "__main__":
    start_real_robot_cameras()