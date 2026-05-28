#!/usr/bin/env python3
"""
Live camera viewer for MuJoCo simulator using Pure OpenCV
Ultra-Fast - 30+ FPS with resizable windows
"""
import argparse
import sys
import time
from pathlib import Path

# Add sim module to path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from sim.sensor_utils import SensorClient, ImageUtils

class CameraViewer:
    def __init__(self, host, port):
        self.client = SensorClient()
        self.client.start_client(server_ip=host, port=port)
        
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0
        self.initialized_windows = set() # Controle para criar janelas apenas uma vez
        
    def _process_image(self, cam_name, img_data):
        """Decode and process image based on camera type"""
        if isinstance(img_data, str):
            try:
                img = ImageUtils.decode_image(img_data)
            except Exception:
                return None
        elif isinstance(img_data, np.ndarray):
            img = img_data
        else:
            return None

        if img is None or not isinstance(img, np.ndarray):
            return None

        # Inverte de RGB (Padrão IA/VR) para BGR (Padrão Monitor/OpenCV)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_out = img
            
        return img_out

    def start(self):
        """Start the live viewer with OpenCV"""
        print("⏳ Aguardando conexão com as lentes do simulador...")
        
        data = self.client.receive_message()
        if not data or "images" not in data:
            print("❌ Nenhuma câmera encontrada no stream!")
            return

        print(f"\n{'='*60}")
        print("📹 Live Resizable Viewer (OPENCV TURBO) started!")
        print("-> ARRASTE as bordas das janelas para expandir.")
        print("-> Pressione 'Q' para sair.")
        print(f"{'='*60}\n")
        
        try:
            while True:
                data = self.client.receive_message()
                
                # Cálculo de FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_time)
                    self.frame_count = 0
                    self.last_time = current_time

                images_dict = data.get("images", {})
                
                for cam_name, img_data in images_dict.items():
                    img_bgr = self._process_image(cam_name, img_data)
                    if img_bgr is None:
                        continue

                    # --- MÁGICA DA EXPANSÃO ---
                    # Se for a primeira vez que vemos esta câmera, configuramos a janela como redimensionável
                    window_name = f"Sensor: {cam_name}"
                    if window_name not in self.initialized_windows:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        # Mantém a proporção correta (não achata) enquanto você expande
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
                        self.initialized_windows.add(window_name)

                    # Desenha a tarja do FPS (ajustada para ser visível mesmo em telas grandes)
                    cv2.rectangle(img_bgr, (5, 5), (160, 45), (0, 0, 0), -1)
                    cv2.putText(img_bgr, f'FPS: {self.fps:.1f}', (15, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Renderiza na janela redimensionável
                    cv2.imshow(window_name, img_bgr)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping viewer...")
        finally:
            self.client.stop_client()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Live camera viewer for MuJoCo simulator")
    parser.add_argument("--host", type=str, default="localhost", help="Simulator host address")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port")
    args = parser.parse_args()
    
    viewer = CameraViewer(host=args.host, port=args.port)
    viewer.start()

if __name__ == "__main__":
    main()