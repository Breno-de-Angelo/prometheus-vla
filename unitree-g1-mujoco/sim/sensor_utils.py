"""Standalone sensor utilities for camera image publishing via ZMQ"""
import base64
import json
from dataclasses import dataclass
from typing import Any, Dict

import cv2
import numpy as np
import zmq


@dataclass
class ImageMessageSchema:
    timestamps: Dict[str, float]
    images: Dict[str, np.ndarray]

    def serialize(self) -> Dict[str, Any]:
        serialized_msg = {"timestamps": self.timestamps, "images": {}}
        for key, image in self.images.items():
            # Proteção dupla: garante que manda PNG (Lossless) para 16-bits
            if 'depth' in key.lower():
                serialized_msg["images"][key] = ImageUtils.encode_depth_image(image)
            else:
                serialized_msg["images"][key] = ImageUtils.encode_image(image)
        return serialized_msg

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> "ImageMessageSchema":
        timestamps = data.get("timestamps", {})
        images = {}
        for key, value in data.get("images", {}).items():
            if isinstance(value, str):
                # Descomprime do jeito certo com base no nome da lente
                if 'depth' in key.lower():
                    images[key] = ImageUtils.decode_depth_image(value)
                else:
                    images[key] = ImageUtils.decode_image(value)
            else:
                images[key] = value
        return ImageMessageSchema(timestamps=timestamps, images=images)

    def asdict(self) -> Dict[str, Any]:
        return {"timestamps": self.timestamps, "images": self.images}


class SensorServer:
    """ZMQ-based sensor server for publishing camera images"""
    
    def start_server(self, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)  # high water mark
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")
        print(f"Sensor server running at tcp://*:{port}")

        self.message_sent = 0
        self.message_dropped = 0

    def stop_server(self):
        self.socket.close()
        self.context.term()

    def send_message(self, data: Dict[str, Any]):
        try:
            json_str = json.dumps(data)
            self.socket.send_string(json_str, flags=zmq.NOBLOCK)
        except zmq.Again:
            self.message_dropped += 1
            print(f"[Warning] message dropped: {self.message_dropped}")
        self.message_sent += 1

        if self.message_sent % 100 == 0:
            print(
                f"[Sensor server] Message sent: {self.message_sent}, message dropped: {self.message_dropped}"
            )


class SensorClient:
    """ZMQ-based sensor client for subscribing to camera images"""
    
    def start_client(self, server_ip: str, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.CONFLATE, True)  # last msg only.
        self.socket.setsockopt(zmq.RCVHWM, 3)  # queue size 3 for receive buffer
        self.socket.connect(f"tcp://{server_ip}:{port}")

    def stop_client(self):
        self.socket.close()
        self.context.term()

    def receive_message(self):
        json_str = self.socket.recv_string()
        return json.loads(json_str)


class ImageUtils:
    """Utilities for encoding/decoding images for network transmission"""
    
    @staticmethod
    def encode_image(image: np.ndarray) -> str:
        """Encode numpy image to base64-encoded JPEG string"""
        _, color_buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return base64.b64encode(color_buffer).decode("utf-8")

    @staticmethod
    def encode_depth_image(image: np.ndarray) -> str:
        """Encode depth image to base64-encoded PNG string"""
        depth_compressed = cv2.imencode(".png", image)[1].tobytes()
        return base64.b64encode(depth_compressed).decode("utf-8")

    @staticmethod
    def decode_image(image: str) -> np.ndarray:
        """Decode base64-encoded JPEG string to numpy image (Returns RGB)"""
        color_data = base64.b64decode(image)
        color_array = np.frombuffer(color_data, dtype=np.uint8)
        bgr_image = cv2.imdecode(color_array, cv2.IMREAD_COLOR)
        # CORREÇÃO: Retorna o array em RGB nativo para o cliente
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def decode_depth_image(image: str) -> np.ndarray:
        """Decode base64-encoded PNG string to depth image"""
        depth_data = base64.b64decode(image)
        depth_array = np.frombuffer(depth_data, dtype=np.uint8)
        return cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)

