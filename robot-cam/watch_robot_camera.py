import base64, json
import cv2
import numpy as np
import zmq

ROBOT_IP = "192.168.68.71"
PORT = 5555

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.setsockopt(zmq.SUBSCRIBE, b"")
sock.setsockopt(zmq.RCVHWM, 2)
sock.setsockopt(zmq.RCVTIMEO, 5000)
sock.connect(f"tcp://{ROBOT_IP}:{PORT}")
print(f"connected to tcp://{ROBOT_IP}:{PORT}", flush=True)

try:
    while True:
        try:
            raw = sock.recv_string()
        except zmq.Again:
            print("timeout — sem frames. Server tá rodando? IP certo?", flush=True)
            break
        msg = json.loads(raw)
        cams = msg.get("images", {})
        for name in ("head_camera", "head_camera_depth"):
            if name in cams:
                jpg = base64.b64decode(cams[name])
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                cv2.imshow(name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
