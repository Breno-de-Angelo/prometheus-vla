#!/usr/bin/env python
"""Host do sim para inferência distribuída.

Roda o MuJoCo (viewer + câmeras por ZMQ na porta 5555), lança o run_g1_server.py como
subprocesso (bridge DDS<->ZMQ do corpo e das mãos, portas 6000-6003) e abre janelas de RGB/DEPTH.
A VLA roda noutra máquina em modo ZMQ, apontando o --robot-ip para este host.

Uso (a partir de unitree-g1-mujoco/):  python laptop_sim_host.py
"""
import base64
import json
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread

import cv2
import mujoco
import numpy as np
import zmq

from env import make_env

_REPO = Path(__file__).resolve().parent.parent
_RUN_G1_SERVER = _REPO / "lerobot-ext" / "robot" / "unitree_g1" / "run_g1_server.py"


def camera_display(stop: Event, port: int = 5555):
    """Abre janelas cv2 de RGB e DEPTH lendo a stream ZMQ que o sim publica."""
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://localhost:{port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.RCVTIMEO, 500)
    try:
        while not stop.is_set():
            try:
                data = json.loads(sub.recv_string())
            except zmq.Again:
                continue
            imgs = data.get("images", {})
            if "head_camera" in imgs:
                rgb = cv2.imdecode(np.frombuffer(base64.b64decode(imgs["head_camera"]), np.uint8),
                                   cv2.IMREAD_COLOR)
                if rgb is not None:
                    cv2.imshow("RGB - head_camera (sim)", rgb)
            if "head_camera_depth" in imgs:
                d = cv2.imdecode(np.frombuffer(base64.b64decode(imgs["head_camera_depth"]), np.uint8),
                                 cv2.IMREAD_UNCHANGED)
                if d is not None:
                    if d.ndim == 3:
                        d = d[:, :, 0]
                    dn = cv2.normalize(d.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imshow("DEPTH - head_camera_depth (sim)", cv2.applyColorMap(dn, cv2.COLORMAP_JET))
            cv2.waitKey(1)
    finally:
        cv2.destroyAllWindows()
        sub.close()
        ctx.term()


def main():
    # bridge DDS<->ZMQ (corpo + mãos) na rede; G1_DDS_IFACE=lo casa o DDS do sim
    import os
    _srv_env = dict(os.environ, G1_DDS_IFACE="lo")
    srv = subprocess.Popen([sys.executable, str(_RUN_G1_SERVER)], env=_srv_env)

    # sim + câmeras (ZMQ 5555) + DDS + viewer
    env = make_env(cameras=["head_camera", "head_camera_depth"])
    dt = env.sim_dt

    # distância mão<->copo (~1Hz)
    de = env.sim_env
    _hand = mujoco.mj_name2id(de.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_1_link")
    _cup = mujoco.mj_name2id(de.mj_model, mujoco.mjtObj.mjOBJ_BODY, "objeto_customizado")
    _n = 0

    # janelas RGB + DEPTH
    stop = Event()
    tcam = Thread(target=camera_display, args=(stop,), name="CamDisplay", daemon=True)
    tcam.start()

    print("[host] rodando. Ctrl-C para parar.", flush=True)
    try:
        while True:
            t0 = time.time()
            env.step()
            _n += 1
            if _n % 250 == 0 and _hand >= 0 and _cup >= 0:   # ~1Hz
                h = de.mj_data.xpos[_hand]
                c = de.mj_data.xpos[_cup]
                print(f"[reach] mao={np.round(h,3)} copo={np.round(c,3)} "
                      f"dist={float(np.linalg.norm(h - c)) * 100:.1f}cm  (dz={float(h[2]-c[2])*100:+.1f})", flush=True)
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("[host] parando.")
    finally:
        stop.set()
        tcam.join(timeout=2.0)
        try:
            env.close()
        except Exception:
            pass
        srv.terminate()
        try:
            srv.wait(timeout=4.0)
        except Exception:
            srv.kill()


if __name__ == "__main__":
    main()
