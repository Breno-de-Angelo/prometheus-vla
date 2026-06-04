#!/usr/bin/env python3
"""
Host de VALIDAÇÃO headless do sim G1 (mesa + copo já estão na cena scene_43dof.xml).

- Sobe o MuJoCo offscreen (sem janela), com o UnitreeSdk2Bridge (DDS no 'lo').
- Sobe o run_g1_server (ponte ZMQ<->DDS do corpo, portas 6000-6003) em subprocesso.
- Publica câmeras em ZMQ 5555 (o cliente ZMQCamera precisa disso).
- A cada ~20Hz grava em /tmp/sim_val/state.jsonl:
    t, body_q[29], left_hand_q[7], right_hand_q[7],
    body_cmd_q/kp (braços), left/right_hand_cmd_q/kp/tau  <-- o que o robô MANDOU
  => prova direta do "salto" (braço q saltando) e do kp da mão esquerda.
- Renderiza frames 3a-pessoa em /tmp/sim_val/frames/f_<ms>.png:
    periodicamente e sempre que o arquivo /tmp/sim_val/GRAB existir (toque p/ capturar).

Uso (de unitree-g1-mujoco/):  MUJOCO_GL=egl python sim_validate_host.py --seconds 60
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import yaml
import mujoco

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from sim.simulator_factory import SimulatorFactory, init_channel  # noqa: E402

_REPO = _HERE.parent
_RUN_G1_SERVER = _REPO / "lerobot-ext" / "robot" / "unitree_g1" / "run_g1_server.py"

OUT = Path("/tmp/sim_val")
FRAMES = OUT / "frames"
GRAB = OUT / "GRAB"
STATE = OUT / "state.jsonl"

# índices de braço dentro dos 29 motores do corpo (G1 29dof)
LEFT_ARM = list(range(15, 22))
RIGHT_ARM = list(range(22, 29))
ARM = LEFT_ARM + RIGHT_ARM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--no-bridge", action="store_true", help="não sobe run_g1_server")
    ap.add_argument("--no-cam", action="store_true", help="não publica câmeras")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    if STATE.exists():
        STATE.unlink()
    for f in FRAMES.glob("*.png"):
        f.unlink()

    with open(_HERE / "config.yaml") as f:
        config = yaml.safe_load(f)

    init_channel(config=config)

    cam_cfgs = {} if args.no_cam else {
        "head_camera": {"height": 480, "width": 640},
        "head_camera_depth": {"height": 480, "width": 640},
    }
    sim = SimulatorFactory.create_simulator(
        config=config, env_name="default",
        onscreen=False, offscreen=not args.no_cam, camera_configs=cam_cfgs,
    )
    de = sim.sim_env
    bridge = sim.unitree_bridge
    print(f"[host] sim criado. bridge={bridge is not None} hands L={len(de.left_hand_index)} R={len(de.right_hand_index)}", flush=True)

    srv = None
    if not args.no_bridge:
        env = dict(os.environ, G1_DDS_IFACE="lo")
        srv_log = open(OUT / "run_g1_server.log", "w")
        srv = subprocess.Popen([sys.executable, str(_RUN_G1_SERVER)], env=env,
                               stdout=srv_log, stderr=subprocess.STDOUT)
        print(f"[host] run_g1_server PID={srv.pid} (log {OUT/'run_g1_server.log'})", flush=True)

    if not args.no_cam:
        sim.start_image_publish_subprocess(start_method=config.get("MP_START_METHOD", "spawn"),
                                           camera_port=5555)
        print("[host] câmeras publicando em tcp://localhost:5555", flush=True)

    # renderer 3a-pessoa
    renderer = mujoco.Renderer(de.mj_model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.azimuth = 130
    cam.elevation = -25
    cam.distance = 2.2
    cam.lookat = np.array([0.25, 0.0, 0.9])

    def grab(tag):
        mujoco.mj_forward(de.mj_model, de.mj_data)
        renderer.update_scene(de.mj_data, camera=cam)
        img = renderer.render()  # RGB HxWx3
        import imageio.v2 as imageio
        ms = int((time.time() - t0) * 1000)
        path = FRAMES / f"f_{ms:07d}_{tag}.png"
        imageio.imwrite(path, img)
        return path

    sim_dt = de.sim_dt
    viewer_every = max(1, int(config.get("VIEWER_DT", 0.02) / sim_dt))
    image_every = max(1, int(config.get("IMAGE_DT", 0.0333) / sim_dt))
    fh = open(STATE, "w")
    t0 = time.time()
    sim_cnt = 0
    last_log = 0.0
    last_frame = 0.0
    fh_log_period = 0.05   # 20 Hz
    frame_period = 2.0     # frame automático a cada 2s

    def log_row():
        obs = de.prepare_obs()
        row = {
            "t": round(time.time() - t0, 4),
            "body_q": [round(float(x), 4) for x in obs["body_q"]],
            "left_hand_q": [round(float(x), 4) for x in obs.get("left_hand_q", [])],
            "right_hand_q": [round(float(x), 4) for x in obs.get("right_hand_q", [])],
        }
        if bridge is not None and bridge.low_cmd:
            lc = bridge.low_cmd.motor_cmd
            row["arm_cmd_q"] = {i: round(float(lc[i].q), 4) for i in ARM}
            row["arm_cmd_kp"] = {i: round(float(lc[i].kp), 3) for i in ARM}
        if bridge is not None and getattr(bridge, "left_hand_cmd", None):
            lh = bridge.left_hand_cmd.motor_cmd
            rh = bridge.right_hand_cmd.motor_cmd
            row["left_hand_cmd_kp"] = [round(float(lh[i].kp), 3) for i in range(7)]
            row["left_hand_cmd_q"] = [round(float(lh[i].q), 4) for i in range(7)]
            row["left_hand_cmd_tau"] = [round(float(lh[i].tau), 4) for i in range(7)]
            row["right_hand_cmd_kp"] = [round(float(rh[i].kp), 3) for i in range(7)]
            row["right_hand_cmd_q"] = [round(float(rh[i].q), 4) for i in range(7)]
        fh.write(json.dumps(row) + "\n")
        fh.flush()

    grab("start")
    log_row()
    print(f"[host] rodando {args.seconds}s. Toque {GRAB} p/ capturar frame.", flush=True)
    end = t0 + args.seconds
    try:
        while time.time() < end:
            de.sim_step()
            sim_cnt += 1
            now = time.time() - t0
            if sim_cnt % viewer_every == 0:
                de.update_viewer()
            if not args.no_cam and sim_cnt % image_every == 0:
                de.update_render_caches()
            if now - last_log >= fh_log_period:
                log_row()
                last_log = now
            if now - last_frame >= frame_period:
                grab("auto")
                last_frame = now
            if GRAB.exists():
                p = grab("manual")
                GRAB.unlink()
                print(f"[host] frame manual -> {p}", flush=True)
            # ritmo
            el = (time.time() - t0) - now
            sleep = sim_dt - el
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        grab("end")
        log_row()
        fh.close()
        print(f"[host] fim. state={STATE} frames={FRAMES}", flush=True)
        try:
            sim.close()
        except Exception:
            pass
        if srv is not None:
            srv.terminate()
            try:
                srv.wait(timeout=4)
            except Exception:
                srv.kill()


if __name__ == "__main__":
    main()
