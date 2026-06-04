#!/usr/bin/env python3
"""Renderiza poses do G1 na cena (mesa+copo) p/ comparar salto vs repouso.
MUJOCO_GL=egl python render_pose.py"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import imageio.v2 as imageio
from pathlib import Path

HF = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/models--lerobot--unitree-g1-mujoco/snapshots/a38dc8617f0fca51b38e9354dc58ee35ad850fb5"))
os.chdir(HF / "assets")
m = mujoco.MjModel.from_xml_path("scene_43dof.xml")
d = mujoco.MjData(m)

# qposadr das juntas de braço (do modelo)
ADR = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in [
    "left_shoulder_pitch_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_elbow_joint"]}

renderer = mujoco.Renderer(m, height=480, width=640)
cam = mujoco.MjvCamera()
cam.azimuth = 130; cam.elevation = -22; cam.distance = 2.3
cam.lookat = np.array([0.25, 0.0, 0.95])

OUT = Path("/tmp/sim_val/frames")
OUT.mkdir(parents=True, exist_ok=True)


def render(tag, shoulder_pitch, elbow):
    mujoco.mj_resetData(m, d)
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[ADR["left_shoulder_pitch_joint"]] = shoulder_pitch
    d.qpos[ADR["right_shoulder_pitch_joint"]] = shoulder_pitch
    d.qpos[ADR["left_elbow_joint"]] = elbow
    d.qpos[ADR["right_elbow_joint"]] = elbow
    mujoco.mj_forward(m, d)
    renderer.update_scene(d, camera=cam)
    img = renderer.render()
    p = OUT / f"pose_{tag}.png"
    imageio.imwrite(p, img)
    print(f"render {tag}: shoulder_pitch={shoulder_pitch} elbow={elbow} -> {p}")


# Frame 0 medido no diag: shoulder_pitch~0.05, elbow~0.42 (braços apoiados/fletidos)
render("repouso_COM_fix", 0.05, 0.42)
# Sem fix: 1o send_action crava q=0 com kp=80 -> braços retos pra cima (salto)
render("salto_SEM_fix", 0.0, 0.0)
print("OK")
