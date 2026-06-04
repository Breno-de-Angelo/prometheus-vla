#!/usr/bin/env python3
"""
Teleop do braco DIREITO do G1 no sim por WEBCAM (MediaPipe Pose), via ZMQ.

Captura seu braco na webcam, estima ombro/cotovelo/punho (pose 3D mundo do MediaPipe),
mapeia (retargeting geometrico aproximado) pras juntas do braco direito do G1 (motores
22-28) e streama pro sim (mesmo protocolo do replay). Tecla **G** = fecha a mao (grasp);
solta = abre. **Q** = sair.

RODA NO VENV ISOLADO (nao no g1):
    source ~/webcam_teleop_env/bin/activate
    python pickup_experiments/webcam_teleop.py [--flip-pitch] [--flip-roll] [--flip-yaw] [--cam 0]

Dica: fique de frente pra camera, tronco visivel. Mova o braco direito; o robo segue.
Os sinais/ganhos sao calibraveis pelas flags (a direcao pode vir invertida dependendo da camera).
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import zmq
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (PoseLandmarker, PoseLandmarkerOptions, RunningMode)
import mediapipe as mp

MODEL = str(Path.home() / "webcam_teleop_env/pose_landmarker_lite.task")
ADDR = "tcp://127.0.0.1:6001"
# indices MediaPipe Pose
R_SH, R_EL, R_WR = 12, 14, 16
HAND_OPEN = [0.0] * 7
HAND_CLOSE = [0.0, -1.5, -1.5, 1.5, 1.5, 1.5, 1.5]
# limites das juntas do braco direito (rad) p/ clamp [pitch,roll,yaw,elbow,wr,wp,wy]
ARM_LO = np.array([-3.0, -1.6, -2.6, -0.1, -1.9, -1.6, -1.6])
ARM_HI = np.array([2.7, 0.5, 2.6, 2.3, 1.9, 1.6, 1.6])


def build_body(arm7, leg_kp=120.0, leg_kd=2.5, arm_kp=200.0, arm_kd=4.0):
    out = []
    for i in range(29):
        if 22 <= i <= 28:
            q = float(arm7[i - 22]); kp, kd = arm_kp, arm_kd
        else:
            q = 0.0; kp, kd = leg_kp, leg_kd
        out.append({"idx": i, "q": q, "kp": kp, "kd": kd})
    return out


def build_hand(vals7, kp=20.0, kd=1.0):
    return [{"idx": i, "q": float(vals7[i]), "kp": kp, "kd": kd} for i in range(7)]


def retarget(world, args):
    """world: lista de landmarks 3D (x dir-do-sujeito, y baixo, z p/ camera).
    Devolve 7 angulos do braco direito do G1 (aprox)."""
    def v(i):
        lm = world[i]
        return np.array([lm.x, lm.y, lm.z])
    S, E, W = v(R_SH), v(R_EL), v(R_WR)
    ua = E - S            # braco (ombro->cotovelo)
    fa = W - E            # antebraco (cotovelo->punho)
    nua = ua / (np.linalg.norm(ua) + 1e-6)
    nfa = fa / (np.linalg.norm(fa) + 1e-6)

    # MediaPipe world: x=esquerda+ do sujeito, y=baixo+, z=frente(-)/tras(+) aprox.
    # shoulder_pitch: braco abaixado (nua.y~+1) -> ~0 ; braco pra frente (nua.z~-1) -> levanta
    pitch = np.arctan2(-nua[2], nua[1])      # frente/baixo
    roll = np.arcsin(np.clip(-nua[0], -1, 1))  # braco pro lado
    # elbow: flexao entre -ua e fa
    elbow = np.arccos(np.clip(np.dot(-nua, nfa), -1, 1))
    # yaw/wrist: aprox neutro (dificil sem mais landmarks)
    yaw = 0.0
    q = np.array([pitch, roll, yaw, elbow, 0.0, 0.0, 0.0])
    # sinais calibraveis
    if args.flip_pitch: q[0] = -q[0]
    if args.flip_roll: q[1] = -q[1]
    if args.flip_yaw: q[2] = -q[2]
    q[0] += args.pitch_offset
    return np.clip(q, ARM_LO, ARM_HI)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--flip-pitch", action="store_true")
    ap.add_argument("--flip-roll", action="store_true")
    ap.add_argument("--flip-yaw", action="store_true")
    ap.add_argument("--pitch-offset", type=float, default=0.0)
    ap.add_argument("--smooth", type=float, default=0.3, help="suavizacao (0=sem, ->1 mais suave)")
    args = ap.parse_args()

    ctx = zmq.Context(); sock = ctx.socket(zmq.PUSH); sock.connect(ADDR)
    print(f"[webcam] ZMQ -> {ADDR}")

    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=RunningMode.VIDEO, num_poses=1)
    landmarker = PoseLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(args.cam)
    grasp = False
    q_smooth = None
    t0 = time.time()
    print("[webcam] G = fecha a mao | Q = sair. Fique de frente pra camera.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # espelho (mais intuitivo)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - t0) * 1000)
        res = landmarker.detect_for_video(mpimg, ts)

        status = "SEM POSE"
        if res.pose_world_landmarks:
            q = retarget(res.pose_world_landmarks[0], args)
            q_smooth = q if q_smooth is None else (args.smooth * q_smooth + (1 - args.smooth) * q)
            arm = q_smooth
            msg = {"body_motors": build_body(arm),
                   "left_hand": build_hand(HAND_OPEN),
                   "right_hand": build_hand(HAND_CLOSE if grasp else HAND_OPEN, kp=25 if grasp else 20)}
            sock.send_string(json.dumps(msg))
            status = f"pitch={arm[0]:+.2f} roll={arm[1]:+.2f} elbow={arm[3]:+.2f} | GRASP={'ON' if grasp else 'off'}"
            # desenha landmarks do braco (em coords de imagem)
            if res.pose_landmarks:
                h, w = frame.shape[:2]
                for i in (R_SH, R_EL, R_WR):
                    lm = res.pose_landmarks[0][i]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 7, (0, 255, 0), -1)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if grasp else (255, 255, 0), 2)
        cv2.imshow("Webcam Teleop (G=grasp, Q=sair)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q')):
            break
        elif k in (ord('g'), ord('G')):
            grasp = not grasp
            print(f"[webcam] GRASP {'ON' if grasp else 'off'}")

    cap.release(); cv2.destroyAllWindows(); sock.close(); ctx.term()
    print("[webcam] encerrado")


if __name__ == "__main__":
    main()
