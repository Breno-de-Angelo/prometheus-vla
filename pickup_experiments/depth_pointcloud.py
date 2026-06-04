#!/usr/bin/env python3
"""
Reconstroi a NUVEM DE PONTOS do depth da D435i (gravado no dataset) e plota como
bolinhas DENTRO do MuJoCo, junto da cena scene_43dof.xml. Serve pra comparar a
geometria REAL (o que a camera viu no mundo real) com o mundo SIMULADO (mesa, copo)
e ajustar posicao/tamanho do copo ate baterem.

Como funciona:
  - decodifica 1 frame do video de profundidade (e do RGB, pra colorir os pontos)
  - depth_m = valor_8bit / DEPTH_SCALE  (default 100 -> valor em cm)
  - retroprojeta cada pixel valido usando os intrinsecos da head_camera_depth (fovy=58)
  - transforma do frame da camera pro mundo usando a pose da camera no sim
    (robo na pose de pe, pelvis Z=0.910, faixa elastica)
  - injeta os pontos como esferas no user_scn do viewer passivo

Uso:
    conda activate g1
    python pickup_experiments/depth_pointcloud.py [--frame 0] [--ep N] \
        [--scale 100] [--step 4] [--data right]

Controles: orbite com o mouse. 'q' pra sair (na janela). Os pontos sao estaticos.
"""
import argparse
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
XML = HERE.parent / "unitree-g1-mujoco/assets/scene_43dof.xml"
DEPTH_MP4 = {
    "right": "/tmp/depth.mp4",       # baixado de pick_up_the_cup_2026-04-30
}
RGB_MP4 = {
    "right": "/tmp/rgb.mp4",
}
PARQUET = {
    "right": HERE.parent / "datasets/pick_up_the_cup_right_ref/data.parquet",
}
# braco direito ds idx 7-13 -> motores qpos do braco direito
RARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
BASE_Z = 0.910


def read_video_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"nao consegui ler frame {idx} de {path}")
    return fr  # BGR


def episode_frame0(df, ep):
    """indice global do frame 0 do episodio ep (offset cumulativo)."""
    lens = df.groupby("episode_index").size()
    off = int(lens.loc[:ep - 1].sum()) if ep > 0 else 0
    return off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="right")
    ap.add_argument("--ep", type=int, default=None, help="usa frame0 deste episodio")
    ap.add_argument("--frame", type=int, default=0, help="indice global do frame (se --ep ausente)")
    ap.add_argument("--scale", type=float, default=128.0, help="valor_8bit/scale = metros (calibrado)")
    ap.add_argument("--step", type=int, default=4, help="subamostragem de pixels (maior=menos pontos)")
    ap.add_argument("--maxd", type=float, default=2.0, help="descarta pontos alem de X metros")
    ap.add_argument("--torso", type=float, default=31.0,
                    help="inclinacao do tronco (waist pitch) em graus, + = pra frente (calibrado)")
    ap.add_argument("--pitch", type=float, default=0.0, help="(nao usado; ver --torso)")
    ap.add_argument("--cam-fwd", type=float, default=0.0,
                    help="desloca a origem da camera ao longo do eixo de visao em metros "
                         "(negativo = pra dentro da cabeca)")
    ap.add_argument("--minclip", type=float, default=0.05,
                    help="descarta pontos a menos de X metros da camera (ruido/campo proximo)")
    ap.add_argument("--max-spheres", type=int, default=6000,
                    help="limite de bolinhas renderizadas (acima disso o viewer estoura)")
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET[args.data])
    frame_idx = episode_frame0(df, args.ep) if args.ep is not None else args.frame
    print(f"[pcd] frame global {frame_idx}")

    depth = read_video_frame(DEPTH_MP4[args.data], frame_idx)[:, :, 0].astype(np.float32)
    try:
        rgb = read_video_frame(RGB_MP4[args.data], frame_idx)[:, :, ::-1]  # BGR->RGB
    except SystemExit:
        rgb = None

    H, W = depth.shape
    dm = depth / args.scale  # metros

    # ---- modelo + pose da camera ----
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    d.qpos[2] = BASE_Z
    # poe o braco direito na pose do frame (so cosmetico)
    if args.ep is not None or args.frame:
        a = np.stack(df["action"].values)[frame_idx]
        for k, jn in enumerate(RARM_JOINTS):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            d.qpos[m.jnt_qposadr[jid]] = a[7 + k]
    mujoco.mj_forward(m, d)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera_depth")
    waist_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")]

    # ---- intrinsecos (fovy=58) ----
    fovy = 58.0
    fy = (H / 2) / np.tan(np.radians(fovy / 2)); fx = fy
    cx, cy = W / 2, H / 2

    # grade de pixels validos (pre-calculada)
    vu = [(v, u) for v in range(0, H, args.step) for u in range(0, W, args.step)
          if args.minclip < dm[v, u] <= args.maxd]
    zz = np.array([dm[v, u] for v, u in vu])
    xc = np.array([(u - cx) / fx for v, u in vu]) * zz
    yc = np.array([(v - cy) / fy for v, u in vu]) * zz
    Pc = np.stack([xc, -yc, -zz], axis=1)        # pontos no frame da camera (mujoco)
    if rgb is not None:
        Cc = np.array([[*(rgb[v, u] / 255.0), 1.0] for v, u in vu])
    else:
        Cc = np.tile([0.1, 0.6, 1.0, 1.0], (len(vu), 1))
    # subamostra pro orcamento de esferas
    budget = args.max_spheres
    if len(Pc) > budget:
        sel = np.random.choice(len(Pc), budget, replace=False)
        Pc, Cc = Pc[sel], Cc[sel]
    print(f"[pcd] {len(Pc)} pontos desenhados (step={args.step}, scale={args.scale})")

    def cloud(torso_deg, neck_deg):
        """inclina o TRONCO (waist pitch, real, robo inclina junto) e a CABECA/camera
        (neck: gira a camera no lugar, sem junta real). Devolve (cam_pos, pontos_mundo)."""
        d.qpos[waist_qadr] = np.radians(torso_deg)
        mujoco.mj_forward(m, d)
        cp = d.cam_xpos[cid].copy()
        R = d.cam_xmat[cid].reshape(3, 3).copy()
        if neck_deg:
            th = np.radians(neck_deg)
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(th), -np.sin(th)],
                           [0, np.sin(th), np.cos(th)]])
            R = R @ Rx
        cp = cp + R[:, 2] * (-args.cam_fwd)
        return cp, cp + Pc @ R.T

    state = {"torso": float(args.torso), "neck": float(args.pitch), "dirty": False}

    # ---- viewer interativo ----
    def fill(viewer):
        cp, PW = cloud(state["torso"], state["neck"])
        n = 0
        mujoco.mjv_initGeom(viewer.user_scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.02, 0, 0]), cp, np.eye(3).flatten(),
                            np.array([1.0, 0.0, 1.0, 1.0], np.float32))
        n += 1
        for i in range(len(PW)):
            mujoco.mjv_initGeom(viewer.user_scn.geoms[n], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.005, 0, 0]), PW[i],
                                np.eye(3).flatten(), Cc[i].astype(np.float32))
            n += 1
        viewer.user_scn.ngeom = n
        print(f"[pcd] tronco={state['torso']:+.1f}  cabeca/cam={state['neck']:+.1f} deg  "
              f"(Z medio da nuvem={PW[:,2].mean():.3f}, mesa sim topo=0.86)")

    def key_cb(keycode):
        if keycode == 264:      # DOWN -> inclina TRONCO pra frente (robo inclina junto)
            state["torso"] += 1.0; state["dirty"] = True
        elif keycode == 265:    # UP -> volta o tronco
            state["torso"] -= 1.0; state["dirty"] = True
        elif keycode == 46:     # '.' -> inclina so a CABECA/CAMERA pra baixo
            state["neck"] += 1.0; state["dirty"] = True
        elif keycode == 44:     # ',' -> volta a cabeca/camera
            state["neck"] -= 1.0; state["dirty"] = True

    state["dirty"] = False
    with mujoco.viewer.launch_passive(m, d, key_callback=key_cb) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 1
        print(f"[pcd] maxgeom={viewer.user_scn.maxgeom}. Setas ↑/↓ (ou ./,) inclinam o rosto do robo.")
        fill(viewer)
        while viewer.is_running():
            if state["dirty"]:
                fill(viewer); state["dirty"] = False
            viewer.sync()
            time.sleep(0.05)


if __name__ == "__main__":
    main()
