#!/usr/bin/env python3
"""
SYNC AO VIVO sim<->real: a cada frame da câmera real, detecta os ArUcos da mesa,
recalcula a pose da câmera (PnP + ajuste de plano) e move a câmera/marcadores/copo
do SIM em tempo real (atualiza pose direto no MuJoCo, sem recarregar o modelo).
Mostra REAL | SIM(alinhado) lado a lado.

Câmera real via ZMQ :5555. Intrínsecos de assets/realsense_color_intrinsics.json.
Roda no env g1 (foreground, por causa do GUI). Q/ESC fecha.

    conda activate g1
    MUJOCO_GL=egl python live_sync.py --host 192.168.68.71
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import cv2
import zmq
import mujoco

HERE = Path(__file__).parent
SIM_DIR = HERE / "unitree-g1-mujoco"
SCENE_XML = SIM_DIR / "assets" / "scene_43dof.xml"
sys.path.insert(0, str(SIM_DIR))
from sim.sensor_utils import SensorClient, ImageUtils  # noqa: E402

INTR_CACHE = HERE / "assets" / "realsense_color_intrinsics.json"
TABLE_IDS = [0, 1, 2, 3]
CUP_IDS = [4, 5, 6, 7]
KNOWN = set(TABLE_IDS) | set(CUP_IDS)
SIZE_M = {**{i: 0.060 for i in TABLE_IDS}, **{i: 0.025 for i in CUP_IDS}}
SIM_TABLE_CENTER = np.array([0.45, 0.0, 0.756])
CUP_RADIUS_M = 0.044
SCALE = 0.866  # depth/pnp calibrado (invariante p/ a imagem; afeta só métrica/copo)


def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s
    else:
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            s = math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
            w,x,y,z = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
        elif i == 1:
            s = math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
            w,x,y,z = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
        else:
            s = math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
            w,x,y,z = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


def flat_quat(Rwm):
    """Marcador deitado na mesa: normal SEMPRE pra cima (+z), yaw do PnP.
    Evita o flip da normal do IPPE que escondia marcadores (face p/ baixo)."""
    xdir = Rwm[:, 0].copy()
    xdir[2] = 0.0
    if np.linalg.norm(xdir) < 1e-6:
        xdir = Rwm[:, 1].copy(); xdir[2] = 0.0
    yaw = math.atan2(xdir[1], xdir[0])
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def build_model(W, H, fovy):
    """Monta a scene alinhada UMA vez: aligned_cam + group4 + tampo grande."""
    xml = SCENE_XML.read_text()
    cam = (f'<camera name="aligned_cam" pos="0.4 0 1.3" xyaxes="1 0 0 0 1 0" '
           f'fovy="{fovy:.3f}"/>')
    xml = xml.replace('<camera name="global_view"', cam + '\n    <camera name="global_view"')
    xml = re.sub(r'(<global\b)', r'\1 offwidth="848" offheight="480"', xml, count=1)
    for g in ["geometria_mesa", "toalha_mesa", "aruco0", "aruco1", "aruco2",
              "aruco3", "copo_geom_visual"]:
        xml = re.sub(rf'(name="{g}")', r'\1 group="4"', xml, count=1)
    z_tab = SIM_TABLE_CENTER[2] - 0.001
    big = (f'<geom name="align_tabletop" type="box" size="2.0 2.0 0.001" '
           f'pos="{SIM_TABLE_CENTER[0]:.3f} {SIM_TABLE_CENTER[1]:.3f} {z_tab:.4f}" '
           f'material="toalha_mat" group="4" contype="0" conaffinity="0"/>')
    xml = xml.replace('<geom name="floor"', big + '\n    <geom name="floor"')
    xml = xml.replace('diffuse="0.6 0.6 0.6"', 'diffuse="0.3 0.3 0.3"')
    tmp = SCENE_XML.parent / "_live_aligned.xml"
    tmp.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    return model


def make_detector():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 35
    p.adaptiveThreshWinSizeStep = 4
    D = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.ArucoDetector(D, p)


def solve(bgr, detector, K, dist):
    """Detecta + PnP. Retorna {id: (tvec(3), R(3x3))} e a imagem com overlay."""
    out = bgr.copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    res = {}
    if ids is not None:
        for c, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            if mid not in KNOWN:
                continue
            half = SIZE_M[mid] / 2
            objp = np.array([[-half,half,0],[half,half,0],[half,-half,0],[-half,-half,0]], np.float32)
            ok, rvec, tvec = cv2.solvePnP(objp, c[0], K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                R, _ = cv2.Rodrigues(rvec)
                res[mid] = (tvec.flatten(), R)
                col = (0,255,0) if mid in TABLE_IDS else (255,255,0)
                cv2.aruco.drawDetectedMarkers(out, [c], np.array([[mid]]), col)
    return res, out


def compute_pose(res):
    """Dos marcadores da mesa (>=3) -> (cam_pos, R_world_cam, {id:(pos,quat)}, cup_xy)."""
    tab = {m: res[m] for m in TABLE_IDS if m in res}
    if len(tab) < 3:
        return None
    ids = sorted(tab)
    P = {m: tab[m][0] * SCALE for m in ids}
    Pp = np.array([P[m] for m in ids])
    c = Pp.mean(0)
    _, _, Vt = np.linalg.svd(Pp - c)
    n = Vt[-1]
    if np.dot(n, -c) < 0:
        n = -n
    n /= np.linalg.norm(n)
    xr = P[ids[0]] - c; xr -= np.dot(xr, n) * n; xr /= np.linalg.norm(xr)
    yr = np.cross(n, xr)
    R_tc = np.column_stack([xr, yr, n]).T
    B = np.eye(3); center = SIM_TABLE_CENTER
    C = np.diag([1.0, -1.0, -1.0])
    cam_pos = center + B @ (R_tc @ (-c))
    R_world_cam = B @ R_tc @ C
    markers = {}
    for m in ids:
        wp = center + B @ (R_tc @ (P[m] - c))
        Rwm = B @ R_tc @ tab[m][1]
        markers[m] = (wp, flat_quat(Rwm))
    # copo
    cup_xy = None
    cups = [m for m in CUP_IDS if m in res]
    if cups:
        cs = []
        for m in cups:
            Pc = res[m][0] * SCALE
            wp = center + B @ (R_tc @ (Pc - c))
            nrm = B @ (R_tc @ res[m][1][:, 2])
            nh = np.array([nrm[0], nrm[1], 0.0])
            if np.linalg.norm(nh) > 1e-6:
                nh /= np.linalg.norm(nh)
                ctr = wp - CUP_RADIUS_M * nh
                cs.append(ctr[:2])
        if cs:
            cup_xy = np.mean(cs, axis=0)
    return cam_pos, R_world_cam, markers, cup_xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.68.71")
    ap.add_argument("--port", type=int, default=5555)
    args = ap.parse_args()

    intr = json.loads(INTR_CACHE.read_text())
    W, H = intr["width"], intr["height"]
    K = np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1]], float)
    dist = np.array(intr.get("coeffs", [0]*5), float)
    fovy = math.degrees(2 * math.atan((H / 2) / intr["fy"]))

    model = build_model(W, H, fovy)
    data = mujoco.MjData(model)
    data.qpos[3:7] = [1, 0, 0, 0]
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "aligned_cam")
    gid = {m: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"aruco{m}") for m in TABLE_IDS}
    cup_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "junta_livre_bloco")
    cup_qadr = model.jnt_qposadr[cup_jid]
    renderer = mujoco.Renderer(model, height=H, width=W)
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[4] = 1
    detector = make_detector()

    client = SensorClient()
    client.start_client(server_ip=args.host, port=args.port)
    client.socket.setsockopt(zmq.RCVTIMEO, 2000)
    win = "REAL  |  SIM (sync ao vivo)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"[live-sync] {args.host}:{args.port}  fovy={fovy:.2f}  Q/ESC fecha")

    last_n = 0
    best = {"n": -1}            # melhor frame = MAIS marcadores vistos juntos
    out_dir = HERE / "runs"
    out_dir.mkdir(exist_ok=True)
    while True:
        try:
            msg = client.receive_message()
            s = msg.get("images", msg).get("head_camera")
            if s is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break
                continue
            real = ImageUtils.decode_image(s) if isinstance(s, str) else s
        except Exception:
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break
            continue

        res, overlay = solve(real, detector, K, dist)
        pose = compute_pose(res)
        if pose is not None:
            cam_pos, Rwc, markers, cup_xy = pose
            model.cam_pos[cam_id] = cam_pos
            model.cam_quat[cam_id] = mat_to_quat(Rwc)
            for m, (wp, q) in markers.items():
                model.geom_pos[gid[m]] = wp - np.array([0.45, 0, 0.0])  # local da mesa
                model.geom_quat[gid[m]] = q
            if cup_xy is not None:
                data.qpos[cup_qadr:cup_qadr+3] = [cup_xy[0], cup_xy[1], 0.756]
                data.qpos[cup_qadr+3:cup_qadr+7] = [1, 0, 0, 0]
            last_n = len(markers)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="aligned_cam", scene_option=opt)
        sim = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

        # guarda o MELHOR frame = mais marcadores (mesa+copo) detectados juntos.
        # salva NA HORA (não só no Q) num arquivo fixo, pra não depender de fechar certo.
        n_total = len(res)
        if pose is not None and n_total > best["n"]:
            best = {
                "n": n_total,
                "n_table": len(markers),
                "table_ids": sorted(m for m in res if m in TABLE_IDS),
                "cup_ids": sorted(m for m in res if m in CUP_IDS),
                "cam_pos": cam_pos.tolist(),
                "cam_quat": mat_to_quat(Rwc).tolist(),
                "fovy": fovy,
                "markers_world": {int(m): markers[m][0].tolist() for m in markers},
                "cup_xy": (None if cup_xy is None else cup_xy.tolist()),
            }
            cv2.imwrite(str(out_dir / "live_calib_best.png"),
                        np.concatenate([overlay, sim], axis=1))
            (out_dir / "live_calib_best.json").write_text(json.dumps(best, indent=2))
            print(f"[calib] novo melhor: {best['n']} marcadores "
                  f"(mesa={best['table_ids']} copo={best['cup_ids']}) -> runs/live_calib_best.*",
                  flush=True)

        cv2.putText(overlay, f"REAL  mesa={sorted(m for m in res if m in TABLE_IDS)} "
                    f"copo={sorted(m for m in res if m in CUP_IDS)}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(sim, f"SIM alinhado ({last_n} mesa)  | MELHOR ate agora: {best['n']} marcadores",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        combined = np.concatenate([overlay, sim], axis=1)
        cv2.imshow(win, combined)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    client.stop_client()
    cv2.destroyAllWindows()

    if best["n"] > 0:
        print(f"[calib] melhor frame final: {best['n']} marcadores -> runs/live_calib_best.json",
              flush=True)
    else:
        print("[calib] nenhum frame com >=3 marcadores da mesa; nada salvo", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
