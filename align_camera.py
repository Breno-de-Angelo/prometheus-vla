#!/usr/bin/env python3
"""
Alinha a câmera do SIM com a REAL usando a constelação de ArUcos da mesa
(opção data-driven: deriva o layout do próprio real). Gera um sim "aligned"
e um overlay real⊕sim pra VALIDAR antes de mexer na head_camera de verdade.

Entrada: um run de record_sync_data.py (manifest.json com tvec por marcador).
Passos:
  1. pose mediana de cada marcador da mesa no frame da câmera (PnP).
  2. corrige escala absoluta pelo depth (z_depth/z_pnp ~0.86; PnP infla por
     marcador != 60mm). --no-scale pra desligar.
  3. ajusta um plano à constelação -> frame da MESA (origem, eixos, normal).
  4. mapeia a constelação pro tampo da mesa do sim e põe uma câmera 'aligned_cam'
     na MESMA pose relativa; fovy dos intrínsecos reais; render 848x480.
  5. overlay real⊕sim + lado a lado -> /tmp/align_overlay.png; reprojeta os
     marcadores no sim e mede o erro em pixels vs real (validação quantitativa).

Uso:
    python align_camera.py                       # usa o run mais recente
    python align_camera.py --run runs/sync_XXXX  # run específico
    python align_camera.py --real-frame runs/.../rgb/000100.png   # frame p/ overlay
"""
import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import cv2

HERE = Path(__file__).parent
SIM_DIR = HERE / "unitree-g1-mujoco"
SCENE_XML = SIM_DIR / "assets" / "scene_43dof.xml"
TABLE_IDS = [0, 1, 2, 3]
CUP_IDS = [4, 5, 6, 7]
CUP_RADIUS_M = 0.044   # raio do cilindro de colisão do copo (scene_43dof.xml)

# tampo da mesa no sim (world): mesa body em (0.45,0,0), toalha topo ~z=0.754.
SIM_TABLE_CENTER = np.array([0.45, 0.0, 0.756])
SIM_TABLE_NORMAL = np.array([0.0, 0.0, 1.0])
SIM_TABLE_X = np.array([-1.0, 0.0, 0.0])   # "pra frente do robô" aponta -x? ver nota
SIM_TABLE_Y = np.array([0.0, -1.0, 0.0])


def latest_run():
    runs = sorted((HERE / "runs").glob("sync_*"))
    if not runs:
        sys.exit("nenhum run em runs/sync_*; rode record_sync_data.py primeiro")
    return runs[-1]


def median_rvec(manifest, mid):
    """Rotação mediana (matriz 3x3 cam<-marker) do marcador mid sobre os frames."""
    Rs = []
    for f in manifest["frames"]:
        mk = f["markers"].get(str(mid))
        if mk and mk.get("rvec"):
            R, _ = cv2.Rodrigues(np.array(mk["rvec"], float))
            Rs.append(R)
    if not Rs:
        return None
    # média simples + reortonormalização (rvecs pouco dispersos aqui)
    M = np.mean(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def mat_to_quat(R):
    """Matriz rotação -> quaternion MuJoCo (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def fit_plane(P):
    """Ajusta plano a Nx3. Retorna (centroide, normal_unit) com normal p/ a câmera."""
    c = P.mean(axis=0)
    U, S, Vt = np.linalg.svd(P - c)
    n = Vt[-1]
    if np.dot(n, -c) < 0:   # câmera na origem; normal aponta p/ a câmera
        n = -n
    return c, n / np.linalg.norm(n)


def look_rotation_world(cam_pos, target, up_hint=np.array([0, 0, 1.0])):
    """Matriz 3x3 (colunas = eixos x,y,z da câmera em world) olhando p/ target.
    Convenção MuJoCo: câmera olha por -z, y pra cima."""
    z = cam_pos - target            # -view dir (câmera olha -z)
    z = z / np.linalg.norm(z)
    x = np.cross(up_hint, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(np.array([0, 1.0, 0]), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--real-frame", default=None, help="PNG do real p/ overlay (default: meio do run)")
    ap.add_argument("--no-scale", action="store_true", help="não corrigir escala pelo depth")
    ap.add_argument("--out", default="/tmp/align_overlay.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    run = Path(args.run) if args.run else latest_run()
    manifest = json.loads((run / "manifest.json").read_text())
    intr = manifest["intrinsics"]
    W, H = manifest["resolution"]
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float64)
    fovy = math.degrees(2 * math.atan((H / 2) / intr["fy"]))
    print(f"[align] run={run.name}  fovy(real)={fovy:.2f}°")

    agg = manifest["aggregate"]
    P_cam, ratios = {}, []
    for mid in TABLE_IDS:
        a = agg.get(str(mid))
        if a and a.get("tvec_median"):
            P_cam[mid] = np.array(a["tvec_median"], float)
            if a.get("z_pnp_median_m") and a.get("z_depth_median_m"):
                ratios.append(a["z_depth_median_m"] / a["z_pnp_median_m"])
    if len(P_cam) < 3:
        sys.exit(f"preciso de >=3 marcadores da mesa; só tenho {sorted(P_cam)}")

    scale = 1.0 if args.no_scale or not ratios else float(np.median(ratios))
    print(f"[align] marcadores={sorted(P_cam)}  escala(depth/pnp)={scale:.3f}"
          + ("" if args.no_scale else "  (aplicada)"))
    for mid in P_cam:
        P_cam[mid] = P_cam[mid] * scale

    ids = sorted(P_cam)
    P = np.array([P_cam[m] for m in ids])
    c, n = fit_plane(P)
    cam_height = abs(np.dot(-c, n))
    # ângulo entre o eixo óptico (+z cam) e a normal da mesa
    optical = np.array([0, 0, 1.0])
    tilt = math.degrees(math.acos(np.clip(abs(np.dot(optical, n)), -1, 1)))
    print(f"[align] altura câmera->mesa={cam_height:.3f}m  tilt(óptico vs normal mesa)={tilt:.1f}°")

    # frame da MESA no real (coords da câmera OpenCV): origem=c, z=n, x=proj de (P0-c)
    xr = P[0] - c
    xr = xr - np.dot(xr, n) * n
    xr = xr / np.linalg.norm(xr)
    yr = np.cross(n, xr)
    R_ct = np.column_stack([xr, yr, n])           # table->cam (colunas = eixos mesa em cam)
    R_tc = R_ct.T                                 # cam->table

    # base do frame da mesa no WORLD do sim (mesa horizontal, normal +z)
    B = np.eye(3)
    center = SIM_TABLE_CENTER

    # marcadores: posição em world = center + B * (coords no frame da mesa)
    marker_world = {m: center + B @ (R_tc @ (P_cam[m] - c)) for m in ids}
    # câmera (origem do frame da câmera) no frame da mesa, depois world
    cam_world = center + B @ (R_tc @ (np.zeros(3) - c))
    # rotação da câmera: eixos OpenCV(cam)->world = B @ R_tc ; converte p/ convenção
    # MuJoCo (x dir, y cima, -z frente) = OpenCV(x,y,z) com y,z invertidos.
    C = np.diag([1.0, -1.0, -1.0])
    R_world_cam = B @ R_tc @ C
    xax, yax = R_world_cam[:, 0], R_world_cam[:, 1]
    print(f"[align] cam_world={np.round(cam_world,3)}")

    # orientação de cada marcador no world = B · (R_tc · R_cam<-marker)
    import re
    marker_quat = {}
    for m in ids:
        Rcm = median_rvec(manifest, m)
        if Rcm is not None:
            Rwm = B @ R_tc @ Rcm
            marker_quat[m] = mat_to_quat(Rwm)

    # --- escreve scene temporária: reposiciona+orienta aruco0..3 + add aligned_cam ---
    xml = SCENE_XML.read_text()
    for m in ids:
        # mesa body está em world 0.45 0 0, então pos local = world - (0.45,0,0)
        lp = marker_world[m] - np.array([0.45, 0.0, 0.0])
        xml = re.sub(rf'(name="aruco{m}"[^>]*?pos=")[^"]*(")',
                     rf'\g<1>{lp[0]:.4f} {lp[1]:.4f} {lp[2]:.4f}\g<2>', xml)
        if m in marker_quat:                       # injeta/atualiza quat do geom
            q = marker_quat[m]
            qstr = f'quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}"'
            if re.search(rf'name="aruco{m}"[^>]*?quat="', xml):
                xml = re.sub(rf'(name="aruco{m}"[^>]*?quat=")[^"]*(")',
                             rf'\g<1>{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}\g<2>', xml)
            else:
                xml = re.sub(rf'(name="aruco{m}")', rf'\1 {qstr}', xml, count=1)

    # --- posiciona o COPO via marcador(es) do copo (4-7) ---
    cup_centers = []
    for m in CUP_IDS:
        a = agg.get(str(m))
        if not (a and a.get("tvec_median")):
            continue
        Pc = np.array(a["tvec_median"], float) * scale
        wpos = center + B @ (R_tc @ (Pc - c))
        Rcm = median_rvec(manifest, m)
        if Rcm is None:
            continue
        # normal do marcador (z do frame do marcador) -> world; componente horizontal
        nrm = B @ (R_tc @ Rcm[:, 2])
        nh = np.array([nrm[0], nrm[1], 0.0])
        if np.linalg.norm(nh) < 1e-6:
            continue
        nh /= np.linalg.norm(nh)
        # PnP: z aponta p/ fora da face (em direção à câmera) = p/ fora do copo.
        # centro do copo = posição do marcador recuada do raio p/ dentro.
        ctr = wpos - CUP_RADIUS_M * nh
        cup_centers.append([ctr[0], ctr[1]])
        print(f"[cup] id{m}: marcador world={np.round(wpos,3)} -> centro copo xy=({ctr[0]:.3f},{ctr[1]:.3f})")
    if cup_centers:
        cc = np.mean(cup_centers, axis=0)
        z_cup = 0.756   # base do copo na toalha (igual ao default)
        xml = re.sub(r'(name="objeto_customizado" pos=")[^"]*(")',
                     rf'\g<1>{cc[0]:.4f} {cc[1]:.4f} {z_cup:.4f}\g<2>', xml)
        print(f"[cup] copo reposicionado p/ ({cc[0]:.3f},{cc[1]:.3f},{z_cup})")
    else:
        print("[cup] nenhum marcador de copo -> copo fica na pose default")
    cam_line = (f'<camera name="aligned_cam" pos="{cam_world[0]:.4f} {cam_world[1]:.4f} {cam_world[2]:.4f}" '
                f'xyaxes="{xax[0]:.4f} {xax[1]:.4f} {xax[2]:.4f} {yax[0]:.4f} {yax[1]:.4f} {yax[2]:.4f}" '
                f'fovy="{fovy:.3f}"/>')
    xml = xml.replace('<camera name="global_view"', cam_line + '\n    <camera name="global_view"')
    # framebuffer offscreen grande o bastante p/ 848x480
    import re as _re
    if "offwidth" not in xml:
        xml = _re.sub(r'(<global\b)', r'\1 offwidth="848" offheight="480"', xml, count=1)
    # põe mesa/toalha/arucos/copo no GROUP 4 p/ renderizar SÓ a cena (esconde o robô)
    for gname in ["geometria_mesa", "toalha_mesa", "aruco0", "aruco1", "aruco2",
                  "aruco3", "copo_geom_visual"]:
        xml = _re.sub(rf'(name="{gname}")', r'\1 group="4"', xml, count=1)
    # tampo GRANDE só p/ o render de validação: a toalha real preenche o quadro
    # inteiro; a mesa do sim é menor (markers caíam fora e "flutuavam"). Plano
    # plano em z da mesa, group 4, sem colisão. (NÃO altera a scene de verdade.)
    z_tab = SIM_TABLE_CENTER[2] - 0.001
    big = (f'<geom name="align_tabletop" type="box" size="2.0 2.0 0.001" '
           f'pos="{SIM_TABLE_CENTER[0]:.3f} {SIM_TABLE_CENTER[1]:.3f} {z_tab:.4f}" '
           f'material="toalha_mat" group="4" contype="0" conaffinity="0"/>')
    xml = xml.replace('<geom name="floor"', big + '\n    <geom name="floor"')
    # baixa a luz (toalha branca estourava e matava o contraste dos marcadores)
    xml = xml.replace('diffuse="0.6 0.6 0.6"', 'diffuse="0.3 0.3 0.3"')
    xml = xml.replace('<light pos="0 0 1.5" dir="0 0 -1" directional="true"/>',
                      '<light pos="0 0 1.5" dir="0 0 -1" directional="true" diffuse="0.3 0.3 0.3"/>')

    import mujoco
    tmp = Path(tempfile.mkdtemp()) / "scene_aligned.xml"
    # mesmo dir de assets pra resolver meshes/texturas
    tmp = SCENE_XML.parent / "scene_aligned.xml"
    tmp.write_text(xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
        data = mujoco.MjData(model)
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        r = mujoco.Renderer(model, height=H, width=W)
        opt = mujoco.MjvOption()           # renderiza SÓ o group 4 (cena, sem robô)
        opt.geomgroup[:] = 0
        opt.geomgroup[4] = 1
        r.update_scene(data, camera="aligned_cam", scene_option=opt)
        sim_rgb = r.render()
        r.close()
    finally:
        tmp.unlink(missing_ok=True)
    sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

    # frame real p/ overlay
    if args.real_frame:
        real = cv2.imread(args.real_frame)
    else:
        rgbs = sorted((run / "rgb").glob("*.png"))
        real = cv2.imread(str(rgbs[len(rgbs) // 2])) if rgbs else np.zeros_like(sim_bgr)

    # validação: detecta aruco no sim alinhado e compara posição com o real
    def centers(bgr):
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        D = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 35
        p.adaptiveThreshWinSizeStep = 4
        cr, ids, _ = cv2.aruco.ArucoDetector(D, p).detectMarkers(g)
        out = {}
        if ids is not None:
            for cc, i in zip(cr, ids.flatten()):
                out[int(i)] = cc[0].mean(axis=0)
        return out
    cs_sim, cs_real = centers(sim_bgr), centers(real)
    errs = []
    for m in ids:
        if m in cs_sim and m in cs_real:
            errs.append(np.linalg.norm(cs_sim[m] - cs_real[m]))
    print(f"[align] sim detectou ids={sorted(cs_sim)}  erro repro médio="
          f"{np.mean(errs):.1f}px" if errs else "[align] sim detectou ids=" + str(sorted(cs_sim)))

    blend = cv2.addWeighted(real, 0.5, sim_bgr, 0.5, 0)
    side = np.concatenate([real, sim_bgr], axis=1)
    full = np.concatenate([side, np.concatenate([blend, np.zeros_like(blend)], axis=1)], axis=0)
    cv2.imwrite(args.out, full)
    cv2.imwrite(args.out.replace(".png", "_blend.png"), blend)
    print(f"[ok] overlay -> {args.out}  (cima: REAL|SIM ; baixo-esq: blend)")

    if args.show:
        cv2.imshow("align", full)
        while cv2.waitKey(100) & 0xFF not in (ord("q"), 27):
            pass
    os._exit(0)


if __name__ == "__main__":
    main()
