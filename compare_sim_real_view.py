#!/usr/bin/env python3
"""
Captura o frame RGB do robô (ZMQ :5555) e a render da head_camera do sim,
monta lado a lado pra comparar o enquadramento sim <-> real.

No frame REAL, detecta os ArUcos DICT_4X4_50:
  - mesa = ids 0-3 (60 mm)   -> desenhados em VERDE
  - copo = ids 4-7 (25 mm)   -> desenhados em CIANO
Se intrínsecos forem informados (--fx/--fy/--cx/--cy), também desenha os
eixos de pose de cada marcador (base p/ o alinhamento sim<->real).

Uso típico (laptop conectando ao robô que publica em :5555):
    conda activate g1
    python compare_sim_real_view.py --host <IP_DO_ROBO> --show

Sem --show, salva só o PNG (útil em SSH sem display):
    python compare_sim_real_view.py --host <IP_DO_ROBO> --out /tmp/sim_real.png
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# EGL tem que ser escolhido ANTES de importar mujoco (render offscreen sem display).
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import cv2

HERE = Path(__file__).parent
SIM_DIR = HERE / "unitree-g1-mujoco"
sys.path.insert(0, str(SIM_DIR))
from sim.sensor_utils import SensorClient, ImageUtils  # noqa: E402

# scene que o sim realmente usa (config.yaml: ROBOT_SCENE = assets/scene_43dof.xml)
SCENE_XML = SIM_DIR / "assets" / "scene_43dof.xml"

# ids -> grupo (bate com assets/aruco/generate_aruco.py)
TABLE_IDS = {0, 1, 2, 3}
CUP_IDS = {4, 5, 6, 7}
TABLE_SIZE_M = 0.060
CUP_SIZE_M = 0.025
KNOWN_IDS = TABLE_IDS | CUP_IDS  # ids válidos do nosso setup; resto = falso positivo

INTR_CACHE = HERE / "assets" / "realsense_color_intrinsics.json"

# snippet que roda NO ROBÔ (env g1 com pyrealsense2) pra ler os intrínsecos de
# fábrica do perfil color, sem abrir um segundo stream (não conflita com o server).
_INTR_PROBE = r"""
import pyrealsense2 as rs, json
ctx = rs.context(); out=None
for dev in ctx.query_devices():
    for sensor in dev.query_sensors():
        for prof in sensor.get_stream_profiles():
            if prof.stream_type()!=rs.stream.color: continue
            v=prof.as_video_stream_profile()
            if v.width()=={W} and v.height()=={H} and prof.fps()==30:
                i=v.get_intrinsics()
                out={{"fx":i.fx,"fy":i.fy,"cx":i.ppx,"cy":i.ppy,"width":i.width,
                      "height":i.height,"model":str(i.model),"coeffs":list(i.coeffs),
                      "serial":dev.get_info(rs.camera_info.serial_number),
                      "device":dev.get_info(rs.camera_info.name),"stream":"color"}}
print(json.dumps(out))
"""


def get_real_intrinsics(w, h, ssh_host, remote_py):
    """Intrínsecos REAIS da D435 (não chute). Ordem: cache local -> SSH -> None.

    Cacheia em assets/realsense_color_intrinsics.json pra próximas execuções
    funcionarem offline (e mais rápido).
    """
    # 1) cache, se bater a resolução
    if INTR_CACHE.exists():
        try:
            c = json.loads(INTR_CACHE.read_text())
            if c.get("width") == w and c.get("height") == h:
                print(f"[intr] cache {INTR_CACHE.name}: fx={c['fx']:.1f} fy={c['fy']:.1f} "
                      f"cx={c['cx']:.1f} cy={c['cy']:.1f}")
                return c
        except Exception:
            pass
    # 2) consulta via SSH ao robô (env com pyrealsense2)
    try:
        probe = _INTR_PROBE.format(W=w, H=h)
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=8", ssh_host, f"{remote_py} -"],
            input=probe, capture_output=True, text=True, timeout=30,
        )
        line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "null"
        info = json.loads(line)
        if info:
            INTR_CACHE.parent.mkdir(parents=True, exist_ok=True)
            INTR_CACHE.write_text(json.dumps(info, indent=2))
            print(f"[intr] SSH {ssh_host}: fx={info['fx']:.1f} fy={info['fy']:.1f} "
                  f"cx={info['cx']:.1f} cy={info['cy']:.1f} (cacheado)")
            return info
        print(f"[intr] SSH não achou perfil color {w}x{h}: {r.stderr.strip()[:200]}")
    except Exception as e:
        print(f"[intr] SSH falhou ({e}); seguindo sem intrínsecos reais")
    return None


def grab_real_frame(host, port, timeout_s):
    """Conecta ao SensorServer e devolve o primeiro frame head_camera (BGR)."""
    client = SensorClient()
    client.start_client(server_ip=host, port=port)
    # CONFLATE entrega só a última msg; dá um tempo pro PUB/SUB casar.
    client.socket.setsockopt(__import__("zmq").RCVTIMEO, int(timeout_s * 1000))
    print(f"[real] conectando em tcp://{host}:{port} (timeout {timeout_s:.0f}s)...")
    t0 = time.time()
    rgb = None
    while time.time() - t0 < timeout_s:
        try:
            data = client.receive_message()
        except Exception:
            continue
        images = data.get("images", data)
        rgb = images.get("head_camera")
        if rgb is None:
            continue
        if isinstance(rgb, str):
            rgb = ImageUtils.decode_image(rgb)  # devolve BGR (cv2.imdecode)
        break
    client.stop_client()
    if rgb is None:
        raise TimeoutError(
            f"sem frame 'head_camera' em {timeout_s:.0f}s. O robô está publicando em "
            f"tcp://{host}:{port}? (cheque realsense_server.py e o IP/tunel)"
        )
    print(f"[real] frame recebido: {rgb.shape}")
    return rgb


def detect_aruco(bgr, K=None, dist=None):
    """Desenha marcadores detectados (e eixos de pose se K dado). Retorna cópia + infos."""
    out = bgr.copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # refino subpixel dos cantos (pose mais estável) + janela adaptativa mais
    # ampla pra pegar marcador oblíquo/pequeno. Mantemos o perímetro no default
    # pra NÃO criar falso positivo (id fora do set 0-7 é descartado abaixo).
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 35
    params.adaptiveThreshWinSizeStep = 4
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)

    found = []
    if ids is None:
        cv2.putText(out, "ArUco: nenhum marcador", (8, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out, found

    ids = ids.flatten()
    for c, mid in zip(corners, ids):
        mid = int(mid)
        if mid not in KNOWN_IDS:
            continue  # ruído / marcador de outro setup -> ignora
        group = "mesa" if mid in TABLE_IDS else "copo"
        color = (0, 255, 0) if mid in TABLE_IDS else (255, 255, 0)
        cv2.aruco.drawDetectedMarkers(out, [c], np.array([[mid]]), color)
        found.append((mid, group))

        if K is not None:
            size = TABLE_SIZE_M if mid in TABLE_IDS else CUP_SIZE_M
            half = size / 2.0
            objp = np.array([[-half, half, 0], [half, half, 0],
                             [half, -half, 0], [-half, -half, 0]], np.float32)
            ok, rvec, tvec = cv2.solvePnP(objp, c[0], K, dist,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                cv2.drawFrameAxes(out, K, dist, rvec, tvec, size * 0.7, 2)
                z = float(tvec[2][0])
                ctr = c[0].mean(axis=0).astype(int)
                cv2.putText(out, f"{z:.2f}m", (ctr[0] - 20, ctr[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    tags = ", ".join(f"{m}({g})" for m, g in sorted(found))
    cv2.putText(out, f"ArUco: {tags}", (8, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out, found


def render_sim_head(width=640, height=480):
    """Renderiza a head_camera do sim na pose default (RGB).

    Resolução nativa do offscreen é 640x480 (env.py / framebuffer do XML);
    pedir mais que isso estoura o framebuffer. Mantemos 640x480 e o caller
    redimensiona pra casar a altura do frame real.
    """
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    # quaternion da base válido (MjData zera o qpos)
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    if cid < 0:
        raise RuntimeError("camera 'head_camera' não encontrada no scene")
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera="head_camera")
    rgb = renderer.render()
    renderer.close()
    return rgb  # RGB


def label(img, text):
    """Faixa de título ACIMA da imagem (não cobre o conteúdo/ArUcos do topo)."""
    band = np.zeros((30, img.shape[1], 3), dtype=img.dtype)
    cv2.putText(band, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return np.concatenate([band, img], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="IP do robô que publica em :5555")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--out", default="/tmp/sim_real_view.png")
    ap.add_argument("--show", action="store_true", help="abre janela cv2 (precisa de display)")
    # intrínsecos: buscados AUTOMATICAMENTE (cache local -> SSH ao robô).
    ap.add_argument("--ssh-host", default="g1robot",
                    help="alias/host SSH do robô p/ ler os intrínsecos da D435")
    ap.add_argument("--remote-py", default="~/miniconda3/envs/g1/bin/python",
                    help="python no robô com pyrealsense2")
    ap.add_argument("--no-pose", action="store_true", help="não estimar pose (só detectar)")
    ap.add_argument("--live", action="store_true",
                    help="loop contínuo (janela atualiza em tempo real; Q ou ESC fecha)")
    args = ap.parse_args()

    # render do sim UMA vez (pose default; estático) e prepara o painel
    print("[sim] renderizando head_camera...")
    sim_bgr0 = cv2.cvtColor(render_sim_head(640, 480), cv2.COLOR_RGB2BGR)

    # intrínsecos (cache -> SSH); precisa de w/h, então pega 1 frame primeiro
    first = grab_real_frame(args.host, args.port, args.timeout)
    h, w = first.shape[:2]
    K = None
    dist = np.zeros(5)
    if not args.no_pose:
        intr = get_real_intrinsics(w, h, args.ssh_host, args.remote_py)
        if intr is not None:
            K = np.array([[intr["fx"], 0, intr["cx"]],
                          [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float64)
            if any(intr.get("coeffs", [])):
                dist = np.array(intr["coeffs"], np.float64)
        else:
            print("[intr] sem intrínsecos reais -> sem estimativa de pose")

    # casa a altura do sim com a do real e rotula
    if sim_bgr0.shape[0] != h:
        s = h / sim_bgr0.shape[0]
        sim_bgr0 = cv2.resize(sim_bgr0, (int(sim_bgr0.shape[1] * s), h))
    sim_panel = label(sim_bgr0.copy(), "SIM head_camera + ArUcos (pose default)")

    def compose(real_bgr):
        rv, found = detect_aruco(real_bgr, K, dist)
        rv = label(rv, f"REAL (robo)  {w}x{h}")
        return np.concatenate([rv, sim_panel], axis=1), found

    if not args.live:
        combined, found = compose(first)
        cv2.imwrite(args.out, combined)
        print(f"[ok] salvo em {args.out}  | ArUcos: {found if found else 'nenhum'}")
        if args.show:
            cv2.imshow("REAL  |  SIM", combined)
            print("pressione Q pra fechar")
            while cv2.waitKey(100) & 0xFF not in (ord("q"), 27):
                pass
            cv2.destroyAllWindows()
        os._exit(0)

    # ---- modo AO VIVO ----
    import zmq
    client = SensorClient()
    client.start_client(server_ip=args.host, port=args.port)
    client.socket.setsockopt(zmq.RCVTIMEO, 2000)
    win = "REAL (ArUco)  |  SIM"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[live] janela aberta — Q ou ESC pra sair")
    combined, found = compose(first)
    last = time.time()
    while True:
        try:
            data = client.receive_message()
            img = data.get("images", data).get("head_camera")
            if img is not None:
                real_bgr = ImageUtils.decode_image(img) if isinstance(img, str) else img
                combined, found = compose(real_bgr)
                last = time.time()
        except Exception:
            pass
        # HUD de fps/idade do frame
        age = time.time() - last
        hud = combined.copy()
        # topo (área vazia) pra não cobrir ArUcos; +44 fica logo abaixo da faixa de título
        cv2.putText(hud, f"ArUco: {[f'{m}({g})' for m,g in found] or 'nenhum'}  age={age:.1f}s",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, hud)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    client.stop_client()
    cv2.destroyAllWindows()
    os._exit(0)  # evita o EGLError barulhento no teardown


if __name__ == "__main__":
    main()
