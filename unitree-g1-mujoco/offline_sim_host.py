#!/usr/bin/env python
"""Host OFFLINE pra inferência distribuída SEM robô e SEM render do sim.

Modo "replay aberto / auditar decisões": a VLA (na Atena) enxerga a IMAGEM e o STATE
de uma RUN REAL gravada (um episódio do dataset LeRobot), frame a frame, decide as
ações, e o host só captura as decisões (o RunRecorder da própria inferência grava
train/log/run_<ts>/chunks.jsonl com o que a VLA viu + o que ela mandou). O sim MuJoCo
(visualização das ações) é uma camada OPCIONAL que entra depois — este script é o
"robô fake" que serve a observação do dataset.

Fala EXATAMENTE o protocolo ZMQ que a VLA (UnitreeG1Dex3 em modo socket) espera, igual
ao run_g1_server.py — mas em vez de DDS/MuJoCo, a fonte é o dataset:
  :5555 PUB  imagem head_camera   {"images": {"head_camera": b64_jpeg}}
  :6001 PUB  state do corpo       {"topic":"rt/lowstate",        "data": {motor_state[35],...}}
  :6002 PUB  state das mãos       {"topic":"rt/dex3/{side}/state","data": {side,motor_state[7],...}}
  :6000 PULL comando do corpo  (recebe da VLA; drena/loga)
  :6003 PULL comando das mãos  (recebe da VLA; drena/loga)

Mapeamento dataset(right14 observation.state[14]) -> motores:
  state[0:7]  = braço direito  -> lowstate.motor_state[22:29].q  (G1 29-DoF: dir = 22-28)
  state[7:14] = dedos direitos -> handstate(right).motor_state[0:7].q

Uso (no PC, a partir de unitree-g1-mujoco/):
  python offline_sim_host.py --dataset-root ../datasets/_merged/G1_Dex3_pick_white_cup_right8_1squeeze \\
      --episode 0 --loop --log-cmp /tmp/offline_cmp.jsonl
  # na Atena:  CUDA_VISIBLE_DEVICES=1 python inference_realtime_pi05d_right14.py \\
  #              --robot-ip <IP_DO_PC> --checkpoint ... --fps 30 --rtc --live ... --sim
"""
import argparse
import base64
import glob
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import zmq

# ---- dashboard ao vivo (MJPEG): robô 3ª-pessoa + o que a VLA vê (dataset) + métricas ----
LIVE = {"global": None, "head": None, "stats": "aguardando a VLA...", "lock": threading.Lock()}

_PAGE = ("""<!doctype html><html><head><meta charset=utf-8><title>G1 eval offline (replay aberto)</title>
<style>body{background:#0d0d0f;color:#e8e8ea;font-family:system-ui,sans-serif;margin:0;padding:14px}
h1{font-size:17px;margin:0 0 4px}.sub{color:#888;font-size:13px;margin-bottom:12px}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
.card{background:#17171b;border:1px solid #26262c;border-radius:10px;padding:10px}
h2{font-size:13px;margin:2px 0 8px;color:#7db5ff;font-weight:600}
img{display:block;border-radius:6px;background:#000}
#stats{font-size:14px;line-height:1.7;white-space:pre;background:#000;padding:10px 14px;border-radius:8px;color:#6ee06e;min-width:300px}</style></head>
<body><h1>G1 — eval offline (REPLAY ABERTO)</h1>
<div class=sub>A VLA vê o <b>vídeo do dataset</b> (mundo real) e manda comandos; o robô do simulador só <b>visualiza</b> o que ela faria.</div>
<div class=row>
<div class=card><h2>Robô no simulador — 3ª pessoa</h2><img src="/global.mjpg" width=640 height=480></div>
<div class=card><h2>Status ao vivo</h2><div id=stats>aguardando...</div></div>
</div>
<script>setInterval(async()=>{try{document.getElementById('stats').textContent=await(await fetch('/stats')).text()}catch(e){}},400)</script>
</body></html>""").encode("utf-8")


class _LiveHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
            self.wfile.write(_PAGE); return
        if self.path.startswith("/stats"):
            with LIVE["lock"]:
                b = LIVE["stats"].encode()
            self.send_response(200); self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        key = "global" if "global" in self.path else "head" if "head" in self.path else None
        if key:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache"); self.end_headers()
            try:
                while True:
                    with LIVE["lock"]:
                        buf = LIVE[key]
                    if buf is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
                    time.sleep(1 / 15.0)
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_response(404); self.end_headers()


def start_dashboard(port):
    srv = ThreadingHTTPServer(("0.0.0.0", port), _LiveHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[dashboard] AO VIVO em http://localhost:{port}", flush=True)
    return srv


def _jpg(img_bgr, q=80):
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return buf.tobytes() if ok else None

# Portas e topics — IDÊNTICOS a run_g1_server.py / unitree_sdk2_socket.py
LOWCMD_PORT, LOWSTATE_PORT, HANDSTATE_PORT, HANDCMD_PORT, CAM_PORT = 6000, 6001, 6002, 6003, 5555
kTopicLowState = "rt/lowstate"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"
NUM_MOTORS = 35
NUM_HAND_MOTORS = 7
# G1 29-DoF: braço direito = motores 22..28 (ARM_JOINT_IDS = 15..28; dir = 22-28)
RIGHT_ARM_MOTOR_IDS = tuple(range(22, 29))   # 7 juntas
# nomes das juntas (ordem do dataset right14 observation.state)
ARM_NAMES = ["ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "Elbow", "WristRoll", "WristPitch", "WristYaw"]
FINGER_NAMES = ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]


# ---------------------------------------------------------------- dataset reader
class EpisodeSource:
    """Lê um episódio do dataset LeRobot v3: observation.state[14] do parquet + frames
    do head_camera.mp4 (seek por from_timestamp). Entrega (rgb_uint8_HWC, state14) por frame."""

    def __init__(self, root: str, episode: int):
        self.root = Path(root)
        self.ep = episode
        meta_files = sorted(glob.glob(str(self.root / "meta/episodes/chunk-*/*.parquet")))
        if not meta_files:
            raise FileNotFoundError(f"meta/episodes não encontrado em {root}")
        meta = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
        row = meta[meta["episode_index"] == episode]
        if len(row) == 0:
            raise ValueError(f"episódio {episode} não existe (tem {meta['episode_index'].max()+1})")
        row = row.iloc[0]
        # parquet de dados do episódio
        dci, dfi = int(row["data/chunk_index"]), int(row["data/file_index"])
        dpath = self.root / f"data/chunk-{dci:03d}/file-{dfi:03d}.parquet"
        df = pd.read_parquet(dpath)
        df = df[df["episode_index"] == episode] if "episode_index" in df.columns else df
        self.state = np.vstack(df["observation.state"].values).astype(np.float32)  # [n,14]
        self.n = len(self.state)
        # DEPTH REAL (HF Image: PNG uint16 em mm, no parquet) — guarda os bytes PNG por frame.
        # Os bytes já são o PNG uint16 que o consumidor (ZMQCamera/OmniView) espera: só base64.
        dcol = "observation.images.head_camera_depth"
        if dcol in df.columns:
            self.depth_png = [v.get("bytes") if isinstance(v, dict) else None for v in df[dcol].values]
            ndepth = sum(1 for b in self.depth_png if b)
            print(f"[ep{episode}] depth REAL: {ndepth}/{self.n} frames (PNG uint16 mm)", flush=True)
        else:
            self.depth_png = None
        # vídeo + janela temporal do episódio
        vkey = "videos/observation.images.head_camera"
        vci, vfi = int(row[f"{vkey}/chunk_index"]), int(row[f"{vkey}/file_index"])
        self.vpath = self.root / f"videos/observation.images.head_camera/chunk-{vci:03d}/file-{vfi:03d}.mp4"
        self.t_from = float(row[f"{vkey}/from_timestamp"])
        self.cap = cv2.VideoCapture(str(self.vpath))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._start_frame = int(round(self.t_from * self.fps))
        print(f"[ep{episode}] {self.n} frames | state[14] | vídeo {self.vpath.name} "
              f"@ {self.fps:.0f}fps (start={self._start_frame})", flush=True)

        self.frames = None  # cache dos frames do episódio (preenchido por preload())

    def preload(self):
        """Carrega todos os frames do episódio na RAM (leitura SEQUENCIAL, sem seek por
        frame) — elimina o lag/travamento do cap.set() a cada chamada de rgb()."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)
        frames = []
        for _ in range(self.n):
            ok, f = self.cap.read()
            if not ok:
                break
            frames.append(f)
        self.frames = frames
        print(f"[ep{self.ep}] preload: {len(frames)} frames na RAM", flush=True)
        return len(frames)

    def rgb(self, i: int) -> np.ndarray:
        """Frame i do episódio (BGR uint8 como o cv2.imdecode entrega no robô real)."""
        if self.frames is not None:
            return self.frames[min(i, len(self.frames) - 1)] if self.frames else np.zeros((480, 848, 3), np.uint8)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame + i)
        ok, f = self.cap.read()
        if not ok:
            return np.zeros((480, 848, 3), np.uint8)
        return f  # BGR (o robô real publica jpeg que decodifica em BGR; a VLA faz BGR->RGB)

    def state14(self, i: int) -> np.ndarray:
        return self.state[min(i, self.n - 1)]

    def close(self):
        self.cap.release()


# ---------------------------------------------------------------- ZMQ messages
def _motor(q=0.0):
    return {"q": float(q), "dq": 0.0, "tau_est": 0.0, "temperature": 25.0}


def lowstate_msg(state14: np.ndarray) -> dict:
    """{"topic":"rt/lowstate","data":{motor_state[35], imu_state, wireless_remote, mode_machine}}.
    Só preenche o braço DIREITO (motores 22-28) com state14[0:7]; resto = 0."""
    motors = [_motor() for _ in range(NUM_MOTORS)]
    for k, midx in enumerate(RIGHT_ARM_MOTOR_IDS):
        motors[midx] = _motor(state14[k])
    data = {
        "motor_state": motors,
        "imu_state": {"quaternion": [1.0, 0.0, 0.0, 0.0], "gyroscope": [0.0] * 3,
                      "accelerometer": [0.0] * 3, "rpy": [0.0] * 3, "temperature": 25.0},
        "wireless_remote": base64.b64encode(bytes(40)).decode("ascii"),
        "mode_machine": 0,
    }
    return {"topic": kTopicLowState, "data": data}


def handstate_msg(state14: np.ndarray, side: str) -> dict:
    """Mão DIREITA = state14[7:14]; esquerda = zeros. Inclui press_sensor_state (tátil) zerado."""
    if side == "right":
        qs = state14[7:14]
    else:
        qs = np.zeros(7, np.float32)
    motors = [_motor(qs[i]) for i in range(NUM_HAND_MOTORS)]
    press = [{"pressure": [0.0] * 12, "temperature": [25.0] * 12} for _ in range(NUM_HAND_MOTORS)]
    topic = kTopicDex3RightState if side == "right" else kTopicDex3LeftState
    return {"topic": topic, "data": {"side": side, "motor_state": motors, "press_sensor_state": press}}


_DUMMY_DEPTH_PNG = None


def _dummy_depth_b64(h, w):
    """PNG uint16 zerado — só pra a ZMQCamera de head_camera_depth não bloquear o
    get_observation (o modelo right14 é RGB-only; o conteúdo do depth não importa)."""
    global _DUMMY_DEPTH_PNG
    if _DUMMY_DEPTH_PNG is None:
        ok, buf = cv2.imencode(".png", np.zeros((h, w), np.uint16))
        _DUMMY_DEPTH_PNG = base64.b64encode(buf).decode("ascii") if ok else ""
    return _DUMMY_DEPTH_PNG


def image_msg(bgr: np.ndarray, depth_png: bytes = None) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    b64 = base64.b64encode(buf).decode("ascii") if ok else ""
    h, w = bgr.shape[:2]
    # depth REAL do dataset (PNG uint16 mm) se houver; senão dummy zerado
    depth_b64 = base64.b64encode(depth_png).decode("ascii") if depth_png else _dummy_depth_b64(h, w)
    t = time.time()
    return json.dumps({
        "images": {"head_camera": b64, "head_camera_depth": depth_b64},
        "timestamps": {"head_camera": t, "head_camera_depth": t},
    })


# ---------------------------------------------------------------- MuJoCo viz (P2)
class MujocoViz:
    """Aplica os comandos recebidos da VLA ao G1 do MuJoCo (via DDS rt/lowcmd) e
    grava um vídeo 3ª-pessoa do robô se movendo — a "visualização das decisões".
    O sim NÃO realimenta a VLA (a VLA vê o dataset); aqui é só pra VER o robô."""

    def __init__(self, video_out: str, fps: float = 30.0, width: int = 640, height: int = 480):
        import os as _os
        _os.environ.setdefault("MUJOCO_GL", "egl")
        _os.environ.setdefault("G1_DDS_IFACE", "lo")
        import sys as _sys
        _here = Path(__file__).resolve().parent
        for p in (str(_here), str(_here.parent / "lerobot-ext/robot/unitree_g1")):
            if p not in _sys.path:
                _sys.path.insert(0, p)
        import mujoco
        from env import make_env
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, HandCmd_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__HandCmd_
        from unitree_sdk2py.core.channel import ChannelPublisher
        self._mj = mujoco
        self._new_lowcmd = unitree_hg_msg_dds__LowCmd_
        self._new_handcmd = unitree_hg_msg_dds__HandCmd_

        print("[mujoco] criando env headless (sem publish de câmera)...", flush=True)
        # Desliga onscreen+offscreen do env TEMPORARIAMENTE (senão ele cria um contexto
        # GL que conflita com o nosso Renderer EGL). Restaura o config.yaml depois, pra
        # não afetar o uso live (laptop_sim_host quer o viewer).
        import yaml
        cfg_path = _here / "config.yaml"
        _orig_cfg = cfg_path.read_text()
        _cfg = yaml.safe_load(_orig_cfg)
        _cfg["ENABLE_ONSCREEN"] = False
        _cfg["ENABLE_OFFSCREEN"] = False
        cfg_path.write_text(yaml.safe_dump(_cfg))
        try:
            self.env = make_env(cameras=[], publish_images=False)
        finally:
            cfg_path.write_text(_orig_cfg)  # restaura sempre
        self.de = self.env.sim_env
        self.mjm, self.mjd = self.de.mj_model, self.de.mj_data
        self.n_substep = max(1, int(round((1.0 / fps) / self.env.sim_dt)))  # ~8 steps/frame

        self.lowcmd_pub = ChannelPublisher("rt/lowcmd", hg_LowCmd); self.lowcmd_pub.Init()
        self.lhand_pub = ChannelPublisher("rt/dex3/left/cmd", HandCmd_); self.lhand_pub.Init()
        self.rhand_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_); self.rhand_pub.Init()
        self._cur_low = None
        self._cur_lhand = None
        self._cur_rhand = None
        self._hold = None      # pose inicial a SEGURAR após reset (até a VLA voltar ao frame 0)
        self._hold_n = 0

        # renderer offscreen + câmera 3ª pessoa olhando o torso
        self.renderer = mujoco.Renderer(self.mjm, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.6, 150.0, -20.0
        tid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.cam.lookat[:] = self.mjd.xpos[tid] if tid >= 0 else [0, 0, 1.0]
        self.cam_track_id = tid
        self.writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        self.video_out = video_out
        # ids p/ medir o que o robô do SIM faz (mão alcança o copo? levanta?)
        self.hand_b = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "right_hand_index_1_link")
        self.cup_b = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "objeto_customizado")
        self.cup_z0 = float(self.mjd.xpos[self.cup_b][2]) if self.cup_b >= 0 else 0.0
        self.last_rgb = None  # último frame 3ª-pessoa (RGB) p/ o dashboard
        # câmera de CABEÇA do robô (1ª pessoa no sim) — renderizada da head_camera do modelo
        self.head_cam_id = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        self.last_head_rgb = None
        # estado inicial (robô na pose default + copo no lugar) p/ resetar a cada loop
        self.qpos0 = self.mjd.qpos.copy()
        self.qvel0 = self.mjd.qvel.copy()
        self.arm_adr = [self.jadr(n) for n in ARM_JOINTS]
        self.fin_adr = [self.jadr(n) for n in FINGER_JOINTS]
        print(f"[mujoco] gravando vídeo da viz -> {video_out} ({width}x{height}@{fps:.0f}, {self.n_substep} substeps)", flush=True)

    def reset_to_initial(self, state14):
        """Volta o robô do sim à POSE INICIAL DO VÍDEO (braço+dedos do frame 0) e o copo à
        posição inicial; zera velocidades. Chamado quando o loop do episódio reinicia, pra
        cada repetição começar limpa (sem a deriva/copo-derrubado do loop anterior)."""
        self.mjd.qpos[:] = self.qpos0
        self.mjd.qvel[:] = self.qvel0
        for a, q in zip(self.arm_adr, state14[:7]):
            if a is not None:
                self.mjd.qpos[a] = float(q)
        for a, q in zip(self.fin_adr, state14[7:14]):
            if a is not None:
                self.mjd.qpos[a] = float(q)
        self._mj.mj_forward(self.mjm, self.mjd)
        # SEGURA o robô travado nessa pose inicial até a VLA "voltar ao frame 0" (drenar o
        # buffer RTC ~1.5s) — senão o comando do FIM do reach (ainda na fila) puxa o robô
        # pra a pose errada logo após o snap.
        self._hold = np.asarray(state14, dtype=float)
        self._hold_n = 0

    def jadr(self, name):
        jid = self._mj.mj_name2id(self.mjm, self._mj.mjtObj.mjOBJ_JOINT, name)
        return self.mjm.jnt_qposadr[jid] if jid >= 0 else None

    def render_kinematic(self, arm_adr, fin_adr, arm7, fin7):
        """Playback CINEMÁTICO: seta as juntas do braço+dedos direto e renderiza (sem
        dinâmica) — mostra EXATAMENTE a trajetória de junta que a VLA decidiu, sem risco
        de o robô cair/derrubar. O copo fica parado; dist mão-copo é geométrica."""
        for a, q in zip(arm_adr, arm7):
            if a is not None:
                self.mjd.qpos[a] = q
        for a, q in zip(fin_adr, fin7):
            if a is not None:
                self.mjd.qpos[a] = q
        self._mj.mj_forward(self.mjm, self.mjd)
        if self.cam_track_id >= 0:
            self.cam.lookat[:] = self.mjd.xpos[self.cam_track_id]
        self.renderer.update_scene(self.mjd, self.cam)
        rgb = self.renderer.render()
        self.last_rgb = rgb
        self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def metric(self):
        """(dist mão-copo cm, lift do copo cm) do robô do SIM seguindo os comandos da VLA."""
        if self.hand_b < 0 or self.cup_b < 0:
            return None, None
        import numpy as _np
        d = float(_np.linalg.norm(self.mjd.xpos[self.hand_b] - self.mjd.xpos[self.cup_b])) * 100
        l = (float(self.mjd.xpos[self.cup_b][2]) - self.cup_z0) * 100
        return d, l

    def update_body_cmd(self, body_dict: dict):
        if not body_dict:
            return
        data = body_dict.get("data", body_dict)
        cmd = self._new_lowcmd()
        cmd.mode_pr = data.get("mode_pr", 0); cmd.mode_machine = data.get("mode_machine", 0)
        for i, mc in enumerate(data.get("motor_cmd", [])):
            if i >= 35:
                break
            cmd.motor_cmd[i].mode = mc.get("mode", 0); cmd.motor_cmd[i].q = mc.get("q", 0.0)
            cmd.motor_cmd[i].dq = mc.get("dq", 0.0); cmd.motor_cmd[i].kp = mc.get("kp", 0.0)
            cmd.motor_cmd[i].kd = mc.get("kd", 0.0); cmd.motor_cmd[i].tau = mc.get("tau", 0.0)
        self._cur_low = cmd

    def update_hand_cmd(self, hand_dict: dict):
        if not hand_dict:
            return
        topic = hand_dict.get("topic", "")
        data = hand_dict.get("data", hand_dict)
        cmd = self._new_handcmd()
        for i, mc in enumerate(data.get("motor_cmd", [])):
            if i >= 7:
                break
            cmd.motor_cmd[i].mode = mc.get("mode", 0); cmd.motor_cmd[i].q = mc.get("q", 0.0)
            cmd.motor_cmd[i].dq = mc.get("dq", 0.0); cmd.motor_cmd[i].kp = mc.get("kp", 0.0)
            cmd.motor_cmd[i].kd = mc.get("kd", 0.0); cmd.motor_cmd[i].tau = mc.get("tau", 0.0)
        if "right" in topic:
            self._cur_rhand = cmd
        else:
            self._cur_lhand = cmd

    def _render_frame(self):
        if self.cam_track_id >= 0:
            self.cam.lookat[:] = self.mjd.xpos[self.cam_track_id]
        self.renderer.update_scene(self.mjd, self.cam)
        rgb = self.renderer.render()  # RGB uint8
        self.last_rgb = rgb
        self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def render_head(self):
        """Renderiza a head_camera do robô do sim (1ª pessoa). Chamada só quando há
        dashboard (args.port) — é um 2º render por frame (custo ~igual ao 3ª-pessoa)."""
        if self.head_cam_id < 0:
            return
        self.renderer.update_scene(self.mjd, camera=self.head_cam_id)
        self.last_head_rgb = self.renderer.render()

    def step_and_render(self):
        # Após um reset, SEGURA a pose inicial (cinemática) até a VLA "voltar ao frame 0".
        # Libera quando o comando dela p/ o braço se aproxima da pose inicial (drenou o
        # buffer RTC) ou após ~3s de timeout — só então a dinâmica/reach reassume.
        if self._hold is not None:
            for a, q in zip(self.arm_adr, self._hold[:7]):
                if a is not None:
                    self.mjd.qpos[a] = float(q)
            for a, q in zip(self.fin_adr, self._hold[7:14]):
                if a is not None:
                    self.mjd.qpos[a] = float(q)
            self.mjd.qvel[:] = 0.0
            self._mj.mj_forward(self.mjm, self.mjd)
            self._hold_n += 1
            released = False
            if self._cur_low is not None:
                vla_arm = np.array([self._cur_low.motor_cmd[m].q for m in range(22, 29)])
                released = float(np.linalg.norm(vla_arm - self._hold[:7])) < 0.25
            if released or self._hold_n > 90:
                self._hold = None
            self._render_frame()
            return
        for _ in range(self.n_substep):
            if self._cur_low is not None:
                self.lowcmd_pub.Write(self._cur_low)
            if self._cur_rhand is not None:
                self.rhand_pub.Write(self._cur_rhand)
            if self._cur_lhand is not None:
                self.lhand_pub.Write(self._cur_lhand)
            self.de.sim_step()   # só física (render é feito por nós, abaixo)
        self._render_frame()

    def close(self):
        try:
            self.writer.release()
            self.renderer.close()
            self.env.close()
        except Exception:
            pass
        print(f"[mujoco] vídeo salvo -> {self.video_out}", flush=True)


# ---------------------------------------------------------------- replay de decisões (SEM GPU)
ARM_JOINTS = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
              "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
FINGER_JOINTS = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                 "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint",
                 "right_hand_middle_1_joint"]


def replay_decisions(args):
    """Reproduz no robô do sim as ações JÁ DECIDIDAS pela VLA (chunks.jsonl), sem GPU/VLA.
    Playback cinemático + dashboard ao vivo: robô 3ª-pessoa + frame do dataset (casado por
    estado) + métrica dist mão-copo. É o que a VLA mandou o braço fazer na inferência offline."""
    chunks = [json.loads(l) for l in open(args.replay_decisions) if l.strip()]
    traj = np.array([c["actions"][0] for c in chunks], np.float32)        # [N,14] 1ª ação/inferência
    states = np.array([c.get("state_raw", c["state"]) for c in chunks], np.float32)  # [N,14]
    print(f"[replay] {len(traj)} decisões da VLA carregadas de {Path(args.replay_decisions).name}", flush=True)

    viz = MujocoViz(args.video, fps=args.fps)
    if args.port:
        start_dashboard(args.port)
    arm_adr = [viz.jadr(n) for n in ARM_JOINTS]
    fin_adr = [viz.jadr(n) for n in FINGER_JOINTS]

    # fonte do dataset, em ORDEM NATURAL (o episódio toca 0->fim contínuo, 1 vídeo só).
    src = EpisodeSource(args.dataset_root, args.episode)
    src.preload()                              # mata o lag do seek por frame
    ds_states = src.state  # [n,14]
    # para CADA frame do episódio (em ordem), a decisão da VLA mais compatível com o
    # estado daquele frame — assim o vídeo do dataset avança natural e o robô segue.
    kstar = [int(np.argmin(np.linalg.norm(states - ds_states[m], axis=1))) for m in range(src.n)]
    N = len(traj)
    print(f"[replay] AO VIVO em http://localhost:{args.port} — episódio em ordem; o robô segue a decisão da VLA", flush=True)
    try:
        loops = 0
        a_ema = traj[kstar[0]].copy()
        while True:
            for m in range(src.n):
                tgt = traj[kstar[m]]
                a_ema = 0.6 * a_ema + 0.4 * tgt   # alisa o robô (sem saltos entre decisões)
                ds_bgr = src.rgb(m)
                viz.render_kinematic(arm_adr, fin_adr, a_ema[:7], a_ema[7:14])
                if args.port:
                    d, l = viz.metric()
                    gjpg = _jpg(cv2.cvtColor(viz.last_rgb, cv2.COLOR_RGB2BGR))
                    hjpg = _jpg(ds_bgr)
                    closed = a_ema[10] > 0.7  # index_0 fechado?
                    stats = (f"REPLAY das decisões da VLA (sem GPU)\n"
                             f"run:      {Path(args.replay_decisions).stem}\n"
                             f"episódio: {args.episode}   frame {m+1}/{src.n}   (loop {loops})\n"
                             f"decisão casada: {kstar[m]+1}/{N}\n\n"
                             f"braço do sim:\n"
                             f"  dist mão→copo: {d:6.1f} cm\n"
                             f"  lift do copo:  {l:6.1f} cm\n"
                             f"  mão: {'FECHANDO' if closed else 'aberta'}")
                    with LIVE["lock"]:
                        if gjpg: LIVE["global"] = gjpg
                        if hjpg: LIVE["head"] = hjpg
                        LIVE["stats"] = stats
                if m % 20 == 0:
                    d, l = viz.metric()
                    print(f"\r[replay] loop {loops} frame {m+1}/{src.n}  dist={d:.1f}cm  ", end="", flush=True)
                time.sleep(1.0 / args.fps)
            loops += 1
            if not args.loop:
                print("\n[replay] terminou (use --loop pra repetir).", flush=True)
                break
    except KeyboardInterrupt:
        print("\n[replay] parado.", flush=True)
    finally:
        viz.close()
        src.close()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Host offline: serve um episódio do dataset pra VLA remota")
    ap.add_argument("--dataset-root", required=True, help="raiz do dataset LeRobot v3")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--loop", action="store_true", help="repete o episódio em loop")
    ap.add_argument("--bind", default="0.0.0.0", help="interface de bind (0.0.0.0 = acessível pela rede)")
    ap.add_argument("--log-cmp", default="", help="jsonl: por frame, state do dataset + AÇÃO real do dataset "
                    "+ comando recebido da VLA (pra comparar as decisões offline)")
    ap.add_argument("--mujoco", action="store_true", help="liga a visualização: aplica os cmd da VLA a um G1 do "
                    "MuJoCo (via DDS) e grava um vídeo 3ª-pessoa do robô se movendo")
    ap.add_argument("--video", default="/tmp/offline_viz.mp4", help="saída do vídeo da viz MuJoCo")
    ap.add_argument("--trace", default="", help="jsonl: por frame, o que a VLA COMANDA (braço+dedos) vs o que o "
                    "robô do sim ATINGE + dist/lift + flag de hold — pra auditar tracking e grasp")
    ap.add_argument("--port", type=int, default=8013, help="porta do dashboard web ao vivo (0=desliga)")
    ap.add_argument("--replay-decisions", default="", help="caminho de um chunks.jsonl: reproduz as ações JÁ "
                    "decididas pela VLA no robô do sim (sem GPU/VLA) com dashboard ao vivo")
    ap.add_argument("--replay-interp", type=int, default=6, help="sub-frames interpolados entre decisões (suaviza)")
    args = ap.parse_args()

    if args.replay_decisions:
        replay_decisions(args)
        return

    viz = MujocoViz(args.video, fps=args.fps) if args.mujoco else None
    if args.port:
        start_dashboard(args.port)

    src = EpisodeSource(args.dataset_root, args.episode)
    # ação REAL do dataset (pra comparar) — pode não existir em todos
    real_action = None
    try:
        dpath = src.vpath  # não; recarrega o parquet de ação
    except Exception:
        pass

    ctx = zmq.Context()
    cam = ctx.socket(zmq.PUB); cam.bind(f"tcp://{args.bind}:{CAM_PORT}")
    lowstate = ctx.socket(zmq.PUB); lowstate.bind(f"tcp://{args.bind}:{LOWSTATE_PORT}")
    handstate = ctx.socket(zmq.PUB); handstate.bind(f"tcp://{args.bind}:{HANDSTATE_PORT}")
    lowcmd = ctx.socket(zmq.PULL); lowcmd.bind(f"tcp://{args.bind}:{LOWCMD_PORT}"); lowcmd.setsockopt(zmq.RCVTIMEO, 0)
    handcmd = ctx.socket(zmq.PULL); handcmd.bind(f"tcp://{args.bind}:{HANDCMD_PORT}"); handcmd.setsockopt(zmq.RCVTIMEO, 0)
    print(f"[offline-host] bind {args.bind}: cam :{CAM_PORT} lowstate :{LOWSTATE_PORT} "
          f"handstate :{HANDSTATE_PORT} | lowcmd :{LOWCMD_PORT} handcmd :{HANDCMD_PORT}", flush=True)
    print("[offline-host] esperando a VLA conectar (rode a inferência na Atena com --robot-ip <este_ip>)...", flush=True)
    time.sleep(1.0)  # deixa os SUB/PUSH da VLA conectarem antes de publicar

    cmp_f = open(args.log_cmp, "w") if args.log_cmp else None
    trace_f = open(args.trace, "w") if args.trace else None
    period = 1.0 / args.fps
    i = 0
    n_cmd = 0
    last_body_cmd = None
    if viz is not None:
        viz.reset_to_initial(src.state14(0))  # começa na pose inicial do vídeo (não na default do MuJoCo)
    try:
        while True:
            t0 = time.perf_counter()
            st = src.state14(i)
            ds_bgr = src.rgb(i)            # frame do dataset que a VLA realmente vê
            ds_depth = src.depth_png[i] if (src.depth_png and i < len(src.depth_png)) else None
            cam.send_string(image_msg(ds_bgr, ds_depth))
            lowstate.send_string(json.dumps(lowstate_msg(st)))
            handstate.send_string(json.dumps(handstate_msg(st, "right")))
            handstate.send_string(json.dumps(handstate_msg(st, "left")))

            # drena comandos da VLA (não bloqueia) — guarda o mais recente de cada
            while True:
                try:
                    last_body_cmd = json.loads(lowcmd.recv_string(zmq.NOBLOCK)); n_cmd += 1
                except zmq.Again:
                    break
            last_hand_cmd = None
            while True:
                try:
                    last_hand_cmd = json.loads(handcmd.recv_string(zmq.NOBLOCK))
                except zmq.Again:
                    break

            # MuJoCo: aplica os cmd ao robô da sim e renderiza o frame do vídeo
            if viz is not None:
                if last_body_cmd is not None:
                    viz.update_body_cmd(last_body_cmd)
                if last_hand_cmd is not None:
                    viz.update_hand_cmd(last_hand_cmd)
                viz.step_and_render()

            if cmp_f is not None and last_body_cmd is not None:
                # extrai o q comandado do braço direito (motores 22-28) do lowcmd recebido
                mc = last_body_cmd.get("data", {}).get("motor_cmd", [])
                vla_arm = [mc[mid].get("q") if mid < len(mc) else None for mid in RIGHT_ARM_MOTOR_IDS]
                cmp_f.write(json.dumps({"frame": i, "state_arm": st[:7].round(4).tolist(),
                                        "vla_arm_cmd": [round(v, 4) if v is not None else None for v in vla_arm]}) + "\n")

            # TRACE: o que a VLA COMANDA vs o que o robô do sim ATINGE (auditar tracking+grasp)
            if trace_f is not None and viz is not None:
                bmc = (last_body_cmd or {}).get("data", {}).get("motor_cmd", []) if last_body_cmd else []
                hmc = (last_hand_cmd or {}).get("data", {}).get("motor_cmd", []) if last_hand_cmd else []
                vla_arm = [round(bmc[m].get("q"), 4) if m < len(bmc) else None for m in RIGHT_ARM_MOTOR_IDS]
                vla_fin = [round(hmc[m].get("q"), 4) if m < len(hmc) else None for m in range(7)]
                sim_arm = [round(float(viz.mjd.qpos[a]), 4) if a is not None else None for a in viz.arm_adr]
                sim_fin = [round(float(viz.mjd.qpos[a]), 4) if a is not None else None for a in viz.fin_adr]
                d, l = viz.metric()
                trace_f.write(json.dumps({"h": n_cmd, "f": i, "hold": viz._hold is not None,
                                          "va": vla_arm, "vf": vla_fin, "sa": sim_arm, "sf": sim_fin,
                                          "dist": round(d, 2) if d is not None else None,
                                          "lift": round(l, 2) if l is not None else None}) + "\n")

            # atualiza o dashboard ao vivo
            if args.port:
                gjpg = _jpg(cv2.cvtColor(viz.last_rgb, cv2.COLOR_RGB2BGR)) if (viz and viz.last_rgb is not None) else None
                # 1ª pessoa: head_camera RENDERIZADA do robô do sim (≠ RGB do dataset);
                # fallback pro frame do dataset se não houver viz.
                if viz is not None:
                    viz.render_head()
                hjpg = (_jpg(cv2.cvtColor(viz.last_head_rgb, cv2.COLOR_RGB2BGR))
                        if (viz and viz.last_head_rgb is not None) else _jpg(ds_bgr))
                conn = "CONTROLANDO" if n_cmd > 0 else "aguardando conexão"
                stats = (f"episódio:  {args.episode}\n"
                         f"frame:     {i}/{src.n}\n"
                         f"VLA:       {conn}   (cmds={n_cmd})\n")
                if viz is not None:
                    d, l = viz.metric()
                    if d is not None:
                        stats += (f"\nrobô do SIM (segue os cmd da VLA):\n"
                                  f"  dist mão→copo: {d:6.1f} cm\n"
                                  f"  lift do copo:  {l:6.1f} cm\n"
                                  f"  (no replay aberto o sim diverge do real\n"
                                  f"   depois de uns frames — é só visualização)")
                with LIVE["lock"]:
                    if gjpg is not None:
                        LIVE["global"] = gjpg
                    if hjpg is not None:
                        LIVE["head"] = hjpg
                    LIVE["stats"] = stats

            if i % 30 == 0:
                print(f"\r[offline-host] frame {i}/{src.n}  cmds recebidos={n_cmd}   ", end="", flush=True)

            i += 1
            if i >= src.n:
                if args.loop:
                    i = 0
                    if viz is not None:
                        viz.reset_to_initial(src.state14(0))  # robô do sim volta à pose inicial do vídeo
                else:
                    print("\n[offline-host] episódio terminou.", flush=True)
                    break
            dt = period - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[offline-host] parando.", flush=True)
    finally:
        if cmp_f:
            cmp_f.close()
        if trace_f:
            trace_f.close()
        if viz is not None:
            viz.close()
        src.close()
        for s in (cam, lowstate, handstate, lowcmd, handcmd):
            s.close()
        ctx.term()


if __name__ == "__main__":
    main()
