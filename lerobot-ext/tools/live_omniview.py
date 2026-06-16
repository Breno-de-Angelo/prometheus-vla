#!/usr/bin/env python3
"""Dashboard web AO VIVO estilo OmniView para a teleoperação/gravação do G1.

Espelha a TELA DE DETALHE do OmniView (RGB · head_camera, DEPTH · turbo,
TRAJETÓRIA 3D · end-effector dir., RAMPAS por junta e TÁTIL · Dex3-1) — só que
em tempo real, lendo os dados enquanto você teleopera, em vez de um episódio já
gravado.

Fontes (dois PUBs ZMQ que o pipeline de gravação já expõe):
  • imagens   : tcp://<cam-host>:5555  (RGB jpg + Depth png uint16) — mesmo do view_cam_live
  • telemetria: tcp://127.0.0.1:5557    (state/action/tátil) — publicado por live_bridge.py

Arquitetura: duas threads ZMQ (bloqueantes) atualizam o último frame de cada
fonte; um servidor aiohttp serve o webapp estático e um broadcast websocket
empurra (a) cada novo frame de imagem e (b) cada nova amostra de telemetria pros
browsers conectados. O navegador monta as rampas/trajetória/tátil em JS.

Uso:
    python tools/live_omniview.py --host 127.0.0.1            # sim
    python tools/live_omniview.py --host 192.168.123.164      # robô real
    python tools/live_omniview.py --http-port 8013 --no-browser
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import socket
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import zmq
from aiohttp import WSMsgType, web

logging.basicConfig(level=logging.INFO, format="[live_omniview] %(message)s")
logger = logging.getLogger("live_omniview")

WEBAPP_DIR = Path(__file__).resolve().parent / "live_webapp"

# Último frame de cada fonte + contadores de sequência (pra mandar só o que mudou).
LATEST = {"img": None, "img_seq": 0, "tele": None, "tele_seq": 0, "quest": None}
_LOCK = threading.Lock()


# ----------------------------------------------------------------------------- imagens
def _jpg_dataurl(bgr: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def _colorize_depth(depth: np.ndarray):
    """uint16 (sim: metros*38, real: mm) -> (dataurl turbo, meta). Igual ao view_cam_live."""
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    vis_min = float(os.environ.get("DEPTH_VIS_MIN_M", "0.2"))
    vis_max = float(os.environ.get("DEPTH_VIS_MAX_M", "1.5"))
    valid = depth > 0
    if not valid.any():
        black = np.zeros((*depth.shape, 3), dtype=np.uint8)
        return _jpg_dataurl(black), {"min": 0, "max": 0, "valid": 0, "visMin": vis_min, "visMax": vis_max}
    # sim publica metros*38 (valores baixos); robô real publica mm.
    scale = 38.0 if float(depth[valid].max()) < 150.0 else 1000.0
    depth_m = depth.astype(np.float32) / scale
    norm = np.clip((depth_m - vis_min) / (vis_max - vis_min), 0.0, 1.0)
    # TURBO; perto (norm baixo) -> vermelho => usa (1-norm). Inválidos pretos.
    d_vis = ((1.0 - norm) * 255).astype(np.uint8)
    d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_TURBO)
    d_color[~valid] = (0, 0, 0)
    meta = {
        "min": round(float(depth_m[valid].min()) * 1000.0),   # mm
        "max": round(float(depth_m[valid].max()) * 1000.0),
        "valid": round(100.0 * float(valid.mean())),
        "visMin": vis_min,
        "visMax": vis_max,
    }
    return _jpg_dataurl(d_color), meta


def _image_thread(host: str, port: int, stop: threading.Event, show_depth: bool, rgb_bytes: bool):
    ctx = zmq.Context.instance()
    _last_frame_warn = [0.0]   # rate-limit do log de frame ruim (1 a cada 5s)
    while not stop.is_set():
        sock = ctx.socket(zmq.SUB)
        # CONFLATE só serve no modo rgb_bytes (mensagem de 1 parte). O stream
        # completo :5555 é MULTIPART (zmq.compressed.v1) e CONFLATE quebra
        # multipart no ZMQ — então lá usamos só RCVHWM=1 + drain manual p/ pegar
        # sempre o frame mais recente sem entupir.
        if rgb_bytes:
            sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.RCVHWM, 1)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        sock.connect(f"tcp://{host}:{port}")
        logger.info("imagens: conectado a tcp://%s:%d (%s)", host, port, "rgb-bytes" if rgb_bytes else "multipart/json")
        try:
            while not stop.is_set():
                # --- modo rgb_bytes: JPEG cru, 1 parte (canal :5558 do VR) ---
                if rgb_bytes:
                    try:
                        jpg = sock.recv()
                    except zmq.Again:
                        continue
                    except Exception:
                        break
                    out = {"rgb": "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")}
                    with _LOCK:
                        if not LATEST.get("img"):
                            LATEST["img"] = {}
                        LATEST["img"].update(out)
                        LATEST["img_seq"] += 1
                    continue

                # --- modo completo :5555: recv_multipart + drain p/ frame mais recente ---
                try:
                    parts = sock.recv_multipart()
                    while True:  # descarta frames acumulados, fica só com o último
                        try:
                            parts = sock.recv_multipart(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                except zmq.Again:
                    continue
                except Exception as e:
                    logger.warning("imagens: recv falhou (%s: %s); reconectando", type(e).__name__, e)
                    break

                try:
                    data = json.loads(parts[0])
                    images = data.get("images", data)
                    protocol = data.get("protocol")
                    rgb_info = images.get("head_camera")
                    depth_info = images.get("head_camera_depth")
                    out = {}

                    # RGB — multipart (dict c/ índice 'part', bytes JPEG crus) ou legado (base64 str).
                    # O JPEG da câmera é codificado na convenção do OpenCV (cv2.imencode = BGR). O MODELO
                    # usa cv2.imdecode → vê o RGB CORRETO. Mas o BROWSER lê JPEG na convenção oposta (RGB),
                    # então o passthrough sairia com R↔B trocado. Pra exibir EXATAMENTE o array que o modelo
                    # vê: decodificamos como o modelo (cv2.imdecode) e re-encodamos browser-fiel via
                    # _jpg_dataurl (que espera BGR) → _m[:, :, ::-1] é o BGR do que o modelo enxerga.
                    _rgb_jpg = None
                    if isinstance(rgb_info, dict) and protocol == "zmq.compressed.v1":
                        idx = rgb_info.get("part")
                        if idx is not None and idx < len(parts):
                            _rgb_jpg = parts[idx]
                    elif isinstance(rgb_info, str):
                        _rgb_jpg = base64.b64decode(rgb_info)
                    if _rgb_jpg is not None:
                        _m = cv2.imdecode(np.frombuffer(_rgb_jpg, np.uint8), cv2.IMREAD_COLOR)  # = visão do modelo
                        out["rgb"] = (_jpg_dataurl(_m[:, :, ::-1]) if _m is not None
                                      else "data:image/jpeg;base64," + base64.b64encode(_rgb_jpg).decode("ascii"))

                    # Depth — multipart (PNG uint16 crus) ou legado (base64 str)
                    if show_depth:
                        depth = None
                        if isinstance(depth_info, dict) and protocol == "zmq.compressed.v1":
                            idx = depth_info.get("part")
                            if idx is not None and idx < len(parts):
                                depth = cv2.imdecode(np.frombuffer(parts[idx], np.uint8), cv2.IMREAD_UNCHANGED)
                        elif isinstance(depth_info, str):
                            depth = cv2.imdecode(np.frombuffer(base64.b64decode(depth_info), np.uint8), cv2.IMREAD_UNCHANGED)
                        if depth is not None:
                            url, meta = _colorize_depth(depth)
                            out["depth"] = url
                            out["depthMeta"] = meta
                except Exception as e:
                    # frame ruim não pode matar a thread: descarta, loga (com folga) e segue
                    now = time.time()
                    if now - _last_frame_warn[0] > 5.0:
                        _last_frame_warn[0] = now
                        logger.warning("imagens: frame descartado (%s: %s)", type(e).__name__, e)
                    continue

                if out:
                    with _LOCK:
                        if not LATEST.get("img"):
                            LATEST["img"] = {}
                        LATEST["img"].update(out)
                        LATEST["img_seq"] += 1
        except Exception as e:
            logger.warning("imagens: thread caiu (%s: %s); reconectando em 0.5s", type(e).__name__, e)
        finally:
            sock.close(0)
        if not stop.is_set():
            time.sleep(0.5)  # fonte caiu; tenta reconectar


def _quest_thread(quest_adb: str, stop: threading.Event):
    """Sonda o Quest via ADB a cada 2s -> LATEST['quest'] (True/False).

    `quest_adb` = "<ip>:5555" (o headset). Vazio => não sonda (status fica 'n/d').
    'device' no get-state = headset alcançável na rede (proxy de "Quest conectado").
    """
    import subprocess
    if not quest_adb:
        return
    while not stop.is_set():
        connected = False
        try:
            r = subprocess.run(["adb", "-s", quest_adb, "get-state"],
                               capture_output=True, text=True, timeout=4)
            connected = (r.returncode == 0 and "device" in r.stdout)
        except Exception:
            connected = False
        with _LOCK:
            LATEST["quest"] = connected
        stop.wait(2.0)


def _tele_thread(port: int, stop: threading.Event):
    ctx = zmq.Context.instance()

    def _handle(msg: str):
        try:
            pkt = json.loads(msg)
        except Exception:
            return
        with _LOCK:
            # mapa de atenção da VLA (inferência): vai pro canal de IMAGEM.
            # attn_jpg = overlay pronto (legado); attn_hm = só o heatmap JET
            # pequeno, que o navegador compõe por cima do RGB ao vivo.
            attn = pkt.pop("attn_jpg", None)
            attn_hm = pkt.pop("attn_hm", None)
            if attn or attn_hm:
                if not LATEST.get("img"):
                    LATEST["img"] = {}
                if attn:
                    LATEST["img"]["attn"] = attn
                if attn_hm:
                    LATEST["img"]["attn_hm"] = attn_hm
                LATEST["attn_ts"] = time.time()
                LATEST["img_seq"] += 1
                if not pkt or set(pkt) <= {"type"}:
                    return
            LATEST["tele"] = pkt
            LATEST["tele_seq"] += 1

    while not stop.is_set():
        sock = ctx.socket(zmq.SUB)
        # SEM CONFLATE: o canal carrega DOIS tipos de pacote (telemetria e attn_jpg da
        # inferência) e CONFLATE guardaria só o último, fazendo um apagar o outro.
        # Em vez disso, drena a fila e guarda o mais recente de CADA tipo.
        sock.setsockopt(zmq.RCVHWM, 8)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        sock.connect(f"tcp://127.0.0.1:{port}")
        logger.info("telemetria: conectado a tcp://127.0.0.1:%d", port)
        try:
            while not stop.is_set():
                try:
                    msg = sock.recv_string()
                except zmq.Again:
                    continue
                except Exception as e:
                    logger.warning("telemetria: recv falhou (%s: %s); reconectando", type(e).__name__, e)
                    break
                _handle(msg)
                # drena o backlog sem bloquear (fica só com o mais recente de cada tipo)
                while True:
                    try:
                        _handle(sock.recv_string(flags=zmq.NOBLOCK))
                    except zmq.Again:
                        break
                    except Exception:
                        break
        except Exception as e:
            logger.warning("telemetria: thread caiu (%s: %s); reconectando em 0.5s", type(e).__name__, e)
        finally:
            sock.close(0)
        if not stop.is_set():
            time.sleep(0.5)


# ----------------------------------------------------------------------------- web / ws
async def _safe_close(ws):
    try:
        await ws.close()
    except Exception:
        pass


async def _ws_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    clients = request.app["clients"]
    # SÓ 1 CLIENTE: derruba as conexões anteriores. Reconexões órfãs (o ws cai sem o
    # servidor perceber e o browser reconecta) acumulavam no set e multiplicavam o
    # broadcast — 1 aba virava 5 clientes e a CPU do viewer estourava. O cliente novo
    # substitui o antigo; o discard é imediato (para o broadcast já), o close roda em
    # segundo plano pra não travar caso o socket velho esteja morto.
    for old in list(clients):
        clients.discard(old)
        asyncio.create_task(_safe_close(old))
    clients.add(ws)
    logger.info("ws conectado (%d clientes)", len(clients))
    try:
        async for m in ws:  # só mantém viva; o broadcast empurra os frames
            if m.type == WSMsgType.ERROR:
                break
    finally:
        request.app["clients"].discard(ws)
        logger.info("ws desconectado (%d clientes)", len(request.app["clients"]))
    return ws


async def _broadcast(app: web.Application):
    sent_img = sent_tele = 0
    tick = 0
    last_img_tick = -10**9
    while True:
        await asyncio.sleep(1 / 60)
        tick += 1
        clients = app["clients"]
        if not clients:
            sent_img, sent_tele = LATEST["img_seq"], LATEST["tele_seq"]
            continue
        payloads = []
        # limita os frames de câmera (só visualização) p/ não estourar a memória do
        # browser com data/blobs grandes a 30Hz — 848×480 real era pesado demais.
        img_every = app["img_every"]
        send_img = (tick - last_img_tick) >= img_every
        with _LOCK:
            # atenção parada (inferência off há >5s): tira do pacote pra não reenviar
            # heatmap fóssil 30x/s pela rede.
            img = LATEST["img"]
            if img and ("attn" in img or "attn_hm" in img) \
                    and time.time() - LATEST.get("attn_ts", 0.0) > 5.0:
                img.pop("attn", None)
                img.pop("attn_hm", None)
            if send_img and LATEST["img_seq"] != sent_img and LATEST["img"] is not None:
                sent_img = LATEST["img_seq"]
                last_img_tick = tick
                payloads.append({"type": "img", **LATEST["img"]})
            if LATEST["tele_seq"] != sent_tele and LATEST["tele"] is not None:
                sent_tele = LATEST["tele_seq"]
                payloads.append(LATEST["tele"])
            if tick % 30 == 0:   # ~0.5s: status de conexão do Quest (PC/G1 o browser deduz sozinho)
                payloads.append({"type": "links", "quest": LATEST["quest"]})
        for p in payloads:
            txt = json.dumps(p)
            for ws in list(clients):
                try:
                    await ws.send_str(txt)
                except Exception:
                    clients.discard(ws)


async def _on_startup(app: web.Application):
    app["broadcast_task"] = asyncio.create_task(_broadcast(app))


async def _on_cleanup(app: web.Application):
    app["broadcast_task"].cancel()
    for ws in list(app["clients"]):
        await ws.close()


def build_app() -> web.Application:
    app = web.Application()
    app["clients"] = set()
    app.router.add_get("/ws", _ws_handler)
    app.router.add_get("/", lambda r: web.HTTPFound("/live.html"))
    app.router.add_static("/", str(WEBAPP_DIR), show_index=False)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


def _open_browser_when_ready(host: str, port: int):
    target = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    for _ in range(60):
        try:
            with socket.create_connection((target, port), timeout=0.5):
                pass
        except OSError:
            time.sleep(0.5)
            continue
        url = f"http://{target}:{port}/live.html"
        logger.info("abrindo browser em %s", url)
        webbrowser.open(url)
        return


def main():
    ap = argparse.ArgumentParser(description="Dashboard web ao vivo (estilo OmniView) do G1")
    ap.add_argument("--host", default="127.0.0.1", help="host do PUB de câmeras (sim=127.0.0.1)")
    ap.add_argument("--cam-port", type=int, default=5555)
    ap.add_argument("--tele-port", type=int, default=5557)
    ap.add_argument("--http-host", default="127.0.0.1")
    ap.add_argument("--http-port", type=int, default=8013)
    ap.add_argument("--quest-adb", default="", help="<ip>:5555 do Quest p/ status via adb (vazio=n/d)")
    ap.add_argument("--img-fps", type=float, default=30.0,
                    help="teto de FPS dos frames de câmera enviados ao browser (só viz; dataset intocado)")
    ap.add_argument("--rgb-bytes", action="store_true",
                    help="recebe o stream como JPEG bytes RGB-only, sem JSON/depth")
    ap.add_argument("--no-depth", action="store_true",
                    help="não decodifica/coloriza depth no OmniView LIVE (alivia CPU; dataset intocado)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    stop = threading.Event()
    show_depth = (not args.rgb_bytes and not args.no_depth
                  and os.environ.get("G1_OMNIVIEW_DEPTH", "1") not in ("", "0", "false", "False"))
    t_img = threading.Thread(target=_image_thread, args=(args.host, args.cam_port, stop, show_depth, args.rgb_bytes), daemon=True)
    t_tel = threading.Thread(target=_tele_thread, args=(args.tele_port, stop), daemon=True)
    t_quest = threading.Thread(target=_quest_thread, args=(args.quest_adb, stop), daemon=True)
    t_img.start()
    t_tel.start()
    t_quest.start()
    if not show_depth:
        logger.info("depth: desativado no OmniView LIVE (RGB + telemetria apenas)")

    if not args.no_browser:
        threading.Thread(
            target=_open_browser_when_ready, args=(args.http_host, args.http_port), daemon=True
        ).start()

    logger.info("dashboard em http://%s:%d/live.html  (Ctrl-C pra sair)", args.http_host, args.http_port)
    app = build_app()
    app["img_every"] = max(1, round(60.0 / max(1.0, args.img_fps)))  # ticks de 1/60s entre frames
    try:
        web.run_app(app, host=args.http_host, port=args.http_port, print=None, handle_signals=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
