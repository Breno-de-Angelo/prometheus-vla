#!/usr/bin/env python3
"""Viewer ao vivo das câmeras RGB e Depth do robô via ZMQ."""
import argparse
import os
import sys
import time
import json
import numpy as np
import cv2
from pathlib import Path

_STATUS_FILE = "/tmp/g1_record_status.json"
_status_cache = {"episode": 0, "start_time": time.time()}

def _read_status():
    try:
        with open(_STATUS_FILE) as f:
            _status_cache.update(json.load(f))
    except Exception:
        pass
    return _status_cache

sys.path.insert(0, str(Path(__file__).parent / "unitree-g1-mujoco"))
from sim.sensor_utils import SensorClient, ImageUtils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    client = SensorClient()
    client.start_client(server_ip=args.host, port=args.port)
    print(f"[CamViewer] Conectado a tcp://{args.host}:{args.port} — pressione Q para fechar.")

    while True:
        try:
            data = client.receive_message()
        except Exception:
            continue

        if not data:
            continue

        images = data.get("images", data)
        rgb   = images.get("head_camera")
        depth = images.get("head_camera_depth")

        panels = []

        if rgb is not None:
            if isinstance(rgb, str):
                rgb = ImageUtils.decode_image(rgb)
            # decode_image já retorna BGR (cv2.imdecode); cv2.imshow espera BGR → mostra
            # direto. (Antes fazia RGB2BGR assumindo input RGB, trocando R↔B → tudo azulado.)
            rgb_bgr = rgb
            panels.append(("RGB", rgb_bgr))

        if depth is not None:
            if isinstance(depth, str):
                depth = ImageUtils.decode_image(depth)
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            # ──────────────────────────────────────────────────────────────────
            # ⚠️ ESTA COLORIZAÇÃO (JET) É *PURAMENTE VISUALIZAÇÃO* PRO OPERADOR.
            # NADA disto vai pro dataset nem pra VLA — é só pro humano ver no viewer.
            #
            # O depth REALMENTE GRAVADO é o uint16 RAW em milímetros, em outro caminho:
            #   • servidor PRODUZ o depth em mm e envia PNG uint16 (1 canal, lossless):
            #       lerobot-ext/Scripts_Prometheus_int/realsense_server.py:48-53 (depth_mm)
            #   • cliente DECODIFICA como uint16 (IMREAD_UNCHANGED), sem colorizar, no
            #     monkeypatch `_patched_zmq_read`:
            #       lerobot-ext/init_lerobot_record_v2.py (~linha 118, "INJEÇÃO 3.1")
            #     → é ESSE array uint16 (mm) que o LeRobot grava como observação.
            # Mudar a cor aqui NÃO altera o dataset (nem precisa rodar no robô).
            # ──────────────────────────────────────────────────────────────────
            # Convenção: VERMELHO = perto, AZUL = longe.
            # Usa FAIXA ABSOLUTA FIXA (em metros), NÃO auto-estica por frame. Auto-esticar
            # (min/max ou percentis do frame) faz uma cena toda perto — mesa a 0.5-1m —
            # ser espalhada no arco-íris inteiro e parecer "longe". Com faixa fixa, a cor
            # significa distância real. Ajuste a faixa por env DEPTH_VIS_MIN_M/MAX_M.
            vis_min = float(os.environ.get("DEPTH_VIS_MIN_M", "0.2"))
            vis_max = float(os.environ.get("DEPTH_VIS_MAX_M", "1.5"))
            valid = depth > 0
            if valid.any():
                # unidade: sim publica metros*38 (valores baixos); robô real publica mm.
                # detecta pela magnitude e converte pra metros.
                scale = 38.0 if float(depth[valid].max()) < 150.0 else 1000.0
                depth_m = depth.astype(np.float32) / scale
                norm = np.clip((depth_m - vis_min) / (vis_max - vis_min), 0.0, 1.0)
                # JET: 0=azul, 255=vermelho → perto (norm baixo) vira vermelho. Daí (1-norm).
                d_vis = ((1.0 - norm) * 255).astype(np.uint8)
                d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_JET)
                d_color[~valid] = (0, 0, 0)   # inválidos PRETOS (não vermelhos)
                lab_lo, lab_hi = float(depth_m[valid].min()), float(depth_m[valid].max())
                label = (f"Depth real [{lab_lo:.2f}-{lab_hi:.2f}m] | "
                         f"escala fixa {vis_min:.1f}-{vis_max:.1f}m  vermelho=perto azul=longe")
            else:
                d_color = np.zeros((*depth.shape, 3), dtype=np.uint8)
                label = "Depth [sem dados]"
            panels.append((label, d_color))

        if not panels:
            continue

        # redimensiona tudo para mesma altura e concatena lado a lado
        h = max(p[1].shape[0] for p in panels)
        strips = []
        for label, img in panels:
            scale = h / img.shape[0]
            w2 = int(img.shape[1] * scale)
            resized = cv2.resize(img, (w2, h))
            cv2.putText(resized, label, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            strips.append(resized)

        combined = np.concatenate(strips, axis=1)

        # HUD: episódio + timer
        st = _read_status()
        elapsed = time.time() - st["start_time"]
        mins, secs = divmod(int(elapsed), 60)
        hud = f"EP {st['episode']:03d}  {mins:02d}:{secs:02d}"
        # fundo semitransparente atrás do texto
        (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(combined, (8, combined.shape[0] - th - 16), (tw + 16, combined.shape[0] - 4), (0, 0, 0), -1)
        cv2.putText(combined, hud, (12, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Robot Cameras", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    client.stop_client()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
