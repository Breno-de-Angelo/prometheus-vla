#!/usr/bin/env python3
"""Viewer ao vivo das câmeras RGB e Depth do robô via ZMQ."""
import argparse
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
            # Convenção pedida: VERMELHO = mais perto, AZUL = mais longe.
            # depth do servidor: valor alto = mais longe; 0 = pixel inválido (sem leitura).
            valid = depth > 0
            if valid.any():
                # percentis 5–95 dos VÁLIDOS → ignora inválidos e outliers, dá contraste estável.
                lo = float(np.percentile(depth[valid], 5))
                hi = float(np.percentile(depth[valid], 95))
                hi = hi if hi > lo else lo + 1.0
                norm = np.clip((depth.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
                # JET: 0=azul, 255=vermelho. perto (depth baixo) → 255 (vermelho);
                # longe (depth alto) → 0 (azul). Daí o (1 - norm).
                d_vis = ((1.0 - norm) * 255).astype(np.uint8)
                d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_JET)
                d_color[~valid] = (0, 0, 0)   # inválidos PRETOS (não vermelhos)
                lab_lo, lab_hi = int(depth[valid].min()), int(depth[valid].max())
            else:
                d_color = np.zeros((*depth.shape, 3), dtype=np.uint8)
                lab_lo, lab_hi = 0, 0
            label = f"Depth [{lab_lo}-{lab_hi}]  vermelho=perto azul=longe"
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
