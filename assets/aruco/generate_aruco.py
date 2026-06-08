#!/usr/bin/env python3
"""
Gera marcadores ArUco pra registro sim<->real (mesa + copo).

- DICT_4X4_50, IDs 0-3 = mesa, 4-7 = copo.
- Tamanho FÍSICO embutido via DPI (imprima em 100% / "tamanho real" pra pose correta).
- Quiet zone branca de 1 célula ao redor (exigida pela detecção).
- Rótulo de id/tamanho numa faixa abaixo (fora da quiet zone).
- Emite markers.json com id -> grupo, tamanho_mm, arquivo (consumido pelo detector e pelo sim).

Uso: python assets/aruco/generate_aruco.py
"""
import cv2
import numpy as np
import json
import os
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "markers")      # PNGs p/ IMPRIMIR (com rótulo)
TEX = os.path.join(HERE, "textures")     # PNGs limpos p/ o SIM (sem rótulo, quadrados)
os.makedirs(OUT, exist_ok=True)
os.makedirs(TEX, exist_ok=True)

DICT_NAME = "DICT_4X4_50"
DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
DPI = 300  # imprima em 100% pra respeitar o tamanho físico

GROUPS = {
    "table": {"ids": [0, 1, 2, 3], "size_mm": 60},  # mesa: estáticos, maiores
    "cup":   {"ids": [4, 5, 6, 7], "size_mm": 25},  # copo: pequenos (objeto pequeno)
}

def mm_to_px(mm):
    return round(mm / 25.4 * DPI)

meta = {"dictionary": DICT_NAME, "dpi": DPI, "markers": []}

for group, cfg in GROUPS.items():
    side = mm_to_px(cfg["size_mm"])
    qz = side // 6  # 1 célula de quiet zone (marcador 4x4 + borda = 6 células)
    for mid in cfg["ids"]:
        m = cv2.aruco.generateImageMarker(DICT, mid, side)
        m = cv2.copyMakeBorder(m, qz, qz, qz, qz, cv2.BORDER_CONSTANT, value=255)
        # textura LIMPA p/ o sim (marcador + quiet zone, sem rótulo); 512x512 p/ o render
        tex = cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(tex).save(os.path.join(TEX, f"{group}_id{mid:02d}.png"))
        # faixa de rótulo abaixo (não interfere na quiet zone do marcador)
        h, w = m.shape
        label_h = max(40, side // 8)
        canvas = np.full((h + label_h, w), 255, np.uint8)
        canvas[:h, :] = m
        cv2.putText(canvas, f"{group} id={mid}  {cfg['size_mm']}mm",
                    (qz, h + label_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    side / 1400, 0, max(1, side // 400), cv2.LINE_AA)
        fname = f"{group}_id{mid:02d}_{cfg['size_mm']}mm.png"
        path = os.path.join(OUT, fname)
        Image.fromarray(canvas).save(path, dpi=(DPI, DPI))  # DPI -> tamanho físico
        meta["markers"].append({
            "id": mid, "group": group, "size_mm": cfg["size_mm"], "file": f"markers/{fname}",
        })
        print(f"[ok] {fname}  ({w}x{h+label_h}px @ {DPI}dpi -> marcador {cfg['size_mm']}mm)")

with open(os.path.join(HERE, "markers.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"\n[meta] {len(meta['markers'])} marcadores -> markers.json")
