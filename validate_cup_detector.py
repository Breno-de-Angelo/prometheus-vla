#!/usr/bin/env python3
# Passo 1 da 4b: valida um detector open-vocab pra ROTULAR o copo (centroide) no dataset.
# Extrai N frames de fases variadas, roda GroundingDINO/OWLv2 com "a white cup." e salva
# overlays (bbox+centroide) + um grid.jpg + imprime cx,cy,conf. Serve pra decidir se os
# rótulos da grounding loss são viáveis (e qual prompt/threshold/detector usar).
# Precisa de REDE p/ baixar os pesos (NÃO usar HF_HUB_OFFLINE aqui).
import sys, argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "lerobot-ext"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

IMG = "observation.images.head_camera"

ap = argparse.ArgumentParser()
ap.add_argument("--repo-id", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
ap.add_argument("--prompt", default="a white cup.")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--episodes", type=int, default=8)
ap.add_argument("--outdir", default="/tmp/cup_detect_val")
ap.add_argument("--box-thr", type=float, default=0.20)
ap.add_argument("--text-thr", type=float, default=0.20)
args = ap.parse_args()

out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[load] {args.model} on {dev}")
proc = AutoProcessor.from_pretrained(args.model)
model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(dev).eval()

ds = LeRobotDataset(args.repo_id, root=args.root, episodes=list(range(args.episodes)), video_backend="pyav")
idx = np.linspace(0, len(ds) - 1, args.n).astype(int)


def to_pil(t):
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    a = np.squeeze(a)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = a * 255 if float(a.max()) <= 1.5 else a
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB")


hits = 0
print(f"{'frame':>6} | {'cx':>5} {'cy':>5} | {'conf':>5} | bbox")
for i in idx:
    img = to_pil(ds[int(i)][IMG]); W, H = img.size
    inp = proc(images=img, text=args.prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        outp = model(**inp)
    res = proc.post_process_grounded_object_detection(
        outp, inp.input_ids, box_threshold=args.box_thr, text_threshold=args.text_thr,
        target_sizes=[(H, W)])[0]
    dr = ImageDraw.Draw(img)
    if len(res["scores"]):
        j = int(res["scores"].argmax())
        b = [float(x) for x in res["boxes"][j].tolist()]
        conf = float(res["scores"][j])
        mx, my = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        dr.rectangle(b, outline=(0, 255, 0), width=3)
        dr.ellipse([mx - 5, my - 5, mx + 5, my + 5], fill=(255, 0, 0))
        print(f"{int(i):6d} | {mx/W:5.2f} {my/H:5.2f} | {conf:5.2f} | {[round(x) for x in b]}")
        hits += 1
    else:
        print(f"{int(i):6d} | {'--':>5} {'--':>5} | {'0':>5} | (nada)")
    img.save(out / f"det_{int(i):05d}.jpg")

# grid de todos os overlays p/ inspeção rápida
cols = 4; rows = (args.n + cols - 1) // cols
tw, th = 240, 136
grid = Image.new("RGB", (cols * tw, rows * th), (18, 18, 22))
for k, p in enumerate(sorted(out.glob("det_*.jpg"))):
    grid.paste(Image.open(p).resize((tw, th)), ((k % cols) * tw, (k // cols) * th))
grid.save(out / "grid.jpg")
print(f"\n[ok] {hits}/{len(idx)} frames com copo detectado (conf>thr)")
print(f"[ok] overlays + grid.jpg em {out}/")
