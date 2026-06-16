#!/usr/bin/env python3
# Converte uma run gravada pela inferência (--record: chunks.jsonl + frames.jsonl + rgb/depth/attn)
# para o formato de replay do gen_episode_replay.py (manifest.json + f/*.jpg), pra tocar no
# build_replay_html.py JÁ EXISTENTE. Num deploy não há GT humano -> "gt" = ação EXECUTADA no robô.
# Também salva analysis.png (4 painéis) com os achados.
# Uso: gen_deploy_replay.py <run_dir> <out_replay_dir> [label]
import sys, json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(sys.argv[1]); OUT = Path(sys.argv[2]); LABEL = sys.argv[3] if len(sys.argv) > 3 else "as-is-deploy"
(OUT / "f").mkdir(parents=True, exist_ok=True)
OUT_W = 300
ARM = ["ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "Elbow", "WristRoll", "WristPitch", "WristYaw"]
IDX1_TGT = 1.74  # right_hand_index_1 alvo fechado -> proxy de squeeze executado

ch = [json.loads(l) for l in open(R / "chunks.jsonl") if l.strip()]
fr = [json.loads(l) for l in open(R / "frames.jsonl") if l.strip()]
meta = json.load(open(R / "meta.json"))
# run name + link do wandb (do train_config.json do checkpoint, automático)
wandb_url = ""; run_name = meta.get("checkpoint", "")
try:
    import os.path as _osp
    _tc = json.load(open(_osp.join(meta.get("checkpoint", ""), "train_config.json")))
    run_name = _tc.get("job_name") or run_name
    _w = _tc.get("wandb") or {}
    if _w.get("run_id"):
        wandb_url = f"https://wandb.ai/{_w.get('entity') or 'prometheus-lcad'}/{_w.get('project') or 'prometheus_g1'}/runs/{_w['run_id']}"
except Exception as _e:
    print("aviso: wandb do train_config indisponível:", _e)
t = np.array([c["t"] for c in ch], dtype=float); t0 = t[0]; tt = t - t0
ft = np.array([f["t"] for f in fr], dtype=float) - t0
fa = np.array([[f["action"].get("kRight%s.q" % nm, 0.0) for nm in ARM] for f in fr], dtype=float)
fidx1 = np.array([f["action"].get("right_hand_index_1_joint.q", 0.0) for f in fr], dtype=float)
idxf = np.clip(np.searchsorted(ft, tt), 0, len(fr) - 1)


def save_jpg(rgb, p):
    img = Image.fromarray(rgb)
    if img.width > OUT_W:
        img = img.resize((OUT_W, int(img.height * OUT_W / img.width)), Image.BILINEAR)
    img.convert("RGB").save(p, quality=80)


def colorize_depth(d):
    v = d[d > 0]
    lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (float(d.min()), float(d.max()))
    n = np.clip((d.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]  # ->RGB


frames = []
for i, c in enumerate(ch):
    rgb = cv2.imread(str(R / c["rgb"]))[:, :, ::-1]   # BGR->RGB
    H, W = rgb.shape[:2]
    save_jpg(np.ascontiguousarray(rgb), OUT / "f" / f"rgb_{i:03d}.jpg")
    if c.get("depth") and (R / c["depth"]).exists():
        d = cv2.imread(str(R / c["depth"]), cv2.IMREAD_UNCHANGED)
        save_jpg(np.ascontiguousarray(colorize_depth(d)), OUT / "f" / f"dep_{i:03d}.jpg")
    if c.get("attn") and (R / c["attn"]).exists():
        heat = cv2.imread(str(R / c["attn"]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        g = heat.shape[0]; pad = (1.0 - H / W) / 2.0
        r0, r1 = int(round(g * pad)), int(round(g * (1.0 - pad)))
        hc = heat[max(0, r0):min(g, r1), :]
        hc = cv2.resize(hc, (W, H), interpolation=cv2.INTER_CUBIC)
        rng = float(hc.max() - hc.min()) or 1.0
        hm = cv2.applyColorMap((255 * (hc - hc.min()) / rng).astype(np.uint8), cv2.COLORMAP_JET)
        over = cv2.addWeighted(np.ascontiguousarray(rgb[:, :, ::-1]), 0.6, hm, 0.4, 0)  # BGR
        over = cv2.resize(over, (OUT_W, int(H * OUT_W / W)))
        cv2.imwrite(str(OUT / "f" / f"att_{i:03d}.jpg"), over, [cv2.IMWRITE_JPEG_QUALITY, 80])
    chunk = np.array(c["actions"], dtype=float)       # [50,8] físico: 7 rad + squeeze
    pred = chunk[0]
    gt_j = fa[idxf[i]]
    gt_sq = float(np.clip(fidx1[idxf[i]] / IDX1_TGT, 0, 1))
    frames.append(dict(
        i=i, pred_sq=round(float(pred[7]), 3), gt_sq=round(gt_sq, 3),
        pred_j=[round(float(x), 4) for x in pred[:7]],
        gt_j=[round(float(x), 4) for x in gt_j],
        chunk=[[round(float(v), 4) for v in chunk[k][:7]] + [round(float(chunk[k][7]), 3)]
               for k in range(len(chunk))],
    ))

man = dict(label=LABEL, ckpt=meta.get("checkpoint", ""), cfg=meta, wandb_url=wandb_url, run_name=run_name,
           episode="DEPLOY (robô real)", mode="8dim", seed=None, n_frames=len(frames), fps=8, frames=frames)
(OUT / "manifest.json").write_text(json.dumps(man))
print("[ok] manifest:", OUT / "manifest.json", len(frames), "frames")

# ---------- figura de análise ----------
A = np.array([np.array(c["actions"], dtype=float) for c in ch])
S = np.array([c["state_raw"] for c in ch], dtype=float)
sq = A[:, 0, 7]
attn = np.array([cv2.imread(str(R / c["attn"]), cv2.IMREAD_GRAYSCALE).astype(float) for c in ch])
an = attn / (attn.sum((1, 2), keepdims=True) + 1e-9)
ent = -(an * np.log(an + 1e-12)).sum((1, 2)); me = float(np.log(256))
yy, xx = np.mgrid[0:16, 0:16]; comx = (an * xx).sum((1, 2)); comy = (an * yy).sum((1, 2))
gap = np.abs(fa[idxf] - A[:, 0, :7])
flips = int(np.abs(np.diff((sq > 0.5).astype(int))).sum())
fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
ax[0].plot(tt, sq, c="crimson", lw=1); ax[0].axhline(0.7, ls=":", c="gray"); ax[0].axhline(0.3, ls=":", c="gray")
ax[0].set_ylabel("squeeze"); ax[0].set_title("GRASP: fechado %.0f%% do tempo, %d flip-flops -> pose-driven/erratico" % (100 * (sq > 0.7).mean(), flips))
for i in [0, 3, 5]:
    ax[1].plot(tt, S[:, i], lw=1, label=ARM[i])
ax[1].legend(fontsize=8); ax[1].set_ylabel("braco (rad)"); ax[1].set_title("BRACO: ombro std %.2f (quase parado) | cotovelo std %.2f (faz tudo)" % (S[:, 0].std(), S[:, 3].std()))
ax[2].plot(tt, comx, lw=1, label="COM x"); ax[2].plot(tt, comy, lw=1, label="COM y"); ax[2].set_ylim(0, 15); ax[2].legend(fontsize=8)
ax[2].set_ylabel("attn COM"); ax[2].set_title("ATENCAO COM ~fixa no centro (std x+y=%.2f) -> nao rastreia o copo" % (comx.std() + comy.std()))
ax[3].plot(tt, 100 * ent / me, c="purple", lw=1); ax[3].axhline(85, ls=":", c="red"); ax[3].set_ylabel("entropia attn (%)"); ax[3].set_xlabel("t (s)")
ax[3].set_title("ATENCAO DIFUSA ~%.0f%% do max + gap previsto-vs-exec %.3f rad (sem lag)" % (100 * ent.mean() / me, gap.mean()))
fig.tight_layout(); fig.savefig(OUT / "analysis.png", dpi=92); plt.close(fig)
print("[ok] analysis.png")
