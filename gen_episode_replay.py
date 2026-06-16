#!/usr/bin/env python3
# Grava o REPLAY de 1 episódio completo rodando o modelo de fato (como a eval/inferência
# offline): por frame, prevê a ação (predict_action_chunk) gravando a ATENÇÃO real do
# action expert sobre a imagem, e salva RGB + depth + atenção + métricas (força/comando/GT).
# Roda 1× e gera os assets; o player HTML toca em loop sem re-rodar o modelo.
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cv2

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lerobot-ext"))
sys.path.insert(0, str(REPO / "lerobot-ext" / "train"))

import probe_saliency as PS
from attn_recorder import AttnRecorder, overlay_heatmap

IMG = "observation.images.head_camera"
DEPTH = "observation.images.head_camera_depth"
OUT_W = 300


def save_jpg(arr_rgb, path, w=OUT_W):
    img = Image.fromarray(arr_rgb)
    if img.width > w:
        img = img.resize((w, int(img.height * w / img.width)), Image.BILINEAR)
    img.convert("RGB").save(path, quality=80)


def depth_rgb(sample):
    d = sample.get(DEPTH)
    if d is None:
        return None
    a = d.detach().cpu().numpy() if torch.is_tensor(d) else np.asarray(d)
    a = np.squeeze(a).astype(np.float32)
    valid = a[a > 0]
    lo, hi = (np.percentile(valid, 2), np.percentile(valid, 98)) if valid.size else (a.min(), a.max())
    n = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    return PS.colorize(n)


def unnorm_arm(norm7, stats):
    # normalizado [-1,1] -> radianos (junta), via q01/q99 do dataset
    q01 = np.asarray(stats["action"]["q01"], dtype=np.float32)[:7]
    q99 = np.asarray(stats["action"]["q99"], dtype=np.float32)[:7]
    return (np.asarray(norm7, dtype=np.float32) + 1.0) / 2.0 * (q99 - q01) + q01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--mode", default="8dim")
    ap.add_argument("--label", default="run")
    ap.add_argument("--seed", type=int, default=None,
                    help="se setado, fixa o ruído do flow-matching (mesmo seed todo frame) "
                         "-> trajetória determinística/lisa; None = ruído fresco por frame")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); (outdir / "f").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] policy {args.ckpt}")
    policy = PS.PI05Policy.from_pretrained(args.ckpt, strict=False).to(device).eval()
    preproc, _ = PS.make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.ckpt)
    print(f"[load] dataset ep {args.episode}")
    ds = PS.LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode], video_backend="pyav")
    stats = ds.meta.stats
    fps = int(getattr(ds.meta, "fps", 15) or 15)
    n = len(ds)
    print(f"[run] {n} frames, fps={fps}")

    rec = AttnRecorder().install()
    frames = []
    for i in range(n):
        sample = ds[i]
        rgb = PS.img_to_hwc_uint8(sample[IMG])
        H, W = rgb.shape[:2]
        batch = PS.make_batch(sample, policy, device)
        # predict_action_chunk gravando a atenção
        with torch.no_grad():
            if args.seed is not None:
                torch.manual_seed(args.seed)   # mesmo ruído todo frame -> determinístico
            with rec:
                a = policy.predict_action_chunk(preproc(PS.clone(batch))).detach()
        chunk = a[0].float().cpu().numpy()[:, :8]  # [50, 8] normalizado (todo o chunk)
        pred = chunk[0]  # 1ª ação do chunk = a que seria executada
        heat = rec.heatmap()

        save_jpg(rgb, outdir / "f" / f"rgb_{i:03d}.jpg")
        dr = depth_rgb(sample)
        if dr is not None:
            save_jpg(dr, outdir / "f" / f"dep_{i:03d}.jpg")
        if heat is not None:
            band = H / W
            g = heat.shape[0]; pad = (1.0 - band) / 2.0
            r0, r1 = int(round(g * pad)), int(round(g * (1.0 - pad)))
            hc = heat[max(0, r0):min(g, r1), :]
            over = overlay_heatmap(rgb, hc, out_width=OUT_W)  # BGR
            cv2.imwrite(str(outdir / "f" / f"att_{i:03d}.jpg"), over, [cv2.IMWRITE_JPEG_QUALITY, 80])

        gt = np.asarray(sample["action"], dtype=np.float32)
        chunk_j = unnorm_arm(chunk[:, :7], stats)        # [50,7] juntas em radianos
        chunk_sq = (chunk[:, 7] + 1.0) / 2.0             # [50] squeeze 0-1 (recuperação 8dim)
        frames.append(dict(
            i=i,
            pred_sq=round(PS.pred_squeeze(pred, args.mode), 3),
            gt_sq=round(PS.gt_squeeze(gt, args.mode), 3),
            pred_j=[round(float(x), 4) for x in unnorm_arm(pred[:7], stats)],   # juntas previstas (rad)
            gt_j=[round(float(x), 4) for x in gt[:7]],                          # juntas reais (rad)
            chunk=[[round(float(v), 4) for v in chunk_j[k]] + [round(float(chunk_sq[k]), 3)]
                   for k in range(len(chunk_j))],                              # 50 x [7 juntas rad + squeeze]
        ))
        if i % 25 == 0:
            print(f"  frame {i}/{n} pred_sq={frames[-1]['pred_sq']:.2f} gt_sq={frames[-1]['gt_sq']:.2f}")

    man = dict(label=args.label, ckpt=args.ckpt, episode=args.episode, mode=args.mode,
               seed=args.seed, n_frames=n, fps=fps, frames=frames)
    (outdir / "manifest.json").write_text(json.dumps(man))
    print(f"[ok] {outdir}/manifest.json  ({n} frames)")


if __name__ == "__main__":
    main()
