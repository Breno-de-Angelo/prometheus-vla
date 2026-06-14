#!/usr/bin/env python3
# Gera os assets visuais do probe de grounding p/ o infografico: RGB, depth,
# atencao real (attn_recorder) e saliencia (oclusao) de frames OPEN/CLOSED, mais
# o chunk predito x GT. Reusa probe_saliency + attn_recorder. Saida: PNGs + manifest.json.
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
OUT_W = 424


def _resize_save(arr_rgb, path):
    img = Image.fromarray(arr_rgb)
    if img.width > OUT_W:
        img = img.resize((OUT_W, int(img.height * OUT_W / img.width)), Image.BILINEAR)
    img.save(path)
    return path.name


def depth_png(sample, path):
    d = sample.get(DEPTH)
    if d is None:
        return None
    a = d.detach().cpu().numpy() if torch.is_tensor(d) else np.asarray(d)
    a = np.squeeze(a).astype(np.float32)
    valid = a[a > 0]
    lo, hi = (np.percentile(valid, 2), np.percentile(valid, 98)) if valid.size else (a.min(), a.max())
    n = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    return _resize_save(PS.colorize(n), path)


def attn_png(policy, preproc, sample, rgb, path):
    # roda predict_action_chunk gravando a atencao do action expert sobre a imagem
    batch = PS.make_batch(sample, policy, next(policy.parameters()).device)
    rec = AttnRecorder().install()
    with rec:
        with torch.no_grad():
            policy.predict_action_chunk(preproc(PS.clone(batch)))
    heat = rec.heatmap()
    if heat is None:
        return None
    # crop da banda de padding vertical do resize_with_pad (848x480 -> 224 quadrado):
    # a imagem ocupa a faixa central (H/W) do quadrado; topo/base sao padding preto.
    band = rgb.shape[0] / rgb.shape[1]
    g = heat.shape[0]
    pad = (1.0 - band) / 2.0
    r0, r1 = int(round(g * pad)), int(round(g * (1.0 - pad)))
    heat = heat[max(0, r0):min(g, r1), :]
    over_bgr = overlay_heatmap(rgb, heat, out_width=OUT_W)
    cv2.imwrite(str(path), over_bgr)
    return path.name


def arm_norm(raw_action, stats):
    # normaliza as 7 dims do braco com q01/q99 do dataset -> [-1,1] (igual ao treino)
    q01 = np.asarray(stats["action"]["q01"], dtype=np.float32)[:7]
    q99 = np.asarray(stats["action"]["q99"], dtype=np.float32)[:7]
    raw = np.asarray(raw_action, dtype=np.float32)[:7]
    return np.clip(2.0 * (raw - q01) / (q99 - q01 + 1e-6) - 1.0, -1.2, 1.2).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", default="8dim")
    ap.add_argument("--n-each-frames", type=int, default=2)
    ap.add_argument("--n-agg", type=int, default=12)
    ap.add_argument("--sal-grid", type=int, default=8)
    ap.add_argument("--train-eps", type=int, default=214)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] policy {args.ckpt}")
    policy = PS.PI05Policy.from_pretrained(args.ckpt, strict=False).to(device).eval()
    preproc, _ = PS.make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.ckpt)
    print(f"[load] dataset {args.repo_id}")
    ds = PS.LeRobotDataset(args.repo_id, root=args.root, episodes=list(range(args.train_eps)),
                           video_backend="pyav")
    stats = ds.meta.stats

    acts = np.asarray(ds.hf_dataset.select_columns(["action"])["action"], dtype=np.float32)
    sq_all = np.array([PS.gt_squeeze(a, args.mode) for a in acts])
    open_idx = np.where(sq_all < 0.05)[0]
    closed_idx = np.where(sq_all > 0.95)[0]
    rng = np.random.default_rng(0)
    open_sel = rng.choice(open_idx, min(args.n_agg, len(open_idx)), replace=False)
    closed_sel = rng.choice(closed_idx, min(args.n_agg, len(closed_idx)), replace=False)
    open_imgs = [ds[int(i)][IMG] for i in open_sel]
    closed_imgs = [ds[int(i)][IMG] for i in closed_sel]

    # agregados (sep / sq_img / arm_img) + tabela OPEN/CLOSED real/zero/swap
    rows = []
    for label, sel, pool in (("OPEN", open_sel, closed_imgs), ("CLOSED", closed_sel, open_imgs)):
        for j, i in enumerate(sel):
            sample = ds[int(i)]
            batch = PS.make_batch(sample, policy, device)
            a_real = PS.pred_chunk(policy, preproc, batch)
            bz = PS.clone(batch); bz[IMG] = torch.zeros_like(bz[IMG])
            a_zero = PS.pred_chunk(policy, preproc, bz)
            bs = PS.clone(batch); bs[IMG] = pool[j % len(pool)].to(device).unsqueeze(0).float()
            a_swap = PS.pred_chunk(policy, preproc, bs)
            rows.append(dict(label=label, real=PS.pred_squeeze(a_real, args.mode),
                             zero=PS.pred_squeeze(a_zero, args.mode), swap=PS.pred_squeeze(a_swap, args.mode),
                             arm_zero=float(np.abs(a_zero[:7] - a_real[:7]).mean()),
                             arm_swap=float(np.abs(a_swap[:7] - a_real[:7]).mean())))

    def m(label, key):
        return float(np.mean([r[key] for r in rows if r["label"] == label]))
    sep = m("CLOSED", "real") - m("OPEN", "real")
    sq_img = float(np.mean([abs(r["real"] - r["zero"]) for r in rows])) + \
        float(np.mean([abs(r["real"] - r["swap"]) for r in rows]))
    arm_img = float(np.mean([r["arm_zero"] + r["arm_swap"] for r in rows]))

    # frames detalhados: N OPEN + N CLOSED com RGB/depth/atencao/saliencia + chunk x GT
    frames = []
    nf = args.n_each_frames
    for label, sel, pool in (("CLOSED", closed_sel[:nf], open_imgs), ("OPEN", open_sel[:nf], closed_imgs)):
        for k, i in enumerate(sel):
            i = int(i)
            tag = f"{label.lower()}{k}"
            sample = ds[i]
            rgb = PS.img_to_hwc_uint8(sample[IMG])
            H, W = rgb.shape[:2]
            batch = PS.make_batch(sample, policy, device)
            fill = float(batch[IMG].mean())
            a_real = PS.pred_chunk(policy, preproc, batch)
            bz = PS.clone(batch); bz[IMG] = torch.zeros_like(bz[IMG])
            a_zero = PS.pred_chunk(policy, preproc, bz)
            swap_img = pool[k % len(pool)]
            bs = PS.clone(batch); bs[IMG] = swap_img.to(device).unsqueeze(0).float()
            a_swap = PS.pred_chunk(policy, preproc, bs)
            base, heat = PS.occlusion_map(policy, preproc, batch, args.mode, args.sal_grid, fill)
            rec = dict(
                tag=tag, label=label, idx=i,
                rgb=_resize_save(rgb, outdir / f"rgb_{tag}.png"),
                depth=depth_png(sample, outdir / f"depth_{tag}.png"),
                attn=attn_png(policy, preproc, sample, rgb, outdir / f"attn_{tag}.png"),
                sal=_resize_save(PS.make_overlay(rgb, heat, W, H), outdir / f"sal_{tag}.png"),
                gt_squeeze=PS.gt_squeeze(np.asarray(sample["action"], dtype=np.float32), args.mode),
                pred_real=PS.pred_squeeze(a_real, args.mode),
                pred_zero=PS.pred_squeeze(a_zero, args.mode),
                pred_swap=PS.pred_squeeze(a_swap, args.mode),
                pred_arm=[float(x) for x in a_real[:7]],
                gt_arm=arm_norm(np.asarray(sample["action"], dtype=np.float32), stats),
            )
            frames.append(rec)
            print(f"[frame] {tag} idx={i} gt_sq={rec['gt_squeeze']:.2f} "
                  f"pred real/zero/swap={rec['pred_real']:.2f}/{rec['pred_zero']:.2f}/{rec['pred_swap']:.2f}")

    try:
        _ts = json.loads((Path(args.ckpt).parent / "training_state" / "training_step.json").read_text())
        _step = int(_ts.get("step", 0))
    except Exception:
        _step = 0
    manifest = dict(
        run="y32omum0", run_name="cup_pi05_right8_armstate7_valfix_lf", step=_step,
        ckpt=args.ckpt, mode=args.mode, n_open=int(len(open_idx)), n_closed=int(len(closed_idx)),
        sep=sep, sq_img_sens=sq_img, arm_img_sens=arm_img,
        table=dict(open_real=m("OPEN", "real"), open_zero=m("OPEN", "zero"), open_swap=m("OPEN", "swap"),
                   closed_real=m("CLOSED", "real"), closed_zero=m("CLOSED", "zero"), closed_swap=m("CLOSED", "swap")),
        frames=frames,
    )
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ok] sep={sep:+.3f} sq_img={sq_img:.3f} arm_img={arm_img:.3f} frames={len(frames)}")
    print(f"[ok] {outdir}/manifest.json")


if __name__ == "__main__":
    main()
