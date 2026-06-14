#!/usr/bin/env python3
# Probe de REACHING: o braço previsto SEGUE a imagem (servo visual) ou ignora
# (trajetoria proprioceptiva/open-loop)? E ele ALCANCA certo (predito x GT)?
# Usa a fase do aperto como proxy da fase de reaching: OPEN = aproximando,
# CLOSED = no copo. Compara com a sensibilidade do squeeze (que ja usa muito a imagem).
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lerobot-ext"))

import probe_saliency as PS   # make_batch, pred_chunk, clone, gt_squeeze, pred_squeeze, PI05Policy...

IMG = "observation.images.head_camera"


def arm_norm(raw, stats):
    q01 = np.asarray(stats["action"]["q01"], dtype=np.float32)[:7]
    q99 = np.asarray(stats["action"]["q99"], dtype=np.float32)[:7]
    raw = np.asarray(raw, dtype=np.float32)[:7]
    return np.clip(2.0 * (raw - q01) / (q99 - q01 + 1e-6) - 1.0, -1.5, 1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", default="8dim")
    ap.add_argument("--n-each", type=int, default=20)
    ap.add_argument("--train-eps", type=int, default=214)
    ap.add_argument("--out", default="/tmp/probe_arm_reaching.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] policy {args.ckpt}")
    policy = PS.PI05Policy.from_pretrained(args.ckpt, strict=False).to(device).eval()
    preproc, _ = PS.make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.ckpt)
    print(f"[load] dataset {args.repo_id}")
    ds = PS.LeRobotDataset(args.repo_id, root=args.root, episodes=list(range(args.train_eps)),
                           video_backend="pyav")
    stats = ds.meta.stats

    acts = np.asarray(ds.hf_dataset.select_columns(["action"])["action"], dtype=np.float32)
    sq = np.array([PS.gt_squeeze(a, args.mode) for a in acts])
    open_idx = np.where(sq < 0.05)[0]
    closed_idx = np.where(sq > 0.95)[0]
    rng = np.random.default_rng(0)
    open_sel = rng.choice(open_idx, min(args.n_each, len(open_idx)), replace=False)
    closed_sel = rng.choice(closed_idx, min(args.n_each, len(closed_idx)), replace=False)
    # pool de troca: frames de fase OPOSTA (config de braço/copo diferente)
    open_imgs = [ds[int(i)][IMG] for i in open_sel]
    closed_imgs = [ds[int(i)][IMG] for i in closed_sel]

    res = {}
    for label, sel, pool in (("OPEN", open_sel, closed_imgs), ("CLOSED", closed_sel, open_imgs)):
        arm_zero, arm_swap, arm_err, sq_zero, sq_swap = [], [], [], [], []
        for j, i in enumerate(sel):
            sample = ds[int(i)]
            batch = PS.make_batch(sample, policy, device)
            a_real = PS.pred_chunk(policy, preproc, batch)
            bz = PS.clone(batch); bz[IMG] = torch.zeros_like(bz[IMG])
            a_zero = PS.pred_chunk(policy, preproc, bz)
            bs = PS.clone(batch); bs[IMG] = pool[j % len(pool)].to(device).unsqueeze(0).float()
            a_swap = PS.pred_chunk(policy, preproc, bs)
            arm_zero.append(float(np.abs(a_zero[:7] - a_real[:7]).mean()))
            arm_swap.append(float(np.abs(a_swap[:7] - a_real[:7]).mean()))
            gt_arm = arm_norm(np.asarray(sample["action"], dtype=np.float32), stats)
            arm_err.append(float(np.abs(a_real[:7] - gt_arm).mean()))
            sq_zero.append(abs(PS.pred_squeeze(a_real, args.mode) - PS.pred_squeeze(a_zero, args.mode)))
            sq_swap.append(abs(PS.pred_squeeze(a_real, args.mode) - PS.pred_squeeze(a_swap, args.mode)))
        res[label] = dict(
            n=len(sel),
            arm_img_sens=float(np.mean(arm_zero) + np.mean(arm_swap)),
            arm_reach_err=float(np.mean(arm_err)),
            squeeze_img_sens=float(np.mean(sq_zero) + np.mean(sq_swap)),
        )

    arm_all = float(np.mean([res[l]["arm_img_sens"] for l in res]))
    sq_all = float(np.mean([res[l]["squeeze_img_sens"] for l in res]))
    err_all = float(np.mean([res[l]["arm_reach_err"] for l in res]))

    print("\n================ REACHING PROBE ================")
    print(f"{'fase':10s} | {'braço→img':>10s} | {'squeeze→img':>12s} | {'erro alcance':>13s}")
    for l in ("OPEN", "CLOSED"):
        r = res[l]
        ph = "aproximando" if l == "OPEN" else "no copo"
        print(f"{l:10s} | {r['arm_img_sens']:10.3f} | {r['squeeze_img_sens']:12.3f} | "
              f"{r['arm_reach_err']:13.3f}   ({ph})")
    print(f"{'OVERALL':10s} | {arm_all:10.3f} | {sq_all:12.3f} | {err_all:13.3f}")
    print("\n--- leitura ---")
    ratio = sq_all / (arm_all + 1e-6)
    print(f"o squeeze usa a imagem ~{ratio:.1f}x mais que o braço ({sq_all:.3f} vs {arm_all:.3f})")
    print(f"erro de alcance do braço (predito x GT, norm) = {err_all:.3f} "
          f"(baixo = alcança certo na distribuição de treino)")
    verdict = ("o braço IGNORA a imagem (open-loop/proprioceptivo)" if arm_all < 0.2
               else "o braço usa a imagem de forma relevante")
    print(f"VEREDITO: {verdict}")
    Path(args.out).write_text(json.dumps(dict(
        per_phase=res, arm_img_sens=arm_all, squeeze_img_sens=sq_all,
        arm_reach_err=err_all, ratio_sq_over_arm=ratio), indent=2))
    print(f"[ok] {args.out}")


if __name__ == "__main__":
    main()
