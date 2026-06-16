#!/usr/bin/env python3
# Probe focado: o squeeze previsto sai CRAVADO (0/1) ou MORNO (~0.4)? E ele
# segue a IMAGEM ou ignora? Roda em frames REAIS de TREINO (memorizados).
#
# --mode 8dim  : action[7] = squeeze (q01=0/q99=1 => raw = (norm+1)/2, exato)
# --mode 14dim : dedos action[8:14] = squeeze x RIGHT_TARGET. Recupera squeeze
#                do normalizado por dim (sinal depende de RIGHT_TARGET[d]) e media.
#
# predict_action_chunk devolve a acao NORMALIZADA. 3 condicoes de imagem por
# frame: real / zerada / trocada (imagem de um frame de squeeze OPOSTO). Se o
# squeeze previsto NAO muda entre real/zerada/trocada => a politica ignora a
# imagem (grounding quebrado).

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "lerobot-ext"))

from lerobot.policies.factory import make_pre_post_processors  # noqa
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa

IMG = "observation.images.head_camera"
RIGHT_TARGET = np.array([0.0, -0.92, -1.74, 1.57, 1.74, 1.57, 1.74])  # dims 7..13


def clone(b):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}


def make_batch(sample, policy, device, task="Pick up the white cup"):
    out = {}
    for name in policy.config.input_features:
        v = sample[name]
        if not torch.is_tensor(v):
            v = torch.as_tensor(v)
        out[name] = v.to(device).unsqueeze(0).float()
    out["task"] = task
    return out


def pred_chunk(policy, preproc, batch):
    b = preproc(clone(batch))
    with torch.no_grad():
        c = policy.predict_action_chunk(b).detach()
    return c[0, 0].float().cpu().numpy()  # primeira acao do chunk, NORMALIZADA


def gt_squeeze(action_raw, mode):
    if mode == "8dim":
        return float(action_raw[7])
    return float(action_raw[10] / RIGHT_TARGET[3])  # index_0 (T=1.57), limpo positivo


def pred_squeeze(norm_action, mode):
    # recupera squeeze do espaco normalizado (q01/q99 = faixa de squeeze x T)
    if mode == "8dim":
        return (float(norm_action[7]) + 1.0) / 2.0
    vals = []
    for d in range(8, 14):
        T = RIGHT_TARGET[d - 7]
        n = float(norm_action[d])
        vals.append((n + 1.0) / 2.0 if T > 0 else (1.0 - n) / 2.0)
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", choices=["8dim", "14dim"], default="8dim")
    ap.add_argument("--n-each", type=int, default=12)
    ap.add_argument("--train-eps", type=int, default=214)  # eps 0..213 = treino
    ap.add_argument("--out", default="/tmp/probe_grasp.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] policy de {args.ckpt} (mode={args.mode})")
    policy = PI05Policy.from_pretrained(args.ckpt, strict=False).to(device).eval()
    preproc, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.ckpt)

    print(f"[load] dataset {args.repo_id} root={args.root} (treino eps 0..{args.train_eps-1})")
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=list(range(args.train_eps)),
                        video_backend="pyav")
    n = len(ds)

    # frames OPEN/CLOSED lendo a coluna action SEM decodar video
    acts = np.asarray(ds.hf_dataset.select_columns(["action"])["action"], dtype=np.float32)
    assert len(acts) == n, f"hf_dataset {len(acts)} != ds {n}"
    sq_all = np.array([gt_squeeze(a, args.mode) for a in acts])
    open_idx = np.where(sq_all < 0.05)[0]
    closed_idx = np.where(sq_all > 0.95)[0]
    rng = np.random.default_rng(0)
    open_sel = rng.choice(open_idx, min(args.n_each, len(open_idx)), replace=False)
    closed_sel = rng.choice(closed_idx, min(args.n_each, len(closed_idx)), replace=False)
    print(f"[frames] open={len(open_sel)} closed={len(closed_sel)} "
          f"(de {len(open_idx)} open / {len(closed_idx)} closed)")

    open_imgs = [ds[int(i)][IMG] for i in open_sel]
    closed_imgs = [ds[int(i)][IMG] for i in closed_sel]

    rows = []
    for label, sel, swap_pool in (("OPEN", open_sel, closed_imgs),
                                  ("CLOSED", closed_sel, open_imgs)):
        for j, i in enumerate(sel):
            sample = ds[int(i)]
            gt_sq = gt_squeeze(np.asarray(sample["action"], dtype=np.float32), args.mode)
            batch = make_batch(sample, policy, device)
            a_real = pred_chunk(policy, preproc, batch)
            bz = clone(batch); bz[IMG] = torch.zeros_like(bz[IMG])
            a_zero = pred_chunk(policy, preproc, bz)
            bs = clone(batch); bs[IMG] = swap_pool[j % len(swap_pool)].to(device).unsqueeze(0).float()
            a_swap = pred_chunk(policy, preproc, bs)
            rows.append(dict(label=label, idx=int(i), gt_sq=gt_sq,
                             pred_sq_real=pred_squeeze(a_real, args.mode),
                             pred_sq_zero=pred_squeeze(a_zero, args.mode),
                             pred_sq_swap=pred_squeeze(a_swap, args.mode),
                             arm_absdiff_zero=float(np.abs(a_zero[:7] - a_real[:7]).mean()),
                             arm_absdiff_swap=float(np.abs(a_swap[:7] - a_real[:7]).mean())))

    def agg(label, key):
        v = [r[key] for r in rows if r["label"] == label]
        return float(np.mean(v)), float(np.std(v))

    print("\n================ RESULTADO (mode=%s) ================" % args.mode)
    print(f"{'cond':8s} | {'GT sq':>6s} | pred_sq real / zerada / trocada")
    for label in ("OPEN", "CLOSED"):
        print(f"{label:8s} | {agg(label,'gt_sq')[0]:6.3f} | "
              f"real={agg(label,'pred_sq_real')[0]:.3f}±{agg(label,'pred_sq_real')[1]:.3f}  "
              f"zerada={agg(label,'pred_sq_zero')[0]:.3f}  trocada={agg(label,'pred_sq_swap')[0]:.3f}")

    sep = agg("CLOSED", "pred_sq_real")[0] - agg("OPEN", "pred_sq_real")[0]
    sq_img = float(np.mean([abs(r["pred_sq_real"]-r["pred_sq_zero"]) for r in rows])) \
        + float(np.mean([abs(r["pred_sq_real"]-r["pred_sq_swap"]) for r in rows]))
    arm_img = float(np.mean([r["arm_absdiff_zero"]+r["arm_absdiff_swap"] for r in rows]))
    print("\n--- leitura ---")
    print(f"SEPARACAO squeeze (closed-open) = {sep:+.3f}  (perto de 1=CRAVA; perto de 0=MORNO)")
    print(f"sensibilidade SQUEEZE a imagem = {sq_img:.3f}  (perto de 0=IGNORA imagem)")
    print(f"sensibilidade BRACO a imagem (norm) = {arm_img:.3f}")
    Path(args.out).write_text(json.dumps(dict(mode=args.mode, rows=rows, sep=sep,
        sq_img_sens=sq_img, arm_img_sens=arm_img), indent=2))
    print(f"[ok] {args.out}")


if __name__ == "__main__":
    main()
