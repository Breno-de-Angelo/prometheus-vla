#!/usr/bin/env python3
# Probe (a) da revisão: STATE-SWAP no BRAÇO.
# Pergunta do professor: "o braço segue mais o state do que a imagem?"
# Mesma imagem, troco o `observation.state` por um de FASE OPOSTA (OPEN<->CLOSED,
# poses de braço bem diferentes) e meço o quanto a AÇÃO DO BRAÇO (a[:7]) muda.
# No MESMO run e com o MESMO seed do sampler, também troco/zero a IMAGEM -> assim
# comparo braço→state vs braço→imagem de forma pareada (a única coisa que muda é o
# que eu perturbo). Se braço→state >> braço→imagem, o braço segue o state (atalho
# proprioceptivo) e ignora a câmera — fecha o diagnóstico de causal confusion.
#
# Reaproveita o load/predict de probe_saliency.py (PS), igual ao probe_arm_reaching.py.
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "lerobot-ext"))

import probe_saliency as PS  # make_batch, pred_chunk, clone, gt_squeeze, PI05Policy, ...

IMG = "observation.images.head_camera"
STATE = "observation.state"


def as_state_tensor(v, device):
    t = v.clone().detach() if torch.is_tensor(v) else torch.as_tensor(np.asarray(v, dtype=np.float32))
    return t.to(device).unsqueeze(0).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", default="8dim")
    ap.add_argument("--n-each", type=int, default=20)
    ap.add_argument("--train-eps", type=int, default=214)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="/tmp/probe_arm_state_swap.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] policy {args.ckpt}")
    policy = PS.PI05Policy.from_pretrained(args.ckpt, strict=False).to(device).eval()
    preproc, _ = PS.make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.ckpt)
    print(f"[load] dataset {args.repo_id}")
    ds = PS.LeRobotDataset(args.repo_id, root=args.root, episodes=list(range(args.train_eps)),
                           video_backend="pyav")

    acts = np.asarray(ds.hf_dataset.select_columns(["action"])["action"], dtype=np.float32)
    sq = np.array([PS.gt_squeeze(a, args.mode) for a in acts])
    open_idx = np.where(sq < 0.05)[0]
    closed_idx = np.where(sq > 0.95)[0]
    rng = np.random.default_rng(0)
    open_sel = rng.choice(open_idx, min(args.n_each, len(open_idx)), replace=False)
    closed_sel = rng.choice(closed_idx, min(args.n_each, len(closed_idx)), replace=False)
    # pools de FASE OPOSTA: states e imagens
    open_states = [ds[int(i)][STATE] for i in open_sel]
    closed_states = [ds[int(i)][STATE] for i in closed_sel]
    open_imgs = [ds[int(i)][IMG] for i in open_sel]
    closed_imgs = [ds[int(i)][IMG] for i in closed_sel]

    res = {}
    for label, sel, st_pool, im_pool in (("OPEN", open_sel, closed_states, closed_imgs),
                                         ("CLOSED", closed_sel, open_states, open_imgs)):
        arm_state, arm_img_swap, arm_img_zero, dstate = [], [], [], []
        for j, i in enumerate(sel):
            sample = ds[int(i)]
            batch = PS.make_batch(sample, policy, device)
            a_real = PS.pred_chunk(policy, preproc, batch)

            # --- STATE-SWAP: mesma imagem, state de fase oposta ---
            new_state = st_pool[j % len(st_pool)]
            bss = PS.clone(batch); bss[STATE] = as_state_tensor(new_state, device)
            a_sswap = PS.pred_chunk(policy, preproc, bss)
            arm_state.append(float(np.abs(a_sswap[:7] - a_real[:7]).mean()))
            dstate.append(float(np.abs(np.asarray(sample[STATE], dtype=np.float32)[:7]
                                       - np.asarray(new_state, dtype=np.float32)[:7]).mean()))

            # --- IMAGE-SWAP / ZERO: mesmo state, imagem perturbada (pareado) ---
            bis = PS.clone(batch); bis[IMG] = im_pool[j % len(im_pool)].to(device).unsqueeze(0).float()
            a_iswap = PS.pred_chunk(policy, preproc, bis)
            arm_img_swap.append(float(np.abs(a_iswap[:7] - a_real[:7]).mean()))
            bz = PS.clone(batch); bz[IMG] = torch.zeros_like(bz[IMG])
            a_zero = PS.pred_chunk(policy, preproc, bz)
            arm_img_zero.append(float(np.abs(a_zero[:7] - a_real[:7]).mean()))

        res[label] = dict(
            n=len(sel),
            arm_state_swap=float(np.mean(arm_state)),                       # braço muda ao trocar o STATE
            arm_img_swap=float(np.mean(arm_img_swap)),                      # ...ao trocar a IMAGEM (pareado)
            arm_img_sens=float(np.mean(arm_img_zero) + np.mean(arm_img_swap)),  # zero+swap (= métrica antiga)
            state_delta=float(np.mean(dstate)),
        )

    st_all = float(np.mean([res[l]["arm_state_swap"] for l in res]))
    imsw_all = float(np.mean([res[l]["arm_img_swap"] for l in res]))
    imzs_all = float(np.mean([res[l]["arm_img_sens"] for l in res]))
    ratio = st_all / (imsw_all + 1e-6)

    print("\n============== STATE-SWAP PROBE (braço) ==============")
    print(f"{'fase':9s} | {'braço→state':>11s} | {'braço→img(sw)':>13s} | {'braço→img(z+s)':>14s} | {'Δstate':>7s}")
    for l in ("OPEN", "CLOSED"):
        r = res[l]
        print(f"{l:9s} | {r['arm_state_swap']:11.3f} | {r['arm_img_swap']:13.3f} | "
              f"{r['arm_img_sens']:14.3f} | {r['state_delta']:7.3f}")
    print(f"{'OVERALL':9s} | {st_all:11.3f} | {imsw_all:13.3f} | {imzs_all:14.3f} |")
    print("\n--- leitura ---")
    print(f"o braço muda ~{ratio:.1f}x MAIS ao trocar o STATE do que ao trocar a IMAGEM "
          f"({st_all:.3f} vs {imsw_all:.3f})")
    verdict = ("o braço SEGUE o STATE e ignora a imagem (atalho proprioceptivo CONFIRMADO)"
               if ratio >= 3.0 else
               "o braço usa state e imagem de forma comparável" if ratio >= 1.0 else
               "o braço usa mais a IMAGEM que o state")
    print(f"VEREDITO ({args.label}): {verdict}")
    Path(args.out).write_text(json.dumps(dict(
        label=args.label, ckpt=args.ckpt, per_phase=res,
        arm_state_swap=st_all, arm_img_swap=imsw_all, arm_img_sens=imzs_all,
        ratio_state_over_img=ratio, verdict=verdict), indent=2))
    print(f"[ok] {args.out}")


if __name__ == "__main__":
    main()
