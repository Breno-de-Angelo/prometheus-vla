#!/usr/bin/env python3
# Probe de SALIENCIA por OCLUSAO + camera RGB + forca do aperto, pra VISUALIZAR
# onde (e quanto) a politica usa a imagem pra decidir o SQUEEZE (fechar a mao).
#
# Reaproveita o load/predict do probe_grasp_grounding.py. Pra um frame CLOSED e um
# OPEN, desliza um quadrado cinza pela imagem (grid GxG): em cada posicao re-preve
# o squeeze e mede |Δsqueeze|. Mapa de calor = "onde a imagem decide o fechar".
# - Baseline que IGNORA a imagem -> mapa fraco/chapado (Δ pequeno em qualquer lugar).
# - Politica aterrada -> ponto quente no copo/mao (Δ grande ali).
# Salva: rgb_*.png, sal_*.png (overlay) e manifest.json (numeros + picos + sensib.).

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "lerobot-ext"))

from lerobot.policies.factory import make_pre_post_processors  # noqa
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa

IMG = "observation.images.head_camera"
RIGHT_TARGET = np.array([0.0, -0.92, -1.74, 1.57, 1.74, 1.57, 1.74])


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


def pred_chunk(policy, preproc, batch, seed=0):
    # FIXA o ruido do amostrador de flow-matching: assim o unico que varia entre
    # oclusoes e a IMAGEM, nao o ruido. Sem isso o "pico" da oclusao vira ruido do
    # sampler (num modelo cego, ocluir qualquer patch da ~0 com ruido fixo).
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    b = preproc(clone(batch))
    with torch.no_grad():
        c = policy.predict_action_chunk(b).detach()
    return c[0, 0].float().cpu().numpy()


def gt_squeeze(a, mode):
    return float(a[7]) if mode == "8dim" else float(a[10] / RIGHT_TARGET[3])


def pred_squeeze(n, mode):
    if mode == "8dim":
        return (float(n[7]) + 1.0) / 2.0
    vals = []
    for d in range(8, 14):
        T = RIGHT_TARGET[d - 7]
        v = float(n[d])
        vals.append((v + 1.0) / 2.0 if T > 0 else (1.0 - v) / 2.0)
    return float(np.mean(vals))


def img_to_hwc_uint8(t):
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    a = np.squeeze(a)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, -1)
    if a.dtype != np.uint8:
        a = a * 255.0 if float(np.nanmax(a)) <= 1.5 else a
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def colorize(norm):  # HxW [0,1] -> HxWx3 uint8 (turbo-ish)
    x = np.clip(norm, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def occlusion_map(policy, preproc, batch, mode, grid, fill, source=None):
    # source=None -> oclusao CINZA (remove conteudo). source=tensor (1,3,H,W) ->
    # oclusao por TROCA: injeta o conteudo de `source` naquele bloco (decompoe o
    # efeito de trocar a imagem inteira -> localiza a regiao que vira a decisao).
    base = pred_squeeze(pred_chunk(policy, preproc, batch), mode)
    _, _, H, W = batch[IMG].shape
    heat = np.zeros((grid, grid), np.float32)
    ys = np.linspace(0, H, grid + 1).astype(int)
    xs = np.linspace(0, W, grid + 1).astype(int)
    for r in range(grid):
        for c in range(grid):
            b = clone(batch)
            sl = (slice(None), slice(None), slice(ys[r], ys[r + 1]), slice(xs[c], xs[c + 1]))
            if source is None:
                b[IMG][sl] = fill
            else:
                b[IMG][sl] = source[sl]
            sq = pred_squeeze(pred_chunk(policy, preproc, b), mode)
            heat[r, c] = abs(sq - base)
    return base, heat


def make_overlay(rgb, heat, W, H):
    hn = heat / (heat.max() + 1e-8)
    big = np.asarray(Image.fromarray((hn * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.0
    color = colorize(big)
    alpha = (0.62 * big)[..., None]
    return (rgb * (1 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)


def save_frame(outdir, tag, sample, policy, preproc, mode, grid, device,
               swap_src=None, swap_grid=8):
    batch = make_batch(sample, policy, device)
    rgb = img_to_hwc_uint8(sample[IMG])
    H, W = rgb.shape[:2]
    fill = float(batch[IMG].mean())
    base, heat = occlusion_map(policy, preproc, batch, mode, grid, fill)
    Image.fromarray(rgb).save(outdir / f"rgb_{tag}.png")
    Image.fromarray(make_overlay(rgb, heat, W, H)).save(outdir / f"sal_{tag}.png")
    out = dict(tag=tag, base_squeeze=base, peak_delta=float(heat.max()),
               heat_mean=float(heat.mean()), heat_p95=float(np.percentile(heat, 95)),
               rgb=f"rgb_{tag}.png", sal=f"sal_{tag}.png", H=H, W=W)
    if swap_src is not None:
        src = swap_src.to(device).unsqueeze(0).float()
        _, sheat = occlusion_map(policy, preproc, batch, mode, swap_grid, fill, source=src)
        Image.fromarray(make_overlay(rgb, sheat, W, H)).save(outdir / f"salswap_{tag}.png")
        out.update(salswap=f"salswap_{tag}.png", swap_grid=swap_grid,
                   swap_peak=float(sheat.max()), swap_mean=float(sheat.mean()),
                   swap_p95=float(np.percentile(sheat, 95)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", choices=["8dim", "14dim"], default="8dim")
    ap.add_argument("--grid", type=int, default=14)
    ap.add_argument("--swap-grid", type=int, default=8)
    ap.add_argument("--n-each", type=int, default=12)
    ap.add_argument("--train-eps", type=int, default=214)
    ap.add_argument("--label", default="run")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] policy {args.ckpt} (mode={args.mode})")
    policy = PI05Policy.from_pretrained(args.ckpt, strict=False).to(device).eval()
    preproc, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.ckpt)
    print(f"[load] dataset {args.repo_id} root={args.root}")
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=list(range(args.train_eps)),
                        video_backend="pyav")

    acts = np.asarray(ds.hf_dataset.select_columns(["action"])["action"], dtype=np.float32)
    sq_all = np.array([gt_squeeze(a, args.mode) for a in acts])
    open_idx = np.where(sq_all < 0.05)[0]
    closed_idx = np.where(sq_all > 0.95)[0]
    rng = np.random.default_rng(0)
    open_sel = rng.choice(open_idx, min(args.n_each, len(open_idx)), replace=False)
    closed_sel = rng.choice(closed_idx, min(args.n_each, len(closed_idx)), replace=False)

    # sensibilidades agregadas (igual ao probe de grounding) p/ painel "quanto usa"
    rows = []
    open_imgs = [ds[int(i)][IMG] for i in open_sel]
    closed_imgs = [ds[int(i)][IMG] for i in closed_sel]
    for label, sel, swap_pool in (("OPEN", open_sel, closed_imgs),
                                  ("CLOSED", closed_sel, open_imgs)):
        for j, i in enumerate(sel):
            sample = ds[int(i)]
            batch = make_batch(sample, policy, device)
            a_real = pred_chunk(policy, preproc, batch)
            bz = clone(batch); bz[IMG] = torch.zeros_like(bz[IMG])
            a_zero = pred_chunk(policy, preproc, bz)
            bs = clone(batch); bs[IMG] = swap_pool[j % len(swap_pool)].to(device).unsqueeze(0).float()
            a_swap = pred_chunk(policy, preproc, bs)
            rows.append(dict(label=label,
                             real=pred_squeeze(a_real, args.mode),
                             zero=pred_squeeze(a_zero, args.mode),
                             swap=pred_squeeze(a_swap, args.mode),
                             arm_zero=float(np.abs(a_zero[:7] - a_real[:7]).mean()),
                             arm_swap=float(np.abs(a_swap[:7] - a_real[:7]).mean())))

    def mean(label, key):
        return float(np.mean([r[key] for r in rows if r["label"] == label]))
    sep = mean("CLOSED", "real") - mean("OPEN", "real")
    sq_img = float(np.mean([abs(r["real"] - r["zero"]) for r in rows])) \
        + float(np.mean([abs(r["real"] - r["swap"]) for r in rows]))
    arm_img = float(np.mean([r["arm_zero"] + r["arm_swap"] for r in rows]))

    # saliencia em 1 frame CLOSED + 1 OPEN representativos (indices deterministicos)
    closed_i, open_i = int(closed_sel[0]), int(open_sel[0])
    closed_sample, open_sample = ds[closed_i], ds[open_i]
    # swap_src cruzado: no CLOSED injeta a imagem do OPEN (e vice-versa) bloco a bloco
    frames = [
        save_frame(outdir, "closed", closed_sample, policy, preproc, args.mode, args.grid,
                   device, swap_src=open_sample[IMG], swap_grid=args.swap_grid),
        save_frame(outdir, "open", open_sample, policy, preproc, args.mode, args.grid,
                   device, swap_src=closed_sample[IMG], swap_grid=args.swap_grid),
    ]
    for f in frames:
        f["idx"] = closed_i if f["tag"] == "closed" else open_i

    manifest = dict(label=args.label, ckpt=args.ckpt, mode=args.mode, grid=args.grid,
                    closed_real=mean("CLOSED", "real"), closed_zero=mean("CLOSED", "zero"),
                    closed_swap=mean("CLOSED", "swap"),
                    open_real=mean("OPEN", "real"), open_zero=mean("OPEN", "zero"),
                    open_swap=mean("OPEN", "swap"),
                    sep=sep, sq_img_sens=sq_img, arm_img_sens=arm_img, frames=frames)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ok] sep={sep:+.3f} sq_img={sq_img:.3f} arm_img={arm_img:.3f} "
          f"gray_peak_closed={frames[0]['peak_delta']:.3f} "
          f"swap_peak_closed={frames[0].get('swap_peak', -1):.3f} "
          f"swap_mean_closed={frames[0].get('swap_mean', -1):.4f}")
    print(f"[ok] {outdir}/manifest.json")


if __name__ == "__main__":
    main()
