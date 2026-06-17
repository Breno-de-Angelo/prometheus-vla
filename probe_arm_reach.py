#!/usr/bin/env python3
# Probe do BRACO / reach: o reach depende de ONDE o copo esta na imagem?
# Analogo a probe_grasp_grounding, mas:
#   (1) seleciona frames de REACH (squeeze ABERTO, ~meio do episodio) -> 1 por
#       episodio = copo em posicoes VARIADAS (as 238 demos tem copo espalhado);
#   (2) mede a mudanca do BRACO (7 dims, normalizado) sob imagem real/zerada/trocada;
#   (3) TESTE DE TRACKING: detecta o centroide do copo (branco) na imagem e mede se,
#       ao trocar a imagem por uma com o copo MAIS LONGE, o braco muda MAIS
#       (correlacao copo-desloc x braco-muda + bins perto/longe). Se nao muda -> o
#       reach IGNORA a posicao do copo (segue prior) = a causa do "desce em cima".
#
# predict_action_chunk devolve a acao NORMALIZADA. Mede o 1o passo (commit) e o
# ULTIMO passo do chunk (alvo do reach). Deteccao de copo = numpy puro (sem cv2).

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


def pred_full(policy, preproc, batch):
    """chunk inteiro NORMALIZADO -> (T, A)"""
    b = preproc(clone(batch))
    with torch.no_grad():
        c = policy.predict_action_chunk(b).detach()
    return c[0].float().cpu().numpy()


def gt_squeeze(action_raw, mode):
    if mode == "8dim":
        return float(action_raw[7])
    return float(action_raw[10] / RIGHT_TARGET[3])


def cup_uv(img_tensor):
    """centroide do copo BRANCO (numpy puro): pixel claro + baixa saturacao.
    Retorna (u,v) normalizado [0,1] ou None."""
    a = img_tensor.detach().cpu().numpy()
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    a = a.astype(np.float32)
    if a.max() <= 1.01:
        a = a * 255.0
    mx = a.max(2); mn = a.min(2); sat = mx - mn
    mask = (mx > 160) & (sat < 45)              # claro E baixa saturacao = branco (copo)
    ys, xs = np.where(mask)
    if len(xs) < 40:                            # fallback mais frouxo p/ frames escuros
        mask = (mx > 145) & (sat < 55)
        ys, xs = np.where(mask)
        if len(xs) < 25:
            return None
    H, W = a.shape[0], a.shape[1]
    # MEDIANA resiste a brilho espalhado da mesa (vs media)
    return (float(np.median(xs)) / W, float(np.median(ys)) / H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", choices=["8dim", "14dim"], default="8dim")
    ap.add_argument("--n-eps", type=int, default=36)      # frames de reach (1/episodio)
    ap.add_argument("--swaps-per", type=int, default=4)   # trocas por frame (varia desloc do copo)
    ap.add_argument("--train-eps", type=int, default=214)
    ap.add_argument("--out", default="/tmp/probe_arm_reach.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] policy {args.ckpt} (mode={args.mode})")
    policy = PI05Policy.from_pretrained(args.ckpt, strict=False).to(device).eval()
    preproc, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.ckpt)

    print(f"[load] dataset {args.repo_id} root={args.root}")
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=list(range(args.train_eps)),
                        video_backend="pyav")
    acts = np.asarray(ds.hf_dataset.select_columns(["action"])["action"], dtype=np.float32)
    sq_all = np.array([gt_squeeze(a, args.mode) for a in acts])
    ep_idx = np.asarray(ds.hf_dataset.select_columns(["episode_index"])["episode_index"], dtype=np.int64)

    # 1 frame de REACH por episodio: squeeze aberto, janela 25-55% do episodio
    rng = np.random.default_rng(0)
    sel = []
    for ep in range(args.train_eps):
        fr = np.where(ep_idx == ep)[0]
        if len(fr) < 6:
            continue
        lo, hi = fr[int(0.25 * len(fr))], fr[int(0.55 * len(fr))]
        cand = [int(i) for i in range(lo, hi + 1) if sq_all[i] < 0.1]
        if cand:
            sel.append(cand[len(cand) // 2])
    rng.shuffle(sel)
    sel = sel[:args.n_eps]
    print(f"[frames] {len(sel)} frames de reach (1/episodio, squeeze aberto)")

    # pre-carrega imagens + centroide do copo
    imgs, uvs = {}, {}
    for i in sel:
        im = ds[i][IMG]
        imgs[i] = im
        uvs[i] = cup_uv(im)
    n_cup = sum(1 for i in sel if uvs[i] is not None)
    print(f"[cup] centroide detectado em {n_cup}/{len(sel)} frames")

    rows, pairs = [], []
    for i in sel:
        batch = make_batch(ds[i], policy, device)
        ch_real = pred_full(policy, preproc, batch)
        bz = clone(batch); bz[IMG] = torch.zeros_like(bz[IMG])
        ch_zero = pred_full(policy, preproc, bz)
        # mudanca do braco: 1o passo (commit) e ultimo passo (alvo do reach)
        arm0_zero = float(np.abs(ch_zero[0, :7] - ch_real[0, :7]).mean())
        armE_zero = float(np.abs(ch_zero[-1, :7] - ch_real[-1, :7]).mean())
        sq0_real = (ch_real[0, 7] + 1) / 2 if args.mode == "8dim" else None

        # trocas: varias outras imagens (varia o deslocamento do copo)
        others = [j for j in sel if j != i]
        rng.shuffle(others)
        for j in others[:args.swaps_per]:
            bs = clone(batch); bs[IMG] = imgs[j].to(device).unsqueeze(0).float()
            ch_swap = pred_full(policy, preproc, bs)
            arm0_swap = float(np.abs(ch_swap[0, :7] - ch_real[0, :7]).mean())
            armE_swap = float(np.abs(ch_swap[-1, :7] - ch_real[-1, :7]).mean())
            cup_disp = None
            if uvs[i] is not None and uvs[j] is not None:
                cup_disp = float(np.hypot(uvs[i][0]-uvs[j][0], uvs[i][1]-uvs[j][1]))
            pairs.append(dict(i=i, j=j, cup_disp=cup_disp,
                              arm0_swap=arm0_swap, armE_swap=armE_swap))
        rows.append(dict(idx=i, arm0_zero=arm0_zero, armE_zero=armE_zero,
                         sq0_real=(float(sq0_real) if sq0_real is not None else None),
                         cup_uv=uvs[i]))

    # ---------- metricas ----------
    arm0_zero_m = float(np.mean([r["arm0_zero"] for r in rows]))
    armE_zero_m = float(np.mean([r["armE_zero"] for r in rows]))
    arm0_swap_m = float(np.mean([p["arm0_swap"] for p in pairs]))
    armE_swap_m = float(np.mean([p["armE_swap"] for p in pairs]))
    arm_img_sens0 = arm0_zero_m + arm0_swap_m          # forma identica ao probe da mao (1o passo)
    arm_img_sensE = armE_zero_m + armE_swap_m          # no ALVO do reach (ultimo passo)

    # tracking: o braco muda MAIS quando o copo trocado esta MAIS LONGE?
    P = [p for p in pairs if p["cup_disp"] is not None]
    track_corr0 = track_corrE = float("nan")
    near = far = None
    if len(P) >= 6:
        d = np.array([p["cup_disp"] for p in P])
        a0 = np.array([p["arm0_swap"] for p in P])
        aE = np.array([p["armE_swap"] for p in P])
        track_corr0 = float(np.corrcoef(d, a0)[0, 1])
        track_corrE = float(np.corrcoef(d, aE)[0, 1])
        med = np.median(d)
        near = dict(arm0=float(a0[d <= med].mean()), armE=float(aE[d <= med].mean()), n=int((d <= med).sum()))
        far = dict(arm0=float(a0[d > med].mean()), armE=float(aE[d > med].mean()), n=int((d > med).sum()))

    print("\n================ PROBE BRACO / REACH (mode=%s) ================" % args.mode)
    print(f"frames de reach: {len(rows)} | pares de troca: {len(pairs)} (com copo: {len(P)})")
    print("\n--- sensibilidade do BRACO a imagem (norm; perto de 0 = IGNORA a imagem) ---")
    print(f"  1o passo (commit):  zerada {arm0_zero_m:.3f} + trocada {arm0_swap_m:.3f} = {arm_img_sens0:.3f}")
    print(f"  ULTIMO passo (alvo): zerada {armE_zero_m:.3f} + trocada {armE_swap_m:.3f} = {arm_img_sensE:.3f}")
    print("\n--- TRACKING: trocar p/ copo MAIS LONGE faz o braco mudar MAIS? ---")
    print(f"  correlacao(desloc_copo, muda_braco): 1o passo {track_corr0:+.3f} | alvo {track_corrE:+.3f}")
    if near and far:
        print(f"  copo trocado PERTO (n={near['n']}): braco muda {near['armE']:.3f} (alvo)")
        print(f"  copo trocado LONGE (n={far['n']}): braco muda {far['armE']:.3f} (alvo)")
        ratio = far['armE'] / max(1e-6, near['armE'])
        print(f"  razao longe/perto = {ratio:.2f}x  ({'TRACKA' if ratio>1.3 else 'NAO tracka (segue prior)'} a posicao do copo)")
    Path(args.out).write_text(json.dumps(dict(
        mode=args.mode, n_frames=len(rows), n_pairs=len(pairs), n_cup_pairs=len(P),
        arm_img_sens_commit=arm_img_sens0, arm_img_sens_target=arm_img_sensE,
        arm0_zero=arm0_zero_m, arm0_swap=arm0_swap_m, armE_zero=armE_zero_m, armE_swap=armE_swap_m,
        track_corr_commit=track_corr0, track_corr_target=track_corrE,
        near=near, far=far, rows=rows, pairs=pairs), indent=2, default=str))
    print(f"\n[ok] {args.out}")


if __name__ == "__main__":
    main()
