#!/usr/bin/env python3
# Linear probe de GROUNDING (features visuais CONGELADAS do pi05_base).
# Decide a rota do "run4" (grounding loss): se o cue está nas features congeladas ->
# head no expert basta (VLM congelado, barato); se não -> destravar (KI, caro).
#
# Extração: monkey-patch em embed_image (=paligemma get_image_features) -> tokens de
# imagem [1, n_tok, dim], mean-pool -> vetor por frame. Decodificador LINEAR (logreg
# torch, L2), held-out: FIT em eps de TREINO, TESTE em eps de VAL. Baseline = propriocepção.
#
# Probe B (DECISÃO, o que vale): entre frames AINDA ABERTOS, "prestes a fechar"
#   (janela [t_close-K, t_close)) vs "longe de fechar" (antes de t_close-2K). Ambos abertos
#   -> remove o atalho trivial "dedos visivelmente fechados".
# Probe A (sanidade): open vs closed (provavelmente trivial; AUC baixo seria forte sinal).
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "lerobot-ext"))
from lerobot.policies.factory import make_pre_post_processors          # noqa
from lerobot.policies.pi05.modeling_pi05 import PI05Policy             # noqa
from lerobot.datasets.lerobot_dataset import LeRobotDataset           # noqa

IMG = "observation.images.head_camera"
STATE = "observation.state"


def clone(b): return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}


def make_batch(sample, policy, device, task="Pick up the white cup"):
    out = {}
    for name in policy.config.input_features:
        v = sample[name]
        if not torch.is_tensor(v): v = torch.as_tensor(v)
        out[name] = v.to(device).unsqueeze(0).float()
    out["task"] = task
    return out


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0: return float("nan")
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def logreg(Xtr, ytr, Xte, wd=1e-2, iters=600, lr=0.05):
    Xtr = torch.tensor(Xtr, dtype=torch.float32); ytr = torch.tensor(ytr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    lin = torch.nn.Linear(Xtr.shape[1], 1); opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    lf = torch.nn.BCEWithLogitsLoss()
    for _ in range(iters):
        opt.zero_grad(); lf(lin(Xtr).squeeze(1), ytr).backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(lin(Xte).squeeze(1)).numpy()


def gt_sq(a): return float(a[7])  # 8dim: action[7] = squeeze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--train-eps", type=int, default=214)
    ap.add_argument("--K", type=int, default=15, help="janela 'prestes a fechar' (frames)")
    ap.add_argument("--cap", type=int, default=400, help="max por classe no fit")
    ap.add_argument("--out", default="/tmp/probe_linear.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] policy {args.ckpt}")
    policy = PI05Policy.from_pretrained(args.ckpt, strict=False).to(device).eval()
    preproc, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.ckpt)

    pwe = policy.model.paligemma_with_expert
    _orig = pwe.embed_image; cap = {}
    def wrap(img):
        out = _orig(img); cap["f"] = out.detach().float(); return out
    pwe.embed_image = wrap

    info = json.loads((Path(args.root) / "meta" / "info.json").read_text())
    n_total = info["total_episodes"]
    print(f"[info] total_eps={n_total}  treino=0..{args.train_eps-1}  val={args.train_eps}..{n_total-1}")

    def labels_for(ds):
        """Devolve dict de listas de índices globais p/ probe A e B."""
        acts = np.asarray(ds.hf_dataset.select_columns(["action"])["action"], dtype=np.float32)
        epi = np.asarray(ds.hf_dataset.select_columns(["episode_index"])["episode_index"])
        sq = np.array([gt_sq(a) for a in acts])
        A_open, A_closed, B_pre, B_far = [], [], [], []
        K = args.K
        for e in np.unique(epi):
            g = np.where(epi == e)[0]; sqe = sq[g]
            A_open += list(g[sqe < 0.05]); A_closed += list(g[sqe > 0.95])
            cl = np.where(sqe > 0.5)[0]
            if len(cl) == 0: continue
            tc = cl[0]
            B_pre += [int(g[j]) for j in range(max(0, tc - K), tc) if sqe[j] < 0.5]
            B_far += [int(g[j]) for j in range(0, max(0, tc - 2 * K)) if sqe[j] < 0.05]
        return dict(A_open=A_open, A_closed=A_closed, B_pre=B_pre, B_far=B_far)

    def feats_for(ds, idxs):
        F, S = [], []
        for i in idxs:
            s = ds[int(i)]; b = make_batch(s, policy, device)
            with torch.no_grad():
                _ = policy.predict_action_chunk(preproc(clone(b)))
            F.append(cap["f"][0].mean(0).cpu().numpy())
            S.append(np.asarray(s[STATE], dtype=np.float32))
        return np.array(F), np.array(S)

    def balance(a, b, capn, rng):
        a = np.array(a); b = np.array(b); m = min(len(a), len(b), capn)
        return (rng.choice(a, m, replace=False), rng.choice(b, m, replace=False), m)

    rng = np.random.default_rng(0)
    print("[load] dataset treino + val")
    ds_tr = LeRobotDataset(args.repo_id, root=args.root, episodes=list(range(args.train_eps)), video_backend="pyav")
    ds_va = LeRobotDataset(args.repo_id, root=args.root, episodes=list(range(args.train_eps, n_total)), video_backend="pyav")
    L_tr = labels_for(ds_tr); L_va = labels_for(ds_va)
    print(f"[frames] treino: A_open={len(L_tr['A_open'])} A_closed={len(L_tr['A_closed'])} "
          f"B_pre={len(L_tr['B_pre'])} B_far={len(L_tr['B_far'])}")
    print(f"[frames] val:    A_open={len(L_va['A_open'])} A_closed={len(L_va['A_closed'])} "
          f"B_pre={len(L_va['B_pre'])} B_far={len(L_va['B_far'])}")

    def stdz(Xtr, Xte):
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6; return (Xtr - mu) / sd, (Xte - mu) / sd
    bacc = lambda y, s: float(((s > 0.5).astype(int) == y).mean())

    def run_probe(name, pos_key, neg_key, capn):
        ptr, ntr, mtr = balance(L_tr[pos_key], L_tr[neg_key], capn, rng)
        pte, nte, mte = balance(L_va[pos_key], L_va[neg_key], max(60, capn // 3), rng)
        if mtr < 10 or mte < 5:
            print(f"\n[{name}] amostras insuficientes (fit={mtr} test={mte}) — pulado"); return None
        Fi_tr, St_tr = feats_for(ds_tr, np.concatenate([ptr, ntr]))
        Fi_te, St_te = feats_for(ds_va, np.concatenate([pte, nte]))
        ytr = np.concatenate([np.ones(mtr), np.zeros(mtr)]); yte = np.concatenate([np.ones(mte), np.zeros(mte)])
        Zf, Zt = stdz(Fi_tr, Fi_te); s_img = logreg(Zf, ytr, Zt, wd=1e-2)
        Zf2, Zt2 = stdz(St_tr, St_te); s_st = logreg(Zf2, ytr, Zt2, wd=1e-3)
        ai, as_ = auc(yte, s_img), auc(yte, s_st)
        print(f"\n[{name}]  fit={mtr}/cl  test={mte}/cl")
        print(f"  IMAGEM (features congeladas): AUC={ai:.3f}  bal_acc={bacc(yte,s_img):.3f}")
        print(f"  PROPRIOCEPÇÃO (baseline):     AUC={as_:.3f}  bal_acc={bacc(yte,s_st):.3f}   (chance=0.5)")
        return dict(auc_img=ai, auc_state=as_, n_fit=int(mtr), n_test=int(mte))

    print("\n==================== LINEAR PROBE — features CONGELADAS ====================")
    rB = run_probe("PROBE B — DECISÃO (prestes-a-fechar vs longe, AMBOS abertos)", "B_pre", "B_far", args.cap)
    rA = run_probe("PROBE A — sanidade (open vs closed)", "A_closed", "A_open", args.cap)

    print("\n--- leitura ---")
    if rB:
        if rB["auc_img"] >= 0.70:
            print(f"  Probe B: IMAGEM separa 'prestes a fechar' (AUC {rB['auc_img']:.2f}) -> o cue de DECISÃO")
            print(f"           ESTÁ nas features congeladas -> rota A (head no expert, VLM congelado) basta.")
        elif rB["auc_img"] <= 0.58:
            print(f"  Probe B: IMAGEM ~cega pra decisão (AUC {rB['auc_img']:.2f}) -> features não têm o cue")
            print(f"           -> precisa DESTRAVAR (KI). Compare com propriocepção={rB['auc_state']:.2f}.")
        else:
            print(f"  Probe B: sinal fraco/ambíguo (AUC {rB['auc_img']:.2f}) — ver vs propriocepção {rB['auc_state']:.2f}.")
    Path(args.out).write_text(json.dumps(dict(probeB=rB, probeA=rA, K=args.K), indent=2))
    print(f"[ok] {args.out}")


if __name__ == "__main__":
    main()
