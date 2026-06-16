#!/usr/bin/env python3
# Complemento do probe de grounding: ablação da PROPRIOCEPÇÃO (observation.state).
# Pergunta: o que carrega a decisão de fechar (squeeze)? A imagem já foi descartada
# (probe anterior). Aqui testamos o state, com IMAGEM SEMPRE REAL:
#   - baseline        : state real
#   - state_zero      : zera o state inteiro [14]
#   - state_swap      : state de um frame do bucket OPOSTO (in-distribution)
#   - arm_zero        : zera só o braço (dims 0-6), mantém dedos medidos
#   - fingers_zero    : zera só os dedos medidos (dims 7-13), mantém braço
# Se a SEPARAÇÃO do squeeze (closed-open) COLAPSA numa condição -> esse input
# é o que carrega o "abrir/fechar". Separa pose do braço de autocorrelação dos dedos.

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
STATE = "observation.state"
RIGHT_TARGET = np.array([0.0, -0.92, -1.74, 1.57, 1.74, 1.57, 1.74])


def clone(b): return {k:(v.clone() if torch.is_tensor(v) else v) for k,v in b.items()}

def make_batch(sample, policy, device, task="Pick up the white cup"):
    out={}
    for name in policy.config.input_features:
        v=sample[name]
        if not torch.is_tensor(v): v=torch.as_tensor(v)
        out[name]=v.to(device).unsqueeze(0).float()
    out["task"]=task
    return out

def pred_chunk(policy, preproc, batch):
    b=preproc(clone(batch))
    with torch.no_grad():
        return policy.predict_action_chunk(b).detach()[0,0].float().cpu().numpy()

def gt_squeeze(a, mode):
    return float(a[7]) if mode=="8dim" else float(a[10]/RIGHT_TARGET[3])

def pred_squeeze(n, mode):
    if mode=="8dim": return (float(n[7])+1.0)/2.0
    vals=[]
    for d in range(8,14):
        T=RIGHT_TARGET[d-7]; x=float(n[d])
        vals.append((x+1.0)/2.0 if T>0 else (1.0-x)/2.0)
    return float(np.mean(vals))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True); ap.add_argument("--mode", choices=["8dim","14dim"], default="8dim")
    ap.add_argument("--n-each", type=int, default=12); ap.add_argument("--train-eps", type=int, default=214)
    ap.add_argument("--out", default="/tmp/probe_proprio.json")
    args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load] policy {args.ckpt} (mode={args.mode})")
    policy=PI05Policy.from_pretrained(args.ckpt, strict=False).to(device).eval()
    preproc,_=make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.ckpt)
    print(f"[load] dataset {args.repo_id} root={args.root}")
    ds=LeRobotDataset(args.repo_id, root=args.root, episodes=list(range(args.train_eps)), video_backend="pyav")
    n=len(ds)
    acts=np.asarray(ds.hf_dataset.select_columns(["action"])["action"], dtype=np.float32)
    sq_all=np.array([gt_squeeze(a,args.mode) for a in acts])
    open_idx=np.where(sq_all<0.05)[0]; closed_idx=np.where(sq_all>0.95)[0]
    rng=np.random.default_rng(0)
    open_sel=rng.choice(open_idx, min(args.n_each,len(open_idx)), replace=False)
    closed_sel=rng.choice(closed_idx, min(args.n_each,len(closed_idx)), replace=False)
    print(f"[frames] open={len(open_sel)} closed={len(closed_sel)}")

    # pools de STATE p/ swap (bucket oposto)
    open_states=[torch.as_tensor(ds[int(i)][STATE]).float() for i in open_sel]
    closed_states=[torch.as_tensor(ds[int(i)][STATE]).float() for i in closed_sel]

    CONDS=["baseline","state_zero","state_swap","arm_zero","fingers_zero"]
    rows=[]
    for label, sel, swap_states in (("OPEN",open_sel,closed_states),("CLOSED",closed_sel,open_states)):
        for j,i in enumerate(sel):
            base=make_batch(ds[int(i)], policy, device)
            r={"label":label,"idx":int(i)}
            for c in CONDS:
                b=clone(base)
                if c=="state_zero":   b[STATE]=torch.zeros_like(b[STATE])
                elif c=="state_swap": b[STATE]=swap_states[j%len(swap_states)].to(device).unsqueeze(0)
                elif c=="arm_zero":   b[STATE]=b[STATE].clone(); b[STATE][:, :7]=0.0
                elif c=="fingers_zero": b[STATE]=b[STATE].clone(); b[STATE][:, 7:14]=0.0
                r[c]=pred_squeeze(pred_chunk(policy,preproc,b), args.mode)
            rows.append(r)

    def mean(label,c): return float(np.mean([r[c] for r in rows if r["label"]==label]))
    print(f"\n========== ABLAÇÃO DA PROPRIOCEPÇÃO (mode={args.mode}, imagem sempre REAL) ==========")
    print(f"{'condição':14s} | {'OPEN':>6s} | {'CLOSED':>6s} | {'separação':>9s}")
    base_sep=None
    for c in CONDS:
        o,cl=mean("OPEN",c),mean("CLOSED",c); sep=cl-o
        if c=="baseline": base_sep=sep
        print(f"{c:14s} | {o:6.3f} | {cl:6.3f} | {sep:+9.3f}")
    print("\n--- leitura ---")
    print("Se a SEPARAÇÃO colapsa (→~0) numa condição, esse input é o que CARREGA o abrir/fechar.")
    print(f"baseline separação = {base_sep:+.3f}")
    Path(args.out).write_text(json.dumps({"mode":args.mode,"rows":rows}, indent=2))
    print(f"[ok] {args.out}")

if __name__=="__main__":
    main()
