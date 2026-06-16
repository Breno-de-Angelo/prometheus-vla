#!/usr/bin/env python3
# Teste decisivo: o pi05 USA observation.state? Com SEED FIXA (tira o ruído),
# compara predict_action_chunk com state real vs zerado vs trocado.
# Se as saídas forem IDÊNTICAS -> o state é ignorado (a ablação anterior foi artefato).
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "lerobot-ext"))
from lerobot.policies.factory import make_pre_post_processors  # noqa
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa

IMG = "observation.images.head_camera"; STATE = "observation.state"
CKPT = "train_output/cup_pi05_right8_1squeeze_lf/checkpoints/best/pretrained_model"
ROOT = "datasets/_merged/G1_Dex3_pick_white_cup_right8_1squeeze"
REPO = "lewislf/G1_Dex3_pick_white_cup_right8_1squeeze"

def clone(b): return {k:(v.clone() if torch.is_tensor(v) else v) for k,v in b.items()}

def make_batch(sample, policy, device):
    out={}
    for name in policy.config.input_features:
        v=sample[name]
        if not torch.is_tensor(v): v=torch.as_tensor(v)
        out[name]=v.to(device).unsqueeze(0).float()
    out["task"]="Pick up the white cup"
    return out

def pred(policy, preproc, batch, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    b=preproc(clone(batch))
    with torch.no_grad():
        return policy.predict_action_chunk(b).detach()[0,0].float().cpu().numpy()

def main():
    dev=torch.device("cuda")
    print("[load]"); policy=PI05Policy.from_pretrained(CKPT, strict=False).to(dev).eval()
    preproc,_=make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT)
    ds=LeRobotDataset(REPO, root=ROOT, episodes=list(range(214)), video_backend="pyav")
    # confirma que o input_features inclui o state
    print("input_features:", list(policy.config.input_features.keys()))
    print("state em input_features?", STATE in policy.config.input_features)

    acts=np.asarray(ds.hf_dataset.select_columns(["action"])["action"], dtype=np.float32)[:,7]
    open_i=np.where(acts<0.05)[0][:3]; closed_i=np.where(acts>0.95)[0][:3]
    for lbl,idxs in (("OPEN",open_i),("CLOSED",closed_i)):
        for i in idxs:
            base=make_batch(ds[int(i)], policy, dev)
            a1=pred(policy,preproc,base,0)                              # real, seed 0
            a2=pred(policy,preproc,base,0)                              # real, seed 0 (deve ser == a1)
            bz=clone(base); bz[STATE]=torch.zeros_like(bz[STATE]); az=pred(policy,preproc,bz,0)   # zerado, seed 0
            bo=clone(base); bo[STATE]=bo[STATE]+5.0;             ao=pred(policy,preproc,bo,0)      # state absurdo, seed 0
            d_seed   = float(np.abs(a1-a2).max())   # sanity: seed fixa reproduz?
            d_zero   = float(np.abs(a1-az).max())   # state zerado muda algo?
            d_absurd = float(np.abs(a1-ao).max())   # state +5 muda algo?
            print(f"{lbl} idx{int(i):5d} | seed-repro Δmax={d_seed:.2e} | state-ZERO Δmax={d_zero:.2e} | state+5 Δmax={d_absurd:.2e} | sq={(a1[7]+1)/2:.3f}")
    print("\nLEITURA: se state-ZERO e state+5 derem Δ≈0 (e seed-repro=0), o modelo IGNORA o state.")

if __name__=="__main__": main()
