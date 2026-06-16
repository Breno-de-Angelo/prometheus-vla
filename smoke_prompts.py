#!/usr/bin/env python3
# Smoke dos PROMPTS (revisão do prof): com o dataset REAL, imprime os prompts gerados
# pelo state-regularizer p/ confirmar a mecânica ANTES de gastar 20k steps:
#   (1) ~50% dos prompts de TREINO saem SEM "State:" (dropout p=0.5)
#   (2) os prompts COM State têm 32 bins (max_state_dim)
#   (3) só as 7 primeiras dims recebem ruído (padding 8..32 fica intacto)
#   (4) VAL/DEPLOY (regularizer desligado) = sempre "State:" completo, sem ruído
# Roda em CPU (só metadados do dataset + o step), não toca a GPU do treino.
import sys, argparse
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "lerobot-ext"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import OBS_STATE
import train.state_regularizer as SR

ap = argparse.ArgumentParser()
ap.add_argument("--repo-id", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--p", type=float, default=0.5)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--seed", type=int, default=1000)
args = ap.parse_args()

ds = LeRobotDataset(args.repo_id, root=args.root, episodes=list(range(5)), video_backend="pyav")
st = ds.meta.stats["observation.state"]
q01 = np.asarray(st["q01"], np.float32)[:7]; q99 = np.asarray(st["q99"], np.float32)[:7]
idx = np.linspace(0, len(ds) - 1, args.n).astype(int)
raw = np.stack([np.asarray(ds[int(i)]["observation.state"], np.float32)[:7] for i in idx])
norm = np.clip(2.0 * (raw - q01) / (q99 - q01 + 1e-6) - 1.0, -1, 1)  # quantile-norm -> [-1,1]
state = torch.tensor(norm)


def run_step(enabled):
    step = Pi05PrepareStateTokenizerProcessorStep(max_state_dim=32)
    tr = {TransitionKey.OBSERVATION: {OBS_STATE: state.clone()},
          TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick up the white cup"] * len(idx)}}
    SR._STATE_REG["enabled"] = enabled
    return step(tr)[TransitionKey.COMPLEMENTARY_DATA]["task"]


def bins_of(p):
    return p.split("State:")[1].split(";")[0].split() if "State:" in p else None


SR.install_state_regularizer(dropout_prob=args.p, noise_bins=args.k, state_dim=7, seed=args.seed)

print("=" * 80)
print(f"TREINO — regularizer ATIVO (dropout p={args.p}, ruído k={args.k} nas 7 dims, seed={args.seed})")
print("=" * 80)
tp = run_step(True)
no_state = 0
for i, p in enumerate(tp):
    b = bins_of(p)
    no_state += b is None
    print(f"[{i:2d}] ({'SEM State' if b is None else f'{len(b)} bins':9s}) {p!r}")
print(f"\n(1) sem State: {no_state}/{len(tp)} = {100*no_state/len(tp):.0f}%   (esperado ~{100*args.p:.0f}%)")
sb = next((len(bins_of(p)) for p in tp if bins_of(p)), 0)
print(f"(2) prompts COM State têm {sb} bins   (esperado 32)")

print("\n" + "=" * 80)
print("VAL/DEPLOY — regularizer DESLIGADO (state completo, sem ruído)")
print("=" * 80)
vp = run_step(False)
for i, p in enumerate(vp[:4]):
    print(f"[{i:2d}] {p!r}")
with_state = sum(bins_of(p) is not None for p in vp)
print(f"\n(4) com State: {with_state}/{len(vp)} = {100*with_state/len(vp):.0f}%   (esperado 100%)")

print("\n" + "=" * 80)
print("(3) RUÍDO só nas 7 primeiras dims (treino-com-state vs val, MESMO frame)")
print("=" * 80)
shown = False
for i in range(len(idx)):
    tb, vb = bins_of(tp[i]), bins_of(vp[i])
    if tb is not None:
        tbi = list(map(int, tb)); vbi = list(map(int, vb))
        d = [a - b for a, b in zip(tbi, vbi)]
        print(f"frame {i}:")
        print(f"  val    bins = {vbi}")
        print(f"  treino bins = {tbi}")
        print(f"  Δ 7 reais   = {d[:7]}   (cada ∈ [-{args.k}, {args.k}])")
        print(f"  Δ pad 8-32  = conjunto {set(d[7:])}   (deve ser {{0}})")
        ok = all(abs(x) <= args.k for x in d[:7]) and set(d[7:]) == {0} and len(tbi) == 32
        print(f"  => {'OK ✓' if ok else 'FALHOU ✗'}: ruído só nas 7, padding intacto, 32 bins")
        shown = True
        break
if not shown:
    print("  (nenhum prompt de treino com State neste batch — aumente --n)")
