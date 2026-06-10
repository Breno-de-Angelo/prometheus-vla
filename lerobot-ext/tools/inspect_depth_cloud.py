#!/usr/bin/env python3
"""Inspeção visual do pipeline de nuvem de pontos do pi05-depth (auditoria FASE 4).

Pega N frames aleatórios do dataset, roda o pipeline COMPLETO usado no treino
(back-projection com intrínsecos da config → crop de workspace → FPS 1024) e
salva scatter plots 3D em PNG, para verificar que mesa/copo dominam a nuvem.

Uso:
    python lerobot-ext/tools/inspect_depth_cloud.py \
        --config lerobot-ext/config/train/train_cup_pi05_right14_depth.yaml \
        --root lerobot-ext/datasets/G1_Dex3_right14_dataset/20260608_205432 \
        --n 5 --out /tmp/depth_cloud_inspect
"""
import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "lerobot-ext"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from train.depth_encoder import depth_to_pointcloud, validate_intrinsics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML de treino (lê depth_intrinsics/depth_scale/depth_workspace/depth_key)")
    ap.add_argument("--root", required=True, help="root do dataset LeRobot")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="/tmp/depth_cloud_inspect")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    intr = validate_intrinsics(cfg.get("depth_intrinsics"))
    scale = cfg.get("depth_scale")
    assert scale is not None, "config sem depth_scale"
    workspace = cfg.get("depth_workspace")
    depth_key = cfg.get("depth_key", "observation.images.head_camera_depth")
    print(f"[config] intrinsics={intr} scale={scale} workspace={workspace}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="local/inspect", root=args.root, video_backend="pyav")
    random.seed(args.seed)
    frames = random.sample(range(ds.num_frames), args.n)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in frames:
        depth = ds[i][depth_key].float()  # [1, H, W]
        torch.manual_seed(args.seed)  # pré-subsample determinístico p/ inspeção
        pc = depth_to_pointcloud(depth.unsqueeze(0), intr, num_points=1024,
                                 depth_scale=scale, workspace=workspace)[0]  # (3, 1024)
        x, y, z = pc[0].numpy(), pc[1].numpy(), pc[2].numpy()
        nz = (pc != 0).any(0).sum().item()

        fig = plt.figure(figsize=(14, 5))
        fig.suptitle(f"frame {i} — {nz}/1024 pontos não-nulos — "
                     f"z[{z[z > 0].min() if (z > 0).any() else 0:.2f}, {z.max():.2f}]m")
        ax = fig.add_subplot(131, projection="3d")
        ax.scatter(x, y, z, s=2, c=z, cmap="viridis")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_title("nuvem 3D (crop+FPS)")
        ax2 = fig.add_subplot(132)
        ax2.scatter(x, z, s=2, c=z, cmap="viridis")
        ax2.set_xlabel("x (m)"); ax2.set_ylabel("z (m)"); ax2.set_title("vista de topo (x-z)")
        ax2.invert_yaxis()
        ax3 = fig.add_subplot(133)
        ax3.scatter(x, -torch.tensor(y).numpy(), s=2, c=z, cmap="viridis")
        ax3.set_xlabel("x (m)"); ax3.set_ylabel("-y (m)"); ax3.set_title("vista frontal (x,-y)")
        path = out_dir / f"cloud_frame{i:05d}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[ok] {path}  ({nz}/1024 pontos, z mediana={z[z > 0].mean():.3f}m)" if (z > 0).any()
              else f"[ok] {path}  (NUVEM VAZIA — checar crop/intrínsecos!)")

    print(f"\n{args.n} PNGs em {out_dir}/")


if __name__ == "__main__":
    main()
