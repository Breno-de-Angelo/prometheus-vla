#!/usr/bin/env python3
"""Junta N datasets LeRobot locais (mesmo schema) num só e sobe pro HF Hub.

Uso:
    python lerobot-ext/tools/merge_and_push.py \
        --inputs datasets/G1_Dex3_depth_tactil_dataset/20260609_124014 \
                 datasets/G1_Dex3_depth_tactil_dataset/20260609_125917 \
        --output-repo-id lewislf/G1_Dex3_pick_white_cup_v2 \
        --output-dir datasets/_merged/G1_Dex3_pick_white_cup_v2 \
        --push            # tire pra só fazer o merge local sem subir
"""
import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="pastas (root) dos datasets locais a juntar")
    ap.add_argument("--output-repo-id", required=True, help="ex: lewislf/G1_Dex3_pick_white_cup_v2")
    ap.add_argument("--output-dir", required=True, help="pasta onde gravar o dataset merged")
    ap.add_argument("--push", action="store_true", help="sobe pro HF Hub depois do merge")
    ap.add_argument("--private", action="store_true", help="repo HF privado")
    args = ap.parse_args()

    datasets = []
    for i, root in enumerate(args.inputs):
        root = Path(root).resolve()
        # repo_id local é só rótulo; o que importa é o root.
        ds = LeRobotDataset(repo_id=f"local/input_{i}", root=root)
        print(f"[load] {root}  ->  {ds.num_episodes} eps, {ds.num_frames} frames")
        datasets.append(ds)

    total_eps = sum(d.num_episodes for d in datasets)
    total_frames = sum(d.num_frames for d in datasets)
    print(f"[merge] {len(datasets)} datasets -> {total_eps} eps, {total_frames} frames -> {args.output_repo_id}")

    merged = merge_datasets(datasets, output_repo_id=args.output_repo_id, output_dir=args.output_dir)
    print(f"[merge] OK: {merged.num_episodes} eps, {merged.num_frames} frames em {args.output_dir}")
    assert merged.num_episodes == total_eps, f"esperava {total_eps} eps, vieram {merged.num_episodes}"

    if args.push:
        print(f"[push] subindo para HF: {args.output_repo_id} (private={args.private})...")
        merged.push_to_hub(private=args.private)
        print("[push] ✅ concluído.")
    else:
        print("[push] pulado (use --push pra subir).")


if __name__ == "__main__":
    main()
