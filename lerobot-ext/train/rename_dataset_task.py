#!/usr/bin/env python
"""
Renomeia a string de `task` de um dataset LeRobot, sem tocar nos dados.

Por que isso importa
────────────────────
A string de `task` é o ÚNICO mecanismo de condicionamento por linguagem das
políticas `pi05depth` e `openvladepth`. O prompt entregue ao modelo é montado
diretamente dela:

    "In: What action should the robot take to {task}?\\nOut:"

Se a string não descreve o que a demonstração realmente faz, o modelo aprende a
associação errada — e o problema só aparece quando você tenta um comando novo.

O que o script altera
─────────────────────
A tarefa mora em dois arquivos de METADADOS:

    meta/tasks.parquet            índice = string, coluna task_index
    meta/episodes/**/*.parquet    coluna `tasks`, uma lista por episódio

Os arquivos de dados (`data/**/*.parquet`) guardam só o `task_index` inteiro, que
NÃO muda. Por isso renomear é barato e reversível — nenhum vídeo é reescrito.

Uso
───
    python train/rename_dataset_task.py \\
        --dataset meu_dataset/pick_up_the_cup_2026-06-09 \\
        --from "pick up the cup" \\
        --to   "pick up the white mug and place it to the right" \\
        --dry-run

    # depois, sem --dry-run (um backup de meta/ é criado automaticamente)

Escrever a string nova
──────────────────────
Em inglês, imperativo, minúsculas — é o formato do pré-treino do OpenVLA.
Descreva a AÇÃO COMPLETA da demonstração, não só o começo:

    ✓ "pick up the white mug and place it to the right"
    ✗ "pick up the cup"        (descreve metade da demo)
    ✗ "tarefa 1"               (não descreve nada)
"""

from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path

import pandas as pd


def _episode_files(root: Path) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Renomeia a string de `task` de um dataset LeRobot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dataset", required=True, help="raiz do dataset (a pasta com meta/, data/, videos/)")
    ap.add_argument("--from", dest="old", required=True, help="string atual")
    ap.add_argument("--to", dest="new", required=True, help="string nova")
    ap.add_argument("--dry-run", action="store_true", help="mostra o que mudaria, sem escrever")
    args = ap.parse_args()

    root = Path(args.dataset)
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        print(f"[ERRO] {tasks_path} não encontrado. O --dataset aponta para a raiz certa?")
        return 1

    tasks = pd.read_parquet(tasks_path)
    print(f"\ntasks.parquet atual ({tasks_path}):")
    print(tasks.to_string())

    if args.old not in tasks.index:
        print(f"\n[ERRO] A tarefa {args.old!r} não existe neste dataset.")
        print(f"       Tarefas presentes: {[str(t) for t in tasks.index]}")
        return 1
    if args.new in tasks.index:
        print(f"\n[ERRO] A tarefa {args.new!r} já existe. Renomear criaria duas entradas iguais.")
        return 1

    eps = _episode_files(root)
    n_eps_afetados = 0
    for f in eps:
        df = pd.read_parquet(f)
        if "tasks" in df.columns:
            n_eps_afetados += sum(args.old in list(t) for t in df["tasks"])

    print(f"\nMudança:\n  {args.old!r}\n  → {args.new!r}")
    print(f"\nArquivos afetados:")
    print(f"  meta/tasks.parquet                 (1 linha)")
    print(f"  meta/episodes/**/*.parquet         ({len(eps)} arquivo(s), {n_eps_afetados} episódio(s))")
    print(f"  data/**/*.parquet                  (NENHUM — guardam só task_index)")
    print(f"  videos/**/*.mp4                    (NENHUM)")

    if args.dry_run:
        print("\n[DRY-RUN] Nada foi escrito. Rode de novo sem --dry-run para aplicar.")
        return 0

    # Backup só dos metadados — é tudo que este script toca.
    backup = root / "meta_backup_rename"
    if backup.exists():
        print(f"\n[ERRO] {backup} já existe. Remova ou renomeie antes de prosseguir.")
        return 1
    shutil.copytree(root / "meta", backup)
    print(f"\nBackup de meta/ em {backup}")

    # 1. tasks.parquet — troca o índice, preserva o task_index
    tasks = tasks.rename(index={args.old: args.new})
    tasks.to_parquet(tasks_path)
    print(f"  ✓ meta/tasks.parquet")

    # 2. episodes — a coluna `tasks` é uma lista por episódio
    for f in eps:
        df = pd.read_parquet(f)
        if "tasks" not in df.columns:
            continue
        df["tasks"] = df["tasks"].apply(
            lambda lst: [args.new if t == args.old else t for t in list(lst)]
        )
        df.to_parquet(f)
    print(f"  ✓ meta/episodes/ ({len(eps)} arquivo(s))")

    # 3. Conferência
    check = pd.read_parquet(tasks_path)
    print("\ntasks.parquet novo:")
    print(check.to_string())
    if args.new not in check.index:
        print("\n[ERRO] A verificação falhou — restaure a partir do backup.")
        return 1

    print(
        "\nPronto. Se este dataset já foi usado num treino, o checkpoint antigo\n"
        "continua com o prompt velho embutido no processador — retreine para que\n"
        "o modelo veja a string nova."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
