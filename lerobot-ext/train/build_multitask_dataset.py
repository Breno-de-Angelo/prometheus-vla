#!/usr/bin/env python
"""
Junta vários datasets de tarefa única num único dataset multi-tarefa.

Por que este script existe
──────────────────────────
Neste fork do LeRobot o `MultiLeRobotDataset` está desligado
(`lerobot/datasets/factory.py:115` levanta NotImplementedError), então não dá
para apontar o YAML de treino para vários datasets. O caminho é juntar em disco
antes e treinar em cima do resultado.

A parte que costuma doer é a validação: `aggregate_datasets` exige que `fps`,
`robot_type` e o dicionário de `features` INTEIRO sejam idênticos entre todos os
datasets (`lerobot/datasets/aggregate.py:47`). Uma câmera a mais, uma resolução
diferente ou uma junta extra em um deles e o merge recusa — depois de você já ter
esperado a cópia de dezenas de GB.

Por isso o `--dry-run` faz a mesma validação **antes** de escrever qualquer coisa,
e aponta exatamente qual feature diverge em qual dataset.

Uso
───
    # sempre primeiro:
    python train/build_multitask_dataset.py \
        --datasets meu_dataset/pick_up_the_cup_2026-06-20 \
                   meu_dataset/place_cup_on_coffee_stand_2026-06-20 \
        --output-repo-id local/g1_dex3_multitask \
        --output-dir meu_dataset/g1_dex3_multitask \
        --dry-run

    # depois, sem --dry-run

Ver docs/DATASETS_MULTITAREFA.md para o fluxo completo de gravação.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Carregamento
# ══════════════════════════════════════════════════════════════════════════════


def _load_meta(spec: str) -> tuple[str, Path | None, LeRobotDatasetMetadata]:
    """
    Aceita um caminho local ou um repo_id do Hub.

    Para caminho local, o repo_id vira `local/<nome-da-pasta>` — o LeRobot precisa
    de um, mas ele é só um rótulo quando `root` está preenchido.
    """
    path = Path(spec)
    if path.exists():
        repo_id = f"local/{path.name}"
        return repo_id, path, LeRobotDatasetMetadata(repo_id, root=path)
    return spec, None, LeRobotDatasetMetadata(spec)


# ══════════════════════════════════════════════════════════════════════════════
# Validação (a mesma que o aggregate_datasets faz, só que antes e explicando)
# ══════════════════════════════════════════════════════════════════════════════


def validate(metas: list[tuple[str, Path | None, LeRobotDatasetMetadata]]) -> bool:
    """Compara fps, robot_type e features. Retorna True se dá para juntar."""
    ref_name, _, ref = metas[0]
    ok = True

    print("\n" + "=" * 78)
    print("VALIDAÇÃO DE SCHEMA")
    print("=" * 78)
    print(f"referência: {ref_name}   (fps={ref.fps}, robot_type={ref.robot_type})\n")

    for name, _, meta in metas[1:]:
        problems: list[str] = []

        if meta.fps != ref.fps:
            problems.append(f"fps {meta.fps} ≠ {ref.fps}")
        if meta.robot_type != ref.robot_type:
            problems.append(f"robot_type '{meta.robot_type}' ≠ '{ref.robot_type}'")

        faltando = sorted(set(ref.features) - set(meta.features))
        sobrando = sorted(set(meta.features) - set(ref.features))
        if faltando:
            problems.append(f"features ausentes: {faltando}")
        if sobrando:
            problems.append(f"features a mais: {sobrando}")

        for key in sorted(set(ref.features) & set(meta.features)):
            a, b = ref.features[key], meta.features[key]
            if a.get("shape") != b.get("shape"):
                problems.append(f"'{key}' shape {b.get('shape')} ≠ {a.get('shape')}")
            elif a.get("dtype") != b.get("dtype"):
                problems.append(f"'{key}' dtype {b.get('dtype')} ≠ {a.get('dtype')}")

        if problems:
            ok = False
            print(f"  ✗ {name}")
            for p in problems:
                print(f"      {p}")
        else:
            print(f"  ✓ {name}")

    if not ok:
        print(
            "\nO merge vai recusar. Alinhe o schema antes: mesmas câmeras, mesmas\n"
            "resoluções, mesma dimensão de estado/ação, mesmo fps.\n"
            "Para migrar um dataset já gravado, veja `modify_features` e\n"
            "`add_features` em lerobot/datasets/dataset_tools.py."
        )
    return ok


def report_tasks(metas: list[tuple[str, Path | None, LeRobotDatasetMetadata]]) -> bool:
    """
    Lista as strings de `task` de cada dataset.

    Retorna True se o conjunto resultante tem mais de uma tarefa distinta — que é
    a condição para o treino multi-tarefa fazer sentido.
    """
    print("\n" + "=" * 78)
    print("TAREFAS")
    print("=" * 78)

    todas: set[str] = set()
    for name, _, meta in metas:
        tasks = [str(t) for t in meta.tasks.index]
        todas.update(tasks)
        print(f"  {name}")
        print(f"      {meta.total_episodes} episódios, {meta.total_frames} frames")
        for t in tasks:
            print(f"      task: {t!r}")

    print(f"\n  → {len(todas)} tarefa(s) distinta(s) no dataset unificado")

    if len(todas) < 2:
        print(
            "\n  AVISO: uma tarefa só. Um VLA condicionado por linguagem treinado\n"
            "  assim aprende a ignorar o prompt — trocar o comando não vai mudar o\n"
            "  comportamento. Confira se cada gravação recebeu a sua própria string\n"
            "  de `task` (docs/DATASETS_MULTITAREFA.md, seção 2.2)."
        )
        return False
    return True


def report_disk(metas, output_dir: Path) -> None:
    """O merge COPIA os dados — estima o espaço necessário."""
    total = 0
    for name, root, _ in metas:
        if root is None:
            print(f"\n  {name}: no Hub, tamanho não estimado localmente")
            continue
        size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
        total += size
        print(f"  {name}: {size / 1e9:.1f} GB")

    dest = output_dir if output_dir.exists() else output_dir.parent
    livre = shutil.disk_usage(dest).free
    print(f"\n  necessário ≈ {total / 1e9:.1f} GB | livre em {dest}: {livre / 1e9:.1f} GB")
    if total > livre * 0.9:
        print("  AVISO: espaço apertado. O merge copia, não move nem cria link.")


# ══════════════════════════════════════════════════════════════════════════════
# Merge
# ══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser(
        description="Junta datasets de tarefa única num dataset multi-tarefa.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--datasets", nargs="+", required=True,
        help="caminhos locais ou repo_ids, um por tarefa",
    )
    ap.add_argument("--output-repo-id", required=True, help="ex: local/g1_dex3_multitask")
    ap.add_argument("--output-dir", required=True, help="onde escrever o dataset unificado")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="valida schema, tarefas e espaço em disco sem escrever nada",
    )
    ap.add_argument(
        "--allow-single-task", action="store_true",
        help="prossegue mesmo se o resultado tiver uma tarefa só (só para debug)",
    )
    args = ap.parse_args()

    if len(args.datasets) < 2:
        print("[ERRO] Informe ao menos dois datasets para juntar.")
        return 1

    output_dir = Path(args.output_dir)

    print(f"\nCarregando metadados de {len(args.datasets)} dataset(s)...")
    try:
        metas = [_load_meta(s) for s in args.datasets]
    except Exception as e:
        print(f"[ERRO] Falha ao carregar metadados: {e}")
        return 1

    schema_ok = validate(metas)
    multitask_ok = report_tasks(metas)

    print("\n" + "=" * 78)
    print("ESPAÇO EM DISCO")
    print("=" * 78)
    report_disk(metas, output_dir)

    if not schema_ok:
        print("\n[ERRO] Schema incompatível — merge abortado.")
        return 1

    if not multitask_ok and not args.allow_single_task:
        print("\n[ERRO] Resultado teria uma tarefa só. Use --allow-single-task se for intencional.")
        return 1

    if args.dry_run:
        print("\n[DRY-RUN] Nada foi escrito. Rode de novo sem --dry-run para juntar.")
        return 0

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"\n[ERRO] {output_dir} já existe e não está vazio. Remova ou escolha outro caminho.")
        return 1

    print("\n" + "=" * 78)
    print(f"JUNTANDO em {output_dir}")
    print("=" * 78)

    from lerobot.datasets.dataset_tools import merge_datasets

    datasets = [
        LeRobotDataset(repo_id, root=root) if root else LeRobotDataset(repo_id)
        for repo_id, root, _ in metas
    ]
    merged = merge_datasets(datasets, args.output_repo_id, output_dir=output_dir)

    print("\n" + "=" * 78)
    print("PRONTO")
    print("=" * 78)
    print(f"  episódios: {merged.meta.total_episodes}")
    print(f"  frames:    {merged.meta.total_frames}")
    print(f"  tarefas:   {list(merged.meta.tasks.index)}")
    print(f"\nNo YAML de treino:\n"
          f"  dataset:\n"
          f"    repo_id: \"{args.output_repo_id}\"\n"
          f"    root: \"{output_dir}\"\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
