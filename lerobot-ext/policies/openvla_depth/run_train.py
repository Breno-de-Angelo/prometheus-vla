#!/usr/bin/env python
"""
OpenVLA-Depth — motor de treinamento.

O loop em si é o mesmo do `policies/pi0_depth/run_train.py`: aquele arquivo já é
genérico (resolve política e processadores via `make_policy` /
`make_pre_post_processors` do LeRobot) e traz coisas que queremos manter — split
de validação, `BestValTracker`, shutdown limpo no Ctrl+C, `ColorJitter` só em RGB
e o reset das stats de profundidade para identidade. Duplicar 950 linhas para
trocar duas importações seria pior.

O que este módulo acrescenta é a checagem que só faz sentido num VLA multi-tarefa:
avisar (ou abortar) quando o dataset tem uma única string de `task`. Um modelo
condicionado por linguagem treinado com um comando só aprende a ignorar o texto —
e o sintoma aparece semanas depois, quando o segundo comando não funciona.
"""

from __future__ import annotations

import logging
import sys

import yaml
from termcolor import colored

# Registra a política antes de qualquer coisa: o `make_policy` do LeRobot resolve
# "openvladepth" pelo registro de subclasses do PreTrainedConfig.
from . import configuration_openvla  # noqa: F401

# Motor compartilhado.
from ..pi0_depth.run_train import train  # noqa: F401


def _dataset_tasks(repo_id: str, root: str | None = None) -> list[str] | None:
    """
    Lê as strings de tarefa do dataset a partir dos metadados.

    `root` é obrigatório para datasets locais (`meu_dataset/...`): sem ele o
    LeRobot tentaria baixar `repo_id` do Hub e falharia.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(repo_id, root=root)
        return [str(t) for t in meta.tasks.index]
    except Exception as e:  # metadados indisponíveis não devem bloquear o treino
        logging.warning(f"[OpenVLA-D] Não foi possível ler as tarefas de '{repo_id}': {e}")
        return None


def check_multitask(config_path: str) -> None:
    """
    Confere se o dataset tem mais de uma tarefa e se `override_task` não está
    anulando o condicionamento por linguagem.

    Aborta quando `policy.require_multitask: true` no YAML.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    policy_cfg = cfg.get("policy", {}) or {}
    dataset_cfg = cfg.get("dataset", {}) or {}
    repo_id = dataset_cfg.get("repo_id")
    root = dataset_cfg.get("root")
    require = bool(policy_cfg.get("require_multitask", False))
    override = policy_cfg.get("override_task")

    if override:
        print(
            colored(
                f"[AVISO] override_task='{override}' — o campo `task` do dataset será "
                "ignorado e todos os episódios verão o mesmo prompt. Isso desliga o "
                "multi-tarefa. Remova do YAML para treinar de verdade.",
                "yellow",
                attrs=["bold"],
            )
        )

    if not repo_id:
        return

    tasks = _dataset_tasks(repo_id, root)
    if tasks is None:
        return

    print(colored(f"\n[OpenVLA-D] Tarefas em '{repo_id}': {len(tasks)}", "cyan", attrs=["bold"]))
    for t in tasks:
        print(f"    • {t}")
    print()

    if len(tasks) < 2:
        msg = (
            f"O dataset '{repo_id}' tem apenas {len(tasks)} tarefa distinta. "
            "Um VLA condicionado por linguagem treinado assim aprende a ignorar o "
            "texto do prompt: o comando não vai mudar o comportamento na inferência.\n"
            "Grave demos de tarefas diferentes (cada uma com sua string de `task`) "
            "ou use `train/config/openvla_depth_multitask.yaml` como referência."
        )
        if require:
            print(colored(f"[ERRO] {msg}", "red", attrs=["bold"]))
            sys.exit(1)
        print(colored(f"[AVISO] {msg}", "yellow", attrs=["bold"]))


def main():
    from lerobot.utils.import_utils import register_third_party_plugins

    config_arg = next((a for a in sys.argv if "--config_path" in a), None)
    if config_arg:
        try:
            check_multitask(config_arg.split("=", 1)[-1])
        except Exception as e:
            logging.warning(f"[OpenVLA-D] Checagem de multi-tarefa pulada: {e}")

    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
