#!/usr/bin/env python
"""Treino do FastWAM-D.

Ao contrário do `policies/act_depth/run_train.py`, este não é um fork do laço
de treino do LeRobot — ele só REGISTRA a política e chama o `lerobot-train` de
fábrica. O motivo é deliberado:

  - o laço forkeado do ACT-D existe por causa de coisas que só ele tem
    (curriculum de posição neutra, gate de incerteza, freeze seletivo por
    encoder). O FastWAM-D não precisa de nada disso;
  - um modelo de 5B depende de `accelerate`, gradient checkpointing e
    checkpoint/retomada exatos por amostra, que o laço de fábrica já resolve e
    um fork precisaria perseguir a cada rebase;
  - validação em episódios separados existe nativamente na 0.6.1 via
    `dataset.eval_split` + `eval_steps` — não é preciso o `val_dataset` do fork.

O que este arquivo ACRESCENTA ao laço de fábrica é uma coisa só: guardar
**apenas o melhor checkpoint**, em vez de um a cada `save_freq`. O motivo é
disco. Um checkpoint do FastWAM-D carrega o expert de vídeo de 5B inteiro
(~10 GB de pesos, mais o estado do otimizador); 20 mil steps salvando a cada
2 mil dão dez desses. O laço nativo não tem noção de "melhor" — o `eval_loss`
do `eval_steps` só é registrado no log — e ele nunca apaga checkpoint antigo.
Em vez de forkear o laço por causa disso, os três nomes que decidem ONDE e SE
salvar são trocados aqui (ver `_instala_melhor_apenas`).

    cd lerobot-ext
    python -m policies.fastwam_depth.run_train \\
        --config_path=config/train/fastwamdepth_white_cup_on_dripper.yaml
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys

import numpy as np
from pathlib import Path
from typing import Any

# O `cwd` precisa estar no path para o `make_policy` reimportar
# `policies.fastwam_depth.modeling_fastwam_depth` pela convenção de nomes
# (ver `lerobot/policies/factory.py::_get_policy_cls_from_policy_name`).
sys.path.append(os.getcwd())

# Import com efeito colateral, e é o ponto de existir deste arquivo: o
# decorador `@PreTrainedConfig.register_subclass("fastwamdepth")` só roda
# quando o módulo é importado. Sem isto o draccus rejeita
# `policy.type: fastwamdepth` como tipo desconhecido.
from policies.fastwam_depth.configuration_fastwam_depth import FastWAMDepthConfig  # noqa: F401
from lerobot.scripts import lerobot_train
from lerobot.scripts.lerobot_train import train


# ══════════════════════════════════════════════════════════════════════════
# "Só o melhor checkpoint"
# ══════════════════════════════════════════════════════════════════════════
# O laço nativo faz, na MESMA iteração e nesta ordem: calcula o `eval_loss` dos
# episódios held-out e o registra no log; depois, se for step de salvar, chama
# `get_step_checkpoint_dir` -> `save_checkpoint` -> `update_last_checkpoint`
# (`lerobot/scripts/lerobot_train.py`). Trocamos os três:
#
#   get_step_checkpoint_dir -> escreve numa área de estágio, não em checkpoints/NNNNNN
#   save_checkpoint         -> só escreve se o eval_loss melhorou; depois publica o
#                              estágio por cima de checkpoints/best, atomicamente
#   update_last_checkpoint  -> o link `last` aponta sempre para `best`
#
# Assim o disco guarda UM checkpoint (dois por alguns segundos, durante a troca)
# em vez de um por `save_freq`.

_MELHOR: dict[str, Any] = {"eval_loss": float("inf"), "step": None, "visto_em": None}

_PADRAO_EVAL = re.compile(r"eval_loss=([0-9.eE+-]+)")


class _CapturaEvalLoss(logging.Handler):
    """Lê o `eval_loss` da linha que o laço nativo já registra.

    É a via menos invasiva: no laço de fábrica o `eval_loss` é variável local,
    não entra no `MetricsTracker` e (sem wandb) só aparece em
    `logging.info(f"step {step}: eval_loss={eval_loss:.4f}")`. Se um rebase
    mudar esse texto, o `save_checkpoint` abaixo avisa e salva assim mesmo — o
    desfecho ruim seria terminar 20 mil steps sem nada em disco.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # As mensagens DESTE módulo são ignoradas: o `save_checkpoint` abaixo
        # também registra o valor do eval, e sem esta guarda ele realimentaria
        # o handler — um save pulado repor-ia o `visto_em` que ele mesmo acabou
        # de consumir, e o aviso de "nenhum eval_loss lido" nunca dispararia.
        if record.pathname == __file__:
            return
        try:
            casou = _PADRAO_EVAL.search(record.getMessage())
        except Exception:  # noqa: BLE001 - um handler não pode derrubar o treino
            return
        if casou:
            _MELHOR["visto_em"] = float(casou.group(1))


def _publica(estagio: Path, destino: Path) -> None:
    """Troca `destino` por `estagio` sem deixar o melhor checkpoint corrompido.

    Renomear é atômico dentro do mesmo sistema de arquivos, então o que está em
    `best` a qualquer instante é sempre um checkpoint íntegro — o anterior ou o
    novo. Escrever direto por cima de `best` arriscaria perder os dois se o
    treino morresse no meio da escrita.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    antigo = destino.with_name(destino.name + ".old")
    if antigo.exists():
        shutil.rmtree(antigo)
    if destino.exists():
        destino.rename(antigo)
    estagio.rename(destino)
    if antigo.exists():
        shutil.rmtree(antigo)


def _instala_eval_representativo() -> None:
    """Faz o `max_eval_samples` amostrar o held-out INTEIRO, não só o começo.

    O laço nativo monta o subconjunto de validação com
    `(task_arr == t).nonzero()[0][:per_task]` (`lerobot_train.py`), ou seja, os
    PRIMEIROS N quadros de cada tarefa. Como os episódios são concatenados em
    ordem, isso pega só o início do primeiro episódio held-out — a fase de
    aproximação — e nunca o momento de fechar a mão, que é justamente onde este
    modelo erra. Um `eval_loss` medido assim melhora enquanto a garra piora.

    A troca é no `Subset`: ele aparece uma única vez no laço, e é exatamente
    para isto. Os índices viram um espaçamento uniforme sobre todo o conjunto de
    validação, cobrindo todos os episódios held-out do começo ao fim.

    Com mais de uma tarefa a amostragem deixa de ser equilibrada por tarefa e
    passa a ser proporcional ao tamanho de cada uma — o que é razoável, mas é
    diferente do que o de origem faz. Hoje o dataset tem uma tarefa só.
    """
    import torch.utils.data as tud

    subset_original = tud.Subset

    class SubsetUniforme(subset_original):
        def __init__(self, dataset, indices):
            n = len(indices)
            total = len(dataset)
            if 0 < n < total:
                indices = np.linspace(0, total - 1, n).round().astype(int).tolist()
                logging.info(
                    "[FastWAM-D] validação: %d amostras espaçadas uniformemente sobre %d "
                    "quadros held-out (em vez dos %d primeiros).", n, total, n
                )
            super().__init__(dataset, indices)

    tud.Subset = SubsetUniforme


def _instala_handler() -> None:
    """Põe o `_CapturaEvalLoss` na raiz, sem duplicar se já estiver lá."""
    raiz = logging.getLogger()
    if not any(isinstance(h, _CapturaEvalLoss) for h in raiz.handlers):
        raiz.addHandler(_CapturaEvalLoss())


def _instala_melhor_apenas() -> None:
    save_original = lerobot_train.save_checkpoint
    update_original = lerobot_train.update_last_checkpoint
    init_logging_original = lerobot_train.init_logging

    def get_step_checkpoint_dir(output_dir, total_steps, step):  # noqa: ARG001
        return Path(output_dir) / "checkpoints" / "best.tmp"

    def save_checkpoint(checkpoint_dir, step, cfg, *args: Any, **kwargs: Any):
        eval_loss = _MELHOR["visto_em"]
        _MELHOR["visto_em"] = None

        if eval_loss is None:
            logging.warning(
                "step %s: nenhum eval_loss lido desde o último save — salvando assim mesmo. "
                "Confira se `eval_steps` > 0 e se `save_freq` é múltiplo dele.",
                step,
            )
        elif eval_loss >= _MELHOR["eval_loss"]:
            logging.info(
                "step %s: eval %.4f não supera o melhor (%.4f, do step %s) — nada é escrito.",
                step, eval_loss, _MELHOR["eval_loss"], _MELHOR["step"],
            )
            return
        else:
            logging.info(
                "step %s: eval %.4f é o novo melhor (era %.4f) — publicando em checkpoints/best.",
                step, eval_loss, _MELHOR["eval_loss"],
            )

        estagio = Path(checkpoint_dir)
        if estagio.exists():
            shutil.rmtree(estagio)
        save_original(estagio, step, cfg, *args, **kwargs)
        _publica(estagio, estagio.with_name("best"))

        if eval_loss is not None:
            _MELHOR["eval_loss"] = eval_loss
            _MELHOR["step"] = step

    def update_last_checkpoint(checkpoint_dir):
        # O que chega é a área de estágio, que já não existe: o link tem que
        # apontar para `best`, que é onde o `--resume` e o `--policy.path` vão
        # procurar. Se o save foi pulado por não ser o melhor, `best` continua
        # sendo o de antes e o link segue válido.
        melhor = Path(checkpoint_dir).with_name("best")
        if melhor.exists():
            update_original(melhor)

    def init_logging(*args: Any, **kwargs: Any):
        # O `init_logging` do LeRobot faz `logger.handlers.clear()`
        # (`utils/utils.py`) e ele roda DENTRO do `train()` — ou seja, depois
        # daqui. Sem reinstalar o handler em seguida, ele é varrido antes do
        # primeiro eval e o `save_checkpoint` abaixo nunca vê um eval_loss:
        # cai no caminho do aviso e salva em todo `save_freq`, virando "o
        # último" em vez de "o melhor". Foi o que aconteceu no step 1000 da
        # primeira corrida.
        init_logging_original(*args, **kwargs)
        _instala_handler()

    lerobot_train.get_step_checkpoint_dir = get_step_checkpoint_dir
    lerobot_train.save_checkpoint = save_checkpoint
    lerobot_train.update_last_checkpoint = update_last_checkpoint
    lerobot_train.init_logging = init_logging
    _instala_handler()


def main() -> None:
    _instala_melhor_apenas()
    _instala_eval_representativo()
    train()


if __name__ == "__main__":
    main()
