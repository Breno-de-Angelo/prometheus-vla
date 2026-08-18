#!/usr/bin/env python
"""Espelha para o wandb um treino do LeRobot que JÁ ESTÁ RODANDO.

O `WandBLogger` do LeRobot é construído dentro do `train()`, a partir do
`cfg.wandb`, e não há como injetá-lo num processo vivo. Este script contorna
isso por fora: acompanha o arquivo de log do treino, interpreta as linhas que o
`MetricsTracker` já imprime e as reenvia ao wandb. O treino não é tocado.

O que ele NÃO tem, e é bom saber antes de olhar o painel: gradientes, histograma
de pesos e as métricas de sistema do processo de treino. O que sai daqui é
exatamente o que aparece no log — loss, grad norm, lr, tempo de step e de
dataloading, e o eval_loss dos episódios held-out.

    python wandb_sidecar.py --log /data/train_output/fastwamdepth.log \\
        --project prometheus_g1 --name fastwamdepth_latent
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

import wandb

# `INFO ... step:1K smpl:4K ep:22 epch:0.18 loss:0.234 grdn:5.6 lr:1.0e-04 ...`
LINHA_TREINO = re.compile(r"\bstep:(\S+)\s+smpl:")
PAR = re.compile(r"([A-Za-z_]+[A-Za-z_0-9]*):([-+0-9][^\s]*)")
# `INFO ... step 500: eval_loss=0.1234`
LINHA_EVAL = re.compile(r"step (\d+): eval_loss=([0-9.eE+-]+)")
# a linha que o nosso run_train imprime quando publica o melhor checkpoint
LINHA_MELHOR = re.compile(r"step (\d+): eval ([0-9.]+) é o novo melhor")

SUFIXOS = {"K": 1e3, "M": 1e6, "B": 1e9}


def _num(texto: str) -> float | None:
    """Converte os valores do log, inclusive os abreviados (`1K`, `4M`)."""
    texto = texto.rstrip(",")
    mult = SUFIXOS.get(texto[-1:].upper())
    if mult is not None:
        texto = texto[:-1]
    else:
        mult = 1.0
    try:
        return float(texto) * mult
    except ValueError:
        return None


def inicio_da_corrida(caminho: Path) -> int:
    """Byte onde começa a ÚLTIMA corrida do arquivo.

    O log é aberto em modo append: um relançamento escreve no mesmo arquivo,
    depois de tudo o que as tentativas anteriores deixaram. Reenviar aquilo
    misturaria corridas diferentes no mesmo gráfico.
    """
    marcador = b"Creating dataset"
    dados = caminho.read_bytes()
    pos = dados.rfind(marcador)
    return 0 if pos < 0 else dados.rfind(b"\n", 0, pos) + 1


def treino_vivo(padrao: str) -> bool:
    return subprocess.run(["pgrep", "-f", padrao], capture_output=True).returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--project", default="prometheus_g1")
    ap.add_argument("--name", default=None)
    ap.add_argument("--log-freq", type=int, default=50,
                    help="o `log_freq` do treino: o step exato de cada linha de treino sai daqui, "
                         "porque o `step:` impresso vem abreviado (1K) e perderia precisão")
    ap.add_argument("--proc", default="policies.fastwam_depth",
                    help="padrão do pgrep que identifica o treino; quando ele some, o sidecar encerra")
    ap.add_argument("--desde-o-inicio", action="store_true", default=True)
    args = ap.parse_args()

    run = wandb.init(project=args.project, name=args.name, resume="allow",
                     notes="espelho do log — ver wandb_sidecar.py")
    print(f"[sidecar] {run.url}", flush=True)

    with args.log.open("rb") as f:
        f.seek(inicio_da_corrida(args.log))
        n_treino = 0
        ocioso = 0
        while True:
            linha = f.readline()
            if not linha:
                if not treino_vivo(args.proc):
                    ocioso += 1
                    if ocioso > 3:  # dá tempo de drenar o que ficou no buffer
                        break
                time.sleep(2)
                continue
            ocioso = 0
            texto = linha.decode("utf-8", "replace")

            casou = LINHA_EVAL.search(texto)
            if casou:
                run.log({"eval/loss": float(casou.group(2))}, step=int(casou.group(1)))
                continue

            casou = LINHA_MELHOR.search(texto)
            if casou:
                run.log({"eval/melhor_loss": float(casou.group(2))}, step=int(casou.group(1)))
                continue

            if LINHA_TREINO.search(texto):
                n_treino += 1
                step = n_treino * args.log_freq
                metricas = {}
                for chave, valor in PAR.findall(texto.split("step:", 1)[1]):
                    v = _num(valor)
                    if v is not None:
                        metricas[f"train/{chave}"] = v
                if metricas:
                    run.log(metricas, step=step)

    run.finish()
    print("[sidecar] treino encerrado, run fechada", flush=True)


if __name__ == "__main__":
    main()
