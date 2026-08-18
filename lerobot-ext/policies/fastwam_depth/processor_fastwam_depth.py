#!/usr/bin/env python
"""Pré/pós-processadores do FastWAM-D.

Não há nada a mudar em relação ao FastWAM de origem, e o motivo vale registrar:
o `normalization_mapping` dele já marca `VISUAL: IDENTITY`
(`configuration_fastwam.py`), então o mapa de profundidade — declarado como
feature VISUAL — **atravessa o normalizador intacto**, na unidade nativa do
dataset (milímetros).

É exatamente o que o enxerto precisa. Nas outras políticas de profundidade
daqui (`policies/act_depth/processor_act.py`) foi preciso remover o depth da
lista de features do normalizador na mão, porque lá o VISUAL é MEAN_STD e a
normalização por estatística destruiria a escala métrica.

Este módulo existe para o `run_train` importar um nome estável e para esta
nota ficar perto de quem for mexer.
"""

from __future__ import annotations

import torch

from lerobot.policies.fastwam.processor_fastwam import make_fastwam_pre_post_processors

from .configuration_fastwam_depth import FastWAMDepthConfig


def make_fastwamdepth_pre_post_processors(
    config: FastWAMDepthConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
):
    """Mesmos pipelines do FastWAM — ver a nota no topo do módulo."""
    return make_fastwam_pre_post_processors(config, dataset_stats)
