#!/usr/bin/env python
"""Configuração do FastWAM-D — FastWAM com profundidade métrica.

Estende o `FastWAMConfig` do LeRobot sem tocar nele: o que muda aqui é só
como a profundidade entra no modelo. Ver `README.md` para o desenho.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.utils.constants import OBS_STATE

# Modos de fusão da profundidade, do mais forte ao mais barato.
DEPTH_MODE_LATENT = "latent"
DEPTH_MODE_TOKEN = "token"
DEPTH_MODE_OFF = "off"
DEPTH_MODES = (DEPTH_MODE_LATENT, DEPTH_MODE_TOKEN, DEPTH_MODE_OFF)


@PreTrainedConfig.register_subclass("fastwamdepth")
@dataclass
class FastWAMDepthConfig(FastWAMConfig):
    """FastWAM com o mapa de profundidade do dataset entrando no modelo.

    Args:
        depth_mode: como a profundidade chega ao DiT.

            - ``"latent"`` (padrão): a profundidade é codificada pelo MESMO VAE
              congelado do Wan e concatenada ao latente do vídeo no canal, na
              entrada do `patch_embedding`. Cada token do DiT passa a carregar
              a distância do próprio pedaço de imagem — é o modo que dá
              correspondência espacial de verdade.
            - ``"token"``: a profundidade vira UM token global no contexto da
              cross-attention (média espacial do latente projetada em
              ``text_dim``), no mesmo lugar onde a propriocepção entra. Barato,
              não mexe no `patch_embedding` — serve de linha de base para
              medir quanto o modo ``latent`` está de fato ganhando.
            - ``"off"``: desliga; roda como o FastWAM de origem. É o controle
              do experimento.

        depth_unit: unidade em que o dataset entrega a profundidade. O LeRobot
            0.6.1 usa milímetros (``depth_output_unit``).
        depth_min, depth_max: faixa métrica (em METROS) mapeada para [0, 1]
            antes do VAE. Fora dela satura. O padrão cobre manipulação sobre
            mesa; a cena do café fica entre ~0,3 m e ~1,8 m.
        depth_use_log: mapeia em log em vez de linear. Ligado por padrão pelo
            mesmo motivo que o LeRobot grava em log (`datasets/depth_utils.py`):
            o erro do sensor cresce com a distância, então gastar resolução
            perto é o que interessa para manipulação.
        depth_train_patch_embedding: destrava o `patch_embedding` do expert de
            vídeo mesmo com ``freeze_video_expert=True``. No modo ``latent``
            isso é obrigatório — os canais novos entram zerados e precisam
            aprender —, e é barato: é uma Conv3d, não os 30 blocos de 5B.
    """

    depth_mode: str = DEPTH_MODE_LATENT
    depth_unit: str = "mm"
    depth_min: float = 0.05
    depth_max: float = 5.0
    depth_use_log: bool = True
    depth_train_patch_embedding: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.depth_mode not in DEPTH_MODES:
            raise ValueError(f"`depth_mode` deve ser um de {DEPTH_MODES}, recebeu {self.depth_mode!r}.")
        if self.depth_unit not in ("mm", "m"):
            raise ValueError(f"`depth_unit` deve ser 'mm' ou 'm', recebeu {self.depth_unit!r}.")
        if not 0.0 < self.depth_min < self.depth_max:
            raise ValueError(
                f"É preciso 0 < depth_min < depth_max; recebeu {self.depth_min} e {self.depth_max}."
            )
        if self.depth_use_log and self.depth_min <= 0.0:
            raise ValueError("`depth_min` tem que ser > 0 para o mapeamento logarítmico.")

    # ── Features ────────────────────────────────────────────────────────────
    @property
    def depth_feature_keys(self) -> list[str]:
        """Câmeras de profundidade declaradas em `input_features`.

        A regra é o sufixo do nome (`..._depth`), a mesma que o resto do repo
        usa — `robot/unitree_g1/unitree_g1.py` decide por ela quantos canais
        declarar, e o `cameras/zmq/camera_zmq.py` decide por ela como decodificar.
        """
        if not self.input_features:
            return []
        return sorted(k for k in self.input_features if k.startswith("observation.images.") and k.endswith("depth"))

    @property
    def rgb_feature_keys(self) -> list[str]:
        """Câmeras de cor, na MESMA ordem que o `_stack_video_from_images` monta o vídeo."""
        if not self.input_features:
            return []
        return sorted(
            k
            for k in self.input_features
            if k.startswith("observation.images.") and not k.endswith("depth")
        )

    def rgb_owner_of(self, depth_key: str) -> str | None:
        """Câmera de cor a que um mapa de profundidade pertence.

        `observation.images.head_camera_depth` → `observation.images.head_camera`.
        É isso que põe a profundidade no MESMO pedaço da imagem concatenada que
        a cor dela ocupa — sem isso a geometria entraria deslocada, o que é
        pior do que não ter profundidade nenhuma.
        """
        dono = depth_key.removesuffix("_depth")
        return dono if dono in self.input_features else None

    def set_dataset_feature_metadata(self, dataset_features: dict[str, Any]) -> None:
        """Reconstrói as features visuais a partir das chaves reais do dataset.

        O `make_policy` chama este gancho assim que conhece o dataset e ele
        DESCARTA o `input_features` do YAML. O de origem
        (`FastWAMConfig.set_dataset_feature_metadata`) trata toda chave
        `observation.images.*` como câmera de cor: reparte a largura do mosaico
        entre TODAS elas e declara cada uma com 3 canais. Com o mapa de
        profundidade no dataset isso erra duas vezes — a profundidade nasceria
        com 3 canais e 149 px de largura, e as duas câmeras de cor passariam a
        somar 298 em vez de 448 (é o `ValueError` que estoura ao criar a
        política).

        Aqui a largura é repartida SÓ entre as câmeras de cor, e a profundidade
        entra na resolução nativa dela, com os canais que o dataset gravou —
        ela não ocupa largura no mosaico de cor, viaja pelo latente (ver
        `README.md`). Os canais vêm do dataset em vez de fixos em 1 de
        propósito: um dataset antigo, gravado quando a profundidade era imagem
        de 8 bits em 3 canais, tem que estourar no `validate_features` com a
        mensagem que diz o que houve, e não ser silenciosamente redeclarado
        como se fosse métrico.
        """
        chaves_imagem = sorted(
            chave
            for chave, feature in dataset_features.items()
            if chave.startswith("observation.images.") and feature.get("dtype") in ("video", "image")
        )
        if not chaves_imagem:
            return

        chaves_depth = [k for k in chaves_imagem if k.endswith("depth")]
        chaves_rgb = [k for k in chaves_imagem if not k.endswith("depth")]
        if not chaves_rgb:
            raise ValueError(
                "O dataset só tem câmeras de profundidade "
                f"({chaves_depth}). O FastWAM-D é um enxerto no vídeo de COR: "
                "sem uma câmera de cor não há mosaico onde encaixar a geometria."
            )

        altura, largura_total = self.image_size
        largura_camera = largura_total // len(chaves_rgb)
        novas: dict[str, PolicyFeature] = {
            chave: PolicyFeature(type=FeatureType.VISUAL, shape=(3, altura, largura_camera))
            for chave in chaves_rgb
        }

        for chave in chaves_depth:
            forma = tuple(dataset_features[chave].get("shape") or ())
            if len(forma) != 3:
                raise ValueError(
                    f"Feature de profundidade `{chave}` com shape {forma} no dataset; "
                    "esperava (H, W, C)."
                )
            h_depth, w_depth, c_depth = (int(v) for v in forma)
            # A resolução nativa é preservada: quem redimensiona é o
            # `monta_video_profundidade`, que precisa da fatia do mosaico e não
            # da largura por câmera declarada aqui.
            novas[chave] = PolicyFeature(type=FeatureType.VISUAL, shape=(c_depth, h_depth, w_depth))

        if self.proprio_dim is not None and OBS_STATE in dataset_features:
            novas[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(self.proprio_dim,))

        self.input_features = novas
        self.validate_features()

    def validate_features(self) -> None:
        """Valida as features de cor com a regra do FastWAM, ignorando a profundidade.

        O `validate_features` de origem exige que TODA feature de imagem tenha
        3 canais e que as larguras somem `image_size[1]` — ele foi escrito
        assumindo que imagem = câmera de cor. O mapa de profundidade tem 1
        canal e não ocupa largura nenhuma no mosaico (ele viaja por fora, pelo
        latente), então sai da conta antes de chamar o de cima.
        """
        chaves_depth = self.depth_feature_keys
        if not chaves_depth:
            super().validate_features()
            return

        guardadas = {k: self.input_features.pop(k) for k in chaves_depth}
        try:
            super().validate_features()
        finally:
            self.input_features.update(guardadas)

        for chave, feature in guardadas.items():
            shape = tuple(feature.shape)
            if len(shape) != 3 or shape[0] != 1:
                raise ValueError(
                    f"Feature de profundidade `{chave}` precisa ter shape (1, H, W) — o dataset "
                    f"a grava como mapa de 1 canal em milímetros. Recebeu {shape}."
                )
            if self.depth_mode != DEPTH_MODE_OFF and self.rgb_owner_of(chave) is None:
                raise ValueError(
                    f"`{chave}` não tem câmera de cor correspondente em `input_features`. "
                    f"Esperava `{chave.removesuffix('_depth')}` — é ela que define em que "
                    f"pedaço do mosaico a profundidade entra."
                )
