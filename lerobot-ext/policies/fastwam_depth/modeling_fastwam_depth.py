#!/usr/bin/env python
"""FastWAM-D — FastWAM enxertado com o mapa de profundidade métrico do dataset.

Ver `README.md` para o desenho e o porquê de cada escolha. Em uma frase: a
profundidade é codificada pelo MESMO VAE congelado do Wan e concatenada ao
latente do vídeo no eixo de canais, na entrada do `patch_embedding` — que é
ampliado com **inicialização em zero** para não estragar o pré-treino.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.policies.fastwam.modeling_fastwam import (
    FastWAMPolicy,
    _batch_to_infer_kwargs,
    _action_from_model_output,
    _resize_frames,
)

from .configuration_fastwam_depth import (
    DEPTH_MODE_LATENT,
    DEPTH_MODE_OFF,
    DEPTH_MODE_TOKEN,
    FastWAMDepthConfig,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Preparo da profundidade
# ══════════════════════════════════════════════════════════════════════════

def mapeia_para_unidade(profundidade: Tensor, config: FastWAMDepthConfig) -> Tensor:
    """Profundidade métrica → [0, 1], que é o domínio de entrada do VAE do Wan.

    O mapeamento é FIXO (vem de `depth_min`/`depth_max` do config), nunca por
    quadro. Normalizar por min/max de cada frame parece inofensivo e destrói
    justamente o que se quer: a escala vira relativa, e o modelo aprende
    contraste de profundidade em vez de distância. Um objeto a 40 cm numa cena
    vazia ficaria idêntico a um a 3 m numa cena cheia.

    Em log por padrão, pela mesma razão que o LeRobot grava em log
    (`datasets/depth_utils.py::quantize_depth`): o erro do sensor cresce com a
    distância, então resolução perto vale mais para manipulação.

    Pixel sem medida vira 0 (o fundo do intervalo): o dataset devolve o próprio
    `depth_min` da quantização (~1 cm) para eles, e 0 é o que menos mente.
    """
    escala = 0.001 if config.depth_unit == "mm" else 1.0
    metros = profundidade.to(dtype=torch.float32) * escala

    inferior = float(config.depth_min)
    superior = float(config.depth_max)
    valido = torch.isfinite(metros) & (metros > inferior)

    if config.depth_use_log:
        import math

        seguro = metros.clamp(min=inferior)
        normal = (torch.log(seguro) - math.log(inferior)) / (math.log(superior) - math.log(inferior))
    else:
        normal = (metros - inferior) / (superior - inferior)

    return torch.where(valido, normal.clamp(0.0, 1.0), torch.zeros_like(normal))


def monta_video_profundidade(
    batch: dict[str, Tensor], config: FastWAMDepthConfig, num_frames: int
) -> Tensor | None:
    """Monta o "vídeo" de profundidade no MESMO mosaico do vídeo de cor.

    O FastWAM concatena as câmeras lado a lado na largura, em ordem alfabética
    da chave (`_stack_video_from_images`). Aqui a profundidade é montada com o
    mesmo layout: cada câmera de cor ocupa uma fatia, e a fatia recebe o mapa
    de profundidade DELA — ou zeros, quando aquela câmera não tem profundidade.

    É isso que faz o token do DiT que olha para um pedaço da imagem receber a
    distância daquele mesmo pedaço. Empilhar a profundidade em outra ordem (ou
    esticá-la sobre o mosaico inteiro) seria pior do que não ter profundidade:
    o modelo aprenderia uma correspondência espacial falsa.

    Devolve `[B, 3, T, H, W]` em [0, 1] (3 canais porque é o VAE do Wan que
    codifica, e ele espera RGB), ou None se não houver nenhuma profundidade.
    """
    chaves_rgb = [k for k in config.rgb_feature_keys if k in batch]
    if not chaves_rgb:
        return None

    mapa_depth = {}
    for chave_depth in config.depth_feature_keys:
        if chave_depth not in batch:
            continue
        dono = config.rgb_owner_of(chave_depth)
        if dono is not None:
            mapa_depth[dono] = batch[chave_depth]
    if not mapa_depth:
        return None

    altura = int(config.image_size[0])
    largura_camera = int(config.image_size[1]) // len(chaves_rgb)

    fatias = []
    for chave_rgb in chaves_rgb:
        profundidade = mapa_depth.get(chave_rgb)
        if profundidade is None:
            # Câmera sem sensor de profundidade: fatia zerada. O zero já é o
            # "sem medida" do mapeamento, então o modelo vê a mesma coisa que vê
            # num pixel inválido — não um valor de distância inventado.
            referencia = batch[chave_rgb]
            forma = list(referencia.shape)
            forma[-3] = 1  # 1 canal
            profundidade = torch.zeros(forma, dtype=torch.float32, device=referencia.device)
        else:
            profundidade = mapeia_para_unidade(profundidade, config)
        fatias.append(_resize_frames(profundidade, (altura, largura_camera)))

    imagem = torch.cat(fatias, dim=-1) if len(fatias) > 1 else fatias[0]

    if imagem.ndim == 4:
        # [B, 1, H, W] — quadro único (inferência ao vivo) → repete no tempo.
        imagem = imagem.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)
    elif imagem.ndim == 5:
        # [B, T, 1, H, W] — pilha temporal do delta-timestamp → [B, 1, T, H, W].
        imagem = imagem.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Profundidade com shape inesperado: {tuple(imagem.shape)}")

    # O VAE do Wan espera 3 canais; replicamos o mapa. Custa banda de VAE, mas
    # evita treinar um encoder novo só para a profundidade e mantém o latente
    # exatamente alinhado com o do RGB, que é o ponto de todo o enxerto.
    return imagem.repeat(1, 3, 1, 1, 1)


# ══════════════════════════════════════════════════════════════════════════
# Política
# ══════════════════════════════════════════════════════════════════════════

class FastWAMDepthPolicy(FastWAMPolicy):
    """FastWAM com profundidade métrica no latente (ou como token de contexto)."""

    config_class = FastWAMDepthConfig
    name = "fastwamdepth"

    def __init__(self, config: FastWAMDepthConfig, dataset_stats=None, **kwargs: Any):
        super().__init__(config, dataset_stats, **kwargs)
        self.config: FastWAMDepthConfig = config

        # Latentes de profundidade do forward corrente. É estado de vida curta:
        # quem escreve limpa no finally. O gancho do `patchify` levanta erro se
        # achar isto vazio, em vez de concatenar zeros — treinar com profundidade
        # silenciosamente ausente é o pior desfecho possível aqui.
        self._latentes_depth: Tensor | None = None
        self._canais_video: int = 0

        if config.depth_mode == DEPTH_MODE_LATENT:
            self._amplia_patch_embedding()
            self._instala_gancho_patchify()
        elif config.depth_mode == DEPTH_MODE_TOKEN:
            self._constroi_token_profundidade()

    # ── Montagem do enxerto ─────────────────────────────────────────────────
    @property
    def _expert_video(self) -> nn.Module:
        expert = getattr(self.model, "video_expert", None)
        if expert is None:
            raise RuntimeError("FastWAM-D precisa do `video_expert` do FastWAM.")
        return expert

    def _amplia_patch_embedding(self) -> None:
        """Duplica os canais de entrada da Conv3d de patch, com os novos ZERADOS.

        `wan/model.py`: `patch_embedding = nn.Conv3d(in_dim, dim, k=patch, s=patch)`.
        Os 48 primeiros canais continuam sendo os pesos pré-treinados do Wan; os
        48 novos (a profundidade) entram zerados. No passo zero o modelo produz
        EXATAMENTE a mesma saída de antes — o prior de vídeo fica intacto — e a
        contribuição da profundidade cresce a partir do treino.

        Inicializar os canais novos com ruído (o padrão do PyTorch) injetaria
        lixo num modelo de 5B já treinado logo no primeiro passo.
        """
        conv: nn.Conv3d = self._expert_video.patch_embedding
        self._canais_video = int(conv.in_channels)
        canais_novos = self._canais_video * 2  # vídeo + profundidade, mesmo VAE

        ampliada = nn.Conv3d(
            canais_novos,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        ).to(device=conv.weight.device, dtype=conv.weight.dtype)

        with torch.no_grad():
            ampliada.weight.zero_()
            ampliada.weight[:, : self._canais_video].copy_(conv.weight)
            if conv.bias is not None:
                ampliada.bias.copy_(conv.bias)

        self._expert_video.patch_embedding = ampliada

        if self.config.depth_train_patch_embedding:
            # O `freeze_video_expert` do config congela os 5B do expert de vídeo,
            # e sem esta linha congelaria junto os canais novos — que nasceram
            # zerados. O modelo treinaria para sempre sem enxergar profundidade.
            ampliada.requires_grad_(True)

        logger.info(
            f"[FastWAM-D] patch_embedding {self._canais_video} → {canais_novos} canais "
            f"(novos em zero, treináveis={ampliada.weight.requires_grad})."
        )

    def _instala_gancho_patchify(self) -> None:
        """Concatena a profundidade ao latente na única porta que o DiT usa.

        `wan/video_dit.py::pre_dit` chama `self.patchify(x)` uma vez por forward,
        e é o único lugar onde o latente vira token. Envolver o método aqui
        evita ter que passar um argumento novo por toda a cadeia
        (`pre_dit` → `training_loss` / `infer_action`), que é código do fork
        e vai ser rebaseado.
        """
        expert = self._expert_video
        patchify_original = expert.patchify

        def patchify_com_profundidade(x: Tensor) -> Tensor:
            profundidade = self._latentes_depth
            if profundidade is None:
                raise RuntimeError(
                    "FastWAM-D: latentes de profundidade ausentes no forward. Isto é bug de "
                    "quem chamou (o depth é preparado em `forward`/`predict_action_chunk`), "
                    "não dado faltando — concatenar zeros aqui treinaria o modelo às cegas."
                )
            if profundidade.shape[0] != x.shape[0]:
                raise RuntimeError(
                    f"FastWAM-D: batch da profundidade ({profundidade.shape[0]}) != do vídeo ({x.shape[0]})."
                )
            if profundidade.shape[2] > x.shape[2]:
                # A inferência roda o expert de vídeo só sobre o PRIMEIRO quadro
                # (`infer_action` faz prefill do KV cache, não gera vídeo), então
                # o latente de profundidade chega mais longo no tempo. Corta.
                profundidade = profundidade[:, :, : x.shape[2]]
            if profundidade.shape[2] != x.shape[2]:
                raise RuntimeError(
                    f"FastWAM-D: T da profundidade ({profundidade.shape[2]}) < T do vídeo ({x.shape[2]})."
                )
            profundidade = profundidade.to(device=x.device, dtype=x.dtype)
            return patchify_original(torch.cat([x, profundidade], dim=1))

        expert.patchify = patchify_com_profundidade

    def _constroi_token_profundidade(self) -> None:
        """Modo `token`: um único token de contexto, no lugar da propriocepção.

        Linha de base barata para o experimento. Sem correspondência espacial:
        o modelo recebe "a cena tem esta geometria", não "este pedaço está a
        tantos centímetros".
        """
        dim_texto = int(self._expert_video.text_dim)
        canais = int(self._expert_video.patch_embedding.in_channels)
        self._canais_video = canais
        self.encoder_token_depth = nn.Linear(canais, dim_texto)
        logger.info(f"[FastWAM-D] token de profundidade: Linear({canais} → {dim_texto}).")

    # ── Carregamento de checkpoint ──────────────────────────────────────────
    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        """Carrega o checkpoint base do FastWAM AMPLIANDO o patch_embedding.

        O loader de origem descarta tensores com shape incompatível e deixa o
        parâmetro no valor recém-inicializado. Para o `patch_embedding` isso
        seria desastroso e silencioso: como ampliamos a entrada de 48 para 96
        canais, o shape não bate, e o modelo começaria com a Conv3d de patch
        ZERADA — jogando fora o pré-treino de vídeo inteiro, que é a única
        razão de usar FastWAM.

        Aqui o tensor do checkpoint é copiado nos canais de vídeo e o restante
        (profundidade) fica em zero, exatamente como no `_amplia_patch_embedding`.
        """
        from safetensors.torch import load_file

        estado = load_file(model_file, device="cpu")
        estado_modelo = model.state_dict()

        ampliados = []
        for chave, tensor_ckpt in list(estado.items()):
            if not chave.endswith("patch_embedding.weight") or chave not in estado_modelo:
                continue
            forma_modelo = tuple(estado_modelo[chave].shape)
            forma_ckpt = tuple(tensor_ckpt.shape)
            if forma_modelo == forma_ckpt:
                continue
            mesma_saida = forma_modelo[0] == forma_ckpt[0] and forma_modelo[2:] == forma_ckpt[2:]
            if not (mesma_saida and forma_modelo[1] > forma_ckpt[1]):
                continue
            ampliado = torch.zeros(forma_modelo, dtype=tensor_ckpt.dtype)
            ampliado[:, : forma_ckpt[1]] = tensor_ckpt
            estado[chave] = ampliado
            ampliados.append(f"{chave}: {forma_ckpt} → {forma_modelo}")

        if ampliados:
            logger.info("[FastWAM-D] patch_embedding ampliado no load: " + "; ".join(ampliados))

        faltando, sobrando = model.load_state_dict(estado, strict=False)
        if faltando:
            logger.warning(f"[FastWAM-D] {len(faltando)} tensores sem valor no checkpoint: {faltando[:8]}")
        if map_location and map_location != "cpu":
            model.to(map_location)
        return model

    # ── Caminho dos dados ───────────────────────────────────────────────────
    def _prepara_profundidade(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Codifica a profundidade e guarda o latente; devolve o batch SEM ela.

        Tirar as chaves de profundidade do batch não é limpeza: o
        `_stack_video_from_images` do FastWAM varre tudo que começa com
        `observation.images.` e concatena na largura. Com o mapa de
        profundidade lá dentro, ou o `torch.cat` quebra por causa dos canais,
        ou (se alguém declarasse 3 canais) ele entraria como se fosse mais uma
        câmera de cor.
        """
        if self.config.depth_mode == DEPTH_MODE_OFF:
            return batch

        limpo = {k: v for k, v in batch.items() if k not in self.config.depth_feature_keys}

        video_depth = monta_video_profundidade(batch, self.config, self.config.model_video_frames)
        if video_depth is None:
            raise KeyError(
                f"FastWAM-D não achou profundidade no batch. Esperava uma de "
                f"{self.config.depth_feature_keys}. Use `depth_mode: off` para rodar sem."
            )

        with torch.no_grad():
            # Sem gradiente de propósito: o VAE é congelado e nada antes dele é
            # treinável, então guardar o grafo só gastaria memória de ativação —
            # e é justamente ela que aperta ao treinar um modelo de 5B.
            latente = self.model._encode_video_latents(
                video_depth.to(device=self.model.device, dtype=self.model.torch_dtype)
            )

        if self.config.depth_mode == DEPTH_MODE_LATENT:
            self._latentes_depth = latente
        else:  # token
            agrupado = latente.mean(dim=(2, 3, 4))  # [B, C]
            token = self.encoder_token_depth(agrupado.to(self.encoder_token_depth.weight.dtype))
            self._latentes_depth = token.unsqueeze(1)  # [B, 1, text_dim]

        return limpo

    def _anexa_token_ao_contexto(self, sample: dict[str, Tensor]) -> None:
        token = self._latentes_depth
        if token is None:
            return
        contexto = sample["context"]
        mascara = sample["context_mask"]
        token = token.to(device=contexto.device, dtype=contexto.dtype)
        sample["context"] = torch.cat([contexto, token], dim=1)
        sample["context_mask"] = torch.cat(
            [mascara, torch.ones((mascara.shape[0], 1), dtype=mascara.dtype, device=mascara.device)],
            dim=1,
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        limpo = self._prepara_profundidade(batch)
        try:
            if self.config.depth_mode == DEPTH_MODE_TOKEN:
                sample = self._batch_to_training_sample(limpo)
                self._anexa_token_ao_contexto(sample)
                perda, metricas = self.model.training_loss(sample)
                return perda, dict(metricas or {})
            return super().forward(limpo)
        finally:
            self._latentes_depth = None

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        """Prediz um chunk de ações, um item do batch por vez.

        O laço por item é o mesmo do FastWAM de origem (o `infer_action` do Wan
        só aceita batch 1). Aqui ele é explícito porque a profundidade guardada
        precisa corresponder ao item que está sendo processado — um estado
        global com o batch inteiro dentro seria fatiado errado.
        """
        if self.config.depth_mode == DEPTH_MODE_OFF:
            return super().predict_action_chunk(batch, **kwargs)

        self.eval()
        limpo = self._prepara_profundidade(batch)
        latentes = self._latentes_depth
        try:
            infer_kwargs = _batch_to_infer_kwargs(batch=limpo, config=self.config)

            if self.config.depth_mode == DEPTH_MODE_TOKEN:
                contexto, mascara = self._contexto_de_inferencia(infer_kwargs)
                infer_kwargs["context"], infer_kwargs["context_mask"] = contexto, mascara
                infer_kwargs["prompt"] = None

            tamanho = int(infer_kwargs["input_image"].shape[0])
            acoes = []
            for i in range(tamanho):
                self._latentes_depth = latentes[i : i + 1]
                por_item = dict(infer_kwargs)
                por_item["input_image"] = infer_kwargs["input_image"][i : i + 1]
                for chave in ("proprio", "context", "context_mask"):
                    valor = infer_kwargs.get(chave)
                    if isinstance(valor, Tensor):
                        por_item[chave] = valor[i : i + 1]
                prompt = infer_kwargs.get("prompt")
                if isinstance(prompt, (list, tuple)):
                    por_item["prompt"] = prompt[i]
                acoes.append(_action_from_model_output(self.model.infer_action(**por_item)))
            saida = torch.cat(acoes, dim=0)
        finally:
            self._latentes_depth = None

        destino = infer_kwargs["input_image"].device
        return saida.to(device=destino, dtype=torch.float32)

    def _contexto_de_inferencia(self, infer_kwargs: dict[str, Any]) -> tuple[Tensor, Tensor]:
        """Contexto de texto já codificado, para poder pendurar o token de profundidade nele."""
        contexto, mascara = infer_kwargs.get("context"), infer_kwargs.get("context_mask")
        if contexto is None or mascara is None:
            prompt = infer_kwargs.get("prompt")
            if prompt is None:
                raise KeyError("FastWAM-D no modo `token` precisa de `prompt` ou de `context` no batch.")
            contexto, mascara = self.model.encode_prompt(prompt)
        token = self._latentes_depth.to(device=contexto.device, dtype=contexto.dtype)
        return (
            torch.cat([contexto, token], dim=1),
            torch.cat(
                [mascara, torch.ones((mascara.shape[0], 1), dtype=mascara.dtype, device=mascara.device)],
                dim=1,
            ),
        )

    # ── Otimizador ──────────────────────────────────────────────────────────
    def get_optim_params(self) -> list[Tensor]:
        """Parâmetros do pai mais os do enxerto, sem repetir.

        O `get_optim_params` de origem devolve os parâmetros do DiT e do encoder
        de propriocepção. O `patch_embedding` ampliado e o encoder do token não
        estão nessa lista — sem isto eles nunca receberiam atualização, e o
        modelo treinaria para sempre com os canais de profundidade em zero.
        """
        params = list(super().get_optim_params())
        vistos = {id(p) for p in params}

        extras: list[Tensor] = []
        if self.config.depth_mode == DEPTH_MODE_LATENT:
            extras.extend(self._expert_video.patch_embedding.parameters())
        elif self.config.depth_mode == DEPTH_MODE_TOKEN:
            extras.extend(self.encoder_token_depth.parameters())

        for p in extras:
            if p.requires_grad and id(p) not in vistos:
                params.append(p)
                vistos.add(id(p))
        return params
