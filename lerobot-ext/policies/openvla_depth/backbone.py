#!/usr/bin/env python
"""
Carregamento do backbone OpenVLA (Prismatic VLM) sem depender do `trust_remote_code`.

## Por que reconstruir em vez de usar AutoModelForVision2Seq

O código remoto publicado junto do `openvla/openvla-7b` foi escrito para
`transformers==4.40.1` e `timm==0.9.10`. O ambiente deste projeto roda
`transformers>=5.x` (exigido pelo LeRobot), e as duas coisas não coexistem no
mesmo venv. Em vez de fixar versões incompatíveis, remontamos aqui a mesma
arquitetura a partir de três peças estáveis:

    vision_backbone   → dois ViTs do timm (DINOv2-L + SigLIP-so400m), fundidos
                        por concatenação no eixo de canais
    projector         → MLP de 3 camadas (fc1 → GELU → fc2 → GELU → fc3)
    language_model    → LlamaModel (transformers)

e carregamos os pesos direto do safetensors do checkpoint. As dimensões de cada
peça são **inferidas do próprio state dict**, não hardcoded — se você trocar o
backbone por outro Prismatic, continua funcionando.

## Detalhe que costuma passar batido

O Prismatic não usa a saída final do ViT: ele pega os patches da **penúltima
camada** (`n = len(blocks) - 2`) via `get_intermediate_layers`, sem LayerNorm
final. Reproduzir isso é obrigatório — usar a saída final desalinha as features
em relação ao projector pré-treinado e o modelo sai lixo sem dar erro nenhum.

## Diagnóstico

Se os prefixos de chave do checkpoint mudarem, rode:

    python -m policies.openvla_depth.backbone --inspect openvla/openvla-7b

que imprime os prefixos reais e as formas, sem carregar o modelo na GPU.
"""

from __future__ import annotations

import glob
import logging
import os
import sys

import torch
import torch.nn as nn

# Prefixos de chave no safetensors do openvla-7b.
PREFIX_DINOV2 = "vision_backbone.featurizer."
PREFIX_SIGLIP = "vision_backbone.fused_featurizer."
PREFIX_PROJECTOR = "projector."
PREFIX_LLM = "language_model.model."

# Estatísticas de normalização de cada torre visual. O Prismatic normaliza a
# MESMA imagem de dois jeitos diferentes e empilha em 6 canais.
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# Download / leitura do checkpoint
# ══════════════════════════════════════════════════════════════════════════════


def _resolve_local_dir(repo_or_path: str) -> str:
    """Retorna um diretório local com os arquivos do checkpoint."""
    if os.path.isdir(repo_or_path):
        return repo_or_path

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_or_path,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    )


def load_openvla_state_dict(repo_or_path: str) -> dict[str, torch.Tensor]:
    """Carrega todos os shards safetensors do checkpoint num único dict (em CPU)."""
    from safetensors.torch import load_file

    local_dir = _resolve_local_dir(repo_or_path)
    shards = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"Nenhum arquivo .safetensors encontrado em '{local_dir}'. "
            "O checkpoint baixou completo?"
        )

    state_dict: dict[str, torch.Tensor] = {}
    for shard in shards:
        state_dict.update(load_file(shard, device="cpu"))
    logging.info(f"[OpenVLA] {len(state_dict)} tensores carregados de {len(shards)} shard(s).")
    return state_dict


def _split_by_prefix(state_dict: dict, prefix: str) -> dict:
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def inspect_checkpoint(repo_or_path: str, depth: int = 2) -> None:
    """Imprime os prefixos de chave e formas do checkpoint (ferramenta de diagnóstico)."""
    state_dict = load_openvla_state_dict(repo_or_path)

    groups: dict[str, list[str]] = {}
    for key in state_dict:
        prefix = ".".join(key.split(".")[:depth])
        groups.setdefault(prefix, []).append(key)

    print(f"\n{'=' * 78}\nCheckpoint: {repo_or_path}\n{'=' * 78}")
    for prefix in sorted(groups):
        keys = groups[prefix]
        sample = keys[0]
        print(f"{prefix:<45} {len(keys):>5} tensores   ex: {tuple(state_dict[sample].shape)}")

    print(f"\n{'-' * 78}\nDimensões inferidas:")
    for name, key in [
        ("projector.fc1.weight", "projector.fc1.weight"),
        ("projector.fc2.weight", "projector.fc2.weight"),
        ("projector.fc3.weight", "projector.fc3.weight"),
        ("llm embed_tokens", "language_model.model.embed_tokens.weight"),
    ]:
        if key in state_dict:
            print(f"  {name:<28} {tuple(state_dict[key].shape)}")
        else:
            print(f"  {name:<28} AUSENTE  ← prefixo mudou, ajuste backbone.py")
    print(f"{'=' * 78}\n")


def compare_keys(
    repo_or_path: str,
    dinov2_model: str = "vit_large_patch14_reg4_dinov2.lvd142m",
    siglip_model: str = "vit_so400m_patch14_siglip_224",
    image_size: int = 224,
) -> bool:
    """
    Confere, componente por componente, se as chaves do checkpoint batem com os
    módulos que construímos localmente.

    Existe por um motivo específico: o `openvla-7b` foi salvo com `timm==0.9.10`,
    e este projeto roda `timm>=1.0`. Se os nomes de parâmetro mudaram entre as
    versões, `build_openvla_backbone` vai falhar — e é muito melhor descobrir
    isso aqui, num diff legível, do que no meio da inicialização do treino.

    O LLM é construído em `meta device`: comparamos os nomes das chaves sem
    alocar os 7B de pesos.

    Retorna True se todos os componentes batem.
    """
    state_dict = load_openvla_state_dict(repo_or_path)
    local_dir = _resolve_local_dir(repo_or_path)
    ok = True

    def _report(name: str, expected: set, found: set) -> bool:
        missing = sorted(k for k in expected - found if "rotary_emb.inv_freq" not in k)
        unexpected = sorted(found - expected)
        status = "OK" if not missing and not unexpected else "DIVERGE"
        print(f"\n─── {name} — {status} ───")
        print(f"    módulo local: {len(expected)} chaves | checkpoint: {len(found)} chaves")
        if missing:
            print(f"    faltando no checkpoint ({len(missing)}): {missing[:10]}")
        if unexpected:
            print(f"    sobrando no checkpoint ({len(unexpected)}): {unexpected[:10]}")
        return not missing and not unexpected

    vision = FusedVisionBackbone(dinov2_model, siglip_model, image_size=image_size)
    ok &= _report(
        f"DINOv2 ({dinov2_model})",
        set(vision.featurizer.state_dict()),
        set(_split_by_prefix(state_dict, PREFIX_DINOV2)),
    )
    ok &= _report(
        f"SigLIP ({siglip_model})",
        set(vision.fused_featurizer.state_dict()),
        set(_split_by_prefix(state_dict, PREFIX_SIGLIP)),
    )

    sd_proj = _split_by_prefix(state_dict, PREFIX_PROJECTOR)
    try:
        projector = _build_projector_from_state_dict(sd_proj)
        ok &= _report("projector", set(projector.state_dict()), set(sd_proj))
    except KeyError as e:
        print(f"\n─── projector — DIVERGE ───\n    {e}")
        ok = False

    sd_llm = _split_by_prefix(state_dict, PREFIX_LLM)
    try:
        with torch.device("meta"):
            llm = _build_llama(local_dir, sd_llm)
        ok &= _report("language_model (Llama)", set(llm.state_dict()), set(sd_llm))
    except Exception as e:
        print(f"\n─── language_model — não foi possível construir ───\n    {e}")
        ok = False

    print(f"\n{'=' * 78}")
    if ok:
        print("Todos os componentes batem — `load_mode: native` deve carregar sem erro.")
    else:
        print(
            "Há divergências acima. Antes de mexer no código, confira a versão do timm:\n"
            "  o checkpoint foi salvo com timm==0.9.10.\n"
            "Se só a torre visual diverge, instalar `timm==0.9.10` num venv separado e\n"
            "reexportar os pesos costuma ser mais simples que remapear as chaves."
        )
    print("=" * 78 + "\n")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Torre visual fundida (DINOv2 + SigLIP)
# ══════════════════════════════════════════════════════════════════════════════


class FusedVisionBackbone(nn.Module):
    """
    Duas torres ViT rodando em paralelo sobre a mesma imagem, concatenadas no
    eixo de features.

    Entrada:  pixel_values [B, 6, 224, 224] — canais 0:3 normalizados para DINOv2,
              canais 3:6 normalizados para SigLIP.
    Saída:    patches [B, 256, 2176]
    """

    def __init__(self, dinov2_model: str, siglip_model: str, image_size: int = 224):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "O backbone visual do OpenVLA precisa do `timm`. Instale com:\n"
                "    pip install 'timm>=1.0.0,<1.1.0'"
            ) from e

        self.featurizer = timm.create_model(
            dinov2_model, pretrained=False, num_classes=0, img_size=image_size
        )
        self.fused_featurizer = timm.create_model(
            siglip_model, pretrained=False, num_classes=0, img_size=image_size
        )

        # Penúltima camada, sem norm final — comportamento do Prismatic.
        self._dinov2_layer = len(self.featurizer.blocks) - 2
        self._siglip_layer = len(self.fused_featurizer.blocks) - 2

        self.embed_dim = self.featurizer.embed_dim + self.fused_featurizer.embed_dim

    @staticmethod
    def _patches(featurizer: nn.Module, x: torch.Tensor, layer: int) -> torch.Tensor:
        out = featurizer.get_intermediate_layers(x, n={layer})
        # get_intermediate_layers devolve tupla de tensores [B, num_patches, D]
        return out[0] if isinstance(out, (tuple, list)) else out

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.shape[1] != 6:
            raise ValueError(
                f"FusedVisionBackbone espera 6 canais (DINOv2 + SigLIP), "
                f"recebeu {pixel_values.shape[1]}. Use build_fused_pixel_values()."
            )
        dino_in, siglip_in = pixel_values[:, :3], pixel_values[:, 3:]
        dino_patches = self._patches(self.featurizer, dino_in, self._dinov2_layer)
        siglip_patches = self._patches(self.fused_featurizer, siglip_in, self._siglip_layer)
        return torch.cat([dino_patches, siglip_patches], dim=2)


def build_fused_pixel_values(rgb: torch.Tensor) -> torch.Tensor:
    """
    Converte RGB em [0, 1] com shape [B, 3, H, W] no tensor de 6 canais que a
    torre fundida espera. A imagem já deve estar em 224×224.
    """
    device, dtype = rgb.device, rgb.dtype

    def _norm(x, mean, std):
        m = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
        s = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)
        return (x - m) / s

    return torch.cat([_norm(rgb, DINOV2_MEAN, DINOV2_STD), _norm(rgb, SIGLIP_MEAN, SIGLIP_STD)], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# Projector
# ══════════════════════════════════════════════════════════════════════════════


class PrismaticProjector(nn.Module):
    """MLP de 3 camadas que leva os patches visuais ao espaço de embedding do LLM."""

    def __init__(self, vision_dim: int, inner_dim: int, llm_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, inner_dim, bias=True)
        self.fc2 = nn.Linear(inner_dim, llm_dim, bias=True)
        self.fc3 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()
        self.act_fn2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.act_fn2(self.fc2(self.act_fn1(self.fc1(x)))))


def _build_projector_from_state_dict(sd: dict) -> PrismaticProjector:
    """Infere as dimensões do projector a partir das formas no checkpoint."""
    required = ["fc1.weight", "fc2.weight", "fc3.weight"]
    missing = [k for k in required if k not in sd]
    if missing:
        raise KeyError(
            f"Projector incompleto no checkpoint (faltam {missing}). "
            f"Chaves presentes: {sorted(sd)[:10]}. "
            "Rode `python -m policies.openvla_depth.backbone --inspect <repo>`."
        )
    inner_dim, vision_dim = sd["fc1.weight"].shape
    llm_dim = sd["fc2.weight"].shape[0]
    return PrismaticProjector(vision_dim, inner_dim, llm_dim)


# ══════════════════════════════════════════════════════════════════════════════
# Backbone completo
# ══════════════════════════════════════════════════════════════════════════════


class OpenVLABackbone(nn.Module):
    """
    Prismatic VLM sem o `lm_head`: entrega hidden states, que é tudo que o head
    OFT precisa. Descartar o lm_head economiza ~260M parâmetros (32000 × 4096).
    """

    def __init__(self, vision: FusedVisionBackbone, projector: PrismaticProjector, language_model):
        super().__init__()
        self.vision = vision
        self.projector = projector
        self.language_model = language_model
        self.llm_dim = language_model.config.hidden_size

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings()(input_ids)

    def embed_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.projector(self.vision(pixel_values))


def _build_llama(local_dir: str, state_dict: dict):
    """Instancia o LlamaModel a partir do config.json do checkpoint Prismatic."""
    import json

    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaModel

    cfg_path = os.path.join(local_dir, "config.json")
    text_config = None
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            raw = json.load(f)
        text_config = raw.get("text_config") or raw.get("llm_config")

    if text_config is None:
        # Sem text_config no config.json: deduz do state dict. Cobre Llama-2 7B.
        vocab_size, hidden = state_dict["embed_tokens.weight"].shape
        n_layers = 1 + max(
            int(k.split(".")[1]) for k in state_dict if k.startswith("layers.") and k.split(".")[1].isdigit()
        )
        logging.warning(
            "[OpenVLA] config.json sem `text_config` — deduzindo LlamaConfig do state dict "
            f"(hidden={hidden}, layers={n_layers}, vocab={vocab_size})."
        )
        text_config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden,
            "num_hidden_layers": n_layers,
            "intermediate_size": state_dict["layers.0.mlp.gate_proj.weight"].shape[0],
            "num_attention_heads": hidden // 128,
            "num_key_value_heads": hidden // 128,
        }

    text_config.pop("architectures", None)
    llama_config = LlamaConfig(**text_config)
    llama_config._attn_implementation = "sdpa"
    return LlamaModel(llama_config)


def build_openvla_backbone(config) -> OpenVLABackbone:
    """
    Constrói o backbone e carrega os pesos do checkpoint.

    `config` é um OPENVLADEPTHConfig. Respeita `config.load_mode`.
    """
    if config.load_mode == "remote_code":
        return _build_via_remote_code(config)

    local_dir = _resolve_local_dir(config.pretrained_backbone)
    state_dict = load_openvla_state_dict(local_dir)

    sd_dino = _split_by_prefix(state_dict, PREFIX_DINOV2)
    sd_siglip = _split_by_prefix(state_dict, PREFIX_SIGLIP)
    sd_proj = _split_by_prefix(state_dict, PREFIX_PROJECTOR)
    sd_llm = _split_by_prefix(state_dict, PREFIX_LLM)

    for name, sd, prefix in [
        ("DINOv2", sd_dino, PREFIX_DINOV2),
        ("SigLIP", sd_siglip, PREFIX_SIGLIP),
        ("projector", sd_proj, PREFIX_PROJECTOR),
        ("language_model", sd_llm, PREFIX_LLM),
    ]:
        if not sd:
            raise KeyError(
                f"Nenhuma chave com prefixo '{prefix}' ({name}) no checkpoint "
                f"'{config.pretrained_backbone}'. Rode:\n"
                f"    python -m policies.openvla_depth.backbone --inspect {config.pretrained_backbone}\n"
                "e ajuste os PREFIX_* no topo de backbone.py."
            )

    vision = FusedVisionBackbone(
        config.dinov2_model, config.siglip_model, image_size=config.image_resolution[0]
    )
    _load_strict(vision.featurizer, sd_dino, "vision.featurizer")
    _load_strict(vision.fused_featurizer, sd_siglip, "vision.fused_featurizer")

    projector = _build_projector_from_state_dict(sd_proj)
    _load_strict(projector, sd_proj, "projector")

    language_model = _build_llama(local_dir, sd_llm)
    _load_strict(language_model, sd_llm, "language_model")

    if vision.embed_dim != projector.fc1.in_features:
        raise ValueError(
            f"Incompatibilidade de dimensão: torre visual entrega {vision.embed_dim} "
            f"features mas o projector espera {projector.fc1.in_features}. "
            f"Confira dinov2_model/siglip_model no YAML."
        )

    logging.info(
        f"[OpenVLA] Backbone pronto — visão {vision.embed_dim}d → projector → "
        f"LLM {language_model.config.hidden_size}d "
        f"({language_model.config.num_hidden_layers} camadas)."
    )
    return OpenVLABackbone(vision, projector, language_model)


def _load_strict(module: nn.Module, sd: dict, name: str) -> None:
    """Carrega pesos e falha alto se sobrar/faltar chave — silêncio aqui vira modelo aleatório."""
    missing, unexpected = module.load_state_dict(sd, strict=False)
    # `rotary_emb.inv_freq` é recomputado pelo transformers moderno e não vem no
    # checkpoint; qualquer outra chave faltando é erro real.
    missing = [k for k in missing if "rotary_emb.inv_freq" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"[OpenVLA] Falha ao carregar '{name}':\n"
            f"  faltando ({len(missing)}): {missing[:8]}\n"
            f"  inesperadas ({len(unexpected)}): {unexpected[:8]}\n"
            "Isso normalmente significa que a versão do timm/transformers mudou os nomes "
            "dos parâmetros. Rode o --inspect e compare."
        )
    logging.info(f"[OpenVLA] '{name}': {len(sd)} tensores carregados.")


def _build_via_remote_code(config) -> OpenVLABackbone:
    """
    Caminho alternativo: usa o código remoto oficial do OpenVLA.

    Só funciona num venv com `transformers==4.40.1` e `timm==0.9.10`, que é
    incompatível com o LeRobot. Mantido para quem quiser validar o backbone
    isoladamente. Ver docs/OPENVLA_DEPTH.md, seção "Ambiente".
    """
    from transformers import AutoModelForVision2Seq

    logging.warning(
        "[OpenVLA] load_mode='remote_code': exige transformers==4.40.1 / timm==0.9.10. "
        "Se o import falhar, use load_mode='native'."
    )
    vla = AutoModelForVision2Seq.from_pretrained(
        config.pretrained_backbone,
        trust_remote_code=True,
        torch_dtype=getattr(torch, config.dtype),
        low_cpu_mem_usage=True,
    )

    vision_raw = vla.vision_backbone
    projector = vla.projector
    language_model = vla.language_model.model  # LlamaForCausalLM → LlamaModel

    # Embrulha a torre remota para expor a mesma interface da nativa.
    class _RemoteVision(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.embed_dim = projector.fc1.in_features

        def forward(self, pixel_values):
            return self.inner(pixel_values)

    return OpenVLABackbone(_RemoteVision(vision_raw), projector, language_model)


# ══════════════════════════════════════════════════════════════════════════════
# CLI de diagnóstico
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Inspeciona um checkpoint OpenVLA/Prismatic.")
    ap.add_argument("--inspect", default="openvla/openvla-7b", help="repo_id ou caminho local")
    ap.add_argument("--depth", type=int, default=2, help="profundidade do agrupamento de prefixos")
    ap.add_argument(
        "--compare",
        action="store_true",
        help="também compara as chaves do checkpoint com os módulos locais "
        "(pega incompatibilidade de versão do timm antes do treino)",
    )
    args = ap.parse_args()
    inspect_checkpoint(args.inspect, depth=args.depth)
    if args.compare:
        sys.exit(0 if compare_keys(args.inspect) else 1)
