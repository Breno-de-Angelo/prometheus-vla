import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# UTILITÁRIO: carregamento de pesos pré-treinados
# ══════════════════════════════════════════════════════════════

def _load_pretrained_weights(
    model: nn.Module,
    source: str,
    cache_dir: Optional[str] = None,
    prefix_remap: Optional[dict] = None,
) -> nn.Module:
    """
    Carrega pesos pré-treinados em `model` com strict=False.

    `source` pode ser:
      - Caminho local absoluto ou relativo:  "/data/ckpts/pointnet.pth"
      - URL HuggingFace (hf://):             "hf://danasone/dp3-pointnet/pointnet.pth"
      - URL direta (https://):               "https://example.com/model.pth"
      - None / "none" / "":                  sem pré-treino (skip silencioso)

    `prefix_remap` é um dicionário opcional para renomear prefixos de chaves
    do checkpoint antes de tentar o match:
        {"encoder.": ""}   →  remove o prefixo "encoder." de todas as chaves

    Retorna o modelo (in-place) com as chaves compatíveis carregadas.
    """
    if not source or source.lower() in ("none", "false", ""):
        logger.info("[PretrainedDepth] Nenhuma fonte configurada — iniciando do zero.")
        return model

    # ── Resolve o checkpoint ──────────────────────────────────
    ckpt_path: Optional[str] = None

    if source.startswith("hf://"):
        # Formato: hf://repo_id/filename
        # Ex:      hf://danasone/dp3-pointnet/pointnet_encoder.pth
        rest = source[len("hf://"):]
        parts = rest.split("/")
        if len(parts) < 2:
            raise ValueError(f"URL HuggingFace inválida: '{source}'. Formato: hf://repo_id/filename")
        repo_id = "/".join(parts[:-1])
        filename = parts[-1]
        try:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
            )
            logger.info(f"[PretrainedDepth] Baixado de HuggingFace: {repo_id}/{filename}")
        except ImportError:
            raise ImportError(
                "Instale huggingface_hub para usar fontes hf://: "
                "pip install huggingface_hub"
            )

    elif source.startswith("https://") or source.startswith("http://"):
        ckpt_path = _download_url(source, cache_dir)

    else:
        # Caminho local
        ckpt_path = source
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"[PretrainedDepth] Arquivo não encontrado: '{ckpt_path}'"
            )

    # ── Carrega o state_dict ──────────────────────────────────
    raw = torch.load(ckpt_path, map_location="cpu")

    # Checkpoints podem ter o state_dict em sub-chaves comuns
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "model_state_dict", "encoder"):
            if key in raw:
                raw = raw[key]
                logger.info(f"[PretrainedDepth] Usando sub-chave '{key}' do checkpoint.")
                break

    # Aplica remapeamento de prefixos
    if prefix_remap:
        remapped = {}
        for k, v in raw.items():
            new_k = k
            for old_prefix, new_prefix in prefix_remap.items():
                if k.startswith(old_prefix):
                    new_k = new_prefix + k[len(old_prefix):]
                    break
            remapped[new_k] = v
        raw = remapped

    # ── Carrega com strict=False (ignora chaves incompatíveis) ─
    missing, unexpected = model.load_state_dict(raw, strict=False)

    total_ckpt  = len(raw)
    total_model = len(model.state_dict())
    loaded      = total_model - len(missing)

    logger.info(
        f"[PretrainedDepth] Carregado: {loaded}/{total_model} tensors compatíveis "
        f"| checkpoint tinha {total_ckpt} chaves "
        f"| faltando {len(missing)} | inesperado {len(unexpected)}"
    )
    if missing:
        logger.debug(f"[PretrainedDepth] Chaves faltando (iniciadas do zero): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        logger.debug(f"[PretrainedDepth] Chaves ignoradas do checkpoint: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    return model


def _download_url(url: str, cache_dir: Optional[str]) -> str:
    """Download simples via torch.hub com cache."""
    import hashlib
    fname = hashlib.md5(url.encode()).hexdigest() + "_" + url.split("/")[-1]
    cache = Path(cache_dir or Path.home() / ".cache" / "prometheus_depth")
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / fname
    if dest.exists():
        logger.info(f"[PretrainedDepth] Usando cache: {dest}")
        return str(dest)
    logger.info(f"[PretrainedDepth] Baixando: {url}")
    import urllib.request
    urllib.request.urlretrieve(url, dest)
    return str(dest)


# ══════════════════════════════════════════════════════════════
# POINTNET ENCODER
# ══════════════════════════════════════════════════════════════
class PointNetEncoder(nn.Module):
    """
    Codificador 3D clássico para nuvem de pontos.

    Pontos fortes:
      - Leve e rápido (poucos parâmetros)
      - Bom para objetos com geometria simples
      - Treinamento estável, sem atenção

    Limitações:
      - Max-pooling global perde estrutura local da cena
      - Sem modelagem de relações entre pontos vizinhos
      - Menos discriminativo para cenas complexas multi-objeto

    Compatibilidade com pesos pré-treinados (strict=False):
      - DP3 (Diffusion Policy 3D): layers conv1/conv2/conv3 carregam direto
      - Pesos próprios salvos via save_pretrained_depth_encoder()
    """
    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1   = nn.Linear(1024, 512)
        self.fc2   = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, N]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # [B, output_dim]


# ══════════════════════════════════════════════════════════════
# POINT TRANSFORMER ENCODER
# ══════════════════════════════════════════════════════════════

def _knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    k vizinhos mais próximos — puro PyTorch, zero custom ops.

    Args:
        x: [B, 3, N]
        k: número de vizinhos
    Returns:
        idx: [B, N, k]
    """
    x_t   = x.permute(0, 2, 1)                                         # [B, N, 3]
    inner = torch.bmm(x_t, x)                                          # [B, N, N]
    sq    = (x ** 2).sum(dim=1, keepdim=True).permute(0, 2, 1)         # [B, N, 1]
    dist  = sq + sq.permute(0, 2, 1) - 2 * inner                       # [B, N, N]
    idx   = dist.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]     # [B, N, k]
    return idx


class PointTransformerLayer(nn.Module):
    """
    Camada de atenção vetorial do Point Transformer (Zhao et al., 2021).

    Para cada ponto p_i:
      1. Projeta features p_i e seus k-NNs {p_j} em Q, K, V
      2. Calcula position encoding relativo: delta(p_i - p_j)
      3. Attention weight: softmax( gamma( phi(p_i) - psi(p_j) + delta ) )
      4. Output: soma ponderada dos valores + encoding posicional

    Compatibilidade com pesos externos (strict=False):
      - Chaves phi/psi/alpha/delta/gamma/proj_out/norm carregam direto
        se o checkpoint usar as mesmas dimensões.
    """

    def __init__(self, in_dim: int, out_dim: int, k: int = 16):
        super().__init__()
        self.k       = k
        self.out_dim = out_dim

        self.phi   = nn.Linear(in_dim, out_dim)   # query
        self.psi   = nn.Linear(in_dim, out_dim)   # key
        self.alpha = nn.Linear(in_dim, out_dim)   # value

        self.delta = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.gamma = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        self.proj_out = nn.Linear(out_dim, out_dim)
        self.norm     = nn.LayerNorm(out_dim)
        self.skip     = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   [B, N, C]
            xyz: [B, N, 3]
        Returns:
            [B, N, out_dim]
        """
        B, N, C = x.shape

        idx     = _knn(xyz.permute(0, 2, 1), self.k)                    # [B, N, k]
        idx_exp = idx.unsqueeze(-1).expand(B, N, self.k, C)
        x_exp   = x.unsqueeze(2).expand(B, N, N, C)
        x_nbr   = torch.gather(x_exp, 2, idx_exp)                       # [B, N, k, C]

        idx_xyz = idx.unsqueeze(-1).expand(B, N, self.k, 3)
        xyz_exp = xyz.unsqueeze(2).expand(B, N, N, 3)
        xyz_nbr = torch.gather(xyz_exp, 2, idx_xyz)                     # [B, N, k, 3]

        xyz_diff = xyz.unsqueeze(2) - xyz_nbr                           # [B, N, k, 3]
        pos_enc  = self.delta(xyz_diff)                                  # [B, N, k, out_dim]

        q      = self.phi(x).unsqueeze(2)                               # [B, N, 1, out_dim]
        k_feat = self.psi(x_nbr)                                        # [B, N, k, out_dim]
        v      = self.alpha(x_nbr) + pos_enc                            # [B, N, k, out_dim]

        attn = self.gamma(q - k_feat + pos_enc)                         # [B, N, k, out_dim]
        attn = F.softmax(attn, dim=2)

        out = (attn * v).sum(dim=2)                                     # [B, N, out_dim]
        out = self.proj_out(out)
        out = self.norm(out + self.skip(x))
        return out


class PointTransformerEncoder(nn.Module):
    """
    Encoder Point Transformer para nuvem de pontos — puro PyTorch.

    Pontos fortes vs PointNet:
      - Atenção LOCAL entre vizinhos k-NN
      - Position encoding relativo
      - Mais discriminativo para cenas com múltiplos objetos

    Parâmetros recomendados (YAML):
      point_transformer_k:      16    # vizinhos (8-32)
      point_transformer_layers: 3     # profundidade (2-4)
      point_transformer_dim:    256   # dim interna (128-512)

    Compatibilidade com pesos externos (strict=False):
      - Chaves input_embed / layers.N.{phi,psi,alpha,delta,gamma,proj_out,norm}
        carregam direto se hidden_dim bater.
      - global_attn e head geralmente não carregam (dims diferentes) →
        inicializados do zero automaticamente.
    """

    def __init__(
        self,
        output_dim: int = 512,
        k: int = 16,
        num_layers: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.k = k

        self.input_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        dims = [hidden_dim] * (num_layers + 1)
        self.layers = nn.ModuleList([
            PointTransformerLayer(in_dim=dims[i], out_dim=dims[i + 1], k=k)
            for i in range(num_layers)
        ])

        self.global_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, N]
        Returns:
            [B, output_dim]
        """
        xyz  = x.permute(0, 2, 1)           # [B, N, 3]
        feat = self.input_embed(xyz)         # [B, N, hidden_dim]

        for layer in self.layers:
            feat = layer(feat, xyz)

        attn_scores  = self.global_attn(feat)          # [B, N, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        global_feat  = (attn_weights * feat).sum(dim=1) # [B, hidden_dim]

        return self.head(global_feat)                   # [B, output_dim]


# ══════════════════════════════════════════════════════════════
# UTILITÁRIO: salvar encoder treinado para reusar como pretrained
# ══════════════════════════════════════════════════════════════

def save_pretrained_depth_encoder(model: nn.Module, path: str) -> None:
    """
    Salva o state_dict do encoder para reusar em experimentos futuros.

    Uso:
        from depth_encoder import save_pretrained_depth_encoder
        save_pretrained_depth_encoder(policy.pointnet, "checkpoints/pt_encoder_ep50.pth")

    Para carregar no próximo treino, no YAML:
        depth_pretrained_weights: "checkpoints/pt_encoder_ep50.pth"
    """
    torch.save({"state_dict": model.state_dict()}, path)
    logger.info(f"[PretrainedDepth] Encoder salvo em: {path}")


# ══════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════

def build_depth_encoder(config) -> nn.Module:
    """
    Cria o encoder de profundidade baseado em config.

    Parâmetros relevantes no YAML
    ─────────────────────────────
    depth_encoder_type: "pointnet"          # ou "point_transformer"
    point_transformer_k: 16
    point_transformer_layers: 3
    point_transformer_dim: 256

    # Pesos pré-treinados — QUALQUER UM dos formatos abaixo:
    #
    # 1. Sem pré-treino (padrão — inicia do zero):
    depth_pretrained_weights: null
    #
    # 2. Arquivo local (seu próprio checkpoint salvo):
    depth_pretrained_weights: "/data/ckpts/pointnet_run1.pth"
    #
    # 3. HuggingFace (baixa automático com cache):
    depth_pretrained_weights: "hf://danasone/dp3-pointnet/pointnet_encoder.pth"
    #
    # 4. URL direta:
    depth_pretrained_weights: "https://example.com/encoder.pth"
    #
    # Remapeamento de prefixos do checkpoint (opcional):
    # Útil quando o checkpoint foi salvo com um wrapper diferente.
    # Ex: chaves "encoder.conv1.weight" → remove "encoder."
    depth_pretrained_prefix_remap:
      "encoder.": ""
      "backbone.": ""
    """
    encoder_type  = getattr(config, "depth_encoder_type", "pointnet")
    output_dim    = config.dim_model
    pretrained    = getattr(config, "depth_pretrained_weights", None)
    prefix_remap  = getattr(config, "depth_pretrained_prefix_remap", None)
    cache_dir     = getattr(config, "depth_pretrained_cache_dir", None)

    # ── Instancia ──────────────────────────────────────────────
    if encoder_type == "point_transformer":
        k          = getattr(config, "point_transformer_k", 16)
        num_layers = getattr(config, "point_transformer_layers", 3)
        hidden_dim = getattr(config, "point_transformer_dim", 256)
        logger.info(
            f"[ACT-D] Usando Point Transformer — "
            f"k={k}, layers={num_layers}, hidden_dim={hidden_dim}, output_dim={output_dim}"
        )
        model = PointTransformerEncoder(
            output_dim=output_dim,
            k=k,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )
    else:
        logger.info(f"[ACT-D] Usando PointNet — output_dim={output_dim}")
        model = PointNetEncoder(output_dim=output_dim)

    # ── Carrega pesos pré-treinados (se configurado) ───────────
    if pretrained:
        model = _load_pretrained_weights(
            model,
            source=pretrained,
            cache_dir=cache_dir,
            prefix_remap=prefix_remap,
        )
    else:
        logger.info("[ACT-D] depth_pretrained_weights não configurado — iniciando do zero.")

    return model


# ══════════════════════════════════════════════════════════════
# depth_to_pointcloud (inalterado)
# ══════════════════════════════════════════════════════════════

def depth_to_pointcloud(
    depth_tensor: torch.Tensor,
    intrinsics: dict,
    num_points: int = 1024,
    depth_unit: str = "mm",
    z_max: float = 5.0,
) -> torch.Tensor:
    """
    Projeta mapa de profundidade em nuvem de pontos 3D.

    O tensor chega `[B, 1, H, W]` na unidade nativa do dataset — MILÍMETROS,
    que é o padrão do LeRobot 0.6.1 (`depth_output_unit`). Nada de normalizar:
    `processor_act.py` tira a profundidade do normalizador de propósito, porque
    a projeção pinhole precisa da distância métrica de verdade.

    Isto mudou: até a migração o dataset guardava profundidade como imagem RGB
    de 8 bits (0–2000 mm espremidos em 0–255), então o tensor chegava em [0,1]
    e o código fazia `z = tensor * 2.0`. Com o mapa de profundidade nativo esse
    fator ficou errado por três ordens de grandeza — um dataset gravado no
    formato novo treinado com o fator velho põe a cena a 1,2 km de distância.

    Args:
        depth_unit: unidade do tensor de entrada, "mm" ou "m". Ela vem de
            `dataset.depth_output_unit`; o resto da função trabalha em metros.
        z_max: distância máxima (m) aceita na nuvem — ver o filtro lá embaixo.
    """
    if depth_unit not in ("mm", "m"):
        raise ValueError(f"depth_unit deve ser 'mm' ou 'm', recebeu {depth_unit!r}")
    para_metros = 0.001 if depth_unit == "mm" else 1.0

    B, C, H, W = depth_tensor.shape
    device = depth_tensor.device

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    grid_x = grid_x.float().unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.float().unsqueeze(0).expand(B, -1, -1)

    z  = depth_tensor[:, 0, :, :] * para_metros
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    x  = (grid_x - cx) * z / fx
    y  = (grid_y - cy) * z / fy

    point_cloud = torch.stack((x, y, z), dim=1).view(B, 3, -1)

    sampled_pcs = []
    for b in range(B):
        pc         = point_cloud[b]
        # Piso de 5 cm: pixel sem medida volta da desquantização como o próprio
        # `depth_min` (1 cm), não como zero — sem o piso ele viraria uma parede
        # falsa colada na lente. Teto de `z_max` porque a RealSense devolve
        # alguns pixels saturados (o dataset tem max de 65 m); um punhado deles
        # domina a escala da nuvem e a PointNet aprende o ruído.
        valid_mask = (pc[2, :] > 0.05) & (pc[2, :] < z_max)
        valid_pc   = pc[:, valid_mask]

        if valid_pc.shape[1] >= num_points:
            indices = torch.randperm(valid_pc.shape[1], device=device)[:num_points]
            sampled_pcs.append(valid_pc[:, indices])
        else:
            pad = torch.zeros((3, num_points - valid_pc.shape[1]), device=device)
            sampled_pcs.append(torch.cat([valid_pc, pad], dim=1))

    return torch.stack(sampled_pcs)  # [B, 3, num_points]