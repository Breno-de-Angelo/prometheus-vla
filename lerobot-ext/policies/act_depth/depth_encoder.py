import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# POINTNET ENCODER (arquitetura original)
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
    """
    def __init__(self, output_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        # x shape: [Batch, 3, Num_Points]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        # Max Pooling global (perde estrutura local — limitação do PointNet)
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # [Batch, output_dim]


# ══════════════════════════════════════════════════════════════
# POINT TRANSFORMER ENCODER (nova arquitetura, mais robusta)
# ══════════════════════════════════════════════════════════════

def _knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Retorna os k vizinhos mais próximos para cada ponto.
    
    Args:
        x: [B, 3, N]
        k: número de vizinhos
    Returns:
        idx: [B, N, k] índices dos k-NNs
    """
    # Distância euclidiana ao quadrado via produto interno
    # x^T x: [B, N, N]
    x_t = x.permute(0, 2, 1)   # [B, N, 3]
    inner = torch.bmm(x_t, x)  # [B, N, N]
    sq = (x ** 2).sum(dim=1, keepdim=True).permute(0, 2, 1)  # [B, N, 1]
    dist = sq + sq.permute(0, 2, 1) - 2 * inner              # [B, N, N]
    # Retorna os k menores (excluindo o próprio ponto — idx 0)
    idx = dist.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]  # [B, N, k]
    return idx


def _gather_local(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Coleta os k vizinhos de cada ponto.
    
    Args:
        x:   [B, C, N]
        idx: [B, N, k]
    Returns:
        [B, C, N, k]
    """
    B, C, N = x.shape
    k = idx.shape[2]
    # Expande idx para todos os canais
    idx_expanded = idx.unsqueeze(1).expand(B, C, N, k)  # [B, C, N, k]
    x_expanded = x.unsqueeze(-1).expand(B, C, N, k)     # [B, C, N, k]
    # Coleta ao longo de N
    x_exp2 = x.unsqueeze(2).expand(B, C, N, N)          # [B, C, N, N]
    out = torch.gather(x_exp2, 3, idx_expanded)          # [B, C, N, k]
    return out


class PointTransformerLayer(nn.Module):
    """
    Camada de atenção vetorial do Point Transformer (Zhao et al., 2021).
    
    Diferente do PointNet, calcula atenção entre cada ponto e seus k vizinhos,
    capturando relações locais de forma explícita. Isso é análogo ao que o
    ResNet faz com convoluções — mas invariante à permutação e à densidade.

    Para cada ponto p_i:
      1. Projeta features p_i e seus k-NNs {p_j} em Q, K, V
      2. Calcula position encoding relativo: delta(p_i - p_j)
      3. Attention weight: softmax( gamma( phi(p_i) - psi(p_j) + delta ) )
      4. Output: soma ponderada dos valores + encoding posicional
    """

    def __init__(self, in_dim: int, out_dim: int, k: int = 16):
        super().__init__()
        self.k = k
        self.out_dim = out_dim

        # Projeções Q, K, V
        self.phi = nn.Linear(in_dim, out_dim)   # query
        self.psi = nn.Linear(in_dim, out_dim)   # key
        self.alpha = nn.Linear(in_dim, out_dim) # value

        # Position encoding relativo (coordenadas 3D → out_dim)
        self.delta = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        # MLP de atenção: mapeia diferença QK+pos para pesos de atenção
        self.gamma = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        # Projeção de saída
        self.proj_out = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        # Skip connection se dimensões diferem
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   [B, N, C]  features dos pontos
            xyz: [B, N, 3]  coordenadas 3D dos pontos
        Returns:
            [B, N, out_dim]
        """
        B, N, C = x.shape

        # 1. Acha os k vizinhos baseado em coordenadas xyz
        idx = _knn(xyz.permute(0, 2, 1), self.k)  # [B, N, k]

        # 2. Coleta features dos vizinhos: [B, N, k, C]
        x_flat = x  # [B, N, C]

        # Gather neighbors para features e coordenadas
        # idx: [B, N, k]
        idx_exp = idx.unsqueeze(-1).expand(B, N, self.k, C)  # [B, N, k, C]
        x_exp = x_flat.unsqueeze(2).expand(B, N, N, C)       # [B, N, N, C]
        x_nbr = torch.gather(x_exp, 2, idx_exp)              # [B, N, k, C]

        idx_xyz = idx.unsqueeze(-1).expand(B, N, self.k, 3)  # [B, N, k, 3]
        xyz_exp = xyz.unsqueeze(2).expand(B, N, N, 3)        # [B, N, N, 3]
        xyz_nbr = torch.gather(xyz_exp, 2, idx_xyz)          # [B, N, k, 3]

        # 3. Diferença de posição relativa (position encoding)
        xyz_diff = xyz.unsqueeze(2) - xyz_nbr  # [B, N, k, 3]
        pos_enc = self.delta(xyz_diff)          # [B, N, k, out_dim]

        # 4. Q, K, V
        q = self.phi(x).unsqueeze(2)     # [B, N, 1, out_dim]
        k_feat = self.psi(x_nbr)        # [B, N, k, out_dim]
        v = self.alpha(x_nbr) + pos_enc # [B, N, k, out_dim]  (valor + pos)

        # 5. Pesos de atenção vetorial
        attn = self.gamma(q - k_feat + pos_enc)  # [B, N, k, out_dim]
        attn = F.softmax(attn, dim=2)             # normaliza sobre vizinhos

        # 6. Agrega com atenção
        out = (attn * v).sum(dim=2)   # [B, N, out_dim]
        out = self.proj_out(out)

        # 7. Residual + LayerNorm
        out = self.norm(out + self.skip(x))
        return out


class PointTransformerEncoder(nn.Module):
    """
    Encoder Point Transformer para nuvem de pontos.
    
    Pontos fortes vs PointNet:
      - Atenção LOCAL entre vizinhos k-NN (não só max-pooling global)
      - Position encoding relativo: capta a geometria local explicitamente
      - Muito mais discriminativo para cenas complexas com múltiplos objetos
      - Performance comparável ao ResNet para imagens, mas para nuvens 3D
    
    Limitações vs PointNet:
      - Mais lento e pesado (k-NN a cada forward, mais parâmetros)
      - Requer mais dados para convergir bem
      - Sensível ao valor de k e ao número de pontos amostrados
    
    Parâmetros recomendados (YAML):
      pointnet_num_points: 512  # Reduzir de 1024 se VRAM apertada
      point_transformer_k: 16  # Vizinhos (8-32, equilíbrio custo-qualidade)
      point_transformer_layers: 3  # Profundidade (2-4)
      point_transformer_dim: 256   # Dim interna (128-512)
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

        # Embedding inicial: 3D coords → hidden_dim
        self.input_embed = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pilha de camadas Point Transformer
        dims = [hidden_dim] + [hidden_dim] * (num_layers - 1) + [hidden_dim]
        self.layers = nn.ModuleList([
            PointTransformerLayer(in_dim=dims[i], out_dim=dims[i + 1], k=k)
            for i in range(num_layers)
        ])

        # Pooling: atenção global (mais expressivo que max-pooling do PointNet)
        self.global_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Projeção final para output_dim (dim_model do ACT)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, N]  nuvem de pontos (saída de depth_to_pointcloud)
        Returns:
            [B, output_dim]
        """
        B, _, N = x.shape

        # Transpõe: [B, N, 3]
        xyz = x.permute(0, 2, 1)

        # Embedding inicial usando coordenadas XYZ como features
        feat = self.input_embed(xyz)  # [B, N, hidden_dim]

        # Aplica as camadas PT com self-atenção local
        for layer in self.layers:
            feat = layer(feat, xyz)   # [B, N, hidden_dim]

        # Pooling global com atenção (substitui max-pooling do PointNet)
        # Calcula score de importância para cada ponto
        attn_scores = self.global_attn(feat)         # [B, N, 1]
        attn_weights = F.softmax(attn_scores, dim=1) # [B, N, 1]
        global_feat = (attn_weights * feat).sum(dim=1)  # [B, hidden_dim]

        return self.head(global_feat)  # [B, output_dim]


# ══════════════════════════════════════════════════════════════
# FACTORY: escolhe o encoder baseado no config
# ══════════════════════════════════════════════════════════════
def build_depth_encoder(config) -> nn.Module:
    """
    Cria o encoder de profundidade baseado em config.depth_encoder_type.
    
    Valor no YAML:
        depth_encoder_type: "pointnet"             # Padrão: leve e rápido
        depth_encoder_type: "point_transformer"    # Novo: mais robusto
    
    Args:
        config: ACTConfig com os parâmetros de depth
    Returns:
        nn.Module com interface forward(x: [B,3,N]) → [B, dim_model]
    """
    encoder_type = getattr(config, "depth_encoder_type", "pointnet")
    output_dim = config.dim_model

    if encoder_type == "point_transformer":
        k = getattr(config, "point_transformer_k", 16)
        num_layers = getattr(config, "point_transformer_layers", 3)
        hidden_dim = getattr(config, "point_transformer_dim", 256)
        print(
            f"[ACT-D] Usando Point Transformer — "
            f"k={k}, layers={num_layers}, hidden_dim={hidden_dim}, output_dim={output_dim}"
        )
        return PointTransformerEncoder(
            output_dim=output_dim,
            k=k,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )
    else:
        # Padrão: PointNet (compatibilidade com checkpoints existentes)
        print(f"[ACT-D] Usando PointNet — output_dim={output_dim}")
        return PointNetEncoder(output_dim=output_dim)


# ══════════════════════════════════════════════════════════════
# depth_to_pointcloud (inalterado — usado por ambos encoders)
# ══════════════════════════════════════════════════════════════
def depth_to_pointcloud(depth_tensor, intrinsics, num_points=1024):
    """
    Projeta mapa de profundidade em nuvem de pontos 3D.
    
    O tensor de profundidade chega normalizado em [0,1], onde 1.0 = 2 metros.
    Aplica projeção pinhole inversa para recuperar coordenadas XYZ reais.
    """
    B, C, H, W = depth_tensor.shape
    device = depth_tensor.device

    # Malha de pixels
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij',
    )
    grid_x = grid_x.float().unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.float().unsqueeze(0).expand(B, -1, -1)

    # Recupera profundidade em metros (0→0m, 1→2m)
    z = depth_tensor[:, 0, :, :] * 2.0

    # Projeção pinhole inversa
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    x = (grid_x - cx) * z / fx
    y = (grid_y - cy) * z / fy

    # Monta nuvem e amostra N pontos válidos
    point_cloud = torch.stack((x, y, z), dim=1).view(B, 3, -1)

    sampled_pcs = []
    for b in range(B):
        pc = point_cloud[b]
        valid_mask = pc[2, :] > 0.05  # descarta ruído < 5cm
        valid_pc = pc[:, valid_mask]

        if valid_pc.shape[1] >= num_points:
            indices = torch.randperm(valid_pc.shape[1], device=device)[:num_points]
            sampled_pcs.append(valid_pc[:, indices])
        else:
            pad = torch.zeros((3, num_points - valid_pc.shape[1]), device=device)
            sampled_pcs.append(torch.cat([valid_pc, pad], dim=1))

    return torch.stack(sampled_pcs)  # [B, 3, num_points]