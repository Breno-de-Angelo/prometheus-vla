import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_intrinsics(camera_intrinsics: dict | None) -> dict:
    """Exige intrínsecos explícitos da câmera de depth — sem default silencioso.

    O default antigo (fx=600, fy=600, cx=320, cy=240) era nominal de um frame
    640x480; com o stream real de 848x480 ele distorcia a nuvem lateralmente
    (cx correto ≈ 424) sem nenhum aviso. Default errado calado é pior que crash.
    """
    if camera_intrinsics is None:
        raise ValueError(
            "camera_intrinsics não informado. Configure no YAML de treino o bloco:\n"
            "  depth_intrinsics: {fx: ..., fy: ..., cx: ..., cy: ...}\n"
            "(na inferência, passe --depth-fx/--depth-fy/--depth-cx/--depth-cy).\n"
            "Para ler os valores REAIS do sensor, rode no robô:\n"
            "  python lerobot-ext/tools/dump_realsense_intrinsics.py --width 848 --height 480"
        )
    missing = {"fx", "fy", "cx", "cy"} - set(camera_intrinsics)
    if missing:
        raise ValueError(f"camera_intrinsics incompleto: faltam {sorted(missing)}")
    return camera_intrinsics


def depth_cloud_sanity_check(depth_tensor: torch.Tensor, depth_scale: float, tag: str = "") -> None:
    """Sanity executado UMA VEZ (1º batch do treino / init da inferência): após a
    conversão para metros, a mediana dos pontos válidos (z > 0.05 m) deve estar em
    [0.2, 3.0] m — fora disso o depth_scale ou a fonte do depth estão errados
    (ex.: PNG16 em mm com scale 1.0, ou tensor [0,1] com scale 0.001)."""
    z = depth_tensor.float().reshape(-1) * depth_scale
    if z.numel() > 1_000_000:  # subsample determinístico (sem mexer no RNG global)
        z = z[:: z.numel() // 1_000_000 + 1]
    valid = z[z > 0.05]
    if valid.numel() == 0:
        raise RuntimeError(
            f"[depth-sanity{tag}] nenhum ponto com z > 0.05 m após depth_scale={depth_scale} "
            f"(depth cru: min={depth_tensor.min().item():.4g}, max={depth_tensor.max().item():.4g})"
        )
    qs = torch.quantile(valid, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0], device=valid.device))
    mn, p5, med, p95, mx = (q.item() for q in qs)
    logging.info(
        f"[depth-sanity{tag}] primeira nuvem (metros): min={mn:.3f} p5={p5:.3f} "
        f"mediana={med:.3f} p95={p95:.3f} max={mx:.3f} (depth_scale={depth_scale})"
    )
    if not (0.2 <= med <= 3.0):
        raise RuntimeError(
            f"[depth-sanity{tag}] mediana dos pontos válidos = {med:.3f} m, fora de "
            f"[0.2, 3.0] m — depth_scale={depth_scale} provavelmente errado para esta fonte"
        )

class PointNetEncoder(nn.Module):
    """Codificador 3D para extrair features globais da Nuvem de Pontos (ACT-D)"""
    def __init__(self, output_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, output_dim)
        # Zero-init da ÚLTIMA projeção (estilo ControlNet): o token de depth entra
        # como no-op no prefixo do modelo pré-treinado (emb = 0) e o gradiente
        # ensina o modelo a usá-lo gradualmente, em vez de injetar ruído de init
        # aleatória desde o step 0. Camadas anteriores mantêm init padrão.
        # `load_injected_from` continua funcionando: load_state_dict sobrescreve.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # x shape: [Batch, 3, Num_Points]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        # Max Pooling (Invariância espacial)
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        x = F.relu(self.fc1(x))
        return self.fc2(x) # [Batch, output_dim]

# Pré-subsample uniforme antes do FPS: limita o custo O(M·K) do FPS num teto fixo
# (mesmo padrão do 3D Diffusion Policy: uniforme → ~16k → FPS → 1024).
FPS_PRE_SUBSAMPLE = 16384


def farthest_point_sampling(points: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    """FPS batched puro em torch, O(M·K) com ops vetorizadas por batch.

    Args:
        points: (B, M, 3) candidatos (lixo nas posições inválidas é ignorado).
        valid:  (B, M) bool — quais candidatos são reais.
        k: nº de pontos a selecionar (assume valid.sum(1) >= k; o chamador trata
           o caso com menos pontos via padding).
    Returns:
        índices (B, k) dos pontos selecionados.
    """
    B, M, _ = points.shape
    device = points.device
    batch = torch.arange(B, device=device)

    idx0 = valid.float().argmax(dim=1)  # primeiro válido de cada amostra
    selected = [idx0]
    last = points[batch, idx0]  # (B, 3)
    min_dist = ((points - last[:, None, :]) ** 2).sum(-1)  # (B, M)
    min_dist = torch.where(valid, min_dist, torch.full_like(min_dist, float("-inf")))

    for _ in range(k - 1):
        nxt = min_dist.argmax(dim=1)
        selected.append(nxt)
        last = points[batch, nxt]
        d = ((points - last[:, None, :]) ** 2).sum(-1)
        min_dist = torch.minimum(min_dist, torch.where(valid, d, min_dist))

    return torch.stack(selected, dim=1)  # (B, k)


def depth_to_pointcloud(
    depth_tensor,
    intrinsics,
    num_points=1024,
    depth_scale: float = 1.0,
    workspace: dict | None = None,
):
    """Projeta pixels de depth no espaço 3D (point cloud) com crop + FPS.

    Args:
        depth_tensor: shape (B, H, W) ou (B, 1, H, W). Tensor de profundidade.
        intrinsics: dict com fx, fy, cx, cy (OBRIGATÓRIO correto — ver validate_intrinsics).
        num_points: nº de pontos amostrados pra alimentar a PointNet.
        depth_scale: multiplicador para converter os valores do tensor em METROS.
        workspace: crop no frame da câmera, em metros — dict opcional
            {"z": [min, max], "x": [min, max], "y": [min, max]} (eixos faltantes
            não são cropados). Vem do YAML (`depth_workspace`); None = sem crop
            além do filtro de ruído z > 0.05 m.
    """
    if depth_tensor.dim() == 3:
        depth_tensor = depth_tensor.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)
    B, C, H, W = depth_tensor.shape
    device = depth_tensor.device

    # 1. Cria a malha de pixels
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid_x = grid_x.float().unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.float().unsqueeze(0).expand(B, -1, -1)

    # 2. Converte valores do depth tensor pra metros (depende do dataset; ver docstring).
    z = depth_tensor[:, 0, :, :] * depth_scale

    # 3. Projeção Pinhole 3D
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    x = (grid_x - cx) * z / fx
    y = (grid_y - cy) * z / fy

    pts = torch.stack((x, y, z), dim=-1).view(B, -1, 3)  # (B, H*W, 3)

    # 4. Máscara de validade: ruído de lente + crop de workspace (se configurado)
    valid = pts[..., 2] > 0.05
    if workspace:
        for axis, dim in (("x", 0), ("y", 1), ("z", 2)):
            if axis in workspace:
                lo, hi = workspace[axis]
                valid &= (pts[..., dim] >= lo) & (pts[..., dim] <= hi)

    # 5. Pré-subsample uniforme para um teto fixo de candidatos (custo do FPS)
    M = pts.shape[1]
    if M > FPS_PRE_SUBSAMPLE:
        keep = torch.randperm(M, device=device)[:FPS_PRE_SUBSAMPLE]
        pts, valid = pts[:, keep], valid[:, keep]

    # 6. Farthest Point Sampling (cobertura espacial >> randperm puro)
    n_valid = valid.sum(dim=1)  # (B,)
    out = torch.zeros(B, 3, num_points, device=device, dtype=pts.dtype)
    enough = n_valid >= num_points
    if enough.any():
        sub_pts, sub_valid = pts[enough], valid[enough]
        idx = farthest_point_sampling(sub_pts, sub_valid, num_points)
        gathered = sub_pts[torch.arange(sub_pts.shape[0], device=device)[:, None], idx]
        out[enough] = gathered.transpose(1, 2)
    for b in torch.nonzero(~enough).flatten().tolist():
        nv = int(n_valid[b].item())
        logging.warning(
            f"[depth_to_pointcloud] amostra {b}: só {nv} pontos válidos após crop "
            f"(< {num_points}) — padding com zeros"
        )
        if nv:
            out[b, :, :nv] = pts[b][valid[b]].T

    return out  # (B, 3, num_points)