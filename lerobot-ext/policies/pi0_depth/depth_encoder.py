import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    """Codificador 3D para extrair features globais da Nuvem de Pontos (ACT-D)"""
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
        # Max Pooling (Invariância espacial)
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        x = F.relu(self.fc1(x))
        return self.fc2(x) # [Batch, output_dim]

def depth_to_pointcloud(depth_tensor, intrinsics, num_points=1024, depth_unit="mm", z_max=5.0):
    """Projeta o mapa de profundidade métrico no espaço 3D.

    O tensor chega `[B, 1, H, W]` na unidade nativa do dataset — MILÍMETROS, o
    padrão do LeRobot 0.6.1 (`depth_output_unit`) — e cru, porque o processador
    tira a profundidade do normalizador.

    Isto MUDOU: até a migração a profundidade era gravada como imagem RGB de 8
    bits (0–2000 mm espremidos em 0–255), o tensor chegava em [0,1] e o código
    fazia `z = tensor * 2.0`. Com o mapa nativo esse fator erra por três ordens
    de grandeza — e em silêncio: o treino roda normal, com a cena a ~1 km.

    Args:
        depth_unit: "mm" ou "m"; o resto da função trabalha em metros.
        z_max: distância máxima (m) aceita na nuvem — ver o filtro abaixo.
    """
    if depth_unit not in ("mm", "m"):
        raise ValueError(f"depth_unit deve ser 'mm' ou 'm', recebeu {depth_unit!r}")
    para_metros = 0.001 if depth_unit == "mm" else 1.0
    B, C, H, W = depth_tensor.shape
    device = depth_tensor.device

    # 1. Cria a malha de pixels
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid_x = grid_x.float().unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.float().unsqueeze(0).expand(B, -1, -1)

    # 2. Unidade do dataset → metros
    z = depth_tensor[:, 0, :, :] * para_metros
    
    # 3. Projeção Pinhole 3D
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    x = (grid_x - cx) * z / fx
    y = (grid_y - cy) * z / fy

    # 4. Agrupa e Amostra (Sampling de 1024 pontos para a PointNet não explodir a VRAM)
    point_cloud = torch.stack((x, y, z), dim=1).view(B, 3, -1)
    
    sampled_pcs = []
    for b in range(B):
        pc = point_cloud[b]
        # Piso de 5 cm: pixel sem medida volta da desquantização como o próprio
        # `depth_min` (1 cm), não como zero. Teto de `z_max` porque a RealSense
        # devolve alguns pixels saturados (o dataset tem max de 65 m), e um
        # punhado deles domina a escala da nuvem.
        valid_mask = (pc[2, :] > 0.05) & (pc[2, :] < z_max)
        valid_pc = pc[:, valid_mask]
        
        if valid_pc.shape[1] > num_points:
            indices = torch.randperm(valid_pc.shape[1], device=device)[:num_points]
            sampled_pcs.append(valid_pc[:, indices])
        else:
            pad = torch.zeros((3, num_points - valid_pc.shape[1]), device=device)
            sampled_pcs.append(torch.cat([valid_pc, pad], dim=1))

    return torch.stack(sampled_pcs)