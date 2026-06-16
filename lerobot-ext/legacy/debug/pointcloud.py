import cv2
import torch
import numpy as np
import open3d as o3d
from policies.act_depth.depth_encoder import depth_to_pointcloud

# 1. Configurações Iniciais
video_path = "meu_dataset/pick_up_the_cup_2026-04-30/videos/observation.images.head_camera_depth/chunk-000/file-000.mp4"
intrinsics = {'fx': 500.0, 'fy': 500.0, 'cx': 320.0, 'cy': 240.0}

# ==========================================
# FUNÇÃO EXTRA: Criar Grade de Piso (Grid)
# ==========================================
def criar_grade_piso(tamanho=2.0, espacamento=0.1, altura=-0.3):
    """
    Cria uma malha de linhas para dar noção de escala.
    Cada quadrado na grade terá 'espacamento' (ex: 0.1 = 10 cm).
    """
    pontos = []
    linhas = []
    idx = 0
    
    # Linhas no eixo X
    for z in np.arange(-tamanho/2, tamanho/2 + espacamento, espacamento):
        pontos.append([-tamanho/2, altura, z])
        pontos.append([tamanho/2, altura, z])
        linhas.append([idx, idx+1])
        idx += 2
        
    # Linhas no eixo Z
    for x in np.arange(-tamanho/2, tamanho/2 + espacamento, espacamento):
        pontos.append([x, altura, -tamanho/2])
        pontos.append([x, altura, tamanho/2])
        linhas.append([idx, idx+1])
        idx += 2
        
    grade = o3d.geometry.LineSet()
    grade.points = o3d.utility.Vector3dVector(pontos)
    grade.lines = o3d.utility.Vector2iVector(linhas)
    # Pinta a grade de cinza escuro
    grade.colors = o3d.utility.Vector3dVector([[0.3, 0.3, 0.3] for i in range(len(linhas))])
    return grade

# ==========================================
# 2. Inicializar a Janela 3D (Open3D)
# ==========================================
print("Iniciando o Player 3D... Pressione 'Q' ou 'ESC' na janela para fechar.")
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Prometheus G1 - Visão 3D Pro", width=1280, height=720)

# 2.1 Adicionar a Nuvem de Pontos
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

# 2.2 Adicionar a Referência de Eixos (XYZ)
# Tamanho 0.2 = As setinhas terão 20 centímetros
eixos = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
vis.add_geometry(eixos)

# 2.3 Adicionar a Grade de Piso (Floor Grid)
# Uma grade de 2 metros por 2 metros, com quadrados a cada 10 cm (0.1m)
grade = criar_grade_piso(tamanho=2.0, espacamento=0.1, altura=-0.2)
vis.add_geometry(grade)

# Configurações visuais (Fundo escuro, pontos maiores)
opt = vis.get_render_option()
opt.background_color = np.asarray([0.05, 0.05, 0.08]) # Fundo quase preto
opt.point_size = 3.0

# ==========================================
# 3. Loop do Vídeo
# ==========================================
cap = cv2.VideoCapture(video_path)
primeiro_frame = True

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    depth_frame = frame[:, :, 0]
    depth_tensor = torch.from_numpy(depth_frame).float() / 255.0
    depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)

    # 5000 pontos para a visualização ficar densa e bonita original é 1024, mas para o player 3D podemos usar mais pontos!
    nuvem_tensor = depth_to_pointcloud(depth_tensor, intrinsics, num_points=8000)

    pontos_numpy = nuvem_tensor[0].transpose(0, 1).cpu().numpy()
    
    # Inverte os eixos para combinar com a câmera padrão do Open3D
    pontos_numpy[:, 1] *= -1 
    pontos_numpy[:, 2] *= -1

    pcd.points = o3d.utility.Vector3dVector(pontos_numpy)

    # Mapa de calor pela distância (Eixo Z)
    z_values = pontos_numpy[:, 2]
    z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-5)
    cores = np.zeros_like(pontos_numpy)
    cores[:, 0] = 1.0 - z_norm # Vermelho (perto)
    cores[:, 1] = z_norm * 0.5 # Verde/Amarelo (médio)
    cores[:, 2] = z_norm       # Azul (longe)
    pcd.colors = o3d.utility.Vector3dVector(cores)

    # Posiciona a câmera no primeiro frame
    if primeiro_frame:
        # Movemos a visão inicial para olhar um pouco de cima e na diagonal
        ctr = vis.get_view_control()
        ctr.set_front([0.0, -0.5, -1.0]) # Olha levemente para baixo
        ctr.set_up([0.0, 1.0, 0.0])      # Define onde é "cima"
        primeiro_frame = False

    vis.update_geometry(pcd)
    
    if not vis.poll_events():
        break
    vis.update_renderer()

cap.release()
vis.destroy_window()