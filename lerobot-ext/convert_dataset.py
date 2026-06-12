import os
import pathlib
import torch
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ==========================================
# 0. Performance: troca SVT-AV1 por x264
#    (3-5x mais rápido, arquivo ~20% maior)
# ==========================================
os.environ["LEROBOT_VIDEO_CODEC"] = "libx264"
os.environ["LEROBOT_VIDEO_CRF"] = "23"          # qualidade x264 (18=alta, 28=baixa)
torch.set_num_threads(16)                         # usa todos os núcleos para tensores

# ==========================================
# 1. Configurações de Caminho
# ==========================================
src_repo_id = "lewislf/G1_Dex3_pick_white_cup_v2"
src_root = "meu_dataset/dataset_g1_cup/"

dst_root = "meu_dataset/dataset_g1_cup_convertido/"
dst_repo_id = "seu_usuario/pick_up_the_cup_convertido"

dst_path = pathlib.Path(dst_root)

# ==========================================
# 2. Detecta episódios já convertidos
# ==========================================
def get_already_converted_episodes(dst_path: pathlib.Path) -> set:
    converted = set()
    for mp4 in dst_path.glob("videos/**/*.mp4"):
        stem = mp4.stem  # "episode_000097"
        if stem.startswith("episode_"):
            try:
                ep_idx = int(stem.split("_")[1])
                converted.add(ep_idx)
            except (IndexError, ValueError):
                pass
    return converted

RESUME = dst_path.exists()

if RESUME:
    already_done = get_already_converted_episodes(dst_path)
    if already_done:
        print(f"[RESUME] Diretório de destino encontrado.")
        print(f"[RESUME] Episódios já convertidos: {len(already_done)} (último: {max(already_done)}) — continuando...\n")
    else:
        import shutil
        print(f"[RESUME] Diretório encontrado mas sem episódios válidos — removendo e recomeçando...\n")
        shutil.rmtree(dst_path)
        RESUME = False
        already_done = set()
else:
    already_done = set()
    print("[INFO] Iniciando conversão do zero...\n")

# ==========================================
# 3. Carrega dataset de origem
# ==========================================
print("Carregando dataset de origem...")
ds_src = LeRobotDataset(repo_id=src_repo_id, root=src_root)
features = ds_src.meta.features.copy()

ep0 = ds_src.meta.episodes[0]
print(f"[DEBUG] Chaves do episódio 0: {list(ep0.keys())}")

if "from" in ep0:
    KEY_START, KEY_END = "from", "to"
elif "start_index" in ep0:
    KEY_START, KEY_END = "start_index", "end_index"
elif "dataset_from_index" in ep0:
    KEY_START, KEY_END = "dataset_from_index", "dataset_to_index"
else:
    raise KeyError(
        f"Não foi possível identificar as chaves de índice dos episódios. "
        f"Chaves disponíveis: {list(ep0.keys())}"
    )

print(f"[DEBUG] Usando chaves: '{KEY_START}' e '{KEY_END}'")

# ==========================================
# 4. Ajuste das Features
# ==========================================
features["observation.images.head_camera_depth"] = {
    "dtype": "video",
    "shape": (480, 848, 3),
    "names": ["height", "width", "channels"],
    "info": {
        "video.fps": 30,
        "video.codec": "libx264",
        "video.pix_fmt": "yuv420p",
        "video.channels": 3,
        "has_audio": False,
        "video.is_depth_map": True
    }
}

features.pop("observation.left_hand_pressure", None)
features.pop("observation.right_hand_pressure", None)

# ==========================================
# 5. Cria ou reabre o dataset de destino
# ==========================================
if RESUME and already_done:
    print("Reabrindo dataset de destino existente...")
    ds_dst = LeRobotDataset(repo_id=dst_repo_id, root=dst_root)
else:
    print("Inicializando novo dataset no padrão Prometheus...")
    ds_dst = LeRobotDataset.create(
        repo_id=dst_repo_id,
        root=dst_root,
        features=features,
        fps=ds_src.meta.fps,
        robot_type=ds_src.meta.robot_type
    )

# ==========================================
# 6. Loop de Conversão (com retomada)
# ==========================================
episodes_to_convert = (
    ds_src.episodes
    if ds_src.episodes is not None
    else range(len(ds_src.meta.episodes))
)

total = len(ds_src.meta.episodes)
skipped = 0

for ep_idx in episodes_to_convert:
    if ep_idx in already_done:
        skipped += 1
        if skipped == 1 or skipped % 10 == 0:
            print(f"[SKIP] Episódio {ep_idx} já convertido ({skipped} pulados até agora)...")
        continue

    print(f"\nConvertendo Episódio {ep_idx} / {total - 1}...")

    ep_meta = ds_src.meta.episodes[ep_idx]
    start_idx = ep_meta[KEY_START]
    end_idx   = ep_meta[KEY_END]

    raw_task = ds_src[start_idx].get("task")
    if raw_task is None:
        raw_task = ep_meta.get("tasks", ["unknown task"])
    task_name = raw_task[0] if isinstance(raw_task, list) else raw_task

    for i in tqdm.tqdm(range(start_idx, end_idx)):
        item = ds_src[i]

        for key in ["index", "episode_index", "frame_index", "timestamp", "task_index"]:
            item.pop(key, None)

        if "task" not in item:
            item["task"] = task_name

        item.pop("observation.left_hand_pressure", None)
        item.pop("observation.right_hand_pressure", None)

        # DEPTH: (1, H, W) int16 → normaliza → 3 canais → (H, W, C)
        depth_raw = item["observation.images.head_camera_depth"]
        depth_clipped = torch.clamp(depth_raw.float(), 0, 2000)
        depth_normalized = depth_clipped / 2000.0
        if depth_normalized.shape[0] == 1:
            depth_3c = depth_normalized.repeat(3, 1, 1)
        else:
            depth_3c = depth_normalized
        item["observation.images.head_camera_depth"] = depth_3c.permute(1, 2, 0)

        # RGB: (C, H, W) → (H, W, C)
        rgb = item["observation.images.head_camera"]
        item["observation.images.head_camera"] = rgb.permute(1, 2, 0)

        ds_dst.add_frame(item)

    ds_dst.save_episode()

print("\nFinalizando dataset (parquets e jsons)...")
ds_dst.finalize()
print("\n✅ Conversão finalizada com sucesso!")