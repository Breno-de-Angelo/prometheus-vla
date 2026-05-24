#!/usr/bin/env python3
"""Distill Humanoid Everyday (USC-GVL) -> cup3-compatible LeRobot v3.0 dataset for a pi05-D DEPTH pretrain.

RUN ENV: conda `ms3`, CWD /home/hercules/prometheus-vla (lerobot 0.4.4 importa lá), HF_HOME=/data/huggingface-models.

VERIFICADO contra os dados reais do HE (2026-05-22):
  depth   = observation.depth.egocentric   float32 [480,640]  em MILIMETROS (range 0..5464)
  action  = [hand14, arm14]                 (corr c/ next-state: action[:14]<->mao 0.67, action[14:]<->braco 0.93)
  juntas  = observation.arm_joints[14] + observation.hand_joints[14]
  tactile = 18 sensores x4 = 72, ~8% NaN, range 0.03..1e5 -> INCOMPATIVEL c/ cup3 [33]x2 -> PULADO
            (o tactil entra so no finetune do cup3, onde o formato esta certo)
  RGB     = observation.images.egocentric   (mp4)

SAIDA (== schema do cup3, sem pressure):
  observation.images.head_camera        video [480,640,3]
  observation.images.head_camera_depth  video [480,640,3]  (8-bit grayscale, 3 canais identicos -> igual cup3)
  observation.state  float32 [28]   action  float32 [28]

REMAP (validado): arm 1:1 ; left hand 1:1 ; right hand index<->middle swap ; action [hand,arm]->[arm,hand].
DEPTH: cup3 guarda depth como 8-bit grayscale 3ch (mean~61). Casamos essa distribuicao:
  u8 = clip(round(depth_m * DEPTH_M2U), 0, 255), DEPTH_M2U=38  -> HE mediana 1.64m -> ~62 (==cup3).
  Treinar com o MESMO depth_scale do cup3 -> PointNet ve entradas comparaveis no pretrain (HE) e finetune (cup3).
"""
import argparse
import json
import os
import re
import shutil
import numpy as np

# ---- remap validado ----
HAND_HE_TO_CUP3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 10, 11]  # right index<->middle swap
DEPTH_M2U = 38.0  # metros -> 8-bit, casado empiricamente com o cup3

CUP3_STATE_NAMES = [
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
    "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1",
    "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1",
]


def remap_hand(hand14_he):
    return np.asarray(hand14_he, dtype=np.float32)[HAND_HE_TO_CUP3]


def remap_state(arm14, hand14_he):
    return np.concatenate([np.asarray(arm14, dtype=np.float32), remap_hand(hand14_he)])  # [28]


def remap_action(he_action28):
    a = np.asarray(he_action28, dtype=np.float32)
    return np.concatenate([a[14:], remap_hand(a[:14])])  # HE [hand,arm] -> cup3 [arm,hand]


def depth_mm_to_u8x3(depth_mm_2d):
    m = np.asarray(depth_mm_2d, dtype=np.float32) * 0.001
    u8 = np.clip(np.round(m * DEPTH_M2U), 0, 255).astype(np.uint8)
    return np.repeat(u8[:, :, None], 3, axis=2)  # [480,640,3]


def stack_depth(parquet_cell):
    """observation.depth.egocentric guardado como object-array de 480 linhas de 640 -> [480,640]."""
    return np.stack([np.asarray(r, dtype=np.float32) for r in parquet_cell])


def decode_rgb_frames(mp4_path):
    import cv2
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# ---- self-test do remap (sem dados) ----
def selftest():
    arm = np.arange(14, dtype=np.float32)
    hand = np.arange(100, 114, dtype=np.float32)
    st = remap_state(arm, hand)
    assert st.shape == (28,)
    assert list(st[21:28]) == [107, 108, 109, 112, 113, 110, 111], st[21:28]
    ac = remap_action(np.concatenate([hand, arm]))  # HE action = [hand, arm]
    assert list(ac[:14]) == list(arm) and list(ac[14:]) == list(remap_hand(hand))
    d = depth_mm_to_u8x3(np.array([[0, 1000, 1640, 5464]], dtype=np.float32))
    assert d.shape == (1, 4, 3) and d[0, 2, 0] == 62, d[0, :, 0]  # 1.64m -> ~62 (==cup3)
    print("[selftest] remap OK | depth 1.64m->", int(d[0, 2, 0]), "(cup3 mean~61)")


# ---- meta helpers ----
def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def list_local_episodes(snap_root):
    """Quais episodios tem parquet+mp4 baixados localmente (intersecao)."""
    pq = {int(n.split("_")[1].split(".")[0]) for n in os.listdir(os.path.join(snap_root, "data", "chunk-000")) if n.endswith(".parquet")}
    vd = os.path.join(snap_root, "videos", "chunk-000", "egocentric")
    mp = {int(n.split("_")[1].split(".")[0]) for n in os.listdir(vd)} if os.path.isdir(vd) else set()
    return sorted(pq & mp)


def episode_meta(snap_root):
    eps = {e["episode_index"]: e for e in _read_jsonl(os.path.join(snap_root, "meta", "episodes.jsonl"))}
    tasks = {}
    tp = os.path.join(snap_root, "meta", "tasks.jsonl")
    if os.path.exists(tp):
        for t in _read_jsonl(tp):
            tasks[t.get("task_index", t.get("index"))] = t
    return eps, tasks


def _clean_task(name):
    """'Basic/pour_water_from_a_kettle_into_a_cup_g1' -> 'pour water from a kettle into a cup'."""
    s = name.split("/")[-1]
    s = re.sub(r"_g1$", "", s).replace("_", " ").strip()
    return s or name


def task_string(ep, tasks):
    # ATENCAO: ep["instruction"] do HE e um CONSTANTE quebrado (mesma frase p/ TODOS os episodios) -> ignorar.
    # A tarefa real esta em ep["tasks"]=[task_index] -> tasks.jsonl["task"] (nome estruturado, distinto por tarefa).
    tl = ep.get("tasks")
    if isinstance(tl, list) and tl:
        t0 = tl[0]
        if isinstance(t0, str) and t0.strip():
            return _clean_task(t0)
        t = tasks.get(t0, {})
        if isinstance(t.get("task"), str) and t["task"].strip():
            return _clean_task(t["task"])
    return "manipulate the object"


FEATURES = {
    "observation.images.head_camera": {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"]},
    "observation.images.head_camera_depth": {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": [28], "names": CUP3_STATE_NAMES},
    "action": {"dtype": "float32", "shape": [28], "names": CUP3_STATE_NAMES},
}


def _add_frame(dst, frame, task):
    dst.add_frame(dict(frame, task=task))  # lerobot 0.4.4: task vai dentro do dict


def build_from_local(snap_root, episode_indices, out_repo, out_root, fps=30):
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    eps_meta, tasks = episode_meta(snap_root)
    dst = LeRobotDataset.create(out_repo, fps=fps, root=out_root, features=FEATURES, use_videos=True, vcodec="h264")
    dst.vcodec = "h264"  # casa o codec do cup3 (h264 yuv420p) + ~5x mais rapido que o svtav1 default
    # lerobot 0.4.4 validate_frame compara value.shape (tupla) com feature['shape'] (lista) -> coage p/ tupla
    for _f in dst.features.values():
        if _f.get("dtype") not in ("image", "video", "string") and isinstance(_f.get("shape"), list):
            _f["shape"] = tuple(_f["shape"])

    for idx in episode_indices:
        pq = os.path.join(snap_root, "data", "chunk-000", f"episode_{idx:06d}.parquet")
        mp4 = os.path.join(snap_root, "videos", "chunk-000", "egocentric", f"episode_{idx:06d}.mp4")
        df = pd.read_parquet(pq)
        rgb = decode_rgb_frames(mp4)
        n = min(len(df), len(rgb))
        if len(df) != len(rgb):
            print(f"  [warn] ep{idx}: {len(df)} linhas vs {len(rgb)} frames RGB -> usando {n}")
        task = task_string(eps_meta.get(idx, {}), tasks)
        for i in range(n):
            _add_frame(dst, {
                "observation.images.head_camera": rgb[i],
                "observation.images.head_camera_depth": depth_mm_to_u8x3(stack_depth(df["observation.depth.egocentric"].iloc[i])),
                "observation.state": remap_state(df["observation.arm_joints"].iloc[i], df["observation.hand_joints"].iloc[i]),
                "action": remap_action(df["action"].iloc[i]),
            }, task)
        dst.save_episode()
        print(f"  [build] ep{idx}: {n} frames | task='{task}'")
    print(f"[build] DONE -> {out_root}")
    return out_root


def download_and_build(repo, episode_indices, out_repo, out_root, meta_root, tmp_dir="/tmp/he_stream", fps=30, min_free_gb=25):
    """Streaming: baixa parquet+mp4 de cada episodio, converte, apaga o cru. Guarda de disco."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = json.load(open(os.path.join(meta_root, "meta", "info.json")))
    csize = int(info.get("chunks_size", 1000))
    eps_meta, tasks = episode_meta(meta_root)
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(tmp_dir, exist_ok=True)
    dst = LeRobotDataset.create(out_repo, fps=fps, root=out_root, features=FEATURES, use_videos=True, vcodec="h264")
    for _f in dst.features.values():
        if _f.get("dtype") not in ("image", "video", "string") and isinstance(_f.get("shape"), list):
            _f["shape"] = tuple(_f["shape"])

    done = 0
    for idx in episode_indices:
        free_gb = shutil.disk_usage(out_root).free / 1e9
        if free_gb < min_free_gb:
            print(f"  [ABORT] disco baixo ({free_gb:.0f} GB < {min_free_gb}) — parando em {done} eps")
            break
        ch = f"chunk-{idx // csize:03d}"
        try:
            pq = hf_hub_download(repo, f"data/{ch}/episode_{idx:06d}.parquet", repo_type="dataset", local_dir=tmp_dir)
            mp4 = hf_hub_download(repo, f"videos/{ch}/egocentric/episode_{idx:06d}.mp4", repo_type="dataset", local_dir=tmp_dir)
        except Exception as e:
            print(f"  [skip] ep{idx}: download falhou ({type(e).__name__}: {e})")
            continue
        try:
            df = pd.read_parquet(pq)
            rgb = decode_rgb_frames(mp4)
            n = min(len(df), len(rgb))
            task = task_string(eps_meta.get(idx, {}), tasks)
            for i in range(n):
                _add_frame(dst, {
                    "observation.images.head_camera": rgb[i],
                    "observation.images.head_camera_depth": depth_mm_to_u8x3(stack_depth(df["observation.depth.egocentric"].iloc[i])),
                    "observation.state": remap_state(df["observation.arm_joints"].iloc[i], df["observation.hand_joints"].iloc[i]),
                    "action": remap_action(df["action"].iloc[i]),
                }, task)
            dst.save_episode()
            done += 1
            if done <= 3 or done % 10 == 0:
                print(f"  [{done}/{len(episode_indices)}] ep{idx} ({n} frames, {free_gb:.0f}GB free) task='{task[:55]}'")
        finally:
            for f in (pq, mp4):
                try:
                    os.remove(f)
                except OSError:
                    pass
    print(f"[download_and_build] DONE {done}/{len(episode_indices)} eps -> {out_root}")
    return out_root


def validate(out_repo, out_root):
    os.environ.setdefault("HF_HOME", "/data/huggingface-models")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(out_repo, root=out_root, video_backend="pyav")
    s = ds[0]
    keys = [k for k in s if k.startswith(("observation", "action"))]
    print("features:", keys)
    st, ac = s["observation.state"], s["action"]
    dep = s["observation.images.head_camera_depth"]
    rgb = s["observation.images.head_camera"]
    print("state", tuple(st.shape), "action", tuple(ac.shape), "rgb", tuple(rgb.shape), "depth", tuple(dep.shape))
    ch0 = dep[0] if dep.ndim == 3 else dep
    print("depth ch0 min/max:", float(ch0.min()), float(ch0.max()), "(esperado >0; cup3-like 0..~200)")
    assert st.shape[-1] == 28 and ac.shape[-1] == 28
    print("[validate] OK — schema cup3 (sem pressure). Treinar com depth_fusion=true, fusion_mode=depth_only.")


def default_meta_root():
    import glob
    g = sorted(glob.glob("/data/huggingface-models/hub/datasets--USC-GVL--humanoid-everyday/snapshots/*/"))
    return g[0].rstrip("/") if g else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--snap-root", help="root do snapshot HF baixado (humanoid-everyday)")
    ap.add_argument("--test-local", type=int, metavar="N", help="converte os primeiros N episodios locais (smoke test)")
    ap.add_argument("--repo", default="USC-GVL/humanoid-everyday")
    ap.add_argument("--grasp-list", default="/home/hercules/prometheus-vla/scripts/he_g1_grasp_episodes.json")
    ap.add_argument("--meta-root", default=None, help="root com meta/ (default: cache HF do humanoid-everyday)")
    ap.add_argument("--download", type=int, metavar="N", help="streaming: baixa+converte os primeiros N da grasp-list")
    ap.add_argument("--download-all", action="store_true", help="streaming: baixa+converte a grasp-list inteira (788)")
    ap.add_argument("--out-repo", default="lewislf/G1_Dex3_HE_depth")
    ap.add_argument("--out-root", default="/data/huggingface-models/lerobot/lewislf/G1_Dex3_HE_depth")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.validate:
        validate(a.out_repo, a.out_root)
    elif a.test_local is not None:
        assert a.snap_root, "--snap-root obrigatorio"
        locs = list_local_episodes(a.snap_root)[: a.test_local]
        print(f"[test-local] episodios locais: {locs}")
        build_from_local(a.snap_root, locs, a.out_repo, a.out_root)
        validate(a.out_repo, a.out_root)
    elif a.download is not None or a.download_all:
        meta = a.meta_root or default_meta_root()
        idxs = json.load(open(a.grasp_list))
        if a.download is not None:
            idxs = idxs[: a.download]
        print(f"[download] repo={a.repo} | {len(idxs)} eps | meta={meta} | out={a.out_root}")
        download_and_build(a.repo, idxs, a.out_repo, a.out_root, meta_root=meta)
        validate(a.out_repo, a.out_root)
    else:
        ap.print_help()
