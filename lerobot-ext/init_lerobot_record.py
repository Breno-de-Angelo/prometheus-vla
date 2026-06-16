#!/usr/bin/env python
"""
Data Collection Entry Point - HACKER EDITION V9 + VOICE CONTROL
Corrigindo validação de Tuplas e adicionando controle Hands-Free.
"""

import sys
import os

# ---------------------------------------------------------------------------
# 🔄 AUTO-ATIVAÇÃO DO ENV CONDA 'g1'
# Se este script foi lançado FORA do env certo, re-executa com o Python do env
# (sem precisar de `conda activate g1` antes). Tem que vir ANTES dos imports
# pesados (numpy, robot, teleop), que só existem no env. Override: G1_CONDA_ENV.
# A guarda _G1_REEXEC evita loop infinito caso a detecção falhe.
# ---------------------------------------------------------------------------
_TARGET_ENV = os.environ.get("G1_CONDA_ENV", "g1")
_in_target = (os.environ.get("CONDA_DEFAULT_ENV") == _TARGET_ENV
              or f"{os.sep}envs{os.sep}{_TARGET_ENV}{os.sep}" in sys.executable)
if not _in_target and not os.environ.get("_G1_REEXEC"):
    def _find_env_python(env):
        cands = []
        exe = os.environ.get("CONDA_EXE")          # .../bin/conda → base
        if exe:
            base = os.path.dirname(os.path.dirname(exe))
            cands.append(os.path.join(base, "envs", env, "bin", "python"))
        for root in ("~/miniconda3", "~/anaconda3", "~/miniforge3", "~/mambaforge"):
            cands.append(os.path.expanduser(f"{root}/envs/{env}/bin/python"))
        return next((c for c in cands if os.path.exists(c)), None)

    _env_py = _find_env_python(_TARGET_ENV)
    if _env_py:
        _env_prefix = os.path.dirname(os.path.dirname(_env_py))   # .../envs/g1
        os.environ["_G1_REEXEC"] = "1"
        os.environ["CONDA_PREFIX"] = _env_prefix
        os.environ["CONDA_DEFAULT_ENV"] = _TARGET_ENV
        os.environ["PATH"] = os.path.dirname(_env_py) + os.pathsep + os.environ.get("PATH", "")
        print(f"[ENV] 🔄 Ativando conda '{_TARGET_ENV}' automaticamente...", file=sys.stderr)
        os.execv(_env_py, [_env_py] + sys.argv)
    else:
        print(f"[ENV] ⚠️ Não achei o Python do env conda '{_TARGET_ENV}'. "
              f"Ative manualmente: conda activate {_TARGET_ENV}", file=sys.stderr)

# Promove o editable finder (PEP 660) à frente do PathFinder: senão o
# unitree_sdk2py do ~/.local (PyPI, sem crc.so) mascara o do env. Só reordena
# meta_path, então o ~/.local segue valendo pro resto (ex.: orjson).
try:
    from importlib.machinery import PathFinder as _PathFinder
    _ed = [f for f in sys.meta_path
           if str(getattr(f, "__module__", "")).startswith("__editable__")
           or getattr(f, "__name__", "") == "_EditableFinder"]
    if _ed:
        for f in _ed:
            sys.meta_path.remove(f)
        _i = next((i for i, f in enumerate(sys.meta_path) if f is _PathFinder), 0)
        sys.meta_path[_i:_i] = _ed
except Exception as _e:
    print(f"[ENV] ⚠️ não consegui priorizar o editable finder: {_e}", file=sys.stderr)

# Silencia o aviso cosmético do pkg_resources (vindo do pygame) — terminal limpo.
import warnings as _warnings
_warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated.*")

import logging
import numpy as np
import threading
import time


class _BootCapture:
    """Acumula o stdout durante imports/patches (antes do run.log existir).
    É despejado no run.log no __main__, mantendo o terminal limpo no boot.
    (stderr NÃO é capturado — erros de import continuam visíveis.)"""

    def __init__(self, real):
        self._real = real
        self._parts = []

    def write(self, s):
        self._parts.append(s)
        return len(s)

    def flush(self):
        pass

    def fileno(self):
        return self._real.fileno()

    def isatty(self):
        return False

    def getvalue(self):
        return "".join(self._parts)


_BOOT_CAP = _BootCapture(sys.stdout)
sys.stdout = _BOOT_CAP

print("[ENV] ⏳ Carregando bibliotecas (torch/lerobot/mujoco), ~10s...", file=sys.stderr)

frame_count = 0

# Buffer global de contrabando
buffer_pressao = {"left": np.zeros(108, dtype=np.float32), "right": np.zeros(108, dtype=np.float32)}

try:
    import robot.unitree_g1
    import teleop.unitree_g1
except ImportError as e:
    print(f"\n[IMPORT ERROR]: {e}", file=sys.stderr)
    sys.exit(1)

# =========================================================================
# 💉 INJEÇÃO 1: Rouba a pressão do Robô
# =========================================================================
# Esta é a classe COMPLETA/pretendida do robô (tátil via get_observation, heartbeat
# anti-watchdog da Dex3, mãos via ZMQ). O make_robot_from_config do lerobot/src
# instanciava a cópia parcial de lerobot/src (sem tátil/heartbeat) — corrigido pela
# INJEÇÃO 0 abaixo, que força make_robot a instanciar ESTA classe (lerobot-ext).
from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3

original_get_obs = UnitreeG1Dex3.get_observation

def patched_get_observation(self):
    global frame_count
    obs = original_get_obs(self)
    
    if obs is not None:
        frame_count += 1
        
        if "left_hand_pressure" in obs:
            lp = obs.pop("left_hand_pressure")
            rp = obs.pop("right_hand_pressure")
            buffer_pressao["left"] = lp
            buffer_pressao["right"] = rp
            
            if frame_count % 50 == 0:
                max_l = np.max(lp)
                max_r = np.max(rp)
                status = "🟢 SENSOR ATIVO" if (max_l > 0 or max_r > 0) else "⚪ ZERADO (Aguardando Toque)"
                #print(f"[DEBUG] Frame {frame_count} | {status} | Max L: {max_l:.0f} | Max R: {max_r:.0f}")
        #else:
            #if frame_count % 50 == 0:
                #print(f"[ERRO] Frame {frame_count} | 🔴 DRIVER NÃO ENVIOU DADOS DE PRESSÃO!")
                
    return obs

UnitreeG1Dex3.get_observation = patched_get_observation

# =========================================================================
# 💉 INJEÇÃO 2: Editar a Planta Baixa do Parquet (TUPLAS!)
# =========================================================================
import lerobot.datasets.utils
original_hw_to_dataset_features = lerobot.datasets.utils.hw_to_dataset_features

def patched_hw_to_dataset_features(features, feature_type, use_videos):
    dataset_features = original_hw_to_dataset_features(features, feature_type, use_videos)
    
    if "observation.state" in dataset_features:
        print("\n[HACK LEROBOT] 🗜️ Configurando colunas do Parquet para Pressão...")
        
        old_names = dataset_features["observation.state"].get("names", [])
        new_names = [n for n in old_names if "pressure" not in n]
        dataset_features["observation.state"]["names"] = new_names
        dataset_features["observation.state"]["shape"] = (len(new_names),)
        
        dataset_features["observation.left_hand_pressure"] = {
            "dtype": "float32", "shape": (108,), "names": [f"left_hand_pressure_{i}" for i in range(108)]
        }
        dataset_features["observation.right_hand_pressure"] = {
            "dtype": "float32", "shape": (108,), "names": [f"right_hand_pressure_{i}" for i in range(108)]
        }
        
        chave_depth = "observation.images.head_camera_depth"
        if chave_depth in dataset_features:
            # depth como imagem uint16 1-canal (PNG), em vez de video h264 8-bit.
            # Deriva a largura do shape que o robot já calculou (cam2_width: 640 no sim,
            # 848 no real) — evita hardcode que crashava o sim depois da mudança 848.
            orig = dataset_features[chave_depth]
            h = orig["shape"][0]
            w = orig["shape"][1]
            print(f"[HACK LEROBOT] Depth -> PNG uint16 1-canal (mm) | shape derivado: ({h},{w},1)")
            dataset_features[chave_depth] = {
                "dtype": "image",
                "shape": (h, w, 1),
                "names": ["height", "width", "channels"],
            }
            
    return dataset_features

lerobot.datasets.utils.hw_to_dataset_features = patched_hw_to_dataset_features

# =========================================================================
# 💉 INJEÇÃO 3: Contrabando de volta pro Empacotador
# =========================================================================
original_build_dataset_frame = lerobot.datasets.utils.build_dataset_frame

def patched_build_dataset_frame(features, obs_dict, prefix="observation."):
    lp = buffer_pressao["left"]
    rp = buffer_pressao["right"]
    
    for i in range(108):
        obs_dict[f"left_hand_pressure_{i}"] = float(lp[i])
        obs_dict[f"right_hand_pressure_{i}"] = float(rp[i])
        
    return original_build_dataset_frame(features, obs_dict, prefix)

lerobot.datasets.utils.build_dataset_frame = patched_build_dataset_frame

# =========================================================================
# 💉 INJEÇÃO 3.1: decodifica a câmera *_depth como uint16 (IMREAD_UNCHANGED)
# A ZMQCamera.read padrão usa IMREAD_COLOR, que converteria a depth pra BGR 8-bit.
# =========================================================================
import lerobot.cameras.zmq.camera_zmq as _zmq_cam_mod
import cv2 as _cv2_dep

def _post_process_zmq_frame(self, frame):
    """Pós-processa o frame cru do ZMQCamera: RGB já vem da fonte; 1 canal na depth.

    O realsense_server.py agora faz cvtColor(BGR2RGB) NA FONTE → o frame já chega RGB.
    Portanto NÃO reconvertemos aqui (single source of truth). Só tratamos a depth.
    (Antes a câmera mandava BGR e a conversão era feita aqui — virou redundante/dobrada.)
    """
    if "depth" not in getattr(self, "camera_name", ""):
        return frame  # já é RGB (convertido no realsense_server) — sem reconversão
    if frame.ndim == 2:
        frame = frame[:, :, None]  # (H,W) -> (H,W,1)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        frame = frame[:, :, 0:1]  # depth triplicada em 3 canais -> 1 canal
    return frame

# IMPORTANTE: patcha read E async_read. O get_observation do robô lê pela
# async_read (unitree_g1.py), então patchar só o read deixava o dataset em BGR
# (mão azul, mesa fria) — bug corrigido em 2026-06-09.
_orig_zmq_read = _zmq_cam_mod.ZMQCamera.read
_orig_zmq_async_read = _zmq_cam_mod.ZMQCamera.async_read

def _patched_zmq_read(self, color_mode=None):
    return _post_process_zmq_frame(self, _orig_zmq_read(self, color_mode))

def _patched_zmq_async_read(self, timeout_ms=10000):
    return _post_process_zmq_frame(self, _orig_zmq_async_read(self, timeout_ms))

_zmq_cam_mod.ZMQCamera.read = _patched_zmq_read
_zmq_cam_mod.ZMQCamera.async_read = _patched_zmq_async_read

# =========================================================================
# 💉 INJEÇÃO 3.2: salva arrays uint16 (1 canal) como PNG 16-bit
# O image_array_to_pil_image padrão exige 3 canais e força uint8 (×255).
# =========================================================================
import lerobot.datasets.image_writer as _img_writer_mod
import PIL.Image as _PILImage_dep

_orig_arr_to_pil = _img_writer_mod.image_array_to_pil_image

def _patched_arr_to_pil(image_array, range_check=True):
    if getattr(image_array, "dtype", None) == np.uint16:
        arr = np.squeeze(image_array)  # (H,W,1)/(1,H,W) -> (H,W)
        if arr.ndim != 2:
            raise ValueError(f"depth uint16 esperado 1 canal, recebi shape {image_array.shape}")
        return _PILImage_dep.fromarray(arr)  # PIL modo 'I;16' (16-bit grayscale)
    # depth do sim chega como uint8 1-canal (triplicado pelo image_publish_utils)
    if getattr(image_array, "ndim", 0) == 3 and image_array.shape[2] == 1:
        return _PILImage_dep.fromarray(np.squeeze(image_array).astype(np.uint8), mode="L")
    return _orig_arr_to_pil(image_array, range_check)

_img_writer_mod.image_array_to_pil_image = _patched_arr_to_pil

# Patch direto em write_image para garantir que depth seja salvo mesmo quando
# o worker thread resolve image_array_to_pil_image pelo namespace local do módulo.
_orig_write_image = _img_writer_mod.write_image

def _patched_write_image(image, fpath, compress_level=1):
    import PIL.Image as _PIL
    try:
        if isinstance(image, np.ndarray):
            img = _patched_arr_to_pil(image)
            if image.dtype == np.uint16:
                # Depth uint16 e salvo a cada frame no buffer temporario do LeRobot.
                # compress_level=6 economiza disco, mas consome CPU e causa lag no live view.
                compress_level = min(compress_level, 1)
            elif (
                os.environ.get("G1_RGB_TEMP_FAST", "1") not in ("", "0", "false", "False")
                and image.dtype == np.uint8
                and getattr(image, "ndim", 0) == 3
                and image.shape[2] == 3
                and "head_camera" in str(fpath)
            ):
                # RGB de video e temporario: o LeRobot so precisa desses arquivos
                # para montar o mp4 no save_episode. Salvar PNG por frame durante
                # a teleop custa CPU e causa lag no OmniView. BMP nao comprime:
                # usa mais disco, mas e muito mais leve em CPU. Mantemos o nome
                # .png porque o encoder procura frame-XXXXXX.png; PIL detecta pelo
                # magic byte ao abrir.
                mode = os.environ.get("G1_RGB_TEMP_FAST_FORMAT", "BMP").upper()
                if mode == "JPEG":
                    img.save(fpath, format="JPEG", quality=int(os.environ.get("G1_RGB_TEMP_JPEG_QUALITY", "90")))
                else:
                    img.save(fpath, format="BMP")
                return
        elif isinstance(image, _PIL.Image):
            img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        img.save(fpath, compress_level=compress_level)
    except Exception as e:
        print(f"Error writing image {fpath}: {e}")

_img_writer_mod.write_image = _patched_write_image

# =========================================================================
# 💉 INJEÇÃO 3.3: FIX do batch encoding do LeRobot v3.0
# BUG do core: _batch_save_episode_video monta o path do META/EPISODES file
# usando o índice do DATA file (data/chunk_index, data/file_index). O data
# pagina a cada ~5 episódios (depth uint16 embutido é grande), mas o
# meta/episodes cabe muitos episódios em file-000 → no episódio 5 ele tenta
# ler meta/episodes/chunk-000/file-001.parquet (inexistente) →
# FileNotFoundError no encoding final. TODA gravação com >5 episódios crashava
# ao encerrar (os dados ficavam salvos, mas o vídeo RGB não era encodado).
# FIX: usar o índice de meta/episodes (o correto para esse path).
# =========================================================================
import pandas as _pd_enc_fix
import logging as _lg_enc_fix
import lerobot.datasets.lerobot_dataset as _ld_mod
from lerobot.datasets.utils import (
    DEFAULT_EPISODES_PATH as _DEF_EP_PATH,
    load_episodes as _load_eps_fix,
)

def _patched_batch_save_episode_video(self, start_episode, end_episode=None):
    if end_episode is None:
        end_episode = self.num_episodes
    _lg_enc_fix.info(f"[FIX-ENC] batch encoding episodios {start_episode}..{end_episode - 1}")
    # Sempre recarrega do disco: em modo --resume o meta.episodes tem apenas os
    # episódios originais em memória e não inclui os recém-gravados.
    if hasattr(self.meta, "_flush_metadata_buffer"):
        self.meta._flush_metadata_buffer()
    if hasattr(self.meta, "_close_writer"):
        self.meta._close_writer()
    self.meta.episodes = _load_eps_fix(self.root)

    # Mapa episode_index REAL -> posição na meta (a meta pode estar incompleta ou
    # fora de ordem após resume; acessar por posição [ep] quebra com IndexError).
    _ep_col = list(self.meta.episodes["episode_index"])
    _ep_to_pos = {int(e): i for i, e in enumerate(_ep_col)}

    def _meta_idx(ep):
        pos = _ep_to_pos.get(int(ep))
        if pos is None:
            # episódio recém-gravado ainda sem linha de meta -> assume chunk/file 0
            return 0, 0
        e = self.meta.episodes[pos]
        return e["meta/episodes/chunk_index"], e["meta/episodes/file_index"]

    chunk_idx, file_idx = _meta_idx(start_episode)
    episode_df_path = self.root / _DEF_EP_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    episode_df = _pd_enc_fix.read_parquet(episode_df_path)

    for ep_idx in range(start_episode, end_episode):
        _lg_enc_fix.info(f"[FIX-ENC] encoding videos do episodio {ep_idx}")
        c2, f2 = _meta_idx(ep_idx)
        if c2 != chunk_idx or f2 != file_idx:
            episode_df.to_parquet(episode_df_path)
            self.meta.episodes = _load_eps_fix(self.root)
            chunk_idx, file_idx = c2, f2
            episode_df_path = self.root / _DEF_EP_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
            episode_df = _pd_enc_fix.read_parquet(episode_df_path)

        video_ep_metadata = {}
        for video_key in self.meta.video_keys:
            video_ep_metadata.update(self._save_episode_video(video_key, ep_idx))
        video_ep_metadata.pop("episode_index")
        if self.meta.latest_episode is None:
            self.meta.latest_episode = {}
        self.meta.latest_episode.update(
            {k: [v] if not isinstance(v, list) else v for k, v in video_ep_metadata.items()})
        video_ep_df = _pd_enc_fix.DataFrame(
            {k: ([float(v)] if "timestamp" in k else [v]) for k, v in video_ep_metadata.items()},
            index=[ep_idx],
        ).convert_dtypes(dtype_backend="pyarrow")
        episode_df = episode_df.combine_first(video_ep_df)
        episode_df.to_parquet(episode_df_path)
        self.meta.episodes = _load_eps_fix(self.root)

_ld_mod.LeRobotDataset._batch_save_episode_video = _patched_batch_save_episode_video
print("[INJEÇÃO 3.3] FIX do batch encoding aplicado (usa meta/episodes index, não data)")

# =========================================================================
# 🔧 INJEÇÃO 3.4: FIX do _check_cached_episodes_sufficient para resume local
# O método original acessa self.meta.episodes por posição e quebra quando a
# meta está incompleta (ex: eps 6-53 faltando após crash no encoding).
# Este patch verifica só: (1) os episodes estão nos parquets de dados, e
# (2) o arquivo de vídeo chunk-000/file-000.mp4 existe no disco.
# =========================================================================
def _patched_check_cached_episodes_sufficient(self):
    import torch as _torch
    if self.hf_dataset is None or len(self.hf_dataset) == 0:
        return False
    available = {
        ep.item() if isinstance(ep, _torch.Tensor) else int(ep)
        for ep in self.hf_dataset.unique("episode_index")
    }
    if not available:
        return False
    # Verifica só se os arquivos de vídeo do chunk-000 existem (sem lookup por posição na meta)
    for vid_key in self.meta.video_keys:
        vid_path = self.root / f"videos/{vid_key}/chunk-000/file-000.mp4"
        if not vid_path.exists():
            return False
    return True

_ld_mod.LeRobotDataset._check_cached_episodes_sufficient = _patched_check_cached_episodes_sufficient
print("[INJEÇÃO 3.4] FIX do _check_cached_episodes_sufficient aplicado (resume com meta incompleta)")

# =========================================================================
# 🎤 INJEÇÃO 4: Comandos de Voz e Teclado (Setas, Pulo Duplo e PAUSE!)
# =========================================================================
import lerobot.utils.control_utils
import threading
import time

original_init_keyboard = lerobot.utils.control_utils.init_keyboard_listener

global_events = None

def patched_init_keyboard():
    global global_events
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    global_events = events
    return None, events

lerobot.utils.control_utils.init_keyboard_listener = patched_init_keyboard

# Listener de teclado global DESATIVADO — captura de qualquer janela é perigoso com robô real.
# Controle feito exclusivamente pelo Quest (botões X/Y).
# =========================================================================

# =========================================================================
# 🧊 INJEÇÃO 5: Hack de Congelamento Motor (Pause/Play)
# =========================================================================
robot_paused = False  # começa destravado; o "esperar X" é feito via controller_enabled.

original_send_action = UnitreeG1Dex3.send_action

# 🌅 SOFT-START: UMA vez por sessão, ao 1º comando, rampa da posição MEDIDA atual
# (braço caído na mesa) até o alvo (pose neutra) em ~SOFTSTART_DURATION_S segundos,
# pra não dar o "tranco" de jogar o braço direto pra pose neutra com kp alto.
# É time-based (usa time.time()), então imune a jitter do loop. NÃO relê a medida a
# cada frame (captura só no início) → não recria o loop de queda do freeze.
_softstart = {"done": False, "t0": None, "init": None}
SOFTSTART_DURATION_S = 3.0

def _read_measured(robot, action):
    """Posição q MEDIDA das juntas presentes na action (braços + cintura + mãos)."""
    from robot.unitree_g1.g1_utils import G1_29_JointIndex, G1_29_JointArmIndex
    pos = {}
    ls = getattr(robot, "_lowstate", None)
    if ls is not None:
        for motor in list(G1_29_JointIndex) + list(G1_29_JointArmIndex):
            key = f"{motor.name}.q"
            if key in action:
                pos[key] = ls.motor_state[motor.value].q
    lh = getattr(robot, "_left_hand_state", None)
    if lh is not None:
        for i, name in enumerate(robot.left_hand_joint_names):
            key = f"{name}.q"
            if key in action:
                pos[key] = lh.motor_state[i].q
    rh = getattr(robot, "_right_hand_state", None)
    if rh is not None:
        for i, name in enumerate(robot.right_hand_joint_names):
            key = f"{name}.q"
            if key in action:
                pos[key] = rh.motor_state[i].q
    return pos

def patched_send_action(self, action):
    global robot_paused

    # 🌅 SOFT-START (uma vez): rampa medido → alvo pra não dar tranco no startup.
    if not _softstart["done"]:
        cur = _read_measured(self, action)
        if _softstart["t0"] is None:
            if not cur:
                # _lowstate ainda não populado — espera o próximo frame (sem mover).
                return original_send_action(self, action)
            _softstart["t0"] = time.time()
            _softstart["init"] = cur
            print(f"\n[SOFT-START] 🌅 Rampa de {SOFTSTART_DURATION_S:.0f}s: posição atual → pose neutra. "
                  f"NÃO aperte X ainda.", flush=True)
        alpha = min((time.time() - _softstart["t0"]) / SOFTSTART_DURATION_S, 1.0)
        ramped = {k: (1.0 - alpha) * _softstart["init"].get(k, v) + alpha * v
                  for k, v in action.items()}
        # Sincroniza last_action_q com a rampa para o smoother não brigar com ela.
        if hasattr(self, "last_action_q"):
            for k, v in ramped.items():
                if k in self.last_action_q:
                    self.last_action_q[k] = v

        if alpha >= 1.0:
            _softstart["done"] = True
            print("[SOFT-START] ✅ Completo — pode apertar X para teleoperar.", flush=True)
        return original_send_action(self, ramped)

    if robot_paused:
        # 🧊 CONGELAMENTO: NÃO reler a posição MEDIDA dos motores.
        # Quando travado, get_action() já devolve a pose PRÉ-freeze ESTÁTICA
        # (self.body_joints/hand_joints não mudam, pois o IK não roda). Comandar
        # essa pose FIXA com kp=100 segura o braço contra a gravidade.
        #
        # Reler a posição medida (como era feito antes) criava um loop
        # "segue a gravidade pra baixo": ao comandar a medida, o erro de
        # posição → 0 → torque → 0 → o braço cede sob gravidade → no frame
        # seguinte relê a medida (já mais baixa) → comanda mais baixo → o braço
        # DESCE lentamente ("perdendo força"). Ao destravar, o clutch re-ancorava
        # na pose pré-freeze (que o IK não atualizou) e o braço LEVANTAVA de volta.
        #
        # Sincroniza last_action_q com o alvo PRÉ-freeze (não a medida) para o
        # smoother ver delta=0 e não cair no fallback de inicializar com a medida.
        if hasattr(self, "last_action_q"):
            for k, v in action.items():
                self.last_action_q[k] = v
        return original_send_action(self, action)

    return original_send_action(self, action)

UnitreeG1Dex3.send_action = patched_send_action

# =========================================================================
# 💉 INJEÇÃO 7: hook de save_episode — ASYNC (thread de fundo)
# =========================================================================
# O save_episode original bloqueava o loop de controle por ~2-5s (image_writer
# flush + compute_stats + parquet + video encode). Durante esse tempo o robô
# parava de receber comandos e, ao retomar, executava o movimento acumulado —
# perigoso no robô real.
#
# SOLUÇÃO: snapshot do episode_buffer no thread principal (instantâneo) + reset
# imediato do buffer (o próximo add_frame já pode rodar) + todo o trabalho pesado
# em uma thread de fundo.
#
# SEGURANÇA DO SNAPSHOT:
#   • episode_buffer é um dict de listas Python e arrays NumPy — copiável com
#     {k: list(v) if isinstance(v, list) else v.copy() if hasattr(v,'copy') else v}.
#   • Passamos o snapshot como episode_data para original_save_episode → o código
#     original usa essa cópia e não toca mais em self.episode_buffer.
#   • clear_episode_buffer() é chamado NO THREAD PRINCIPAL logo após o snapshot
#     (antes de disparar a thread), zerando o buffer para o próximo episódio.
#     Isso evita qualquer race entre a thread de save e add_frame do próximo ep.
#   • A thread de save nunca lê nem escreve self.episode_buffer após o dispatch.
#
# SERIALIZAÇÃO:
#   • _save_queue é uma queue.Queue(maxsize=10). patched_save_episode enfileira o
#     snapshot e retorna imediatamente — a thread de controle NUNCA espera o encode.
#   • _save_worker_thread (daemon, persistente) consome 1 save por vez em série,
#     preservando a ordem exigida por meta.save_episode (que muta estado global).
#   • atexit envia a sentinela None e chama queue.join() — garante flush limpo no Ctrl-C.
import lerobot.datasets.lerobot_dataset
import atexit
import queue as _queue_mod
original_save_episode = lerobot.datasets.lerobot_dataset.LeRobotDataset.save_episode

# Fila de saves: o patched_save_episode enfileira o trabalho e retorna imediatamente.
# O worker persistente (_save_worker_thread) consome 1 save por vez em série,
# preservando a ordem exigida por meta.save_episode (muta estado compartilhado).
# A thread de controle NUNCA mais fica travada esperando o encode anterior (~13s).
_save_queue: _queue_mod.Queue = _queue_mod.Queue(maxsize=10)
_save_error: list = []  # acumula Exception da thread de save (propagadas no log)

# Thread-local flag: quando True, _wait_image_writer() vira no-op.
# Setamos para True na thread de save porque já drenamos o writer no thread principal.
_save_thread_local = threading.local()

# Episódio que está sendo salvo em background (None = nenhum save ativo).
# Usado pelo patch de clear_episode_buffer para garantir que o path de discard
# (B press durante save assíncrono) crie o próximo buffer com o índice correto.
_active_save_episode_index: int | None = None

_original_wait_image_writer = lerobot.datasets.lerobot_dataset.LeRobotDataset._wait_image_writer

def _patched_wait_image_writer(self):
    """No-op quando chamado da thread de save: o drain já foi feito no thread principal."""
    if getattr(_save_thread_local, "skip_wait", False):
        return
    _original_wait_image_writer(self)

lerobot.datasets.lerobot_dataset.LeRobotDataset._wait_image_writer = _patched_wait_image_writer

_original_clear_episode_buffer = lerobot.datasets.lerobot_dataset.LeRobotDataset.clear_episode_buffer

def _delete_episode_image_dirs(dataset, episode_index) -> None:
    import shutil as _shutil, numpy as _np
    if isinstance(episode_index, _np.ndarray):
        episode_index = episode_index.item() if episode_index.size == 1 else int(episode_index[0])
    for cam_key in getattr(dataset.meta, "camera_keys", dataset.meta.image_keys + dataset.meta.video_keys):
        img_dir = dataset._get_image_file_dir(int(episode_index), cam_key)
        if img_dir.is_dir():
            _shutil.rmtree(img_dir)

def _patched_clear_episode_buffer(self, delete_images: bool = True) -> None:
    """Garante que o próximo buffer use o índice correto mesmo durante save assíncrono.

    Quando B (discard) é pressionado enquanto a thread de save está rodando,
    meta.total_episodes pode estar em estado intermediário (decrementado pela thread
    de save para passar no validate). Sem o patch, create_episode_buffer() usaria
    o valor errado e o próximo episódio gravaria no diretório errado.
    """
    if delete_images:
        if self.image_writer is not None:
            self._wait_image_writer()
        ep_idx = self.episode_buffer.get("episode_index", None)
        if ep_idx is not None:
            _delete_episode_image_dirs(self, ep_idx)
    # Se há um save assíncrono ativo, usa o próximo índice explícito para evitar
    # race com meta.total_episodes; caso contrário usa o comportamento padrão.
    if _active_save_episode_index is not None:
        self.episode_buffer = self.create_episode_buffer(episode_index=_active_save_episode_index + 1)
    else:
        self.episode_buffer = self.create_episode_buffer()

lerobot.datasets.lerobot_dataset.LeRobotDataset.clear_episode_buffer = _patched_clear_episode_buffer

def _save_queue_worker():
    """Worker persistente: consome saves da fila em série (preserva ordem de meta).

    Recebe tuplas (dataset_ref, episode_data_snapshot, ep_idx) ou None (sentinela
    de encerramento). Processa 1 save por vez — a serialização é mantida, mas a
    thread de controle não precisa mais esperar o encode anterior terminar."""
    global _active_save_episode_index
    while True:
        item = _save_queue.get()
        if item is None:            # sentinela de encerramento
            _save_queue.task_done()
            break
        dataset_ref, episode_data_snapshot, ep_idx = item
        _active_save_episode_index = ep_idx
        _save_thread_local.skip_wait = True
        try:
            # episode_data=snapshot faz o save usar nossa cópia, não self.episode_buffer.
            original_save_episode(dataset_ref, episode_data=episode_data_snapshot)
        except Exception as exc:
            _save_error.append(exc)
            print(f"[ASYNC-SAVE] ❌ Exceção na thread de save (ep {ep_idx}): {exc}", flush=True)
        finally:
            try:
                _delete_episode_image_dirs(dataset_ref, ep_idx)
            except Exception as exc:
                print(f"[ASYNC-SAVE] ⚠️ Falha limpando imagens temporárias do ep {ep_idx}: {exc}", flush=True)
            _active_save_episode_index = None
            _save_thread_local.skip_wait = False
            _save_queue.task_done()


_save_worker_thread = threading.Thread(
    target=_save_queue_worker, daemon=True, name="lerobot-save-worker")
_save_worker_thread.start()


_queue_drained = False
def _drain_save_queue():
    global _queue_drained
    if _queue_drained:
        return
    _queue_drained = True
    """Drena a fila e espera todos os saves terminarem (chamado pelo atexit)."""
    pending = _save_queue.qsize()
    print(f"[ASYNC-SAVE] ⏳ Aguardando saves/encodes pendentes antes de encerrar (fila={pending})...", flush=True)
    _save_queue.put(None)   # sentinela: manda o worker parar após processar o que já está na fila
    _save_queue.join()      # bloqueia até task_done() de TODOS os itens (inclusive o None)
    if _save_error:
        print(f"[ASYNC-SAVE] ⚠️ {len(_save_error)} erro(s) durante saves em background:", flush=True)
        for e in _save_error:
            print(f"  - {e}", flush=True)
    print("[ASYNC-SAVE] ✅ Fila de saves drenada.", flush=True)


atexit.register(_drain_save_queue)


def _snapshot_episode_buffer(episode_buffer: dict) -> dict:
    """Cria cópia rasa do episode_buffer: listas são copiadas, arrays são clonados."""
    snap = {}
    for k, v in episode_buffer.items():
        if isinstance(v, list):
            snap[k] = list(v)
        elif hasattr(v, "copy"):
            snap[k] = v.copy()
        else:
            snap[k] = v
    return snap


_original_finalize = lerobot.datasets.lerobot_dataset.LeRobotDataset.finalize
def _patched_finalize(self, *args, **kwargs):
    print("[ASYNC-SAVE] Interceptado finalize(), aguardando saves pendentes...", flush=True)
    _drain_save_queue()
    return _original_finalize(self, *args, **kwargs)
lerobot.datasets.lerobot_dataset.LeRobotDataset.finalize = _patched_finalize

def patched_save_episode(self, *args, **kwargs):
    # 🛡️ GUARD anti-crash de SAVE VAZIO ou ENCERRAR:
    # Se o usuário apertar Y (stop_recording), não queremos salvar o episódio parcial.
    import lerobot.utils.control_utils
    global_events = getattr(lerobot.utils.control_utils, "global_events", None)
    if global_events and global_events.get("stop_recording"):
        print("\n[GUARD] ⚠️ Encerrando o sistema (Y pressionado) — descartando episódio parcial.", flush=True)
        print("[ASYNC-SAVE] O encerramento vai aguardar todos os saves/encodes já enfileirados.", flush=True)
        try:
            self.clear_episode_buffer()
        except Exception:
            pass
        return None

    eb = getattr(self, "episode_buffer", None)
    size = eb.get("size", 0) if isinstance(eb, dict) else 0
    if size == 0:
        print("\n[GUARD] ⚠️ save_episode com 0 frames (A repetido / antes do 1º frame) — "
              "pulando save vazio e limpando eventos.", flush=True)
        if global_events is not None:
            global_events["exit_early"] = False
            global_events["rerecord_episode"] = False
        try:
            self.clear_episode_buffer()
        except Exception:
            self.episode_buffer = self.create_episode_buffer()
        return None

    # 1) Drena o image_writer NO THREAD PRINCIPAL antes de fazer o snapshot.
    #    Isso garante que todos os PNGs do episódio atual estejam em disco
    #    antes de a thread de save começar a encodar o vídeo.
    #    Este drain é O(queue_size) de escritas PNG — muito mais rápido (~100ms)
    #    que o save completo (~13s), logo o impacto no loop de controle é mínimo.
    if self.image_writer is not None:
        self.image_writer.wait_until_done()

    # 2) Snapshot imutável do buffer atual — feito no thread principal, O(n) de listas.
    episode_data_snapshot = _snapshot_episode_buffer(self.episode_buffer)

    # 3) Cria o próximo buffer com episode_index=N+1 explicitamente.
    #    NÃO tocamos em meta.total_episodes: a thread de save chama meta.save_episode
    #    normalmente, que o incrementa de N→N+1. Usar meta.total_episodes aqui seria
    #    um race: a thread de save pode estar em qualquer ponto da sua execução.
    next_episode_index = episode_data_snapshot["episode_index"] + 1
    try:
        _delete_episode_image_dirs(self, next_episode_index)
    except Exception:
        pass
    try:
        self.episode_buffer = self.create_episode_buffer(episode_index=next_episode_index)
    except Exception:
        self.episode_buffer = self.create_episode_buffer(episode_index=next_episode_index)

    # Referência local para o worker não fechar sobre `self` de forma inesperada.
    _dataset_ref = self
    ep_idx = episode_data_snapshot["episode_index"]

    # Registra o episódio ativo para que _patched_clear_episode_buffer (path de
    # discard B) também use o índice correto durante o save assíncrono.
    global _active_save_episode_index
    _active_save_episode_index = ep_idx

    # Enfileira o save — o worker processa em série, SEM bloquear o controle.
    # Se a fila estiver cheia (maxsize=10 episódios), put() bloqueia até haver vaga
    # (cenário extremo que nunca deve ocorrer na prática).
    try:
        _save_queue.put((_dataset_ref, episode_data_snapshot, ep_idx), block=True, timeout=2.0)
        qsize = _save_queue.qsize()
        print(f"[ASYNC-SAVE] 🚀 ep {ep_idx} enfileirado "
              f"(fila={qsize} — loop de controle livre).", flush=True)
    except _queue_mod.Full:
        # Fila cheia: processa sincronamente pra não perder o episódio.
        print(f"[ASYNC-SAVE] ⚠️ Fila cheia ao salvar ep {ep_idx} — "
              "salvando sincronamente (evita perda de dados).", flush=True)
        _save_thread_local.skip_wait = True
        try:
            original_save_episode(_dataset_ref, episode_data=episode_data_snapshot)
        except Exception as exc:
            _save_error.append(exc)
            print(f"[ASYNC-SAVE] ❌ Erro no save síncrono de fallback: {exc}", flush=True)
        finally:
            try:
                _delete_episode_image_dirs(_dataset_ref, ep_idx)
            except Exception as exc:
                print(f"[ASYNC-SAVE] ⚠️ Falha limpando imagens temporárias do ep {ep_idx}: {exc}", flush=True)
            _active_save_episode_index = None
            _save_thread_local.skip_wait = False

    return None


lerobot.datasets.lerobot_dataset.LeRobotDataset.save_episode = patched_save_episode

# =========================================================================
# 💉 INJEÇÃO 8: Logger de Debug (CSV por episódio)
# =========================================================================
import csv as _csv_mod
import time as _time_mod
from pathlib import Path as _Path

_dbg_log = {"file": None, "writer": None, "episode": -1}

def _dbg_open(episode_idx, dataset_root):
    if _dbg_log["file"]:
        _dbg_log["file"].close()
    log_dir = _Path(dataset_root) / "debug_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"episode_{episode_idx:06d}.csv"
    _dbg_log["file"] = open(path, "w", newline="")
    _dbg_log["writer"] = None  # criado na 1ª linha (lazy, pra pegar colunas dinamicamente)
    _dbg_log["episode"] = episode_idx

def _dbg_write(row: dict):
    if _dbg_log["file"] is None:
        return
    if _dbg_log["writer"] is None:
        _dbg_log["writer"] = _csv_mod.DictWriter(_dbg_log["file"], fieldnames=list(row.keys()))
        _dbg_log["writer"].writeheader()
    _dbg_log["writer"].writerow(row)
    _dbg_log["file"].flush()

# Captura obs + action a cada frame
_last_action_for_log = {}
original_send_action_for_log = UnitreeG1Dex3.send_action

def _patched_send_action_log(self, action):
    global _last_action_for_log
    _last_action_for_log = dict(action)
    return original_send_action_for_log(self, action)

UnitreeG1Dex3.send_action = _patched_send_action_log

_original_build_frame_for_log = lerobot.datasets.utils.build_dataset_frame

def _patched_build_frame_log(features, obs_dict, prefix="observation."):
    result = _original_build_frame_for_log(features, obs_dict, prefix)
    try:
        row = {"timestamp": _time_mod.time()}
        # obs escalares (juntas, pressão)
        for k, v in obs_dict.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                row[f"obs.{k}"] = float(v)
        # estatísticas das câmeras
        for cam_key in ("head_camera", "head_camera_depth"):
            img = obs_dict.get(cam_key)
            if img is not None and isinstance(img, np.ndarray):
                row[f"cam.{cam_key}.mean"] = float(img.mean())
                row[f"cam.{cam_key}.min"]  = float(img.min())
                row[f"cam.{cam_key}.max"]  = float(img.max())
                row[f"cam.{cam_key}.shape"] = str(img.shape)
        # action enviada
        for k, v in _last_action_for_log.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                row[f"act.{k}"] = float(v)
        _dbg_write(row)
    except Exception:
        pass
    return result

lerobot.datasets.utils.build_dataset_frame = _patched_build_frame_log

# Abre novo CSV a cada episódio e fecha ao salvar
_original_save_ep_log = lerobot.datasets.lerobot_dataset.LeRobotDataset.save_episode

def _patched_save_ep_log(self, *args, **kwargs):
    if _dbg_log["file"]:
        _dbg_log["file"].close()
        _dbg_log["file"] = None
        _dbg_log["writer"] = None
    return _original_save_ep_log(self, *args, **kwargs)

lerobot.datasets.lerobot_dataset.LeRobotDataset.save_episode = _patched_save_ep_log

# Detecta início de episódio via add_frame (abre o CSV no 1º frame do episódio)
_original_add_frame = lerobot.datasets.lerobot_dataset.LeRobotDataset.add_frame

_STATUS_FILE = "/tmp/g1_record_status.json"
import json as _json_status_mod

def _write_record_status(episode: int, start_time: float):
    try:
        with open(_STATUS_FILE, "w") as f:
            _json_status_mod.dump({"episode": episode, "start_time": start_time}, f)
    except Exception:
        pass

_ep_start_time: float = _time_mod.time()
_perf_last_log: float = 0.0

def _writer_qsize(dataset) -> int | None:
    try:
        writer = getattr(dataset, "image_writer", None)
        q = getattr(writer, "queue", None)
        return q.qsize() if q is not None and hasattr(q, "qsize") else None
    except Exception:
        return None

def _patched_add_frame_log(self, frame):
    global _ep_start_time, _perf_last_log
    ep_idx = self.meta.total_episodes
    if ep_idx != _dbg_log["episode"] and _dbg_log["file"] is None:
        root = self.root
        _dbg_open(ep_idx, root)
        _ep_start_time = _time_mod.time()
        _write_record_status(ep_idx, _ep_start_time)
    t0 = _time_mod.perf_counter()
    try:
        return _original_add_frame(self, frame)
    finally:
        dt_ms = (_time_mod.perf_counter() - t0) * 1000.0
        now = _time_mod.time()
        if dt_ms > 25.0 or now - _perf_last_log > 2.0:
            _perf_last_log = now
            qsize = _writer_qsize(self)
            print(f"[PERF] add_frame={dt_ms:.1f}ms image_writer_q={qsize}", flush=True)

lerobot.datasets.lerobot_dataset.LeRobotDataset.add_frame = _patched_add_frame_log

# INICIALIZAÇÃO OFICIAL
from lerobot.scripts.lerobot_record import main

# =========================================================================
# 💉 INJEÇÃO 0: make_robot_from_config instancia a classe de lerobot-EXT
# =========================================================================
# BUG RAIZ: lerobot/src/.../robots/utils.py:make_robot_from_config tem import
# hard-coded `from .unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3` → instancia a
# cópia PARCIAL de lerobot/src (sem tátil/pressão no get_observation, sem heartbeat
# anti-watchdog da Dex3). O draccus registra o config de lerobot-ext, mas a instância
# saía de lerobot/src (split-brain). Aqui forçamos a instância correta (lerobot-ext),
# que tem tátil + heartbeat + os fixes (CONFLATE removido, mão esquerda mole) e na qual
# os monkeypatches deste arquivo (soft-start anti-salto, etc.) realmente têm efeito.
import lerobot.scripts.lerobot_record as _lr_mod
import lerobot.robots.utils as _robots_utils_mod
_orig_make_robot = _robots_utils_mod.make_robot_from_config

def _patched_make_robot_from_config(config):
    if getattr(config, "type", None) == "unitree_g1_dex3":
        print("[INJEÇÃO 0] make_robot_from_config -> usando UnitreeG1Dex3 de lerobot-ext (tátil + heartbeat)")
        return UnitreeG1Dex3(config)
    return _orig_make_robot(config)

# Patcha em todos os nomes pelos quais pode ser chamado
_robots_utils_mod.make_robot_from_config = _patched_make_robot_from_config
_lr_mod.make_robot_from_config = _patched_make_robot_from_config

# =========================================================================
# 💉 INJEÇÃO 0.1: DEBOUNCE de "salvar" (exit_early) vazado entre episódios
# =========================================================================
# BUG: apertar A (salvar) repetido/rápido (double-tap) ou logo após o save faz o
# events["exit_early"] "vazar" para o próximo episódio → o record_loop de GRAVAÇÃO
# quebra com 0 frames → save_episode crasha a SESSÃO INTEIRA
#   (ValueError: "must add ... frames before add_episode").
# FIX (na fonte certa, sem tocar lerobot/src vendorizado): envolvemos record_loop e,
# no início da fase de GRAVAÇÃO (dataset presente), descartamos um exit_early tardio.
# record() resolve `record_loop` pelo global do módulo em tempo de chamada, então
# patchar o atributo do módulo basta (mesmo mecanismo da INJEÇÃO 0).
_orig_record_loop = _lr_mod.record_loop

def _patched_record_loop(*args, **kwargs):
    events = kwargs.get("events", args[1] if len(args) > 1 else None)
    dataset = kwargs.get("dataset", None)
    if events is not None and dataset is not None and events.get("exit_early"):
        events["exit_early"] = False
        print("\n[DEBOUNCE] ⏮️ exit_early tardio/repetido descartado no início do episódio "
              "(A apertado de novo durante o save) — evita save de episódio vazio.", flush=True)
    return _orig_record_loop(*args, **kwargs)

_lr_mod.record_loop = _patched_record_loop

# =========================================================================
# 🐞 INJEÇÃO DEBUG-IO: com --debug, grava TUDO enviado/recebido do robô
# =========================================================================
# Embrulha (outermost) send_action e get_observation. Só grava se a env G1_DEBUG_IO
# estiver setada (feito no __main__ quando passa --debug). No-op caso contrário.
from robot.unitree_g1.g1_debug_io import log_send as _io_send, log_recv as _io_recv

_io_orig_send = UnitreeG1Dex3.send_action
def _io_wrapped_send(self, action):
    r = _io_orig_send(self, action)
    _io_send(self)          # lê self.msg / _left_hand_msg / _right_hand_msg já publicados
    return r
UnitreeG1Dex3.send_action = _io_wrapped_send

_io_orig_obs = UnitreeG1Dex3.get_observation
def _io_wrapped_obs(self, *a, **k):
    obs = _io_orig_obs(self, *a, **k)
    _io_recv(self, obs)     # lê _lowstate / _left_hand_state.pressure direto (raw)
    return obs
UnitreeG1Dex3.get_observation = _io_wrapped_obs

# =========================================================================
# 📡 INJEÇÃO LIVE: publica telemetria (state/action/tátil) pro dashboard web
# =========================================================================
# O dashboard estilo OmniView (tools/live_omniview.py) precisa de state/action/
# tátil por frame — dados que só existem como globais DENTRO deste processo. As
# câmeras (RGB+Depth) já saem no PUB ZMQ :5555; aqui abrimos um 2º PUB (:5557)
# e, num wrapper OUTERMOST de get_observation (sempre chamado, gravando ou não),
# publicamos um pacote compacto. É best-effort: nunca derruba a gravação.
sys.path.insert(0, str(_Path(__file__).resolve().parent / "tools"))
try:
    from live_bridge import TelemetryPublisher as _TelemetryPublisher
    _live_pub = _TelemetryPublisher()
except Exception as _e:
    print(f"[LIVE] ⚠️ telemetria ao vivo desativada: {_e}")
    _live_pub = None

if _live_pub is not None:
    _live_orig_obs = UnitreeG1Dex3.get_observation
    _obs_perf_last_log = 0.0

    def _live_wrapped_obs(self, *a, **k):
        nonlocal_obs_perf = None
        t0 = _time_mod.perf_counter()
        obs = _live_orig_obs(self, *a, **k)
        obs_dt_ms = (_time_mod.perf_counter() - t0) * 1000.0
        try:
            global _obs_perf_last_log
            now_perf_log = _time_mod.time()
            if obs_dt_ms > 25.0 or now_perf_log - _obs_perf_last_log > 2.0:
                _obs_perf_last_log = now_perf_log
                print(f"[PERF] get_observation={obs_dt_ms:.1f}ms", flush=True)
            ep = max(0, _dbg_log.get("episode", 0))
            # fase do robô: soft-start (rampa) -> bloqueado (X não apertado) -> desbloqueado
            if not _softstart["done"]:
                phase = "softstart"
            elif robot_paused:
                phase = "locked"
            else:
                phase = "unlocked"
            _live_pub.publish(
                obs=obs,
                action=_last_action_for_log,
                pressure_left=buffer_pressao["left"],
                pressure_right=buffer_pressao["right"],
                episode=ep,
                frame=frame_count,
                t=_time_mod.time(),
                robot_phase=phase,
            )
        except Exception:
            pass
        return obs

    UnitreeG1Dex3.get_observation = _live_wrapped_obs


class IgnoreFPSWarningFilter(logging.Filter):
    def filter(self, record):
        return "Record loop is running slower" not in record.getMessage()

def _free_stale_ports(ports=(5555, 5558, 8012, 5557, 8765)):
    """Libera portas presas por processos órfãos de execuções anteriores.

    Quando o script é morto com kill -9 (ou crasha), podem ficar órfãos segurando
    a porta e causando 'Address already in use' na próxima execução:
      5555 = publicador de imagens ZMQ · 5558 = stream RGB-only do VR
      8012 = servidor Vuer
      5557 = telemetria ao vivo (PUB) · 8765 = dashboard web (OmniView LIVE)
    """
    import os
    import subprocess
    for port in ports:
        try:
            out = subprocess.run(
                ["ss", "-tlnp"], capture_output=True, text=True, timeout=5
            ).stdout
        except Exception:
            return
        for line in out.splitlines():
            if f":{port} " not in line:
                continue
            for tok in line.split():
                if tok.startswith("pid="):
                    pid = tok.split("pid=")[1].split(",")[0]
                    try:
                        os.kill(int(pid), 9)
                        print(f"[LIMPEZA] 🧹 Porta {port} liberada (matou PID órfão {pid})")
                    except (ProcessLookupError, ValueError, PermissionError):
                        pass


# =========================================================================
# 🚦 TERMINAL ENXUTO + PRÉ-FLIGHT
# Concentra a experiência de terminal da gravação em poucas mensagens:
# pré-flight de hardware, handshake do Quest e ações do operador (X/A/B/Y).
# Todo o resto (logs verbosos, tracebacks, libs) é desviado pro run.log do dataset.
# =========================================================================
import re as _re_ui

DEFAULT_QUEST_ADB_IP = "192.168.68.51"

# Marcador interno: força o _TermRouter a ecoar a linha no terminal real.
_UI = "\x00UI\x00"

# As ações do operador chegam do xr_g1_arm.py já com a tag [CONTROLE VR]; aqui
# reescrevemos pra versão curta exibida no terminal enxuto.
_ACTION_REWRITE = [
    (_re_ui.compile("DESTRAVADO"),  '▶️  DESTRAVOU o robô    (X)'),
    (_re_ui.compile("CONGELADO"),   '🧊 TRAVOU o robô       (X)'),
    (_re_ui.compile("SALVANDO"),    '💾 SALVOU episódio     (A)'),
    (_re_ui.compile("DESCARTANDO"), '🗑️  DESCARTOU episódio  (B)'),
    (_re_ui.compile("ENCERRANDO"),  '🛑 ENCERRANDO          (Y)'),
]


class _TermRouter:
    """Substitui sys.stdout/stderr: grava TUDO no run.log e ecoa no terminal real
    só (a) linhas marcadas com _UI (via ui()) e (b) ações do operador (CONTROLE VR)."""

    def __init__(self, logfile, term):
        self._log = logfile
        self._term = term
        self._buf = ""

    def write(self, s):
        try:
            self._log.write(s.replace(_UI, ""))
        except Exception:
            pass
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._echo(line)
        return len(s)

    def _echo(self, line):
        if _UI in line:
            self._term.write(line.replace(_UI, "") + "\n")
            self._term.flush()
            return
        if "[CONTROLE VR]" in line:
            for rx, rep in _ACTION_REWRITE:
                if rx.search(line):
                    self._term.write("   " + rep + "\n")
                    self._term.flush()
                    return

    def flush(self):
        try:
            self._log.flush()
        except Exception:
            pass

    def isatty(self):
        return False


def ui(msg=""):
    """Imprime uma linha no terminal enxuto (e também grava no run.log)."""
    print(_UI + str(msg), flush=True)


def _port_open(host, port, timeout=2.0):
    """True se dá pra abrir uma conexão TCP em host:port (serviço no ar)."""
    import socket
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _probe_camera(host, timeout=5.0):
    """Lê 1 frame do stream ZMQ :5555 e diz (rgb_ok, depth_ok)."""
    import zmq, json
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt_string(zmq.SUBSCRIBE, "")
    s.connect(f"tcp://{host}:5555")
    try:
        parts = s.recv_multipart()
        data = json.loads(parts[0])
        imgs = data.get("images", data)
        return imgs.get("head_camera") is not None, imgs.get("head_camera_depth") is not None
    except Exception:
        return False, False
    finally:
        s.close(0)


def _adb_usb_present():
    """True se há um Quest conectado por USB (device em 'adb devices', sem :5555)."""
    import subprocess
    try:
        out = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return False
    for line in out.splitlines()[1:]:
        toks = line.split()
        if len(toks) >= 2 and toks[1] == "device" and ":5555" not in toks[0]:
            return True
    return False


def _enable_wifi_debug(adb_ip):
    """adb tcpip 5555 (precisa do USB) + adb connect <ip>:5555."""
    import subprocess, time
    try:
        subprocess.run(["adb", "tcpip", "5555"], capture_output=True, text=True, timeout=8)
        time.sleep(1.0)
        r = subprocess.run(["adb", "connect", f"{adb_ip}:5555"],
                           capture_output=True, text=True, timeout=8)
        return "connected" in (r.stdout + r.stderr).lower()
    except Exception:
        return False


def _disable_proximity(adb_ip):
    """Desativa o sensor de proximidade do Quest (mantém a tela ligada o tempo todo)."""
    import subprocess
    try:
        r = subprocess.run(["adb", "-s", f"{adb_ip}:5555", "shell", "am", "broadcast",
                            "-a", "com.oculus.vrpowermanager.prox_close"],
                           capture_output=True, text=True, timeout=8)
        return "Broadcast completed" in r.stdout
    except Exception:
        return False


def _quest_browser_connected():
    """True se há conexão TCP estabelecida na :8012 (browser do Quest no Vuer)."""
    import subprocess
    try:
        out = subprocess.run(
            ["ss", "-tnH", "state", "established", "( sport = :8012 )"],
            capture_output=True, text=True, timeout=5).stdout
        return bool(out.strip())
    except Exception:
        return False


def _robot_iface():
    """Interface de rede com IP na subrede do robô (192.168.123.x)."""
    import subprocess
    try:
        out = subprocess.run(["ip", "-o", "-4", "addr", "show"],
                             capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if "192.168.123." in line:
                return line.split()[1]
    except Exception:
        pass
    return None


def _link_speed_mbps(iface):
    """Velocidade negociada do link em Mbps (via /sys/class/net), ou None."""
    if not iface:
        return None
    try:
        with open(f"/sys/class/net/{iface}/speed") as f:
            v = int(f.read().strip())
            return v if v > 0 else None
    except Exception:
        return None


def _preflight_step(n, total, label, check, fix, wait=True, poll=2.0):
    """Um passo do pré-flight. Se já estiver OK, marca ✅ e segue. Se falhar,
    mostra a instrução de correção e (wait=True) BLOQUEIA re-checando até ficar
    verde, então segue pro próximo. Retorna o estado final (True/False)."""
    import time
    if check():
        ui(f"   ✅ {n}/{total}  {label}")
        return True
    ui(f"   ❌ {n}/{total}  {label}")
    ui("")
    for _l in fix.splitlines():
        ui(f"      {_l}")
    ui("")
    if not wait:
        return False
    ui("      … aguardando você corrigir (Ctrl-C cancela)")
    while not check():
        time.sleep(poll)
    ui(f"   ✅ {n}/{total}  {label}")
    ui("")
    return True


def _stage_manager():
    """Thread (roda junto do main()): handshake do Quest + banners de prontidão.

    Bloqueia o banner 'aperte X' até o browser do Quest conectar no Vuer (:8012)
    e o soft-start terminar. O X em si é botão do controle VR, então só funciona
    com o Quest já conectado — isto aqui só cuida da APRESENTAÇÃO na ordem certa."""
    import time
    # passo 7/7 do pré-flight: espera o Vuer subir (dentro do main()) e o F5 no Quest.
    while not _port_open("127.0.0.1", 8012, timeout=1.0):
        time.sleep(0.5)
    ui("   ⏳ 8/8  Dê F5 uma vez no óculos VR para conectar ao Vuer")
    while not _quest_browser_connected():
        time.sleep(0.5)
    ui("   ✅ 8/8  Óculos VR conectado")
    while not _softstart["done"]:
        time.sleep(0.2)
    ui(); ui(); ui(); ui()
    ui("=" * 79)
    ui('  Tudo pronto para Gravação/Teleoperação do robô. Aperte "X" para desbloquear.')
    ui("=" * 79)
    ui()


_watchdog_stop = threading.Event()
_watchdog_paused = False  # True = watchdog está segurando o pause (não o usuário)


def _watchdog(robot_ip: str):
    """Thread: monitora câmera, Quest WiFi e robô durante a run.

    Começa a checar após o soft-start (sistema operacional). Se qualquer
    conexão cair: congela o robô e avisa. Quando volta: destrava e avisa."""
    import time
    global robot_paused, _watchdog_paused

    while not _softstart["done"] and not _watchdog_stop.is_set():
        time.sleep(0.5)

    INTERVAL = 3.0
    while not _watchdog_stop.is_set():
        robot_ok = _port_open(robot_ip, 6000, timeout=2.0)
        cam_ok   = _port_open(robot_ip, 5555, timeout=2.0)
        quest_ok = _quest_browser_connected()

        lost = []
        if not robot_ok:
            lost.append("Robô (:6000)")
        if not cam_ok:
            lost.append("Câmera (:5555)")
        if not quest_ok:
            lost.append("Quest VR")

        if lost and not _watchdog_paused:
            _watchdog_paused = True
            robot_paused = True
            ui(f"[WATCHDOG] ⚠️  Perdido: {', '.join(lost)} — Robô CONGELADO 🧊")
        elif not lost and _watchdog_paused:
            _watchdog_paused = False
            robot_paused = False
            ui("[WATCHDOG] ✅  Conexões restauradas — Robô DESTRAVADO ▶️")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    import os
    from datetime import datetime
    from pathlib import Path

    if os.environ.get("_G1_REEXEC"):
        print("[ENV] ✅ G1 Ativado", file=sys.stderr)

    cli_args = sys.argv[:]

    if "--config_path" not in str(cli_args):
        print("\n[ERRO]: O argumento '--config_path' é obrigatório.", file=sys.stderr)
        sys.exit(1)

    # Extrai --quest-ip (IP do servidor Vuer = ESTE notebook, p/ o browser do Quest abrir)
    # e --quest-adb-ip (IP do PRÓPRIO Quest na rede, p/ ADB; muda por DHCP). NÃO passa pro lerobot.
    quest_ip = None
    quest_adb_ip = None
    for arg in sys.argv[:]:
        if arg.startswith("--quest-ip="):
            quest_ip = arg.split("=", 1)[1]
            sys.argv.remove(arg)
        elif arg == "--quest-ip":
            idx = sys.argv.index(arg)
            quest_ip = sys.argv[idx + 1]
            sys.argv.pop(idx + 1)
            sys.argv.pop(idx)
        elif arg.startswith("--quest-adb-ip="):
            quest_adb_ip = arg.split("=", 1)[1]
            sys.argv.remove(arg)
        elif arg == "--quest-adb-ip":
            idx = sys.argv.index(arg)
            quest_adb_ip = sys.argv[idx + 1]
            sys.argv.pop(idx + 1)
            sys.argv.pop(idx)

    # --left-arm-limp: braço esquerdo INTEIRO solto (kp=0 nas juntas 15-21) → não segue o
    # controle; você empurra fisicamente pra fora do quadro. Só o lado direito é teleoperado.
    if "--left-arm-limp" in sys.argv:
        sys.argv.remove("--left-arm-limp")
        os.environ["G1_LEFT_ARM_LIMP"] = "1"
        print("[CONFIG] 🫳 Braço ESQUERDO solto (kp=0 nas juntas 15-21) — empurre pra fora do quadro.")

    # --debug: grava TUDO enviado/recebido do robô (send/recv) num JSONL p/ diagnóstico.
    if "--debug" in sys.argv:
        sys.argv.remove("--debug")
        from datetime import datetime as _dt_dbg
        _io_path = f"/tmp/g1_debug_io_{_dt_dbg.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        os.environ["G1_DEBUG_IO"] = _io_path
        print(f"\n[DEBUG-IO] 🐞 Gravando envio/recebimento do robô em: {_io_path}")
        print(f"[DEBUG-IO]    Analise depois com: python lerobot-ext/analyze_debug_io.py {_io_path}\n")

    # Limpa portas presas por execuções anteriores que não encerraram limpo.
    _free_stale_ports()

    force_sim = "--sim" in cli_args or "--simulation=true" in cli_args
    if "--sim" in sys.argv: sys.argv.remove("--sim")

    if force_sim:
        sys.argv.append("--robot.is_simulation=true")
        sys.argv.append("--teleop.is_simulation=true")
    else:
        sys.argv.append("--robot.is_simulation=false")
        sys.argv.append("--teleop.is_simulation=false")

    # =========================================================================
    # 💉 INJEÇÃO 6: Auto-timestamp para datasets (evita sobrescrever runs anteriores)
    # Lê o 'root' base do YAML de config e cria uma subpasta com timestamp,
    # assim cada gravação fica salva separadamente sem precisar apagar nada.
    # =========================================================================
    has_root_arg = any("--dataset.root=" in arg for arg in sys.argv)
    if not has_root_arg:
        import yaml
        config_idx = sys.argv.index("--config_path") + 1
        config_path = sys.argv[config_idx]
        with open(config_path) as f:
            cfg_yaml = yaml.safe_load(f)
        base_root = cfg_yaml.get("dataset", {}).get("root", "datasets/default")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_root = f"{base_root}/{timestamp}"
        Path(timestamped_root).parent.mkdir(parents=True, exist_ok=True)
        sys.argv.append(f"--dataset.root={timestamped_root}")
        print(f"\n[AUTO-DATASET] 📁 Salvando em: {timestamped_root}")

    # =====================================================================
    # 🚦 TERMINAL ENXUTO + PRÉ-FLIGHT
    # =====================================================================
    # --dry-preflight: só roda as checagens e sai (sem gravar) — pra testar setup.
    _dry_preflight = "--dry-preflight" in sys.argv
    if _dry_preflight:
        sys.argv.remove("--dry-preflight")

    # run.log fica junto do dataset desta run — todo log verboso é desviado pra lá.
    if has_root_arg:
        _ds_root = next(a.split("=", 1)[1] for a in sys.argv
                        if a.startswith("--dataset.root="))
    else:
        _ds_root = timestamped_root
    # run.log começa fora de _ds_root (o LeRobot exige criar essa pasta vazio) e é
    # movido pra dentro no fim — ver _finalize_run_log.
    _ds_root_p = Path(_ds_root)
    _ds_root_p.parent.mkdir(parents=True, exist_ok=True)
    _run_log_tmp = _ds_root_p.parent / f".{_ds_root_p.name}.run.log"
    _run_log_final = _ds_root_p / "run.log"
    _run_log_path = str(_run_log_tmp)

    _LOGFILE = open(_run_log_path, "a", buffering=1)
    try:
        _TERM = open("/dev/tty", "w")
    except OSError:
        _TERM = os.fdopen(os.dup(1), "w")   # cópia do stdout original (sobrevive ao redirect dos fds abaixo)
    sys.stdout = _TermRouter(_LOGFILE, _TERM)
    sys.stderr = sys.stdout
    # despeja no run.log tudo que os imports/patches imprimiram durante o boot
    try:
        _LOGFILE.write(_BOOT_CAP.getvalue())
        _LOGFILE.flush()
    except Exception:
        pass
    # ao final, move o run.log pra dentro do dataset (que o LeRobot já criou).
    import atexit as _atexit_runlog
    def _finalize_run_log():
        try:
            _LOGFILE.flush()
            if _ds_root_p.exists() and _run_log_tmp.exists():
                import shutil
                shutil.move(str(_run_log_tmp), str(_run_log_final))
        except Exception:
            pass
    _atexit_runlog.register(_finalize_run_log)

    # Redireciona os fds 1/2 do SO pro run.log: captura saída de código C (libx264/
    # ffmpeg via PyAV) e de subprocessos que escrevem direto no fd, fora do
    # sys.stdout/stderr do Python. O terminal enxuto segue pelo _TERM (fd separado).
    try:
        os.dup2(_LOGFILE.fileno(), 1)
        os.dup2(_LOGFILE.fileno(), 2)
    except Exception:
        pass
    # logging verboso -> run.log (tira os StreamHandlers que sujariam o terminal)
    _root_lg = logging.getLogger()
    for _h in list(_root_lg.handlers):
        if isinstance(_h, logging.StreamHandler) and not isinstance(_h, logging.FileHandler):
            _root_lg.removeHandler(_h)
    _fh = logging.FileHandler(_run_log_path)
    _fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    _root_lg.addHandler(_fh)
    _root_lg.setLevel(logging.INFO)
    ui(f"📁 Dataset: {_ds_root}")
    ui(f"📝 Log completo desta run: {_run_log_final}")

    # ---- PRÉ-FLIGHT (só no robô real) ----
    # Cada passo orienta a correção e espera ficar verde antes de ir pro próximo.
    # (--dry-preflight não bloqueia: só reporta o estado e sai.)
    _PREFLIGHT_ROBOT_IP = "192.168.123.164"
    _adb_ip = quest_adb_ip or DEFAULT_QUEST_ADB_IP
    _all_ok = True
    if not force_sim:
        ui()
        ui("🚦 PRÉ-FLIGHT — verificando hardware")
        _wait = not _dry_preflight
        _steps = [
            (f"Robô (./start_robot.sh @ {_PREFLIGHT_ROBOT_IP}:6000-6003)",
             lambda: _port_open(_PREFLIGHT_ROBOT_IP, 6000, timeout=3.0),
             "Conecte o cabo de ethernet do Robô ao seu Notebook.\n"
             "Entre na interface de rede do Prometheus."),
            ("Rede Gigabit (link 1000 Mb/s)",
             lambda: (_link_speed_mbps(_robot_iface()) or 0) >= 1000,
             "Rede negociou abaixo de 1000 Mb/s → a câmera engasga (~10 Hz).\n"
             "Reassente/troque o cabo ethernet (Cat5e/6) FIRME nos dois lados —\n"
             "contato marginal derruba pra 100."),
            ("Câmera RGB (head_camera @ :5555)",
             lambda: _probe_camera(_PREFLIGHT_ROBOT_IP, timeout=4.0)[0],
             "Suba os servidores no Robô (start_robot.sh) e confira a RealSense D435i conectada."),
            ("Câmera Depth (head_camera_depth, uint16)",
             lambda: _probe_camera(_PREFLIGHT_ROBOT_IP, timeout=4.0)[1],
             "O depth vem do mesmo realsense_server do passo anterior.\n"
             "Se só o depth faltou, reinicie o realsense_server no Robô."),
            ("Quest via USB (adb devices)",
             _adb_usb_present,
             "Coloque o óculos no rosto e aperte o botão do lado direito para ligá-lo."),
            (f"Depuração WiFi (adb tcpip 5555 → {_adb_ip})",
             lambda: _enable_wifi_debug(_adb_ip),
             "Mantenha o Quest no cabo USB e autorize este computador no headset."),
            ("Sensor de proximidade off (óculos fica ligado)",
             lambda: _disable_proximity(_adb_ip),
             "Coloque/segure o Quest com a tela acesa e autorize este computador."),
        ]
        # 1-7 = hardware (aqui); 8 = "dê F5 no óculos VR" (no _stage_manager, após o
        # Vuer subir dentro do main()).
        _PREFLIGHT_TOTAL = 8
        for _i, (_lbl, _chk, _fix) in enumerate(_steps, 1):
            _ok = _preflight_step(_i, _PREFLIGHT_TOTAL, _lbl, _chk, _fix, wait=_wait)
            _all_ok = _all_ok and _ok
        if not _wait:
            ui("   ⏭️  8/8  Dê F5 no óculos VR — verificado só na gravação (não no --dry-preflight)")
            if not _all_ok:
                ui("❌ PRÉ-FLIGHT: há itens pendentes acima.")

    if _dry_preflight:
        sys.exit(0 if (force_sim or _all_ok) else 1)

    logging.getLogger().addFilter(IgnoreFPSWarningFilter())
    logging.getLogger("lerobot").addFilter(IgnoreFPSWarningFilter())

    # Mantém o Quest 2 acordado e (opcionalmente) abre o Vuer no browser
    import subprocess
    # IP de ADB do Quest (o PRÓPRIO headset, não o laptop). Muda por DHCP, então é
    # configurável via --quest-adb-ip. Default = último IP conhecido do Quest.
    # NÃO é o --quest-ip (esse é o IP do laptop/servidor Vuer).
    QUEST_ADB = f"{quest_adb_ip}:5555" if quest_adb_ip else "192.168.68.51:5555"

    def _adb(*args):
        return subprocess.run(["adb", "-s", QUEST_ADB] + list(args),
                              capture_output=True, text=True, timeout=5)

    # Abre o viewer em processo separado e registra para matar ao sair.
    # Default: dashboard web estilo OmniView (RGB+Depth+Trajetória+Rampas+Tátil ao vivo).
    # Fallback: a janela OpenCV simples antiga, via env G1_VIEWER=cv2.
    
    # 🚀 OTIMIZAÇÃO CRÍTICA DE REDE (Proxy ZMQ Local)
    # Evita que init_lerobot_record, xr_g1_arm e live_omniview abram 3 streams de 60 MB/s
    # sobrecarregando a placa de rede Gigabit do laptop/robô.
    import zmq as _zmq
    import threading as _threading
    import base64 as _b64_vr
    import json as _json_vr
    ROBOT_IP = "127.0.0.1" if force_sim else "192.168.123.164"
    def _zmq_proxy():
        try:
            ctx = _zmq.Context.instance()
            f = ctx.socket(_zmq.SUB)
            f.setsockopt(_zmq.RCVHWM, 1)
            f.setsockopt(_zmq.LINGER, 0)
            f.connect(f"tcp://{ROBOT_IP}:5555")
            f.setsockopt_string(_zmq.SUBSCRIBE, "")
            
            b = ctx.socket(_zmq.PUB)
            b.setsockopt(_zmq.SNDHWM, 1)
            b.setsockopt(_zmq.LINGER, 0)
            b.bind("tcp://127.0.0.1:5555")
            
            while True:
                parts = f.recv_multipart()
                while True:
                    try:
                        parts = f.recv_multipart(flags=_zmq.NOBLOCK)
                    except _zmq.Again:
                        break
                try:
                    b.send_multipart(parts, flags=_zmq.NOBLOCK)
                except _zmq.Again:
                    continue
        except Exception as e:
            print(f"[ZMQ Proxy] Erro no proxy local: {e}")

    def _zmq_vr_rgb_proxy():
        """Publica só head_camera como JPEG bytes para o Quest/Vuer.

        O stream :5555 continua completo (RGB+Depth) para dataset/OmniView. Este
        canal :5558 evita depth/base64 no caminho crítico do óculos.
        """
        try:
            ctx = _zmq.Context.instance()
            f = ctx.socket(_zmq.SUB)
            f.setsockopt(_zmq.RCVHWM, 1)
            f.setsockopt(_zmq.LINGER, 0)
            f.connect("tcp://127.0.0.1:5555")
            f.setsockopt_string(_zmq.SUBSCRIBE, "")

            b = ctx.socket(_zmq.PUB)
            b.setsockopt(_zmq.SNDHWM, 1)
            b.setsockopt(_zmq.LINGER, 0)
            b.bind("tcp://127.0.0.1:5558")

            while True:
                parts = f.recv_multipart()
                while True:
                    try:
                        parts = f.recv_multipart(flags=_zmq.NOBLOCK)
                    except _zmq.Again:
                        break
                try:
                    data = _json_vr.loads(parts[0])
                    head = data.get("images", data).get("head_camera")
                    if isinstance(head, dict) and data.get("protocol") == "zmq.compressed.v1":
                        part = head.get("part")
                        if head.get("encoding") != "jpeg" or part is None or part >= len(parts):
                            continue
                        b.send(parts[part], flags=_zmq.NOBLOCK)
                    elif isinstance(head, str):
                        b.send(_b64_vr.b64decode(head), flags=_zmq.NOBLOCK)
                    else:
                        continue
                except _zmq.Again:
                    continue
                except Exception:
                    continue
        except Exception as e:
            print(f"[ZMQ VR Proxy] Erro no proxy RGB local: {e}")

    if not force_sim:
        _t_proxy = _threading.Thread(target=_zmq_proxy, daemon=True)
        _t_proxy.start()
    else:
        print("[ZMQ Proxy] Sim: usando publisher local :5555 direto (sem forwarder principal)")
    _t_vr_proxy = _threading.Thread(target=_zmq_vr_rgb_proxy, daemon=True)
    _t_vr_proxy.start()
    
    # Agora que o proxy local está rodando em 127.0.0.1:5555, direcionamos todos os clientes pra ele!
    cam_host = "127.0.0.1"
    os.environ["G1_VR_CAM_PORT"] = "5558"
    
    # Em vez de modificar sys.argv (que quebra o draccus com dicionários aninhados),
    # nós interceptamos a função record() para modificar a config depois do parse.
    import lerobot.scripts.lerobot_record as _lr_mod
    _orig_record_call = getattr(_lr_mod, "record", None)
    if _orig_record_call:
        def _patched_record_call(*args, **kwargs):
            if not args and not kwargs:
                import draccus
                _lr_mod.register_third_party_plugins()
                cfg = draccus.parse(_lr_mod.RecordConfig)
                # Durante a gravação, add_frame salva imagens temporárias por câmera.
                # Processos exigem pickle/cópia de arrays grandes e podem piorar o
                # lag; por padrão ficamos em threads e aceleramos o RGB temporário
                # salvando JPEG rápido no patch de write_image acima.
                if hasattr(cfg, "dataset"):
                    if getattr(cfg.dataset, "num_image_writer_processes", 0) == 0:
                        cfg.dataset.num_image_writer_processes = int(os.environ.get("G1_IMAGE_WRITER_PROCESSES", "0"))
                    if getattr(cfg.dataset, "num_image_writer_threads_per_camera", 0) == 4:
                        cfg.dataset.num_image_writer_threads_per_camera = int(os.environ.get("G1_IMAGE_WRITER_THREADS_PER_CAMERA", "8"))
                    print(
                        "[IMAGE-WRITER] processos="
                        f"{cfg.dataset.num_image_writer_processes} "
                        "threads/cam="
                        f"{cfg.dataset.num_image_writer_threads_per_camera}",
                        flush=True,
                    )
                if hasattr(cfg, "robot") and hasattr(cfg.robot, "cameras"):
                    for cam_name, cam_cfg in cfg.robot.cameras.items():
                        if getattr(cam_cfg, "type", "") == "zmq" and hasattr(cam_cfg, "video"):
                            cam_cfg.video.host = "127.0.0.1"
                            cam_cfg.video.port = 5555
                return _orig_record_call(cfg)
            return _orig_record_call(*args, **kwargs)
        _lr_mod.record = _patched_record_call
    
    import atexit as _atexit
    if os.environ.get("G1_VIEWER", "").lower() == "cv2":
        viewer_script = str(Path(__file__).resolve().parent.parent / "view_cam_live.py")
        _viewer_proc = subprocess.Popen(
            [sys.executable, viewer_script, "--host", cam_host, "--port", "5555"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"[CamViewer] 📷 Viewer OpenCV iniciado (RGB + Depth via Proxy em {cam_host}:5555)")
    else:
        viewer_script = str(Path(__file__).resolve().parent / "tools" / "live_omniview.py")
        _quest_adb = f"{quest_adb_ip}:5555" if quest_adb_ip else ""
        # Default: lê o stream COMPLETO :5555 (multipart RGB+Depth) → mostra depth no dashboard.
        # A depth roda em processo separado (não pesa no loop de gravação).
        #
        # G1_OMNIVIEW_RGB_ONLY=1 → volta ao modo antigo: lê o :5558 (JPEG RGB-only do proxy
        # VR), sem depth. Mais leve de CPU e aproveita o proxy :5558 que já roda. Use quando
        # não precisar ver a depth ao vivo.
        _rgb_only = os.environ.get("G1_OMNIVIEW_RGB_ONLY", "0") not in ("", "0", "false", "False")
        if _rgb_only:
            _viewer_proc = subprocess.Popen(
                [sys.executable, viewer_script, "--host", cam_host,
                 "--cam-port", "5558", "--tele-port", "5557", "--http-port", "8765",
                 "--quest-adb", _quest_adb, "--img-fps", "30", "--rgb-bytes", "--no-depth"],
                stdout=_LOGFILE, stderr=_LOGFILE,
            )
            print("[OmniView LIVE] 🖥️  Dashboard web iniciado — RGB-only :5558, 30 FPS, depth OFF — abrindo http://127.0.0.1:8765/live.html")
            print("                (G1_OMNIVIEW_RGB_ONLY ligado; tire-o pra ver a depth via :5555)")
        else:
            _viewer_proc = subprocess.Popen(
                [sys.executable, viewer_script, "--host", cam_host,
                 "--cam-port", "5555", "--tele-port", "5557", "--http-port", "8765",
                 "--quest-adb", _quest_adb, "--img-fps", "30"],
                stdout=_LOGFILE, stderr=_LOGFILE,
            )
            print("[OmniView LIVE] 🖥️  Dashboard web iniciado — RGB+Depth :5555, 30 FPS — abrindo http://127.0.0.1:8765/live.html")
            print("                (G1_OMNIVIEW_RGB_ONLY=1 = RGB-only sem depth; G1_VIEWER=cv2 volta à janela OpenCV antiga)")
    _atexit.register(lambda: _viewer_proc.poll() is None and _viewer_proc.terminate())

    # ADB connect + sensor de proximidade já foram tratados no PRÉ-FLIGHT (itens 5 e 6).
    # Aqui só abrimos o Vuer no browser do Quest (depende do --quest-ip).
    try:
        if quest_ip:
            vuer_url = f"https://{quest_ip}:8012/?grid=False&ws=wss://{quest_ip}:8012"
            # Fecha o browser completamente antes de abrir (limpa guias e experiência VR anterior)
            _adb("shell", "am force-stop com.oculus.browser")
            import time; time.sleep(1)
            # URL precisa de aspas simples no shell Android p/ o & não ser interpretado
            _adb("shell", f"am start -a android.intent.action.VIEW -d '{vuer_url}'")
            print(f"[Quest] 🌐 Browser aberto em {vuer_url}")
    except Exception:
        print("[Quest] ⚠️ Não consegui abrir o browser no Quest via ADB.")

    # Handshake do Quest + banners de prontidão + watchdog rodam em paralelo ao main().
    if not force_sim:
        _threading.Thread(target=_stage_manager, daemon=True).start()
        _threading.Thread(
            target=_watchdog, args=(_PREFLIGHT_ROBOT_IP,), daemon=True
        ).start()

    import signal
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _watchdog_stop.set()
        ui("")
        ui("🛑 Gravação finalizada pelo usuário.")
        sys.exit(0)
    except Exception as e:
        _watchdog_stop.set()
        import traceback
        traceback.print_exc()  # traceback completo vai pro run.log
        ui("")
        ui(f"❌ ERRO: {type(e).__name__}: {e}")
        ui(f"   Detalhes em: {_run_log_final if _ds_root_p.exists() else _run_log_tmp}")
        sys.exit(1)
