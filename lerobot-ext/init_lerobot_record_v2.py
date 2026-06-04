#!/usr/bin/env python
"""
Data Collection Entry Point - HACKER EDITION V9 + VOICE CONTROL
Corrigindo validação de Tuplas e adicionando controle Hands-Free.
"""

import sys
import logging
import numpy as np
import threading
import time

frame_count = 0

# Buffer global de contrabando
buffer_pressao = {"left": np.zeros(33, dtype=np.float32), "right": np.zeros(33, dtype=np.float32)}

try:
    import robot.unitree_g1
    import teleop.unitree_g1
except ImportError as e:
    print(f"\n[IMPORT ERROR]: {e}")
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
            "dtype": "float32", "shape": (33,), "names": [f"left_hand_pressure_{i}" for i in range(33)]
        }
        dataset_features["observation.right_hand_pressure"] = {
            "dtype": "float32", "shape": (33,), "names": [f"right_hand_pressure_{i}" for i in range(33)]
        }
        
        chave_depth = "observation.images.head_camera_depth"
        if chave_depth in dataset_features:
            # depth como imagem uint16 1-canal (PNG), em vez de video h264 8-bit
            print("[HACK LEROBOT] Depth -> PNG uint16 1-canal (mm)")
            dataset_features[chave_depth] = {
                "dtype": "image",
                "shape": (480, 848, 1),
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
    
    for i in range(33):
        obs_dict[f"left_hand_pressure_{i}"] = float(lp[i])
        obs_dict[f"right_hand_pressure_{i}"] = float(rp[i])
        
    return original_build_dataset_frame(features, obs_dict, prefix)

lerobot.datasets.utils.build_dataset_frame = patched_build_dataset_frame

# =========================================================================
# 💉 INJEÇÃO 3.1: decodifica a câmera *_depth como uint16 (IMREAD_UNCHANGED)
# A ZMQCamera.read padrão usa IMREAD_COLOR, que converteria a depth pra BGR 8-bit.
# =========================================================================
import lerobot.cameras.zmq.camera_zmq as _zmq_cam_mod
import base64 as _b64_dep
import json as _json_dep
import cv2 as _cv2_dep

_orig_zmq_read = _zmq_cam_mod.ZMQCamera.read

def _patched_zmq_read(self, color_mode=None):
    if "depth" not in getattr(self, "camera_name", ""):
        # CORREÇÃO DE COR: o servidor publica BGR (RealSense rs.format.bgr8) e o
        # ZMQCamera.read faz cv2.imdecode (retorna BGR) SEM converter → o dataset
        # ficava gravado em BGR. O LeRobot e os backbones das VLAs (ImageNet/SigLIP)
        # esperam RGB. Convertemos aqui para gravar RGB de verdade.
        frame = _orig_zmq_read(self, color_mode)            # BGR (imdecode)
        return _cv2_dep.cvtColor(frame, _cv2_dep.COLOR_BGR2RGB)  # → RGB pro dataset

    if not self.is_connected or self.socket is None:
        from lerobot.utils.errors import DeviceNotConnectedError
        raise DeviceNotConnectedError(f"{self} is not connected.")
    try:
        message = self.socket.recv_string()
    except Exception as e:
        if type(e).__name__ == "Again":
            raise TimeoutError(f"{self} timeout after {self.timeout_ms}ms") from e
        raise

    images = _json_dep.loads(message).get("images", {})
    img_b64 = images.get(self.camera_name) or (next(iter(images.values())) if images else None)
    if img_b64 is None:
        raise RuntimeError(f"{self} no images in message")

    raw = _b64_dep.b64decode(img_b64)
    frame = _cv2_dep.imdecode(np.frombuffer(raw, np.uint8), _cv2_dep.IMREAD_UNCHANGED)  # uint16 (H,W)
    if frame is None:
        raise RuntimeError(f"{self} failed to decode depth image")
    if frame.ndim == 2:
        frame = frame[:, :, None]  # (H,W) -> (H,W,1)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        frame = frame[:, :, 0:1]  # depth triplicada em 3 canais -> 1 canal
    return frame

_zmq_cam_mod.ZMQCamera.read = _patched_zmq_read

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
        elif isinstance(image, _PIL.Image):
            img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        img.save(fpath, compress_level=compress_level)
    except Exception as e:
        print(f"Error writing image {fpath}: {e}")

_img_writer_mod.write_image = _patched_write_image

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
# 💉 INJEÇÃO 7: hook de save_episode (síncrono — não resetar warmup)
# =========================================================================
# NOTA: save_episode bloqueia o loop por ~2-5s (image_writer flush + compute_stats +
# embed_images). Durante esse tempo o heartbeat segura o robô na última posição
# comandada (não solta o copo, não perde a pose). Uma abordagem async foi tentada
# mas é insegura: episode_buffer e image_writer são estado compartilhado não
# thread-safe entre o save e o novo loop de gravação.
import lerobot.datasets.lerobot_dataset
original_save_episode = lerobot.datasets.lerobot_dataset.LeRobotDataset.save_episode

def patched_save_episode(self, *args, **kwargs):
    # 🛡️ GUARD anti-crash de SAVE VAZIO: se o usuário apertar A (salvar) repetido
    # (double/triple-tap) ou A antes do 1º frame, o record_loop quebra com 0 frames e
    # save_episode -> validate_episode_buffer levanta
    #   ValueError("You must add one or several frames ... before calling add_episode")
    # matando a SESSÃO INTEIRA de gravação. Aqui detectamos buffer vazio, pulamos o save
    # e limpamos as flags vazadas (exit_early/rerecord) pra não cascatear no próximo episódio.
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
    return original_save_episode(self, *args, **kwargs)

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

def _patched_add_frame_log(self, frame):
    global _ep_start_time
    ep_idx = self.meta.total_episodes
    if ep_idx != _dbg_log["episode"] and _dbg_log["file"] is None:
        root = self.root
        _dbg_open(ep_idx, root)
        _ep_start_time = _time_mod.time()
        _write_record_status(ep_idx, _ep_start_time)
    return _original_add_frame(self, frame)

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

class IgnoreFPSWarningFilter(logging.Filter):
    def filter(self, record):
        return "Record loop is running slower" not in record.getMessage()

def _free_stale_ports(ports=(5555, 8012)):
    """Libera portas presas por processos órfãos de execuções anteriores.

    Quando o script é morto com kill -9 (ou crasha), o subprocesso publicador
    de imagens ZMQ (porta 5555) e o servidor Vuer (8012) podem ficar órfãos
    segurando a porta, causando 'Address already in use' na próxima execução.
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


if __name__ == "__main__":
    import os
    from datetime import datetime
    from pathlib import Path

    cli_args = sys.argv[:]

    if "--config_path" not in str(cli_args):
        print("\n[ERRO]: O argumento '--config_path' é obrigatório.")
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

    # Abre viewer de câmera em processo separado e registra para matar ao sair
    cam_host = "192.168.123.164" if not force_sim else "127.0.0.1"
    viewer_script = str(Path(__file__).resolve().parent.parent / "view_cam_live.py")
    _viewer_proc = subprocess.Popen(
        [sys.executable, viewer_script, "--host", cam_host, "--port", "5555"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    import atexit as _atexit
    _atexit.register(lambda: _viewer_proc.poll() is None and _viewer_proc.terminate())
    print(f"[CamViewer] 📷 Viewer iniciado (RGB + Depth em {cam_host}:5555)")

    try:
        # Garante conexão Wi-Fi ADB
        _conn = subprocess.run(["adb", "connect", QUEST_ADB], capture_output=True, text=True, timeout=5)
        print(f"[Quest] adb connect {QUEST_ADB}: {(_conn.stdout or _conn.stderr).strip()}")

        result = _adb("shell", "am", "broadcast", "-a", "com.oculus.vrpowermanager.prox_close")
        if "Broadcast completed" in result.stdout:
            print("[Quest] ✅ Sensor de proximidade desativado — pode pendurar no pescoço.")
        else:
            print(f"[Quest] ⚠️ NÃO desativou (adb={QUEST_ADB}). out={result.stdout.strip()!r} err={result.stderr.strip()!r}")
            print("[Quest]    Cheque: Wireless Debugging ON no Quest + autorizar este PC; IP/porta certos (adb tcpip 5555).")

        if quest_ip:
            vuer_url = f"https://{quest_ip}:8012/?grid=False&ws=wss://{quest_ip}:8012"
            # Fecha o browser completamente antes de abrir (limpa guias e experiência VR anterior)
            _adb("shell", "am force-stop com.oculus.browser")
            import time; time.sleep(1)
            # URL precisa de aspas simples no shell Android p/ o & não ser interpretado
            _adb("shell", f"am start -a android.intent.action.VIEW -d '{vuer_url}'")
            print(f"[Quest] 🌐 Browser aberto em {vuer_url}")
    except Exception:
        print("[Quest] ⚠️ ADB não disponível — sensor de proximidade não foi desativado.")

    import signal
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[SYSTEM]: Gravação finalizada pelo usuário.")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\n[ERRO DE EXECUÇÃO]:")
        traceback.print_exc()
        sys.exit(1)