"""
Teleoperador G1 por câmera RealSense D435i + MediaPipe Hands (sem Quest).

Substitui a fonte de pose do xr_g1_arm.py (headset/controle do Quest) por uma
D435i apontada PARA VOCÊ: o MediaPipe estima a pose da sua mão direita e o
braço DIREITO do G1 a segue. Tudo o que vem depois — IK do braço (G1_29_ArmIK),
clutch (movimento por delta, sem salto), mapeamento pros body_joints/hand_joints
e os eventos de gravação (pausa/salvar/descartar/encerrar) — é IDÊNTICO ao
xr_g1_arm.py, então o init_lerobot_record.py funciona sem mudança nenhuma.

Diferenças de design (decididas com o Luiz):
  • Só o braço DIREITO é teleoperado; o esquerdo fica numa pose neutra fixa.
  • Punho 6DOF: posição XYZ do punho vem da depth alinhada da D435i (métrica),
    a orientação 3DOF vem do plano da palma (punho / MCP do indicador / MCP do
    mínimo) usando os world-landmarks métricos do MediaPipe.
  • Dedos da Dex3 por TECLADO (mão livre): a mão fica neutra e você aperta
    'f' = squeeze (fecha tudo) e 'j' = pinça/trigger — mesma lógica de alvos do
    modo "controller" do xr_g1_arm. Os 4 sinais de grasp também vão pro dataset.
  • Comandos de gravação por TECLADO: espaço = pausa/play, s = salvar,
    d = descartar, q = encerrar (mesmos eventos que X/A/B/Y do Quest disparavam).

Teclas (lidas por pynput, GLOBAIS — funcionam mesmo sem a janela em foco; cuidado
com o robô real):
  espaço  destrava/trava o robô (igual ao X do Quest)
  f       (segura) fecha a mão direita (squeeze)         — solta = abre
  j       (segura) pinça fina da mão direita (trigger)   — solta = abre
  s       salva o episódio e começa o próximo (igual A)
  d       descarta e regrava o episódio (igual B)
  q       encerra a gravação (igual Y)

Envs úteis:
  G1_MP_RS_SERIAL     serial da D435i de mão (se houver mais de uma RealSense)
  G1_MP_HAND          'right' | 'left' | 'any' (default 'any') — qual mão seguir
  G1_MP_PREVIEW       '1' (default) abre janela cv2 com landmarks + HUD; '0' off
  G1_MP_ORIENT        'plane' (default) usa orientação do plano da palma;
                      'fixed' mantém a garra na orientação neutra (3DOF só posição)
  G1_MP_POS_GAIN      ganho da translação câmera→robô (default 1.0)
  G1_MP_FLIP_X/Y/Z    '1' inverte o respectivo eixo de translação (calibração)
"""

import os
import time
import logging
import threading
import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.processor import RobotAction

from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    """True só com valor "real" (evita o footgun de bool('0') == True)."""
    v = os.environ.get(name, "1" if default else "")
    return v not in ("", "0", "false", "False")


# ---------------------------------------------------------------------------
# Mapeamento de eixos câmera RealSense → base do robô.
# RealSense (frame óptico): +X direita, +Y baixo, +Z para a cena (longe da câmera).
# Base do robô (IK G1):     +X frente,  +Y esquerda, +Z cima.
# Como a câmera aponta PARA você, "afastar a mão da câmera" (=+Z cam) é o robô
# esticar o braço para a FRENTE (=+X robô). O clutch usa só o DELTA, então um
# eixo invertido é só "sensação espelhada", não perigo — calibre com G1_MP_FLIP_*.
# ---------------------------------------------------------------------------
_CAM2ROBOT = np.array([
    [0.0, 0.0, 1.0],   # robô X (frente)   =  cam Z
    [-1.0, 0.0, 0.0],  # robô Y (esquerda) = -cam X
    [0.0, -1.0, 0.0],  # robô Z (cima)     = -cam Y
], dtype=float)


@TeleoperatorConfig.register_subclass("realsense_mediapipe_arm")
@dataclass
class RealsenseMediapipeArmConfig(TeleoperatorConfig):
    is_simulation: bool = True
    ee_type: str = "dex3"
    rs_width: int = 640
    rs_height: int = 480
    rs_fps: int = 30


class RealsenseMediapipeArm(Teleoperator):
    config_class = RealsenseMediapipeArmConfig
    name = "realsense_mediapipe_arm"

    def __init__(self, config: RealsenseMediapipeArmConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False

        from robot.unitree_g1.g1_utils import (
            G1_29_JointIndex, LEFT_HAND_JOINT_NAMES, RIGHT_HAND_JOINT_NAMES,
        )
        self._left_hand_names = LEFT_HAND_JOINT_NAMES
        self._right_hand_names = RIGHT_HAND_JOINT_NAMES

        self.body_joints = {f"{motor.name}.q": 0.0 for motor in G1_29_JointIndex}
        self.hand_joints = {f"{name}.q": 0.0 for name in self._left_hand_names + self._right_hand_names}
        # Dims de grasp (NÃO são juntas; propagam o sinal pro dataset, igual xr_g1_arm).
        for _gk in ("left_grasp_squeeze.q", "right_grasp_squeeze.q",
                    "left_grasp_trigger.q", "right_grasp_trigger.q"):
            self.hand_joints[_gk] = 0.0

        # IK do braço.
        self.arm_ik = None
        self.current_arm_q = np.zeros(14)

        # ---- estado do teclado (preenchido pelo listener pynput) ----
        self._key_squeeze = False   # 'f' segurado
        self._key_pinch = False     # 'j' segurado
        self._kb_listener = None
        # EMA do squeeze/pinça (suaviza o 0→1 do teclado; mesmo motivo do Quest).
        self._sq_r = 0.0
        self._tr_r = 0.0
        self._EMA_ALPHA = 0.5

        # Sempre começa TRAVADO; espaço destrava (mesmo contrato do X do Quest).
        self.controller_enabled = False

        # ---- estado do CLUTCH (idêntico ao xr_g1_arm) ----
        self.clutch_anchored = False
        self.ctrl_ref_right = None      # pose da mão (câmera) no instante do destrave
        self.robot_ref_right = None     # pose do punho do robô no instante do destrave
        self.robot_ref_left = None      # esquerdo: pose neutra FIXA (nunca move)
        self.last_right_target = None
        self._last_clutch_time = None

        # ---- estado partilhado câmera/MediaPipe ----
        self._cam_stop = threading.Event()
        self._cam_thread = None
        self._pose_lock = threading.Lock()
        self._latest_hand_pose = None   # 4x4 da mão no frame do ROBÔ (pós-mapeamento)
        self._latest_seq = 0
        self._hand_detected = False
        self._rs_intrin = None

    # ------------------------------------------------------------------ #
    #  Conexão / desconexão                                              #
    # ------------------------------------------------------------------ #
    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return
        logger.info("Carregando URDF e IK do braço G1_29...")
        self.arm_ik = G1_29_ArmIK()
        # Pose neutra do braço esquerdo = FK da config zero; o IK a mantém fixa.
        self.robot_ref_left, _ = self.arm_ik.forward_kinematics(np.zeros(14))

        self._start_keyboard_listener()
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True, name="d435i-mediapipe")
        self._cam_thread.start()

        self._is_connected = True
        logger.info("Teleop RealSense+MediaPipe conectado. Espaço destrava; f/j = grasp.")
        print("\n   🖐️  [MEDIAPIPE] Teleop por câmera ativo. ESPAÇO destrava o robô; "
              "f=fecha, j=pinça, s=salva, d=descarta, q=encerra.\n", flush=True)

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        self._is_connected = False
        self._cam_stop.set()
        if self._cam_thread is not None and self._cam_thread.is_alive():
            self._cam_thread.join(timeout=1.5)
        if self._kb_listener is not None:
            with contextlib.suppress(Exception):
                self._kb_listener.stop()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @property
    def action_features(self) -> dict:
        features = {}
        for key in self.body_joints.keys():
            features[key] = float
        for key in self.hand_joints.keys():
            features[key] = float
        return features

    @property
    def feedback_features(self) -> dict:
        return {"q": np.ndarray}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if "q" in feedback:
            self.current_arm_q = feedback["q"][:14]

    # ------------------------------------------------------------------ #
    #  Teclado (pynput) — destrava/grasp/eventos de gravação             #
    # ------------------------------------------------------------------ #
    def _start_keyboard_listener(self):
        try:
            from pynput import keyboard
        except Exception as e:
            logger.error("pynput indisponível (%s) — teclado desativado. pip install pynput", e)
            return

        def on_press(key):
            try:
                k = key.char.lower() if hasattr(key, "char") and key.char else None
            except Exception:
                k = None
            if key == keyboard.Key.space:
                self._trigger_record_event("toggle_pause")
            elif k == "f":
                self._key_squeeze = True
            elif k == "j":
                self._key_pinch = True
            elif k == "s":
                self._trigger_record_event("save")
            elif k == "d":
                self._trigger_record_event("discard")
            elif k == "q":
                self._trigger_record_event("exit")

        def on_release(key):
            try:
                k = key.char.lower() if hasattr(key, "char") and key.char else None
            except Exception:
                k = None
            if k == "f":
                self._key_squeeze = False
            elif k == "j":
                self._key_pinch = False

        self._kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._kb_listener.daemon = True
        self._kb_listener.start()

    def _trigger_record_event(self, action_type):
        """Mesmos eventos do xr_g1_arm: injeta no script de gravação via __main__."""
        import sys
        if "__main__" not in sys.modules:
            return
        main_mod = sys.modules["__main__"]
        events = getattr(main_mod, "global_events", None)

        if action_type == "save" and events is not None:
            print("\n   🖐️  [MEDIAPIPE] Ação: SALVANDO e gravando o próximo... ✅")
            events["exit_early"] = True
            self.clutch_anchored = False

        elif action_type == "discard" and events is not None:
            print("\n   🖐️  [MEDIAPIPE] Ação: DESCARTANDO e regravando... ❌")
            events["rerecord_episode"] = True
            events["exit_early"] = True
            self.clutch_anchored = False
            try:
                import json as _json, time as _time
                with open("/tmp/g1_record_status.json") as _f:
                    _st = _json.load(_f)
                _st["start_time"] = _time.time()
                with open("/tmp/g1_record_status.json", "w") as _f:
                    _json.dump(_st, _f)
            except Exception:
                pass

        elif action_type == "toggle_pause":
            self.controller_enabled = not self.controller_enabled
            self.clutch_anchored = False
            if self.controller_enabled:
                if hasattr(main_mod, "robot_paused"):
                    main_mod.robot_paused = False
                print("\n   🖐️  [MEDIAPIPE] Robô DESTRAVADO ▶️ (mova a mão direita)")
            else:
                if hasattr(main_mod, "robot_paused"):
                    main_mod.robot_paused = True
                print("\n   🖐️  [MEDIAPIPE] Robô CONGELADO 🧊")

        elif action_type == "exit":
            print("\n   🖐️  [MEDIAPIPE] ENCERRANDO o sistema... 🛑")
            if events is not None:
                events["stop_recording"] = True
                events["exit_early"] = True
            else:
                self.disconnect()
                os._exit(0)

    # ------------------------------------------------------------------ #
    #  Thread da câmera + MediaPipe                                      #
    # ------------------------------------------------------------------ #
    def _camera_loop(self):
        try:
            import pyrealsense2 as rs
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            import cv2
        except Exception as e:
            logger.error("Falha importando pyrealsense2/mediapipe/cv2: %s", e)
            print(f"\n   ❌ [MEDIAPIPE] Não consegui iniciar (import): {e}", flush=True)
            return

        # Modelo da Tasks API (HandLandmarker). O mediapipe novo (0.10.35) não tem a
        # API legada mp.solutions.hands; a Tasks API dá landmarks normalizados E world
        # landmarks métricos (que usamos pra orientação da palma).
        from pathlib import Path as _P
        default_model = _P(__file__).resolve().parents[2] / "assets" / "mediapipe" / "hand_landmarker.task"
        model_path = os.environ.get("G1_MP_MODEL", str(default_model))
        if not os.path.exists(model_path):
            logger.error("Modelo HandLandmarker não encontrado: %s", model_path)
            print(f"\n   ❌ [MEDIAPIPE] Modelo não encontrado: {model_path}\n"
                  "   Baixe: curl -fsSL -o assets/mediapipe/hand_landmarker.task "
                  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task", flush=True)
            return

        w, h, fps = self.config.rs_width, self.config.rs_height, self.config.rs_fps
        pipeline = rs.pipeline()
        cfg = rs.config()
        serial = os.environ.get("G1_MP_RS_SERIAL", "")
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        try:
            profile = pipeline.start(cfg)
        except Exception as e:
            logger.error("Não abri a D435i de mão: %s", e)
            print(f"\n   ❌ [MEDIAPIPE] D435i de mão não abriu: {e}", flush=True)
            return

        align = rs.align(rs.stream.color)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        color_intrin = (profile.get_stream(rs.stream.color)
                        .as_video_stream_profile().get_intrinsics())

        hand_pref = os.environ.get("G1_MP_HAND", "any").lower()
        preview = _env_flag("G1_MP_PREVIEW", default=True)
        orient_mode = os.environ.get("G1_MP_ORIENT", "plane").lower()
        pos_gain = float(os.environ.get("G1_MP_POS_GAIN", "1.0"))
        flip = np.array([
            -1.0 if _env_flag("G1_MP_FLIP_X") else 1.0,
            -1.0 if _env_flag("G1_MP_FLIP_Y") else 1.0,
            -1.0 if _env_flag("G1_MP_FLIP_Z") else 1.0,
        ])

        # IMAGE mode (detecção completa por frame). O VIDEO mode (detect_for_video)
        # com timestamps sintéticos não rastreava e devolvia 0 mãos — verificado na
        # bancada (IMAGE detecta em ~todo frame; VIDEO em nenhum). A 30 FPS o custo
        # extra do IMAGE é irrelevante nesta GPU.
        landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=model_path),
                num_hands=2,
                min_hand_detection_confidence=float(os.environ.get("G1_MP_MIN_CONF", "0.4")),
                running_mode=vision.RunningMode.IMAGE,
            )
        )
        print(f"\n   🖐️  [MEDIAPIPE] D435i de mão OK ({w}x{h}@{fps}, depth_scale={depth_scale:.5f}). "
              f"Mão='{hand_pref}', orient='{orient_mode}'.\n", flush=True)

        while not self._cam_stop.is_set():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except Exception:
                continue
            frames = align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue
            bgr = np.asanyarray(color.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_image)

            chosen = self._select_hand(res, hand_pref)
            detected = chosen is not None
            if detected:
                lm, world_lm = chosen
                pose = self._hand_to_robot_pose(
                    lm, world_lm, depth, color_intrin, depth_scale,
                    rs, orient_mode, pos_gain, flip, w, h,
                )
                if pose is not None:
                    with self._pose_lock:
                        self._latest_hand_pose = pose
                        self._latest_seq += 1
                        self._hand_detected = True
                else:
                    detected = False
            if not detected:
                with self._pose_lock:
                    self._hand_detected = False

            if preview:
                self._draw_preview(cv2, bgr, res, detected, w, h)

        with contextlib.suppress(Exception):
            landmarker.close()
        with contextlib.suppress(Exception):
            pipeline.stop()
        if preview:
            with contextlib.suppress(Exception):
                cv2.destroyWindow("D435i + MediaPipe (teleop G1)")

    def _select_hand(self, res, hand_pref):
        """Escolhe uma mão dos resultados do HandLandmarker. Retorna (landmarks, world_landmarks).

        Na Tasks API cada mão é uma LISTA plana de landmarks (lm[0]..lm[20]); os
        rótulos vêm em res.handedness[i][0].category_name ('Left'/'Right', espelhado).
        """
        if not res.hand_landmarks:
            return None
        labels = []
        if res.handedness:
            for h in res.handedness:
                labels.append(h[0].category_name.lower())  # 'left'/'right' (espelhado!)
        idx = 0
        if hand_pref in ("left", "right") and labels:
            # OBS: rotulado como espelho; o usuário escolhe por tentativa (G1_MP_HAND).
            for i, lab in enumerate(labels):
                if lab == hand_pref:
                    idx = i
                    break
        lm = res.hand_landmarks[idx]
        world = (res.hand_world_landmarks[idx]
                 if res.hand_world_landmarks else None)
        return lm, world

    def _hand_to_robot_pose(self, lm, world_lm, depth_frame, intrin, depth_scale,
                            rs, orient_mode, pos_gain, flip, w, h):
        """Constrói a pose 4x4 da mão no frame do robô.

        Posição: pixel do punho (landmark 0) → depth alinhada → deprojeção métrica
        no frame da câmera → mapeada pro frame do robô (_CAM2ROBOT).
        Orientação: plano da palma a partir dos world-landmarks métricos do MediaPipe
        (punho 0, MCP indicador 5, MCP mínimo 17), também mapeado pro frame do robô.
        """
        wrist = lm[0]
        px = int(np.clip(wrist.x * w, 0, w - 1))
        py = int(np.clip(wrist.y * h, 0, h - 1))
        # Mediana de uma janela 5x5 ao redor do punho → robusto a buracos da depth.
        x0, x1 = max(0, px - 2), min(w, px + 3)
        y0, y1 = max(0, py - 2), min(h, py + 3)
        patch = np.asanyarray(depth_frame.get_data())[y0:y1, x0:x1].astype(np.float32)
        patch = patch[patch > 0]
        if patch.size == 0:
            return None
        z_m = float(np.median(patch)) * depth_scale
        if not (0.15 < z_m < 1.5):   # fora de alcance plausível da mão → ignora
            return None
        cam_xyz = np.array(rs.rs2_deproject_pixel_to_point(intrin, [px, py], z_m), dtype=float)
        robot_pos = (_CAM2ROBOT @ cam_xyz) * pos_gain * flip

        pose = np.eye(4)
        pose[:3, 3] = robot_pos

        if orient_mode != "fixed" and world_lm is not None:
            R = self._palm_rotation(world_lm)
            if R is not None:
                pose[:3, :3] = _CAM2ROBOT @ R
        return pose

    @staticmethod
    def _palm_rotation(world_lm):
        """Frame ortonormal da palma a partir dos world-landmarks (métricos, hand-relative)."""
        def v(i):
            p = world_lm[i]
            return np.array([p.x, p.y, p.z], dtype=float)
        wrist = v(0)
        idx_mcp = v(5)
        pinky_mcp = v(17)
        x_axis = idx_mcp - wrist            # ao longo da palma (rumo aos dedos)
        side = pinky_mcp - wrist
        nx = np.linalg.norm(x_axis)
        if nx < 1e-6:
            return None
        x_axis /= nx
        z_axis = np.cross(x_axis, side)     # normal da palma
        nz = np.linalg.norm(z_axis)
        if nz < 1e-6:
            return None
        z_axis /= nz
        y_axis = np.cross(z_axis, x_axis)
        return np.column_stack([x_axis, y_axis, z_axis])

    def _draw_preview(self, cv2, bgr, res, detected, w, h):
        try:
            if res.hand_landmarks:
                for hlm in res.hand_landmarks:
                    for p in hlm:
                        cx, cy = int(p.x * w), int(p.y * h)
                        cv2.circle(bgr, (cx, cy), 3, (0, 255, 0), -1)
            phase = ("DESTRAVADO ▶" if self.controller_enabled else "TRAVADO ⏸")
            grasp = []
            if self._key_squeeze:
                grasp.append("FECHA(f)")
            if self._key_pinch:
                grasp.append("PINCA(j)")
            hud = f"{phase} | mao:{'OK' if detected else '--'} | {' '.join(grasp) or 'mao neutra'}"
            cv2.putText(bgr, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if self.controller_enabled else (0, 200, 255), 2)
            cv2.putText(bgr, "ESPACO=destrava  f=fecha  j=pinca  s=salva  d=descarta  q=sai",
                        (10, bgr.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("D435i + MediaPipe (teleop G1)", bgr)
            cv2.waitKey(1)
        except Exception:
            pass  # GUI best-effort: nunca derruba a teleop

    # ------------------------------------------------------------------ #
    #  get_action — chamado pelo loop de gravação (~30 Hz)               #
    # ------------------------------------------------------------------ #
    def get_action(self) -> RobotAction:
        if not self._is_connected:
            raise ConnectionError("RealsenseMediapipeArm não está conectado.")

        # 🚨 BLOQUEIO: travado → devolve a pose atual (mãos/braço PARADOS — os dedos
        # também não respondem ao teclado enquanto travado, igual ao xr_g1_arm).
        if not self.controller_enabled:
            return {**self.body_joints, **self.hand_joints}

        # Destravado: dedos seguem o teclado (f=squeeze, j=pinça).
        self._update_fingers()

        with self._pose_lock:
            hand_pose = None if self._latest_hand_pose is None else self._latest_hand_pose.copy()
            detected = self._hand_detected

        # Sem mão detectada: mantém a última pose comandada (não congela o clutch).
        if not detected or hand_pose is None:
            return {**self.body_joints, **self.hand_joints}

        # ---- CLUTCH (idêntico ao xr_g1_arm): move por DELTA, sem salto ----
        now = time.time()
        if (self.clutch_anchored and self._last_clutch_time is not None
                and (now - self._last_clutch_time) > 0.5):
            self.clutch_anchored = False  # voltou de um freeze (encode) → re-ancora
        self._last_clutch_time = now

        if not self.clutch_anchored:
            self.ctrl_ref_right = hand_pose.copy()
            if self.last_right_target is not None:
                self.robot_ref_right = self.last_right_target.copy()
            else:
                _, self.robot_ref_right = self.arm_ik.forward_kinematics(self.current_arm_q)
            self.clutch_anchored = True
            logger.info(">>> CLUTCH ancorado: mova a mão direita para mover o braço.")

        right_target = self.robot_ref_right.copy()
        right_target[:3, 3] = self.robot_ref_right[:3, 3] + (hand_pose[:3, 3] - self.ctrl_ref_right[:3, 3])
        right_target[:3, :3] = (hand_pose[:3, :3] @ self.ctrl_ref_right[:3, :3].T) @ self.robot_ref_right[:3, :3]
        self.last_right_target = right_target.copy()

        # Esquerdo fixo na pose neutra; direito segue a mão. Seed=None → warm-start interno.
        sol_q, _ = self.arm_ik.solve_ik(self.robot_ref_left, right_target, None, None)

        # Esquerdo (0-6): fica ~constante (alvo fixo); mantém o braço parado/neutro.
        self.body_joints["kLeftShoulderPitch.q"] = sol_q[0]
        self.body_joints["kLeftShoulderRoll.q"] = sol_q[1]
        self.body_joints["kLeftShoulderYaw.q"] = sol_q[2]
        self.body_joints["kLeftElbow.q"] = sol_q[3]
        self.body_joints["kLeftWristRoll.q"] = sol_q[4]
        self.body_joints["kLeftWristPitch.q"] = sol_q[5]
        self.body_joints["kLeftWristYaw.q"] = sol_q[6]
        # Direito (7-13): segue a mão.
        self.body_joints["kRightShoulderPitch.q"] = sol_q[7]
        self.body_joints["kRightShoulderRoll.q"] = sol_q[8]
        self.body_joints["kRightShoulderYaw.q"] = sol_q[9]
        self.body_joints["kRightElbow.q"] = sol_q[10]
        self.body_joints["kRightWristRoll.q"] = sol_q[11]
        self.body_joints["kRightWristPitch.q"] = sol_q[12]
        self.body_joints["kRightWristYaw.q"] = sol_q[13]

        return {**self.body_joints, **self.hand_joints}

    def _update_fingers(self):
        """Mão DIREITA por teclado: f=squeeze (fecha tudo), j=pinça (trigger).

        Reaproveita exatamente os alvos do modo 'controller' do xr_g1_arm: squeeze
        × RIGHT_TARGET para o fechamento total + ajuste fino de pinça no trigger.
        A esquerda fica neutra (q=0). EMA suaviza o degrau 0→1 do teclado.
        """
        raw_sq = 1.0 if self._key_squeeze else 0.0
        raw_tr = 1.0 if self._key_pinch else 0.0
        a = self._EMA_ALPHA
        self._sq_r = a * raw_sq + (1 - a) * self._sq_r
        self._tr_r = a * raw_tr + (1 - a) * self._tr_r
        right_squeeze = self._sq_r
        right_trigger = 0.0 if self._tr_r < 0.05 else self._tr_r

        # Registra os sinais de grasp como dims do dataset (igual xr_g1_arm).
        self.hand_joints["right_grasp_squeeze.q"] = float(right_squeeze)
        self.hand_joints["right_grasp_trigger.q"] = float(right_trigger)
        self.hand_joints["left_grasp_squeeze.q"] = 0.0
        self.hand_joints["left_grasp_trigger.q"] = 0.0

        # Alvo de fechamento total = limites do Dex3 direito (ordem thumb,index,middle).
        RIGHT_TARGET = np.array([0.0, -0.920, -1.74, 1.57, 1.74, 1.57, 1.74])
        right_hand_q = right_squeeze * RIGHT_TARGET

        # Pinça fina (mesmos offsets do controller mode do xr_g1_arm).
        PINCH_FORCE = 2.0
        PINCH_OFFSET = 0.2
        PINCH_OFFSET2 = 0.1
        right_hand_q[5] += PINCH_FORCE * right_trigger
        right_hand_q[5] += PINCH_OFFSET * right_trigger
        right_hand_q[6] += PINCH_FORCE * right_trigger
        right_hand_q[6] += PINCH_OFFSET2 * right_trigger
        right_hand_q[0] += -0.5 * right_trigger
        right_hand_q[1] -= 0.8 * right_trigger
        right_hand_q[2] -= 0.8 * right_trigger

        for i, name in enumerate(self._right_hand_names):
            self.hand_joints[f"{name}.q"] = right_hand_q[i]
        # Mão esquerda neutra (aberta).
        for name in self._left_hand_names:
            self.hand_joints[f"{name}.q"] = 0.0
