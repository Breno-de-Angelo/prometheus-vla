#!/usr/bin/env python
"""
Unitree G1 Dex3 Robot - Unified DDS Architecture

This implementation uses the same DDS threading pattern as the body/arm code,
eliminating the multiprocessing complexity of the original Dex3_1_Controller.
"""

from dataclasses import dataclass, field
import logging
import threading
import time
import numpy as np
from functools import cached_property
import os
import sys
import subprocess

# Isso em baixo ele vai usar a class nova Loco para controla o robo com High ou Low Level.
from .unitree_g1 import UnitreeG1, UnitreeG1Config, left_arm_limp_enabled
from lerobot.robots.config import RobotConfig
from .g1_utils import (
    Dex3_1_Left_JointIndex, 
    Dex3_1_Right_JointIndex,
    Dex3_1_Left_PressureTemperatureSensors,
    Dex3_1_Right_PressureTemperatureSensors,
    sensor_index,
    Dex3_Num_Motors,
    DEX3_LEFT_LOWER_LIMITS,
    DEX3_LEFT_UPPER_LIMITS,
    DEX3_RIGHT_LOWER_LIMITS,
    DEX3_RIGHT_UPPER_LIMITS,
    kTopicDex3LeftCommand,
    kTopicDex3RightCommand,
    kTopicDex3LeftState,
    kTopicDex3RightState,
    LEFT_HAND_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
)

from lerobot.processor import RobotAction, RobotObservation

logger = logging.getLogger(__name__)



@dataclass
class HandMotorState:
    """State of a single hand motor."""
    q: float = 0.0  # position


@dataclass
class HandState:
    """State of a single hand (7 motors for Dex3-1)."""
    motor_state: list[HandMotorState] = field(
        default_factory=lambda: [HandMotorState() for _ in range(Dex3_Num_Motors)]
    )
    # Adicionando 27 sensores de pressão e 27 de temperatura por mão
    pressure: np.ndarray = field(default_factory=lambda: np.zeros(108, dtype=np.float32))
    temperature: np.ndarray = field(default_factory=lambda: np.zeros(108, dtype=np.float32))


@RobotConfig.register_subclass("unitree_g1_dex3")
@dataclass
class UnitreeG1Dex3Config(UnitreeG1Config):
    """Configuration for Unitree G1 with Dex3-1 hands."""
    hand_kp: float = 0.8  # Position gain for hand motors
    hand_kd: float = 0.2  # Damping gain for hand motors
    hand_control_dt: float = 0.005  # 100 Hz control loop

    use_loco: bool = False  # False = Low Level (Suporte), True = High Level (Andando)
    
    def __post_init__(self):

        if self.use_loco:
            self.control_mode = "high_level"
        else:
            self.control_mode = "upper_body"
        

        # LÓGICA DE RESOLUÇÃO DINÂMICA
        if self.is_simulation:
            if getattr(self, "sim_backend", "mujoco") != "isaac":
                self.robot_ip = "127.0.0.1"   # mujoco: sim local. isaac: mantem robot_ip do config (ex.: lcad232)
            # Simulação: Leve e rápida para não gargalar a GPU/CPU
            cam2_width = 640
            cam2_height = 480
        else:
            # Hardware Real: D435i em 848x480 (16:9) — FOV horizontal completo (4:3 cortava
            # as laterais) + resolução nativa/ótima do depth da D435i.
            cam2_width = 848
            cam2_height = 480

            
        # Adiciona as câmeras ZMQ ao LeRobot usando as variáveis dinâmicas
        if not self.cameras:
            from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig
            
            self.cameras = {
                # A NOSSA ÚNICA CÂMERA RGB (HD para o VR e para a IA)
                "head_camera": ZMQCameraConfig(
                    server_address=self.robot_ip, port=5555, camera_name="head_camera", width=cam2_width, height=cam2_height
                ),
                
                # DEPTH temporariamente desligado para isolar lag durante gravação.
                # Reative descomentando o bloco abaixo.
                # "head_camera_depth": ZMQCameraConfig(
                #     server_address=self.robot_ip, port=5555, camera_name="head_camera_depth", width=cam2_width, height=cam2_height
                # ),
                #"d435i_ir_left": ZMQCameraConfig(
                #    server_address=self.robot_ip, port=5555, camera_name="d435i_ir_left", width=cam_width, height=cam_height
                #),
                #"d435i_ir_right": ZMQCameraConfig(
                #    server_address=self.robot_ip, port=5555, camera_name="d435i_ir_right", width=cam_width, height=cam_height
                #)
            }


class UnitreeG1Dex3(UnitreeG1):
    """
    Unitree G1 Robot with Dex3-1 Dexterous Hands.
    
    Uses the same DDS threading architecture as the body motors:
    - A background thread subscribes to hand state at 100Hz
    - Hand commands are published directly via DDS
    """
    config_class = UnitreeG1Dex3Config
    name = "unitree_g1_dex3"

    def __init__(self, config: UnitreeG1Dex3Config):
        super().__init__(config)
        
        # Hand state (similar to _lowstate for body)
        self._left_hand_state: HandState | None = None
        self._right_hand_state: HandState | None = None

        
        # Threading control
        self._hand_shutdown_event = threading.Event()
        self._hand_subscribe_thread: threading.Thread | None = None
        
        # DDS publishers/subscribers (initialized in connect)
        self._left_hand_cmd_pub = None
        self._right_hand_cmd_pub = None
        self._left_hand_state_sub = None
        self._right_hand_state_sub = None
        
        # Command messages (initialized in connect)
        self._left_hand_msg = None
        self._right_hand_msg = None

        self._last_action_time = time.time()

        # Buffer da profundidade
        self.latest_depth = np.zeros(768, dtype=np.float32)
        
        # Use joint name constants from g1_utils
        self.left_hand_joint_names = LEFT_HAND_JOINT_NAMES
        self.right_hand_joint_names = RIGHT_HAND_JOINT_NAMES

        # Anti-tremor das mãos: estado do filtro EMA + clamp de velocidade por dedo.
        # O braço (unitree_g1.send_action) já suaviza com smoothing_alpha + max_delta;
        # os dedos Dex3 iam DIRETO do retarget/IK pro DDS (só np.clip nos limites),
        # sem passa-baixa → tremor. Aqui guardamos o último q suavizado por mão.
        self._last_hand_q_left = None
        self._last_hand_q_right = None

        # 🤚 HAND STREAMER (jeito Unitree, fps=100): publica os dedos a 100Hz numa thread
        # dedicada com clip de velocidade contra o COMANDO ANTERIOR (igual ao arm streamer
        # de 250Hz). Desacopla a publicação do record_loop de 30Hz → sem stair-stepping.
        # NOTA: o clip dá SUAVIDADE (rampa), NÃO é o anti-trava — o comando ainda converge
        # pro alvo. Quem evita a sobrecorrente é o kp mola (0.3) do grasp em xr_g1_arm.py.
        # Desligável via G1_HAND_STREAMER=0 → volta ao modo legado (send_action 30Hz + EMA).
        self._hand_target_left: np.ndarray | None = None   # alvo dos 7 dedos (setado por send_action)
        self._hand_target_right: np.ndarray | None = None
        self._hand_lock = threading.Lock()
        self._hand_streamer_stop = threading.Event()
        self._hand_streamer_on = os.environ.get("G1_HAND_STREAMER", "1") not in ("", "0", "false", "False")
        self._hand_streamer_thread: threading.Thread | None = None

    def reset_hands(self, default_positions: list[float] | None = None):
        """Move as mãos para a posição inicial (padrão: totalmente abertas)."""
        if default_positions is None:
            # Define posição aberta para cada junta da mão (valores típicos para Dex3-1)
            # Ajuste conforme a pose usada na coleta de dados
            default_left = np.zeros(Dex3_Num_Motors)
            default_right = np.zeros(Dex3_Num_Motors)
            # Exemplo: abrir completamente (cada junta tem limites diferentes, use 0.0 como referência)
            # Se seu dataset usa valores específicos, carregue de config.default_hand_positions
        else:
            default_left = default_positions[:Dex3_Num_Motors]
            default_right = default_positions[Dex3_Num_Motors:]

        # Aplica suavemente (interpolação linear) para evitar saltos
        total_time = 2.0
        dt = self.config.hand_control_dt
        steps = int(total_time / dt)

        # Obtém posições atuais
        left_current = np.zeros(Dex3_Num_Motors)
        right_current = np.zeros(Dex3_Num_Motors)
        if self._left_hand_state:
            left_current = [s.q for s in self._left_hand_state.motor_state]
        if self._right_hand_state:
            right_current = [s.q for s in self._right_hand_state.motor_state]

        for step in range(steps):
            alpha = step / steps
            left_q = left_current * (1 - alpha) + default_left * alpha
            right_q = right_current * (1 - alpha) + default_right * alpha

            # Prepara ação das mãos
            action = {}
            for i, name in enumerate(self.left_hand_joint_names):
                action[f"{name}.q"] = float(left_q[i])
            for i, name in enumerate(self.right_hand_joint_names):
                action[f"{name}.q"] = float(right_q[i])

            self.send_action(action)  # envia apenas as mãos (as pernas já estão em limp)
            time.sleep(dt)

        logger.info("Mãos resetadas para a posição inicial.")

    def _subscribe_hand_state(self):
        import json
        import zmq
        """
        Background thread that polls hand state via DDS at ~100Hz.
        Similar to _subscribe_motor_state() in UnitreeG1.
        """
        while not self._hand_shutdown_event.is_set():
            start_time = time.time()
            
            # --- 1. LÊ OS MOTORES (VIA SDK NORMAL) ---
            left_msg = self._left_hand_state_sub.Read()
            if left_msg is not None:
                # SALVA a pressão antiga para o HandState() novo não zerar tudo!
                old_p = self._left_hand_state.pressure if self._left_hand_state else np.zeros(108, dtype=np.float32)
                
                left_state = HandState()
                for idx, joint_id in enumerate(Dex3_1_Left_JointIndex):
                    left_state.motor_state[idx].q = left_msg.motor_state[joint_id].q
                
                left_state.pressure = old_p # Restaura a pressão
                self._left_hand_state = left_state
            
            right_msg = self._right_hand_state_sub.Read()
            if right_msg is not None:
                old_p = self._right_hand_state.pressure if self._right_hand_state else np.zeros(108, dtype=np.float32)
                
                right_state = HandState()
                for idx, joint_id in enumerate(Dex3_1_Right_JointIndex):
                    right_state.motor_state[idx].q = right_msg.motor_state[joint_id].q
                
                right_state.pressure = old_p # Restaura a pressão
                self._right_hand_state = right_state

            # --- 2. LÊ A PRESSÃO (VIA ZMQ PURO - O QUE FUNCIONOU NO TESTE) ---
            if hasattr(self, '_pressure_socket') and self._pressure_socket:
                try:
                    while True: # Drena todas as mensagens acumuladas
                        payload = self._pressure_socket.recv(zmq.NOBLOCK)
                        msg_json = json.loads(payload.decode("utf-8"))
                        
                        data = msg_json.get("data", {})
                        side = data.get("side", "")
                        sensors = data.get("press_sensor_state", [])
                        
                        if sensors:
                            # Achata a lista como o LeRobot exige
                            flat_p = []
                            for area in sensors:
                                flat_p.extend([float(x) for x in area.get("pressure", [])])
                            while len(flat_p) < 108:
                                flat_p.append(0.0)
                            
                            # Atualiza direto no estado atual
                            if side == "left" and self._left_hand_state:
                                self._left_hand_state.pressure = np.array(flat_p[:108], dtype=np.float32)
                            elif side == "right" and self._right_hand_state:
                                self._right_hand_state.pressure = np.array(flat_p[:108], dtype=np.float32)
                except zmq.Again:
                    pass # Fila de pressão lida completamente
            
            # Maintain control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.config.hand_control_dt - elapsed)
            time.sleep(sleep_time)

    def connect(self, calibrate: bool = True) -> None:
        """Connect to robot body and hands."""
        # Connect body first
        super().connect(calibrate=calibrate)   
        
        # Skip hand connection in simulation mode
        if self.config.is_simulation:
            logger.info("Simulation mode: Iniciando a ponte da mão")

            # --- ADICIONE/AJUSTE ESTA PARTE ---
            print(f"[UnitreeG1] Forçando conexão ZMQ com a ponte em {self.config.robot_ip}")
            from .unitree_sdk2_socket import ChannelFactoryInitialize

            # Isso 'liga' os sockets internos do LeRobot
            ChannelFactoryInitialize(0, self.config.robot_ip) 
            try:
                # Descobre o caminho absoluto do diretório onde este arquivo está
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # Monta o caminho exato para a pasta robot_control/ponte_mao.py
                ponte_path = os.path.join(current_dir, "robot_control", "ponte_mao.py")
                
                if getattr(self.config, "sim_backend", "mujoco") == "isaac":
                    # isaac: a ponte externa (no lcad232) ja serve as portas; nao lanca a ponte_mao do MuJoCo
                    logger.info("[UnitreeG1] sim_backend=isaac -> usando ponte Isaac externa (sem ponte_mao do MuJoCo)")
                else:
                    logger.info(f"[UnitreeG1] Iniciando ponte ZMQ-DDS em segundo plano: {ponte_path}")
                    # Inicia o processo com o mesmo executável Python que o LeRobot está usando
                    self._ponte_process = subprocess.Popen([sys.executable, ponte_path])
                    logger.info(f"[UnitreeG1] Ponte iniciada com PID: {self._ponte_process.pid}")
                
                # Aguarda 1 segundinho para dar tempo da ponte abrir as portas ZMQ antes do código avançar
                time.sleep(1.0)
            except Exception as e:
                logger.error(f"[UnitreeG1] Falha ao iniciar a ponte automaticamente: {e}") 
        
        # Use ZMQ-based hand communication (SDK-free, no circular import issues)
        # The server on the robot bridges hand DDS topics to ZMQ sockets
        from .unitree_sdk2_socket import (
            ChannelSubscriber,
            ChannelPublisher,
            HandCmdMsg,
            kTopicDex3LeftCommand,
            kTopicDex3RightCommand,
            kTopicDex3LeftState,
            kTopicDex3RightState,
        )
        
        # Initialize hand state subscribers (ZMQ-based)
        self._left_hand_state_sub = ChannelSubscriber(kTopicDex3LeftState, None)
        self._left_hand_state_sub.Init()
        self._right_hand_state_sub = ChannelSubscriber(kTopicDex3RightState, None)
        self._right_hand_state_sub.Init()
        
        # Initialize hand command publishers (ZMQ-based)
        self._left_hand_cmd_pub = ChannelPublisher(kTopicDex3LeftCommand, None)
        self._left_hand_cmd_pub.Init()
        self._right_hand_cmd_pub = ChannelPublisher(kTopicDex3RightCommand, None)
        self._right_hand_cmd_pub.Init()
        
        # Initialize command messages with default gains
        self._left_hand_msg = HandCmdMsg()
        self._right_hand_msg = HandCmdMsg()
        
        kp = self.config.hand_kp
        kd = self.config.hand_kd
        
        for joint_id in Dex3_1_Left_JointIndex:
            # Mode byte: bits 0-3 = id, bits 4-6 = status (0x01 = enabled), bit 7 = timeout
            mode = (joint_id & 0x0F) | (0x01 << 4)
            self._left_hand_msg.motor_cmd[joint_id].mode = mode
            self._left_hand_msg.motor_cmd[joint_id].kp = kp
            self._left_hand_msg.motor_cmd[joint_id].kd = kd
            
        for joint_id in Dex3_1_Right_JointIndex:
            mode = (joint_id & 0x0F) | (0x01 << 4)
            self._right_hand_msg.motor_cmd[joint_id].mode = mode
            self._right_hand_msg.motor_cmd[joint_id].kp = kp
            self._right_hand_msg.motor_cmd[joint_id].kd = kd

        # ==========================================================
        # 💉 INJEÇÃO ZMQ PURA: AQUI É O LUGAR CORRETO!
        # ==========================================================
        import zmq
        self._pure_zmq_ctx = zmq.Context.instance()
        self._pressure_socket = self._pure_zmq_ctx.socket(zmq.SUB)
        self._pressure_socket.connect(f"tcp://{self.config.robot_ip}:6002")
        self._pressure_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        logger.info("📡 [ZMQ Puro] Pescador de Pressão conectado na porta 6002!")
        # ==========================================================
        
        # Start hand state subscription thread (Isto já está aí)
        self._hand_subscribe_thread = threading.Thread(
            target=self._subscribe_hand_state, 
            daemon=True,
            name="Dex3HandStateSubscriber"
        )
        self._hand_subscribe_thread.start()


        self._last_obs = {}
        
        # Wait for first hand state
        timeout = 3.0
        start = time.time()
        while self._left_hand_state is None or self._right_hand_state is None:
            if time.time() - start > timeout:
                logger.warning("Timeout waiting for Dex3 hand state. Hands may not be connected.")
                break
            time.sleep(0.01)
        
        if self._left_hand_state is not None and self._right_hand_state is not None:
            logger.info("Connected to Dex3 Hands via ZMQ.")
        else:
            logger.warning("Dex3 Hands not fully connected - hand state unavailable.")

        # Mão ESQUERDA: comportamento depende de --left-arm-limp (env G1_LEFT_ARM_LIMP).
        #   • SOLTA (opt-in): motor ativo mas kp=kd=tau=0 → zero torque = mole.
        #   • ATIVA (default): mantém kp/kd do connect e semeia q da posição física (igual direita).
        self._left_limp = left_arm_limp_enabled()
        if self._left_limp:
            # mode=0 no Dex3 pode ser "indefinido"; manter mode ativo sem ganhos é mais seguro.
            # Publica várias vezes para garantir que o servidor ZMQ/DDS receba (CONFLATE pode descartar).
            for joint_id in Dex3_1_Left_JointIndex:
                mode = (joint_id & 0x0F) | (0x01 << 4)
                self._left_hand_msg.motor_cmd[joint_id].mode = mode
                self._left_hand_msg.motor_cmd[joint_id].kp = 0.0
                self._left_hand_msg.motor_cmd[joint_id].kd = 0.0
                self._left_hand_msg.motor_cmd[joint_id].q = 0.0
                self._left_hand_msg.motor_cmd[joint_id].tau = 0.0
            for _ in range(5):  # repete para passar pelo CONFLATE
                self._left_hand_cmd_pub.Write(self._left_hand_msg)
                time.sleep(0.02)
            logger.info("Mão ESQUERDA SOLTA (--left-arm-limp): kp=kd=0 (mole).")
        elif self._left_hand_state is not None:
            # ATIVA: kp/kd já vêm de hand_kp/hand_kd (linhas acima); só semeia q p/ não saltar.
            for idx, joint_id in enumerate(Dex3_1_Left_JointIndex):
                self._left_hand_msg.motor_cmd[joint_id].q = self._left_hand_state.motor_state[idx].q

        # Mão DIREITA: semeia com posição física atual para não saltar.
        if self._right_hand_state is not None:
            for idx, joint_id in enumerate(Dex3_1_Right_JointIndex):
                self._right_hand_msg.motor_cmd[joint_id].q = self._right_hand_state.motor_state[idx].q

        # 🤚 HAND STREAMER: semeia os alvos com a posição MEDIDA (sem salto) e inicia a
        # thread de 100Hz. Default ON; G1_HAND_STREAMER=0 volta ao modo legado (30Hz+EMA).
        if self._hand_streamer_on:
            if self._left_hand_state is not None:
                self._hand_target_left = np.array(
                    [m.q for m in self._left_hand_state.motor_state], dtype=float)
            if self._right_hand_state is not None:
                self._hand_target_right = np.array(
                    [m.q for m in self._right_hand_state.motor_state], dtype=float)
            self._hand_streamer_stop.clear()
            self._hand_streamer_thread = threading.Thread(
                target=self._hand_streamer_worker, daemon=True, name="HandStreamer")
            self._hand_streamer_thread.start()
            logger.info("[hand-streamer] thread de 100Hz iniciada (clip de velocidade, jeito Unitree).")

        logger.info("Iniciando Heartbeat Anti-Queda...")
        # Zera o relógio do heartbeat ANTES de iniciá-lo: _last_action_time foi setado lá no
        # __init__, então no 1º tick o gap já passaria de 0.05s e o heartbeat dispararia
        # imediatamente (publicando comando antes do operador apertar X). Resetar evita isso.
        self._last_action_time = time.time()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
        self._heartbeat_thread.start()
        

    def _heartbeat_worker(self):
        """Mantém os motores rígidos quando o PC trava para salvar o episódio.
        Em dry-run (PROMETHEUS_DRY_RUN=1) não publica nada — apenas dorme."""
        import os as _os
        if _os.environ.get("PROMETHEUS_DRY_RUN") == "1":
            logger.info("[heartbeat] dry-run mode → no commands will be sent")
            while not self._hand_shutdown_event.is_set():
                time.sleep(0.5)
            return
        while not self._hand_shutdown_event.is_set():
            # Se passou mais de 0.05s (50ms) sem receber comando do LeRobot, o PC está salvando dados!
            if time.time() - self._last_action_time > 0.05:
                
                # 1. Re-envia o comando de Corpo (Braços e Cintura) para segurar a posição.
                # Quando o ARM STREAMER está VIVO, ELE já publica self.msg a 250Hz
                # (inclusive durante o save) — então o heartbeat não republica o corpo,
                # pra evitar dupla publicação/escrita concorrente do mesmo self.msg.
                # SAFETY NET: se o streamer estava ligado mas a thread MORREU, o heartbeat
                # retoma o corpo (senão o braço ficaria órfão, sem comando → cai).
                _streamer_thr = getattr(self, "_arm_streamer_thread", None)
                _streamer_alive = (getattr(self, "_arm_streamer_on", False)
                                   and _streamer_thr is not None and _streamer_thr.is_alive())
                if not _streamer_alive:
                    if hasattr(self, 'msg') and hasattr(self, 'lowcmd_publisher') and self.lowcmd_publisher is not None:
                        try:
                            self.lowcmd_publisher.Write(self.msg)
                        except Exception:
                            pass
                
                # 2. Re-envia o ÚLTIMO COMANDO das mãos (q comandado + kp), NÃO a posição medida.
                # IMPORTANTE: re-semear da MEDIDA afrouxa a força do grip — o objeto bloqueia os
                # dedos, então q_medido < q_comandado → erro de posição ~0 → torque ~0 → a garra
                # relaxa e SOLTA o copo durante a pausa de encode (~13s) entre episódios. Manter o
                # q COMANDADO preserva a força do grip. (No connect o _*_hand_msg.q já foi semeado
                # da medida, então no startup não há salto, e o firmware Dex3 tem watchdog: sem
                # comando periódico ele re-enrijece, por isso re-publicamos sempre.)
                # MÃOS: quando o HAND STREAMER está VIVO, ELE já publica a 100Hz (inclusive
                # durante o save, mantendo o grip), então o heartbeat NÃO republica as mãos
                # (evita dupla publicação concorrente do mesmo _*_hand_msg sob o _hand_lock).
                # SAFETY NET: se o streamer morreu, o heartbeat reassume as mãos.
                _hstr = getattr(self, "_hand_streamer_thread", None)
                _hand_streamer_alive = (getattr(self, "_hand_streamer_on", False)
                                        and _hstr is not None and _hstr.is_alive())
                if not _hand_streamer_alive:
                    # ESQUERDA: limp → msg com kp=0 (mole); ativa → msg com grip comandado.
                    if self._left_hand_cmd_pub is not None:
                        try:
                            self._left_hand_cmd_pub.Write(self._left_hand_msg)
                        except Exception:
                            pass
                    # DIREITA: re-publica o comando → segura o grip firme durante a pausa.
                    if self._right_hand_cmd_pub is not None:
                        try:
                            self._right_hand_cmd_pub.Write(self._right_hand_msg)
                        except Exception:
                            pass
            
            # Checa o pulso a cada 10 milissegundos
            time.sleep(0.01)

    def _ramp_hand(self, msg, joints, tgt, vmax):
        """Clip de velocidade VETORIAL contra o comando anterior (msg.motor_cmd[].q).
        Escala todas as juntas juntas → preserva a forma do grip. NaN/Inf guard.
        Isto SUAVIZA a trajetória (rampa entre alvos de 30Hz), NÃO previne travamento —
        o anti-trava é o kp mola do grasp. O comando converge pro alvo normalmente."""
        prev = np.array([msg.motor_cmd[jid].q for jid in joints], dtype=float)
        tgt = np.asarray(tgt, dtype=float)
        delta = tgt - prev
        if vmax > 0:
            mx = float(np.max(np.abs(delta))) if delta.size else 0.0
            # scale >= 1.0: nunca divide por zero, nunca passa de vmax por frame.
            scale = max(mx / vmax, 1.0)
            newq = prev + delta / scale
        else:
            newq = prev
        if not np.any(np.isnan(newq)) and not np.any(np.isinf(newq)):
            for i, jid in enumerate(joints):
                msg.motor_cmd[jid].q = float(newq[i])

    def _hand_streamer_worker(self):
        """Publica os dedos a ~100Hz com clip de velocidade (réplica do Dex3_1_Controller
        da Unitree, fps=100, + a rampa do arm streamer). Entrega uma rampa contínua em vez
        de degraus a 30Hz → sem stair-stepping. Lê _hand_target_* (setado por send_action
        sob _hand_lock). Durante o save (PC travado) o alvo não muda → o streamer mantém o
        grip sozinho (substitui a função do heartbeat para as mãos)."""
        dt = 1.0 / 100.0
        fails = 0
        while not self._hand_streamer_stop.is_set():
            t0 = time.perf_counter()
            try:
                with self._hand_lock:
                    tun = getattr(self, "_tuning", None) or {}
                    vmax = float(tun.get("hand_velocity_limit", 10.0)) * dt
                    # ESQUERDA: limp → força q=0 (kp já é 0); senão rampa até o alvo.
                    if getattr(self, "_left_limp", False):
                        for jid in Dex3_1_Left_JointIndex:
                            self._left_hand_msg.motor_cmd[jid].q = 0.0
                    elif self._hand_target_left is not None:
                        self._ramp_hand(self._left_hand_msg, list(Dex3_1_Left_JointIndex),
                                        self._hand_target_left, vmax)
                    # DIREITA: sempre rampa até o alvo.
                    if self._hand_target_right is not None:
                        self._ramp_hand(self._right_hand_msg, list(Dex3_1_Right_JointIndex),
                                        self._hand_target_right, vmax)
                    # Publica (re-checa o stop p/ não escrever depois do disconnect).
                    if not self._hand_streamer_stop.is_set():
                        if self._left_hand_cmd_pub is not None:
                            self._left_hand_cmd_pub.Write(self._left_hand_msg)
                        if self._right_hand_cmd_pub is not None:
                            self._right_hand_cmd_pub.Write(self._right_hand_msg)
                fails = 0
            except Exception as e:
                fails += 1
                if fails <= 3 or fails % 100 == 0:
                    logger.warning(f"[hand-streamer] erro no worker (#{fails}): {type(e).__name__}: {e}")
                if fails >= 10:
                    logger.error("[hand-streamer] 10 falhas consecutivas — encerrando; heartbeat assume as mãos.")
                    break
            elapsed = time.perf_counter() - t0
            self._hand_streamer_stop.wait(max(0.0, dt - elapsed))

    def disconnect(self):
        """Disconnect from robot body and hands."""

        # 🛑 0. PARA O HAND STREAMER antes de mexer nas msgs/publishers.
        if hasattr(self, "_hand_streamer_stop"):
            self._hand_streamer_stop.set()
        _hstr = getattr(self, "_hand_streamer_thread", None)
        if _hstr is not None:
            _hstr.join(timeout=1.0)
            if _hstr.is_alive():
                logger.warning("[hand-streamer] thread não encerrou no timeout (join 1s).")

        # 🛑 1. LÓGICA DE SOLTURA (LIMP MODE) PARA AS MÃOS
        logger.info("Desligando motores das mãos (Limp Mode)...")
        if self._left_hand_cmd_pub is not None and self._right_hand_cmd_pub is not None:
            # Cria mensagens vazias
            relax_left = self._left_hand_msg
            relax_right = self._right_hand_msg
            
            # Força o Status 0 (Desligado) e Kp/Kd = 0 para todos os dedos
            for joint_id in Dex3_1_Left_JointIndex:
                relax_left.motor_cmd[joint_id].mode = (joint_id & 0x0F) | (0x00 << 4)
                relax_left.motor_cmd[joint_id].kp = 0.0
                relax_left.motor_cmd[joint_id].kd = 0.0
                relax_left.motor_cmd[joint_id].tau = 0.0
                
            for joint_id in Dex3_1_Right_JointIndex:
                relax_right.motor_cmd[joint_id].mode = (joint_id & 0x0F) | (0x00 << 4)
                relax_right.motor_cmd[joint_id].kp = 0.0
                relax_right.motor_cmd[joint_id].kd = 0.0
                relax_right.motor_cmd[joint_id].tau = 0.0

            # Dispara a mensagem 5 vezes seguidas para garantir que a placa Dex3 ouviu
            # antes de cortarmos a rede ZMQ
            for _ in range(5):
                self._left_hand_cmd_pub.Write(relax_left)
                self._right_hand_cmd_pub.Write(relax_right)
                time.sleep(0.01)

        # 🛑 2. ENCERRA AS THREADS NORMALMENTE
        # Signal hand thread to stop
        self._hand_shutdown_event.set()
        
        # Wait for hand thread to finish
        if self._hand_subscribe_thread is not None:
            self._hand_subscribe_thread.join(timeout=2.0)
            if self._hand_subscribe_thread.is_alive():
                logger.warning("Hand subscribe thread did not stop cleanly")

        if hasattr(self, 'depth_sub') and self.depth_sub:
            self.depth_sub.close()
        if hasattr(self, 'zmq_ctx') and self.zmq_ctx:
            self.zmq_ctx.term()
        
        # Disconnect body (O LeRobot cuida do resto)
        super().disconnect()

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define action space: body joints (based on control_mode) + hand joints.
        
        - full_body mode: 29 body + 14 hand = 43 joints
        - upper_body mode: 14 arm + 14 hand = 28 joints
        """
        features = super().action_features
        for name in self.left_hand_joint_names:
            features[f"{name}.q"] = float
        for name in self.right_hand_joint_names:
            features[f"{name}.q"] = float
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define observation space: body joints + hand joints + pressure sensors."""
        features = super().observation_features
        for name in self.left_hand_joint_names:
            features[f"{name}.q"] = float
        for name in self.right_hand_joint_names:
            features[f"{name}.q"] = float


        return features

    def get_observation(self) -> RobotObservation:
        """Get observation including hand joint positions and pressure."""
        obs = super().get_observation() or {}
        
        # Add left hand state 
        for i, name in enumerate(self.left_hand_joint_names):
            if self._left_hand_state is not None:
                obs[f"{name}.q"] = float(self._left_hand_state.motor_state[i].q)
            else:
                obs[f"{name}.q"] = 0.0  
        
        # Add right hand state
        for i, name in enumerate(self.right_hand_joint_names):
            if self._right_hand_state is not None:
                obs[f"{name}.q"] = float(self._right_hand_state.motor_state[i].q)
            else:
                obs[f"{name}.q"] = 0.0

        if self._left_hand_state is not None:
            left_p = self._left_hand_state.pressure
            right_p = self._right_hand_state.pressure
        else:
            left_p = np.zeros(108, dtype=np.float32)
            right_p = np.zeros(108, dtype=np.float32)

        # Injeta no dicionário 'obs' para o init_record pescar
        obs["left_hand_pressure"] = left_p.astype(np.float32)
        obs["right_hand_pressure"] = right_p.astype(np.float32)
                
        return obs

    def _smooth_hand_q(self, q: np.ndarray, side: str, seed_state) -> np.ndarray:
        """Anti-tremor dos dedos: EMA passa-baixa + clamp de velocidade por frame.

        Espelha o filtro do braço (unitree_g1.send_action) mas com tuning próprio das
        mãos (mais responsivo, pra não atrasar o grasp). Lê hand_smoothing_alpha e
        hand_max_delta de g1_tuning.json (hot-reload, mesmos defaults se ausentes).
        alpha menor = mais suave; max_delta menor = mais lento/estável.
        """
        tun = self._read_tuning()
        alpha = float(tun.get("hand_smoothing_alpha", 0.4))
        max_delta = float(tun.get("hand_max_delta", 0.3))
        attr = "_last_hand_q_left" if side == "left" else "_last_hand_q_right"
        last = getattr(self, attr)
        if last is None:
            # Semeia da pose MEDIDA (evita tranco no 1º frame), senão do próprio alvo.
            if seed_state is not None:
                last = np.array([m.q for m in seed_state.motor_state], dtype=float)
            else:
                last = q.astype(float).copy()
        else:
            # Anti-trava: re-ancora do medido SOMENTE quando o EMA ultrapassou na direcao
            # OPOSTA ao alvo (ex: last acha que esta fechado mas voce quer abrir).
            # Se alvo e last concordam na direcao (ex: ambos querem fechar contra o objeto),
            # NAO re-ancora — deixa o EMA empurrar com kp alto (grip force).
            # Re-ancoragem incondicional bloqueava o fechamento: medido=0.8, last->1.74,
            # re-ancora pra 0.8 todo frame → dedo nunca passava de 0.8+max_delta=1.3 rad.
            if seed_state is not None:
                measured = np.array([m.q for m in seed_state.motor_state], dtype=float)
                drift = last - measured  # sinalizado: positivo = last a frente do medido
                target_offset = q - measured  # direcao do alvo em relacao ao medido
                # Re-ancora onde last divergiu muito E na direcao CONTRARIA ao alvo
                should_reanchor = (np.abs(drift) > 0.35) & (np.sign(drift) != np.sign(target_offset))
                last = np.where(should_reanchor, measured, last)
        smoothed = (1.0 - alpha) * last + alpha * q
        delta = np.clip(smoothed - last, -max_delta, max_delta)
        out = last + delta
        setattr(self, attr, out)
        return out

    def send_action(self, action: RobotAction) -> RobotAction:
        """Send action to robot including hand commands."""

        self._last_action_time = time.time()

        # Send body action
        super().send_action(action)
        
        # Check if we have hand publishers
        if self._left_hand_cmd_pub is None or self._right_hand_cmd_pub is None:
            return action
        
        # Check if action contains hand commands
        first_joint = self.left_hand_joint_names[0]
        if f"{first_joint}.q" not in action:
            return action
        
        # Extract and clamp left hand targets
        left_q = np.zeros(Dex3_Num_Motors)
        for i, name in enumerate(self.left_hand_joint_names):
            left_q[i] = action.get(f"{name}.q", 0.0)
        left_q = np.clip(left_q, DEX3_LEFT_LOWER_LIMITS, DEX3_LEFT_UPPER_LIMITS)
        
        # Extract and clamp right hand targets
        right_q = np.zeros(Dex3_Num_Motors)
        for i, name in enumerate(self.right_hand_joint_names):
            right_q[i] = action.get(f"{name}.q", 0.0)
        right_q = np.clip(right_q, DEX3_RIGHT_LOWER_LIMITS, DEX3_RIGHT_UPPER_LIMITS)

        # 🤚 MODO STREAMER (default): só atualiza os ALVOS; a thread de 100Hz faz a rampa
        # (clip de velocidade contra o comando anterior) e publica. Não suaviza nem publica
        # aqui — elimina o stair-stepping de 30Hz e evita dupla publicação.
        if getattr(self, "_hand_streamer_on", False):
            with self._hand_lock:
                self._hand_target_right = right_q.astype(float)
                # ESQUERDA: se limp, o streamer força q=0 (kp já é 0); senão persegue o alvo.
                if not getattr(self, "_left_limp", False):
                    self._hand_target_left = left_q.astype(float)
            if not getattr(self, "_left_dbg_done", False):
                kps = [self._left_hand_msg.motor_cmd[j].kp for j in Dex3_1_Left_JointIndex]
                _modo = "SOLTA (limp)" if getattr(self, "_left_limp", False) else "ATIVA"
                print(f"\n   🔧 [SEND_ACTION+STREAMER 100Hz] mão esq {_modo} — kp={kps}", flush=True)
                self._left_dbg_done = True
            return action

        # ───── MODO LEGADO (G1_HAND_STREAMER=0): EMA + clamp + publish a 30Hz ─────
        # 🪶 anti-tremor: passa-baixa EMA + clamp de velocidade nos dedos (ver _smooth_hand_q)
        right_q = self._smooth_hand_q(right_q, "right", self._right_hand_state)

        # Update command messages (direita)
        for idx, joint_id in enumerate(Dex3_1_Right_JointIndex):
            self._right_hand_msg.motor_cmd[joint_id].q = right_q[idx]

        # Mão ESQUERDA: depende de --left-arm-limp (env G1_LEFT_ARM_LIMP).
        if getattr(self, "_left_limp", False):
            # SOLTA: FORÇA kp=0/kd=0/tau=0/q=0 a cada tick (mole), ignorando outros setadores.
            for joint_id in Dex3_1_Left_JointIndex:
                self._left_hand_msg.motor_cmd[joint_id].kp = 0.0
                self._left_hand_msg.motor_cmd[joint_id].kd = 0.0
                self._left_hand_msg.motor_cmd[joint_id].tau = 0.0
                self._left_hand_msg.motor_cmd[joint_id].q = 0.0
        else:
            # ATIVA (default): comando normal do teleop — q da action (kp/kd setados no connect).
            left_q = self._smooth_hand_q(left_q, "left", self._left_hand_state)
            for idx, joint_id in enumerate(Dex3_1_Left_JointIndex):
                self._left_hand_msg.motor_cmd[joint_id].q = left_q[idx]
        if not getattr(self, "_left_dbg_done", False):
            kps = [self._left_hand_msg.motor_cmd[j].kp for j in Dex3_1_Left_JointIndex]
            _modo = "SOLTA (limp)" if getattr(self, "_left_limp", False) else "ATIVA"
            print(f"\n   🔧 [SEND_ACTION] mão esq {_modo} — kp={kps}", flush=True)
            self._left_dbg_done = True
        if self._left_hand_cmd_pub is not None:
            self._left_hand_cmd_pub.Write(self._left_hand_msg)

        # Mão DIREITA: comando normal do teleop
        self._right_hand_cmd_pub.Write(self._right_hand_msg)
        
        return action
