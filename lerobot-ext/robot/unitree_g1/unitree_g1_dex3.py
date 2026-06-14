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
from .unitree_g1_loco import UnitreeG1, UnitreeG1Config
#from .unitree_g1 import UnitreeG1, UnitreeG1Config
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
    pressure: np.ndarray = field(default_factory=lambda: np.zeros(33, dtype=np.float32))
    temperature: np.ndarray = field(default_factory=lambda: np.zeros(33, dtype=np.float32))


@RobotConfig.register_subclass("unitree_g1_dex3")
@dataclass
class UnitreeG1Dex3Config(UnitreeG1Config):
    """Configuration for Unitree G1 with Dex3-1 hands."""
    hand_kp: float = 0.5  # Position gain for hand motors
    hand_kd: float = 0.15  # Damping gain for hand motors
    hand_control_dt: float = 0.005  # 100 Hz control loop

    use_loco: bool = False  # False = Low Level (Suporte), True = High Level (Andando)
    
    def __post_init__(self):

        if self.use_loco:
            self.control_mode = "high_level"
        else:
            self.control_mode = "upper_body"

        if self.is_simulation and not self.remote_sim_ip:
            self.robot_ip = "127.0.0.1"   # só local
        elif self.remote_sim_ip:
            self.robot_ip = self.remote_sim_ip  # aponta câmeras ZMQ para o PC remoto
        # else: mantém robot_ip do hardware real
        

        # LÓGICA DE RESOLUÇÃO DINÂMICA
        if self.is_simulation:
            self.robot_ip = "127.0.0.1"
            # Simulação: Leve e rápida para não gargalar a GPU/CPU
            cam2_width = 848
            cam2_height = 480
        else:
            # Hardware Real: Resolução máxima da Intel RealSense
            cam2_width = 848
            cam2_height = 480

            
        # Adiciona as câmeras ZMQ ao LeRobot usando as variáveis dinâmicas
        if not self.cameras:
            from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig
            
            self.cameras = {
                # A NOSSA ÚNICA CÂMERA RGB (HD para o VR e para a IA)
                "head_camera": ZMQCameraConfig(
                    server_address=self.robot_ip, port=5555, camera_name="head_camera", width=cam2_width, height=cam2_height, warmup_s=6
                ),
                
                # AS 3 LENTES TÉCNICAS (Baixa Resolução para o processamento ser imediato)
                "head_camera_depth": ZMQCameraConfig(
                    server_address=self.robot_ip, port=5555, camera_name="head_camera_depth", width=cam2_width, height=cam2_height, warmup_s=6
                )
                ,
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
                old_p = self._left_hand_state.pressure if self._left_hand_state else np.zeros(33, dtype=np.float32)
                
                left_state = HandState()
                for idx, joint_id in enumerate(Dex3_1_Left_JointIndex):
                    left_state.motor_state[idx].q = left_msg.motor_state[joint_id].q
                
                left_state.pressure = old_p # Restaura a pressão
                self._left_hand_state = left_state
            
            right_msg = self._right_hand_state_sub.Read()
            if right_msg is not None:
                old_p = self._right_hand_state.pressure if self._right_hand_state else np.zeros(33, dtype=np.float32)
                
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
                            while len(flat_p) < 33:
                                flat_p.append(0.0)
                            
                            # Atualiza direto no estado atual
                            if side == "left" and self._left_hand_state:
                                self._left_hand_state.pressure = np.array(flat_p[:33], dtype=np.float32)
                            elif side == "right" and self._right_hand_state:
                                self._right_hand_state.pressure = np.array(flat_p[:33], dtype=np.float32)
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

        logger.info("Iniciando Heartbeat Anti-Queda...")
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
        self._heartbeat_thread.start()
        

    def _heartbeat_worker(self):
        """Mantém os motores rígidos quando o PC trava para salvar o episódio"""
        while not self._hand_shutdown_event.is_set():
            # Se passou mais de 0.05s (50ms) sem receber comando do LeRobot, o PC está salvando dados!
            if time.time() - self._last_action_time > 0.05:
                
                # 1. Re-envia o comando de Corpo (Braços e Cintura) para segurar a posição
                if hasattr(self, 'msg') and hasattr(self, 'lowcmd_publisher') and self.lowcmd_publisher is not None:
                    try:
                        self.lowcmd_publisher.Write(self.msg)
                    except Exception:
                        pass
                
                # 2. Re-envia o comando das Mãos (Dedos Dex3)
                if self._left_hand_cmd_pub is not None and self._right_hand_cmd_pub is not None:
                    try:
                        self._left_hand_cmd_pub.Write(self._left_hand_msg)
                        self._right_hand_cmd_pub.Write(self._right_hand_msg)
                    except Exception:
                        pass
            
            # Checa o pulso a cada 10 milissegundos
            time.sleep(0.01)

    def disconnect(self):
        """Disconnect from robot body and hands."""
        
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
            left_p = np.zeros(33, dtype=np.float32)
            right_p = np.zeros(33, dtype=np.float32)

        # Injeta no dicionário 'obs' para o init_record pescar
        obs["left_hand_pressure"] = left_p.astype(np.float32)
        obs["right_hand_pressure"] = right_p.astype(np.float32)
                
        return obs

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
        
        # Update command messages
        for idx, joint_id in enumerate(Dex3_1_Left_JointIndex):
            self._left_hand_msg.motor_cmd[joint_id].q = left_q[idx]
        for idx, joint_id in enumerate(Dex3_1_Right_JointIndex):
            self._right_hand_msg.motor_cmd[joint_id].q = right_q[idx]
        
        # Publish commands
        self._left_hand_cmd_pub.Write(self._left_hand_msg)
        self._right_hand_cmd_pub.Write(self._right_hand_msg)
        
        return action