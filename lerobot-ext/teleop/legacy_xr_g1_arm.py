import zmq
import time
import cv2
import os
import threading
import logging
import numpy as np
from dataclasses import dataclass
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.processor import RobotAction

# Imports do ecossistema Unitree (ajuste os caminhos conforme sua pasta)
from televuer import TeleVuerWrapper
from .utils.sensor_utils import SensorClient, ImageUtils
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

logger = logging.getLogger(__name__)


def _left_arm_limp() -> bool:
    """True se --left-arm-limp (env G1_LEFT_ARM_LIMP) está ativo.
    Presença com valor real liga; '', '0', 'false' e ausência = desligado
    (evita o footgun de bool('0') == True)."""
    return os.environ.get("G1_LEFT_ARM_LIMP", "") not in ("", "0", "false", "False")


@TeleoperatorConfig.register_subclass("xr_g1_arm")
@dataclass
class XRG1ArmConfig(TeleoperatorConfig):
    img_server_ip: str = "127.0.0.1"
    #img_server_ip: str = "127.0.0.1"
    is_simulation: bool = True
    input_mode: str = "hand"       # 'hand' ou 'controller'
    display_mode: str = "immersive" # 'immersive', 'ego', 'pass-through'
    ee_type: str = "dex3"
    zmq: bool = True
    webrtc: bool = False

class XRG1Arm(Teleoperator):
    config_class = XRG1ArmConfig
    name = "xr_g1_arm"

    def __init__(self, config: XRG1ArmConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False

        # Removemos o hardcode do IP do robô para que o VR possa usar o Proxy Local
        # se configurado, aliviando a rede.
        
        # Carrega os nomes das juntas do seu robô no LeRobot
        from robot.unitree_g1.g1_utils import G1_29_JointIndex, LEFT_HAND_JOINT_NAMES, RIGHT_HAND_JOINT_NAMES
        self._left_hand_names = LEFT_HAND_JOINT_NAMES
        self._right_hand_names = RIGHT_HAND_JOINT_NAMES
        
        self.body_joints = {f"{motor.name}.q": 0.0 for motor in G1_29_JointIndex}
        self.hand_joints = {f"{name}.q": 0.0 for name in self._left_hand_names + self._right_hand_names}

        # Filtro EMA para squeeze e trigger do Quest (que são quasi-binários 0/1).
        # Sem filtro, cada frame onde o squeeze muda de 0→1 vira um step de ±1.5 rad
        # nos dedos → tremor. Alpha=0.2 suaviza em ~5 frames (150ms a 30Hz).
        self._sq_r = 0.0; self._sq_l = 0.0
        self._tr_r = 0.0; self._tr_l = 0.0
        _EMA_ALPHA = 0.5   # mão: ~3 frames (100ms) pra 95% do alvo; 0.2 era lento demais (~15 frames)
        self._EMA_ALPHA = _EMA_ALPHA

        # Inicializa o Wrapper do VR e o Solver IK
        self.tv_wrapper = None
        self.arm_ik = None
        self.hand_retargeter = None

        # Estado atual do robô (necessário como "semente" para o cálculo do IK)
        self.current_arm_q = np.zeros(14)
        self.current_arm_dq = np.zeros(14)


        # Modo "hand": rastreia se as mãos já foram detectadas pelo menos uma vez.
        self.vr_started = False

        # Sempre começa travado — botão X (left_ctrl_aButton) destrava em
        # qualquer modo (hand ou controller).
        self.controller_enabled = False
        self.last_x_state = False
        self.last_y_state = False


        # Estado do CLUTCH (alinhamento relativo). Ao destravar, ancora a pose
        # atual do controle e a pose atual do robô; daí em diante o robô se move
        # apenas pelo DELTA do controle. Isso elimina o "salto" que jogava o
        # braço para a pose absoluta do controle (e forçava contra a mesa).
        self.clutch_anchored = False
        self.ctrl_ref_left = None
        self.ctrl_ref_right = None
        self.robot_ref_left = None
        self.robot_ref_right = None
        # Última pose-alvo comandada para cada braço. Usada para re-ancorar o
        # clutch (ex: ao salvar episódio) sem o robô saltar: o robô continua de
        # onde estava em vez de voltar para FK(0).
        self.last_left_target = None
        self.last_right_target = None
        # Watchdog de tempo: se o loop congelar (ex: encoding de vídeo ao salvar)
        # e voltar depois de um gap, re-ancora o clutch para o robô não aplicar
        # de uma vez o movimento que o controle acumulou durante o freeze.
        self._last_clutch_time = None

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return

        logger.info("Iniciando conexão com o Vuer VR...")
        
        # 1. Inicia o servidor WebRTC / WebSocket do Vuer
        camera_shape = (480, 640) if self.config.is_simulation else (480, 848)
        self.tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=(self.config.input_mode == "hand"),
            binocular=False,
            img_shape=camera_shape,
            display_mode=self.config.display_mode,
            zmq=self.config.zmq,
            webrtc=self.config.webrtc,
            webrtc_url=f"https://{self.config.img_server_ip}:60000/offer",
        )
        
        # 2. Inicia o Solver de Cinemática Inversa dos braços
        logger.info("Carregando URDF e IK do Braço G1_29...")
        self.arm_ik = G1_29_ArmIK()

        # 3. Inicia o Retargeting Das Mãos
        if self.config.ee_type == "dex3":
            logger.info("Iniciando algoritmo de Retargeting para Dex3...")
            self.hand_retargeter = HandRetargeting(HandType.UNITREE_DEX3)

        self._is_connected = True

        # NOVA PARTE: Usando o SensorClient que já funciona!
        if self.config.zmq and self.config.display_mode != "pass-through":
            logger.info(f"Conectando ao feed de vídeo ZMQ via SensorClient em {self.config.img_server_ip}:5555...")
            self.sensor_client = SensorClient()
            self.sensor_client.start_client(server_ip=self.config.img_server_ip, port=5555)

            # Inicia uma thread em background para receber as imagens
            self.video_thread = threading.Thread(target=self._receive_video_feed, daemon=True)
            self.video_thread.start()

        logger.info("VR Teleoperator Conectado! Visite o link do Vuer no navegador do headset.")

    def _receive_video_feed(self):
        while self._is_connected:
            try:
                # Recebe a mensagem estruturada usando o client oficial do seu sim
                data = self.sensor_client.receive_message()
                
                if not data:
                    time.sleep(0.005)
                    continue

                # Extrai a imagem do dicionário (lidando com os dois formatos que seu script suporta)
                img_data = None
                cam_name = "head_camera"  # Nome padrão, mas vamos buscar dinamicamente se falhar
                
                if "images" in data and cam_name in data["images"]:
                    img_data = data["images"][cam_name]
                elif cam_name in data:
                    img_data = data[cam_name]
                else:
                    # Pega a primeira câmera que encontrar no dicionário
                    keys = [k for k in data.keys() if k not in ["timestamps", "images"]]
                    if keys:
                        img_data = data[keys[0]]

                # Se achou a imagem, decodifica usando o ImageUtils
                if img_data is not None:
                    if isinstance(img_data, str):
                        img = ImageUtils.decode_image(img_data)
                    else:
                        img = img_data  # Assume que já é numpy array

                    if img is not None and isinstance(img, np.ndarray):
                        # Converte de BGR para RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Envia para o VR
                        self.tv_wrapper.render_to_xr(img_rgb)

            except Exception as e:
                logger.error(f"Erro no feed de vídeo ZMQ: {e}")
                time.sleep(0.01) # Pausa curta para não floodar o log em caso de erro contínuo

    def disconnect(self) -> None:
        if self._is_connected:
            self._is_connected = False
            
            # Aguarda a thread de vídeo encerrar
            if hasattr(self, 'video_thread') and self.video_thread.is_alive():
                self.video_thread.join(timeout=1.0)
            
            if self.tv_wrapper:
                self.tv_wrapper.close()
                
            # Fecha o client corretamente
            if hasattr(self, 'sensor_client'):
                self.sensor_client.stop_client()

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

    def _trigger_record_event(self, action_type):
        """Injeta comandos diretamente no script de gravação (se ele estiver rodando)"""
        import sys
        import threading
        import time
        import os
        
        # Verifica se estamos rodando dentro do script principal (gravador)
        if "__main__" in sys.modules:
            main_mod = sys.modules["__main__"]
            events = getattr(main_mod, "global_events", None)
            
            if action_type == "save" and events is not None:
                print("\n   🎮 [CONTROLE VR] Ação: SALVANDO e gravando o próximo... ✅")
                # Sem reset (reset_time_s=0): só encerra o episódio. O clutch
                # re-ancora para o robô continuar no mesmo estado, sem salto.
                events["exit_early"] = True
                self.clutch_anchored = False

            elif action_type == "discard" and events is not None:
                print("\n   🎮 [CONTROLE VR] Ação: DESCARTANDO e regravando... ❌")
                events["rerecord_episode"] = True
                events["exit_early"] = True
                self.clutch_anchored = False
                # Reseta o timer do episódio no viewer OpenCV imediatamente
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
                    # Destravado: libera o gravador para seguir o VR
                    if hasattr(main_mod, "robot_paused"):
                        main_mod.robot_paused = False
                    print("\n   🎮 [CONTROLE VR] Ação: Robô DESTRAVADO ▶️")
                else:
                    # Congelado: para o robô na posição física atual
                    if hasattr(main_mod, "robot_paused"):
                        main_mod.robot_paused = True
                    print("\n   🎮 [CONTROLE VR] Ação: Robô CONGELADO 🧊")

            elif action_type == "exit":
                print("\n   🎮 [CONTROLE VR] Ação: ENCERRANDO o sistema... 🛑")
                # Se estiver no gravador, manda o sinal de parada global
                if events is not None:
                    events["stop_recording"] = True
                    events["exit_early"] = True
                else:
                    # Se estiver só testando a teleoperação isolada, força o fechamento
                    self.disconnect()
                    os._exit(0)
        else:
            # Modo teleoperação normal - ignora silenciosamente
            pass

    def get_action(self) -> RobotAction:
        if not self._is_connected:
             raise ConnectionError("XR Teleoperator não está conectado.")
        

        # 1. Pega os dados do Headset VR
        tele_data = self.tv_wrapper.get_tele_data()

        # =========================
        # CONTROLE ESQUERDO (X = Pause/Play | Y = Encerrar)
        # =========================
        x_pressed = getattr(tele_data, "left_ctrl_aButton", False) # Botão X físico
        y_pressed = getattr(tele_data, "left_ctrl_bButton", False) # Botão Y físico

        # Detecta clique no X (Pause/Play)
        if x_pressed and not self.last_x_state:
            self._trigger_record_event("toggle_pause")

        # Detecta clique no Y (Sair / Encerrar)
        if y_pressed and not self.last_y_state:
            self._trigger_record_event("exit")

        self.last_x_state = x_pressed
        self.last_y_state = y_pressed

        # =========================
        # CONTROLE DE GRAVAÇÃO (A = Salvar / B = Descartar)
        # =========================
        a_pressed_right = getattr(tele_data, "right_ctrl_aButton", False)
        b_pressed_right = getattr(tele_data, "right_ctrl_bButton", False)

        # Inicializa as variáveis de estado de borda se não existirem
        if not hasattr(self, "last_a_right_state"): self.last_a_right_state = False
        if not hasattr(self, "last_b_right_state"): self.last_b_right_state = False

        # Verifica clique no A (Salvar)
        if a_pressed_right and not self.last_a_right_state:
            self._trigger_record_event("save")

        # Verifica clique no B (Descartar)
        if b_pressed_right and not self.last_b_right_state:
            self._trigger_record_event("discard")

        self.last_a_right_state = a_pressed_right
        self.last_b_right_state = b_pressed_right

        # =========================
        # 🚨 BLOQUEIO TOTAL AQUI
        # =========================
        if not self.controller_enabled:
            return {**self.body_joints, **self.hand_joints}

        # Em modo "hand": garante que as mãos estejam sendo rastreadas antes de mover.
        if self.config.input_mode == "hand":
            has_right_hand = np.any(tele_data.right_hand_pos != 0.0)
            has_left_hand = np.any(tele_data.left_hand_pos != 0.0)
            if not (has_right_hand or has_left_hand):
                if self.vr_started:
                    logger.warning("Rastreamento de mãos não detectado (VR inativo). Mantendo posição.")
                    self.vr_started = False
                return {**self.body_joints, **self.hand_joints}
            if not self.vr_started:
                logger.info(">>> MÃOS DETECTADAS NO VR! MOVENDO G1...")
                self.vr_started = True


        # 2. Calcula IK dos Braços (retorna 14 ângulos) — com CLUTCH
        # Watchdog: se houve um congelamento do loop (> 0.5s, típico do encoding
        # de vídeo ao salvar), re-ancora para o robô não saltar ao retomar.
        now = time.time()
        if self.clutch_anchored and self._last_clutch_time is not None and (now - self._last_clutch_time) > 0.5:
            self.clutch_anchored = False
        self._last_clutch_time = now

        # Ancora a referência no 1º frame após destravar: a partir daqui o robô
        # parte da pose ATUAL e segue apenas o DELTA do controle (sem salto).
        if not self.clutch_anchored:
            self.ctrl_ref_left = tele_data.left_wrist_pose.copy()
            self.ctrl_ref_right = tele_data.right_wrist_pose.copy()
            # Re-ancoragem (ex: após salvar): parte da ÚLTIMA pose comandada para
            # o robô continuar de onde estava, sem salto. Na 1ª vez (sem
            # histórico), usa a FK da config atual (q≈0, comandada no countdown).
            if self.last_left_target is not None:
                self.robot_ref_left = self.last_left_target.copy()
                self.robot_ref_right = self.last_right_target.copy()
            else:
                self.robot_ref_left, self.robot_ref_right = self.arm_ik.forward_kinematics(self.current_arm_q)
            self.clutch_anchored = True
            logger.info(">>> CLUTCH ancorado: robô parado. Mova os controles para movê-lo.")

        # Alvo = pose de referência do robô + movimento RELATIVO do controle
        # (translação somada no mundo; rotação relativa aplicada à rotação base)
        left_target = self.robot_ref_left.copy()
        left_target[:3, 3] = self.robot_ref_left[:3, 3] + (tele_data.left_wrist_pose[:3, 3] - self.ctrl_ref_left[:3, 3])
        left_target[:3, :3] = (tele_data.left_wrist_pose[:3, :3] @ self.ctrl_ref_left[:3, :3].T) @ self.robot_ref_left[:3, :3]

        right_target = self.robot_ref_right.copy()
        right_target[:3, 3] = self.robot_ref_right[:3, 3] + (tele_data.right_wrist_pose[:3, 3] - self.ctrl_ref_right[:3, 3])
        right_target[:3, :3] = (tele_data.right_wrist_pose[:3, :3] @ self.ctrl_ref_right[:3, :3].T) @ self.robot_ref_right[:3, :3]

        # Guarda a pose-alvo para permitir re-ancorar sem salto (ex: ao salvar)
        self.last_left_target = left_target.copy()
        self.last_right_target = right_target.copy()

        # Seed = None → o solver faz warm-start da PRÓPRIA solução anterior
        # (self.init_data) e o custo de suavidade (var_q_last) passa a penalizar o
        # salto desde o último frame. Antes passávamos self.current_arm_q, que NUNCA
        # é atualizado (send_feedback não é chamado no record loop) e ficava em zeros:
        # isso resetava o seed pra zero todo frame e anulava o smooth cost → o IPOPT
        # caía em soluções ligeiramente diferentes a cada frame (cotovelo "pulando") → tremor.
        sol_q, _ = self.arm_ik.solve_ik(left_target, right_target, None, None)

        # Mapeia os 14 ângulos para o dicionário do LeRobot
        # Esquerdo (índices 0 a 6)
        self.body_joints["kLeftShoulderPitch.q"] = sol_q[0]
        self.body_joints["kLeftShoulderRoll.q"]  = sol_q[1]
        self.body_joints["kLeftShoulderYaw.q"]   = sol_q[2]
        self.body_joints["kLeftElbow.q"]         = sol_q[3]
        self.body_joints["kLeftWristRoll.q"]     = sol_q[4]
        self.body_joints["kLeftWristPitch.q"]    = sol_q[5]
        self.body_joints["kLeftWristYaw.q"]      = sol_q[6]

        # Direito (índices 7 a 13)
        self.body_joints["kRightShoulderPitch.q"] = sol_q[7]
        self.body_joints["kRightShoulderRoll.q"]  = sol_q[8]
        self.body_joints["kRightShoulderYaw.q"]   = sol_q[9]
        self.body_joints["kRightElbow.q"]         = sol_q[10]
        self.body_joints["kRightWristRoll.q"]     = sol_q[11]
        self.body_joints["kRightWristPitch.q"]    = sol_q[12]
        self.body_joints["kRightWristYaw.q"]      = sol_q[13]

        # 3. Calcula o Retargeting das Mãos (Dedos)
        # 3. Calcula o Retargeting das Mãos (Dedos)
        if self.config.ee_type == "dex3":
            
            # =========================================================
            # MODO 1: RASTREAMENTO PELAS MÃOS (HAND TRACKING)
            # =========================================================
            if self.config.input_mode == "hand":
                # CORREÇÃO 2: Formatação e cálculo correto dos vetores das mãos (Conforme Dex3_1_Controller)
                left_hand_data = tele_data.left_hand_pos.reshape(25, 3)
                right_hand_data = tele_data.right_hand_pos.reshape(25, 3)
                
                # Só calcula se a mão foi detectada no frame atual
                if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])):
                    
                    ref_left_value = left_hand_data[self.hand_retargeter.left_indices[1,:]] - left_hand_data[self.hand_retargeter.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeter.right_indices[1,:]] - right_hand_data[self.hand_retargeter.right_indices[0,:]]

                    left_hand_q = self.hand_retargeter.left_retargeting.retarget(ref_left_value)[self.hand_retargeter.left_dex_retargeting_to_hardware]
                    right_hand_q = self.hand_retargeter.right_retargeting.retarget(ref_right_value)[self.hand_retargeter.right_dex_retargeting_to_hardware]

                    # --- AJUSTE DE PINÇA (OFFSET FIXO PARA TOQUE LEVE) ---
                    OFFSET_ESQUEDA = 0.17 
                    OFFSET_DIREITA = 0.0

                    left_hand_q[5] -= OFFSET_ESQUEDA
                    left_hand_q[6] -= OFFSET_ESQUEDA
                    right_hand_q[3] += OFFSET_DIREITA
                    right_hand_q[4] += OFFSET_DIREITA
                    
                    # NOVO: DETECÇÃO REAL DE PUNHO
                    dist_medio_esq = np.linalg.norm(left_hand_data[14] - left_hand_data[0])
                    dist_medio_dir = np.linalg.norm(right_hand_data[14] - right_hand_data[0])
                    
                    punho_esq = np.clip((0.15 - dist_medio_esq) / 0.09, 0.0, 1.0)
                    punho_dir = np.clip((0.15 - dist_medio_dir) / 0.09, 0.0, 1.0)
                    
                    FORCA_PUNHO = 0.8 
                    
                    left_hand_q[3] -= (FORCA_PUNHO * punho_esq)
                    left_hand_q[4] -= (FORCA_PUNHO * punho_esq)
                    left_hand_q[5] -= (FORCA_PUNHO * punho_esq) 
                    left_hand_q[6] -= (FORCA_PUNHO * punho_esq)
                    
                    right_hand_q[5] += (FORCA_PUNHO * punho_dir)
                    right_hand_q[6] += (FORCA_PUNHO * punho_dir)
                    right_hand_q[3] += (FORCA_PUNHO * punho_dir)
                    right_hand_q[4] += (FORCA_PUNHO * punho_dir)

                    # Por padrão controla as DUAS mãos; com --left-arm-limp só a direita.
                    if not _left_arm_limp():
                        for i, name in enumerate(self._left_hand_names):
                            self.hand_joints[f"{name}.q"] = left_hand_q[i]

                    for i, name in enumerate(self._right_hand_names):
                        self.hand_joints[f"{name}.q"] = right_hand_q[i]

            # =========================================================
            # MODO 2: RASTREAMENTO POR CONTROLES (VR CONTROLLERS)
            # =========================================================
            elif self.config.input_mode == "controller":
                
                # --- HACK DE MEMÓRIA: INJEÇÃO DE IMPEDÂNCIA (KP/KD) ---
                if not hasattr(self, "kp_hacked"):
                    import gc
                    for obj in gc.get_objects():
                        if type(obj).__name__ == "UnitreeG1Dex3":
                            
                            # "MOLA" (grasp complacente) — design original que NAO trava.
                            # kp alto (1.0/2.0) contra objeto/batente => erro de posicao sustentado
                            # => tau=kp*erro grande => sobrecorrente => firmware Dex3 desliga o motor.
                            # kp=0.3 mantem o torque baixo mesmo no limite => fecha sem travar.
                            NOVO_KP = 0.3   # dedos: mola
                            NOVO_KD = 0.2
                            KP_BASE_POLEGAR = 0.8  # polegar um pouco mais firme p/ conseguir retornar
                            
                            # Mão ESQUERDA: por padrão recebe o MESMO kp da direita (ATIVA).
                            # Com --left-arm-limp (env G1_LEFT_ARM_LIMP) fica mole (kp=0).
                            _left_limp = _left_arm_limp()
                            if hasattr(obj, "_left_hand_msg") and obj._left_hand_msg is not None:
                                for i in range(7):
                                    if _left_limp:
                                        obj._left_hand_msg.motor_cmd[i].kp = 0.0
                                        obj._left_hand_msg.motor_cmd[i].kd = 0.0
                                        obj._left_hand_msg.motor_cmd[i].q = 0.0
                                        obj._left_hand_msg.motor_cmd[i].tau = 0.0
                                    else:
                                        obj._left_hand_msg.motor_cmd[i].kp = KP_BASE_POLEGAR if i == 0 else NOVO_KP
                                        obj._left_hand_msg.motor_cmd[i].kd = NOVO_KD

                            if hasattr(obj, "_right_hand_msg") and obj._right_hand_msg is not None:
                                for i in range(7):
                                    obj._right_hand_msg.motor_cmd[i].kp = KP_BASE_POLEGAR if i == 0 else NOVO_KP
                                    obj._right_hand_msg.motor_cmd[i].kd = NOVO_KD
                                    
                            self.kp_hacked = True
                            print(f"\n   🪽 [HACK] Kp ajustado! Dedos em {NOVO_KP}, mas base do polegar em {KP_BASE_POLEGAR} para conseguir retornar.")
                            break

                # --- LEITURA DOS GATILHOS E SQUEEZE COM FILTRO EMA ---
                # O squeeze e o trigger do Quest são quasi-binários (saltam 0→1 abruptamente).
                # Sem filtro, cada salto vira um step de ±1.5 rad nos dedos → tremor visível.
                # EMA (alpha=0.2) suaviza em ~5 frames / 150ms sem atrasar a resposta perceptivelmente.
                raw_tr_l = np.clip((10.0 - tele_data.left_ctrl_triggerValue)  / 10.0, 0.0, 1.0)
                raw_tr_r = np.clip((10.0 - tele_data.right_ctrl_triggerValue) / 10.0, 0.0, 1.0)
                raw_sq_l = np.clip(tele_data.left_ctrl_squeezeValue,  0.0, 1.0)
                raw_sq_r = np.clip(tele_data.right_ctrl_squeezeValue, 0.0, 1.0)
                a = self._EMA_ALPHA
                self._tr_l = a * raw_tr_l + (1 - a) * self._tr_l
                self._tr_r = a * raw_tr_r + (1 - a) * self._tr_r
                self._sq_l = a * raw_sq_l + (1 - a) * self._sq_l
                self._sq_r = a * raw_sq_r + (1 - a) * self._sq_r
                left_trigger  = 0.0 if self._tr_l < 0.05 else self._tr_l
                right_trigger = 0.0 if self._tr_r < 0.05 else self._tr_r
                left_squeeze  = self._sq_l
                right_squeeze = self._sq_r

                # =========================
                # LÓGICA DE MOVIMENTO
                # =========================
                left_hand_q = np.zeros(7)
                right_hand_q = np.zeros(7)

                # Alvos de FECHAMENTO TOTAL = limites do Dex3 (g1_utils.DEX3_*_LIMITS).
                # Antes usava ±1.5 (genérico), que subutilizava o range real (até ±1.74)
                # e ainda clipava o thumb_1 → a mão só fechava ~85%. Agora bate o limite.
                # Ordem: [thumb_0(rot), thumb_1, thumb_2, index_0, index_1, middle_0, middle_1]
                LEFT_TARGET  = np.array([0.0,  0.920,  1.74, -1.57, -1.74, -1.57, -1.74])
                RIGHT_TARGET = np.array([0.0, -0.920, -1.74,  1.57,  1.74,  1.57,  1.74])

                # Grip completo
                left_hand_q  = left_squeeze  * LEFT_TARGET
                right_hand_q = right_squeeze * RIGHT_TARGET

                # =========================
                # PINÇA E ROTAÇÃO (AJUSTE FINO)
                # =========================
                PINCH_FORCE = 2.0
                PINCH_OFFSET = 0.2
                PINCH_OFFSET2 = 0.1

                LEFT_INDEX_ID  = 5   
                RIGHT_INDEX_ID = 5
   

                # Aplica movimento do indicador
                left_hand_q[5]   += -PINCH_FORCE * left_trigger
                right_hand_q[5] +=  PINCH_FORCE * right_trigger

                # Offset fixo
                left_hand_q[5]   += -PINCH_OFFSET * left_trigger
                right_hand_q[5] +=  PINCH_OFFSET * right_trigger

                # Aplica movimento do indicador
                left_hand_q[6]   += -PINCH_FORCE * left_trigger
                right_hand_q[6] +=  PINCH_FORCE * right_trigger

                # Offset fixo
                left_hand_q[6]   += -PINCH_OFFSET2 * left_trigger
                right_hand_q[6] +=  PINCH_OFFSET2 * right_trigger    

                # ROTAÇÃO DO POLEGAR
                # Nota: Inverti o sinal da mão direita para +0.5, pois mãos costumam ser espelhadas.
                # Se a mão direita passar a girar para o lado errado, pode voltar para -0.5.
                left_hand_q[0]  += -0.5 * left_trigger
                right_hand_q[0] +=  -0.5 * right_trigger 

                # CURVATURA EXTRA
                #left_hand_q[5]   += -0.5 * left_trigger
                #right_hand_q[5] +=  -0.5 * right_trigger 

                # Polegar acompanha pinça
                left_hand_q[1] += 0.8 * left_trigger
                left_hand_q[2] += 0.8 * left_trigger

                right_hand_q[1] -= 0.8 * right_trigger
                right_hand_q[2] -= 0.8 * right_trigger


                # =========================
                # APLICAÇÃO FINAL
                # Por padrão controla as DUAS mãos. Com --left-arm-limp, a esquerda
                # fica travada em aberto (q=0) e só a direita é controlada.
                # =========================
                if not _left_arm_limp():
                    for i, name in enumerate(self._left_hand_names):
                        self.hand_joints[f"{name}.q"] = left_hand_q[i]
                for i, name in enumerate(self._right_hand_names):
                    self.hand_joints[f"{name}.q"] = right_hand_q[i]

        # Concatena os dicionários e retorna a Ação Final
        action_data = {**self.body_joints, **self.hand_joints}
        return action_data
