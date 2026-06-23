#!/usr/bin/env python
"""
Inference Entry Point — Remote (Cliente ZMQ)
=============================================
Roda no seu PC local. Conecta ao servidor de inferência (Atena/A100),
envia observações e recebe chunks de ação para controlar o robô.

O modelo NÃO é carregado aqui — fica 100% na Atena.
Este script só precisa: Python + ZMQ + robô.

Arquitetura:
  Processo filho (RemoteInferenceProcess): obs_queue → ZMQ → Atena → action_queue
  Processo pai  (main/control loop):       action_queue → robot.send_action()

  O socket ZMQ fica isolado no processo filho — sem GIL compartilhado com o
  loop de controle, mesmo que o RTT para a Atena seja longo (100-200ms+).

Uso:
  python init_lerobot_inference_remote_v2.py --server=<IP> [OPÇÕES]

Opções:
  --server=<IP>          (obrigatório) IP da Atena com o servidor de inferência
  --port=<INT>           Porta do servidor (padrão: 5600)
  --sim                  Modo simulação (sem robô real)
  --cam-robot=<IP>       Stream ZMQ de câmera externa (OBRIGATÓRIO para --v
                         funcionar — sem isso e sem --fake-video, não existe
                         nenhuma fonte de imagem e a janela fica preta)
  --port-cam=<PORTA>     Porta do stream de câmera (padrão: 5555)
  --fake-video=<PATH>    Injeta imagem ou vídeo na câmera RGB
  --fake-depth=<PATH>    Injeta vídeo ou imagem de depth
  --chunk=<INT>          Ações por chunk (padrão: 60)
  --lead=<INT>           Pede nova inferência com N ações restantes (padrão: 50)
  --fps=<INT>            Hz do loop de controle (padrão: 30)
  --debug                Loga ações e tempos no terminal
  --log                  Ativa gravação do log em arquivo .txt
  --log-path=<DIR>       Pasta do log (padrão: ./logs/)
  --v                    Abre janela de visualização da câmera.
                         Requer --cam-robot=<IP> ou --fake-video=<PATH>.
  --v-control            Abre janela com controles de player
  --v-attn               Abre janela de attention map. O SERVIDOR (Atena)
                         precisa enviar o payload de atenção na resposta
                         ZMQ (campo "attn"); se o servidor não suportar,
                         a janela mostra "Aguardando dados do servidor...".
  -h, --help             Mostra esta mensagem

Exemplos:
  # Servidor na Atena (192.168.1.100):
  python init_lerobot_inference_remote_v2.py \\
      --server=192.168.1.100 --port=5600 \\
      --chunk=100 --lead=80 --fps=30 --debug

  # Com câmera ZMQ e visualização:
  python init_lerobot_inference_remote_v2.py \\
      --server=192.168.1.100 \\
      --cam-robot=192.168.123.164 \\
      --chunk=100 --lead=80 --fps=30 --v --v-attn
"""

import os
import sys
import time
import multiprocessing as mp
from queue import Empty, Full
from datetime import datetime

import numpy as np
import cv2

# ── ZMQ + msgpack ──────────────────────────────────────────────────────
try:
    import zmq
    import msgpack
    import msgpack_numpy as m
    m.patch()
except ImportError:
    print("❌ Instale as dependências: pip install pyzmq msgpack msgpack-numpy")
    sys.exit(1)

# ── Reutiliza helpers do script original ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from init_lerobot_inference_async_v2 import (
        VideoControlWindow,
        AttnMapWindow,
        setup_cameras,
        get_camera_frames,
    )
except ImportError as e:
    print(f"❌ Não consegui importar init_lerobot_inference_async_v2: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────
# Helpers de serialização
# ─────────────────────────────────────────────────────────────────────

def _pack(data: dict) -> bytes:
    return msgpack.packb(data, default=m.encode)

def _unpack(raw: bytes) -> dict:
    return msgpack.unpackb(raw, object_hook=m.decode, raw=False)


def _obs_to_serializable(obs: dict) -> dict:
    """
    Converte o dict de observação para tipos serializáveis via msgpack.
    Numpy arrays são mantidos (msgpack_numpy cuida deles).
    Outros tipos são convertidos para lista.
    """
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = np.array(v, dtype=np.float32)
        elif isinstance(v, (int, float)):
            out[k] = float(v)
        elif isinstance(v, str):
            out[k] = v
        else:
            try:
                out[k] = np.array(v, dtype=np.float32)
            except Exception:
                pass  # ignora chaves não-serializáveis
    return out


# ─────────────────────────────────────────────────────────────────────
# Processo de inferência remota (multiprocessing — ZMQ isolado do GIL)
# ─────────────────────────────────────────────────────────────────────

def _remote_process_worker(
    obs_queue: mp.Queue,
    action_queue: mp.Queue,
    log_queue: mp.Queue,
    attn_queue: mp.Queue,
    stop_event: mp.Event,
    server_ip: str,
    server_port: int,
    actions_per_chunk: int,
    want_attn: bool,
    debug: bool,
):
    """
    Roda em PROCESSO separado — o socket ZMQ fica isolado do loop de controle.
    Comunicação com o pai via mp.Queue (apenas numpy/python puro, picklable).
    """
    import zmq, msgpack, msgpack_numpy as m_np
    m_np.patch()
    import numpy as _np

    def _connect():
        s = ctx.socket(zmq.REQ)
        # LINGER=0: ao fechar, descarta mensagens pendentes na hora em vez
        # de travar esperando elas saírem — importante para reconectar rápido.
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, 15_000)
        s.setsockopt(zmq.SNDTIMEO, 5_000)
        s.connect(f"tcp://{server_ip}:{server_port}")
        return s

    print(f"🔌 [remote_proc] Conectando ao servidor {server_ip}:{server_port}...")
    ctx     = zmq.Context()
    socket  = _connect()
    print("✅ [remote_proc] Conectado!")

    req_count  = 0
    t_total_ms = 0.0
    _attn_warned = False

    while not stop_event.is_set():
        try:
            queue_item = obs_queue.get(timeout=0.5)
        except Exception:
            continue

        obs, obs_step = queue_item if isinstance(queue_item, tuple) else (queue_item, 0)

        t0 = time.perf_counter()
        try:
            obs_serial = _obs_to_serializable(obs)
            msg = {
                "obs": obs_serial,
                "obs_step": obs_step,
                "actions_per_chunk": actions_per_chunk,
                # Pede ao servidor para incluir o payload de atenção na
                # resposta. Servidores antigos que não conhecem esse campo
                # simplesmente o ignoram — não quebra nada.
                "want_attn": want_attn,
            }
            socket.send(msgpack.packb(msg, default=m_np.encode))

            t_sent = time.perf_counter()
            raw  = socket.recv()
            t_recv = time.perf_counter()
            resp = msgpack.unpackb(raw, object_hook=m_np.decode, raw=False)
            t_unpack = time.perf_counter()

            if debug:
                recv_ms   = (t_recv - t_sent) * 1000
                unpack_ms = (t_unpack - t_recv) * 1000
                if unpack_ms > 200:
                    print(f"\n⚠️  [remote_proc] unpack demorou {unpack_ms:.0f}ms "
                          f"({len(raw)/1e6:.1f}MB recebidos) — payload grande, "
                          f"considere reduzir actions_per_chunk ou checar se "
                          f"attn_np está sendo mandado sem .tolist() no servidor.")

            if "error" in resp:
                print(f"\n❌ [remote_proc] Servidor reportou erro: {resp['error']}")
                continue

            chunk_np     = [_np.array(a, dtype=_np.float32) for a in resp["chunk_np"]]
            obs_step_ret = int(resp["obs_step"])

            t_rtt_ms    = (time.perf_counter() - t0) * 1000
            req_count  += 1
            t_total_ms += t_rtt_ms

            if debug:
                avg = t_total_ms / req_count
                print(f"\n🌐 [remote] RTT={t_rtt_ms:.1f}ms | chunk={len(chunk_np)} | avg={avg:.1f}ms")

            # Envia chunk bruto para o log do pai
            try:
                log_queue.put_nowait((chunk_np, obs_step_ret))
            except Exception:
                pass

            # Envia chunk para o loop de controle
            try:
                action_queue.put_nowait((chunk_np, obs_step_ret))
            except Exception:
                try: action_queue.get_nowait()
                except Exception: pass
                action_queue.put_nowait((chunk_np, obs_step_ret))

            # Payload de atenção, se pedido e se o servidor suportar
            if want_attn:
                attn_raw = resp.get("attn")
                if attn_raw is not None:
                    payload = {
                        "attn_np":      _np.array(attn_raw["attn_np"], dtype=_np.float32),
                        "rgb_frame":    _np.array(attn_raw["rgb_frame"], dtype=_np.uint8),
                        "action_chunk": chunk_np,
                        # Geometria de tokens (PI05DEPTH) — ausente para
                        # policies com layout fixo como ACT. update_from_data
                        # trata None como "usa geometria ACT clássica".
                        "meta": attn_raw.get("meta"),
                    }
                    try:
                        if attn_queue.full():
                            try: attn_queue.get_nowait()
                            except Exception: pass
                        attn_queue.put_nowait(payload)
                    except Exception:
                        pass
                elif not _attn_warned:
                    print("\n⚠️  [remote_proc] --v-attn pedido, mas o servidor não "
                          "enviou o campo 'attn' na resposta. Verifique se o "
                          "servidor de inferência suporta want_attn.")
                    _attn_warned = True

        except zmq.Again:
            # CRÍTICO: depois de um timeout, o socket REQ fica num estado
            # interno travado (a máquina de estados send→recv→send→recv foi
            # quebrada). Reenviar no MESMO socket trava para sempre — é
            # exatamente o que causava "servidor parece morto" depois do
            # primeiro timeout. A única forma confiável de recuperar é
            # fechar e recriar o socket.
            print(f"\n⏱️  [remote_proc] Timeout aguardando servidor (>{15}s). "
                  f"Reconectando socket...")
            try:
                socket.close()
            except Exception:
                pass
            socket = _connect()
            print("🔌 [remote_proc] Socket reconectado.")
        except Exception as e:
            import traceback
            print(f"\n❌ [remote_proc] Erro: {e}")
            traceback.print_exc()
            # Qualquer erro inesperado também pode ter deixado o socket REQ
            # fora de sincronia — reconecta por segurança.
            try:
                socket.close()
            except Exception:
                pass
            socket = _connect()

    socket.close()
    ctx.term()
    print("🔌 [remote_proc] Desconectado.")


class RemoteInferenceProcess:
    """
    Substitui inf_thread (threading.Thread) por um processo filho real.

    Interface pública:
      .obs_queue    → mp.Queue — envia (obs, obs_step) para o filho
      .action_queue → mp.Queue — recebe (chunk_np, obs_step) do filho
      .log_queue    → mp.Queue — recebe (chunk_np, obs_step) para gravar no log
      .attn_queue   → mp.Queue — recebe payload de atenção (só se want_attn=True
                       e o servidor suportar) para AttnMapWindow
      .start()      → lança o processo filho
      .stop(timeout)→ sinaliza parada e aguarda
    """

    def __init__(
        self,
        *,
        server_ip: str,
        server_port: int,
        actions_per_chunk: int,
        want_attn: bool = False,
        debug: bool,
    ):
        self._kwargs = dict(
            server_ip=server_ip,
            server_port=server_port,
            actions_per_chunk=actions_per_chunk,
            want_attn=want_attn,
            debug=debug,
        )
        self.obs_queue    = mp.Queue(maxsize=1)
        self.action_queue = mp.Queue(maxsize=2)
        self.log_queue    = mp.Queue(maxsize=50)
        self.attn_queue   = mp.Queue(maxsize=2)
        self._stop_event  = mp.Event()
        self._proc: mp.Process | None = None

    def start(self):
        self._proc = mp.Process(
            target=_remote_process_worker,
            args=(
                self.obs_queue,
                self.action_queue,
                self.log_queue,
                self.attn_queue,
                self._stop_event,
            ),
            kwargs=self._kwargs,
            daemon=True,
            name="remote_inference_proc",
        )
        self._proc.start()

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        if self._proc is not None and self._proc.is_alive():
            self._proc.join(timeout=timeout)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2.0)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    if any(f in sys.argv for f in ["-h", "--help"]):
        print(__doc__)
        sys.exit(0)

    # ── Args ──────────────────────────────────────────────────────────
    server_ip         = None
    server_port       = 5600
    is_sim            = False
    fake_video_path   = None
    fake_depth_path   = None
    cam_robot_ip      = None
    cam_port          = "5555"
    debug_mode        = False
    show_video        = False
    show_video_control = False
    show_attn         = False
    actions_per_chunk = 60
    lead_actions      = 50
    fps               = 30
    log_enabled       = False
    log_path          = None
    limite_de_mudanca = 10.0  # limite de mudança entre ações consecutivas para detectar inconsistências

    for arg in sys.argv[1:]:
        if arg.startswith("--server="):
            server_ip = arg.split("=", 1)[1]
        elif arg.startswith("--port="):
            server_port = int(arg.split("=", 1)[1])
        elif arg in ["--sim", "--simulation=true"]:
            is_sim = True
        elif arg.startswith("--fake-video="):
            fake_video_path = arg.split("=", 1)[1]
        elif arg.startswith("--fake-depth="):
            fake_depth_path = arg.split("=", 1)[1]
        elif arg.startswith("--cam-robot="):
            cam_robot_ip = arg.split("=", 1)[1]
        elif arg.startswith("--port-cam="):
            cam_port = arg.split("=", 1)[1]
        elif arg.startswith("--chunk="):
            actions_per_chunk = int(arg.split("=", 1)[1])
        elif arg.startswith("--lead="):
            lead_actions = int(arg.split("=", 1)[1])
        elif arg.startswith("--fps="):
            fps = int(arg.split("=", 1)[1])
        elif arg == "--debug":
            debug_mode = True
        elif arg == "--log":
            log_enabled = True
        elif arg.startswith("--log-path="):
            log_path = arg.split("=", 1)[1]
            log_enabled = True
        elif arg == "--v":
            show_video = True
        elif arg == "--v-control":
            show_video_control = True
            show_video = True
        elif arg == "--v-attn":
            show_attn = True
        elif arg.startswith("--inconsistency="):
            limite_de_mudanca = float(arg.split("=", 1)[1])

    if server_ip is None:
        print("❌ ERRO: --server=<IP_DA_ATENA> é obrigatório.")
        sys.exit(1)

    if show_video and cam_robot_ip is None and fake_video_path is None:
        print("⚠️  --v pedido, mas nenhuma fonte de câmera foi configurada "
              "(falta --cam-robot=<IP> ou --fake-video=<PATH>).")
        print("   A janela vai abrir, mas vai ficar PRETA porque "
              "obs['head_camera'] nunca é preenchido.")

    lead_actions = min(lead_actions, actions_per_chunk)
    loop_dt      = 1.0 / fps

    # ── Log ───────────────────────────────────────────────────────────
    arquivo_log       = None
    nome_arquivo_log  = None
    _chunk_id_counter = 0

    if log_enabled:
        log_dir = log_path or "./logs"
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo_log = os.path.join(log_dir, f"log_{ts}.txt")
        arquivo_log = open(nome_arquivo_log, "w", buffering=1)
        arquivo_log.write(f"# remote inference log — server={server_ip}:{server_port}\n")
        arquivo_log.write(f"# chunk={actions_per_chunk} lead={lead_actions} fps={fps}\n")
        arquivo_log.write("---\n")
        print(f"📝 Log em: {nome_arquivo_log}")

    # ── Câmeras ───────────────────────────────────────────────────────
    stream_client, fake_cap, fake_img_rgb, fake_depth_cap, fake_depth_img = setup_cameras(
        cam_robot_ip, cam_port, fake_video_path, fake_depth_path
    )

    # ── Robô ──────────────────────────────────────────────────────────
    from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
    print(f"⏳ Conectando ao Unitree G1 (sim={is_sim})...")
    g1_config = UnitreeG1Dex3Config(
        #robot_ip="192.168.123.164",
        robot_ip="127.0.0.1",
        control_mode="upper_body",
        is_simulation=is_sim,
    )
    robot = UnitreeG1Dex3(g1_config)
    robot.connect()
    print("✅ Robô conectado!")

    for cam in robot.cameras.values():
        if hasattr(cam, "timeout_ms"):
            cam.timeout_ms = 800

    joint_names = [
        "kLeftShoulderPitch.q",  "kLeftShoulderRoll.q",  "kLeftShoulderYaw.q",
        "kLeftElbow.q",          "kLeftWristRoll.q",      "kLeftWristPitch.q",
        "kLeftWristyaw.q",
        "kRightShoulderPitch.q", "kRightShoulderRoll.q", "kRightShoulderYaw.q",
        "kRightElbow.q",         "kRightWristRoll.q",     "kRightWristPitch.q",
        "kRightWristYaw.q",
        "left_hand_thumb_0_joint.q",  "left_hand_thumb_1_joint.q",
        "left_hand_thumb_2_joint.q",  "left_hand_middle_0_joint.q",
        "left_hand_middle_1_joint.q", "left_hand_index_0_joint.q",
        "left_hand_index_1_joint.q",
        "right_hand_thumb_0_joint.q", "right_hand_thumb_1_joint.q",
        "right_hand_thumb_2_joint.q", "right_hand_index_0_joint.q",
        "right_hand_index_1_joint.q", "right_hand_middle_0_joint.q",
        "right_hand_middle_1_joint.q",
    ]

    # ── Janela OpenCV ─────────────────────────────────────────────────
    vc = None
    if show_video_control:
        vc = VideoControlWindow("Visao da IA — Remote")
        vc.create()

    aw = None
    if show_attn:
        # use_depth aqui é só cosmético (rótulos/histórico do painel) — o
        # fatiamento real do heatmap usa o campo "meta" vindo do servidor
        # quando presente (geometria PI05DEPTH), senão a geometria fixa ACT.
        # True é o default seguro: os checkpoints atuais (actdepth, pi05depth)
        # sempre treinam com depth_3d ativo.
        aw = AttnMapWindow(
            "Attention Map — Remote",
            use_depth=False,
            use_vae=False,
            action_dim=len(joint_names),
        )
        aw.create()

    # ── Processo de inferência remota (multiprocessing — ZMQ isolado) ──
    inf_proc = RemoteInferenceProcess(
        server_ip=server_ip,
        server_port=server_port,
        actions_per_chunk=actions_per_chunk,
        want_attn=show_attn,
        debug=debug_mode,
    )
    inf_proc.start()

    # Aliases para o código do loop (mesma interface de antes)
    obs_queue         = inf_proc.obs_queue
    action_queue      = inf_proc.action_queue
    network_log_queue = inf_proc.log_queue   # chunks brutos → log

    print(f"🚀 Loop de controle iniciado — {fps} Hz | chunk={actions_per_chunk} | lead={lead_actions}")
    print(f"   Servidor de inferência: {server_ip}:{server_port}")
    print("   [Ctrl+C para parar]\n")

    # ── Loop de controle (idêntico ao original) ───────────────────────
    current_step          = 0
    active_chunks         = []
    waiting_for_inference = False
    _diag_elapsed_sum     = 0.0
    _diag_loops           = 0
    _last_action          = None

    try:
        while True:
            start_t = time.perf_counter()

            # 1. Lê observação do robô / câmera
            obs_valid = True
            try:
                obs = robot.get_observation()
            except TimeoutError as e:
                print(f"\n⚠️  Timeout de câmera: {e}. Mantendo ação do buffer.")
                obs_valid = False
                obs = None
            
            if obs is not None and not obs:
                obs_valid = False

            if obs_valid and obs is not None:
                obs, fake_img_rgb = get_camera_frames(
                    obs, stream_client, fake_cap, fake_img_rgb,
                    fake_depth_cap=fake_depth_cap,
                    fake_depth_img=fake_depth_img,
                )

            # 2. Visualização (opcional)
            if show_video and obs_valid and obs is not None:
                rgb = obs.get("head_camera")
                if rgb is not None:
                    if vc is not None:
                        frame = vc.process(rgb)
                        alive = vc.show(frame)
                        if not alive:
                            raise KeyboardInterrupt
                    else:
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Camera — Remote Inference", bgr)
                        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                            raise KeyboardInterrupt

            # 3. Calcula ações restantes
            actions_left = 0
            if active_chunks:
                newest = active_chunks[-1]
                actions_left = (newest["start_step"] + len(newest["chunk"])) - current_step

            # 4. Alimenta fila de inferência
            if obs_valid and (actions_left <= lead_actions) and not waiting_for_inference:
                try:
                    obs_queue.put_nowait((obs, current_step))
                    waiting_for_inference = True
                except Full:
                    try:
                        obs_queue.get_nowait()
                    except Empty:
                        pass
                    obs_queue.put_nowait((obs, current_step))
                    waiting_for_inference = True

            # 5. Recebe novo chunk
            if not action_queue.empty():
                try:
                    new_chunk, obs_step = action_queue.get_nowait()
                    waiting_for_inference = False
                    # IMPORTANTE: usa current_step (agora), não obs_step (quando a
                    # obs foi capturada). A inferência remota tem latência de rede +
                    # GPU; se indexarmos pelo obs_step, o chunk chega "atrasado" e os
                    # primeiros N steps dele são pulados de uma vez — isso é o que
                    # causa os trancos/movimentos estranhos nos boundaries de chunk.
                    active_chunks.append({
                        "start_step": obs_step,
                        "chunk": new_chunk,
                    })
                except (Empty, ValueError):
                    pass

            # Limpa chunks antigos
            active_chunks = [
                c for c in active_chunks
                if (c["start_step"] + len(c["chunk"])) > current_step
            ]

            # 6. Ensembling ACT
            if active_chunks:
                actions_for_now = []
                for c in active_chunks:
                    idx = current_step - c["start_step"]
                    if 0 <= idx < len(c["chunk"]):
                        actions_for_now.append(c["chunk"][idx])

                if actions_for_now:
                    k = 0.1
                    exp_weights = np.exp(-k * np.arange(len(actions_for_now)))
                    exp_weights = exp_weights / exp_weights.sum()
                    action_numpy = np.sum(
                        np.array(actions_for_now) * exp_weights[:, None], axis=0
                    )

                    # Guarda para proteção de inconsistência
                    if _last_action is not None:
                        diff = np.max(np.abs(action_numpy - _last_action))
                        if diff > limite_de_mudanca:
                            if debug_mode:
                                print(f"\n⚠️  Inconsistência detectada: diff={diff:.3f} > {limite_de_mudanca} — suavizando")
                            action_numpy = 0.5 * action_numpy + 0.5 * _last_action
                    _last_action = action_numpy.copy()

                    current_step += 1

                    # Log
                    if arquivo_log is not None:
                        vals = "\t".join(f"{v:.6f}" for v in action_numpy.tolist())
                        arquivo_log.write(
                            f"EXECUTED\t{current_step}\t{time.time():.6f}\t{vals}\n"
                        )
                        if log_enabled:
                            while True:
                                try:
                                    raw_chunk, obs_step_net = network_log_queue.get_nowait()
                                    _chunk_id_counter += 1
                                    for si, row in enumerate(raw_chunk):
                                        rvals = "\t".join(f"{v:.6f}" for v in row.tolist())
                                        arquivo_log.write(
                                            f"NETWORK\t{_chunk_id_counter}\t{si}\t{obs_step_net}\t{rvals}\n"
                                        )
                                except Exception:
                                    break

                    if debug_mode:
                        arm = " | ".join([f"{v:.3f}" for v in action_numpy[:7]])
                        print(f"\r🤖 [remote] E: [{arm}] chunks_ativos={len(active_chunks)}", end="", flush=True)

                    action_dict = {name: float(action_numpy[i]) for i, name in enumerate(joint_names)}
                    robot.send_action(action_dict)

            else:
                if debug_mode:
                    print("\r⏳ Aguardando servidor...", end="", flush=True)

            # 7. Diagnóstico periódico
            elapsed = time.perf_counter() - start_t
            _diag_elapsed_sum += elapsed
            _diag_loops += 1
            if _diag_loops >= 100:
                avg_dt_ms = (_diag_elapsed_sum / _diag_loops) * 1000
                if debug_mode:
                    print(f"\n📊 Loop dt médio: {avg_dt_ms:.1f}ms ({1000/avg_dt_ms:.1f} Hz)")
                _diag_loops = 0
                _diag_elapsed_sum = 0.0

            # 8. Drena attn_queue → atualiza AttnMapWindow (payload vem do
            #    processo filho, que só preenche se o servidor enviar "attn")
            if aw is not None:
                try:
                    while True:
                        payload = inf_proc.attn_queue.get_nowait()
                        aw.update_from_data(payload)
                except Exception:
                    pass

            # 9. Exibe janela de attention map
            if aw is not None:
                alive = aw.show()
                if not alive:
                    raise KeyboardInterrupt

            # Mantém cadência
            sleep_t = loop_dt - (time.perf_counter() - start_t)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n🛑 Parando...")

    finally:
        inf_proc.stop(timeout=5.0)

        if arquivo_log is not None and not arquivo_log.closed:
            try:
                while True:
                    raw_chunk, obs_step_net = network_log_queue.get_nowait()
                    _chunk_id_counter += 1
                    for si, row in enumerate(raw_chunk):
                        rvals = "\t".join(f"{v:.6f}" for v in row.tolist())
                        arquivo_log.write(
                            f"NETWORK\t{_chunk_id_counter}\t{si}\t{obs_step_net}\t{rvals}\n"
                        )
            except Exception:
                pass
            arquivo_log.write("---END---\n")
            arquivo_log.close()
            print(f"💾 Log salvo em: {nome_arquivo_log}")

        if fake_cap is not None:
            fake_cap.release()
        if fake_depth_cap is not None:
            fake_depth_cap.release()
        if stream_client is not None:
            stream_client.stop_client()
        robot.disconnect()
        if vc is not None:
            vc.destroy()
        if aw is not None:
            aw.destroy()
        if show_video:
            cv2.destroyAllWindows()
        print("✅ Encerrado com segurança.")


if __name__ == "__main__":
    # "spawn" isola o processo filho do estado do pai (CUDA, sockets, etc.)
    mp.set_start_method("spawn", force=False)
    main()