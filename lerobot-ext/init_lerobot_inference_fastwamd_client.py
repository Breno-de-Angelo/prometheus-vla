#!/usr/bin/env python
"""
Cliente de Inferência — FastWAM-D (roda no SEU PC, com o robô)
===============================================================
O modelo fica 100% na athena (`init_lerobot_inference_fastwamd_server.py`).
Aqui só roda o loop de controle: lê o robô e as câmeras, manda a observação
pela rede, recebe o chunk de ações e executa. Nenhum peso é carregado — este
script precisa de Python, ZMQ, OpenCV e o robô.

Arquitetura (a mesma dos clientes antigos, e pelo mesmo motivo):

  processo filho : obs_queue → ZMQ → athena → action_queue + debug_queue
  processo pai   : action_queue → robot.send_action()   a 30 Hz

  O socket fica isolado no filho para que o RTT da rede (e os ~1,5 s de
  inferência de um modelo de 6 B) nunca segurem o GIL do loop de controle.

O que este cliente tem de novo em relação ao `init_lerobot_inference_client.py`:

  - manda as DUAS câmeras de cor (cabeça e pulso) e a profundidade métrica;
  - manda a frase da tarefa (o FastWAM é condicionado por linguagem);
  - chunk limitado a 32 ações (`action_horizon` do modelo, contra 60 do ACT-D);
  - painel de depuração com quatro quadrantes (`--v-debug`): atenção do DiT,
    profundidade crua × a que o modelo recebeu, nuvem de pontos e temperatura
    dos motores.

Banda: a observação vai crua, ~3 MB por inferência (duas câmeras 848x480 mais a
profundidade de 16 bits). A ~1 inferência por segundo isso é nada numa LAN, e
mandar cru evita que compressão com perda mude o que o modelo vê em relação ao
que ele viu no treino.

Uso:
  python init_lerobot_inference_fastwamd_client.py --server=<IP_ATHENA> [OPÇÕES]

Opções:
  --server=<IP>        (obrigatório) IP da athena
  --port=<INT>         porta do servidor (padrão: 5600)
  --sim                modo simulação (o robô vira 127.0.0.1; exige o MuJoCo no ar)
  --robot-ip=<IP>      IP do G1 real (padrão: 192.168.123.164). Ignorado com --sim.
  --replay=<PATH:EP>   ALIMENTA O MODELO COM UM EPISÓDIO GRAVADO em vez das
                       câmeras. Ex: --replay=meu_dataset/white_cup_on_dripper_2026-08-11:25
                       O robô (MuJoCo ou real) executa as ações resultantes, então
                       dá para VER o que o modelo faz com entrada que ele reconhece.
                       É o teste que separa "o modelo não aprendeu" de "o simulador
                       não se parece com o treino".
  --estado-robo        no replay, usa a propriocepção do ROBÔ em vez da gravada.
                       Padrão é usar a gravada: assim a observação inteira é
                       exatamente uma que o modelo viu no treino. Com esta flag
                       a imagem vem do dataset e as juntas do robô — mais
                       realista, mas se o robô divergir a observação vira uma
                       combinação que nunca existiu.
  --replay-uma-vez     encerra ao fim do episódio (padrão: repete em loop, para
                       dar tempo de olhar o painel e o robô sem ficar relançando)
  --rampa=<SEG>        tempo para sair da pose atual do robô até a primeira ação
                       do modelo (padrão: 2.0; 0 desliga). Os episódios começam
                       com os braços numa pose de prontidão (no ep 25, cotovelo
                       esquerdo a 1,378 rad) e o robô começa em zeros — sem a
                       rampa, a primeira ação é um salto instantâneo para lá.
  --sem-reducao        manda a observação crua (~3 MB) em vez de reduzida (~600 KB).
                       A redução não muda o que o modelo vê — ela só faz do lado
                       do cliente o mesmo resize que o servidor faria.
  --depth-legado       converte profundidade de 3 canais 0–255 de volta para mm.
                       Só para o servidor de câmera do MuJoCo, que ainda publica
                       no formato antigo. Perde resolução (~8 mm por degrau) —
                       serve para ver o caminho funcionando, não para medir.
  --cam-robot=<IP>     stream ZMQ de câmera externa
  --port-cam=<PORTA>   porta do stream (padrão: 5555)
  --task=<STR>         frase da tarefa; sem isto o servidor usa a dele
  --chunk=<INT>        ações por chunk (padrão: 32 = o horizonte do modelo)
  --lead=<INT>         pede nova inferência com N ações restantes (padrão: 24)
  --fps=<INT>          Hz do loop de controle (padrão: 30)
  --v-debug            abre o painel de depuração (os quatro quadrantes)
  --intrinsics=fx,fy,cx,cy   intrínsecos da câmera para a nuvem de pontos
                       (padrão: 617,617,424,240 — nominais da RealSense 848x480)
  --debug              loga ações e tempos
  --log                grava log em arquivo
  --log-path=<DIR>     pasta do log (padrão: ./logs/)
  -h, --help           esta mensagem

Exemplo:
  python init_lerobot_inference_fastwamd_client.py \\
      --server=10.9.8.252 --cam-robot=192.168.123.164 \\
      --chunk=32 --lead=24 --fps=30 --v-debug --debug
"""

import os
import sys
import time
import multiprocessing as mp
from datetime import datetime
from queue import Empty, Full

# ⚠️ O zmq vem ANTES de qualquer coisa que puxe torch (aqui, o
# `init_lerobot_inference_async_v2`). Ordem trocada = `Segmentation fault` sem
# mensagem nenhuma; ver a nota no topo do servidor.
try:
    import zmq
    import msgpack
    import msgpack_numpy as m
    m.patch()
except ImportError:
    print("❌ Instale as dependências: pip install pyzmq msgpack msgpack-numpy")
    sys.exit(1)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from init_lerobot_inference_async_v2 import setup_cameras, get_camera_frames  # noqa: E402
from viz_debug_fastwamd import INTRINSECOS_PADRAO, PainelDebug  # noqa: E402

# Mesma ordem do dataset e do servidor. Ver a nota lá: são 29 juntas, com o
# `kWaistYaw.q` no meio e as duas mãos em ordens de dedo diferentes.
JUNTAS_G1 = [
    "kLeftShoulderPitch.q", "kLeftShoulderRoll.q", "kLeftShoulderYaw.q",
    "kLeftElbow.q", "kLeftWristRoll.q", "kLeftWristPitch.q", "kLeftWristyaw.q",
    "kRightShoulderPitch.q", "kRightShoulderRoll.q", "kRightShoulderYaw.q",
    "kRightElbow.q", "kRightWristRoll.q", "kRightWristPitch.q", "kRightWristYaw.q",
    "kWaistYaw.q",
    "left_hand_thumb_0_joint.q", "left_hand_thumb_1_joint.q", "left_hand_thumb_2_joint.q",
    "left_hand_middle_0_joint.q", "left_hand_middle_1_joint.q",
    "left_hand_index_0_joint.q", "left_hand_index_1_joint.q",
    "right_hand_thumb_0_joint.q", "right_hand_thumb_1_joint.q", "right_hand_thumb_2_joint.q",
    "right_hand_index_0_joint.q", "right_hand_index_1_joint.q",
    "right_hand_middle_0_joint.q", "right_hand_middle_1_joint.q",
]

# O que viaja na observação. Tudo o que não está aqui fica no PC: mandar o
# `obs` inteiro do robô (dq, tau, IMU, pressão) triplicaria o payload sem o
# modelo usar nada disso.
CAMERAS = ("head_camera", "right_wrist_camera", "head_camera_depth")


def _pack(data: dict) -> bytes:
    return msgpack.packb(data, default=m.encode)


def _unpack(raw: bytes) -> dict:
    return msgpack.unpackb(raw, object_hook=m.decode, raw=False)


def reduz_para_a_rede(obs_rede: dict) -> dict:
    """Encolhe a observação sem mudar o que o modelo vê.

    As duas câmeras de cor são redimensionadas para 224x224 — exatamente o que o
    `_stack_video_from_images` faz do lado do servidor, com a MESMA interpolação
    (bilinear com antialias). Mandar 848x480 é jogar banda fora: o modelo
    descarta esses pixels no primeiro passo.

    A profundidade vai comprimida em PNG de 16 bits, que é SEM PERDA. Ela não é
    redimensionada aqui de propósito: o modelo normaliza (log) antes de
    redimensionar, e interpolar milímetros crus não é a mesma conta que
    interpolar o log deles — diferença pequena, mas que não precisa existir.

    Efeito: ~3 MB por requisição viram ~600 KB.
    """
    import torch

    reduzido = dict(obs_rede)
    for cam in ("head_camera", "right_wrist_camera"):
        quadro = reduzido.get(cam)
        if quadro is None or quadro.shape[:2] == (224, 224):
            continue
        t = torch.from_numpy(np.ascontiguousarray(quadro)).permute(2, 0, 1).unsqueeze(0).float()
        t = torch.nn.functional.interpolate(
            t, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
        )
        reduzido[cam] = t.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()

    profundidade = reduzido.pop("head_camera_depth", None)
    if profundidade is not None:
        import cv2

        ok, buf = cv2.imencode(".png", np.ascontiguousarray(profundidade.astype(np.uint16)))
        if ok:
            reduzido["head_camera_depth_png"] = buf.tobytes()
        else:
            reduzido["head_camera_depth"] = profundidade
    return reduzido


def converte_depth_legado(mapa: np.ndarray) -> np.ndarray:
    """Profundidade no formato ANTIGO (3 canais, 0–255) de volta para milímetros.

    O servidor de câmera do MuJoCo ainda publica assim — cinza replicado em três
    canais, com 0–2000 mm espremidos em 0–255. É perda de informação que não dá
    para desfazer: 255 degraus para 2 metros são ~8 mm por degrau, contra o
    milímetro do sensor real. Serve para VER o caminho funcionando no simulador,
    não para tirar conclusão sobre precisão de profundidade.
    """
    if mapa.ndim == 3:
        mapa = mapa[..., 0]
    return (mapa.astype(np.float32) * (2000.0 / 255.0)).astype(np.uint16)


def observacao_para_rede(obs: dict, depth_legado: bool = False) -> dict:
    """Só as juntas e as câmeras, em tipos que o msgpack entende."""
    saida: dict = {}
    for nome in JUNTAS_G1:
        saida[nome] = float(obs.get(nome, 0.0))
    for cam in CAMERAS:
        quadro = obs.get(cam)
        if quadro is None:
            continue
        if depth_legado and cam.endswith("depth"):
            quadro = converte_depth_legado(np.asarray(quadro))
        saida[cam] = np.ascontiguousarray(quadro)
    return saida


def le_temperaturas(robot) -> dict[str, float]:
    """Temperatura por junta, direto do `lowstate` do SDK.

    Vem do estado bruto e não do `get_observation()` de propósito: aquele dict
    é o mesmo que a gravação de dataset consome, e acrescentar chave nele
    mudaria o schema do que é gravado. Aqui é só telemetria de tela.
    """
    temperaturas: dict[str, float] = {}
    lowstate = getattr(robot, "_lowstate", None)
    if lowstate is None:
        return temperaturas

    try:
        for motor in robot.body_joint_index:
            valor = getattr(lowstate.motor_state[motor.value], "temperature", None)
            if valor is not None:
                temperaturas[motor.name] = float(valor)
    except Exception:  # noqa: BLE001 - telemetria nunca derruba o controle
        pass

    for lado, atributo in (("left", "_left_hand_state"), ("right", "_right_hand_state")):
        estado = getattr(robot, atributo, None)
        temp = getattr(estado, "temperature", None) if estado is not None else None
        if temp is None:
            continue
        try:
            valores = [float(v) for v in np.asarray(temp).ravel() if float(v) > 0.0]
            if valores:
                temperaturas[f"mao {lado} (max)"] = max(valores)
        except Exception:  # noqa: BLE001
            pass
    return temperaturas


def monta_mosaico(obs: dict) -> np.ndarray | None:
    """Cabeça e pulso lado a lado, do jeito que o modelo concatena.

    O FastWAM ordena as câmeras por nome (`_stack_video_from_images`), e
    `head_camera` < `right_wrist_camera` no alfabeto — por isso a cabeça fica à
    esquerda. Desenhar o painel em outra ordem faria o mapa de atenção apontar
    para a câmera errada.
    """
    import cv2

    quadros = []
    for cam in ("head_camera", "right_wrist_camera"):
        quadro = obs.get(cam)
        if quadro is None:
            continue
        quadros.append(cv2.resize(quadro, (224, 224), interpolation=cv2.INTER_AREA))
    if not quadros:
        return None
    return np.hstack(quadros) if len(quadros) > 1 else quadros[0]


class ReplayDeEpisodio:
    """Serve os quadros de um episódio gravado no lugar das câmeras.

    Para que serve: o render do MuJoCo é outra distribuição visual, e um modelo
    com pouco treino não transfere. Com o replay, a imagem que chega ao modelo é
    exatamente do tipo que ele viu no treino, e o robô do simulador executa as
    ações resultantes. Se ele se mexer direito aqui e não se mexer com a câmera
    do simulador, a diferença está na imagem — não no modelo nem no controle.

    A profundidade sai do dataset já em milímetros e 1 canal, pelo decodificador
    do LeRobot. Ler o mesmo vídeo com `cv2.VideoCapture` (o caminho do
    `--fake-depth`) devolveria 8 bits em 3 canais e destruiria a escala métrica.
    """

    def __init__(self, raiz: str, episodio: int, usar_estado_gravado: bool = True):
        # O `LeRobotDataset` consulta o Hub para resolver a versão do dataset
        # MESMO com `root` local, e aí um repo_id inventado devolve 401 e derruba
        # o replay. Offline é a verdade aqui: os dados estão todos em disco.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.ds = LeRobotDataset(repo_id=os.path.basename(os.path.normpath(raiz)),
                                 root=raiz, episodes=[episodio])
        self.episodio = episodio
        self.usar_estado_gravado = usar_estado_gravado
        self.i = 0
        self.n = len(self.ds)
        if self.n == 0:
            raise SystemExit(f"episódio {episodio} vazio em {raiz}")
        print(f"🎬 Replay: {raiz} episódio {episodio} — {self.n} quadros "
              f"({'estado gravado' if usar_estado_gravado else 'estado do robô'})")

    @staticmethod
    def _uint8(t) -> np.ndarray:
        a = t.numpy()
        if a.ndim == 3 and a.shape[0] in (1, 3):
            a = np.transpose(a, (1, 2, 0))
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        return a

    def aplica(self, obs: dict, em_loop: bool) -> tuple[dict, bool]:
        """Devolve (obs com os quadros gravados, acabou)."""
        if self.i >= self.n:
            if not em_loop:
                return obs, True
            self.i = 0

        amostra = self.ds[self.i]
        self.i += 1

        obs = dict(obs)
        obs["head_camera"] = self._uint8(amostra["observation.images.head_camera"])
        obs["right_wrist_camera"] = self._uint8(amostra["observation.images.right_wrist_camera"])
        obs["head_camera_depth"] = (
            amostra["observation.images.head_camera_depth"].numpy().squeeze().astype(np.uint16)
        )
        if self.usar_estado_gravado:
            estado = amostra["observation.state"].numpy()
            for k, nome in enumerate(JUNTAS_G1):
                obs[nome] = float(estado[k])
        return obs, False


# ═════════════════════════════════════════════════════════════════════════
# Processo de inferência remota
# ═════════════════════════════════════════════════════════════════════════
def _worker_remoto(server_ip, server_port, n_acoes, quer_debug, task, verbose, depth_legado, reduzir,
                   obs_queue, action_queue, debug_queue, log_queue, stop_evt):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 30000)
    socket.setsockopt(zmq.SNDTIMEO, 10000)
    endereco = f"tcp://{server_ip}:{server_port}"
    socket.connect(endereco)
    print(f"🔗 [inferência] conectado a {endereco}")

    while not stop_evt.is_set():
        try:
            obs, passo = obs_queue.get(timeout=0.5)
        except Empty:
            continue

        try:
            obs_rede = observacao_para_rede(obs, depth_legado)
            if reduzir:
                obs_rede = reduz_para_a_rede(obs_rede)
            pedido = {
                "obs": obs_rede,
                "obs_step": int(passo),
                "actions_per_chunk": int(n_acoes),
                "want_debug": bool(quer_debug),
            }
            if task:
                pedido["task"] = task

            t0 = time.perf_counter()
            socket.send(_pack(pedido))
            resposta = _unpack(socket.recv())
            rtt_ms = (time.perf_counter() - t0) * 1000.0

            if "error" in resposta:
                print(f"\n❌ [servidor] {resposta['error']}")
                continue

            chunk = np.asarray(resposta["chunk_np"], dtype=np.float32)
            action_queue.put((chunk, int(resposta.get("obs_step", passo))))
            try:
                log_queue.put_nowait((chunk, int(resposta.get("obs_step", passo))))
            except Full:
                pass

            if quer_debug:
                # O tempo do servidor viaja junto com o payload de debug (e
                # mesmo quando ele vem vazio), porque é o número que diz se o
                # `--chunk` está dimensionado: chunk tem que cobrir uma
                # inferência inteira, senão o buffer de ações zera no meio.
                payload = dict(resposta.get("debug") or {})
                payload["infer_ms"] = float(resposta.get("infer_ms", 0.0))
                payload["travadas"] = int(resposta.get("travadas", 0))
                try:
                    debug_queue.put_nowait(payload)
                except Full:
                    pass

            travadas = int(resposta.get("travadas", 0))
            if travadas:
                # Sinal de que a observação está fora do que o modelo viu no
                # treino (simulador, câmera preta, cena diferente). Não é ruído
                # de operação: é o aviso de que a ação seria absurda sem a trava.
                print(f"\n⚠️  {travadas} valores travados na faixa do dataset — "
                      "observação fora da distribuição de treino.")
            if verbose:
                print(f"\n📡 [inferência] {resposta.get('infer_ms', 0):.0f} ms no servidor "
                      f"| {rtt_ms:.0f} ms de ida e volta | chunk {chunk.shape}"
                      f"{f' | travadas {travadas}' if travadas else ''}")

        except zmq.Again:
            # Um REQ/REP que estourou o tempo fica com a máquina de estados
            # travada: sem recriar o socket, todo envio seguinte falha.
            print("\n⚠️  [inferência] servidor não respondeu — reconectando")
            socket.close()
            socket = ctx.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 30000)
            socket.setsockopt(zmq.SNDTIMEO, 10000)
            socket.connect(endereco)
        except Exception as erro:  # noqa: BLE001
            print(f"\n❌ [inferência] {type(erro).__name__}: {erro}")

    socket.close()
    ctx.term()


class ProcessoInferenciaRemota:
    def __init__(self, server_ip, server_port, n_acoes, quer_debug, task, verbose, depth_legado=False,
                 reduzir=True):
        self.obs_queue = mp.Queue(maxsize=1)
        self.action_queue = mp.Queue(maxsize=4)
        self.debug_queue = mp.Queue(maxsize=2)
        self.log_queue = mp.Queue(maxsize=64)
        self._stop = mp.Event()
        self._args = (server_ip, server_port, n_acoes, quer_debug, task, verbose, depth_legado, reduzir)
        self._proc: mp.Process | None = None

    def start(self):
        self._proc = mp.Process(
            target=_worker_remoto,
            args=(*self._args, self.obs_queue, self.action_queue,
                  self.debug_queue, self.log_queue, self._stop),
            daemon=True,
        )
        self._proc.start()

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        if self._proc is not None and self._proc.is_alive():
            self._proc.join(timeout=timeout)
            if self._proc.is_alive():
                self._proc.terminate()


# ═════════════════════════════════════════════════════════════════════════
# Principal
# ═════════════════════════════════════════════════════════════════════════
def main():
    if any(f in sys.argv for f in ["-h", "--help"]):
        print(__doc__)
        sys.exit(0)

    import cv2

    server_ip = None
    server_port = 5600
    is_sim = False
    robot_ip = "192.168.123.164"
    depth_legado = False
    replay_raiz = None
    replay_ep = 0
    replay_loop = True
    estado_do_robo = False
    reduzir = True
    rampa_s = 2.0
    cam_robot_ip = None
    cam_port = "5555"
    task = None
    n_acoes = 32
    lead = 24
    fps = 30
    ver_debug = False
    verbose = False
    log_ativo = False
    log_path = None
    intrinsecos = dict(INTRINSECOS_PADRAO)
    limite_de_mudanca = 10.0

    for arg in sys.argv[1:]:
        if arg.startswith("--server="):
            server_ip = arg.split("=", 1)[1]
        elif arg.startswith("--port="):
            server_port = int(arg.split("=", 1)[1])
        elif arg in ("--sim", "--simulation=true"):
            is_sim = True
        elif arg.startswith("--robot-ip="):
            robot_ip = arg.split("=", 1)[1]
        elif arg == "--depth-legado":
            depth_legado = True
        elif arg.startswith("--replay="):
            valor = arg.split("=", 1)[1]
            replay_raiz, _, ep = valor.rpartition(":")
            if not replay_raiz:
                replay_raiz, replay_ep = valor, 0
            else:
                replay_ep = int(ep)
        elif arg == "--estado-robo":
            estado_do_robo = True
        elif arg == "--replay-uma-vez":
            replay_loop = False
        elif arg == "--sem-reducao":
            reduzir = False
        elif arg.startswith("--rampa="):
            rampa_s = float(arg.split("=", 1)[1])
        elif arg.startswith("--cam-robot="):
            cam_robot_ip = arg.split("=", 1)[1]
        elif arg.startswith("--port-cam="):
            cam_port = arg.split("=", 1)[1]
        elif arg.startswith("--task="):
            task = arg.split("=", 1)[1]
        elif arg.startswith("--chunk="):
            n_acoes = int(arg.split("=", 1)[1])
        elif arg.startswith("--lead="):
            lead = int(arg.split("=", 1)[1])
        elif arg.startswith("--fps="):
            fps = int(arg.split("=", 1)[1])
        elif arg == "--v-debug":
            ver_debug = True
        elif arg.startswith("--intrinsics="):
            fx, fy, cx, cy = (float(v) for v in arg.split("=", 1)[1].split(","))
            intrinsecos = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        elif arg == "--debug":
            verbose = True
        elif arg == "--log":
            log_ativo = True
        elif arg.startswith("--log-path="):
            log_path = arg.split("=", 1)[1]
            log_ativo = True
        elif arg.startswith("--inconsistency="):
            limite_de_mudanca = float(arg.split("=", 1)[1])

    if server_ip is None:
        print("❌ ERRO: --server=<IP_DA_ATHENA> é obrigatório.")
        sys.exit(1)

    if n_acoes > 32:
        print(f"⚠️  --chunk={n_acoes} passa do horizonte do FastWAM-D (32). "
              "O servidor recorta; use 32 para não contar com ação que não existe.")
    lead = min(lead, n_acoes)
    loop_dt = 1.0 / fps

    arquivo_log = None
    nome_arquivo_log = None
    if log_ativo:
        pasta = log_path or "./logs"
        os.makedirs(pasta, exist_ok=True)
        nome_arquivo_log = os.path.join(
            pasta, f"fastwamd_{datetime.now():%Y%m%d_%H%M%S}.txt")
        arquivo_log = open(nome_arquivo_log, "w", buffering=1)
        arquivo_log.write(f"# fastwam-d remoto — servidor={server_ip}:{server_port}\n")
        arquivo_log.write(f"# chunk={n_acoes} lead={lead} fps={fps}\n---\n")
        print(f"📝 Log em: {nome_arquivo_log}")

    stream_client, fake_cap, fake_img_rgb, fake_depth_cap, fake_depth_img = setup_cameras(
        cam_robot_ip, cam_port, None, None
    )

    from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
    print(f"⏳ Conectando ao Unitree G1 (sim={is_sim}, ip={robot_ip})...")
    # Estes três TÊM que casar com o `config/record/*.yaml` que gravou o dataset.
    # Não é preferência: o `use_waist_yaw` define se o `kWaistYaw` entra no vetor
    # de ação E se o roll/pitch da cintura recebem ganho de travamento. Com ele
    # desligado, o yaw comandado é ignorado em silêncio (a cintura fica em zero)
    # e o tronco fica mole — que é a causa conhecida de o robô tombar para a
    # frente (ver o comentário no `unitree_g1.py`, na trava da cintura).
    cfg_robo = UnitreeG1Dex3Config(
        robot_ip=robot_ip,
        control_mode="upper_body",
        is_simulation=is_sim,
        use_waist_yaw=True,      # 29 dims: o dataset foi gravado assim
        use_wrist_camera=True,   # o FastWAM-D usa DUAS câmeras de cor
    )
    if replay_raiz:
        # No replay a imagem vem do dataset, então as câmeras do robô só
        # atrapalhariam: sem um servidor de imagem publicando, o
        # `get_observation()` estoura em TimeoutError antes de qualquer coisa.
        # É preciso limpar DEPOIS do __post_init__ — ele preenche o dicionário
        # justamente quando ele está vazio.
        cfg_robo.cameras = {}
        print("🎬 Replay ativo: câmeras do robô desligadas (a imagem vem do dataset)")
    robot = UnitreeG1Dex3(cfg_robo)
    robot.connect()
    print("✅ Robô conectado!")
    for cam in robot.cameras.values():
        if hasattr(cam, "timeout_ms"):
            cam.timeout_ms = 800

    # ── Conferência do espaço de ação e das câmeras ──────────────────────────
    # Antes de mandar a primeira ação. Junta que o robô não controla é aceita
    # sem reclamar e simplesmente não se move: o modelo pensa que comandou, o
    # robô fica com aquele eixo em zero, e o resultado é uma postura que nunca
    # existiu no dataset.
    try:
        conferencia = robot.get_observation() or {}
    except Exception as erro:  # noqa: BLE001
        print(f"❌ Não consegui ler uma observação do robô: {erro}")
        sys.exit(1)

    faltando_juntas = [n for n in JUNTAS_G1 if n not in conferencia]
    faltando_cameras = (
        [] if replay_raiz
        else [c for c in ("head_camera", "right_wrist_camera") if c not in robot.cameras]
    )
    if faltando_juntas or faltando_cameras:
        print("\n❌ O robô não expõe o que a política precisa — abortando ANTES de mandar ação.")
        if faltando_juntas:
            print(f"   juntas ausentes ({len(faltando_juntas)}): {faltando_juntas[:6]}"
                  f"{' ...' if len(faltando_juntas) > 6 else ''}")
            print("   → o dataset foi gravado com `use_waist_yaw: true` e 29 dims.")
        if faltando_cameras:
            print(f"   câmeras ausentes: {faltando_cameras}")
            print(f"   → configuradas: {sorted(robot.cameras)}")
        sys.exit(1)
    print(f"✅ {len(JUNTAS_G1)} juntas e {len(robot.cameras)} câmeras conferidas "
          f"({', '.join(sorted(robot.cameras))})")

    # A pose de onde a rampa parte é a do ROBÔ, lida agora. No replay o `obs` do
    # laço carrega o estado GRAVADO, que já é o destino — partir dele faria a
    # rampa não fazer nada.
    pose_inicial = np.array([float(conferencia.get(n, 0.0)) for n in JUNTAS_G1], dtype=np.float32)

    replay = None
    if replay_raiz:
        replay = ReplayDeEpisodio(replay_raiz, replay_ep,
                                  usar_estado_gravado=not estado_do_robo)

    painel = None
    if ver_debug:
        painel = PainelDebug(intrinsecos=intrinsecos)
        painel.create()

    inf = ProcessoInferenciaRemota(server_ip, server_port, n_acoes, ver_debug, task, verbose,
                                   depth_legado, reduzir)
    inf.start()

    print(f"🚀 Loop de controle — {fps} Hz | chunk={n_acoes} | lead={lead}")
    print(f"   Servidor: {server_ip}:{server_port}")
    print("   [Ctrl+C para parar]\n")

    passo_atual = 0
    chunks_ativos: list[dict] = []
    esperando = False
    ultima_acao = None
    soma_dt = 0.0
    n_ciclos = 0
    ultimo_infer_ms = 0.0
    ultimas_travadas = 0
    n_rampa = max(1, int(rampa_s * fps))

    try:
        while True:
            t_inicio = time.perf_counter()

            obs_valida = True
            try:
                obs = robot.get_observation()
            except TimeoutError as erro:
                print(f"\n⚠️  Timeout de câmera: {erro}. Mantendo ação do buffer.")
                obs, obs_valida = None, False
            if obs is not None and not obs:
                obs_valida = False

            if obs_valida and obs is not None:
                if replay is not None:
                    obs, acabou = replay.aplica(obs, replay_loop)
                    if acabou:
                        print(f"\n🎬 Episódio {replay.episodio} terminou "
                              f"({replay.n} quadros).")
                        break
                else:
                    obs, fake_img_rgb = get_camera_frames(
                        obs, stream_client, fake_cap, fake_img_rgb,
                        fake_depth_cap=fake_depth_cap, fake_depth_img=fake_depth_img,
                    )

            # ── Painel ────────────────────────────────────────────────────
            if painel is not None:
                if obs_valida and obs is not None:
                    prof = obs.get("head_camera_depth")
                    if prof is not None and depth_legado:
                        prof = converte_depth_legado(np.asarray(prof))
                    painel.define_imagens(rgb_mosaico=monta_mosaico(obs), depth_mm=prof)
                    painel.define_temperaturas(le_temperaturas(robot))
                try:
                    while True:
                        payload = inf.debug_queue.get_nowait()
                        ultimo_infer_ms = float(payload.get("infer_ms", ultimo_infer_ms))
                        ultimas_travadas = int(payload.get("travadas", 0))
                        painel.define_debug_servidor(payload)
                except Exception:
                    pass

            # ── Quantas ações ainda restam ────────────────────────────────
            restantes = 0
            if chunks_ativos:
                mais_novo = chunks_ativos[-1]
                restantes = (mais_novo["inicio"] + len(mais_novo["chunk"])) - passo_atual

            if obs_valida and restantes <= lead and not esperando:
                try:
                    inf.obs_queue.put_nowait((obs, passo_atual))
                    esperando = True
                except Full:
                    try:
                        inf.obs_queue.get_nowait()
                    except Empty:
                        pass
                    try:
                        inf.obs_queue.put_nowait((obs, passo_atual))
                        esperando = True
                    except Full:
                        pass

            if not inf.action_queue.empty():
                try:
                    chunk, passo_obs = inf.action_queue.get_nowait()
                    esperando = False
                    chunks_ativos.append({"inicio": passo_obs, "chunk": chunk})
                except (Empty, ValueError):
                    pass

            chunks_ativos = [
                c for c in chunks_ativos
                if (c["inicio"] + len(c["chunk"])) > passo_atual
            ]

            # ── Ensembling temporal ───────────────────────────────────────
            if chunks_ativos:
                candidatas = []
                for c in chunks_ativos:
                    i = passo_atual - c["inicio"]
                    if 0 <= i < len(c["chunk"]):
                        candidatas.append(c["chunk"][i])

                if candidatas:
                    pesos = np.exp(-0.1 * np.arange(len(candidatas)))
                    pesos /= pesos.sum()
                    acao = np.sum(np.array(candidatas) * pesos[:, None], axis=0)

                    # ── Rampa de entrada ──────────────────────────────
                    # A primeira ação do modelo é a pose em que a demonstração
                    # começa, que não é onde o robô está. Sem interpolar, ele
                    # salta — feio no simulador e perigoso no robô real, onde um
                    # degrau de posição vira torque de pico.
                    if rampa_s > 0 and passo_atual < n_rampa and pose_inicial is not None:
                        t = (passo_atual + 1) / n_rampa
                        acao = (1.0 - t) * pose_inicial + t * acao
                        if passo_atual == 0:
                            print(f"\n🛗 Rampa de {rampa_s:.1f} s da pose atual até a do modelo "
                                  f"({n_rampa} ciclos).")

                    if ultima_acao is not None:
                        salto = float(np.max(np.abs(acao - ultima_acao)))
                        if salto > limite_de_mudanca:
                            if verbose:
                                print(f"\n⚠️  Salto de {salto:.2f} rad entre ações — suavizando")
                            acao = 0.5 * acao + 0.5 * ultima_acao
                    ultima_acao = acao.copy()
                    passo_atual += 1

                    if arquivo_log is not None:
                        vals = "\t".join(f"{v:.6f}" for v in acao.tolist())
                        arquivo_log.write(f"EXECUTED\t{passo_atual}\t{time.time():.6f}\t{vals}\n")

                    if verbose:
                        braco = " | ".join(f"{v:.3f}" for v in acao[:7])
                        print(f"\r🤖 [{braco}] chunks={len(chunks_ativos)}", end="", flush=True)

                    robot.send_action({nome: float(acao[i]) for i, nome in enumerate(JUNTAS_G1)})
            elif verbose:
                print("\r⏳ Aguardando o servidor...", end="", flush=True)

            # ── Cadência e diagnóstico ────────────────────────────────────
            decorrido = time.perf_counter() - t_inicio
            soma_dt += decorrido
            n_ciclos += 1
            if n_ciclos >= 100:
                dt_ms = soma_dt / n_ciclos * 1000
                if verbose:
                    print(f"\n📊 Loop dt médio: {dt_ms:.1f} ms ({1000 / dt_ms:.1f} Hz)")
                soma_dt, n_ciclos = 0.0, 0

            if painel is not None:
                painel.define_cabecalho(
                    (f"replay ep{replay.episodio} {replay.i}/{replay.n} | " if replay else "")
                    + f"FastWAM-D | passo {passo_atual} | chunks {len(chunks_ativos)} | "
                    f"restam {max(restantes, 0)} | servidor {ultimo_infer_ms:.0f} ms"
                    + (f" | TRAVADAS {ultimas_travadas}" if ultimas_travadas else "")
                )
                if not painel.show():
                    raise KeyboardInterrupt

            sobra = loop_dt - (time.perf_counter() - t_inicio)
            if sobra > 0:
                time.sleep(sobra)

    except KeyboardInterrupt:
        print("\n🛑 Parando...")

    finally:
        inf.stop(timeout=5.0)
        if arquivo_log is not None and not arquivo_log.closed:
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
        if painel is not None:
            painel.destroy()
        cv2.destroyAllWindows()
        print("✅ Encerrado com segurança.")

        # `os._exit` em vez de deixar o interpretador desmontar sozinho: com
        # zmq, torch, mujoco e OpenCV/Qt no mesmo processo, a ordem de descarga
        # das bibliotecas C++ produz um `Segmentation fault (core dumped)` DEPOIS
        # desta linha. Não é falha de execução — o trabalho já terminou e tudo
        # foi fechado acima —, mas assusta e polui o terminal com core dump.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=False)
    main()
