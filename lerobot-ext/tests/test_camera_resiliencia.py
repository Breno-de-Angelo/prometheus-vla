"""Testes do `_read_camera`: engasgo de rede não pode derrubar a teleoperação.

Sobem um servidor ZMQ falso (mesmo protocolo `zmq.raw.v1` do servidor de imagem do
robô) e exercitam o caminho real — `ZMQCamera` de verdade, `UnitreeG1._read_camera` de
verdade —, inclusive os modos de falha, que são justamente os que nunca dá para
provocar de propósito com o robô na frente.

Rodar de `lerobot-ext/`:

    conda activate prometheus-vla
    python -m pytest tests/test_camera_resiliencia.py -v
"""

import json
import threading
import time

import numpy as np
import pytest
import zmq

from lerobot.cameras.zmq.camera_zmq import ZMQCamera
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig

from robot.unitree_g1.config_unitree_g1 import UnitreeG1Config
from robot.unitree_g1.unitree_g1 import UnitreeG1 as UnitreeG1Base
from robot.unitree_g1.unitree_g1_loco import UnitreeG1 as UnitreeG1Loco

# A lógica está duplicada nos dois arquivos. Quem roda hoje é a de `unitree_g1_loco`
# (é dela que `UnitreeG1Dex3` herda), mas a cópia de `unitree_g1.py` é testada junto
# para as duas não divergirem em silêncio.
CLASSES = pytest.mark.parametrize(
    "classe_robo", [UnitreeG1Loco, UnitreeG1Base], ids=["loco (em uso)", "base"]
)

ALTURA, LARGURA = 48, 64
NOME_CAM = "head_camera"


class ServidorFalso:
    """Publica quadros como o servidor de imagem do robô, e sabe emudecer.

    `mudo` é o ponto do teste: é o engasgo de Wi-Fi que antes matava a sessão.
    """

    def __init__(self, porta: int, hz: float = 60.0):
        self.porta = porta
        self.periodo = 1.0 / hz
        self.mudo = threading.Event()
        self._parar = threading.Event()
        self._thread = None
        self.contador = 0

    def __enter__(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUB)
        sock.bind(f"tcp://127.0.0.1:{self.porta}")

        def laco():
            while not self._parar.is_set():
                if not self.mudo.is_set():
                    quadro = np.full((ALTURA, LARGURA, 3), self.contador % 256, dtype=np.uint8)
                    cabecalho = {
                        "protocol": "zmq.raw.v1",
                        "images": {
                            NOME_CAM: {"part": 1, "dtype": "uint8", "shape": list(quadro.shape)}
                        },
                    }
                    sock.send_multipart([json.dumps(cabecalho).encode(), quadro.tobytes()])
                    self.contador += 1
                time.sleep(self.periodo)
            sock.close()
            ctx.term()

        self._thread = threading.Thread(target=laco, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._parar.set()
        self._thread.join(timeout=2.0)


def _porta_livre() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    porta = s.getsockname()[1]
    s.close()
    return porta


def _montar(classe_robo, porta: int, timeout_leitura_ms: int, carencia_s: float):
    """Devolve (robo, camera) prontos, sem tocar em DDS nem em hardware."""
    cfg_cam = ZMQCameraConfig(
        server_address="127.0.0.1",
        port=porta,
        camera_name=NOME_CAM,
        width=LARGURA,
        height=ALTURA,
        # `fps` só existe aqui para satisfazer o `__post_init__` de UnitreeG1Config, que
        # exige width/height/fps. A config real do dex3 sobrescreve `__post_init__` sem
        # chamar o super, então passa com fps=None — não é o caso que queremos testar.
        fps=30,
        timeout_ms=5000,
        warmup_s=5,
    )
    cfg = UnitreeG1Config(
        robot_ip="127.0.0.1",
        cameras={NOME_CAM: cfg_cam},
        camera_read_timeout_ms=timeout_leitura_ms,
        camera_grace_s=carencia_s,
    )
    robo = classe_robo(cfg)
    cam = robo._cameras[NOME_CAM]
    cam.connect()
    return robo, cam


def _ler_ate_engasgar(robo, cam, limite_s: float = 3.0):
    """Lê até a leitura passar a ser servida pelo cache, e devolve esse quadro.

    Emudecer o servidor não emudece a rede na mesma hora: um quadro já publicado ainda
    chega e é entregue como novo. Sem drenar isso, o teste comparava o quadro reusado
    com um quadro anterior ao último bom e falhava de vez em quando. `_cam_mudo_desde`
    só ganha entrada no caminho de reuso, então é o sinal exato de que entramos nele.
    """
    fim = time.monotonic() + limite_s
    while time.monotonic() < fim:
        quadro = robo._read_camera(NOME_CAM, cam)
        if NOME_CAM in robo._cam_mudo_desde:
            return quadro
    raise AssertionError("o servidor falso não emudeceu dentro do limite")


@CLASSES
def test_leitura_normal_entrega_quadros_novos(classe_robo):
    porta = _porta_livre()
    with ServidorFalso(porta):
        robo, cam = _montar(classe_robo, porta, timeout_leitura_ms=300, carencia_s=10.0)
        try:
            vistos = {robo._read_camera(NOME_CAM, cam)[0, 0, 0] for _ in range(5)}
            assert len(vistos) > 1, f"quadros não avançaram: {vistos}"
        finally:
            cam.disconnect()


@CLASSES
def test_engasgo_reusa_ultimo_quadro_em_vez_de_derrubar(classe_robo):
    """O caso que motivou tudo: rede para por uns quadros, sessão TEM de continuar."""
    porta = _porta_livre()
    with ServidorFalso(porta) as srv:
        robo, cam = _montar(classe_robo, porta, timeout_leitura_ms=100, carencia_s=10.0)
        try:
            robo._read_camera(NOME_CAM, cam)
            srv.mudo.set()
            bom = _ler_ate_engasgar(robo, cam)

            for _ in range(5):
                reusado = robo._read_camera(NOME_CAM, cam)  # não pode levantar
                assert np.array_equal(reusado, bom)

            srv.mudo.clear()
            time.sleep(0.2)
            assert robo._read_camera(NOME_CAM, cam) is not None
        finally:
            cam.disconnect()


@CLASSES
def test_quadro_reusado_e_copia(classe_robo):
    """Quem consome a observação pode escrever no array (HUD, anotação).

    Sem a cópia, essa escrita corromperia o quadro guardado e o engasgo seguinte
    devolveria lixo.
    """
    porta = _porta_livre()
    with ServidorFalso(porta) as srv:
        robo, cam = _montar(classe_robo, porta, timeout_leitura_ms=100, carencia_s=10.0)
        try:
            robo._read_camera(NOME_CAM, cam)
            srv.mudo.set()

            primeiro = _ler_ate_engasgar(robo, cam)
            original = primeiro.copy()
            primeiro[:] = 0  # consumidor "estraga" o array que recebeu

            segundo = robo._read_camera(NOME_CAM, cam)
            assert np.array_equal(segundo, original), "o cache foi corrompido pelo consumidor"
        finally:
            cam.disconnect()


@CLASSES
def test_silencio_alem_da_carencia_levanta(classe_robo):
    """Servidor caído não é engasgo: seguir aqui gravaria dataset com imagem parada."""
    porta = _porta_livre()
    with ServidorFalso(porta) as srv:
        robo, cam = _montar(classe_robo, porta, timeout_leitura_ms=50, carencia_s=0.5)
        try:
            robo._read_camera(NOME_CAM, cam)
            srv.mudo.set()

            limite = time.monotonic() + 5.0
            with pytest.raises(TimeoutError, match="sem nenhum quadro novo"):
                while time.monotonic() < limite:
                    robo._read_camera(NOME_CAM, cam)
                pytest.fail("não levantou dentro da carência")
        finally:
            cam.disconnect()


@CLASSES
def test_sem_nenhum_quadro_falha_de_cara(classe_robo):
    """Câmera que nunca respondeu é endereço errado, não engasgo — falhar é o certo,
    e falha na subida, não no meio da demonstração."""
    porta = _porta_livre()  # ninguém publicando nesta porta
    robo, cam = _montar(classe_robo, porta, timeout_leitura_ms=100, carencia_s=10.0)
    try:
        with pytest.raises(TimeoutError, match="timeout after"):
            robo._read_camera(NOME_CAM, cam)
    finally:
        cam.disconnect()


@CLASSES
def test_carencia_conta_do_primeiro_engasgo_e_zera_ao_voltar(classe_robo):
    """Um engasgo longo mas dentro da carência não pode 'acumular' com o próximo."""
    porta = _porta_livre()
    with ServidorFalso(porta) as srv:
        robo, cam = _montar(classe_robo, porta, timeout_leitura_ms=50, carencia_s=1.0)
        try:
            robo._read_camera(NOME_CAM, cam)

            for _ in range(2):
                srv.mudo.set()
                fim = time.monotonic() + 0.6  # 60% da carência, duas vezes
                while time.monotonic() < fim:
                    robo._read_camera(NOME_CAM, cam)
                srv.mudo.clear()
                time.sleep(0.3)
                robo._read_camera(NOME_CAM, cam)  # quadro novo → zera o relógio

            assert NOME_CAM not in robo._cam_mudo_desde
        finally:
            cam.disconnect()
