#!/usr/bin/env python
"""
Servidor de Inferência — FastWAM-D (roda na athena)
====================================================
Carrega o FastWAM-D e responde chunks de ação por ZMQ. O robô fica com o
cliente (`init_lerobot_inference_fastwamd_client.py`), no seu PC.

Por que um servidor separado dos antigos: o FastWAM-D tem 6 B de parâmetros e
depende do VAE do Wan, do encoder de texto UMT5 e do enxerto de profundidade —
nada disso cabe no notebook, e o `init_lerobot_inference_server.py` foi escrito
em volta do ACT-D/PI05 (uma câmera, sem texto, chunk de 60).

Diferenças que importam em relação ao servidor antigo:

  - **duas câmeras de cor** (cabeça e pulso direito), concatenadas na largura
    pelo próprio modelo — o cliente manda as duas cruas;
  - **profundidade métrica** em milímetros, 1 canal, que viaja pelo latente e
    não pelo mosaico de cor;
  - **task em texto** — o FastWAM é condicionado por linguagem, então a frase
    da tarefa vai em toda requisição (ou fica fixa por `--task=`);
  - **chunk de no máximo 32** ações (`action_horizon` do modelo), contra 60 do
    ACT-D. Pedir mais que isso não gera mais ação, só recorta.

Protocolo ZMQ (REQ/REP, msgpack):

  cliente → { obs: {...}, obs_step: int, actions_per_chunk: int,
              want_debug: bool, task: str|None }
  servidor → { chunk_np: [[float]], obs_step: int, infer_ms: float,
               debug?: { attn: [[float]], depth: [[float]] } }
  em erro  → { error: str }

  `want_debug` é opcional e caro só quando ligado: ele instala os ganchos de
  captura (`policies/fastwam_depth/debug_inferencia.py`) durante aquela
  inferência. Cliente que não pede não paga nada.

Uso:
  python init_lerobot_inference_fastwamd_server.py --checkpoint=<PATH> [OPÇÕES]

Opções:
  --checkpoint=<PATH>  (obrigatório) caminho do `pretrained_model`
                       (ex: /data/train_output/.../checkpoints/best/pretrained_model)
  --port=<INT>         porta ZMQ (padrão: 5600)
  --host=<IP>          interface de escuta (padrão: 0.0.0.0)
  --task=<STR>         frase da tarefa (padrão: "place the white cup on the dripper").
                       O cliente pode sobrescrever a cada requisição.
  --device=<STR>       cuda | cuda:1 | cpu (padrão: cuda se houver)
  --debug              loga cada inferência
  -h, --help           esta mensagem

Exemplo na athena:
  HF_HOME=/data/.cache/huggingface CUDA_VISIBLE_DEVICES=2 \\
  python init_lerobot_inference_fastwamd_server.py \\
      --checkpoint=/data/train_output/fastwamdepth_white_cup_on_dripper/checkpoints/best/pretrained_model \\
      --port=5600 --debug
"""

import os
import sys
import time
import traceback

# ⚠️ ORDEM: o zmq TEM que ser importado ANTES do torch.
#
# Ao contrário, o processo morre com `Segmentation fault (core dumped)` sem
# imprimir uma linha — o crash acontece depois, no meio do import do lerobot,
# e nem o `faulthandler` pega. A causa é o libstdc++: o do sistema é mais
# antigo que o exigido pelo numpy/torch do ambiente, e quem é carregado
# primeiro define qual fica valendo para todo o processo. Importado antes, o
# pyzmq puxa o libstdc++ do conda e o resto encontra os símbolos que precisa.
#
# Sintoma de bancada, se você duvidar: `python -c "import zmq, torch"` funciona
# e `python -c "import torch, zmq"` não.
try:
    import zmq
    import msgpack
    import msgpack_numpy as m
    m.patch()
except ImportError:
    print("❌ Instale as dependências: pip install pyzmq msgpack msgpack-numpy")
    sys.exit(1)

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policies.fastwam_depth.configuration_fastwam_depth import FastWAMDepthConfig  # noqa: E402,F401
from policies.fastwam_depth.debug_inferencia import CapturaDebug  # noqa: E402
from policies.fastwam_depth.modeling_fastwam_depth import FastWAMDepthPolicy  # noqa: E402

TAREFA_PADRAO = "place the white cup on the dripper"

# Ordem EXATA do dataset (`meta/info.json`). Não é a mesma dos scripts antigos:
# são 29 e não 28 (entra o `kWaistYaw.q` no meio), e as duas mãos não seguem a
# mesma ordem de dedos — a esquerda vai thumb/middle/index e a direita
# thumb/index/middle. Errar isto embaralha juntas em silêncio: o modelo recebe
# a posição do indicador no lugar da do médio e a ação sai plausível e errada.
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


class LinhaDeBaseDaAtencao:
    """Separa o que o modelo OLHOU do que ele sempre olha.

    O mapa cru é dominado por *attention sink*: as primeiras colunas de cada
    linha da grade recebem a maior parte do peso em TODO quadro, independente da
    cena — medido neste checkpoint, coluna 0 com média 0,48 e coluna 1 com 0,27,
    contra ~0,03 no resto, praticamente idênticas em quadros completamente
    diferentes. É destino "nulo" do transformer, não informação.

    Como o sumidouro é estacionário e o conteúdo varia, o que interessa é o
    DESVIO em relação ao que aquele token costuma receber. Esta classe mantém a
    média móvel e devolve `max(atual - base, 0)` normalizado.

    Aquecimento: nas primeiras requisições a média ainda é ruim, então o peso do
    quadro novo decai como 1/n até o piso — assim a base converge rápido em vez
    de arrastar o primeiro quadro por dezenas de inferências.
    """

    def __init__(self, piso_alpha: float = 0.05):
        self.base: np.ndarray | None = None
        self.n = 0
        self.piso_alpha = piso_alpha

    def relativo(self, mapa: np.ndarray) -> tuple[np.ndarray, int]:
        if self.base is None or self.base.shape != mapa.shape:
            self.base = mapa.astype(np.float64).copy()
            self.n = 1
            return np.zeros_like(mapa), self.n

        # Desvio RELATIVO, não absoluto: `(atual - base) / base`. Medido, a
        # subtração pura não bastava — o sumidouro não é perfeitamente estável, e
        # uma flutuação para cima dele (base 0,48) sobrevivia à subtração e
        # continuava ganhando de conteúdo real (base 0,03). Em proporção, +0,10
        # na coluna do sumidouro é 20% e +0,10 num token de conteúdo é 300%.
        desvio = np.clip((mapa - self.base) / (self.base + 1e-4), 0.0, None)
        pico = float(desvio.max())
        relativo = (desvio / pico) if pico > 0 else np.zeros_like(desvio)

        alpha = max(self.piso_alpha, 1.0 / (self.n + 1))
        self.base = (1 - alpha) * self.base + alpha * mapa
        self.n += 1
        return relativo.astype(np.float32), self.n


def _pack(data: dict) -> bytes:
    return msgpack.packb(data, default=m.encode)


def _unpack(raw: bytes) -> dict:
    return msgpack.unpackb(raw, object_hook=m.decode, raw=False)


# ─────────────────────────────────────────────────────────────────────
# Observação
# ─────────────────────────────────────────────────────────────────────
def monta_obs(obs: dict, config, task: str) -> dict:
    """Dict de observação cru — sem batch e sem normalização; o preprocessor faz o resto.

    As imagens vão em [0, 1] e a profundidade em MILÍMETROS, porque é assim que
    o dataset entrega e é o que o `mapeia_para_unidade` espera. Dividir a
    profundidade por 255 (como o caminho antigo fazia, de quando ela era imagem
    de 8 bits) não quebra nada visivelmente e destrói a escala métrica.
    """
    raw: dict = {}

    estado = [float(obs.get(nome, 0.0)) for nome in JUNTAS_G1]
    raw["observation.state"] = torch.tensor(estado, dtype=torch.float32)

    for chave in config.rgb_feature_keys:
        curto = chave.removeprefix("observation.images.")
        imagem = obs.get(curto)
        if imagem is None:
            raise KeyError(
                f"Câmera '{curto}' não veio na observação. O FastWAM-D foi treinado com "
                f"{[k.removeprefix('observation.images.') for k in config.rgb_feature_keys]} "
                "e o mosaico depende das duas: faltando uma, as fatias entram deslocadas."
            )
        raw[chave] = torch.from_numpy(np.ascontiguousarray(imagem)).permute(2, 0, 1).float().div(255.0)

    for chave in config.depth_feature_keys:
        curto = chave.removeprefix("observation.images.")
        profundidade = obs.get(curto)
        if profundidade is None and f"{curto}_png" in obs:
            # PNG de 16 bits — compressão SEM PERDA, para a profundidade não
            # ocupar 800 KB por requisição. `IMREAD_UNCHANGED` é obrigatório:
            # `IMREAD_COLOR` (o padrão) devolveria 8 bits em 3 canais e os
            # milímetros virariam lixo, do mesmo jeito que já aconteceu no
            # decodificador ZMQ (ver docs/PROFUNDIDADE_NATIVA.md).
            import cv2

            bruto = np.frombuffer(obs[f"{curto}_png"], dtype=np.uint8)
            profundidade = cv2.imdecode(bruto, cv2.IMREAD_UNCHANGED)
            if profundidade is None:
                raise ValueError(f"não consegui decodificar o PNG de profundidade de '{curto}'.")
        if profundidade is None:
            raise KeyError(
                f"Profundidade '{curto}' não veio na observação. Rodar sem ela exige um "
                "checkpoint treinado com `depth_mode: off` — o modo latente levanta erro "
                "de propósito, em vez de concatenar zeros e inferir às cegas."
            )
        mapa = np.squeeze(profundidade)
        if mapa.ndim != 2:
            raise ValueError(
                f"Profundidade deveria ser [H, W] de 1 canal, veio {np.shape(profundidade)}. "
                "Servidor de câmera publicando cinza de 3 canais?"
            )
        raw[chave] = torch.from_numpy(np.ascontiguousarray(mapa)).float().unsqueeze(0)

    raw["task"] = task
    return raw


# ─────────────────────────────────────────────────────────────────────
# Servidor
# ─────────────────────────────────────────────────────────────────────
def carrega_limites_de_acao(checkpoint: str, margem: float = 0.10):
    """Faixa de ação vista no dataset, com folga, para servir de trava.

    Vem do mesmo arquivo de estatísticas que o postprocessor usa para
    desnormalizar (`action.min` / `action.max`), então é literalmente "o que os
    teleoperadores demonstraram". Nada além disso deveria sair daqui: uma
    política com pouco treino, recebendo imagem fora da distribuição (o
    simulador, uma câmera preta, uma cena diferente), produz ação de magnitude
    absurda — medi 10⁵ rad com imagem de ruído puro. O modelo não tem como
    saber que aquilo é impossível; a trava tem.

    A margem existe para não engessar extrapolação legítima nas bordas do que
    foi demonstrado — 10% da faixa de cada junta, não um valor fixo.
    """
    from pathlib import Path

    from safetensors.torch import load_file

    arquivos = sorted(Path(checkpoint).glob("*unnormalizer_processor.safetensors"))
    if not arquivos:
        print("⚠️  Sem estatísticas de ação no checkpoint — a trava de faixa fica DESLIGADA.")
        return None, None

    stats = load_file(str(arquivos[0]))
    if "action.min" not in stats or "action.max" not in stats:
        print("⚠️  Estatísticas sem action.min/action.max — trava DESLIGADA.")
        return None, None

    minimo = stats["action.min"].to(torch.float32)
    maximo = stats["action.max"].to(torch.float32)
    folga = (maximo - minimo).abs() * margem
    return minimo - folga, maximo + folga


def carrega_politica(checkpoint: str, device: torch.device):
    print(f"⏳ Carregando FastWAM-D de: {checkpoint}")
    config = FastWAMDepthConfig.from_pretrained(checkpoint)
    politica = FastWAMDepthPolicy.from_pretrained(checkpoint, config=config)
    politica.to(device)
    politica.eval()
    politica.guardar_video_depth = True  # o painel de profundidade do cliente lê daqui

    print(f"✅ FastWAM-D carregado — depth_mode={config.depth_mode} | "
          f"horizonte={config.action_horizon} | câmeras={len(config.rgb_feature_keys)}")
    return politica, config


def run_server(checkpoint: str, host: str, port: int, task_padrao: str,
               device_str: str | None, debug: bool):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🖥️  Device: {device}")
    if device.type == "cuda" and torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(device.index or 0)}")

    politica, config = carrega_politica(checkpoint, device)

    from lerobot.policies.factory import make_pre_post_processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=checkpoint
    )
    print("✅ Pre/postprocessors carregados do checkpoint")

    lim_min, lim_max = carrega_limites_de_acao(checkpoint)
    if lim_min is not None:
        print(f"🛡️  Trava de faixa ativa: [{float(lim_min.min()):+.3f}, "
              f"{float(lim_max.max()):+.3f}] rad (faixa do dataset + 10%)")

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    endereco = f"tcp://{host}:{port}"
    socket.bind(endereco)
    print(f"🔌 Escutando em {endereco}")
    print(f"   Tarefa padrão: {task_padrao!r}")
    print("   Aguardando o cliente...\n")

    n_req = 0
    ms_total = 0.0
    linha_base = LinhaDeBaseDaAtencao()

    try:
        while True:
            msg = _unpack(socket.recv())
            obs = msg["obs"]
            obs_step = int(msg.get("obs_step", 0))
            n_acoes = int(msg.get("actions_per_chunk", config.action_horizon))
            want_debug = bool(msg.get("want_debug", False))
            task = msg.get("task") or task_padrao

            # Pedir mais que o horizonte não gera ação nova — o modelo emite
            # `action_horizon` de cada vez. Recortar em silêncio faria o cliente
            # acreditar num buffer que não existe e zerar no meio do movimento.
            if n_acoes > config.action_horizon:
                n_acoes = config.action_horizon

            t0 = time.perf_counter()
            try:
                raw_obs = monta_obs(obs, config, task)
                batch = preprocessor(raw_obs)
                batch.pop("action", None)

                captura = CapturaDebug(politica) if want_debug else None
                with torch.inference_mode():
                    if captura is not None:
                        with captura:
                            chunk = politica.predict_action_chunk(batch)
                    else:
                        chunk = politica.predict_action_chunk(batch)

                    chunk = chunk[0, :n_acoes, :]
                    processado = postprocessor(chunk.unsqueeze(1))
                    if isinstance(processado, dict):
                        processado = processado["action"]
                    bruto = processado.squeeze(1).to(torch.float32).cpu()

                # ── Trava de faixa ────────────────────────────────────────
                nao_finitos = int((~torch.isfinite(bruto)).sum())
                if nao_finitos:
                    # NaN/inf não se conserta com clamp: o chunk inteiro é lixo
                    # e mandá-lo seria pior que não responder.
                    raise ValueError(
                        f"a política devolveu {nao_finitos} valores não finitos (NaN/inf) — "
                        "chunk descartado."
                    )
                travadas = 0
                if lim_min is not None:
                    fora = ((bruto < lim_min) | (bruto > lim_max))
                    travadas = int(fora.sum())
                    if travadas:
                        pior = float((bruto - bruto.clamp(lim_min, lim_max)).abs().max())
                        print(f"⚠️  step {obs_step}: {travadas} de {bruto.numel()} valores fora da "
                              f"faixa do dataset (pior excesso {pior:.3f} rad) — travados.")
                    bruto = bruto.clamp(lim_min, lim_max)
                chunk_np = bruto.numpy()

                infer_ms = (time.perf_counter() - t0) * 1000.0
                resposta = {
                    "chunk_np": chunk_np,
                    "obs_step": obs_step,
                    "infer_ms": infer_ms,
                    "travadas": travadas,
                }

                if captura is not None:
                    payload = {}
                    cru = captura.mapa_atencao(normalizar=False)
                    if cru is not None:
                        faixa = float(cru.max() - cru.min())
                        payload["attn"] = (
                            ((cru - cru.min()) / faixa) if faixa > 0 else np.zeros_like(cru)
                        ).astype(np.float32)
                        relativo, n_base = linha_base.relativo(cru)
                        payload["attn_rel"] = relativo
                        payload["attn_base_n"] = int(n_base)
                    profundidade = captura.mapa_profundidade()
                    if profundidade is not None:
                        payload["depth"] = profundidade.astype(np.float32)
                    if payload:
                        resposta["debug"] = payload

                socket.send(_pack(resposta))

                n_req += 1
                ms_total += infer_ms
                if debug:
                    print(f"[{n_req:04d}] step={obs_step} chunk={chunk_np.shape} "
                          f"{infer_ms:7.1f} ms (média {ms_total / n_req:6.1f}) "
                          f"{'debug' if want_debug else ''}")

            except Exception as erro:  # noqa: BLE001 - o servidor não pode morrer com o cliente
                traceback.print_exc()
                socket.send(_pack({"error": f"{type(erro).__name__}: {erro}"}))

    except KeyboardInterrupt:
        print("\n⏹️  Encerrando servidor.")
    finally:
        socket.close()
        ctx.term()


def main():
    if any(f in sys.argv for f in ["-h", "--help"]):
        print(__doc__)
        sys.exit(0)

    checkpoint = None
    host = "0.0.0.0"
    port = 5600
    task = TAREFA_PADRAO
    device = None
    debug = False

    for arg in sys.argv[1:]:
        if arg.startswith("--checkpoint="):
            checkpoint = arg.split("=", 1)[1]
        elif arg.startswith("--host="):
            host = arg.split("=", 1)[1]
        elif arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])
        elif arg.startswith("--task="):
            task = arg.split("=", 1)[1]
        elif arg.startswith("--device="):
            device = arg.split("=", 1)[1]
        elif arg == "--debug":
            debug = True

    if not checkpoint:
        print("❌ ERRO: --checkpoint=<PATH> é obrigatório.")
        print("   Ex: --checkpoint=/data/train_output/fastwamdepth_white_cup_on_dripper/"
              "checkpoints/best/pretrained_model")
        sys.exit(1)
    if not os.path.isdir(checkpoint):
        print(f"❌ ERRO: {checkpoint} não existe.")
        sys.exit(1)

    run_server(checkpoint, host, port, task, device, debug)


if __name__ == "__main__":
    main()
