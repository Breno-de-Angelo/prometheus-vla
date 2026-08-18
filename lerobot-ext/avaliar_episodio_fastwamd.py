#!/usr/bin/env python
"""
Replay de episódio contra o servidor de inferência — FastWAM-D
===============================================================
Roda um episódio GRAVADO através do servidor e compara, quadro a quadro, a ação
prevista com a que o teleoperador executou. Sem robô, sem simulador.

Por que este teste e não o simulador: o render do MuJoCo é outra distribuição
visual, e um modelo com pouco treino não transfere. Aqui a entrada é exatamente
o tipo de imagem que ele viu no treino — se ele não acertar isto, não vai
acertar nada.

O que importa olhar:

  **episódio de treino × episódio held-out.** Erro pequeno no de treino e grande
  no held-out é a assinatura de decoreba. Rodar os dois lado a lado é o ponto
  deste script.

  **as juntas da mão.** "Pegar a xícara" acontece nos dedos. Um erro médio bonito
  espalhado por 29 juntas pode esconder uma mão que nunca fecha.

Uso (na athena, com o servidor no ar):
  python avaliar_episodio_fastwamd.py --episodios=0,25 [OPÇÕES]

Opções:
  --episodios=<n,n>   quais episódios rodar (padrão: 0,25 — um de treino e um held-out)
  --server=<IP>       servidor de inferência (padrão: 127.0.0.1)
  --port=<INT>        porta (padrão: 5600)
  --passo=<INT>       de quantos em quantos quadros pedir uma inferência
                      (padrão: 32 = o horizonte; é a cadência real de operação)
  --root=<PATH>       raiz do dataset
  --saida=<PATH>      PNG do gráfico (padrão: ./avaliacao_fastwamd.png)
  -h, --help          esta mensagem
"""

import os
import sys

# zmq ANTES do torch — ver a nota no topo do servidor.
try:
    import zmq
    import msgpack
    import msgpack_numpy as m
    m.patch()
except ImportError:
    print("❌ pip install pyzmq msgpack msgpack-numpy")
    sys.exit(1)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RAIZ_PADRAO = "meu_dataset/white_cup_on_dripper_2026-08-11"

# Os grupos existem para o erro não virar uma média que esconde tudo. A mão é o
# que decide se a xícara foi pega; o braço é o que decide se ele chegou lá.
GRUPOS = [
    ("braco esq", slice(0, 7)),
    ("braco dir", slice(7, 14)),
    ("cintura", slice(14, 15)),
    ("mao esq", slice(15, 22)),
    ("mao dir", slice(22, 29)),
]


def _para_uint8(t):
    a = t.numpy()
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return a


def avalia(episodio: int, juntas, ds, socket, passo: int):
    """Percorre o episódio pedindo um chunk a cada `passo` quadros."""
    indices = [i for i, e in enumerate(ds.hf_dataset["episode_index"]) if int(e) == episodio]
    if not indices:
        raise SystemExit(f"episódio {episodio} não está no dataset carregado")

    previstas, gravadas, travadas_total = [], [], 0
    for k in range(0, len(indices) - 1, passo):
        amostra = ds[indices[k]]
        obs = {nome: float(amostra["observation.state"][i]) for i, nome in enumerate(juntas)}
        obs["head_camera"] = _para_uint8(amostra["observation.images.head_camera"])
        obs["right_wrist_camera"] = _para_uint8(amostra["observation.images.right_wrist_camera"])
        obs["head_camera_depth"] = (
            amostra["observation.images.head_camera_depth"].numpy().squeeze().astype(np.uint16)
        )

        socket.send(msgpack.packb(
            {"obs": obs, "obs_step": k, "actions_per_chunk": passo, "want_debug": False},
            default=m.encode))
        r = msgpack.unpackb(socket.recv(), object_hook=m.decode, raw=False)
        if "error" in r:
            raise SystemExit(f"servidor: {r['error']}")
        chunk = np.asarray(r["chunk_np"], dtype=np.float32)
        travadas_total += int(r.get("travadas", 0))

        # As ações gravadas correspondentes a este trecho, para comparar
        # exatamente o mesmo intervalo de tempo que o chunk cobre.
        for j in range(min(len(chunk), len(indices) - 1 - k)):
            alvo = ds[indices[k + j]]["action"].numpy()
            gravadas.append(alvo[0] if alvo.ndim > 1 else alvo)
            previstas.append(chunk[j])
        print(f"   ep {episodio}: quadro {k:4d}/{len(indices)}", end="\r", flush=True)

    return np.array(previstas), np.array(gravadas), travadas_total


def main():
    if any(f in sys.argv for f in ("-h", "--help")):
        print(__doc__)
        sys.exit(0)

    episodios = [0, 25]
    server, port, passo = "127.0.0.1", 5600, 32
    root = RAIZ_PADRAO
    saida = "./avaliacao_fastwamd.png"
    for arg in sys.argv[1:]:
        if arg.startswith("--episodios="):
            episodios = [int(v) for v in arg.split("=", 1)[1].split(",")]
        elif arg.startswith("--server="):
            server = arg.split("=", 1)[1]
        elif arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])
        elif arg.startswith("--passo="):
            passo = int(arg.split("=", 1)[1])
        elif arg.startswith("--root="):
            root = arg.split("=", 1)[1]
        elif arg.startswith("--saida="):
            saida = arg.split("=", 1)[1]

    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    juntas = json.load(open(f"{root}/meta/info.json"))["features"]["action"]["names"]
    ds = LeRobotDataset(repo_id="local", root=root, episodes=episodios)

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 300000)
    socket.connect(f"tcp://{server}:{port}")
    print(f"🔗 servidor {server}:{port} | episódios {episodios} | passo {passo}\n")

    resultados = {}
    for ep in episodios:
        prev, grav, travadas = avalia(ep, juntas, ds, socket, passo)
        erro = np.abs(prev - grav)
        resultados[ep] = (prev, grav, erro, travadas)
        print(f"\n── episódio {ep} ── {len(prev)} quadros | travadas {travadas}")
        print(f"   erro medio geral : {erro.mean():.4f} rad ({np.degrees(erro.mean()):.1f}°)")
        for nome, fatia in GRUPOS:
            print(f"   {nome:10s}: medio {erro[:, fatia].mean():.4f} | "
                  f"pior {erro[:, fatia].max():.4f} rad")

    # ── Gráfico ──────────────────────────────────────────────────────────────
    fig, eixos = plt.subplots(len(episodios), 2, figsize=(14, 4.2 * len(episodios)), squeeze=False)
    for linha, ep in enumerate(episodios):
        prev, grav, erro, travadas = resultados[ep]
        ax = eixos[linha][0]
        ax.plot(erro.mean(axis=1), lw=1.2, label="erro medio (29 juntas)")
        ax.plot(erro[:, 15:29].mean(axis=1), lw=1.2, label="erro medio (maos)")
        ax.set_title(f"episodio {ep} — erro ao longo do tempo")
        ax.set_xlabel("quadro"); ax.set_ylabel("rad"); ax.legend(); ax.grid(alpha=0.3)

        ax = eixos[linha][1]
        # A junta de mão que MAIS se move no episódio, escolhida pelos dados e
        # não fixada no código: neste dataset o `thumb_0` não sai do zero (a base
        # do polegar é fixa) e a garra inteira é acionada junto — plotar a junta
        # errada dá uma linha reta que parece "o modelo não mexe a mão", quando
        # na verdade é a junta que não mexe.
        maos = slice(15, 29)
        idx = 15 + int(np.argmax(grav[:, maos].std(axis=0)))
        ax.plot(grav[:, idx], lw=1.6, label=f"{juntas[idx]} gravado")
        ax.plot(prev[:, idx], lw=1.3, ls="--", label=f"{juntas[idx]} previsto")
        ax.set_title(f"episodio {ep} — a garra (a junta mais ativa da mao)")
        ax.set_xlabel("quadro"); ax.set_ylabel("rad"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(saida, dpi=110)
    print(f"\n📈 gráfico em {saida}")


if __name__ == "__main__":
    main()
