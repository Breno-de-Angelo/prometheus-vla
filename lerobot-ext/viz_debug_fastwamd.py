#!/usr/bin/env python
"""Painel de depuração do FastWAM-D — uma janela, quatro quadrantes.

    ┌──────────────────────┬──────────────────────┐
    │ 1. atenção do DiT    │ 2. profundidade      │
    │    sobre o mosaico   │    (crua e a do      │
    │    das câmeras       │     modelo)          │
    ├──────────────────────┼──────────────────────┤
    │ 3. nuvem de pontos   │ 4. temperatura dos   │
    │    topo + lateral    │    motores do G1     │
    └──────────────────────┴──────────────────────┘

Uma janela só, e não quatro, porque o loop de controle roda a 30 Hz: cada
`cv2.imshow` extra é tempo roubado do ciclo, e quatro janelas soltas viram
quatro `waitKey`. Aqui é um `imshow` por ciclo, com os quadrantes desenhados
num canvas único.

Os quadrantes 1 e 2 vêm do SERVIDOR (só o modelo sabe onde olhou e o que
recebeu como profundidade depois da normalização); os quadrantes 3 e 4 são
locais, calculados do que o robô e a câmera entregam aqui.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# Faixa métrica do painel de profundidade e da nuvem, em metros. É a mesma do
# `depth_min`/`depth_max` do treino: ver a cena com uma escala e o modelo com
# outra é o jeito mais fácil de tirar conclusão errada de um debug.
PROF_MIN_M = 0.05
PROF_MAX_M = 5.0

# Intrínsecos NOMINAIS da RealSense a 848x480 (mesmos do `configuration_act.py`).
# Os reais ainda não foram medidos no robô — `Scripts_Prometheus_int/
# print_camera_intrinsics.py` imprime os de verdade. Com estes, a nuvem tem a
# forma certa mas a escala absoluta pode andar alguns por cento.
INTRINSECOS_PADRAO = {"fx": 617.0, "fy": 617.0, "cx": 424.0, "cy": 240.0}

_VERDE = (90, 200, 90)
_AMARELO = (60, 200, 230)
_VERMELHO = (60, 60, 235)
_CINZA = (150, 150, 150)
_FUNDO = (24, 24, 24)


def _texto(canvas, txt, x, y, escala=0.45, cor=_CINZA, grosso=1):
    """Escreve na tela. Os rótulos deste módulo são ASCII de propósito.

    As fontes Hershey do OpenCV não têm acento nem travessão: qualquer caractere
    fora do ASCII vira `?` na tela. Por isso "Atencao", "-" no lugar de "–" e
    "+/-" no lugar de "±" — feio no código, legível no painel.
    """
    cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, grosso, cv2.LINE_AA)


def _encaixa(imagem: np.ndarray, w: int, h: int) -> np.ndarray:
    """Redimensiona preservando a proporção, com barras escuras no que sobrar.

    O mosaico das câmeras é 2:1 e o quadrante é 4:3; esticar deformaria a cena e,
    junto com ela, o mapa de atenção desenhado por cima — que passaria a apontar
    para um lugar que não corresponde ao pixel real.
    """
    alt, larg = imagem.shape[:2]
    escala = min(w / larg, h / alt)
    nova = cv2.resize(imagem, (max(1, int(larg * escala)), max(1, int(alt * escala))),
                      interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), _FUNDO, dtype=np.uint8)
    y0 = (h - nova.shape[0]) // 2
    x0 = (w - nova.shape[1]) // 2
    canvas[y0:y0 + nova.shape[0], x0:x0 + nova.shape[1]] = nova
    return canvas


def _quadro_vazio(w: int, h: int, aviso: str) -> np.ndarray:
    canvas = np.full((h, w, 3), _FUNDO, dtype=np.uint8)
    _texto(canvas, aviso, 14, h // 2, 0.5, (110, 110, 110))
    return canvas


# ══════════════════════════════════════════════════════════════════════════
# Quadrantes
# ══════════════════════════════════════════════════════════════════════════
def desenha_atencao(rgb_mosaico: np.ndarray | None, mapa: np.ndarray | None,
                    w: int, h: int, relativo: bool = False, base_n: int = 0) -> np.ndarray:
    """Mosaico das câmeras com o mapa de atenção por cima.

    O mapa vem na grade de tokens do DiT (dezenas de células, não pixels), então
    ele é ampliado com interpolação linear — a mancha suave é honesta quanto à
    resolução real: cada célula é um token, e o token é um pedaço da imagem.
    """
    if rgb_mosaico is None:
        return _quadro_vazio(w, h, "1. Atencao - sem imagem")

    bgr = cv2.cvtColor(rgb_mosaico, cv2.COLOR_RGB2BGR) if rgb_mosaico.ndim == 3 else rgb_mosaico
    base = _encaixa(bgr, w, h)

    if mapa is None:
        _texto(base, "1. Atencao - aguardando o servidor", 10, 20, 0.42)
        return base

    # O calor passa pelo MESMO encaixe da imagem: redimensionar os dois com
    # geometrias diferentes desalinharia o mapa em relação à cena.
    calor_rgb = cv2.applyColorMap(
        (np.clip(cv2.resize(mapa.astype(np.float32), (rgb_mosaico.shape[1], rgb_mosaico.shape[0]),
                            interpolation=cv2.INTER_LINEAR), 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_JET)
    colorido = _encaixa(calor_rgb, w, h)
    calor = cv2.cvtColor(colorido, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # 0,55 de imagem e 0,45 de calor: o suficiente para achar a região quente
    # sem perder de vista o objeto que está embaixo dela.
    mistura = cv2.addWeighted(base, 0.55, colorido, 0.45, 0)

    titulo = ("1. Atencao do DiT - desvio da linha de base" if relativo
              else "1. Atencao do DiT (cru - contem o sumidouro)")
    _texto(mistura, titulo, 10, 20, 0.45, (240, 240, 240), 1)
    if relativo and base_n < 10:
        # Com poucas amostras a base ainda é o próprio quadro; dizer isso evita
        # que alguém leia um mapa quase vazio como "o modelo não olha para nada".
        _texto(mistura, f"base aquecendo ({base_n} amostras)", 10, 36, 0.38, (150, 150, 150))
    # O pico é localizado no mapa original e trazido para a tela pelo encaixe,
    # em vez de procurado no colormap — no JET, vermelho e azul escuro têm
    # luminância parecida, e um argmax sobre o cinza cairia no lugar errado.
    ph, pw = np.unravel_index(int(np.argmax(mapa)), mapa.shape)
    escala = min(w / rgb_mosaico.shape[1], h / rgb_mosaico.shape[0])
    lado_w = int(rgb_mosaico.shape[1] * escala)
    lado_h = int(rgb_mosaico.shape[0] * escala)
    pico = (
        int((h - lado_h) // 2 + (ph + 0.5) / mapa.shape[0] * lado_h),
        int((w - lado_w) // 2 + (pw + 0.5) / mapa.shape[1] * lado_w),
    )
    cv2.circle(mistura, (int(pico[1]), int(pico[0])), 9, (255, 255, 255), 2)
    _texto(mistura, "pico", int(pico[1]) + 12, int(pico[0]) - 8, 0.4, (255, 255, 255))
    return mistura


def desenha_profundidade(depth_mm: np.ndarray | None, depth_modelo: np.ndarray | None,
                         w: int, h: int) -> np.ndarray:
    """Duas faixas: a profundidade crua da câmera e a que o modelo recebeu.

    Ver as duas lado a lado é o ponto do painel. A de cima é o que a RealSense
    mandou, em milímetros. A de baixo é depois do log, do recorte de faixa e do
    mosaico — se a de cima tem geometria e a de baixo está preta, a
    profundidade não está chegando ao modelo, e nenhuma métrica de treino
    contaria isso.
    """
    canvas = np.full((h, w, 3), _FUNDO, dtype=np.uint8)
    meia = h // 2

    if depth_mm is not None:
        metros = np.squeeze(depth_mm).astype(np.float32) / 1000.0
        valido = (metros > PROF_MIN_M) & (metros < PROF_MAX_M)
        norm = np.zeros_like(metros)
        norm[valido] = (metros[valido] - PROF_MIN_M) / (PROF_MAX_M - PROF_MIN_M)
        colorido = cv2.applyColorMap((np.clip(1.0 - norm, 0, 1) * 255).astype(np.uint8),
                                     cv2.COLORMAP_TURBO)
        colorido[~valido] = (40, 40, 40)   # sem medida: cinza, não "perto"
        canvas[:meia] = _encaixa(colorido, w, meia)
        medianos = metros[valido]
        faixa = (f"{medianos.min():.2f} a {medianos.max():.2f} m"
                 if medianos.size else "sem medida valida")
        _texto(canvas, f"2. Depth da camera  ({faixa})", 10, 20, 0.45, (240, 240, 240))
    else:
        _texto(canvas, "2. Depth da camera - sem dado", 10, 20, 0.45)

    if depth_modelo is not None:
        mapa = np.clip(np.squeeze(depth_modelo).astype(np.float32), 0.0, 1.0)
        colorido = cv2.applyColorMap((np.clip(1.0 - mapa, 0, 1) * 255).astype(np.uint8),
                                     cv2.COLORMAP_TURBO)
        colorido[mapa <= 0.0] = (40, 40, 40)
        canvas[meia:] = _encaixa(colorido, w, h - meia)
        _texto(canvas, "   o que o modelo recebeu (mosaico normalizado)",
               10, meia + 20, 0.42, (240, 240, 240))
    else:
        _texto(canvas, "   modelo: aguardando servidor", 10, meia + 20, 0.42)

    cv2.line(canvas, (0, meia), (w, meia), (70, 70, 70), 1)
    return canvas


def desenha_nuvem(depth_mm: np.ndarray | None, intrinsecos: dict,
                  w: int, h: int, max_pontos: int = 6000) -> np.ndarray:
    """Nuvem de pontos em duas projeções: topo (XZ) e lateral (YZ).

    É a mesma projeção que alimenta o encoder de profundidade das outras
    políticas (`depth_to_pointcloud`), refeita aqui em numpy para não arrastar
    torch para o cliente — o PC que controla o robô não precisa de modelo
    nenhum carregado.
    """
    canvas = np.full((h, w, 3), _FUNDO, dtype=np.uint8)
    meia = w // 2
    cv2.line(canvas, (meia, 0), (meia, h), (70, 70, 70), 1)
    _texto(canvas, "3. Nuvem - topo (XZ)", 10, 18, 0.42, (240, 240, 240))
    _texto(canvas, "lateral (YZ)", meia + 10, 18, 0.42, (240, 240, 240))

    if depth_mm is None:
        _texto(canvas, "sem profundidade", 10, h // 2, 0.5, (110, 110, 110))
        return canvas

    mapa = np.squeeze(depth_mm).astype(np.float32) / 1000.0
    alt, larg = mapa.shape
    fx, fy = float(intrinsecos["fx"]), float(intrinsecos["fy"])
    cx, cy = float(intrinsecos["cx"]), float(intrinsecos["cy"])
    # Os intrínsecos são da resolução nativa; se o quadro vier redimensionado,
    # eles têm que acompanhar, senão a nuvem sai esticada.
    escala_x = larg / (2.0 * cx)
    escala_y = alt / (2.0 * cy)
    fx, cx = fx * escala_x, cx * escala_x
    fy, cy = fy * escala_y, cy * escala_y

    ys, xs = np.nonzero((mapa > PROF_MIN_M) & (mapa < PROF_MAX_M))
    if xs.size == 0:
        _texto(canvas, "nenhum ponto na faixa", 10, h // 2, 0.5, (110, 110, 110))
        return canvas
    if xs.size > max_pontos:
        escolha = np.random.choice(xs.size, max_pontos, replace=False)
        xs, ys = xs[escolha], ys[escolha]

    z = mapa[ys, xs]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    def _plota(px, pz, x0, faixa_h, faixa_v, rotulo_h, rotulo_v):
        for a, b, cor_z in zip(px, pz, z):
            u = int((a - faixa_h[0]) / (faixa_h[1] - faixa_h[0]) * (meia - 20)) + x0 + 10
            v = int(h - 24 - (b - faixa_v[0]) / (faixa_v[1] - faixa_v[0]) * (h - 50))
            if x0 <= u < x0 + meia and 24 <= v < h:
                t = np.clip((cor_z - PROF_MIN_M) / (PROF_MAX_M - PROF_MIN_M), 0, 1)
                canvas[v, u] = (int(60 + 180 * t), int(220 - 120 * t), int(240 - 200 * t))
        _texto(canvas, rotulo_h, x0 + meia - 60, h - 8, 0.35, (110, 110, 110))
        _texto(canvas, rotulo_v, x0 + 6, 34, 0.35, (110, 110, 110))

    _plota(x, z, 0, (-1.5, 1.5), (PROF_MIN_M, PROF_MAX_M), "X +/-1.5 m", "Z 0-5 m")
    _plota(y, z, meia, (-1.0, 1.0), (PROF_MIN_M, PROF_MAX_M), "Y +/-1.0 m", "Z 0-5 m")
    _texto(canvas, f"{xs.size} pts", 10, h - 8, 0.35, (110, 110, 110))
    return canvas


def desenha_temperatura(temperaturas: dict[str, float] | None, w: int, h: int,
                        alerta: float = 60.0, atencao: float = 45.0) -> np.ndarray:
    """Barras de temperatura por junta, com os dois limiares marcados.

    Os limiares são conservadores de propósito: os motores do G1 aguentam mais,
    mas numa sessão longa de teleoperação/inferência o que interessa é ver a
    subida ANTES do desarme, não descobrir depois.
    """
    canvas = np.full((h, w, 3), _FUNDO, dtype=np.uint8)
    _texto(canvas, "4. Temperatura dos motores", 10, 20, 0.45, (240, 240, 240))

    if not temperaturas:
        _texto(canvas, "sem telemetria (robo em --sim ou sem lowstate)",
               10, h // 2, 0.45, (110, 110, 110))
        return canvas

    itens = list(temperaturas.items())
    topo, base = 34, h - 16
    passo = max(9, (base - topo) // max(1, len(itens)))
    largura_barra = w - 150

    for i, (nome, valor) in enumerate(itens):
        y = topo + i * passo
        if y + 6 > base:
            _texto(canvas, f"... +{len(itens) - i} juntas", 10, base, 0.35, (110, 110, 110))
            break
        curto = nome.replace(".q", "").replace("_joint", "").replace("k", "", 1)[:18]
        _texto(canvas, curto, 8, y + 6, 0.32, (170, 170, 170))
        largura = int(np.clip(valor / 90.0, 0, 1) * largura_barra)
        cor = _VERMELHO if valor >= alerta else (_AMARELO if valor >= atencao else _VERDE)
        cv2.rectangle(canvas, (120, y), (120 + largura, y + max(4, passo - 3)), cor, -1)
        _texto(canvas, f"{valor:.0f}C", w - 28, y + 6, 0.32, cor)

    for limiar, cor in ((atencao, _AMARELO), (alerta, _VERMELHO)):
        x = 120 + int(limiar / 90.0 * largura_barra)
        cv2.line(canvas, (x, topo), (x, base), cor, 1)
    return canvas


# ══════════════════════════════════════════════════════════════════════════
# Janela
# ══════════════════════════════════════════════════════════════════════════
def _exige_opencv_com_gui() -> None:
    """Falha com instrução em vez de deixar a janela simplesmente não aparecer.

    O core do lerobot depende de `opencv-python-headless`, compilado com
    `GUI: NONE`. Com esse build, `cv2.namedWindow` levanta um erro genérico
    ("The function is not implemented") ou, dependendo da versão, não abre nada
    e o programa segue como se estivesse tudo bem — que é o pior desfecho:
    ninguém suspeita do OpenCV, todo mundo suspeita da câmera.
    """
    if "GUI:                           NONE" not in cv2.getBuildInformation():
        return
    raise RuntimeError(
        "Este OpenCV é o build 'headless' (GUI: NONE) — ele não consegue abrir "
        "janela nenhuma.\n"
        "   Conserto (os dois pacotes instalam o mesmo módulo cv2; vence o último):\n"
        "       pip install --no-deps 'opencv-python>=4.9.0,<4.14.0'\n"
        "   Confira com:\n"
        "       python -c \"import cv2; print(cv2.getBuildInformation())\" | grep GUI"
    )


class PainelDebug:
    """Uma janela OpenCV com os quatro quadrantes.

    `atualiza()` é barato e pode ser chamado todo ciclo; os quadrantes que não
    receberam dado novo simplesmente repetem o último desenho.
    """

    def __init__(self, nome: str = "FastWAM-D — Debug",
                 largura_quadrante: int = 480, altura_quadrante: int = 360,
                 intrinsecos: dict | None = None):
        if cv2 is None:
            raise RuntimeError("OpenCV não está instalado: pip install opencv-python")
        self.nome = nome
        self.qw = largura_quadrante
        self.qh = altura_quadrante
        self.intrinsecos = intrinsecos or dict(INTRINSECOS_PADRAO)

        self._rgb_mosaico: np.ndarray | None = None
        self._attn: np.ndarray | None = None
        self._attn_cru: np.ndarray | None = None
        self._attn_e_relativo = False
        self._attn_base_n = 0
        self._depth_mm: np.ndarray | None = None
        self._depth_modelo: np.ndarray | None = None
        self._temperaturas: dict[str, float] | None = None
        self._cabecalho = ""

    def create(self) -> None:
        _exige_opencv_com_gui()
        cv2.namedWindow(self.nome, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.nome, self.qw * 2, self.qh * 2 + 28)

    # ── entradas ────────────────────────────────────────────────────────────
    def define_imagens(self, rgb_mosaico=None, depth_mm=None) -> None:
        if rgb_mosaico is not None:
            self._rgb_mosaico = rgb_mosaico
        if depth_mm is not None:
            self._depth_mm = depth_mm

    def define_debug_servidor(self, payload: dict | None) -> None:
        if not payload:
            return
        # O relativo é o que responde "para onde ele olhou NESTE quadro": o mapa
        # cru é dominado pelo attention sink das primeiras colunas, que acende
        # igual em qualquer cena. Ver a nota em `LinhaDeBaseDaAtencao`, no
        # servidor. O cru fica guardado para quem quiser conferir.
        if payload.get("attn_rel") is not None:
            self._attn = np.asarray(payload["attn_rel"], dtype=np.float32)
            self._attn_e_relativo = True
            self._attn_base_n = int(payload.get("attn_base_n", 0))
        elif payload.get("attn") is not None:
            self._attn = np.asarray(payload["attn"], dtype=np.float32)
            self._attn_e_relativo = False
        if payload.get("attn") is not None:
            self._attn_cru = np.asarray(payload["attn"], dtype=np.float32)
        if payload.get("depth") is not None:
            self._depth_modelo = np.asarray(payload["depth"], dtype=np.float32)

    def define_temperaturas(self, temperaturas: dict[str, float] | None) -> None:
        if temperaturas:
            self._temperaturas = temperaturas

    def define_cabecalho(self, texto: str) -> None:
        self._cabecalho = texto

    # ── desenho ─────────────────────────────────────────────────────────────
    def show(self) -> bool:
        """Desenha e devolve False quando o usuário fecha a janela ou tecla q/ESC."""
        q1 = desenha_atencao(self._rgb_mosaico, self._attn, self.qw, self.qh,
                             relativo=self._attn_e_relativo, base_n=self._attn_base_n)
        q2 = desenha_profundidade(self._depth_mm, self._depth_modelo, self.qw, self.qh)
        q3 = desenha_nuvem(self._depth_mm, self.intrinsecos, self.qw, self.qh)
        q4 = desenha_temperatura(self._temperaturas, self.qw, self.qh)

        corpo = np.vstack([np.hstack([q1, q2]), np.hstack([q3, q4])])
        cabecalho = np.full((28, corpo.shape[1], 3), (16, 16, 16), dtype=np.uint8)
        _texto(cabecalho, self._cabecalho or "FastWAM-D", 10, 19, 0.45, (210, 210, 210))
        canvas = np.vstack([cabecalho, corpo])

        cv2.imshow(self.nome, canvas)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            return False
        # Janela fechada no X: o getWindowProperty vira <1.
        try:
            if cv2.getWindowProperty(self.nome, cv2.WND_PROP_VISIBLE) < 1:
                return False
        except cv2.error:
            return False
        return True

    def destroy(self) -> None:
        try:
            cv2.destroyWindow(self.nome)
        except cv2.error:
            pass
