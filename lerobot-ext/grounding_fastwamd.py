#!/usr/bin/env python
"""
Grounding do FastWAM-D — onde o TEXTO está olhando na imagem
=============================================================
Sem robô, sem simulador, sem servidor. Carrega o checkpoint, passa um quadro
gravado do dataset, e desenha em cima da imagem quanto da atenção de linguagem
de cada pedaço da cena vai para as palavras que você escolher.

    "pegue a caneca branca"  →  ele pinta a caneca branca?

O QUE ISTO MEDE (e o que não mede)
-----------------------------------
Mede o prior de linguagem do **Wan2.2**, não o que aprendemos aqui. O dataset
tem UMA string de tarefa só ("place the white cup on the dripper", idêntica nos
24 episódios) e o treino rodou com `freeze_video_expert: true` — a
cross-attention com o texto mora justamente nos blocos congelados. Nada no
nosso fine-tune ensinou associação texto↔imagem, e nada podia: sem dois
comandos competindo no mesmo batch, o texto é constante e portanto irrelevante.

A consequência boa é que este teste vale com prompt INVENTADO, que nunca
apareceu no treino. A pergunta que ele responde é: o pré-treino em vídeo do Wan
sabe onde está a caneca? Se souber, condicionamento por linguagem é uma questão
de dados de fine-tune. Se não souber, é uma questão de arquitetura.

O NÚMERO QUE APARECE NA FIGURA
-------------------------------
NÃO é atenção crua. Atenção crua aqui mente por DOIS sumidouros empilhados:

  1. **Padding.** O tokenizer preenche até `tokenizer_max_len` (128) e o
     `encode_prompt` (`wan/modular.py:1036`) faz `mask = torch.ones_like(mask)`
     — a cross-attention enxerga as ~100 posições de padding, cujo embedding é
     zero mas cuja chave, depois do viés e do RMSNorm, não é. Medido neste
     checkpoint: **87% da massa vai para o padding.**
  2. **Tokens estruturais.** Dentro dos 13% que sobram, um punhado de tokens
     (o primeiro, o `</s>`) leva quase tudo, em TODO patch. Uma renormalização
     sobre os tokens reais não resolve isto: só troca um sumidouro por outro.

A métrica padrão (`--metrica=espacial`) é a mesma que o painel de ação usa
contra o attention sink: **contraste espacial por token**.

    espacial[p, t] = A[p, t] / média_p(A[p, t])

Ou seja: quanto o patch `p` olha para a palavra `t`, dividido por quanto o
patch MÉDIO olha para ela. Um token sumidouro puxa massa em todo lugar, então
vira 1,0 uniforme e desaparece sozinho; o que sobra é só a variação espacial, e
variação espacial é literalmente a pergunta "onde na imagem está a caneca".

    1,0 = este patch olha para a palavra tanto quanto qualquer outro
    2,0 = olha o dobro do patch médio  → é aqui que a palavra "aterrissa"

A outra (`--metrica=condicional`) é a distribuição P(token | patch)
renormalizada sobre os tokens reais, contra a linha de base do indiferente
(`n_tokens_da_frase / n_tokens_reais`). Ela responde outra pergunta — "deste
patch, que fração da atenção de linguagem vai para a frase" — e é dominada
pelos tokens estruturais. Fica disponível porque é a leitura honesta da massa
absoluta, mas não é a que mostra grounding.

USO (na athena, GPU com ~28 GB livres)
---------------------------------------
Pelo launcher, que já resolve as armadilhas de ambiente (os DOIS caches do
HuggingFace separados, o python por caminho absoluto do home de quem rodou, a
GPU, o `OMP_NUM_THREADS`) — ver `athena/README.md`:

    bash athena/launch_grounding.sh 2 --frases="white cup,dripper,robot"

À mão, se precisar. Note `HF_HUB_CACHE` (leitura dos 25 GB de pesos, comum a
todos) separado de `HF_HOME` (escrita: locks do `datasets`) — apontar os dois
para o `/data` funciona só para o dono daquela pasta, e falha com
`PermissionError` depois de dois minutos carregando o modelo:

    cd ~/DEV/prometheus-vla/lerobot-ext
    HF_HUB_CACHE=/data/.cache/huggingface/hub HF_HOME=$HOME/.cache/huggingface \
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 \
    python grounding_fastwamd.py \
        --checkpoint=/data/train_output/fastwamd_corrida2_step1000/pretrained_model \
        --quadros=0,60,120,180 \
        --frases="white cup,dripper,robot"

Opções:
  --checkpoint=<PATH>  pretrained_model do FastWAM-D (obrigatório)
  --root=<PATH>        raiz do dataset (padrão: meu_dataset/white_cup_on_dripper_2026-08-11)
  --episodio=<INT>     episódio de onde tirar os quadros (padrão: 25)
  --quadros=<n,n,...>  quais quadros do episódio (padrão: 0,60,120,180)
  --task=<STR>         a instrução, ANTES do `prompt_template` do config
                       (padrão: "place the white cup on the dripper")
  --frases=<a,b,...>   as frases a destacar; cada uma vira uma coluna da figura.
                       Precisam aparecer literalmente na `--task`, senão o
                       script lista o que dá para escolher e para.
  --camadas=<a:b>      fatia dos blocos do expert de vídeo a promediar
                       (padrão: todos). Ex: `--camadas=10:20` só o miolo.
  --metrica=<STR>      `espacial` (padrão) ou `condicional` — ver acima.
  --por-camada         em vez da figura normal, uma linha por bloco, para achar
                       em qual profundidade o grounding aparece.
  --saida=<PATH>       PNG de saída (padrão: ./grounding_fastwamd.png)
  --device=<STR>       cuda / cuda:1 / cpu (padrão: cuda se houver)
  -h, --help           esta mensagem

O QUE OLHAR
-----------
Contraste alto e ESPACIALMENTE COERENTE em cima do objeto nomeado é grounding.
Contraste alto espalhado, ou grudado numa borda/canto da imagem, é artefato —
o mesmo padrão para qualquer frase significa que o modelo não está lendo o
texto, está só distribuindo atenção do jeito que sempre distribui. Por isso a
figura sempre traz pelo menos duas frases: a comparação entre colunas é a
medida, não o valor absoluto de uma coluna só.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RAIZ_PADRAO = "meu_dataset/white_cup_on_dripper_2026-08-11"
TAREFA_PADRAO = "place the white cup on the dripper"


# ─────────────────────────────────────────────────────────────────────
# Captura
# ─────────────────────────────────────────────────────────────────────
class CapturaGrounding:
    """Colhe a cross-attention vídeo→texto de UMA inferência.

    Mesma estratégia do `CapturaDebug`: troca dois métodos por fora, colhe, e
    devolve os originais no fim. O modelo não sabe que isto existe.

    A diferença é qual atenção se olha. O `CapturaDebug.mapa_atencao` pega
    ação→vídeo, dentro da self-attention mista do MoT ("que pedaço da cena
    decidiu esta ação"). Aqui é vídeo→texto, dentro da cross-attention de cada
    bloco ("que palavra este pedaço da cena está consultando").
    """

    def __init__(self, policy):
        self.policy = policy
        self.grade: tuple[int, int, int] | None = None    # (T, H, W) de tokens
        self.por_camada: list[np.ndarray] = []            # cada um [S_video, L_texto]
        self._patchify_original = None
        self._cross_original = None

    # ── contexto ────────────────────────────────────────────────────────────
    def __enter__(self) -> "CapturaGrounding":
        self._instala_gancho_grade()
        self._instala_gancho_cross()
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()

    def remove(self) -> None:
        if self._patchify_original is not None:
            self.policy._expert_video.patchify = self._patchify_original
            self._patchify_original = None
        if self._cross_original is not None:
            from lerobot.policies.fastwam.wan.video_dit import FastWAMAttentionBlock

            FastWAMAttentionBlock.apply_cross_attention = self._cross_original
            self._cross_original = None

    # ── ganchos ─────────────────────────────────────────────────────────────
    def _instala_gancho_grade(self) -> None:
        """Anota a grade de tokens lendo a SAÍDA do `patch_embedding`.

        Idêntico ao do `CapturaDebug`, e pelo mesmo motivo: calcular
        `latente / patch_size` aqui seria repetir uma conta que já é feita lá e
        que muda junto com o patch.
        """
        expert = self.policy._expert_video
        original = expert.patchify
        self._patchify_original = original

        def patchify_anotada(x: torch.Tensor):
            saida = original(x)
            if saida.dim() == 5:  # [B, dim, T, H, W]
                self.grade = tuple(int(v) for v in saida.shape[2:])
            return saida

        expert.patchify = patchify_anotada

    def _instala_gancho_cross(self) -> None:
        from lerobot.policies.fastwam.wan.video_dit import FastWAMAttentionBlock

        original = FastWAMAttentionBlock.apply_cross_attention
        self._cross_original = original
        captura = self

        def cross_com_captura(self_bloco, x, context, context_mask=None):
            captura._anota(self_bloco, x, context)
            return original(self_bloco, x, context, context_mask=context_mask)

        FastWAMAttentionBlock.apply_cross_attention = cross_com_captura

    def _anota(self, bloco, x: torch.Tensor, context: torch.Tensor) -> None:
        """Recalcula os pesos de cross-attention deste bloco.

        Recalcular em vez de interceptar é obrigatório: o
        `fastwam_masked_attention` chama SDPA, que devolve só o resultado — a
        matriz de pesos nunca existe como tensor. As contas abaixo são cópia
        fiel do `video_dit.py::apply_cross_attention`; qualquer divergência
        (esquecer o `norm_q`, por exemplo) daria um mapa plausível e errado.

        Só o expert de VÍDEO interessa. Ele se distingue pelo comprimento: as
        consultas dele são a grade de patches (centenas), as do expert de ação
        são o horizonte (32). Sem a grade anotada ainda, não dá para decidir, e
        a chamada é ignorada.
        """
        if self.grade is None:
            return
        s_video = int(np.prod(self.grade))
        if int(x.shape[1]) != s_video:
            return

        try:
            attn = bloco.cross_attn
            n, d = int(attn.num_heads), int(attn.head_dim)

            q = attn.norm_q(attn.q(x.to(attn.q.weight.dtype)))
            k = attn.norm_k(attn.k(context.to(attn.k.weight.dtype)))
            # [B, n, S, d] — mesma partição de cabeças do `fastwam_masked_attention`.
            q = q.detach().view(q.shape[0], q.shape[1], n, d).permute(0, 2, 1, 3).float()
            k = k.detach().view(k.shape[0], k.shape[1], n, d).permute(0, 2, 1, 3).float()

            escala = 1.0 / np.sqrt(d)
            pesos = torch.softmax(q @ k.transpose(-1, -2) * escala, dim=-1)  # [B, n, S, L]
            # Média sobre batch e cabeças. Média entre cabeças esconde a cabeça
            # especialista que às vezes existe, mas somar cabeças com papéis
            # diferentes é o comportamento efetivo da camada — e é o que a
            # saída dela de fato carrega para a frente.
            self.por_camada.append(pesos.mean(dim=(0, 1)).cpu().numpy())
        except Exception as erro:  # noqa: BLE001 — depuração nunca derruba a inferência
            print(f"⚠️  cross-attention não capturada num bloco: {erro}")


# ─────────────────────────────────────────────────────────────────────
# Texto → índices de token
# ─────────────────────────────────────────────────────────────────────
def tokens_do_prompt(policy, prompt: str):
    """`(lista de tokens legíveis, offsets em caracteres)` do prompt final.

    Usa o tokenizer de verdade do modelo (`WanTokenizer.tokenizer` é um
    `AutoTokenizer` do HF), não uma aproximação por espaços: o UMT5 quebra
    "dripper" em pedaços, e escolher os índices errados é escolher outra
    palavra sem perceber.
    """
    tok = policy.model.tokenizer.tokenizer
    enc = tok(prompt, add_special_tokens=True, return_offsets_mapping=True)
    ids = enc["input_ids"]
    return tok.convert_ids_to_tokens(ids), enc["offset_mapping"], len(ids)


def indices_da_frase(prompt: str, offsets, frase: str) -> list[int]:
    """Índices dos tokens que cobrem `frase` dentro de `prompt`."""
    inicio = prompt.lower().find(frase.lower())
    if inicio < 0:
        return []
    fim = inicio + len(frase)
    return [
        i for i, (a, b) in enumerate(offsets)
        if b > a and a < fim and b > inicio  # sobreposição de intervalos, ignorando tokens vazios
    ]


# ─────────────────────────────────────────────────────────────────────
# Mapa
# ─────────────────────────────────────────────────────────────────────
def mapa_de_contraste(pesos: np.ndarray, indices: list[int], n_real: int,
                      grade: tuple[int, int, int], metrica: str = "espacial") -> tuple[np.ndarray, float]:
    """`([h, w] de contraste, massa que foi para o padding)`.

    `pesos` é [S_video, L_texto] direto do softmax, ou seja, ainda com as
    posições de padding dentro. Ver o cabeçalho do arquivo para o porquê de
    cada passo daqui.
    """
    reais = pesos[:, :n_real]
    padding = float(1.0 - reais.sum(axis=1).mean())
    t, h, w = grade

    if metrica == "espacial":
        # Cada coluna dividida pela SUA média espacial. O token sumidouro puxa
        # massa em todo patch, então sua coluna vira ~1,0 em todo lugar e sai da
        # conta sozinha — sem precisar identificá-lo nem removê-lo à mão.
        colunas = reais[:, indices]
        media = colunas.mean(axis=0, keepdims=True)
        contraste = (colunas / np.clip(media, 1e-12, None)).mean(axis=1)
    elif metrica == "condicional":
        massa = reais.sum(axis=1, keepdims=True)
        condicional = reais / np.clip(massa, 1e-8, None)
        contraste = condicional[:, indices].sum(axis=1) / (len(indices) / n_real)
    else:
        raise ValueError(f"métrica desconhecida: {metrica!r} (espacial | condicional)")

    return contraste.reshape(t, h, w).mean(axis=0), padding


def diagnostico_do_texto(pesos: np.ndarray, nomes: list[str], n_real: int, topo: int = 6) -> str:
    """Quais tokens reais levam a massa, em média sobre os patches.

    Existe porque a primeira versão deste script desenhou contraste < 1 em toda
    a imagem e a causa não estava visível na figura: dentro dos tokens reais há
    um segundo sumidouro. Sem esta linha impressa, o mapa parece "sem
    grounding" quando na verdade está medindo a coisa errada.
    """
    medias = pesos[:, :n_real].mean(axis=0)
    ordem = np.argsort(medias)[::-1][:topo]
    total = float(medias.sum())
    return " ".join(f"{nomes[i]}={medias[i] / max(total, 1e-12):.0%}" for i in ordem)


def redimensiona(mapa: np.ndarray, altura: int, largura: int) -> np.ndarray:
    """Grade de tokens → tamanho da imagem, por interpolação bilinear."""
    t = torch.from_numpy(mapa).float()[None, None]
    return torch.nn.functional.interpolate(
        t, size=(altura, largura), mode="bilinear", align_corners=False
    )[0, 0].numpy()


# ─────────────────────────────────────────────────────────────────────
# Dados e modelo
# ─────────────────────────────────────────────────────────────────────
def carrega_politica(checkpoint: str, device: torch.device):
    from policies.fastwam_depth.configuration_fastwam_depth import FastWAMDepthConfig
    from policies.fastwam_depth.modeling_fastwam_depth import FastWAMDepthPolicy

    print(f"⏳ Carregando FastWAM-D de: {checkpoint}")
    config = FastWAMDepthConfig.from_pretrained(checkpoint)
    politica = FastWAMDepthPolicy.from_pretrained(checkpoint, config=config)
    politica.to(device)
    politica.eval()

    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, _ = make_pre_post_processors(policy_cfg=config, pretrained_path=checkpoint)
    print(f"✅ carregado — depth_mode={config.depth_mode} | câmeras={config.rgb_feature_keys}")
    return politica, config, preprocessor


def abre_dataset(raiz: str, episodio: int):
    # O `LeRobotDataset` consulta o Hub para resolver a versão MESMO com `root`
    # local, e um repo_id inventado devolve 401. Offline é a verdade aqui.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id=os.path.basename(os.path.normpath(raiz)), root=raiz, episodes=[episodio]
    )


def monta_batch(amostra: dict, config, task: str) -> dict:
    """Observação crua a partir de um item do dataset, no formato do `monta_obs`.

    O dataset já entrega exatamente o que o preprocessor espera: cor em [0, 1]
    e profundidade de 1 canal em MILÍMETROS. Nada de dividir a profundidade por
    255 — isso não quebra nada visivelmente e destrói a escala métrica.
    """
    raw: dict = {"observation.state": amostra["observation.state"].float()}
    for chave in config.rgb_feature_keys:
        raw[chave] = amostra[chave].float()
    for chave in config.depth_feature_keys:
        raw[chave] = amostra[chave].float()
    raw["task"] = task
    return raw


def imagem_de(amostra: dict, chave: str) -> np.ndarray:
    a = amostra[chave].numpy()
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return a


# ─────────────────────────────────────────────────────────────────────
# Figura
# ─────────────────────────────────────────────────────────────────────
def desenha(linhas: list[dict], frases: list[str], prompt: str, saida: str,
            metrica: str = "espacial", rotulo_linha: str = "quadro") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_l, n_c = len(linhas), len(frases) + 1
    fig, eixos = plt.subplots(n_l, n_c, figsize=(4.2 * n_c, 2.6 * n_l), squeeze=False)

    # Escala de cor comum a TODOS os painéis: sem isso cada mapa é normalizado
    # sozinho e duas frases com grounding muito diferente parecem iguais — que é
    # exatamente a comparação que este script existe para permitir.
    #
    # E centrada em 1,0, que é o neutro das duas métricas: assim "acima da
    # média" e "abaixo da média" têm a mesma extensão de cor e o olho não lê
    # como quente um mapa que só tem ruído em volta de 1.
    todos = np.concatenate([m.ravel() for linha in linhas for m in linha["mapas"]])
    desvio = float(max(0.15, np.percentile(np.abs(todos - 1.0), 99)))
    vmin, vmax = 1.0 - desvio, 1.0 + desvio

    for i, linha in enumerate(linhas):
        eixos[i][0].imshow(linha["imagem"])
        eixos[i][0].set_ylabel(f"{rotulo_linha} {linha['rotulo']}", fontsize=9)
        eixos[i][0].set_title("head_camera" if i == 0 else "", fontsize=9)
        eixos[i][0].set_xticks([]); eixos[i][0].set_yticks([])

        for j, (frase, mapa) in enumerate(zip(frases, linha["mapas"])):
            ax = eixos[i][j + 1]
            ax.imshow(linha["imagem"])
            im = ax.imshow(mapa, cmap="coolwarm", alpha=0.55, vmin=vmin, vmax=vmax)
            if i == 0:
                ax.set_title(f'"{frase}"', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.02, 0.96, f"{mapa.min():.2f}–{mapa.max():.2f}×", transform=ax.transAxes,
                    fontsize=8, color="white", va="top",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1.5, edgecolor="none"))

    rotulo = {
        "espacial": "atenção deste patch à palavra ÷ atenção do patch médio  (1,0 = uniforme)",
        "condicional": "P(frase | patch) ÷ indiferente  (1,0 = não liga)",
    }[metrica]
    barra = fig.colorbar(im, ax=eixos, fraction=0.015, pad=0.01)
    barra.set_label(rotulo, fontsize=9)
    fig.suptitle(f"FastWAM-D — grounding do texto na imagem  ({metrica})\nprompt: {prompt}",
                 fontsize=11)
    fig.savefig(saida, dpi=130, bbox_inches="tight")
    print(f"\n💾 {saida}")


# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    checkpoint = None
    raiz, episodio = RAIZ_PADRAO, 25
    quadros = [0, 60, 120, 180]
    task = TAREFA_PADRAO
    frases = ["white cup", "dripper"]
    camadas = None
    por_camada = False
    metrica = "espacial"
    saida = "grounding_fastwamd.png"
    device_str = None

    for arg in sys.argv[1:]:
        if arg in ("-h", "--help"):
            print(__doc__)
            return
        elif arg.startswith("--checkpoint="):
            checkpoint = arg.split("=", 1)[1]
        elif arg.startswith("--root="):
            raiz = arg.split("=", 1)[1]
        elif arg.startswith("--episodio="):
            episodio = int(arg.split("=", 1)[1])
        elif arg.startswith("--quadros="):
            quadros = [int(v) for v in arg.split("=", 1)[1].split(",") if v.strip()]
        elif arg.startswith("--task="):
            task = arg.split("=", 1)[1]
        elif arg.startswith("--frases="):
            frases = [v.strip() for v in arg.split("=", 1)[1].split(",") if v.strip()]
        elif arg.startswith("--camadas="):
            a, _, b = arg.split("=", 1)[1].partition(":")
            camadas = (int(a) if a else None, int(b) if b else None)
        elif arg == "--por-camada":
            por_camada = True
        elif arg.startswith("--metrica="):
            metrica = arg.split("=", 1)[1]
        elif arg.startswith("--saida="):
            saida = arg.split("=", 1)[1]
        elif arg.startswith("--device="):
            device_str = arg.split("=", 1)[1]
        else:
            print(f"❌ opção desconhecida: {arg}  (use --help)")
            sys.exit(2)

    if not checkpoint:
        print("❌ --checkpoint é obrigatório. Ex:\n"
              "   --checkpoint=/data/train_output/fastwamd_corrida2_step1000/pretrained_model")
        sys.exit(2)

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🖥️  Device: {device}")

    politica, config, preprocessor = carrega_politica(checkpoint, device)
    ds = abre_dataset(raiz, episodio)
    print(f"🎬 {raiz} episódio {episodio} — {len(ds)} quadros")

    prompt = config.prompt_template.format(task=task)
    nomes, offsets, n_real = tokens_do_prompt(politica, prompt)
    print(f"\n📝 prompt final ({n_real} tokens reais de {config.tokenizer_max_len} posições):")
    print(f"   {prompt}")
    print(f"   tokens: {' '.join(nomes)}\n")

    indices = {}
    for frase in frases:
        idx = indices_da_frase(prompt, offsets, frase)
        if not idx:
            print(f"❌ a frase {frase!r} não aparece no prompt final.\n"
                  f"   O prompt é montado pelo `prompt_template` do config:\n"
                  f"     {prompt}\n"
                  f"   Escolha frases que apareçam literalmente nele (ou mude --task).")
            sys.exit(2)
        indices[frase] = idx
        print(f"   {frase!r} → tokens {idx} = {[nomes[i] for i in idx]}")

    # A `head_camera` é a metade ESQUERDA do mosaico porque `rgb_feature_keys`
    # ordena alfabeticamente e `head_camera` < `right_wrist_camera`. Derivar em
    # vez de fixar: acrescentar uma câmera muda a ordem sem avisar.
    chave_head = config.rgb_feature_keys[0]
    n_cams = len(config.rgb_feature_keys)

    linhas = []
    for quadro in quadros:
        if quadro >= len(ds):
            print(f"⚠️  quadro {quadro} não existe (episódio tem {len(ds)}), pulando.")
            continue
        amostra = ds[quadro]
        batch = preprocessor(monta_batch(amostra, config, task))
        batch.pop("action", None)

        captura = CapturaGrounding(politica)
        with torch.inference_mode(), captura:
            politica.predict_action_chunk(batch)

        if not captura.por_camada or captura.grade is None:
            print("❌ nenhuma cross-attention de vídeo capturada. O expert de vídeo "
                  "rodou? (com `depth_mode=off` e cache de vídeo ele pode não ter "
                  "passado pelo bloco neste chunk).")
            sys.exit(1)

        pilha = np.stack(captura.por_camada)  # [n_blocos, S_video, L_texto]
        if camadas is not None:
            pilha = pilha[slice(*camadas)]
        pesos = pilha.mean(axis=0)

        t, h, w = captura.grade
        # Só a fatia da câmera da cabeça: o mosaico tem as duas câmeras lado a
        # lado na largura, e sobrepor o mapa inteiro numa imagem só desloca tudo.
        w_head = w // n_cams
        imagem = imagem_de(amostra, chave_head)

        mapas, padding = [], 0.0
        for frase in frases:
            mapa, padding = mapa_de_contraste(pesos, indices[frase], n_real, captura.grade, metrica)
            mapas.append(redimensiona(mapa[:, :w_head], imagem.shape[0], imagem.shape[1]))

        print(f"   quadro {quadro:4d}: grade {t}×{h}×{w} tokens | "
              f"{len(captura.por_camada)} blocos | padding levou {padding:.0%} | "
              + " | ".join(f"{f!r} {m.min():.2f}–{m.max():.2f}×" for f, m in zip(frases, mapas)))
        print(f"              massa entre os tokens reais: "
              f"{diagnostico_do_texto(pesos, nomes, n_real)}")

        linhas.append({"rotulo": str(quadro), "imagem": imagem, "mapas": mapas,
                       "pesos": pesos, "pilha": np.stack(captura.por_camada),
                       "grade": captura.grade, "w_head": w_head})

    if not linhas:
        print("❌ nenhum quadro válido.")
        sys.exit(1)

    if por_camada:
        # Uma linha por bloco, primeiro quadro só: onde no DiT o grounding nasce.
        base = linhas[0]
        n_blocos = base["pilha"].shape[0]
        linhas = []
        for b in range(n_blocos):
            mapas = [
                redimensiona(
                    mapa_de_contraste(
                        base["pilha"][b], indices[f], n_real, base["grade"], metrica
                    )[0][:, : base["w_head"]],
                    base["imagem"].shape[0], base["imagem"].shape[1],
                )
                for f in frases
            ]
            linhas.append({"rotulo": str(b), "imagem": base["imagem"], "mapas": mapas})

        # A tabela é o resultado; a figura é só para olhar depois. O que decide
        # se existe grounding em ALGUM bloco é a comparação com a frase de
        # controle bloco a bloco — uma média sobre 30 blocos esconde um pico.
        print("\n   bloco | " + " | ".join(f"{f[:12]:>12}" for f in frases))
        for b, linha in enumerate(linhas):
            print(f"   {b:5d} | " + " | ".join(f"{m.max():11.2f}×" for m in linha["mapas"]))
        melhor = {
            f: max(range(len(linhas)), key=lambda b: linhas[b]["mapas"][j].max())
            for j, f in enumerate(frases)
        }
        print("\n   pico por frase: " + " | ".join(f"{f!r} no bloco {b}" for f, b in melhor.items()))

        desenha(linhas, frases, prompt, saida, metrica, rotulo_linha="bloco")
    else:
        desenha(linhas, frases, prompt, saida, metrica)


if __name__ == "__main__":
    main()
