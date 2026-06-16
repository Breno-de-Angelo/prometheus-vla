# Resposta à review — verificação contra o código real

> **Contexto.** A review (`Análise de Falhas em Políticas Robóticas.docx`) foi feita só com o
> `LOG.md`/`RELATORIO.md` que enviamos — **sem acesso ao código**. Abaixo confronto cada tese
> central com o **código e os configs reais** do projeto (π0.5 no Unitree G1 Dex3, *"pick up the
> white cup"*), apontando arquivo:linha. A review é um **bom mapa da literatura de VLA** e acerta
> o quadro clínico, mas **duas das suas teses causais não batem com o nosso setup**.
>
> Data: 2026-06-13. Verificado contra: `lerobot-ext/policies/pi0_depth/modeling_pi05.py`,
> `inference_realtime_pi05d_right14.py`, `lerobot-ext/config/train/train_cup_pi05_right14_rgb_238.yaml`,
> `train_cup_pi05_right8_1squeeze.yaml`.

---

## 1. Onde a review ACERTOU

- **Overfitting extremo é a falha dominante.** Bate com os dados: `train/loss → ~0,008` enquanto a
  `val_loss` *sobe* (0,086 → 0,317 no 14-dim; 0,098 → 0,363 no 8-dim). O A/B `best`-5500 vs
  `last`-20000 deu só ~11% de diferença em sim, e **ambos os `best` falharam no robô real**.
- **O colapso de *visual grounding* é real** (como sintoma): o 8-dim fechou a mão no ar vazio,
  longe do copo — a ação não está condicionada à imagem.
- **A mão de 1-DOF efetivo (squeeze) limita a preensão.** Mesmo o right14 emite 7 juntas de dedo,
  mas o rótulo carrega **um sinal só** (`squeeze × RIGHT_TARGET`). Nenhum dos dois aprende controle
  independente de dedos — é limitação **dos dados**, e a review está certa em apontá-la.
- **Usamos joint-space + ações absolutas** — fato correto (os `action.names` terminam em `.q`;
  nenhuma dimensão é xyz/rpy de end-effector).
- **O alerta contra split frame-level** é correto *em geral* (só não se aplica a nós — ver §2c).
- **Co-training (Open-X / Bridge) e RL sim-real** são técnicas estabelecidas e plausíveis — mas
  como passo *posterior*, não quick-fix (ver §3).

---

## 2. Onde a review ERROU ou EXTRAPOLOU (por ter visto só o `LOG.md`, não o código)

### (a) "Roda a ~3 fps, sem action chunking, inferência *step-wise* → jitter" — **REFUTADO**

- **A review diz:** a mão cicla porque a inferência é passo-a-passo, sem chunking, a 3 fps, com
  observações obsoletas; a solução seria implementar **ACT (action chunking) + Temporal Ensembling**.
- **O código mostra:**
  - A política gera um **chunk de 50 ações** (`predict_action_chunk`) em **ambos** os loops — o
    síncrono (`inference_realtime_pi05d_right14.py:631` / `:634`) e o RTC (`:563-567`). **Não existe
    caminho autoregressivo passo-a-passo no código.**
  - O **controle é a 30 fps** (`step_period = 1.0/args.fps`, fps=30, `:417`) e a **mão a 100 Hz**
    via hand-streamer. Os ~3 fps são a **taxa de replanejamento/inferência** (≈1 chunk a cada
    ~333 ms), **não** a taxa de envio ao robô. As ações do chunk são consumidas a 30 fps por uma
    fila enquanto a próxima inferência roda em *thread* separada — não há "ação obsoleta de 333 ms".
  - **ACT + Temporal Ensembling já estão implementados** via **RTC (Real-Time Chunking)**: a thread
    de inferência prediz o chunk com `inference_delay` e `prev_chunk_left_over` (`:563-567`) e faz
    **inpainting** na emenda (guidance de flow-matching, `--rtc-execution-horizon 25
    --rtc-max-guidance 1.0`). O RTC **estava ligado** nas runs do robô.
  - O **8-dim, com o mesmíssimo chunking e o mesmo código, *não* cicla** (fecha 1× e segura). Se a
    causa fosse "falta de chunking", os **dois** ciclariam.
- **Conclusão:** o jitter do 14-dim está **dentro do chunk predito** (uma sequência de 50 ações que
  oscila abre→fecha), ou seja, é **aprendizado errado** (overfit + grounding pobre), **não artefato
  do loop de inferência**. A alavanca proposta (implementar chunking) já existe.

### (b) "O fine-tuning erodiu o VLM (*representational drift*) → VRA / *Don't Blind Your VLA*" — **REFUTADO na causa**

- **A review diz:** o VLM "esqueceu o que é um copo"; corrigir alinhando as features contra um
  professor SigLIP/DINO congelado (VRA).
- **O código mostra:** com `train_expert_only=true` (nos dois YAMLs), `_set_requires_grad()` faz
  `self.paligemma.eval()` + `requires_grad=False` em **todos** os parâmetros da PaliGemma —
  **incluindo o `vision_tower`** (`modeling_pi05.py:467-470`). Isso é **mais restritivo** que o
  `freeze_vision_encoder` e **vence** o `freeze_vision_encoder=false` que está no config. **O
  encoder de visão não recebe gradiente algum → não pode sofrer *drift*.**
- **Conclusão:** a falha de grounding é **real**, mas a causa proposta é **impossível** neste
  setup. VRA resolve um problema (VLM treinando e degradando) que **não ocorre aqui**. A causa real
  do grounding falho é o **action expert (Gemma-300M, o único que treina) decorando atalhos**
  (frame-number / fundo do laboratório) sem acoplamento visual-motor — agravado por multimodalidade
  nos dados (mesma imagem → ações diferentes entre demos).
- **Ressalva honesta:** VRA *seria* relevante num cenário `train_expert_only=false`, que **não
  testamos**.

### (c) "Split aleatório em nível de *frame* causa *data leakage*" — **NÃO SE APLICA**

- **O código mostra:** o split é por **episódio** (trajectory-level) e **sequencial**: treino =
  eps `[0..213]`, val = eps `[214..237]` (nos dois YAMLs). O filtro do LeRobot é por
  `episode_index` — **nenhum frame vizinho vaza** entre treino e val.
- **Conclusão:** já fazemos o que a review recomenda como *correto*. A ironia: por ser
  **sequencial**, ele introduz **outro** risco (domain shift treino→val) — que nós **medimos** e
  achamos **leve / não-dominante**; o efeito dominante segue sendo overfitting.

---

## 3. Pontos GENUINAMENTE ACIONÁVEIS (priorizados)

1. **Mais dados e mais diversidade** (alvo 500+ eps, cenas/posições variadas). 238 episódios é
   pouco para o action expert decorar sem generalizar — **alavanca #1**.
2. **Repensar o encoding da mão** (sair do squeeze 1-DOF). Sem isso, *nenhuma* arquitetura aprende
   preensão fina — é teto **de dados**, não de modelo.
3. **Regularização contra overfit:** early-stopping mais agressivo e/ou **split aleatório por
   trajetória** (mantendo o átomo = episódio) — remove o domain-shift sequencial e dá um val honesto.
4. **Sinal/loss específico de *grounding* do grasp** (quando/onde fechar, condicionado à imagem) —
   ataca diretamente o sintoma que a review descreveu corretamente.
5. **Co-training (Open-X / Bridge) e RL sim-real:** válidos, mas **depois** dos itens 1–4. Não foram
   usados nos treinos relatados (só os 238 eps); RL exige um sim confiável que ainda não temos.

---

## 4. O que ainda precisamos VERIFICAR/medir

- **Contagem exata de params treináveis.** O `train_expert_only` congela toda a PaliGemma
  (`modeling_pi05.py:467-470`); os treináveis deveriam ser só o action expert + projeções
  (`action_in/out_proj`, `time_mlp`). O "~693 M de ~3,6 B" citado no relatório precisa ser
  reconciliado com isso (~300 M do expert + projeções).
- **O RTC está suprimindo o jitter ou não?** O jitter do 14-dim ocorreu **mesmo com RTC ligado**.
  Medir: gravar o **chunk normalizado bruto** (antes do merge) e ver se as 50 ações já oscilam na
  saída do modelo (confirma "aprendizado errado") vs. oscilação introduzida na emenda (improvável,
  dado o inpainting).
- **Taxa real de gravação do `policy_view.mp4`.** Hoje ele segue o frame rate da câmera amostrada
  por observação. Logar timestamp por frame para separar "taxa de replanejamento" de "taxa de
  captura" e não confundir uma com a outra.
- **Multimodalidade nos dados.** Medir a variância de ação para estados visuais quase-idênticos
  entre episódios (quantifica o "mesma imagem → ações diferentes").

---

## 5. Veredito (uma frase)

A review é **útil como mapa da literatura de VLA** e acerta o quadro clínico (overfitting +
grounding falho + mão 1-DOF), mas as suas duas teses causais centrais — **drift do VLM**
(impossível: a PaliGemma está congelada por `train_expert_only`, vision tower incluso) e
**ausência de chunking / inferência *step-wise* a 3 fps** (falso: usamos action chunking nativo de
50 + RTC com inpainting, controle a 30 fps / mão 100 Hz) — **não batem com o nosso código**; o foco
real é **mais e melhores dados + reencodar a mão + regularizar o overfit**, não VRA nem implementar
um chunking que já existe.
