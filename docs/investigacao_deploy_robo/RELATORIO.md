# Relatório — Treino e deploy das políticas π0.5 (G1 Dex3 · "pick up the white cup")

> Documento para revisão (orientador / revisor de artigo). Duas partes: **(1)** metodologia de
> treino das duas políticas; **(2)** investigação de deploy no **robô real** dos dois melhores
> checkpoints (`best`), com o resultado físico. Todos os números vêm do dado bruto (wandb API,
> logs de inferência, vídeos `policy_view.mp4`, metadados dos checkpoints).
>
> Data: 2026-06-13 · VLA na Atena (A100, env `ms3`) · robô G1 em `10.9.8.73`.

---

# PARTE 1 — Metodologia de treino

## 1. Tarefa, robô e demonstrações
- **Robô:** Unitree G1 humanoide, mão **Dex3** (3 dedos), braço **direito**.
- **Tarefa:** *"pick up the white cup"* — mão aberta em posição variável → alcançar → fechar a mão no copo → levantar. Sequência única (single-pick), ~10 s/episódio.
- **Coleta:** teleoperação por VR. **238 episódios**, **52.952 frames** @ **30 fps**.
- **Sensores gravados:** `head_camera` RGB 848×480, `head_camera_depth` (uint16, mm), pressão tátil Dex3 (108/mão). **Estes treinos usam só RGB + estado** (ablação RGB-puro).

## 2. Modelo base e fine-tuning
- **Base:** **π0.5** (`lerobot/pi05_base`) — VLM **PaliGemma (Gemma-2B)** + **action expert Gemma-300M**; política de **flow-matching** (chunk de 50 ações).
- **Fine-tuning:** `train_expert_only=true` → VLM **congelada**; treinam **~693 M de ~3,6 B** params. `freeze_vision_encoder=false`. `bfloat16` + gradient checkpointing.
- **Entrada:** `head_camera` RGB `[3,480,848]` (→224×224 no PaliGemma) + `observation.state` `[14]` (7 braço + 7 dedos medidos). `n_obs_steps=1`.
- **Saída:** `action` (14 ou 8 dims, §4). `chunk_size=50`, `n_action_steps=50`.
- **Normalização:** VISUAL=IDENTITY (ImageNet), STATE/ACTION=QUANTILES.

## 3. Dataset e split

| Conjunto | Episódios | Qtde | Frames |
|---|---|---|---|
| Treino | 0–213 | 214 | 47.668 |
| Validação | 214–237 | 24 | ~5.284 |

⚠️ Split **sequencial** (por ordem de gravação, não aleatório) → risco de *domain shift* treino↔val. Medido: drift **leve, não-significativo** na mão; efeito dominante = **overfitting**.

## 4. Codificação da ação — a única diferença entre as duas políticas

| | right14 (14 dims) | right8 (8 dims) |
|---|---|---|
| action | 7 braço + **7 dedos** | 7 braço + **1 squeeze** |
| dedos | `squeeze × RIGHT_TARGET` (já expandido) | escalar `squeeze` ∈ [0,1] |
| deploy da mão | direto (7 juntas) | `hand_q = squeeze × RIGHT_TARGET` |
| dataset | `lewislf/G1_Dex3_right14_dataset` (v3_238ep) | `lewislf/G1_Dex3_pick_white_cup_right8_1squeeze` |

`RIGHT_TARGET = [0, −0.92, −1.74, 1.57, 1.74, 1.57, 1.74]` (pose fechada). Dataset right8 gerado por `lerobot-ext/tools/slice_right_arm_1squeeze.py`.

> 🔑 Em **ambos** a mão é **1 grau de liberdade efetivo** (o squeeze). O right14 emite 7 juntas, mas o rótulo carrega 1 sinal só. Logo nenhum aprende controle independente de dedos — limitação **dos dados**.

## 5. Hiperparâmetros (idênticos nos dois treinos)

| Grupo | Valor |
|---|---|
| Optimizer | AdamW · lr **1e-4** · betas [0.9,0.95] · wd 0.01 · grad_clip 1.0 |
| Scheduler | cosine_decay_with_warmup · peak 1e-4 · warmup **2000** · decay **18000** → 1e-5 |
| Batch / Steps | 32 · **20000** (planejado) · num_workers 8 |
| Seed | 1000 |
| Checkpoints | `best` (dominância nas 2 métricas de val) + `last`; eval_freq 500 |

**Augmentations** (máx. 2/amostra): ColorJitter (b/c [0.8,1.2], sat [0.5,1.5], hue [−0.05,0.05]); RandomResizedCrop (scale [0.75,1.0]); RandomErasing (p1, scale [0.02,0.10]).

## 6. Resultados de treino e overfitting

| | 14-dim · `35jrrbk0` | 8-dim · `8hajpdab` |
|---|---|---|
| status | finished (resume=True, 20k steps) | **crashed** ~step 15900 (resume=False) — curva completa: 31 evals |
| train/loss final | 0.0088 | 0.0074 |
| val_loss | 0.086 (mín ~step29) → **0.317** | 0.098 → **0.363** |
| val_action_mse | mín 0.076 (step 5500) → 0.190 fim | mín **0.046** (~step 8000) → 0.054 |
| `best` salvo | **step 5500** (val_loss 0.124, val_mse 0.076) | `best` (~8000) + `015000` + `last` (sem best_meta) |

- **Overfitting confirmado nos dois:** `train/loss`→~0,008 enquanto a **`val_loss` só sobe**. Até uma run do zero (`1sozoy32`, 14-dim) começa em val ≈0,099 e sobe → o **base já é bom no val (~0,10) e qualquer fine-tune overfitta de imediato**.
- **Dedos são o ponto fraco:** no 14-dim o `val_action_mse` dos dedos (dims 8–13) vai de 0,226→0,405 (quase dobra), enquanto o braço fica ~estável (0,04→0,07).
- **A/B em sim (best 5500 vs last 20000), episódio de VALIDAÇÃO 220:** best MSE total 0,373 vs last 0,420 (best generaliza melhor, mas só ~11%); ambos ~4–5× piores no val que no treino. Confirma: overfitting real, e o `best` melhora **pouco**.

---

# PARTE 2 — Investigação de deploy no ROBÔ REAL

Testamos os checkpoints **`best`** (melhores no val) **no robô físico** (não só em sim). VLA na Atena GPU2, conectando direto no robô (`--robot-ip 10.9.8.73`). Comando idêntico nas duas, mudando só `--checkpoint`:
`--fps 30 --hand-kp 0.8 --rtc --live --denoising-steps 5 --rtc-execution-horizon 25 --rtc-max-guidance 1.0 --home-arm-s 3 --open-hand-s 2 --rehome-idle-s 12`.
Cada run grava log + `policy_view.mp4` (head camera 1ª-pessoa — único ponto de vista do robô).

| # | modelo | checkpoint | wandb | log | artefatos | resultado |
|---|---|---|---|---|---|---|
| 1 | 14-dim `rgb238` | best 5500 | `35jrrbk0` | `infer_right14_20260613_122710.log` | `run_20260613_122710` | ❌ não pegou |
| 2 | 8-dim `1squeeze` | best ~8000 | `8hajpdab` | `infer_right14_20260613_123837.log` | `run_20260613_123837` | ❌ não pegou |

## Investigação 1 — 14-dim best (5500) · 12:27→12:29

**NÃO pegou o copo.** Falha = **mão ciclando**.
- Soft-start ok: braço do standby (Elbow 0.76) → HOME das demos (Elbow −0.11) em 3 s; mão abriu.
- **Mão CICLOU abre/fecha 7×** (eventos do log): FECHOU f69→ABRIU f120 (1,7 s) · f293→f576 (9,5 s) · f618→f634 (0,5 s) · f659→f738 (2,6 s) · f866→f1122 (8,6 s) · f1220→f1574 (11,8 s) · FECHOU f2315. **Early-close já no f69.**
- **Pressão travada no baseline ~104,1–104,3k** o tempo todo → **sem grip real**. hand-gate disparou 26×.
- **Vídeo:** copo na mesa → mão do robô aberta ao lado → **uma pessoa apresenta o copo na mão aberta** e ele não fecha em cima → no fim, copo na mesa e **braço recolhido**.

## Investigação 2 — 8-dim best (~8000) · 12:38→12:41

**NÃO pegou o copo.** Falha = **braço erra o alvo** (diferente do 14-dim).
- Modo **RIGHT8** detectado certo (8 dims → squeeze; dedos = squeeze × RIGHT_TARGET).
- **Mão fechou só 1× (f136) e SEGUROU** (não reabriu) · hand-gate só 1×. → **não ciclou** (mais estável que o 14-dim nesse ponto).
- **MAS o braço não apontou pro copo:** vídeo mostra a mão fechando **no ar, no canto inferior-esquerdo**, longe do copo (centro); início com a mão aberta no canto superior-direito; fim com o braço recolhido. **Nunca chegou no copo.** Pressão ~105k (≈baseline).

## Veredito

**Os dois `best` falharam no robô real, por motivos diferentes:**
- **14-dim:** braço chega na região, mas a **mão cicla** (timing de fechamento sem casar com a visão) e não agarra.
- **8-dim:** mão fecha 1× e segura (sem ciclar), mas o **reach erra a mira** e fecha no lugar errado.

**Trocar para o checkpoint `best` (melhor no val) NÃO consertou o deploy.** O gargalo não é a escolha de checkpoint — é a combinação de **(a) overfitting** (o val só sobe; modelo decora as 214 demos), **(b) falta de grounding visual do grasp** (o modelo não condiciona *quando* e *onde* fechar à imagem do copo — erra timing no 14-dim e mira no 8-dim) e **(c) a mão ser 1-DOF (squeeze)**, que impede controle fino de preensão.

**Alavancas reais** (próximos passos): re-treino com **mais dados e diversidade**, **split aleatório** (não sequencial), **early-stop / menos épocas** para reduzir overfit, e **repensar o encoding da mão** (sair do squeeze 1-DOF). A escolha de checkpoint sozinha não resolve.

---

## Mídia (pasta `midia/`)

Vídeos da head_camera do robô (o que a política viu) + frames-chave de cada run:

- **Run 1 (14-dim best 5500):** `run1_14dim_best5500_robo.mp4` · frames: `run1_14dim_inicio.jpg`, `run1_14dim_humano_apresenta_copo.jpg` (pessoa oferece o copo na mão aberta), `run1_14dim_fim_braco_recolhido.jpg`.
- **Run 2 (8-dim best):** `run2_8dim_best_robo.mp4` · frames: `run2_8dim_inicio.jpg`, `run2_8dim_mao_fecha_fora_do_alvo.jpg` (mão fechada no canto, longe do copo), `run2_8dim_fim.jpg`.

*Origem: logs e `policy_view.mp4` em `~/Prometheus/Luiz/prometheus-vla/train/log/run_<ts>/` (Atena); curvas no wandb `prometheus-lcad/prometheus_g1` (runs `35jrrbk0`, `8hajpdab`).*
