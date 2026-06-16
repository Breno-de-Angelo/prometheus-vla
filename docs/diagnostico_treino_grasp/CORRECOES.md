# Correções — hipóteses que foram REFUTADAS/erradas ao longo da investigação

> Esta página existe pra você NÃO carregar conclusões erradas pra frente. Atacamos várias frentes;
> algumas hipóteses caíram quando confrontadas com código/dados. Aqui está o que está **morto** e o
> que ficou no lugar. Marcação: ❌ = estava errado · ✅ = correção vigente.

---

## A) Hipóteses NOSSAS que caíram

### ❌ "O squeeze regride pra média (~0,41) — o modelo hesita no meio"
- **De onde veio:** o `val_action_mse` do squeeze (~0,20–0,24) batia quase exato com a variância crua
  do squeeze [0,1] (0,2376 = MSE de prever a média). Conclusão na hora: o modelo entrega ~0,4 e não
  decide.
- **Por que estava ERRADO (erro de unidade):** o `val_action_mse` é calculado no espaço **NORMALIZADO**
  (QUANTILES; `predict_action_chunk` não desnormaliza). O squeeze tem q01=0/q99=1 → {0,1}→{−1,+1}, e
  nesse espaço o piso de prever a média é **0,95**, não 0,236. O observado 0,20–0,24 está **~4× ABAIXO**
  do piso → o modelo aprende o squeeze, **não** hesita. O "casamento" era coincidência de espaço (cru).
- **✅ Correção (vigente, confirmada pelo probe):** o squeeze sai **CRAVADO** no treino (open→0,005,
  closed→0,847, separação +0,84). O resíduo (~5–6% de frames flipados) é **erro de TIMING**, não
  hesitação. → ver `8DIM.md`.

### ❌ "A val só sobe = o modelo está quebrado / o braço esquece"
- **✅ Correção:** a `val_loss` que sobe é a **proxy de flow-matching** (loss frouxa). A métrica que
  importa (`val_action_mse`, ação gerada) mostra o **braço aprendendo** (cai 0,077→0,025). As duas
  famílias DESCOLAM. → ver `14DIM.md`, `8DIM.md` e `docs/treino_e_dataset/METRICAS_COMPLETAS.md`.

### ❌ "Não temos dado de depth / não dá pra usar profundidade" (dito cedo, 2×)
- **✅ Correção:** o dataset TEM `head_camera_depth` (PNG uint16, mm, 480×848, alinhado ao RGB) e
  tátil [108]. Usado no wiring do dashboard e disponível pra grounding.

### ❌ "O gargalo é a escolha de checkpoint (best vs last)"
- **✅ Correção:** trocar pro `best` NÃO consertou o robô (ambos falharam). O `best` (5500/8000) é o
  ponto útil, mas o gargalo é outro (grounding visual do grasp). → ver `INDEX`.

---

## B) Teses do PROFESSOR que refutamos no código (review em `docs/prof-review/`)

### ❌ "Drift representacional do VLM → usar VRA"
- **✅ Refutado:** `train_expert_only=true` congela TODA a PaliGemma (vision tower incluso,
  `modeling_pi05.py:467-470`). Encoder não recebe gradiente → **não pode driftar** → VRA é moot.

### ❌ "Inferência step-wise a ~3 fps, sem action chunking → implementar ACT"
- **✅ Refutado:** já usamos action chunking nativo (chunk 50) + RTC com inpainting. Os 3 fps são a
  taxa de replanejamento, não de controle (30 fps + mão 100 Hz).

### ❌ "Cuidado com split aleatório por frame (leakage)"
- **✅ Não se aplica:** já usamos split por episódio (trajectory-level), eps 0–213/214–237.

---

## C) O que CONTINUA de pé (não foi refutado)
- Overfitting é real (train→~0,008, val_loss sobe ~3,7×).
- A mão é **1-DOF efetivo** (squeeze; dedos = squeeze×RIGHT_TARGET).
- O `best` está cedo (5500 / 8000); o `last` é overfit.
- A pega NÃO está aterrada na visão — **achado central atual**, ver `INDEX`. Preciso: ✅ **MEDIDO** que a
  imagem não gatilha o grasp (ablação); ⏳ que o gatilho seja a pose/propriocepção é **inferência** (falta `drop-proprio`).
