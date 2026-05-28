# Tutorial: Como Funciona a Validação no PI05-Depth

> Documento baseado no código `run_train.py` + `modeling_pi05.py` do projeto Prometheus G1.
> Mapa de juntas extraído de `info.json` do dataset `pick_up_the_cup_2026-04-30`.

---

## 1. Visão Geral

Durante o treinamento, a cada `eval_freq` steps (configurado como `100` no YAML), o loop
principal pausa e roda a **validação**. O objetivo é medir como o modelo está generalizando
para episódios que ele **nunca viu durante o treino**.

```
Treino (steps 1..5000)
    │
    ├── a cada 100 steps → Validação
    │       ├── Passa o val_dataset pelo modelo (sem gradiente)
    │       ├── Calcula val_loss por batch
    │       ├── Loga loss_per_dim/0..27 no WandB
    │       └── Se val_loss < melhor até agora → salva best_val_checkpoint
    │
    └── a cada 1000 steps → Checkpoint periódico (se save_checkpoint: true)
```

---

## 2. Datasets: Treino vs Validação

```yaml
dataset:       # TREINO — o modelo vê e aprende
  episodes: [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21]  # 17 episódios

val_dataset:   # VALIDAÇÃO — o modelo nunca vê durante o treino
  episodes: [7, 9, 18, 19]                                                   # 4 episódios
```

Os 4 episódios de validação são a "prova real" — se o modelo acerta neles, provavelmente
vai funcionar no robô físico também.

> **Por que separar?** Se você validasse nos mesmos episódios de treino, o modelo poderia
> estar apenas memorizando as trajetórias e pareceria ótimo sem realmente aprender.

---

## 3. O Loop de Validação no Código

```python
# run_train.py — simplificado para entendimento
if is_eval_step and val_dataloader is not None:

    policy.eval()         # Desliga dropout, coloca modelo em modo inferência

    with torch.no_grad(): # Não calcula gradientes — só forward pass
        for val_batch in val_dataloader:

            val_batch = preprocessor(val_batch)   # Normaliza, tokeniza, etc.

            with accelerator.autocast():           # Precisão mista (bfloat16)
                val_loss, val_output_dict = policy.forward(val_batch)

            val_loss_meter.update(val_loss.item()) # Agrega o loss

    policy.train()        # Volta para modo treino
```

Pontos importantes:
- `torch.no_grad()` — nenhum gradiente é calculado, mais rápido e sem custo de memória extra
- `policy.eval()` → `policy.train()` — garante comportamento igual à inferência real no robô
- O `preprocessor` é o **mesmo** do treino — as mesmas normalizações são aplicadas

---

## 4. O que é o `val_loss`

O `val_loss` é o **MSE (Mean Squared Error) do Flow Matching** calculado nos episódios de
validação. O Flow Matching aprende um "campo vetorial" que transforma ruído em ação.

### Como o loss é calculado:

```
1. Pega a ação real do dataset:  actions  [B, chunk_size=50, action_dim=28]
2. Gera ruído aleatório:         noise    [B, 50, 28]
3. Mistura com timestep t:       x_t = t * noise + (1-t) * actions
4. Modelo prediz o campo:        v_t_pred = modelo(x_t, t, imagem, estado)
5. Campo real é:                 v_t_real = noise - actions
6. Loss = MSE(v_t_pred, v_t_real)   ← média sobre batch, chunk e dimensões
```

### Interpretação dos valores — baseada no seu treino real:

| val_loss | O que significa | Quando aconteceu |
|---|---|---|
| 2.482 | Início — modelo ainda aleatório | step 100 |
| 1.169 | Convergindo rápido | step 200 |
| 0.449 | Bom aprendizado | step 1700 — 1º recorde duradouro |
| 0.383 | Melhor resultado do treino | step 5000 — checkpoint final |
| train_loss = 0.046 | Modelo memorizou os demos | step 5000 |

> A diferença entre `train_loss=0.046` e `val_loss=0.383` (fator ~8×) indica overfitting
> leve nas juntas do braço — o modelo memorizou as trajetórias dos 17 demos mas não
> generalizou completamente para os 4 episódios de validação.

---

## 5. `loss_per_dim` — O Diagnóstico por Junta

O código loga o loss **separado por cada uma das 28 dimensões da ação**. Isso permite
saber exatamente qual junta está errando.

```python
# modeling_pi05.py
loss_per_dim = losses.mean(dim=[0, 1]).detach().cpu()  # média sobre batch e chunk
loss_dict = {
    f"loss_per_dim/{i}": v.item() for i, v in enumerate(loss_per_dim)
}
```

---

## 6. Mapa Completo das 28 Dimensões — Unitree G1 Dex3

Extraído do `info.json` do dataset `pick_up_the_cup_2026-04-30`:

```
┌─────┬──────────────────────────────┬──────────────────────────────────────┐
│ dim │ Nome no dataset              │ Descrição                            │
├─────┼──────────────────────────────┼──────────────────────────────────────┤
│  0  │ kLeftShoulderPitch.q         │ Ombro ESQ — elevação frente/trás     │
│  1  │ kLeftShoulderRoll.q          │ Ombro ESQ — abdução lateral          │
│  2  │ kLeftShoulderYaw.q           │ Ombro ESQ — rotação axial            │
│  3  │ kLeftElbow.q                 │ Cotovelo ESQ — flexão/extensão       │
│  4  │ kLeftWristRoll.q             │ Punho ESQ — rotação (pronação/sup.)  │
│  5  │ kLeftWristPitch.q            │ Punho ESQ — flexão dorsal/palmar     │
│  6  │ kLeftWristyaw.q              │ Punho ESQ — desvio ulnar/radial      │
├─────┼──────────────────────────────┼──────────────────────────────────────┤
│  7  │ kRightShoulderPitch.q        │ Ombro DIR — elevação frente/trás     │
│  8  │ kRightShoulderRoll.q         │ Ombro DIR — abdução lateral          │
│  9  │ kRightShoulderYaw.q          │ Ombro DIR — rotação axial            │
│ 10  │ kRightElbow.q                │ Cotovelo DIR — flexão/extensão       │
│ 11  │ kRightWristRoll.q            │ Punho DIR — rotação (pronação/sup.)  │
│ 12  │ kRightWristPitch.q           │ Punho DIR — flexão dorsal/palmar     │
│ 13  │ kRightWristYaw.q             │ Punho DIR — desvio ulnar/radial      │
├─────┼──────────────────────────────┼──────────────────────────────────────┤
│ 14  │ left_hand_thumb_0_joint.q    │ Polegar ESQ — articulação base       │
│ 15  │ left_hand_thumb_1_joint.q    │ Polegar ESQ — articulação média      │
│ 16  │ left_hand_thumb_2_joint.q    │ Polegar ESQ — articulação ponta      │
│ 17  │ left_hand_middle_0_joint.q   │ Dedo médio ESQ — articulação base    │
│ 18  │ left_hand_middle_1_joint.q   │ Dedo médio ESQ — articulação ponta   │
│ 19  │ left_hand_index_0_joint.q    │ Indicador ESQ — articulação base     │
│ 20  │ left_hand_index_1_joint.q    │ Indicador ESQ — articulação ponta    │
├─────┼──────────────────────────────┼──────────────────────────────────────┤
│ 21  │ right_hand_thumb_0_joint.q   │ Polegar DIR — articulação base       │
│ 22  │ right_hand_thumb_1_joint.q   │ Polegar DIR — articulação média      │
│ 23  │ right_hand_thumb_2_joint.q   │ Polegar DIR — articulação ponta      │
│ 24  │ right_hand_index_0_joint.q   │ Indicador DIR — articulação base     │
│ 25  │ right_hand_index_1_joint.q   │ Indicador DIR — articulação ponta    │
│ 26  │ right_hand_middle_0_joint.q  │ Dedo médio DIR — articulação base    │
│ 27  │ right_hand_middle_1_joint.q  │ Dedo médio DIR — articulação ponta   │
└─────┴──────────────────────────────┴──────────────────────────────────────┘
```

---

## 7. Leitura Real dos Logs — Step 5000 (resultado final)

```
dim/ 0  kLeftShoulderPitch   1.354  ← ombro ESQ elevação — ainda errando
dim/ 1  kLeftShoulderRoll    0.444  ← ombro ESQ lateral — ok
dim/ 2  kLeftShoulderYaw     2.077  ← ombro ESQ rotação — PIOR DO BRAÇO ESQ
dim/ 3  kLeftElbow           0.292  ← cotovelo ESQ — ok
dim/ 4  kLeftWristRoll       1.881  ← punho ESQ rotação — alto
dim/ 5  kLeftWristPitch      1.531  ← punho ESQ flexão — alto
dim/ 6  kLeftWristyaw        0.013  ← punho ESQ lateral — quase perfeito
──────────────────────────────────────────────────────────────────────────
dim/ 7  kRightShoulderPitch  0.409  ← ombro DIR — ok (braço passivo)
dim/ 8  kRightShoulderRoll   0.537  ← ombro DIR — ok
dim/ 9  kRightShoulderYaw    0.160  ← ok
dim/10  kRightElbow          0.215  ← ok
dim/11  kRightWristRoll      0.473  ← ok
dim/12  kRightWristPitch     0.258  ← ok
dim/13  kRightWristYaw       0.304  ← ok
──────────────────────────────────────────────────────────────────────────
dim/14  left_hand_thumb_0    0.016  ✅ polegar ESQ — quase perfeito
dim/15  left_hand_thumb_1    0.013  ✅
dim/16  left_hand_thumb_2    0.012  ✅
dim/17  left_hand_middle_0   0.012  ✅ dedo médio ESQ
dim/18  left_hand_middle_1   0.014  ✅
dim/19  left_hand_index_0    0.013  ✅ indicador ESQ
dim/20  left_hand_index_1    0.013  ✅
──────────────────────────────────────────────────────────────────────────
dim/21  right_hand_thumb_0   0.011  ✅ polegar DIR — quase perfeito
dim/22  right_hand_thumb_1   0.112  ✅
dim/23  right_hand_thumb_2   0.107  ✅
dim/24  right_hand_index_0   0.113  ✅ indicador DIR
dim/25  right_hand_index_1   0.112  ✅
dim/26  right_hand_middle_0  0.110  ✅ dedo médio DIR
dim/27  right_hand_middle_1  0.113  ✅
```

### Conclusão do padrão:

O modelo aprendeu perfeitamente **como fechar a mão** para pegar a caneca (dims 14–27
todos abaixo de 0.12). O erro restante está concentrado nas **juntas de rotação do ombro
e punho esquerdos** (dims 0, 2, 4, 5) — as juntas que controlam *como o braço chega* até
a caneca, não *como a mão a segura*.

Isso faz sentido física e geometricamente: a rotação axial do ombro (`kLeftShoulderYaw`)
é a junta mais sensível a pequenas variações na posição inicial do robô e na posição da
caneca. Com 17 demos, pequenas inconsistências entre episódios se traduzem em erro alto
nessas dimensões.

---

## 8. Evolução do Treino — Resumo Cronológico

```
step  100 │ val=2.482 │ dim/2=13.0  dim/5=12.1 │ Início, tudo alto
step  200 │ val=1.169 │ dim/2= 4.9  dim/5= 3.8 │ Queda rápida (-53%)
step 1700 │ val=0.449 │ dim/2= 2.5  dim/5= 1.9 │ 🏆 Primeiro recorde duradouro
step 2000 │ val=0.497 │ dim/2= 2.7  dim/5= 2.3 │ Plateau — braço travou
step 4800 │ val=0.386 │ —                       │ 🏆 Novo recorde (decay do LR)
step 5000 │ val=0.383 │ dim/2= 2.1  dim/5= 1.5 │ 🏆 Final — melhor checkpoint
          │           │                         │
train_loss│    0.046  │ ← fator 8× abaixo do val│ Overfitting leve no braço ESQ
```

O plateau entre steps 1700–4800 foi quebrado pelo **cosine decay do learning rate**
(`scheduler_decay_steps=4500`), que forçou o otimizador a fazer ajustes mais finos
nos últimos 1000 steps.

---

## 9. Best-Val Checkpoint

O `BestValTracker` monitora automaticamente e salva quando bate recorde:

```python
# run_train.py
class BestValTracker:
    def update(self, val_loss, step, ...):
        if val_loss >= self.best_val_loss:
            return False   # Não melhorou, não salva
        self.best_val_loss = val_loss
        save_checkpoint(self.output_dir / "best_val_checkpoint", ...)
        return True
```

```
train_output/pick_up_the_cup_pi05_depth/
└── best_val_checkpoint/          ← USE ESTE para deploy no robô
    ├── pretrained/
    │   └── model.safetensors     ← pesos do step 5000 (val_loss=0.383)
    └── best_val_meta.txt         ← best_val_loss=0.383, best_step=5000
```

> **Regra:** sempre use o `best_val_checkpoint` para deploy, nunca o último checkpoint
> periódico. O último pode ter sofrido overfitting — o melhor generalizou mais.

---

## 10. O que Acompanhar no WandB

Com `wandb.enable: true` e `project: prometheus_g1`, tudo é logado automaticamente.

### Curvas principais:

**`val_loss` vs `loss` (treino):**
- Ambos devem cair juntos no início
- Se `val_loss` para de cair enquanto `loss` continua → overfitting → use `best_val_checkpoint`

**Juntas críticas a monitorar no seu caso:**

| Chave WandB | Junta | Meta para funcionar no robô |
|---|---|---|
| `val/loss_per_dim/2` | kLeftShoulderYaw | < 1.0 |
| `val/loss_per_dim/4` | kLeftWristRoll | < 1.0 |
| `val/loss_per_dim/5` | kLeftWristPitch | < 1.0 |
| `val/loss_per_dim/0` | kLeftShoulderPitch | < 0.8 |

Se após mais demos esses valores caírem abaixo de 1.0, o robô provavelmente vai conseguir
posicionar o braço consistentemente acima da caneca antes de fechar a mão.

---

## 11. Configurações que Afetam a Validação

```yaml
eval_freq: 100        # Valida a cada 100 steps (~45s por validação)
                      # 50 validações em 5000 steps = ~37 min gastos em validação

val_dataset:
  episodes: [7, 9, 18, 19]   # 4 episódios — mínimo funcional
                              # Ideal: 20% do total (com 50 demos → 10 de val)

save_best_checkpoint: true    # Salva automaticamente no melhor val_loss
```

---

## 12. Sinais de Alerta

| Sinal no log | Significado | Ação |
|---|---|---|
| `val_loss` nunca cai abaixo de 1.0 | Modelo não está aprendendo | Verificar LR, `train_expert_only` |
| `val_loss` sobe após um recorde | Overfitting — use `best_val_checkpoint` | Parar ou coletar mais dados |
| `dim/2` > 2.0 no step 5000 | Dataset insuficiente para `kLeftShoulderYaw` | Gravar mais demos com braço consistente |
| `🏆` para de aparecer | Plateau — modelo convergiu para o limite dos dados | Normal após step 3000–4000 com 17 demos |
| `grad_norm` > 50 repetidamente | Gradiente explodindo | Reduzir `optimizer_lr` |
| `train_loss` / `val_loss` > 5× | Overfitting forte | Reduzir steps ou coletar mais dados |