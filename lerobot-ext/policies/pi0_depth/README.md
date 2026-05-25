# Tutorial: Como Funciona a Validação no PI05-Depth

> Documento baseado no código `run_train.py` + `modeling_pi05.py` do projeto Prometheus G1.

---

## 1. Visão Geral

Durante o treinamento, a cada `eval_freq` steps (configurado como `100` no seu YAML), o loop principal pausa e roda a **validação**. O objetivo é medir como o modelo está generalizando para episódios que ele **nunca viu durante o treino**.

```
Treino (steps 1..5000)
    │
    ├── a cada 100 steps → Validação
    │       ├── Passa o val_dataset pelo modelo (sem gradiente)
    │       ├── Calcula val_loss por batch
    │       ├── Loga no WandB
    │       └── Se val_loss < melhor até agora → salva best_val_checkpoint
    │
    └── a cada 1000 steps → Checkpoint periódico (se save_checkpoint: true)
```

---

## 2. Datasets: Treino vs Validação

No seu YAML você separou os episódios manualmente:

```yaml
dataset:          # TREINO — o modelo vê e aprende
  episodes: [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21]  # 17 episódios

val_dataset:      # VALIDAÇÃO — o modelo nunca vê durante o treino
  episodes: [7, 9, 18, 19]                                                   # 4 episódios
```

Os 4 episódios de validação são a sua "prova real" — se o modelo acerta neles, provavelmente vai funcionar no robô físico também.

> **Por que separar?** Se você validasse nos mesmos episódios de treino, o modelo poderia estar apenas memorizando as trajetórias e pareceria ótimo sem realmente aprender.

---

## 3. O Loop de Validação no Código

```python
# run_train.py — simplificado para entendimento
if is_eval_step and val_dataloader is not None:

    policy.eval()   # Desliga dropout, batch norm em modo treino, etc.

    with torch.no_grad():   # Não calcula gradientes — só forward pass
        for val_batch in val_dataloader:

            val_batch = preprocessor(val_batch)   # Normaliza, tokeniza, etc.

            with accelerator.autocast():          # Precisão mista (bfloat16)
                val_loss, val_output_dict = policy.forward(val_batch)

            # Agrega o loss de todos os batches
            val_loss_meter.update(val_loss.item())

    policy.train()  # Volta para modo treino
```

Pontos importantes:
- `torch.no_grad()` — nenhum gradiente é calculado, então é mais rápido e não consome memória extra
- `policy.eval()` → `policy.train()` — garante que o modelo se comporta igual na inferência real
- O `preprocessor` é o **mesmo** do treino — as mesmas normalizações são aplicadas

---

## 4. O que é o `val_loss`

O `val_loss` é o **MSE (Mean Squared Error) do Flow Matching** calculado nos episódios de validação.

### Como o Flow Matching calcula o loss:

```
1. Pega a ação real do dataset:  actions  [B, chunk_size, action_dim]
2. Gera ruído aleatório:         noise    [B, chunk_size, action_dim]
3. Mistura com um timestep t:    x_t = t * noise + (1-t) * actions
4. O modelo tenta prever:        v_t = noise - actions  (o "campo vetorial")
5. Loss = MSE(v_t_predito, v_t_real)
```

Em outras palavras: o modelo aprende a "desfazer o ruído" — e o loss mede o quão bem ele consegue fazer isso nas trajetórias de validação.

### Interpretação dos valores:

| val_loss | Interpretação |
|---|---|
| > 2.0 | Início do treino — normal, modelo ainda aprendendo |
| 1.0 – 2.0 | Convergindo — sinal positivo se estiver caindo |
| 0.5 – 1.0 | Bom aprendizado — verifique se não é overfitting |
| < 0.5 | Excelente — ou overfitting se train_loss também baixo |
| Sobe depois de cair | **Overfitting** — use o `best_val_checkpoint` |

---

## 5. `loss_per_dim` — O Diagnóstico por Junta

Além do loss médio, o código loga o loss **separado por dimensão da ação**. Isso é ouro para entender o que o robô está errando.

```python
# modeling_pi05.py
loss_per_dim = losses.mean(dim=[0, 1]).detach().cpu()   # média sobre batch e chunk
loss_dict = {
    f"loss_per_dim/{i}": v.item() for i, v in enumerate(loss_per_dim)
}
```

### Mapa das dimensões para o Unitree G1 (28 DOF):

```
dims  0– 5  →  Braço: ombro (rot/flex/abd), cotovelo, punho (flex/rot)
dims  6–13  →  Tronco, quadril, joelho, tornozelo
dims 14–21  →  Dedos da mão (8 juntas)
dims 22–27  →  Garras / sensores de pressão integrados
```

### Lendo o seu log do step 100:

```
dim/2:  13.0  ← cotovelo/ombro — CRÍTICO, maior erro
dim/5:  12.1  ← punho rotação  — CRÍTICO
dim/0:   9.7  ← ombro rotação  — alto
dim/4:   8.7  ← punho flexão   — alto
─────────────────────────────────────────────
dim/14–21: 0.4–0.6  ← dedos — quase resolvido!
dim/22–27: 1.0–1.4  ← garras — aprendendo
```

**O que isso significa:** o modelo já sabe abrir/fechar a mão para pegar a caneca (dims 14–21 baixos), mas ainda está errando muito a trajetória do braço até chegar lá (dims 0–5 altos). Isso é esperado no step 100 — o braço tem trajetórias mais longas e com mais variação entre demos.

---

## 6. Best-Val Checkpoint

O `BestValTracker` monitora o `val_loss` e salva automaticamente quando bate um recorde:

```python
# run_train.py
class BestValTracker:
    def update(self, val_loss, step, ...):
        if val_loss >= self.best_val_loss:
            return False   # Não melhorou, não salva

        self.best_val_loss = val_loss
        checkpoint_dir = self.output_dir / "best_val_checkpoint"
        save_checkpoint(checkpoint_dir, ...)   # Salva!
        return True
```

```
train_output/pick_up_the_cup_pi05_depth/
├── best_val_checkpoint/          ← melhor modelo até agora
│   ├── pretrained/
│   │   └── model.safetensors
│   ├── best_val_meta.txt         ← val_loss e step do recorde
│   └── ...
└── checkpoint_005000/            ← checkpoint periódico (se save_checkpoint: true)
```

O `best_val_meta.txt` contém:
```
best_val_loss=0.847231
best_step=3400
total_steps=5000
```

> **Dica:** Sempre use o `best_val_checkpoint` para deploy no robô, nunca o último checkpoint. O último pode ser overfitting — o melhor é o que generalizou melhor nos 4 episódios de validação.

---

## 7. O que Acompanhar no WandB

Com `wandb.enable: true` e `project: prometheus_g1`, todas as métricas são logadas automaticamente.

### Curvas para monitorar:

**`val_loss` vs `loss` (train):**
```
loss (treino):    2.1 → 1.5 → 1.0 → 0.7 → 0.5   ← deve cair sempre
val_loss:         2.4 → 1.8 → 1.2 → 0.9 → 0.8   ← deve cair e estabilizar
```
Se `val_loss` começa a **subir** enquanto `loss` ainda cai → overfitting. Pare e use o `best_val_checkpoint`.

**`loss_per_dim/2` e `loss_per_dim/5`** (as piores):
```
step 100:   13.0 / 12.1   ← onde você está agora
step 500:   deve ser < 6.0 (sinal de convergência)
step 2000:  deve ser < 3.0 (bom)
step 5000:  deve ser < 1.5 (excelente para 17 demos)
```
Se no step 2000 ainda estiverem acima de 5.0, o dataset precisa de mais demos para as trajetórias do braço.

**`best_val_loss`:**
Linha que só desce — cada vez que aparece no log significa que o modelo melhorou de verdade.

---

## 8. Configurações que Afetam a Validação

```yaml
eval_freq: 100        # Valida a cada 100 steps
                      # Mais frequente = mais informação, mas ~45s por validação

val_dataset:
  episodes: [7, 9, 18, 19]   # 4 episódios de validação
                              # Mínimo recomendado: 3–5 episódios
                              # Idealmente: 20% do total de demos

save_best_checkpoint: true    # Salva automaticamente quando val_loss melhora
```

### Custo de tempo por validação:

No seu setup atual (4 episódios de val, batch_size=4, GPU):
```
~45 segundos por validação × 50 validações (a cada 100 em 5000 steps)
= ~37 minutos gastos só em validação durante o treino total
```

Se quiser acelerar, mude `eval_freq: 200` ou `eval_freq: 500` — perde granularidade mas ganha tempo de treino útil.

---

## 9. Fluxo Completo Resumido

```
Step 1..100  →  treina nos 17 episódios
Step 100     →  VALIDAÇÃO
                ├── passa os 4 episódios de val pelo modelo (sem gradiente)
                ├── calcula val_loss = 2.482 (step 100, seu log)
                ├── loga loss_per_dim/0..27 no WandB
                └── salva best_val_checkpoint (primeiro recorde = sempre salva)

Step 101..200 → treina
Step 200      → VALIDAÇÃO
                └── se val_loss < 2.482 → novo recorde → salva checkpoint

... repete até step 5000 ...

Resultado final:
  best_val_checkpoint/  ← use este para o robô
  best_val_meta.txt     ← mostra em qual step o modelo foi melhor
```

---

## 10. Sinais de Alerta Durante o Treino

| Sinal no log | O que significa | O que fazer |
|---|---|---|
| `val_loss` nunca cai abaixo de 2.0 | Modelo não está aprendendo | Verificar LR, aumentar dados |
| `val_loss` sobe após step 2000 | Overfitting | Usar `best_val_checkpoint`, parar cedo |
| `loss_per_dim/2` > 5.0 no step 2000 | Dataset insuficiente para cotovelo | Gravar mais demos |
| `🏆 NOVO RECORDE` para de aparecer | Platô — modelo convergiu | Normal após step 3000–4000 |
| `grad_norm` > 100 repetidamente | Gradiente explodindo | Reduzir `optimizer_lr` |