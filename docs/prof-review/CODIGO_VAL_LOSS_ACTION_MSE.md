2026-06-14 12:20

> **O que é (breve):** código verbatim de `val_loss`, `val_action_mse`, `predict_action_chunk`, loop de validação e config de treino do π0.5 — com `file:linha` e síntese diagnóstica. Resposta ao checklist do professor.
> **Correções:** documento novo (sem versão anterior; nada sobrescrito).

---

# Código de `val_loss`, `val_action_mse`, `predict_action_chunk`, loop de val e config de treino

> Resposta ao checklist do professor. Run de referência: **`6kr7d8nz`** (`cup_pi05_right8_armstate7_lf`),
> que **crashou no step ~12.4k/20k** (queda de rede/energia no lab). Os valores de config vêm do
> **wandb** (a `.yaml` em si está no disco da Atena, offline agora).
> Modelo: **π0.5** (PaliGemma ~3B congelado + expert Gemma 300M treinável), **flow-matching**, chunk=50.
>
> **TL;DR diagnóstico:** a **oscilação** da `val_loss` é **variância de medição** (ruído e `t` são
> re-sorteados a cada eval, sem seed) — não é o modelo mudando. A **subida** é overfit do campo de
> velocidade, que **descola** da `val_action_mse` (o código até comenta isso). **Não há EMA.** O **best
> usa só `val_action_mse`**. A matemática treino↔inferência é **consistente** (as duas métricas medem
> coisas diferentes por design).

---

## 1. `val_loss` — a perda de flow-matching

`modeling_pi05.py` — sorteio de `t` (Beta) e o cálculo do alvo de velocidade:

```python
# modeling_pi05.py:94-98
def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))

# modeling_pi05.py:627-632
def sample_time(self, bsize, device):
    time_beta = sample_beta(self.config.time_sampling_beta_alpha,
                            self.config.time_sampling_beta_beta, bsize, device)
    time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
    return time

# modeling_pi05.py:724-777  (forward de TREINO/LOSS)
def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None) -> Tensor:
    if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)   # N(0,1), SEM seed
    if time is None:
        time = self.sample_time(actions.shape[0], actions.device)  # Beta, SEM seed
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions     # interpolação linear
    u_t = noise - actions                                            # VELOCITY target
    ...
    v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
    return F.mse_loss(u_t, v_t, reduction="none")                    # [B, chunk, dim]
```

Config do sorteio de `t` (`configuration_pi05.py:44-51`): `Beta(α=1.5, β=1.0)`, `scale=0.999`, `offset=0.001`,
`num_inference_steps=10`.

**Respostas:**
- **Como `t` é sorteado:** `Beta(1.5, 1.0)` escalada (`t = beta*0.999 + 0.001`). **Não é uniforme nem logit-normal.** Média ≈ 0.6, enviesada pro lado do **ruído** (`t→1`).
- **Convenção:** `x_t = t·ε + (1−t)·a` (de `a` em t=0 → `ε` em t=1); alvo `u_t = ε − a` (velocity, não `(x−a)/t`).
- **Por-dim:** `F.mse_loss(..., reduction="none")` mantém `[B, chunk, dim]`; a policy faz `loss_per_dim = mean(losses, dim=[0,1])` → vira `dim_eval/val_loss_dim_XX`.

---

## 2. `val_action_mse` — erro da AÇÃO gerada

`run_train.py` (o bloco que adicionamos):

```python
# run_train.py:668-680
with accelerator.autocast():
    pred_actions = policy_for_predict.predict_action_chunk(val_batch)
gt_actions = val_batch[ACTION].to(pred_actions.device)
if gt_actions.dim() == pred_actions.dim():
    dim = min(pred_actions.shape[-1], gt_actions.shape[-1])
    sq = (pred_actions[..., :dim] - gt_actions[..., :dim]) ** 2     # erro²
    val_action_mse_meter.update(sq.mean().item())
    mse_dim = sq.mean(dim=tuple(range(sq.dim() - 1))).float().cpu() # média sobre batch+chunk → [dim]
    ...

# run_train.py:741-745  (split arm/grasp)
if len(_md) >= 8:
    val_log_dict["val_action_mse_arm"]   = float(sum(_md[:7]) / 7)          # dims 0-6  (braço)
    _grasp = _md[7:]
    val_log_dict["val_action_mse_grasp"] = float(sum(_grasp) / len(_grasp)) # dim 7+   (squeeze)
```

**Respostas:**
- **Contra qual GT:** o **chunk inteiro** (todos os 50 passos) previsto vs o chunk GT inteiro — `sq.mean()` média sobre batch **e** todos os timesteps. Não é só a 1ª ação.
- **Espaço:** **normalizado** (QUANTILES q01/q99) — o `predict_action_chunk` aqui sai sem desnormalizar. Por isso `arm≈0.039` e `grasp≈0.238` são em unidades normalizadas.
- **Split:** braço = média das dims 0-6; grasp = média das dims 7+ (aqui só a dim 7, o squeeze).
- **⚠️ Subconjunto:** só os **4 primeiros batches** entram (`max_action_mse_batches=4` → ~128 amostras), enquanto a `val_loss` roda o val inteiro. Isso deixa a `val_action_mse` **mais ruidosa** do que parece.

---

## 3. `predict_action_chunk` / `sample_actions` — a ponte flow→ação

```python
# modeling_pi05.py:618-625
def sample_noise(self, shape, device):
    return torch.normal(mean=0.0, std=1.0, size=shape, ...)   # SEM seed

# modeling_pi05.py:779-856  (sample_actions)
@torch.no_grad()
def sample_actions(self, images, img_masks, tokens, masks, noise=None, num_steps=None, **kwargs):
    if num_steps is None:
        num_steps = self.config.num_inference_steps            # = 10
    if noise is None:
        noise = self.sample_noise((bsize, chunk_size, max_action_dim), device)
    ... # forward do prefixo (imagem+texto) com KV-cache
    dt = -1.0 / num_steps                                       # passo fixo
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt                                  # t: 1.0 → 0.0
        v_t = denoise_step(...)                                 # (ou rtc_processor se --rtc)
        x_t = x_t + dt * v_t                                    # EULER, dt fixo
    return x_t
```

Normalização: `normalize_processor.py:362-377` (QUANTILES, `2(x−q01)/(q99−q01) − 1`); desnorm no
`processor_pi05.py:153-158` (`UnnormalizerProcessorStep`, inverse).

**Respostas:**
- **Ruído inicial:** `torch.normal(0,1)`, **sem seed** → muda a cada chamada.
- **Integration steps:** `num_inference_steps = 10`.
- **Scheduler/timesteps:** **linear determinístico**, `t` de **1.0 → 0.0** em 10 passos, `dt=−1/10`. Integrador **Euler** (`x_t += dt·v_t`). Sem Karras/softmax.
- **Determinístico vs estocástico:** **determinístico** — a única fonte de aleatoriedade é o **ruído inicial**.
- **Norm/desnorm:** **QUANTILES** (q01/q99) → ação em `[-1,1]` no input; desnorm na saída (no deploy). Na `val_action_mse` fica em espaço normalizado.

---

## 4. Loop de validação — **a chave da oscilação**

```python
# run_train.py:510-529  (val dataloader: shuffle=False, batch=cfg.batch_size, val_dataset.episodes)
# run_train.py:662-719
with torch.no_grad():
    for val_batch in val_dataloader:
        val_batch = preprocessor(val_batch)
        with accelerator.autocast():
            val_loss, val_output_dict = policy.forward(val_batch)   # ← ruído+t re-sorteados aqui
        if action_mse_batches_done < max_action_mse_batches:        # só 4 batches
            pred_actions = policy_for_predict.predict_action_chunk(val_batch)
            ...
policy.train()
```

**Respostas (as que mais importam):**
- **🎯 A `val_loss` usa ruído e `t` ALEATÓRIOS a cada validação** (sem seed): o `forward()` recebe `noise=None, time=None` e re-sorteia `N(0,1)` + `Beta(1.5,1.0)` toda vez. **Com o modelo congelado, a `val_loss` já oscilaria sozinha** → a oscilação é **variância estatística de medição**, não o modelo piorando.
- **⚠️ A `val_action_mse` também tem ruído inicial não-seedado** + só 4 batches → também oscila um pouco (menos, por ser o endpoint integrado).
- **⚠️ Achado extra:** o `policy.eval()` está **comentado** (mantém `policy.train()` durante a val — herança de "style prediction" do ACT). Se houver qualquer dropout ativo no expert, isso **adiciona variância** na val. Vale checar se o expert do π0.5 tem dropout > 0 (se tiver, rodar em `eval()` reduziria a oscilação).
- **Val set:** os episódios 214–237 (**24 eps**); batch = `cfg.batch_size` (**32**); `action_mse` só nos 4 primeiros batches.

---

## 5. Config de treino / optimizer / scheduler / best

Valores reais (do **wandb config** desta run):
- **lr = 1e-4**, **AdamW** (betas `[0.9, 0.95]`, **weight_decay = 0.01**, eps 1e-8), **grad_clip_norm = 1.0**.
- **batch_size = 32**, **steps = 20000**. **lr observada caindo 1e-4 → 3e-5** → **cosine decay** (scheduler dá `.step()` por batch).
- Augmentation no treino: ColorJitter + RandomErasing + RandomResizedCrop (forte); val sem aug.

```python
# run_train.py:433     optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
# run_train.py:218-223 grad clip: accelerator.clip_grad_norm_(params, grad_clip_norm) se > 0
# run_train.py:231-233 lr_scheduler.step() a CADA batch (warmup+cosine típico)

# run_train.py:759-783  (CRITÉRIO DO BEST — comentário do próprio código)
# Critério do BEST (revisado 2026-06-10): SÓ val_action_mse ... O critério anterior
# (dominância em val_loss E val_action_mse) congelou o best: as duas métricas andaram
# em direções OPOSTAS (flow-loss subiu 0.098→0.18 enquanto o mse caiu 0.093→0.050).
cur_mse = val_action_mse_meter.avg if val_action_mse_meter.count > 0 else None
if best_checkpoint_step is None:        is_new_best = True
elif cur_mse is not None and best_val_mse is not None:  is_new_best = cur_mse <= best_val_mse
else:                                   is_new_best = cur_loss <= best_val_loss   # fallback
```

**Respostas:**
- **lr/scheduler:** 1e-4, **cosine** (decai por batch; observado 1e-4→3e-5). Warmup: provável (típico lerobot), mas o valor exato está na `.yaml`/factory (Atena offline).
- **weight_decay:** 0.01. **grad_clip:** 1.0. **batch:** 32. **steps:** 20000.
- **EMA:** **NÃO existe** (grep `ema` vazio em `run_train.py`).
- **Best:** **só `val_action_mse`** (`cur_mse <= best_val_mse`); fallback `val_loss` só se action_mse indisponível. O próprio código **documenta a descolagem** entre as duas métricas.

---

## 6. Síntese / diagnóstico

| Sintoma | Causa (medida no código) | Conserto |
|---|---|---|
| `val_loss` **oscila** | ruído + `t` re-sorteados **sem seed** a cada eval (`forward(noise=None,time=None)`); + possivelmente `policy.train()` na val | **fixar/seedar** ruído+`t` no eval, **mediar** sobre N (t,ε) ou uma grade de `t`; rodar em `eval()` |
| `val_loss` **sobe** | overfit do campo de velocidade — **descola** da `val_action_mse` (o código comenta: flow-loss 0.098→0.18 vs mse 0.093→0.050) | não é alarme; se quiser: **EMA**, lr menor, mais dados |
| `val_action_mse` ruidosa | ruído inicial sem seed + **só 4 batches** | seedar o ruído + aumentar `max_action_mse_batches` |
| sem suavização | **não há EMA** | adicionar EMA dos pesos (suaviza curva **e** costuma melhorar o deploy) |

**Consistência treino↔inferência:** a matemática bate — treino mede o erro do campo `v` num ponto
aleatório (`MSE(u_t, v_t)`); inferência integra 10 passos de Euler do mesmo campo. **Não há mismatch**;
`val_loss` (estilo-treino) e `val_action_mse` (estilo-inferência) medem coisas diferentes **por design**.
O sinal que importa pro robô é a `val_action_mse` (e o split **arm 0.039** vs **grasp 0.238** → a pega é o
gargalo, que é problema de **grounding**, não de overfit de `val_loss`).

> Fonte do código: `lerobot/src/lerobot/policies/pi05/modeling_pi05.py`, `.../configuration_pi05.py`,
> `.../processor/normalize_processor.py`, `.../processor/pi05/processor_pi05.py`, `lerobot-ext/train/run_train.py`.
> Config: wandb `prometheus-lcad/prometheus_g1/6kr7d8nz`.
