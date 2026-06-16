# Código de referência — Run 2 / Run 3 (para o professor)

**Escrito:** 2026-06-14 17:06 (-03)
**Correção:** arquivo novo — nada sobrescrito.
**O que é:** os 5 trechos de código que o professor pediu para ver, colados verbatim do repositório (ele não tem acesso ao repo), com uma anotação curta ligando cada um às perguntas dele. Fonte: `prometheus-vla`, branch `Luiz-pi05d`, `lerobot 0.4.4` + `lerobot-ext`.

---

## Resumo das respostas (mapa rápido)

| # | Pergunta | Resposta curta |
|---|---|---|
| 1 | Run 2 do zero? | Sim, fresh do `pi05_base` (`y32omum0`), já rodando. |
| 2 | Mesma seed? | **`seed=1000`** (default fixo, confirmado no dump do run2). Cobre random/numpy/torch/cuda/accelerate + workers. |
| 3 | fork_rng ou explícito? | **fork_rng** (salva/restaura RNG). `sample_actions` aceita `noise=` explícito; `policy.forward` externo não expõe — por isso fork_rng. |
| 4 | `eval()` quebra algo? | Não. pi05 não tem dropout nem batchnorm → `eval()` é higiene, não muda saída. |
| 5 | Params da EMA | `requires_grad=True` = **693.422.112** (expert; PaliGemma congelado fora). Mesmo set do optimizer. |
| 6 | Checkpoint | Já separa raw / optimizer / scheduler / rng / step. Falta só adicionar `ema_state`. |
| 7 | Best | Hoje `val_action_mse` global; per-dim (arm/grasp) já logado. |
| 8/9/10 | Custo | Eval ~2 min hoje; EMA ≈ grátis; dupla eval ≈ +2 min/eval (~+80 min no run, ~10%). |

---

## 1. `make_optimizer_and_scheduler` (`lerobot/optim/factory.py`)

```python
def make_optimizer_and_scheduler(
    cfg: TrainPipelineConfig, policy: PreTrainedPolicy
) -> tuple[Optimizer, LRScheduler | None]:
    params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
    if cfg.optimizer is None:
        raise ValueError("Optimizer config is required but not provided in TrainPipelineConfig")
    optimizer = cfg.optimizer.build(params)
    lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler is not None else None
    return optimizer, lr_scheduler
```

Valores efetivos (do config): AdamW, `lr=1e-4`, `betas=[0.9,0.95]`, `weight_decay=0.01`, `grad_clip_norm=1`; scheduler cosine com `warmup=2000`, `decay_steps=18000`, `decay_lr=1e-5` (mínimo, não zero). `get_optim_params()` é o preset do pi05 e devolve **só os parâmetros treináveis** (`train_expert_only=true` → PaliGemma com `requires_grad=False` fica fora). **É exatamente esse conjunto que a EMA deve seguir.**

---

## 2. Save / load de checkpoint (`lerobot/utils/train_utils.py`)

Estrutura criada por `save_checkpoint`:

```
NNNNNN/                       # step
├── pretrained_model/
│   ├── config.json
│   ├── model.safetensors     # pesos RAW
│   ├── train_config.json
│   └── (processors)
└── training_state/
    ├── optimizer_param_groups.json
    ├── optimizer_state.safetensors
    ├── scheduler_state.json
    ├── rng_state.safetensors
    └── training_step.json
```

```python
def save_training_state(checkpoint_dir, train_step, optimizer=None, scheduler=None):
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir)
    save_rng_state(save_dir)
    if optimizer is not None:
        save_optimizer_state(optimizer, save_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)

def load_training_state(checkpoint_dir, optimizer, scheduler):
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)
    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)
    return step, optimizer, scheduler
```

**Para a EMA:** acrescentar `ema_state.safetensors` em `training_state/`. **Resume** restaura raw (`model.safetensors`) + optimizer + scheduler + rng + ema_state; **deploy** usa os pesos EMA. Nunca salvar só EMA.

---

## 3. Trecho do val-fix (`lerobot-ext/train/run_train_valfix.py`, ~654-690)

```python
max_action_mse_batches = getattr(cfg, "val_action_mse_batches", 16)  # 4 -> 16
action_mse_batches_done = 0
policy_for_predict = accelerator.unwrap_model(policy)

_EVAL_SEED = 1234
vb_idx = 0
with torch.no_grad():
    for val_batch in val_dataloader:
        val_batch = preprocessor(val_batch)
        vb_idx += 1
        # ruído e timestep do flow-matching FIXOS por batch; fork_rng ISOLA o rng do treino
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(_EVAL_SEED + vb_idx)
            with accelerator.autocast():
                val_loss, val_output_dict = policy.forward(val_batch)

        if action_mse_batches_done < max_action_mse_batches:
            # ruído inicial FIXO no predict_action_chunk
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(_EVAL_SEED + 100000 + vb_idx)
                with accelerator.autocast():
                    pred_actions = policy_for_predict.predict_action_chunk(val_batch)
            gt_actions = val_batch[ACTION].to(pred_actions.device)
            # ... mse global + por dimensão (arm/grasp) ...
```

`torch.random.fork_rng(devices=[0])` salva o estado do RNG (CPU+CUDA dev 0) ao entrar e **restaura ao sair** → a validação não consome o RNG do treino. Também há `policy.eval()` no início do loop de validação (restaurado para `train()` no fim). Sobre a preferência por passar `noise/time` explícitos: ver item 4 — dá para o `predict_action_chunk`, mas o `policy.forward` externo não expõe esses argumentos.

---

## 4. `predict_action_chunk` e `sample_actions` (`lerobot/policies/pi05/modeling_pi05.py`)

```python
# nível do MODELO — aceita noise explícito
def sample_actions(self, images, img_masks, tokens, masks, noise=None, num_steps=None, **kwargs):
    if num_steps is None:
        num_steps = self.config.num_inference_steps
    if noise is None:
        noise = self.sample_noise(actions_shape, device)
    ...

# nível da POLICY — repassa **kwargs para sample_actions e já chama self.eval()
def predict_action_chunk(self, batch, **kwargs):
    self.eval()
    images, img_masks = self._preprocess_images(batch)
    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)
    return actions[:, :, :original_action_dim]

# nível da POLICY — usado no val_loss; NÃO expõe noise/time
def forward(self, batch, reduction="mean") -> tuple[Tensor, dict]:
    ...
```

Conclusão: para `val_action_mse` dá para passar `noise=` explícito via `predict_action_chunk(batch, noise=...)`. Para `val_loss`, o `forward` externo só recebe `(batch, reduction)` — passar `noise/time` exigiria mudar a assinatura, então usamos `fork_rng`. Efeito é o mesmo (sem contaminação do RNG do treino). Nota: `predict_action_chunk` já chama `self.eval()` internamente.

---

## 5. Config do Run 2/Run 3 (`train_cup_pi05_right8_armstate7_valfix.yaml`)

```yaml
# policy (pi05)
chunk_size: 50
n_action_steps: 50
train_expert_only: true        # congela o PaliGemma
optimizer_lr: 1.0e-4
scheduler_warmup_steps: 2000
scheduler_decay_steps: 18000
scheduler_decay_lr: 1.0e-5
num_inference_steps: 10        # passos de Euler do flow-matching no predict

# treino
seed: 1000                     # default fixo (confirmado no dump do run2)
steps: 20000
batch_size: 32
num_workers: 8
eval_freq: 500                 # uma validação leva ~2 min
save_freq: 1000
val_action_mse_batches: 16     # default do val-fix (era 4)
```

`set_seed(seed)` cobre `random`, `numpy`, `torch`, `torch.cuda.manual_seed_all` e `accelerate.set_seed`. Não está ligado `torch.use_deterministic_algorithms(True)` → reprodutibilidade alta entre runs (mesma seed → mesma init/ordem/augmentation), mas não bit-exact (kernels CUDA). O pareamento bit-exato vem de logar **raw vs EMA dentro do mesmo run 3** (mesmos pesos, mesmo step).

---

## Run 3 — o que muda (resumo de 1 parágrafo)

Treino novo do zero, mesma config do Run 2 (seed 1000), mantendo os val-fixes (`eval mode`, ruído/timestep fixos via `fork_rng`, `val_action_mse` em 16 batches). Única mudança de modelo: **EMA dos pesos treináveis** (`[p for p in policy.parameters() if p.requires_grad]`, 693M), atualizada após cada `optimizer.step()`, `decay=0.999` com warmup começando no step 0. A validação loga **raw e EMA lado a lado**; o best sai de `val_action_mse_ema`. O checkpoint guarda **raw + EMA**; resume continua do raw, deploy usa EMA.

---

## Respostas às 5 perguntas finais da 2ª rodada (com dado medido)

1. **EMA em fp32 ou no dtype dos pesos?** → **fp32** (~2,77 GB). Cabe: medi a VRAM agora — TRX50 16,8/24,6 GB (~5 GB de folga já somando a EMA), Atena 15,8/82 GB (~64 GB livres). fp32 pela estabilidade numérica.
2. **Gradient accumulation?** → **Não.** Verificado: `Accelerator(...)` sem `gradient_accumulation_steps`, `effective_batch_size = batch_size × num_processes = 32 × 1 = 32`, e o loop faz `backward → optimizer.step → zero_grad → scheduler.step` por batch (1 step/batch). EMA após cada `optimizer.step()` está correto; mesmo assim gateamos em `accelerator.sync_gradients` (no-op hoje, à prova de futuro).
3. **Single ou multi GPU?** → **Single GPU** (`num_processes=1`; TRX50 `CUDA_VISIBLE_DEVICES=0`→`cuda:0`, Atena `=2`→`cuda:0`). Por isso `fork_rng(devices=[0])` funciona hoje; mesmo assim adotamos a versão robusta `[accelerator.device.index]`.
4. **Run 2 e Run 3 mesmo commit exceto EMA?** → Sim. Run 2 rodou com `run_train_valfix.py` (commit `16e94f9`). A EMA entra **atrás de um flag** (`ema_enabled`, default off) no MESMO arquivo: off = idêntico ao que o Run 2 rodou; on = Run 3. Mesma seed (1000), mesmo dataset/scheduler. O caminho raw da validação fica intacto.
5. **Deploy: export EMA separado ou loader aplica `ema_state`?** → **Export separado** — um `pretrained_model/` com `model.safetensors` = pesos EMA aplicados. Vantagem: o script de inferência atual (`inference_realtime_pi05d_right14.py`) carrega sem nenhuma mudança (espera um checkpoint padrão).

## Config EMA fechada

```yaml
ema:
  enabled: true
  decay: 0.999              # 0.9999 atrasa demais para um run de 20k
  warmup: true              # cur_decay = min(decay, (1+t)/(10+t))
  start_step: 0
  update_every: 1           # todo optimizer.step()
  params: requires_grad_only  # os 693M treináveis; PaliGemma congelado fica fora
  eval_with_ema: true       # eval raw + EMA, best por val_action_mse_ema
  save_state: true          # ema_state.safetensors no training_state/; deploy = export EMA
```

> Run 3 começa do zero, mesma config do Run 2, mantendo os val-fixes. A EMA acompanha só os parâmetros treináveis (os mesmos do optimizer), em fp32, atualizada após cada `optimizer.step()` real, `decay=0.999` com warmup desde o step 0. A validação loga raw e EMA lado a lado (e o gap `ema_minus_raw`); o best é por `val_action_mse_ema`, monitorando `grasp_ema`. O checkpoint guarda raw + EMA — resume continua do raw, deploy usa um export EMA.
