# Reprodutibilidade — A/B armstate7 (3 runs)

**Escrito:** 2026-06-14 19:48 (-03)
**O que é:** manifesto pra replicar exatamente as 3 runs do A/B (as-is / val-fixes / EMA) — commit do código, config, máquina, comando, init e checkpoints de cada uma. Inclui os **furos honestos** de reprodutibilidade e como fechá-los.

---

## Setup compartilhado (vale pras 3)

| item | valor |
|---|---|
| **Dataset** | `lewislf/G1_Dex3_pick_white_cup_right8_1squeeze_armstate7` — 238 eps (treino 0–213, val 214–237) |
| | `action[0:7]` = braço direito · `action[7]` = squeeze (0→1) · `state[7]` = braço (armstate7: **sem** os dedos medidos) |
| **Env** | conda `ms3` (py3.10, torch 2.x) |
| **Init (pi05_base)** | **PINADO na revisão `a538eb273274` (2026-06-01)** — NÃO usar o HEAD do `pi05_base`: o commit `7de663` (06-03) adicionou `relative_actions_processor`, que **quebra** no lerobot 0.4.4 (`ImportError` no `make_pre_post_processors`) |
| **Treino** | 20k steps · batch 32 · AdamW lr 1e-4 · warmup 2000 · cosine→1e-5 em 18000 · **seed 1000** |
| **Runtime (obrigatório)** | `OMP_NUM_THREADS=8` (+ MKL/NUMEXPR/OPENBLAS) — senão **deadlock** por explosão de threads em SSH não-interativo |
| **Checkpoints** | `keep_only_best_and_last: true` → só `best` + `last` |
| **wandb** | entity `prometheus-lcad`, project `prometheus_g1`, `disable_artifact: true` · conta luiz-coutinho (chave do `~/.netrc` do laptop) |

---

## Run 1 — AS-IS (`6kr7d8nz`)

| | |
|---|---|
| **Máquina/GPU** | Atena (`hercules@10.9.8.252`), **GPU2** (`CUDA_VISIBLE_DEVICES=2`) |
| **Commit** | **`6004519`** (HEAD da Atena). Código = `run_train.py` **original** (best_meta persist `87b4ffb` + best=val_action_mse `dca37f3`), **sem** val-fixes, **sem** EMA |
| **Config** | `train_cup_pi05_right8_armstate7.yaml` ⚠️ **NÃO está no git** (só na Atena) · `pretrained_path: lerobot/pi05_base` |
| **Init** | `lerobot/pi05_base` via **`HF_HUB_OFFLINE=1`** (cache pré-06-03 = revisão `a538eb273274`), `HF_HOME=/data/huggingface-models` |
| **Comando (resume)** | `python lerobot-ext/train/run_train.py --config_path=train_output/cup_pi05_right8_armstate7_lf/checkpoints/last/pretrained_model/train_config.json --resume=true --wandb.run_id=6kr7d8nz` |
| **Env** | `CUDA_VISIBLE_DEVICES=2 HF_HOME=/data/huggingface-models HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false WANDB_API_KEY=<luiz>` |
| **Checkpoints** | `train_output/cup_pi05_right8_armstate7_lf/checkpoints/{best,last}` |
| **Papel no A/B** | baseline — código original, régua antiga (val_action_mse em 4 batches, ruído não-fixo) |

> **Gotcha do resume:** o `--config_path` tem que apontar pro `train_config.json` do checkpoint (não pro YAML) — senão o lerobot deriva o `pretrained_path` da pasta do YAML e dá `FileNotFoundError`.

---

## Run 2 — VAL-FIXES (`y32omum0`)

| | |
|---|---|
| **Máquina/GPU** | TRX50 (`luiz@100.90.161.46`, aumo), **GPU0** (`CUDA_VISIBLE_DEVICES=0`) |
| **Commit** | ⚠️ **TRX50 não é repo git (foi rsync)** → sem commit. **Equivalente committado = `run_train_valfix.py @ 16e94f9`** (os 4 val-fixes: `policy.eval()` no val, ruído+timestep fixos via `fork_rng`, ruído fixo no `predict_action_chunk`, `val_action_mse` em 16 batches). **Sem** EMA |
| **Config** | `train_cup_pi05_right8_armstate7_valfix.yaml` ⚠️ **NÃO está no git** (só na TRX50) · `pretrained_path: /home/luiz/.../models/pi05_base_0601` |
| **Init** | `models/pi05_base_0601` (revisão `a538eb273274` baixada localmente) |
| **Comando** | `python lerobot-ext/train/run_train.py --config_path=lerobot-ext/config/train/train_cup_pi05_right8_armstate7_valfix.yaml` |
| **Env** | `CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_TOKEN=<hf> WANDB_API_KEY=<luiz>` |
| **Checkpoints** | `train_output/cup_pi05_right8_armstate7_valfix_lf/checkpoints/{best,last}` |
| **Papel no A/B** | a régua nova (val-fixes) — é **MEDIÇÃO**, não muda o modelo vs a as-is |

---

## Run 3 — EMA (a lançar) ✅ a mais limpa

| | |
|---|---|
| **Máquina/GPU** | TRX50 GPU1 **ou** Atena GPU2 (a decidir) |
| **Commit** | **`2f58a3c`** (`run_train_valfix.py` com **EMA** + fix do `fork_rng`) · config no commit **`2938630`**. ✅ **totalmente committado** |
| **Config** | `train_cup_pi05_right8_armstate7_ema.yaml` ✅ **no git** · `ema_enabled:true, ema_decay:0.999, ema_warmup:true, ema_start_step:0, seed:1000` |
| **Init** | TRX50: `models/pi05_base_0601` · Atena: `lerobot/pi05_base` (HF_HUB_OFFLINE) → override com `--policy.pretrained_path=<base>` |
| **Comando** | `python lerobot-ext/train/run_train_valfix.py --config_path=lerobot-ext/config/train/train_cup_pi05_right8_armstate7_ema.yaml [--policy.pretrained_path=<base se Atena>]` |
| **Env** | `CUDA_VISIBLE_DEVICES=<1 ou 2> OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_TOKEN=<hf> WANDB_API_KEY=<luiz>` · **só na TRX50:** `WANDB_HOST_OVERRIDE=gpu-trx50` (na Atena o host já é `atena`) |
| **Checkpoints** | `train_output/cup_pi05_right8_armstate7_ema_lf/checkpoints/{best,last}` — best inclui `training_state/ema_state.safetensors` + `pretrained_model_ema/` (deploy) |
| **Papel no A/B** | + EMA dos pesos treináveis — a **única variável de modelo**. Deploy usa o `pretrained_model_ema/` |

> Validado por smoke (12 steps) em 2026-06-14: EMA cria/atualiza, eval raw+ema, best por `val_action_mse_ema`, `ema_state` + export salvos, host `gpu-trx50` confirmado.

---

## A lógica do A/B
- **Run 1 × Run 2** = as-is × régua. MEDIÇÃO (modelo ~igual) → mostra validação mais limpa/comparável.
- **Run 2 × Run 3** = sem-EMA × EMA. MODELO → mostra o efeito da EMA. Mais o **raw vs ema dentro do run 3** (mesma trajetória) = comparação mais apertada, sem variância run-a-run.

---

## ⚠️ Furos de reprodutibilidade (honesto) + como fechar
1. **Configs da run 1 e run 2 não estão no git** (só nas máquinas). → committar `train_cup_pi05_right8_armstate7.yaml` e `..._valfix.yaml`.
2. **TRX50 não é repo git** (rsync) → run 2 não tem commit; o equivalente fiel é `run_train_valfix.py @ 16e94f9`.
3. **`run_train.py` (as-is) tem mods locais** no laptop; a Atena está em `6004519` — pra reproduzir bit-a-bit, conferir que o `run_train.py` da Atena bate com `6004519` (ou committar o estado exato dela).
4. **Não-determinismo CUDA:** sem `torch.use_deterministic_algorithms(True)`, runs não são bit-exatas entre si — o pareamento forte é o raw×ema **dentro** do run 3.

## Como replicar a run 3 (a mais limpa)
```bash
git checkout 2f58a3c            # ou a branch Luiz-pi05d
conda activate ms3
# TRX50 GPU1:
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
  WANDB_HOST_OVERRIDE=gpu-trx50 HF_TOKEN=<hf> WANDB_API_KEY=<luiz> \
  python lerobot-ext/train/run_train_valfix.py \
  --config_path=lerobot-ext/config/train/train_cup_pi05_right8_armstate7_ema.yaml
```
