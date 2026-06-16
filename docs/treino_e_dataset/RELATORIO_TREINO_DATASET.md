# Relatório de Auditoria — Políticas VLA π0.5 no Unitree G1 Dex3 (14-dim vs 8-dim)

**Tarefa:** "Pick up the white cup" · **Data:** 2026-06-13
**Escopo:** Duas políticas π0.5 idênticas exceto pelo *action space* — 14-dim (`cup_pi05_right14_rgb238_lf`, wandb `35jrrbk0`) e 8-dim (`cup_pi05_right8_1squeeze_lf`, wandb `8hajpdab`).

> **Nota de método.** Todos os números deste relatório foram **aterrados em código e dados reais**, lidos diretamente nesta auditoria: leitura dos parquet/`meta/info.json` dos datasets LeRobot v3.0 (via pyarrow/pandas), inspeção do código de treino (`run_train.py`, `init_lerobot_train.py`, `slice_right_arm_1squeeze.py` e os YAMLs de config) e extração das curvas/configs do W&B via `wandb.Api()` (`scan_history` + `summary` + `config`). Quando algo não estava disponível (ex.: success rate em sim/real), isso é declarado explicitamente. Nada foi inventado ou estimado fora dos dados observados.

---

## Resumo comparativo (lado a lado)

| Item | **14-dim** (`right14_rgb238`) | **8-dim** (`right8_1squeeze`) |
|---|---|---|
| Dataset (root local) | `lerobot-ext/datasets/G1_Dex3_right14_dataset/v3_238ep` | `datasets/_merged/G1_Dex3_pick_white_cup_right8_1squeeze` |
| repo_id (HF) | `lewislf/G1_Dex3_right14_dataset` | `lewislf/G1_Dex3_pick_white_cup_right8_1squeeze` |
| Formato / robô / fps | LeRobot v3.0 / `unitree_g1_dex3` / 30 | LeRobot v3.0 / `unitree_g1_dex3` / 30 |
| Episódios / frames / tasks | 238 / 52 952 / 1 | 238 / 52 952 / 1 |
| Split train / val | [0–213] (214) / [214–237] (24) | [0–213] (214) / [214–237] (24) |
| **action dim** | **14** (7 braço + 7 dedos) | **8** (7 braço + 1 squeeze) |
| **state dim** | 14 (7 braço + 7 dedos medidos) | 14 (7 braço + 7 dedos medidos) — **idêntico** |
| Encoding da mão | 7 juntas comandadas diretamente | 1 escalar `squeeze` ∈ [0,1] → `dedos = squeeze × RIGHT_TARGET` |
| Sensores presentes | RGB (mp4) + depth uint16 (PNG) + tátil [108]×2 | RGB (mp4) + depth uint16 (PNG) + tátil [108]×2 — **idêntico** |
| Run W&B (id) | `35jrrbk0` | `8hajpdab` |
| Status | **finished** (20000 steps) | **crashed** (~15900 de 20000) |
| MIN val_action_mse (step) | **0,075899 @ 5500** | **0,045644 @ 8000** |
| Best checkpoint (deduzido) | step **5500** | step **8000** |
| val_action_mse final | 0,190237 (s20000) | 0,053861 (s15500) |

---

## 1. Como o treino foi feito

### 1.1 Pipeline de treinamento

**Ponto de entrada:** `/home/luiz-aumo/I2CA/prometheus-vla/lerobot-ext/init_lerobot_train.py` (linhas 48-62) — lê o YAML de config via `sys.argv`, importa o módulo `policies` (linha 25) que registra dinamicamente a classe `PI05DepthConfig` no registry, e delega para `run_train.main()`.

**Motor de treinamento:** `/home/luiz-aumo/I2CA/prometheus-vla/lerobot-ext/train/run_train.py`, função `train()` decorada com `@parser.wrap()` (linha 246):

1. **Setup e logging** (251-276): cria `Accelerator` para treino distribuído com DDP; registra config no W&B se habilitado.
2. **Dataset** (285-311): `make_dataset(cfg)` carrega o dataset de treino; `_guard_zero_range_quantile_stats(dataset)` (linha 289) previne divisão por ~zero em dimensões congeladas (encoder LSB < 1e-3 rad); cria val dataset se `cfg.val_dataset` presente.
3. **Policy** (313-383): `make_policy(cfg=cfg.policy, ds_meta=dataset.meta, ...)` instancia o Pi0.5. Se `cfg.depth_fusion=true`, executa injeção de módulos 3D — `inject_pi05_d()` (completo: profundidade + tátil) ou `inject_pi05_depth()` (ablação depth-only). **Em ambas as runs deste relatório `depth_fusion=false` (RGB only), então a injeção 3D NÃO é usada.**
4. **Pre/Post-processor** (391-429): cria normalizadores de input/output com stats do dataset; `make_pre_post_processors()` injeta as estatísticas de normalização QUANTILES.
5. **Otimizador e scheduler** (432-433): Adam(W) com cosine decay.
6. **DataLoader** (479-535): `EpisodeAwareSampler` respeita boundaries de episódio; batch efetivo = `cfg.batch_size × num_processes`.
7. **Loop** (585-896):
   ```python
   for _ in range(step, cfg.steps):
       batch = next(dl_iter)
       batch = preprocessor(batch)          # normaliza input
       train_tracker, output_dict = update_policy(...)  # um passo SGD
       step += 1
   ```
   - `update_policy()` (171-243): forward, loss, backward, `clip_grad_norm_` (=1.0), `optimizer.step()`.
   - **Logging a cada `log_freq`:** escalares (loss, grad_norm, lr) + per-dim (`loss_dim_XX`).
   - **Validação a cada `eval_freq`:** forward nos batches de val + `val_action_mse` via `policy.predict_action_chunk()` (até 4 batches, linha 654).
   - **Critério de best** (752-766): baseado **APENAS** em `val_action_mse` (mudança 2026-06-10, commit `dca37f3`); se melhora, sobrescreve o best como cópia real em disco (774-803).
   - **Checkpoints** (805-848): se `keep_only_best_and_last=true` → rolling `last` + `best`; senão → numerados a cada `save_freq` + symlink `last`.

### 1.2 Hiperparâmetros reais (lado a lado)

| Parâmetro | **right14_rgb238** | **right8_1squeeze** |
|---|---|---|
| Pretrained path | `lerobot/pi05_base` | `lerobot/pi05_base` |
| PaliGemma / action expert | `gemma_2b` / `gemma_300m` | `gemma_2b` / `gemma_300m` |
| Image resolution | [224, 224] | [224, 224] |
| Max state/action dim | 32 / 32 | 32 / 32 |
| Chunk size (n_action_steps) | 50 / 50 | 50 / 50 |
| N obs steps | 1 | 1 |
| Normalization | VISUAL=IDENTITY, STATE=QUANTILES, ACTION=QUANTILES | VISUAL=IDENTITY, STATE=QUANTILES, ACTION=QUANTILES |
| Train expert only | `true` | `true` |
| Freeze vision encoder | `false` | `false` |
| Gradient checkpointing | `true` | `true` |
| Dtype | `bfloat16` | `bfloat16` |
| Optimizer LR | 1.0e-4 | 1.0e-4 |
| Optimizer betas / wd / grad clip | (0.9, 0.95) / 0.01 / 1.0 | (0.9, 0.95) / 0.01 / 1.0 |
| Warmup / decay steps / decay LR | 2000 / 18000 / 1.0e-5 | 2000 / 18000 / 1.0e-5 |
| Total steps | 20000 | 20000 |
| Batch size / num workers | 32 / 8 | 32 / 8 |
| Log / save / eval freq | 100 / 500 / 500 | 100 / 1000 / 500 |
| Depth fusion | `false` (RGB only) | `false` (RGB only) |
| Keep only best/last | `true` | `false` |
| W&B project / entity | `prometheus_g1` / `prometheus-lcad` | `prometheus_g1` / `prometheus-lcad` |

### 1.3 Input/Output features por config

**`train_cup_pi05_right14_rgb_238.yaml`** (linhas 98-109):
```yaml
input_features:
  observation.images.head_camera: { type: VISUAL, shape: [3, 480, 848] }   # RGB 16:9
  observation.state:              { type: STATE,  shape: [14] }            # 7 braço + 7 dedos MEDIDOS
output_features:
  action:                         { type: ACTION, shape: [14] }            # 7 braço + 7 dedos COMANDO
```

**`train_cup_pi05_right8_1squeeze.yaml`** (linhas 101-112):
```yaml
input_features:
  observation.images.head_camera: { type: VISUAL, shape: [3, 480, 848] }   # RGB idêntico
  observation.state:              { type: STATE,  shape: [14] }            # idêntico, NÃO muda
output_features:
  action:                         { type: ACTION, shape: [8] }             # 7 braço COMANDO + 1 squeeze
```

Estado (14 dims, idêntico nas duas): índices [0-6] = 7 juntas do braço direito (medidas); [7-13] = 7 juntas dos dedos (medidas).

### 1.4 A única diferença: action space

- **right14 (14-dim):** output = 7 dims de braço + 7 dims de dedos; a policy prevê a posição-alvo de cada junta do dedo individualmente.
- **right8 (8-dim):** output = 7 dims de braço + 1 escalar `squeeze` ∈ [0,1]; no deploy `right_hand_q = squeeze × RIGHT_TARGET`, com `RIGHT_TARGET = [0.0, -0.920, -1.74, 1.57, 1.74, 1.57, 1.74]` (`slice_right_arm_1squeeze.py`, linhas 29, 207).

### 1.5 Como o dataset 8-dim é gerado (`slice_right_arm_1squeeze.py`)

O script `/home/luiz-aumo/I2CA/prometheus-vla/lerobot-ext/tools/slice_right_arm_1squeeze.py` transforma o dataset `v3_grasp` (action 32-dim / state 28-dim) em right8_1squeeze.

**Mapeamento de índices** (linhas 46-52):
```python
ACTION_INDICES = [7, 8, 9, 10, 11, 12, 13, 29]                         # 7 braço + right_squeeze (idx 29)
STATE_INDICES  = [7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25, 26, 27]  # 7 braço + 7 dedos
```

**5 passos:** (1) fatia colunas `action`/`observation.state` dos parquet via numpy indexing (76-88); (2) atualiza shapes/names em `meta/info.json` (91-101); (3) fatia stats q01/q99/mean/std em `meta/stats.json`, com **guard de zero-range** (65-73) que dá `q99 = q01 + 1.0` em dims congeladas para evitar `NaN` na quantile norm (104-117); (4) fatia stats por episódio em `meta/episodes` (120-141); (5) symlink dos vídeos (economiza espaço) e copia `tasks.parquet` (144-204).

### 1.6 Fluxo resumido

```
train(cfg)
├─ make_dataset(cfg) → LeRobotDataset com stats (q01, q99)
│  └─ _guard_zero_range_quantile_stats() → evita divisão por ~eps
├─ make_policy(cfg.policy, ds_meta) → Pi0.5
│  └─ [se depth_fusion] inject_pi05_d() → PointNet + PressureProj  (NÃO usado aqui)
├─ make_optimizer_and_scheduler → AdamW + CosineDecay
├─ DataLoader com EpisodeAwareSampler
└─ for step in [0, cfg.steps):
   ├─ update_policy(batch, optimizer, policy, grad_clip_norm=1.0)
   ├─ [eval_step] val_action_mse via predict_action_chunk → best = min(val_action_mse)
   └─ [save_step] keep_only_best_and_last ? rolling last+best : numerados + symlink last
```

**Referências de código:** `init_lerobot_train.py:48-62` (entrada); `run_train.py:246` (`train`), `:288-311` (dataset+guard), `:320-324` (policy), `:334-382` (injeção 3D não usada), `:171-243` (`update_policy`), `:585-602` (loop), `:637-803` (val + best); `slice_right_arm_1squeeze.py:46-52` (índices), `:76-141` (transformação).

---

## 2. Dataset 14-dim — estrutura e episódio 0

Dataset: `/home/luiz-aumo/I2CA/prometheus-vla/lerobot-ext/datasets/G1_Dex3_right14_dataset/v3_238ep` (LeRobot v3.0). Episódio escolhido: **0** — single-pick representativo (167 frames, dentro da faixa típica: min 118 / mediana 208 / máx 796). Os dedos vão de totalmente abertos a totalmente fechados no RIGHT_TARGET — grasp completo.

### 2.1 Estrutura geral (`meta/info.json`)

| Campo | Valor |
|---|---|
| codebase_version | v3.0 |
| robot_type | unitree_g1_dex3 |
| fps | 30 |
| total_episodes | 238 |
| total_frames | 52 952 |
| total_tasks | 1 |
| data_path | `data/chunk-{chunk:03d}/file-{file:03d}.parquet` |
| video_path | `videos/{video_key}/chunk-{chunk:03d}/file-{file:03d}.mp4` |

Tasks (`meta/tasks.parquet`): única, `task_index=0` → **"Pick up the white cup"**.

**Features:**

| Feature | dtype | shape | Armazenamento |
|---|---|---|---|
| `action` | float32 | [14] | array embutido no parquet (`list<float>`) |
| `observation.state` | float32 | [14] | array embutido no parquet (`list<float>`) |
| `observation.images.head_camera` | video | [480, 848, 3] | **vídeo mp4 externo** (h264, yuv420p, 30fps) |
| `observation.images.head_camera_depth` | image | [480, 848, 1] | **PNG por-frame embutido** no parquet (`struct<bytes,path>`); uint16 |
| `observation.left_hand_pressure` | float32 | [108] | array embutido |
| `observation.right_hand_pressure` | float32 | [108] | array embutido |
| `timestamp` | float32 | [1] | escalar |
| `frame_index` / `episode_index` / `index` / `task_index` | int64 | [1] | escalar |

Nomes das 14 dims de `action`/`state` (idênticos): braço `kRightShoulderPitch/Roll/Yaw.q`, `kRightElbow.q`, `kRightWristRoll/Pitch/Yaw.q` (0–6); dedos `right_hand_thumb_0/1/2_joint.q`, `right_hand_index_0/1_joint.q`, `right_hand_middle_0/1_joint.q` (7–13).

### 2.2 Layout em disco

- `data/chunk-000/`: **58 arquivos** `file-000…057.parquet`. Cada arquivo agrupa vários episódios consecutivos (não 1:1), limite ~100 MB. Ex.: `file-000` = eps 0–4 (984 linhas), `file-001` = eps 5–8, `file-057` = ep 237. O mapeamento exato vem de `meta/episodes/` (`episode_index → chunk_index, file_index, dataset_from_index, dataset_to_index`).
- `meta/episodes/chunk-000/file-000.parquet`: 1 linha por episódio (238), com `length`, ponteiros de chunk/arquivo, faixas de timestamp do vídeo e stats per-episódio.
- `videos/observation.images.head_camera/chunk-000/file-000.mp4`: **um único mp4** com todos os 52 952 frames concatenados (1765 s); cada episódio é recortado por `from_timestamp/to_timestamp`. O dir `videos/` é symlink para `datasets/_merged/G1_Dex3_pick_white_cup_v3_grasp/videos`.

Schema do parquet (pyarrow): `action: list<float>`, `observation.state: list<float>`, `observation.images.head_camera_depth: struct<bytes:binary, path:string>`, `*_hand_pressure: list<float>`, `timestamp: float`, `frame_index/episode_index/index/task_index: int64`. (`head_camera` RGB NÃO está no parquet — só no mp4.)

### 2.3 Dump do episódio 0

- **Frames:** 167 · **duração:** 5,533 s · **fps real:** 30,0 · **task:** "Pick up the white cup" (`task_index=0`)
- **Timestamps:** 0,0 → 5,533 s; `frame_index` 0→166. Dados em `data/chunk-000/file-000.parquet` (`dataset_from_index=0`, `to_index=167`); vídeo em `file-000.mp4` (ts 0,0→5,567 s).

**`observation.state[14]`** (rad) — min/max/mean e f0 / meio(83) / último(166):

| idx | dim | min | max | mean | f0 | fmid | flast |
|---|---|---|---|---|---|---|---|
| 0 | ShoulderPitch | -0.296 | 0.082 | -0.104 | 0.064 | -0.251 | -0.296 |
| 1 | ShoulderRoll | -0.111 | 0.017 | -0.034 | -0.049 | -0.004 | -0.111 |
| 2 | ShoulderYaw | -0.074 | 0.131 | -0.017 | -0.063 | -0.059 | 0.103 |
| 3 | Elbow | -0.648 | 0.729 | 0.077 | -0.095 | 0.649 | -0.648 |
| 4 | WristRoll | -0.003 | 0.086 | 0.026 | -0.002 | 0.009 | 0.086 |
| 5 | WristPitch | -0.556 | 0.282 | -0.138 | 0.133 | -0.486 | 0.282 |
| 6 | WristYaw | -0.014 | 0.159 | 0.052 | -0.014 | 0.022 | 0.158 |
| 7 | thumb_0 | -0.045 | -0.045 | -0.045 | -0.045 | -0.045 | -0.045 |
| 8 | thumb_1 | -0.221 | -0.064 | -0.146 | -0.064 | -0.220 | -0.213 |
| 9 | thumb_2 | -0.505 | -0.063 | -0.304 | -0.063 | -0.505 | -0.505 |
| 10 | index_0 | 0.029 | 0.401 | 0.232 | 0.029 | 0.400 | 0.401 |
| 11 | index_1 | 0.037 | 0.878 | 0.480 | 0.037 | 0.735 | 0.878 |
| 12 | middle_0 | 0.052 | 0.496 | 0.292 | 0.052 | 0.496 | 0.496 |
| 13 | middle_1 | 0.043 | 0.624 | 0.356 | 0.043 | 0.606 | 0.624 |

> Nota: `state` (juntas medidas) tem amplitude de fechamento menor que `action` (alvo) — os dedos não alcançam fisicamente o alvo comandado (encostam no copo), coerente com um grasp real.

**`action[14]`** (rad) — min/max/mean e f0 / meio(83) / último(166):

| idx | dim | min | max | mean | f0 | fmid | flast |
|---|---|---|---|---|---|---|---|
| 0 | ShoulderPitch | -0.376 | 0.050 | -0.206 | 0.038 | -0.199 | -0.376 |
| 1 | ShoulderRoll | -0.112 | 0.018 | -0.047 | -0.060 | 0.018 | -0.103 |
| 2 | ShoulderYaw | -0.078 | 0.131 | 0.010 | -0.064 | -0.035 | 0.071 |
| 3 | Elbow | -0.700 | 0.701 | -0.100 | -0.149 | 0.402 | -0.683 |
| 4 | WristRoll | -0.007 | 0.100 | 0.051 | -0.006 | 0.055 | 0.085 |
| 5 | WristPitch | -0.598 | 0.265 | -0.121 | 0.093 | -0.507 | 0.184 |
| 6 | WristYaw | -0.012 | 0.167 | 0.085 | -0.012 | 0.078 | 0.135 |
| 7 | thumb_0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 8 | thumb_1 | -0.920 | 0.000 | -0.559 | 0.000 | -0.920 | -0.920 |
| 9 | thumb_2 | -1.740 | 0.000 | -1.056 | 0.000 | -1.740 | -1.740 |
| 10 | index_0 | 0.000 | 1.570 | 0.953 | 0.000 | 1.570 | 1.570 |
| 11 | index_1 | 0.000 | 1.740 | 1.056 | 0.000 | 1.740 | 1.740 |
| 12 | middle_0 | 0.000 | 1.570 | 0.953 | 0.000 | 1.570 | 1.570 |
| 13 | middle_1 | 0.000 | 1.740 | 1.056 | 0.000 | 1.740 | 1.740 |

**`head_camera` (RGB):** [480, 848, 3], uint8, em mp4 (h264 yuv420p, 30fps). Frame do meio (t=2,8 s) em `midia/ep_14dim_rgb.png` — mão Dex3 já fechada segurando o copo branco sobre a bancada.

**`head_camera_depth`:** [480, 848] (canal único), **uint16**, unidade **mm**, PNG embutido por frame (`path=None`, dados inline). Frame do meio: global min/max = 0 / 890 mm; pixels válidos (402 550 px) min/max/mean = 408 / 890 / 636 mm; p1/p50/p99 = 426 / 628 / 865 mm. Zeros = buracos de profundidade em bordas/oclusão. Visualização (turbo) em `midia/ep_14dim_depth.png`.

**Tátil/pressão:** `observation.left_hand_pressure` e `observation.right_hand_pressure`, cada uma float32 [108] (layout 9 sensores × 12 slots). Baseline de repouso ~30000 (não-zero), por isso todos os slots aparecem preenchidos. Ep0 mão direita: min/max/mean = 30000 / 107824 / 52317; pico por-frame 104856 (f0) → 107504 (meio) → 107808 (último), coerente com aumento de contato ao fechar a mão. Mão esquerda (não usada): 30000 / 104864 / 52131.

**Relação dedos ↔ squeeze (confirmação nos dados):** com `RIGHT_TARGET = [0, -0.92, -1.74, 1.57, 1.74, 1.57, 1.74]`, o squeeze implícito por frame (projeção de `action[7:14]` sobre RIGHT_TARGET) reconstrói `recon = squeeze × RIGHT_TARGET` com **erro absoluto máximo = 0,0** (mean 0,0) em todas as 7 dims — ou seja, `action[7:14]` é **exatamente** `squeeze × RIGHT_TARGET`. O squeeze implícito varia de 0,0 a 1,0 (média 0,607).

| frame | squeeze | action[7:14] | recon |
|---|---|---|---|
| f0 | 0.000 | [0, 0, 0, 0, 0, 0, 0] | [0, 0, 0, 0, 0, 0, 0] |
| meio (83) | 1.000 | [0, -0.92, -1.74, 1.57, 1.74, 1.57, 1.74] | idem |
| último (166) | 1.000 | [0, -0.92, -1.74, 1.57, 1.74, 1.57, 1.74] | idem |

Isso comprova que, no canal de dedos da `action`, o dataset 14-dim é **redundante**: os 7 valores são gerados por um único escalar squeeze multiplicando RIGHT_TARGET — exatamente o que o dataset 8-dim codifica diretamente.

**Imagens salvas:**
- `midia/ep_14dim_rgb.png` (80 527 bytes; 848×480 RGB, frame 83 do ep0)
- `midia/ep_14dim_depth.png` (106 631 bytes; turbo da profundidade em mm, frame 83 do ep0)

---

## 3. Dataset 8-dim (squeeze) — estrutura e episódio 0

Dataset: `/home/luiz-aumo/I2CA/prometheus-vla/datasets/_merged/G1_Dex3_pick_white_cup_right8_1squeeze`. LeRobot v3.0, `unitree_g1_dex3`, fps=30, 238 episódios, 52 952 frames, 1 task ("Pick up the white cup"), split `train: 0:238`.

### 3.1 Estrutura (`meta/info.json`) — features

Este dataset é o mesmo conteúdo do `..._v3_grasp` (238 eps, fonte do right14) **re-derivado para action de 8 dims**. Não há campo "fontes" no info.json, mas a linhagem é inferível: o diretório `videos/` é **symlink** para `G1_Dex3_pick_white_cup_v3_grasp/videos` (mesmas câmeras/episódios; só o parquet/action foi reescrito).

| Feature | dtype | shape | Armazenamento |
|---|---|---|---|
| `action` | float32 | **[8]** | array embutido no parquet (list<float>) |
| `observation.state` | float32 | **[14]** | array embutido no parquet (list<float>) |
| `observation.images.head_camera` | video | [480, 848, 3] | mp4 h264 yuv420p, 30fps (via symlink p/ v3_grasp) |
| `observation.images.head_camera_depth` | image | [480, 848, 1] | **PNG embutido no parquet** (struct bytes+path), uint16 |
| `observation.left_hand_pressure` | float32 | [108] | array embutido |
| `observation.right_hand_pressure` | float32 | [108] | array embutido |
| `timestamp` | float32 | [1] | parquet |
| `frame_index`/`episode_index`/`index`/`task_index` | int64 | [1] | parquet |

`action[8]` = 7 juntas do braço direito + **`right_grasp_squeeze.q`** (escalar). `observation.state[14]` = 7 juntas do braço + **7 juntas medidas dos dedos** — ou seja, o state continua **[14]**, idêntico ao right14; só a *action* mudou de 14→8.

### 3.2 Schema do parquet e diferença vs 14-dim

```
action: list<float>                                  # 8 elementos (no 14-dim: 14)
observation.state: list<float>                       # 14 elementos (igual nos dois)
observation.images.head_camera_depth: struct<bytes:binary, path:string>   # PNG embutido
observation.left_hand_pressure / right_hand_pressure: list<float>         # 108 cada
timestamp: float | frame_index/episode_index/index/task_index: int64
```
Única diferença de schema vs o 14-dim: `action` tem **8** floats (7 braço + 1 squeeze) em vez de **14**. Tudo o mais (state[14], depth, tátil, timestamps) é idêntico.

### 3.3 Dump do episódio 0

- **Frames:** 167 · **duração:** 5,533 s (0,000 → 5,533 s) · **task_index:** 0 · frame_index 0→166. Em `data/chunk-000/file-000.parquet`, linhas 0–166. Vídeo: `videos/observation.images.head_camera/chunk-000/file-000.mp4` (from_timestamp=0,0).

**`observation.state[14]`** — 7 braço + 7 dedos medidos:

| dim | nome | min | max | mean | std | first | mid(f83) | last |
|---|---|---|---|---|---|---|---|---|
| 0 | RShoulderPitch | -0.2957 | 0.0815 | -0.1037 | 0.1287 | 0.0641 | -0.2508 | -0.2957 |
| 1 | RShoulderRoll | -0.1112 | 0.0167 | -0.0337 | 0.0368 | -0.0493 | -0.0041 | -0.1112 |
| 2 | RShoulderYaw | -0.0737 | 0.1307 | -0.0169 | 0.0751 | -0.0632 | -0.0593 | 0.1025 |
| 3 | RElbow | -0.6475 | 0.7286 | 0.0765 | 0.4365 | -0.0952 | 0.6488 | -0.6475 |
| 4 | RWristRoll | -0.0030 | 0.0857 | 0.0258 | 0.0290 | -0.0021 | 0.0090 | 0.0857 |
| 5 | RWristPitch | -0.5561 | 0.2818 | -0.1379 | 0.2862 | 0.1325 | -0.4858 | 0.2815 |
| 6 | RWristYaw | -0.0137 | 0.1585 | 0.0517 | 0.0636 | -0.0137 | 0.0224 | 0.1581 |
| 7 | thumb_0 | -0.0451 | -0.0451 | -0.0451 | 0.0000 | -0.0451 | -0.0451 | -0.0451 |
| 8 | thumb_1 | -0.2214 | -0.0643 | -0.1463 | 0.0735 | -0.0644 | -0.2197 | -0.2128 |
| 9 | thumb_2 | -0.5048 | -0.0632 | -0.3037 | 0.2170 | -0.0633 | -0.5047 | -0.5048 |
| 10 | index_0 | 0.0293 | 0.4005 | 0.2315 | 0.1822 | 0.0293 | 0.3995 | 0.4005 |
| 11 | index_1 | 0.0368 | 0.8775 | 0.4798 | 0.4081 | 0.0368 | 0.7347 | 0.8775 |
| 12 | middle_0 | 0.0519 | 0.4956 | 0.2924 | 0.2179 | 0.0519 | 0.4955 | 0.4955 |
| 13 | middle_1 | 0.0427 | 0.6240 | 0.3558 | 0.2842 | 0.0427 | 0.6059 | 0.6240 |

**`action[8]`** — 7 braço + squeeze:

| dim | nome | min | max | mean | std | first | mid(f83) | last |
|---|---|---|---|---|---|---|---|---|
| 0 | RShoulderPitch | -0.3763 | 0.0497 | -0.2060 | 0.1342 | 0.0381 | -0.1994 | -0.3763 |
| 1 | RShoulderRoll | -0.1124 | 0.0180 | -0.0467 | 0.0459 | -0.0598 | 0.0180 | -0.1029 |
| 2 | RShoulderYaw | -0.0784 | 0.1313 | 0.0100 | 0.0795 | -0.0644 | -0.0348 | 0.0713 |
| 3 | RElbow | -0.6998 | 0.7009 | -0.0999 | 0.5069 | -0.1493 | 0.4025 | -0.6832 |
| 4 | RWristRoll | -0.0070 | 0.1000 | 0.0510 | 0.0344 | -0.0055 | 0.0546 | 0.0852 |
| 5 | RWristPitch | -0.5981 | 0.2654 | -0.1208 | 0.3096 | 0.0929 | -0.5067 | 0.1844 |
| 6 | RWristYaw | -0.0119 | 0.1673 | 0.0851 | 0.0647 | -0.0119 | 0.0783 | 0.1346 |
| **7** | **squeeze** | **0.0000** | **1.0000** | **0.6070** | **0.4838** | **0.0000** | **1.0000** | **1.0000** |

**Curva do squeeze (dim 7) ao longo do ep0:**

| % | frame | t (s) | squeeze |
|---|---|---|---|
| 0 | 0 | 0.000 | 0.0000 |
| 10 | 17 | 0.567 | 0.0000 |
| 20 | 33 | 1.100 | 0.0000 |
| 30 | 50 | 1.667 | 0.0000 |
| 40 | 66 | 2.200 | 0.7964 |
| 50 | 83 | 2.767 | 1.0000 |
| 60–100 | 100…166 | 3.33…5.53 | 1.0000 |

Frame em que squeeze cruza ≥0,5: **frame 65 (t≈2,167 s)** — a mão aproxima aberta (squeeze=0) na 1ª metade, fecha rápido entre os frames ~64–66 e permanece travada em 1,0 até o fim (pega e segura o copo). Bate com a imagem do meio (mão já fechada).

**RGB:** (480, 848, 3), uint8, mp4 h264 (via symlink p/ v3_grasp). Frame do meio (f83, t=2,767 s) em `midia/ep_8dim_rgb.png` (85 931 bytes) — mão Dex3 envolvendo o copo branco.

**depth e tátil — ambos presentes neste _merged** (o slice 14→8 NÃO removeu nada; só reescreveu a `action`):
- **depth:** `observation.images.head_camera_depth` — PNG embutido, decodifica para (480, 848) **uint16**; frame do meio min/max = 0–890 mm (buracos=0). Visualização em `midia/ep_8dim_depth.png` (33 252 bytes).
- **tátil:** `observation.left_hand_pressure`/`right_hand_pressure`, ambos [108] float32. Ep0: left 30000 / 104864 / 52131; right 30000 / 107824 / 52317 (baseline ~30000, picos ao encostar).

### 3.4 Relação com o 14-dim (squeeze × RIGHT_TARGET)

Confirmado: o squeeze (action dim 7) ∈ **[0, 1]** tanto no ep0 (min 0,0, max 1,0) quanto no dataset inteiro (`meta/stats.json`: action[7] min=0,0, max=1,0, mean=0,411). A relação com os 7 dedos do right14 é linear: **dedos_right14 = squeeze × RIGHT_TARGET**:
- squeeze=0 → `[0, 0, 0, 0, 0, 0, 0]` (mão aberta)
- squeeze=0,5 → `[0, -0.46, -0.87, 0.785, 0.87, 0.785, 0.87]`
- squeeze=1 → `[0, -0.92, -1.74, 1.57, 1.74, 1.57, 1.74]` (pose fechada plena)

Ou seja, o right8 colapsa as 7 dims de dedo da action do right14 num único grau de liberdade (interpolação no segmento aberto→RIGHT_TARGET). O state continua [14] (mede os 7 dedos reais); só a *saída* da política é comprimida, a *observação* dos dedos permanece completa.

---

## 4. Runs do W&B (`35jrrbk0` · `8hajpdab`)

Projeto: `prometheus-lcad/prometheus_g1`. Entity logada: `luiz-coutinho`. Todos os números abaixo foram extraídos via `wandb.Api()` (`scan_history` + `summary` + `config`).

> **Leitura importante (8-dim).** O `summary._step=189` é apenas o contador de eventos de log do W&B, **não** o nº de passos de treino. Os campos reais de progresso são `train/global_step=15900` e `eval/global_step=15500`. A run treinou ~15.900 de 20.000 passos e então **crashou** (state=`crashed`); não parou em 189 steps.

### Run A — `35jrrbk0` (14-dim, `cup_pi05_right14_rgb238_lf`)

**Identidade / config**

| Campo | Valor |
|---|---|
| name / id / state | cup_pi05_right14_rgb238_lf / 35jrrbk0 / **finished** |
| created_at | 2026-06-10T11:43:54Z |
| runtime | 70.041 s (~19,5 h) |
| steps logados (train) | 200 pontos, global_step 100→**20000** (completou) |
| evals logados | 40 pontos, global_step 500→20000 (eval_freq=500) |
| epochs / episodes vistos | 13,43 epochs / 2873 episódios (samples=640.000) |
| lr / scheduler | 1e-4 (adamw, betas [0.9,0.95], wd 0.01, grad_clip 1) / cosine_decay_with_warmup, peak 1e-4 → 1e-5, warmup 2000, decay 18000 |
| batch_size | 32 |
| action_dim (output) | **14** (`policy.output_features.action.shape=[14]`); state=[14] |
| policy | pi05, chunk_size=50, n_action_steps=50, **train_expert_only=True**, freeze_vision_encoder=False |
| dataset / repo_id | lerobot-ext/datasets/G1_Dex3_right14_dataset/v3_238ep / lewislf/G1_Dex3_right14_dataset |
| keep_only_best_and_last | True · save_freq=500 · resume=True |

**Curvas (eval a cada 500)**

| target | t.step | train/loss | lr | e.step | val_loss | val_action_mse |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100 | 0.172597 | 2.574e-06 | 500 | 0.097651 | 0.173093 |
| 500 | 500 | 0.034070 | 2.256e-05 | 500 | 0.097651 | 0.173093 |
| 2000 | 2000 | 0.024956 | 9.750e-05 | 2000 | 0.097662 | 0.096790 |
| 5500 | 5500 | 0.018212 | 8.113e-05 | 5500 | 0.124385 | **0.075899** (mín mse) |
| 8000 | 8000 | 0.019843 | 6.320e-05 | 8000 | 0.184892 | 0.116017 |
| 10000 | 10000 | 0.027294 | 4.757e-05 | 10000 | 0.215192 | 0.092215 |
| 15000 | 15000 | 0.007992 | 1.623e-05 | 15000 | 0.285113 | 0.185215 |
| 20000 (final) | 20000 | 0.008788 | 1.000e-05 | 20000 | 0.317392 | 0.190237 |

- **MIN val_loss** = 0,085704 @ step **2500** · **MIN val_action_mse** = 0,075899 @ step **5500**
- FINAL: train/loss=0,008788, val_loss=0,317392, val_action_mse=0,190237 (step 20000)
- MIN train/loss global = 0,004382

### Run B — `8hajpdab` (8-dim, `cup_pi05_right8_1squeeze_lf`)

**Identidade / config**

| Campo | Valor |
|---|---|
| name / id / state | cup_pi05_right8_1squeeze_lf / 8hajpdab / **crashed** |
| created_at | 2026-06-10T19:26:49Z |
| runtime | 54.261 s (~15,1 h) |
| steps logados (train) | 159 pontos, global_step 100→**15900** (crashou antes dos 20000) |
| evals logados | 31 pontos, global_step 500→15500 |
| epochs / episodes vistos | 10,67 epochs / 2284 episódios (samples=508.800) |
| lr / scheduler / batch | idênticos ao A (1e-4 peak, cosine warmup 2000 / decay 18000→1e-5; bs=32) |
| action_dim (output) | **8** (`policy.output_features.action.shape=[8]`); state=[14] |
| policy | pi05, chunk_size=50, n_action_steps=50, **train_expert_only=True**, freeze_vision_encoder=False |
| dataset / repo_id | datasets/_merged/G1_Dex3_pick_white_cup_right8_1squeeze / lewislf/G1_Dex3_pick_white_cup_right8_1squeeze |
| keep_only_best_and_last | False · save_freq=1000 · resume=False |

**Curvas (eval a cada 500)**

| target | t.step | train/loss | lr | e.step | val_loss | val_action_mse |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100 | 0.157732 | 2.574e-06 | 500 | 0.097513 | 0.093324 |
| 500 | 500 | 0.050181 | 2.256e-05 | 500 | 0.097513 | 0.093324 |
| 2000 | 2000 | 0.039555 | 9.750e-05 | 2000 | 0.122404 | 0.086998 |
| 5500 | 5500 | 0.017319 | 8.113e-05 | 5500 | 0.179074 | 0.049545 |
| 8000 | 8000 | 0.014004 | 6.320e-05 | 8000 | 0.227762 | **0.045644** (mín mse) |
| 10000 | 10000 | 0.016512 | 4.757e-05 | 10000 | 0.291419 | 0.049954 |
| 15000 | 15000 | 0.006216 | 1.623e-05 | 15000 | 0.371967 | 0.051896 |
| 15900 (último train) | 15900 | 0.007395 | 1.313e-05 | 15500 | 0.363374 | 0.053861 |

- **MIN val_loss** = 0,097513 @ step **500** · **MIN val_action_mse** = 0,045644 @ step **8000**
- ÚLTIMO ponto (crash): train/loss=0,007395 @ s15900; val_loss=0,363374, val_action_mse=0,053861 @ s15500
- MIN train/loss global = 0,002991

### Overfitting (números reais)

Em ambas as runs `train/loss` cai de forma quase monotônica enquanto `val_loss` despenca cedo e depois sobe sem parar — assinatura clássica de overfit.

- **14-dim (35jrrbk0):** train/loss 0,172597 (s100) → 0,034 (s500) → 0,018 (s5500) → **0,008788 (s20000)**. val_loss faz o caminho oposto a partir do vale: mín **0,085704 @ s2500** → 0,124 (s5500) → 0,215 (s10000) → 0,285 (s15000) → **0,317392 (s20000)** — val_loss final é **3,7×** o mínimo. O val_action_mse degrada após o vale: **0,075899 @ s5500** → 0,185 (s15000) → **0,190237 (s20000)**, ~2,5× o mínimo.
- **8-dim (8hajpdab):** train/loss 0,157732 (s100) → 0,050 (s500) → 0,017 (s5500) → **0,006216 (s15000)**. val_loss: mín **0,097513 @ s500** → 0,179 (s5500) → 0,291 (s10000) → **0,371967 @ s15000** (3,8× o mínimo). O val_action_mse aqui é mais plano/baixo (mín **0,045644 @ s8000**, fica ~0,05 até o crash) — porque só tem 1 dim de mão (squeeze) contra 6 de mão no 14-dim.
- **A mão é o que mais sofre.** No per-dim final do 14-dim, as dims do braço (00–06) ficam val_action_mse ≈ 0,011–0,067 e a dim 07 ≈ 5e-5, mas as **6 dims dos dedos (08–13) saturam todas em ≈ 0,404–0,406** — dominam o mse. No 8-dim, idem: braço (dims 00–06) ≈ 0,008–0,051, mas a **dim 07 (squeeze) = 0,2436** no estado final, ~5–30× as dims de braço.

### Checkpoint "best" por run

Critério vigente (commit `dca37f3`): **best = SÓ `val_action_mse` mínimo** (a dominância dupla foi removida porque congelava o best).

- **14-dim (35jrrbk0):** best deduzível = **step 5500** (val_action_mse=0,075899, o mínimo). Com `keep_only_best_and_last=True`, o checkpoint best gravado é esse.
- **8-dim (8hajpdab):** best deduzível = **step 8000** (val_action_mse=0,045644, o mínimo). Com `keep_only_best_and_last=False` e `save_freq=1000`, os checkpoints periódicos (de 1000 em 1000) foram mantidos; o de menor val_action_mse é 8000. Run crashou em ~15900 antes de completar 20000.

> Observação: nenhuma run loga métrica/flag explícita "is_best" ou o step do best no W&B. O "best" acima é **deduzido** do mínimo de `val_action_mse`, que é o critério no código de treino. Caminhos de checkpoint nas configs: 14-dim `train_output/cup_pi05_right14_rgb238_lf/checkpoints/last`; 8-dim `train_output/cup_pi05_right8_1squeeze_lf` (checkpoint_path=None, resume=False).

### Tabela comparativa (W&B)

| Métrica | 14-dim (35jrrbk0) | 8-dim (8hajpdab) |
|---|---:|---:|
| state | finished | **crashed** |
| último train step | 20000 | 15900 |
| último eval step | 20000 | 15500 |
| runtime | ~70.041 s | ~54.261 s |
| action_dim | 14 (7 braço + 7 dedos) | 8 (7 braço + 1 squeeze) |
| train_expert_only | True | True |
| MIN val_loss (step) | 0,085704 (2500) | 0,097513 (500) |
| MIN val_action_mse (step) | **0,075899 (5500)** | **0,045644 (8000)** |
| val_loss final | 0,317392 | 0,363374 |
| val_action_mse final | 0,190237 | 0,053861 |
| train/loss final | 0,008788 | 0,007395 |
| MIN train/loss | 0,004382 | 0,002991 |
| best checkpoint (deduzido) | step 5500 | step 8000 |
| val_loss final / mín | 3,71× | 3,73× |

**Não disponível no W&B:** success rate / métricas de rollout em sim ou real (só há `train/*` e `eval/val_*`), e qualquer flag explícita marcando o step do "best".

---

## 5. Leitura conjunta

- **As duas políticas têm efetivamente 1 grau de liberdade na mão.** Os dados do dataset 14-dim comprovam (erro absoluto máximo = 0,0) que `action[7:14] = squeeze × RIGHT_TARGET` em todo o episódio. Logo, o 14-dim apenas "infla" o mesmo escalar de aperto em 7 juntas via RIGHT_TARGET — não há informação extra de mão nas 7 dims. O 8-dim codifica esse fato diretamente. As políticas só diferem em *como representam a mesma intenção de aperto*.

- **Overfitting é claro e simétrico nas duas runs.** `train/loss` cai quase monotonicamente até ~0,007–0,009 enquanto `val_loss` atinge o vale cedo (s2500 no 14-dim, s500 no 8-dim) e sobe para ~3,7× o mínimo no fim. Isso vale a pena reforçar para a revisão: o checkpoint final (last) NÃO é o melhor; os "best" reais estão em **5500** (14-dim) e **8000** (8-dim) por `val_action_mse`.

- **O custo do overfit recai sobre a mão, não sobre o braço.** No per-dim final, as dims de braço ficam em val_action_mse ~0,01–0,07 nas duas, mas as dims de mão saturam alto: 6 dims de dedo ≈ 0,404–0,406 (14-dim) e a única dim de squeeze = 0,2436 (8-dim). Como o 14-dim tem 6 dims de mão "ruins" contra 1 no 8-dim, seu `val_action_mse` agregado fica estruturalmente mais alto (final 0,190 vs 0,054) — em boa parte um artefato de *quantas* dims de mão entram na média, não necessariamente de a mão estar pior fisicamente.

- **Os dois datasets são o mesmo conteúdo bruto.** Mesmos 238 episódios / 52 952 frames, mesmo `state[14]`, mesmos sensores (RGB mp4 + depth uint16 PNG + tátil [108]×2), com `videos/` em symlink para o `v3_grasp` comum. A única diferença material é a coluna `action` (8 vs 14 floats), gerada pelo slice determinístico `slice_right_arm_1squeeze.py`. Isso torna a comparação 14 vs 8 uma ablação limpa do *action space*.

- **Caveats de comparabilidade direta.** (1) A run 8-dim **crashou em ~15900/20000** (não completou), então o "last" dela não é comparável ao last completo do 14-dim; o que é comparável são os respectivos *best* por `val_action_mse` (5500 vs 8000). (2) `val_action_mse` entre as duas NÃO é apples-to-apples porque é a média sobre dimensões de action de tamanhos diferentes (14 vs 8) e com proporções diferentes de dims de mão. (3) Não há, em nenhuma das runs, métrica de **success rate** em sim/real no W&B — qualquer conclusão sobre desempenho de tarefa real exige rollout, que não está nos dados auditados aqui.

---

## Mídia (pasta `midia/`)

- `ep_14dim_rgb.png` — head_camera RGB, frame 83 do episódio 0 (14-dim): mão Dex3 fechada segurando o copo branco.
- `ep_14dim_depth.png` — profundidade (colormap turbo, mm), mesmo frame.
- `ep_8dim_rgb.png` — head_camera RGB, frame 83 do episódio 0 (8-dim, via symlink p/ v3_grasp).
- `ep_8dim_depth.png` — profundidade (turbo, mm), mesmo frame.
