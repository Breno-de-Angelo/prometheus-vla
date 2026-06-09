# Treino PI05 Base — Braço + Mão Direita (14 dims)

Documentação do pipeline de treinamento do modelo PI05 para a tarefa de pegar o copo
com o Unitree G1 + Dex3, usando apenas o lado direito do robô.

---

## Visão Geral

```
Teleop (28 dims) → slice_right_arm_only.py → Dataset 14 dims → run_train.py → Checkpoint PI05
```

O dataset gravado pela teleoperação tem 28 dimensões (braço e mão de ambos os lados).
Como o treino inicial usa apenas o lado direito, um pré-processamento fatia as dimensões
relevantes antes do treino. Isso evita que o modelo desperdice capacidade aprendendo a
outputar zeros para o lado esquerdo, e elimina divisão por zero na normalização QUANTILES
das dims congeladas.

---

## Dataset

### Origem e localização

| Máquina | Caminho |
|---------|---------|
| Laptop (coleta) | `lerobot-ext/datasets/G1_Dex3_depth_tactil_dataset/<timestamp>` |
| Laptop (treino) | `lerobot-ext/datasets/G1_Dex3_right14_dataset/<timestamp>` |
| Atena (treino) | `~/Prometheus/Luiz/prometheus-vla/lerobot-ext/datasets/G1_Dex3_right14_dataset/<timestamp>` |

### Dataset atual (sanity check)

- **Timestamp:** `20260608_205432`
- **Episódios totais:** 15 (ep0–ep14)
- **Frames totais:** 3027 @ 30 FPS
- **Split:** 12 treino / 3 validação

| Split | Episódios | Frames | Duração total |
|-------|-----------|--------|---------------|
| Treino | ep0–ep11 | 2527 | ~84s |
| Validação | ep12–ep14 | 500 | ~17s |

**Duração por episódio:**

| Episódio | Frames | Duração |
|----------|--------|---------|
| ep00 | 143 | 4.8s |
| ep01 | 269 | 9.0s |
| ep02 | 176 | 5.9s |
| ep03 | 226 | 7.5s |
| ep04 | 194 | 6.5s |
| ep05 | 262 | 8.7s |
| ep06 | 194 | 6.5s |
| ep07 | 179 | 6.0s |
| ep08 | 289 | 9.6s |
| ep09 | 190 | 6.3s |
| ep10 | 234 | 7.8s |
| ep11 | 171 | 5.7s |
| ep12 (val) | 226 | 7.5s |
| ep13 (val) | 131 | 4.4s |
| ep14 (val) | 143 | 4.8s |

### Features

| Feature | Shape | Tipo | Normalização |
|---------|-------|------|-------------|
| `observation.images.head_camera` | [480, 848, 3] | RGB video | IDENTITY |
| `observation.state` | [14] → pad [32] | float32 | QUANTILES |
| `action` | [14] → pad [32] | float32 | QUANTILES |

> O depth (`observation.images.head_camera_depth`) existe no dataset mas **não entra no modelo**
> neste treino (`depth_fusion: false`). Os sensores táteis também não são usados como input.

### Espaço de ação — 14 dimensões

| Idx | Junta | Range (rad) | Obs |
|-----|-------|-------------|-----|
| 0 | `kRightShoulderPitch` | [-0.630, +0.449] | |
| 1 | `kRightShoulderRoll` | [-0.370, +0.103] | |
| 2 | `kRightShoulderYaw` | [-0.558, +0.321] | |
| 3 | `kRightElbow` | [-0.977, +1.045] | maior range |
| 4 | `kRightWristRoll` | [-0.180, +0.285] | |
| 5 | `kRightWristPitch` | [-0.690, +0.573] | |
| 6 | `kRightWristYaw` | [-0.383, +0.430] | |
| 7 | `right_hand_thumb_0` | [0.000, 0.000] | **congelado** (range=0) |
| 8 | `right_hand_thumb_1` | [-0.920, 0.000] | |
| 9 | `right_hand_thumb_2` | [-1.740, 0.000] | maior range da mão |
| 10 | `right_hand_index_0` | [0.000, +1.570] | |
| 11 | `right_hand_index_1` | [0.000, +1.740] | |
| 12 | `right_hand_middle_0` | [0.000, +1.570] | |
| 13 | `right_hand_middle_1` | [0.000, +1.740] | |

---

## Pré-processamento

### Script: `lerobot-ext/tools/slice_right_arm_only.py`

Fatia o dataset 28-dim para 14-dim. Deve ser rodado após cada sessão de coleta, antes do treino.

```bash
python lerobot-ext/tools/slice_right_arm_only.py \
    datasets/G1_Dex3_depth_tactil_dataset/<timestamp>
# saída automática em: datasets/G1_Dex3_right14_dataset/<timestamp>
```

**O que o script faz:**
- Fatia `action` e `observation.state` para os índices [7-13, 21-27] do espaço 28-dim
- Atualiza `meta/info.json` (shape e names)
- Atualiza `meta/stats.json` (min/max/mean/std/quantis por dim)
- Atualiza `meta/episodes/` (stats por episódio)
- Cria `videos/` como symlink para o dataset original (sem duplicar dados)

### Script: `lerobot-ext/tools/lerobot_to_omniview.py`

Converte o dataset para visualização no OmniView (RGB, depth, trajetória, tátil).
Pode ser rodado no dataset original 28-dim ou no right14.

```bash
python lerobot-ext/tools/lerobot_to_omniview.py \
    datasets/G1_Dex3_depth_tactil_dataset/<timestamp>
omniview --data datasets/G1_Dex3_depth_tactil_dataset/<timestamp>
```

---

## Arquitetura do Modelo

**PI05 Base** — Vision-Language-Action com flow matching.

```
Entrada:
  Imagem RGB (480×848) ──► SigLIP (PaliGemma) ──► tokens visuais
  Task text              ──► Gemma 2B (VLM)    ──► tokens de contexto  [CONGELADO]
  State (14→32 dims)     ──┐
                           ├──► Gemma 300M (Action Expert) ──► chunk de 50 ações
  State + ações passadas ──┘                                    [TREINADO]

Saída:
  50 ações futuras (14 dims cada) — horizonte de ~1.67s @ 30 FPS
```

**O que é treinado:**
- Action Expert (Gemma 300M) — aprende o mapeamento visão+estado → ação
- Vision encoder (SigLIP) — fine-tuned (`freeze_vision_encoder: false`)

**O que é congelado:**
- VLM (Gemma 2B) — `train_expert_only: true` — evita catastrofic forgetting com poucos dados

**Pesos base:** `lerobot/pi05_base` — 14GB, já disponível em `~/.cache/huggingface/hub/` na Atena.

---

## Augmentações de Imagem

Aplicadas **apenas ao RGB** em tempo de treino. A cada step, **até 2 de 3** transforms são
sorteados proporcionalmente ao peso:

| Transform | Peso | Frequência ~| Parâmetros |
|-----------|------|-------------|-----------|
| **ColorJitter** | 1.0 | ~40% | brilho ×[0.8,1.2] · contraste ×[0.8,1.2] · sat ×[0.5,1.5] · hue ±0.05 |
| **RandomResizedCrop** | 1.0 | ~40% | corta 75–100% da área, mantém aspect ratio ±10%, redimensiona para 480×848 |
| **RandomErasing** | 0.5 | ~20% | apaga retângulo aleatório (2–10% da área) com preto |

> **Não há augmentação no depth** — `depth_fusion: false` neste treino, depth não entra no modelo.
> Na validação, augmentações são desativadas automaticamente pelo LeRobot.

---

## Hiperparâmetros de Treino

| Parâmetro | Valor atual (sanity check) | Para 300+ eps |
|-----------|---------------------------|---------------|
| Steps | 5000 | 30k–50k |
| Batch size | 32 | 32–64 |
| LR inicial | 1e-4 | 1e-4 |
| LR final | 1e-5 | 1e-5 |
| Warmup steps | 500 | 2000–3000 |
| Decay steps | 4500 | 27k–47k |
| Scheduler | Cosine decay | Cosine decay |
| Chunk size | 50 ações | 50 ações |
| Horizonte | ~1.67s | ~1.67s |
| Checkpoints | a cada 500 steps | a cada 2000 steps |
| dtype | bfloat16 | bfloat16 |
| Gradient checkpointing | true | true |

---

## Infraestrutura de Treino

| Item | Valor |
|------|-------|
| Máquina | Atena (`hercules@10.9.8.252`) |
| GPU | A100 80GB — **CUDA_VISIBLE_DEVICES=1** |
| Env conda | `ms3` |
| Repo | `~/Prometheus/Luiz/prometheus-vla/` (branch `Luiz-pi05d`) |
| Saída | `train_output/cup_pi05_right14_lf/` |
| WandB projeto | `prometheus_g1` |
| WandB entity | `prometheus-lcad` |
| Config | `lerobot-ext/config/train/train_cup_pi05_right14.yaml` |

---

## Como Rodar

### 1. Coletar novos episódios (laptop)

```bash
# gravar — gera dataset 28-dim em G1_Dex3_depth_tactil_dataset/<timestamp>
lerobot-record ...
```

### 2. Pré-processar para 14 dims (laptop)

```bash
python lerobot-ext/tools/slice_right_arm_only.py \
    lerobot-ext/datasets/G1_Dex3_depth_tactil_dataset/<timestamp>
```

### 3. Sincronizar para a Atena (laptop)

```bash
# criar diretório se necessário
ssh hercules@10.9.8.252 "mkdir -p ~/Prometheus/Luiz/prometheus-vla/lerobot-ext/datasets/G1_Dex3_right14_dataset"

# sincronizar parquets e meta
rsync -av lerobot-ext/datasets/G1_Dex3_right14_dataset/ \
  hercules@10.9.8.252:~/Prometheus/Luiz/prometheus-vla/lerobot-ext/datasets/G1_Dex3_right14_dataset/

# sincronizar vídeos (o symlink não funciona na Atena — enviar os arquivos reais)
ssh hercules@10.9.8.252 "rm -f ~/Prometheus/Luiz/prometheus-vla/lerobot-ext/datasets/G1_Dex3_right14_dataset/<timestamp>/videos"
rsync -av lerobot-ext/datasets/G1_Dex3_depth_tactil_dataset/<timestamp>/videos/ \
  hercules@10.9.8.252:~/Prometheus/Luiz/prometheus-vla/lerobot-ext/datasets/G1_Dex3_right14_dataset/<timestamp>/videos/
```

### 4. Atualizar a config (se novo timestamp)

Em `lerobot-ext/config/train/train_cup_pi05_right14.yaml`, atualizar:
```yaml
dataset:
  root: lerobot-ext/datasets/G1_Dex3_right14_dataset/<novo_timestamp>
  episodes: [0, 1, ..., N-3]   # todos exceto os 3 últimos

val_dataset:
  root: lerobot-ext/datasets/G1_Dex3_right14_dataset/<novo_timestamp>
  episodes: [N-2, N-1, N]      # últimos 3 como validação
```

### 5. Disparar o treino (laptop → Atena via SSH)

```bash
cat ~/.config/lf_wandb.key | ssh hercules@10.9.8.252 'IFS= read -r K; \
  export WANDB_API_KEY="$K"; \
  cd ~/Prometheus/Luiz/prometheus-vla && \
  conda activate ms3 && \
  CUDA_VISIBLE_DEVICES=1 python -m train.run_train \
    --config lerobot-ext/config/train/train_cup_pi05_right14.yaml \
    > train/log/cup_pi05_right14_lf.log 2>&1 &'
```

### 6. Monitorar

```bash
# log local
ssh hercules@10.9.8.252 "tail -f ~/Prometheus/Luiz/prometheus-vla/train/log/cup_pi05_right14_lf.log"

# WandB: projeto prometheus_g1, entity prometheus-lcad, run com sufixo _lf
```

---

## Roadmap para Dataset Robusto (300–600 eps)

| Fase | Episódios | O que variar |
|------|-----------|-------------|
| Atual (sanity check) | 15 | posição fixa, 1 cena |
| Fase 2 | ~50 | depurar thumb_0 congelado; confirmar qualidade |
| Fase 3 | ~300 | 5 posições do copo × 2 iluminações × ~30 tentativas |
| Fase 4 | ~500 | incluir ~20% de falhas (copo cai, garra não fecha) |

Quando tiver 300+ eps, ajustar na config:
- `steps: 30000–50000`
- `scheduler_warmup_steps: 2000`
- `scheduler_decay_steps: 28000–48000`
- `batch_size: 64` (se VRAM permitir)
- Split: ~85% treino / 15% validação
