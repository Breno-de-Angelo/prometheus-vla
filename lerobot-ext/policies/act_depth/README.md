# ACT-D — Action Chunking Transformer with Depth

Extensão do ACT original ([Zhao et al., 2023](https://arxiv.org/abs/2304.13705)) com percepção geométrica 3D via PointNet, desenvolvida no **Projeto Prometheus** para o robô humanoide **Unitree G1 com mãos Dex3-1**.

---

## Motivação

O ACT padrão processa apenas imagens RGB monoculares, o que cria um gargalo sensorial para tarefas de manipulação de precisão. A visão 2D é ambígua em profundidade — o mesmo pixel pode corresponder a objetos a distâncias muito diferentes. Seres biológicos resolvem isso com estereopsia; nós resolvemos injetando um token de geometria 3D diretamente no encoder do Transformer.

A hipótese central: **um modelo que sabe onde o objeto está no espaço 3D deve generalizar melhor para variações de posição do que um modelo que só sabe como ele aparece em 2D.**

---

## Arquitetura

### Visão geral

```
Entradas
├── RGB [3, 480, 640]         → ResNet18 → 300 tokens [B, 300, dim]
├── Depth [3, 480, 640]       → depth_to_pointcloud → PointNet → 1 token [B, 1, dim]
├── State [28]                → Linear → 1 token [B, 1, dim]
└── Ação GT [100, 28]         → (só no treino) → VAE Encoder → z [B, 32]

ACT Encoder (N camadas, self-attention)
└── [RGB tokens | Depth token | State token | z token]
         ↓ cross-attention
ACT Decoder (1 camada)
└── 100 query tokens → action head → ações [B, 100, 28]
```

### Pipeline de profundidade

O mapa de profundidade gravado pelo sensor Intel RealSense D435i passa por:

```
depth_raw (z16, mm) → clip(0, 2000mm) → /2000 * 255 → uint8 → vídeo MP4
                                                                    ↓
                                                            LeRobot __getitem__
                                                                    ↓
                                                              / 255 → [0, 1]
                                                                    ↓
                                                        depth_to_pointcloud()
                                                          z = tensor * 2.0  (metros)
                                                          projeção pinhole 3D
                                                          amostra 1024 pontos
                                                                    ↓
                                                            PointNetEncoder
                                                                    ↓
                                                        depth token [B, 1, dim_model]
```

**Bug crítico resolvido:** o `factory.py` do LeRobot sobrescreve as stats de todas as `camera_keys` com valores ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) quando `use_imagenet_stats: true`. O depth também é uma `camera_key`, então era normalizado antes de chegar na PointNet, produzindo valores negativos e zerando todos os pontos válidos. A correção é feita no `run_train.py` resetando as stats do depth para identidade (`mean=0, std=1`) logo após a criação do dataset.

### Token de depth como token próprio no Encoder

A contribuição arquitetural central em relação ao ACT original é que o depth **não é somado ao state token** — ele entra como um token separado na sequência do encoder:

```python
# ANTES (errado — depth desaparecia no state):
state_token = state_token + depth_feature

# AGORA (correto — token próprio com positional embedding):
depth_token = encoder_depth_input_proj(depth_feat).unsqueeze(1)   # [B, 1, dim]
depth_pos   = encoder_depth_pos_embed.weight.unsqueeze(0)          # [1, 1, dim]
encoder_tokens = cat([rgb_tokens, depth_token, state_token, z_token], dim=1)
```

Isso permite ao Transformer aprender cross-attention entre o token de geometria 3D e os 300 tokens RGB — a política pode descobrir que "profundidade nesta região da imagem implica esta ação".

### VAE Encoder com depth e pressão

Durante o treino, o VAE Encoder recebe não só a sequência de ações, mas também os tokens de depth e pressão:

```
VAE input = [cls | depth_token | pressure_token | state | action_sequence]
```

Isso enriquece o prior latente z com contexto geométrico e tátil, tornando o espaço latente mais informativo que o VAE do ACT original.

### PointNet simplificado

```python
Conv1d(3, 64, 1) → Conv1d(64, 128, 1) → Conv1d(128, 1024, 1)
→ MaxPool global → FC(1024, 512) → FC(512, dim_model)
```

Não usa T-Net (transformation network) da PointNet original porque a câmera está fixada na cabeça do G1 — a posição relativa câmera-robô é constante entre episódios, eliminando a necessidade de invariância a rotações.

---

## Módulos e arquivos

| Arquivo | Responsabilidade |
|---|---|
| `modeling_act.py` | `ACTPolicy`, `ACT`, `ACTEncoder`, `ACTDecoder`, `ACTBackbone` |
| `configuration_act.py` | `ACTConfig` — todos os hiperparâmetros, registrado como `"actdepth"` |
| `processor_act.py` | Pipeline de pré/pós-processamento; remove depth do normalizador ImageNet |
| `depth_encoder.py` | `PointNetEncoder`, `depth_to_pointcloud` |
| `neutral_position.py` | Utilitário para calcular posição neutra no espaço normalizado |
| `run_train.py` | Loop de treino com early stopping, best-val checkpoint, correção de stats |

---

## Parâmetros de configuração (YAML)

### Módulos ACT-D

```yaml
policy:
  use_depth_3d: true            # liga PointNet + depth token no encoder
  camera_intrinsics:
    fx: 600.0
    fy: 600.0
    cx: 320.0
    cy: 240.0
  pointnet_num_points: 1024     # pontos amostrados por frame

  use_pressure: false           # pressão tátil Dex3 (requer sensores)
  pressure_feature_dim: 66      # 33 left + 33 right
  pressure_hidden_dim: 256
```

### Arquitetura Transformer

```yaml
  dim_model: 256                # dimensão principal dos tokens
  n_heads: 8                    # cabeças de atenção (dim_model / n_heads = 32)
  dim_feedforward: 1024         # camada FF interna (4× dim_model)
  n_encoder_layers: 4
  n_decoder_layers: 1           # fixo — compatibilidade com ACT original
  use_vae: true
  latent_dim: 32
  n_vae_encoder_layers: 3
```

### Anti-overfitting (dataset pequeno)

```yaml
  dropout: 0.4
  kl_weight: 20.0               # alto → força o decoder a usar tokens visuais
  optimizer_weight_decay: 1e-3
```

### Early stopping

```yaml
early_stop_patience: 0          # 0 = desligado; N = para após N evals sem melhora
early_stop_min_delta: 0.001     # melhora mínima de 0.1% para resetar o contador
```

### Scene Uncertainty Gate

```yaml
  scene_uncertainty_threshold: 0.0   # 0 = desligado (recomendado durante testes)
                                      # 0.5 = ativa em incerteza moderada (deploy)
```

---

## Scene Uncertainty Gate

Durante a inferência, o VAE usa `z = zeros` (sem ruído). A incerteza é estimada pelo desvio padrão médio do espaço latente (`log_sigma`). Quando supera o threshold, a ação predita é interpolada em direção à posição neutra:

```python
blend_alpha = clamp((uncertainty - threshold) / threshold, 0, 1)
action_safe = (1 - alpha) * action_pred + alpha * neutral_position
```

> ⚠️ **Em desenvolvimento:** o cálculo automático da `neutral_position` a partir dos `default_positions` do `UnitreeG1Config` está implementado em `neutral_position.py`, mas o mapeamento entre o espaço físico de radianos e o espaço normalizado da ação ainda apresenta outliers (3 juntas > 3σ) que precisam de ajuste fino. Por isso, recomenda-se manter `scene_uncertainty_threshold: 0.0` durante os testes iniciais e ativar apenas com `neutral_position` validada manualmente.

---

## Backbone visual

O **ResNet18** processa a imagem RGB `[3, 480, 640]` e produz um feature map `[512, 15, 20]` via `layer4`. Cada um dos `15 × 20 = 300` patches representa uma região de `32 × 32 pixels` da imagem original.

O backbone treina junto com o resto do modelo (`lr = optimizer_lr_backbone`). As camadas BatchNorm ficam congeladas via `FrozenBatchNorm2d` para estabilidade em batch sizes pequenos.

```yaml
  vision_backbone: resnet18
  pretrained_backbone_weights: ResNet18_Weights.IMAGENET1K_V1
  optimizer_lr_backbone: 1e-5
```

---

## Treinamento recomendado (21 episódios)

```yaml
steps: 100000          # early stopping para antes se configurado
batch_size: 128
eval_freq: 25
save_best_checkpoint: true
neutral_position_loss_weight: 0.0   # curriculum desligado por padrão

policy:
  dim_model: 256
  dropout: 0.4
  kl_weight: 20.0
  dim_feedforward: 1024
  n_encoder_layers: 4
  n_vae_encoder_layers: 3
```

Com 21 episódios e `dim_model: 256`, o modelo tem ~15M de parâmetros treináveis (excluindo ResNet), o que mantém a proporção parâmetros/frames em nível razoável para evitar memorização.

---

## Referências

- **ACT original:** Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
- **3D-CAVLA:** Bhat et al., "3D-CAVLA: Leveraging Depth and 3D Context to Generalize Vision Language Action Models for Unseen Tasks", CVPR Workshops 2025. [arXiv:2505.05800](https://arxiv.org/abs/2505.05800)
- **PointNet:** Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation", CVPR 2017. [arXiv:1612.00593](https://arxiv.org/abs/1612.00593)
- **LeRobot:** Cadene et al., HuggingFace, 2024. [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)

---

## Status e trabalho futuro

| Componente | Status |
|---|---|
| Depth como token próprio no Encoder | ✅ Implementado |
| Correção de normalização ImageNet no depth | ✅ Implementado |
| PointNet simplificado (sem T-Net) | ✅ Implementado |
| Pressão tátil como token próprio | ✅ Implementado (requer hardware) |
| Early stopping com patience e min_delta | ✅ Implementado |
| Best-val checkpoint automático | ✅ Implementado |
| Scene Uncertainty Gate | ✅ Implementado (gate funcional) |
| **Neutral position automática** | 🔧 Em desenvolvimento — mapeamento físico → normalizado com outliers |
| Depth como mapa espacial 2D (alternativa à PointNet) | 📋 Planejado |
| Coordenadas 3D como positional embedding dos tokens RGB | 📋 Planejado |
| Validação quantitativa depth vs. no-depth (N≥50 episódios) | 📋 Pendente coleta |