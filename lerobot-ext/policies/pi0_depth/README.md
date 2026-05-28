# Manual de Treinamento — Políticas de Imitação com Flow Matching

> Guia completo: treino, validação e deploy.

---

## 1. Visão Geral do Processo

O treinamento de uma política de imitação consiste em ensinar um modelo a reproduzir trajetórias demonstradas por um operador humano. A cada intervalo configurável de steps, o loop de treino pausa para executar uma etapa de validação nos episódios reservados — episódios que o modelo **nunca viu** durante o aprendizado.

```
Treino (steps 1..N)
    │
    ├── a cada eval_freq steps  → Validação
    │       ├── Forward pass sem gradiente nos eps. de validação
    │       ├── Calcula val_loss por batch
    │       ├── Loga métricas (WandB, CSV, etc.)
    │       └── Se val_loss < melhor até agora → salva best_val_checkpoint
    │
    └── a cada save_freq steps  → Checkpoint periódico
```

> **Regra de ouro:** sempre use o `best_val_checkpoint` para deploy no ambiente real, nunca o checkpoint periódico mais recente.

---

## 2. Divisão dos Datasets

Os episódios coletados devem ser divididos em dois conjuntos independentes:

| Conjunto | Papel | Proporção recomendada |
|---|---|---|
| `dataset` (treino) | O modelo vê e aprende. Gradientes são calculados sobre esses episódios. | ~80% dos episódios |
| `val_dataset` (validação) | O modelo **nunca vê** durante o treino. Serve como prova de generalização. | ~20% dos episódios |

Exemplo de configuração YAML:

```yaml
dataset:
  episodes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # 17 eps

val_dataset:
  episodes: [17, 18, 19, 20]  # 4 eps — nunca vistos no treino
```

> **Por que separar?** Se o modelo fosse validado nos mesmos episódios de treino, poderia estar apenas memorizando trajetórias. Os episódios de validação revelam se o modelo realmente generalizou.

---

## 3. O Loop de Validação

A cada `eval_freq` steps, o código executa o seguinte ciclo:

```python
if is_eval_step and val_dataloader is not None:

    policy.eval()          # Desliga dropout — modo inferência

    with torch.no_grad():  # Sem cálculo de gradiente
        for val_batch in val_dataloader:
            val_batch = preprocessor(val_batch)  # Normaliza, tokeniza
            val_loss, output = policy.forward(val_batch)
            val_loss_meter.update(val_loss.item())

    policy.train()         # Volta para modo treino
```

Pontos importantes:

- `torch.no_grad()` — nenhum gradiente é calculado; mais rápido e sem custo extra de memória.
- `policy.eval()` → `policy.train()` — garante comportamento idêntico ao da inferência real no ambiente de deploy.
- O `preprocessor` é o **mesmo** do treino — as mesmas normalizações são aplicadas em ambos os conjuntos.

---

## 4. O que é o val_loss

O `val_loss` é o **MSE (Mean Squared Error) do Flow Matching** calculado sobre os episódios de validação. O Flow Matching aprende um campo vetorial que transforma ruído em ação de forma progressiva.

### Como o loss é calculado

| Passo | Operação | Dimensão típica |
|---|---|---|
| 1 | Pega a ação real do dataset: `actions` | `[B, chunk_size, action_dim]` |
| 2 | Gera ruído aleatório: `noise` | `[B, chunk_size, action_dim]` |
| 3 | Mistura com timestep t: `x_t = t × noise + (1−t) × actions` | `[B, chunk_size, action_dim]` |
| 4 | Modelo prediz o campo: `v_t_pred = modelo(x_t, t, imagem, estado)` | `[B, chunk_size, action_dim]` |
| 5 | Campo real: `v_t_real = noise − actions` | `[B, chunk_size, action_dim]` |
| 6 | `Loss = MSE(v_t_pred, v_t_real)` — média sobre batch, chunk e dims | escalar |

### Interpretação dos valores

| val_loss | Significado | Fase típica |
|---|---|---|
| > 2.0 | Modelo ainda aleatório ou início do treino | Steps iniciais (< 5% do total) |
| 1.0 – 2.0 | Convergindo — aprendizado ativo | Primeira metade do treino |
| 0.4 – 1.0 | Bom aprendizado — generalização razoável | Metade final do treino |
| < 0.4 | Excelente generalização para o número de demos | Fim do treino / mais dados |
| `train_loss` << `val_loss` (fator > 5×) | Overfitting — use `best_val_checkpoint` | Qualquer momento |

> A diferença entre `train_loss` e `val_loss` é natural. Um fator de até 5× é aceitável. Acima disso, o modelo provavelmente memorizou as demonstrações de treino sem generalizar.

---

## 5. loss_per_dim — Diagnóstico por Dimensão

Além do `val_loss` global, o código loga o loss separado por cada dimensão da ação. Isso permite identificar exatamente qual junta ou grau de liberdade está dificultando o aprendizado.

```python
loss_per_dim = losses.mean(dim=[0, 1]).detach().cpu()
loss_dict = {
    f"loss_per_dim/{i}": v.item()
    for i, v in enumerate(loss_per_dim)
}
```

### Como interpretar

- **Loss muito alto (> 2×) em relação às outras dimensões** — aquele grau de liberdade tem alta variância entre demonstrações ou é geometricamente sensível à posição inicial.
- **Loss baixo (< 0.1)** — movimentos simples, repetitivos ou de pequena amplitude que o modelo aprendeu facilmente.
- **Todas as dimensões com loss similar e alto** — o problema é global: possivelmente dados insuficientes ou learning rate inadequado.

### Metas por tipo de dimensão

| Tipo de dimensão | Meta de loss_per_dim | Observação |
|---|---|---|
| Dedos / preensão | < 0.15 | Movimentos simples e repetitivos — fácil de aprender |
| Punho (pitch, roll) | < 1.0 | Moderado — sensível à posição da peça |
| Ombro (pitch, roll) | < 1.0 | Difícil — sensível à posição inicial do robô |
| Rotações axiais (yaw) | < 1.5 | O mais difícil — pequenas variações entre demos geram loss alto |

---

## 6. Best-Val Checkpoint

O `BestValTracker` monitora o `val_loss` a cada validação e salva automaticamente quando um novo mínimo é atingido:

```python
class BestValTracker:
    def update(self, val_loss, step):
        if val_loss >= self.best_val_loss:
            return False   # Não melhorou, não salva
        self.best_val_loss = val_loss
        save_checkpoint(output_dir / "best_val_checkpoint", ...)
        return True
```

### Estrutura de saída

```
train_output/<nome_do_experimento>/
├── best_val_checkpoint/        ← USE ESTE para deploy
│   ├── pretrained_model/
│   │   └── model.safetensors   ← pesos do melhor step
│   └── best_val_meta.txt       ← best_val_loss e best_step
│
└── checkpoints/                ← checkpoints periódicos (não usar)
```

---

## 7. Evolução Típica do Treino

| Step / % | val_loss típico | O que está acontecendo |
|---|---|---|
| 5% | 1.5 – 3.0 | Início — modelo ainda aleatório, queda rápida esperada |
| 10% | 0.8 – 1.5 | Convergência acelerada — maior ganho por step |
| 30–40% | 0.4 – 0.8 | Primeiro patamar — modelo aprendeu os padrões principais |
| 50–80% | plateau | Estabilização — cosine decay do LR quebra o plateau nos últimos steps |
| 100% | mínimo do treino | Melhor generalização — salvo como `best_val_checkpoint` |

> O cosine decay do learning rate (parâmetro `scheduler_decay_steps`) costuma ser decisivo para superar plateaus na segunda metade do treino.

---

## 8. Configurações que Afetam a Validação

| Parâmetro YAML | Efeito | Valor típico |
|---|---|---|
| `eval_freq` | Frequência de validação em steps | 100 – 500 |
| `val_dataset.episodes` | Episódios reservados para validação | ~20% do total |
| `save_best_checkpoint` | Salva automaticamente no melhor val_loss | `true` |
| `scheduler_decay_steps` | Steps para cosine decay do LR | 80–90% do total de steps |
| `training_steps` | Total de steps de treino | Depende do dataset |

---

## 9. Monitoramento das Métricas

Com WandB ou outro logger ativado, as seguintes curvas devem ser acompanhadas:

### Curvas principais

- **`val_loss` vs `loss` (treino):** ambos devem cair juntos no início. Se o `val_loss` para enquanto o `train_loss` continua caindo → overfitting.
- **`loss_per_dim/N`:** monitorar as dimensões com loss mais alto. Se não caem após mais dados, o problema é de consistência nas demonstrações.
- **`grad_norm`:** gradiente explodindo (> 50 repetidamente) indica necessidade de reduzir o learning rate.

---

## 10. Sinais de Alerta

| Sinal no log | Significado | Ação recomendada |
|---|---|---|
| `val_loss` nunca cai abaixo de 1.0 | Modelo não está aprendendo | Verificar LR, arquitetura e preprocessor |
| `val_loss` sobe após um recorde | Overfitting — usar `best_val_checkpoint` | Parar treino ou coletar mais dados |
| `dim/N` > 2.0 no final | Dataset insuficiente para aquela dimensão | Gravar mais demos com posição consistente |
| 🏆 para de aparecer | Plateau — modelo convergiu para o limite dos dados | Normal após ~60–80% dos steps com poucos demos |
| `grad_norm` > 50 repetidamente | Gradiente explodindo | Reduzir `optimizer_lr` |
| `train_loss` / `val_loss` > 5× | Overfitting forte | Reduzir steps ou coletar mais dados |

---

## 11. Recomendações para Melhorar o Desempenho

### Se o val_loss está alto globalmente

- Coletar mais demonstrações, especialmente em posições iniciais variadas.
- Verificar se o preprocessor (normalização) está sendo aplicado corretamente tanto no treino quanto na validação.
- Checar se os episódios de validação são representativos da variedade do dataset.

### Se o val_loss está alto em dimensões específicas

- Regravar demonstrações com maior atenção à consistência do movimento naquelas dimensões.
- Verificar se há ruído ou inconsistência no sensor daquela junta.
- Considerar aumentar o `chunk_size` para dar mais contexto temporal ao modelo.

### Se train_loss e val_loss divergem muito (overfitting)

- Aumentar o número de episódios de validação (meta: 20% do total).
- Adicionar mais demonstrações de treino.
- Reduzir `training_steps` ou confiar no `best_val_checkpoint` como critério de parada.

> Com poucos demos (< 20 episódios), algum overfitting é esperado e aceitável. O `best_val_checkpoint` mitiga isso ao garantir que os pesos usados no deploy sejam os que melhor generalizaram.

---

## 12. Referência Completa do YAML de Configuração

Esta seção descreve cada campo disponível no arquivo de configuração de treinamento.

---

### 12.1 Dataset de Treinamento (`dataset`)

```yaml
dataset:
  repo_id: <usuario>/<nome-do-dataset>   # ID do dataset no HuggingFace Hub
  root: <caminho/local>                  # Caminho local onde o dataset está salvo
  episodes: [0, 1, 2, ...]              # Lista de episódios usados no treino
```

| Campo | Tipo | Descrição |
|---|---|---|
| `repo_id` | string | Identificador do dataset no HuggingFace Hub (`usuario/nome`). Usado para download automático se o dataset não estiver em `root`. |
| `root` | string | Caminho local onde o dataset está ou será salvo. Se o diretório não existir, o sistema faz download a partir de `repo_id`. |
| `episodes` | lista de int | Índices dos episódios usados **exclusivamente** no treino. O modelo calcula gradientes sobre esses episódios. Recomendado: ~80% do total. |

---

### 12.2 Dataset de Validação (`val_dataset`)

```yaml
val_dataset:
  repo_id: <usuario>/<nome-do-dataset>
  root: <caminho/local>
  episodes: [17, 18, 19, 20]
```

Mesmos campos que `dataset`. Os episódios listados aqui **nunca** são vistos durante o treino — servem exclusivamente para medir generalização.

> Recomendado: ~20% do total de episódios. Com menos de 3 episódios de validação, o `val_loss` pode ser ruidoso demais para ser confiável.

---

### 12.3 Política (`policy`)

#### Identificação e repositório

| Campo | Tipo | Descrição |
|---|---|---|
| `type` | string | Tipo da política. Deve corresponder a um tipo registrado no sistema (ex: `pi05depth`, `actdepth`). |
| `repo_id` | string | Repositório HuggingFace de onde os pesos base são carregados para fine-tuning. |
| `push_to_hub` | bool | Se `true`, faz upload do checkpoint final para o `repo_id` ao término do treino. |

#### Modelos base (backbone)

| Campo | Tipo | Descrição |
|---|---|---|
| `paligemma_variant` | string | Variante do modelo de linguagem/visão principal. Ex: `"gemma_2b"`. Controla tamanho e capacidade do VLM. |
| `action_expert_variant` | string | Variante do modelo especialista em ações. Ex: `"gemma_300m"`. Modelo menor que processa estado e gera ações. |
| `image_resolution` | `[H, W]` | Resolução para a qual as imagens são redimensionadas antes de entrar no VLM. Ex: `[224, 224]`. |
| `max_state_dim` | int | Dimensão máxima do vetor de estado. Deve ser igual ao número de juntas/sensores no vetor de observação. |
| `max_action_dim` | int | Dimensão máxima do vetor de ação. Deve ser igual ao número de juntas controladas. |

#### Entradas opcionais

| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `use_depth_3d` | bool | `false` | Ativa o uso de imagem de profundidade como entrada adicional. Requer que `observation.images.<nome>_depth` esteja em `input_features`. |
| `use_pressure` | bool | `false` | Ativa o uso de sensores de pressão/tato como entrada adicional. Requer que `observation.<lado>_hand_pressure` esteja em `input_features`. |

> `use_depth_3d` e `use_pressure` devem ser consistentes com os campos comentados/descomentados em `input_features`. O sistema valida isso no `__post_init__`.

#### Scene Uncertainty Gate

| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `scene_uncertainty_threshold` | float | `0.0` | Limiar de incerteza para o gate. Se `0.0`, o gate está desligado. Valores típicos: `0.05` (conservador) a `0.10` (permissivo). |
| `n_samples_uncertainty` | int | `1` | Número de denoising passes com ruídos diferentes para estimar a incerteza. Auto-ajustado para `3` se `threshold > 0`. Aumentar para `5` dá estimativas mais precisas com custo maior. |

**Como funciona o gate (Flow Matching):** diferente de arquiteturas VAE (que têm `log_sigma` grátis), o Flow Matching estima incerteza rodando `n_samples_uncertainty` passes com ruídos iniciais diferentes. O prefixo VLM é computado **uma vez** com KV-cache; apenas o modelo expert roda N vezes. Com `n=3` o custo extra é moderado.

```
threshold = 0.0  → gate desligado (baseline, sem overhead)
threshold = 0.10 → ativa retorno à posição neutra em alta incerteza
threshold = 0.05 → mais conservador, retorna com mais frequência
```

#### Features de entrada (`input_features`)

Cada entrada é declarada com `type` e `shape`:

```yaml
input_features:
  observation.images.head_camera:
    type: VISUAL
    shape: [3, 480, 640]       # [canais, altura, largura]

  observation.state:
    type: STATE
    shape: [28]                # número de dimensões do estado

  # Descomente para ativar depth (junto com use_depth_3d: true):
  # observation.images.head_camera_depth:
  #   type: VISUAL
  #   shape: [3, 480, 640]

  # Descomente para ativar pressão (junto com use_pressure: true):
  # observation.left_hand_pressure:
  #   type: STATE
  #   shape: [33]
  # observation.right_hand_pressure:
  #   type: STATE
  #   shape: [33]
```

| Tipo | Quando usar |
|---|---|
| `VISUAL` | Imagens RGB ou depth (tensores `[C, H, W]`) |
| `STATE` | Vetores numéricos: posições de juntas, sensores de pressão, etc. |

#### Features de saída (`output_features`)

```yaml
output_features:
  action:
    type: ACTION
    shape: [28]    # deve ser igual a max_action_dim
```

#### Temporalidade

| Campo | Tipo | Descrição |
|---|---|---|
| `chunk_size` | int | Tamanho do chunk de ações previsto por inferência. O modelo prediz `chunk_size` passos de ação de uma vez. Valores maiores dão mais contexto temporal mas aumentam o custo. |
| `n_action_steps` | int | Quantos passos do chunk são efetivamente executados no robô antes de uma nova inferência. Geralmente igual a `chunk_size`. |
| `n_obs_steps` | int | Quantos frames de observação são passados como contexto. `1` = só o frame atual. |

#### Hiperparâmetros do otimizador e scheduler

| Campo | Tipo | Descrição |
|---|---|---|
| `optimizer_lr` | float | Learning rate do otimizador AdamW. Valores típicos: `1e-5` a `1e-4`. Reduzir se `grad_norm` explodir. |
| `optimizer_weight_decay` | float | Penalidade L2 nos pesos. Ajuda a evitar overfitting. Típico: `0.01`. |
| `scheduler_warmup_steps` | int | Steps de warmup linear do LR (de 0 até `optimizer_lr`). Típico: 2–5% do total de steps. |
| `scheduler_decay_steps` | int | Steps totais para o cosine decay do LR. Deve ser próximo do total de `steps` de treino para o decay completar no fim. |

#### Fine-tuning seletivo

| Campo | Tipo | Padrão | Descrição |
|---|---|---|---|
| `train_expert_only` | bool | `false` | Se `true`, congela o VLM principal (PaliGemma) e treina apenas o modelo expert de ações. Recomendado para datasets pequenos — reduz drasticamente o risco de overfitting e acelera o treino. |
| `freeze_vision_encoder` | bool | `false` | Se `true`, congela o encoder de visão (ViT). Use junto com `train_expert_only: true` para máximo controle de quais partes são treinadas. |

> Com poucos demos (< 50 episódios), use `train_expert_only: true` e `freeze_vision_encoder: true`. O VLM já tem capacidade visual pré-treinada; o que precisa aprender é o mapeamento visão→ação específico da tarefa.

---

### 12.4 Configurações de Treinamento

| Campo | Tipo | Descrição |
|---|---|---|
| `output_dir` | string | Diretório onde checkpoints, logs e o `best_val_checkpoint` serão salvos. |
| `steps` | int | Número total de steps de gradiente. Um "step" = um batch processado com atualização de pesos. |
| `batch_size` | int | Número de amostras por step. Valores maiores são mais estáveis mas exigem mais VRAM. |
| `num_workers` | int | Workers do DataLoader para carregamento paralelo de dados. `0` = carregamento no processo principal (mais seguro em ambientes com multiprocessing instável). |
| `log_freq` | int | Frequência (em steps) com que métricas de treino são logadas. |
| `save_freq` | int | Frequência (em steps) para salvar checkpoints periódicos. |
| `save_checkpoint` | bool | Se `false`, desativa os checkpoints periódicos (só `best_val_checkpoint` é salvo). |
| `save_best_checkpoint` | bool | Se `true`, ativa o `BestValTracker` — salva automaticamente sempre que `val_loss` bater recorde. **Recomendado: `true` para qualquer treino de produção.** |
| `neutral_position_loss_weight` | float | Peso do loss de curriculum de posição neutra. `0.0` = desligado. Valores entre `0.1` e `0.3` ensinam o modelo a retornar à posição neutra quando incerto. Requer que `neutral_position` esteja configurado. |

---

### 12.5 Avaliação e Logs

| Campo | Tipo | Descrição |
|---|---|---|
| `eval_freq` | int | Frequência (em steps) para rodar a validação completa no `val_dataset`. |
| `wandb.enable` | bool | Ativa integração com Weights & Biases para logging em tempo real. |
| `wandb.project` | string | Nome do projeto no WandB onde as runs serão agrupadas. |

---

### 12.6 Diferença do Uncertainty Gate: Flow Matching vs VAE

| Aspecto | VAE (ex: ACT) | Flow Matching (ex: PI05) |
|---|---|---|
| **Fonte da incerteza** | `log_sigma` do encoder VAE | Variância entre N denoising passes |
| **Custo** | Zero — já computado no forward pass | N × forward pass do modelo expert |
| **Otimização** | N/A | Prefixo VLM computado 1× com KV-cache |
| **Parâmetro** | `scene_uncertainty_threshold` | `scene_uncertainty_threshold` + `n_samples_uncertainty` |
| **Recomendação** | Pode ativar desde o início | Ativar só após treino básico convergir |