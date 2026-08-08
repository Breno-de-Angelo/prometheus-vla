# Treinar a primeira tarefa, e depois acrescentar a segunda

> Roteiro linear, do dataset que já existe até uma rede que obedece a mais de um
> comando. Para a arquitetura, ver [OPENVLA_DEPTH.md](OPENVLA_DEPTH.md); para as
> regras de gravação e merge, [DATASETS_MULTITAREFA.md](DATASETS_MULTITAREFA.md).

---

## O dataset que temos hoje

`meu_dataset/pick_up_the_cup_2026-06-09`, na atena:

| | |
|---|---|
| tarefa | `pick up the white mug and place it to the right` |
| episódios | 36 (15.535 frames, 30 fps) |
| ação / estado | 28 dims — 14 juntas de braço + 14 de mão Dex3 |
| câmeras | `head_camera` e `head_camera_depth`, 848×480 |
| pressão | presente no schema, mas **zerada** — o sensor não gravou |

> A string de `task` foi corrigida: era `"pick up the cup"`, que descrevia só
> metade da demonstração. O modelo aprende a associação entre o texto e a ação
> inteira, então descrever pela metade ensina a coisa errada. O backup dos
> metadados antigos está em `meta_backup_rename/`.
>
> A pasta ainda se chama `pick_up_the_cup_2026-06-09`. Renomear a pasta é opcional
> (é só um caminho), mas se fizer, atualize `root` nos YAMLs.

---

## Fase 1 — Treinar esta tarefa

### 1.1 Antes de rodar, confira a GPU

Na atena as três A100 são compartilhadas. Já aconteceu de a GPU 0 estar ocupada
por um `train.py` de outra pessoa com 46 GB:

```bash
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
```

Escolha uma GPU com folga e fixe:

```bash
CUDA_VISIBLE_DEVICES=0 python init_lerobot_train_v3.py \
    --config_path=train/config/openvla_depth_cup_atena.yaml
```

Sem `CUDA_VISIBLE_DEVICES` o accelerate tenta usar as três e derruba o job de
outra pessoa — ou o seu.

### 1.2 Orçamento medido

| | |
|---|---|
| s/step (batch 16, LoRA r=32, grad ckpt) | **2,08 s** |
| 12.000 steps ≈ 15 épocas | ~6h56 |
| 12 validações | ~24 min |
| **total** | **~7h25** |

Para um resultado em ~4h40, `--steps=8000` (~10 épocas).

### 1.3 O que o log deve mostrar

```
[OpenVLA] 982 tensores carregados de 3 shard(s).
[OpenVLA] 'vision.featurizer': 48 chaves renomeadas por diferença de versão do timm
[OpenVLA] Backbone pronto — visão 2176d → projector → LLM 4096d (32 camadas).
[OpenVLA-D] Torre visual congelada.
[OpenVLA-D] LoRA r=32 aplicado ao LLM — 80.0M treináveis de 6688M (1.20%).
```

As 48 chaves renomeadas são esperadas (`LayerScale` mudou de nome entre timm 0.9
e 1.0) — não é erro.

Você também vai ver, toda vez:

```
[AVISO] O dataset tem apenas 1 tarefa distinta...
```

Correto e esperado nesta fase. Vira erro fatal só quando `require_multitask: true`.

### 1.4 O que acompanhar

- **`val_loss`** com `save_best_checkpoint` ligado. Quando parar de cair, pode
  encerrar com Ctrl+C — o motor salva o estado antes de sair.
- **`loss_per_dim/0..27`** — o mapa de juntas está em
  [`../policies/pi0_depth/README.md`](../policies/pi0_depth/README.md), seção 6.
  As dims 2, 4 e 5 (yaw do ombro e roll/pitch do punho esquerdos) são as difíceis;
  os dedos (14–21) costumam zerar cedo.
- **`grad_norm`** entre ~1 e ~5. Muito acima disso repetidamente, baixe o
  `optimizer_lr`.

> O `val_loss` do OpenVLA-Depth é **L1**; o do PI05-D é o MSE do flow matching.
> Os números não são comparáveis em valor absoluto — compare tendências e o
> `loss_per_dim` relativo entre juntas.

### 1.5 Testar

```bash
python init_lerobot_inference_v3.py \
    --checkpoint=train/output/openvla_depth_cup_2026-06-09/best_val_checkpoint/pretrained_model \
    --task="pick up the white mug and place it to the right" \
    --sim --v
```

Detalhe: ver [INFERENCIA_COMANDO_TEXTO.md](INFERENCIA_COMANDO_TEXTO.md).

### 1.6 O que esta fase entrega — e o que não entrega

**Entrega:** validação da arquitetura ponta a ponta, um baseline de `val_loss`
comparável com o PI05-D no mesmo robô, e o tempo real por step para dimensionar
os próximos treinos.

**Não entrega:** comportamento multi-tarefa. Com uma tarefa só, o prompt é
constante e o modelo aprende que o texto é irrelevante. Trocar o comando na
inferência não vai mudar nada — e isso é uma propriedade do dataset, não um bug.

---

## Fase 2 — Acrescentar a segunda tarefa

### 2.1 O erro que parece óbvio e não funciona

```
checkpoint da fase 1  →  fine-tune na tarefa 2  →  multi-tarefa?
```

**Não.** Isso produz um modelo que faz a tarefa 2 e esqueceu a tarefa 1.

O condicionamento por linguagem só é aprendido quando **dois comandos competem
dentro do mesmo batch**: duas amostras com imagem parecida, ações diferentes, e o
texto como única coisa que as distingue. No treino sequencial o prompt é constante
em cada fase, então esse conflito nunca acontece — e a fase 2 sobrescreve os pesos
da fase 1.

O raciocínio completo está em
[DATASETS_MULTITAREFA.md](DATASETS_MULTITAREFA.md), seção 4.5.

### 2.2 O caminho que funciona

```
1. Gravar a tarefa 2 como um dataset SEPARADO, com a sua própria string de `task`
2. Juntar os dois datasets em disco
3. Treinar do zero (do backbone do OpenVLA) sobre o dataset unificado
```

O passo 3 recomeça do `openvla/openvla-7b`, não do checkpoint da fase 1. Soa
desperdício, mas as ~7 h de treino são o preço de um condicionamento por
linguagem que realmente funciona.

### 2.3 Antes de gravar: congele o schema

Esta é a decisão irreversível. `lerobot/datasets/aggregate.py:47` exige que
`fps`, `robot_type` e o dicionário de `features` sejam **idênticos** entre os
datasets a juntar. Se a tarefa 2 for gravada com o yaw do tronco (29 juntas) ou
uma câmera a mais, ela **não se junta** com as 36 demos de hoje.

Então decida agora, antes da primeira demo da tarefa 2:

- vai incluir o yaw do tronco? (28 → 29 dims)
- vai acrescentar câmera?
- vai mudar resolução?

Se sim para qualquer uma, a tarefa 1 também precisa ser regravada ou migrada com
`modify_features`. Se não, grave a tarefa 2 exatamente com o schema atual.

### 2.4 Gravar a tarefa 2

Um dataset separado, string de `task` própria, em inglês, descrevendo a ação
inteira:

```
meu_dataset/place_mug_on_coffee_stand_2026-06-25/
    task: "place the white mug on the coffee stand"
```

30–40 demos, e **balanceadas** com a tarefa 1: 36 demos de uma e 5 da outra
produz um modelo que faz sempre a primeira.

Confira depois de gravar:

```bash
python -c "import pandas as pd; print(pd.read_parquet('meu_dataset/<NOVA>/meta/tasks.parquet'))"
```

Se a string vier errada, conserte sem regravar:

```bash
python train/rename_dataset_task.py --dataset meu_dataset/<NOVA> \
    --from "<errada>" --to "<certa>" --dry-run
```

### 2.5 Juntar

```bash
python train/build_multitask_dataset.py \
    --datasets meu_dataset/pick_up_the_cup_2026-06-09 \
               meu_dataset/place_mug_on_coffee_stand_2026-06-25 \
    --output-repo-id local/g1_dex3_multitask \
    --output-dir meu_dataset/g1_dex3_multitask \
    --dry-run
```

O dry-run valida schema, lista as tarefas e estima disco sem escrever nada. Se
passar, rode de novo sem `--dry-run`.

### 2.6 Treinar multi-tarefa

Aponte `train/config/openvla_depth_multitask.yaml` para o dataset unificado e
deixe `require_multitask: true` — assim o treino aborta se o merge tiver saído
com uma tarefa só.

Escale os steps: com 2 tarefas, o modelo precisa ver cada comando tantas vezes
quanto veria numa corrida de tarefa única. ~20.000 steps a 2,08 s ≈ 11 h.

E coloque episódios de **cada** tarefa no `val_dataset` — validação só com
episódios de uma tarefa mede a coisa errada.

### 2.7 O teste que fecha o ciclo

Mesma cena inicial, dois comandos, trajetórias diferentes. Procedimento em
[INFERENCIA_COMANDO_TEXTO.md](INFERENCIA_COMANDO_TEXTO.md), seção 4.

Se as trajetórias saírem iguais, o `val_loss` bom não significa nada — o modelo
está ignorando o texto.

---

## Resumo

```
Fase 1  ──  1 tarefa, 36 demos
            ↳ valida arquitetura, dá baseline vs PI05
            ↳ NÃO produz multi-tarefa

Fase 2  ──  congelar schema
            ↳ gravar tarefa 2 (30-40 demos, balanceada)
            ↳ merge
            ↳ treinar DO ZERO sobre o unificado
            ↳ testar troca de comando
```

Se o objetivo principal é o multi-tarefa, o caminho mais curto não é esperar a
fase 1 terminar — é começar a gravar a tarefa 2, depois de congelar o schema.
