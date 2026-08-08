# OpenVLA-Depth — VLA multi-tarefa com profundidade para o G1 + Dex3

> **O que esta política resolve:** uma única rede neural que recebe o comando em
> texto junto com a visão RGB, o mapa de profundidade e o tato, e executa a ação
> correspondente àquele comando. Trocar `"pick up the cup"` por
> `"place the cup on the coffee stand"` muda o comportamento **sem trocar de
> checkpoint e sem cabeça de saída por tarefa**.

---

## 1. Antes de mais nada: você já tinha metade disso

Vale registrar, porque muda o que é urgente fazer.

O **`pi05depth` já é condicionado por linguagem**. O prompt é montado em
`policies/pi0_depth/processor_pi05.py`:

```python
full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
```

e `cleaned_text` vem do campo `task` de cada frame do dataset LeRobot. Ou seja: o
PI05-D já sabe receber comandos diferentes.

**Só que tem um porém na branch `lerobot-nn`.** O campo `override_task`
(`configuration_pi05.py`, `processor_pi05.py`) força um texto fixo:

```python
current_task = self.override_task if self.override_task is not None else task
```

Com `override_task` preenchido no YAML, **todos os episódios veem o mesmo
prompt** — o modelo aprende a ignorar o texto, e o condicionamento por linguagem
morre. Se você testou multi-tarefa no PI05 e "não funcionou", olhe esse campo
primeiro.

O `actdepth`, esse sim, não tem linguagem nenhuma. Não há como torná-lo
multi-tarefa sem reescrever o encoder.

| | `actdepth` | `pi05depth` | `openvladepth` |
|---|---|---|---|
| Backbone | ResNet18 | PaliGemma 3B | Prismatic / OpenVLA **7B** |
| Visão | RGB | SigLIP | **DINOv2 + SigLIP** |
| Linguagem | ✗ nenhuma | ✓ prompt PaliGemma | ✓ prompt Llama-2 |
| Pré-treino robótico | ✗ | ✓ (π-datasets) | ✓ **970k eps Open X-Embodiment** |
| Decodificação | transformer decoder | flow matching (10 passos) | **OFT paralelo (1 forward)** |
| Profundidade | token PointNet | token PointNet | token PointNet/PointTransformer |
| Tato Dex3 | ✓ | ✓ | ✓ |
| Multi-tarefa | ✗ | ✓ | ✓ |

O ganho real do OpenVLA sobre o PI05 aqui é o **pré-treino em 970k episódios
reais** e um LLM maior — o que costuma se traduzir em generalização melhor para
comandos e objetos fora da distribuição de treino. Não é o condicionamento por
linguagem em si, que o PI05 já tinha.

---

## 2. Arquitetura

### 2.1 Layout da sequência

```
[BOS] [patches RGB ×256] [depth] [pressão] [estado] [prompt ×L] [queries ×C]
 └──────────────────── prefixo (contexto) ─────────────────┘   └─── saída ───┘
```

- **patches RGB** — a mesma imagem passa por DINOv2-L e SigLIP-so400m; as
  features são concatenadas (1024 + 1152 = 2176) e projetadas para 4096.
- **depth** — o mapa de profundidade vira nuvem de pontos (projeção pinhole) e
  passa pelo `PointTransformerEncoder` que você já usa no ACT-D. Sai 1 token.
- **pressão** — os 66 sensores do Dex3 (33 por mão) viram 1 token via MLP.
- **estado** — os 28 ângulos viram 1 token contínuo. Diferente do `pi05depth`,
  que discretiza o estado em 256 bins e escreve tudo no prompt como texto.
- **prompt** — o texto da tarefa, tokenizado com o tokenizer do Llama-2.
- **queries** — `chunk_size` embeddings aprendidos. É deles que sai a ação.

### 2.2 O head de ação (OFT), e por que ele é obrigatório aqui

O OpenVLA original gera ações **autoregressivamente**: cada dimensão vira 1 token
discretizado em 256 bins, mapeado nos últimos 256 tokens do vocabulário do Llama.
Funciona bem no braço de 7 DoF do paper.

No G1 + Dex3 são **28 dimensões**, e vocês usam `chunk_size=50`:

```
28 dims × 50 passos = 1400 tokens autoregressivos por chunk
```

Isso é inviável no robô real — seriam 1400 forwards de um modelo de 7B por chunk.

A solução (OpenVLA-OFT, *Optimized Fine-Tuning*) é anexar `C = chunk_size`
*action queries* ao final da sequência e ler os hidden states dessas posições:

```
prompt + imagens + depth → LLM → hidden[-C:]  →  MLP  →  [B, 50, 28]
                                                          1 forward
```

O custo cai de 1400 forwards para 1. O preço é abandonar os pesos do head de ação
pré-treinado do OpenVLA — mas eles seriam inúteis de qualquer forma, porque o
espaço de ação do G1 (28 juntas absolutas) não tem nada a ver com o do OpenVLA
(7 deltas de end-effector + gripper). O que se aproveita, e é o que importa, é o
**VLM inteiro**: visão, linguagem e o senso comum robótico dos 970k episódios.

### 2.3 Atenção bidirecional nas queries

As action queries usam atenção **bidirecional entre si**, mas continuam causais
em relação ao prefixo:

```
        prefixo    queries
prefixo  causal      ✗        ← prefixo nunca vê as queries
queries   tudo    bidirecional ← query do passo 3 vê a do passo 47
```

É isso que dá coerência temporal ao chunk inteiro num único forward. Sem
bidirecionalidade, a query do passo 3 não saberia para onde o braço vai no passo
47, e o chunk sai "míope".

Implementado com uma máscara 4D pré-construída
(`OpenVLADepthModel._build_attention_mask_4d`). O `transformers` 5.x aceita isso:
`masking_utils.create_causal_mask` devolve intacta qualquer máscara que já venha
em 4D. Desligável com `bidirectional_action_attn: false`.

---

## 3. O mecanismo multi-tarefa — leia esta seção inteira

Todo o multi-tarefa está em **uma única linha** de
`policies/openvla_depth/processor_openvla.py`:

```python
prompt = self.prompt_template.format(task=cleaned.lower())
```

que produz o prompt canônico do OpenVLA:

```
In: What action should the robot take to pick up the cup?
Out:
```

O `task` vem do campo homônimo de cada frame do dataset LeRobot. **Nada mais no
modelo sabe qual tarefa está sendo executada** — não há head por tarefa, nem
one-hot, nem índice.

### 3.1 A consequência prática

> Se todos os episódios do dataset tiverem a **mesma** string de `task`, o modelo
> aprende que o texto é irrelevante e passa a ignorá-lo. Na inferência, mudar o
> comando não muda nada. O treino "converge" normalmente e o problema só aparece
> quando você testa o segundo comando.

Por isso o motor de treino checa isso antes de começar
(`policies/openvla_depth/run_train.py::check_multitask`) e imprime as tarefas
encontradas. Com `require_multitask: true` no YAML, ele **aborta** em vez de
avisar.

### 3.2 Como preparar o dataset

Ao gravar com `init_lerobot_record.py`, dê a cada sessão o texto da sua tarefa,
em vez de reaproveitar um rótulo genérico:

```
sessão 1 → task = "pick up the cup"
sessão 2 → task = "place the cup on the coffee stand"
sessão 3 → task = "hand the cup to the person"
```

Recomendações que fazem diferença:

- **Frases imperativas, em inglês, minúsculas.** O prompt do OpenVLA foi
  pré-treinado nesse formato (`take to {task}?`). Português funciona pior porque
  o pré-treino robótico foi todo em inglês.
- **Balanceie os episódios.** 30 demos de "pick up" e 3 de "place" produzem um
  modelo que faz "pick up" para qualquer comando.
- **Varie a redação da mesma tarefa** se puder (`"pick up the cup"` /
  `"grab the cup"`). Ensina o modelo a atender ao *significado*, não a decorar
  uma string.
- **Coloque episódios de cada tarefa na validação.** Um `val_dataset` só com
  episódios de "pick up" não mede generalização entre comandos.

Se as tarefas estiverem em datasets separados, junte antes com
`train/build_cotraining_dataset.py`.

---

## 4. Arquivos

### Criados

```
policies/openvla_depth/
├── __init__.py                   registro do tipo `openvladepth`
├── configuration_openvla.py      OPENVLADEPTHConfig — todos os knobs do YAML
├── backbone.py                   carregamento do Prismatic/OpenVLA-7B + CLI de diagnóstico
├── modeling_openvla.py           OPENVLADEPTHPolicy + head OFT + máscara 4D
├── processor_openvla.py          prompt multi-tarefa + normalização + tokenização
└── run_train.py                  checagem de multi-tarefa + motor compartilhado

train/config/
├── openvla_depth.yaml            baseline de tarefa única (pareável com pi05_depth_cup3)
└── openvla_depth_multitask.yaml  o config que exercita o multi-tarefa

docs/
└── OPENVLA_DEPTH.md              este arquivo
```

### Modificados

| Arquivo | Mudança |
|---|---|
| `policies/__init__.py` | registra `OPENVLADEPTHConfig` junto com ACT e PI05 |
| `init_lerobot_train_v3.py` | despacha `type: openvladepth` para o motor novo |

### Reaproveitados sem alteração

- `policies/act_depth/depth_encoder.py` — `build_depth_encoder` e
  `depth_to_pointcloud`. O mesmo Point Transformer do ACT-D, então as ablations
  de profundidade continuam comparáveis entre as três políticas.
- `policies/pi0_depth/run_train.py` — o loop de treino. Aquele arquivo já é
  genérico (resolve tudo por `make_policy` / `make_pre_post_processors`) e traz o
  split de validação, o `BestValTracker`, o shutdown limpo no Ctrl+C, o
  `ColorJitter` só em RGB e o reset das stats de profundidade. Duplicar 950
  linhas para trocar duas importações seria pior.

### Como o LeRobot acha tudo isso

Por convenção de nomes, sem registro manual:

```
@PreTrainedConfig.register_subclass("openvladepth")   →  OPENVLADEPTHConfig
OPENVLADEPTHConfig            →  OPENVLADEPTHPolicy   (troca Config por Policy)
configuration_openvla.py      →  modeling_openvla.py  (troca o prefixo do módulo)
configuration_openvla.py      →  processor_openvla.py + make_openvladepth_pre_post_processors
```

Se você renomear qualquer uma dessas peças, o `make_policy` para de encontrar a
política. As funções responsáveis são `_get_policy_cls_from_policy_name` e
`_make_processors_from_policy_config`, em `lerobot/policies/factory.py`.

---

## 5. Ambiente

### 5.1 Dependências adicionais

```bash
pip install --no-deps 'timm>=1.0.0,<1.1.0' 'peft>=0.13.0'
```

- `timm` — as duas torres ViT (DINOv2 e SigLIP).
- `peft` — LoRA. Sem ele, use `use_lora: false` + `train_new_modules_only: true`.

**Use `--no-deps`.** Ambos declaram dependências amplas (`huggingface_hub`,
`safetensors`, `torch`, `torchvision`, `accelerate`, `transformers`) que já estão
instaladas e pinadas neste ambiente. Sem `--no-deps` o pip pode reinstalar
qualquer uma delas numa versão que quebra o resto da stack — e o sintoma costuma
aparecer longe da causa.

Depois de instalar, confira que nada essencial se moveu:

```bash
python -c "
import torch, transformers, accelerate, numpy, huggingface_hub, safetensors
print(torch.__version__, transformers.__version__, accelerate.__version__,
      numpy.__version__, huggingface_hub.__version__, safetensors.__version__)"
```

Referência do que estava na atena **antes** de instalar timm/peft, e que precisa
continuar assim: `torch 2.10.0+cu128`, `transformers 5.6.1`, `accelerate 1.13.0`,
`numpy 1.26.4`, `huggingface_hub 1.11.0`, `safetensors 0.7.0`, `datasets 4.1.0`.

> O aviso `lerobot 0.4.4 requires huggingface-hub<0.36.0 but you have 1.11.0`
> **já existia antes** de qualquer instalação nova. Não foi o timm/peft que
> causou, e a stack funciona assim — não tente "consertar" fazendo downgrade do
> `huggingface_hub`, isso sim quebraria o `datasets` e o LeRobot.

### 5.2 Por que não usamos `trust_remote_code`

O código remoto publicado com o `openvla/openvla-7b` foi escrito para
`transformers==4.40.1` e `timm==0.9.10`. Este projeto roda **transformers 5.6.1**
(exigido pelo LeRobot), e as duas coisas não coexistem no mesmo venv.

Por isso `backbone.py` remonta a arquitetura a partir de três peças estáveis —
dois ViTs do timm, um MLP projector e o `LlamaModel` do transformers — e carrega
os pesos direto do safetensors. **As dimensões de cada peça são inferidas do
próprio state dict**, não hardcoded.

Se ainda assim você quiser o caminho oficial, crie um venv separado com as
versões antigas e use `load_mode: remote_code`. Não é o caminho recomendado.

### 5.3 Detalhe que quebra silenciosamente

O Prismatic **não usa a saída final do ViT**: ele pega os patches da *penúltima*
camada (`n = len(blocks) - 2`), sem LayerNorm final. `FusedVisionBackbone`
reproduz isso. Usar a saída final desalinha as features em relação ao projector
pré-treinado e o modelo sai lixo **sem levantar erro nenhum** — o loss simplesmente
não desce direito. Se você mexer nessa parte, é o primeiro lugar para olhar.

### 5.4 Conferir o checkpoint antes de treinar

```bash
python -m policies.openvla_depth.backbone --inspect openvla/openvla-7b --compare
```

Duas coisas, sem tocar na GPU:

- **`--inspect`** imprime os prefixos de chave reais e as formas. Compare com os
  `PREFIX_*` no topo de `backbone.py`.
- **`--compare`** vai além: constrói localmente as torres timm, o projector e o
  Llama (este em `meta device`, sem alocar os 7B) e faz o diff das chaves,
  componente por componente. Sai com código 1 se algo diverge.

Rode isso **antes do primeiro treino**. O motivo concreto: o `openvla-7b` foi
salvo com `timm==0.9.10` e aqui roda `timm 1.0.28`. Se os nomes de parâmetro
mudaram entre as versões, `--compare` mostra exatamente qual chave, em vez de
você descobrir no meio da inicialização do treino.

O carregamento é **estrito de propósito** (`_load_strict`): uma chave faltando
levanta exceção em vez de deixar um bloco com pesos aleatórios passar
despercebido — um erro que não daria erro nenhum, só um loss que não desce.

### 5.5 O rename do LayerScale (já resolvido)

O `--compare` no checkpoint real apontou uma divergência, e vale registrar porque
é o tipo de coisa que reaparece a cada bump do timm:

```
─── DINOv2 — DIVERGE ───
    módulo local: 343 chaves | checkpoint: 343 chaves
    faltando  (48): ['blocks.0.ls1.gamma',        'blocks.0.ls2.gamma', ...]
    sobrando  (48): ['blocks.0.ls1.scale_factor', 'blocks.0.ls2.scale_factor', ...]
```

Mesma contagem, mesmo sufixo, mesmas formas: é **rename puro** do parâmetro do
`LayerScale`. O checkpoint usa `scale_factor` (o export para HF evita o nome
`gamma`, que o `transformers` renomeia automaticamente); o timm 1.0.x usa `gamma`.

Resolvido por `_KEY_ALIASES` em `backbone.py`, sem precisar de venv com timm
antigo. O alias só é aplicado **se o destino existir no módulo local com a mesma
forma** — sem essa checagem, um alias errado plugaria o tensor errado, e um
modelo com pesos trocados não dá erro, só não converge.

SigLIP, projector e language_model bateram exatos de primeira. O projector saiu
`2176 → 8704 → 4096 → 4096`, e o `lm_head` (32064×4096) é ignorado de propósito:
o head OFT lê hidden states, não logits de vocabulário.

Se um bump futuro do timm trouxer outro rename 1:1, adicione o par em
`_KEY_ALIASES`. Se as **formas** mudarem, aí sim é caso de reexportar os pesos
num venv com `timm==0.9.10`.

---

## 6. Treinar

```bash
# tarefa única — comparável com pi05_depth_cup3
python init_lerobot_train_v3.py --config_path=train/config/openvla_depth.yaml

# multi-tarefa — o caso de uso real
python init_lerobot_train_v3.py --config_path=train/config/openvla_depth_multitask.yaml
```

### 6.1 Na atena, com o dataset local

Config pronto: `train/config/openvla_depth_cup_atena.yaml`
(dataset `meu_dataset/pick_up_the_cup_2026-06-09`).

```bash
source ~/miniconda3/bin/activate g1
cd ~/DEV/prometheus-vla/lerobot-ext

# uma vez: as duas dependências que faltam no env g1 (--no-deps preserva os pins)
pip install --no-deps 'timm>=1.0.0,<1.1.0' 'peft>=0.13.0'

# confira o checkpoint ANTES de gastar horas de GPU (sai com código 1 se diverge)
python -m policies.openvla_depth.backbone --inspect openvla/openvla-7b --compare

# smoke test — meça o tempo por step antes do run longo
CUDA_VISIBLE_DEVICES=0 python init_lerobot_train_v3.py \
    --config_path=train/config/openvla_depth_cup_atena.yaml \
    --steps=30 --eval_freq=15 --batch_size=4 --wandb.enable=false

# treino
CUDA_VISIBLE_DEVICES=0 python init_lerobot_train_v3.py \
    --config_path=train/config/openvla_depth_cup_atena.yaml
```

> **`CUDA_VISIBLE_DEVICES=0` não é opcional.** Na atena a GPU 1 está com um
> `VLLM::EngineCore` (74 GB) e a GPU 2 com `ollama` (19 GB). Só a GPU 0 está
> livre. Sem essa variável o accelerate tenta usar as três e o job morre — ou,
> pior, derruba o serviço de outra pessoa.

Dataset local exige `root` no YAML, além de `repo_id`:

```yaml
dataset:
  repo_id: "local/pick_up_the_cup_2026-06-09"   # só um rótulo
  root: "meu_dataset/pick_up_the_cup_2026-06-09" # onde os arquivos estão
```

Sem `root`, o LeRobot tenta baixar `repo_id` do Hub e falha.

### Dimensionamento (A100/H100)

| VRAM | LoRA | batch | gradient checkpointing |
|---|---|---|---|
| 80 GB | r=32–64 | 16 | ligado |
| 40 GB | r=32 | 8 | ligado |
| 24 GB | r=16 | 1–2 | ligado, e ainda é apertado |

> Em 8 GB (o notebook RTX 5070) o OpenVLA-7B **não treina**, nem com QLoRA. Para
> inferência local, veja a seção 9.

### Learning rates

Dois grupos separados, montados em `OPENVLADEPTHPolicy.get_optim_params`:

| grupo | módulos | LR padrão | por quê |
|---|---|---|---|
| backbone | adaptadores LoRA sobre o LLM | `5e-4` | valor do paper OpenVLA |
| novos | depth encoder, tato, estado, queries, head | `1e-3` | começam do zero, precisam andar mais rápido |

---

## 7. O que acompanhar no WandB

O `loss_dict` mantém o formato do PI05-D, então seus painéis continuam valendo:

- `loss` / `val_loss` — L1 médio sobre as 28 dimensões (o padrão do OFT).
  **Não é comparável em valor absoluto com o `val_loss` do PI05-D**, que é o MSE
  do flow matching. Compare tendências, não números.
- `loss_per_dim/0..27` — o mesmo mapa de juntas documentado em
  `policies/pi0_depth/README.md` seção 6. As dims 0/2/4/5 (ombro e punho
  esquerdos) continuam sendo as difíceis.
- `l1_loss` — L1 explícito, para comparar entre `loss_type` diferentes.

**No multi-tarefa, o sinal mais importante não está no loss agregado.** Um
`val_loss` bom pode esconder um modelo que faz bem a tarefa majoritária e ignora
as outras. Rode a validação por tarefa, ou no mínimo confira na inferência que
comandos diferentes produzem trajetórias diferentes a partir da mesma observação
inicial — é o teste decisivo do condicionamento por linguagem.

---

## 8. Uncertainty gate

Mesma ideia do ACT-D e do PI05-D, mas o mecanismo mudou porque o head OFT é
determinístico (não há ruído de denoising para variar).

Aqui a incerteza vem de **MC-dropout no head**: o LLM roda **uma vez** — é ele
que domina o custo — e só o MLP do head roda N vezes com dropout ativo sobre o
mesmo hidden state. A dispersão entre as amostras é o proxy de incerteza.

```yaml
policy:
  scene_uncertainty_threshold: 0.10   # 0.0 desliga (padrão)
  # n_samples_uncertainty e action_head_dropout são ligados automaticamente
```

Acima do limiar, a ação é misturada com o buffer `neutral_position` na mesma
proporção usada pelas outras duas políticas — as ablations seguem comparáveis.

---

## 8.5 Achados no dataset `pick_up_the_cup_2026-06-09`

Três coisas apareceram ao inspecionar o dataset na atena. As duas primeiras
**afetam também o `pi05depth` e o `actdepth`**, não só a política nova.

### 8.5.1 Os intrínsecos da câmera estão errados nos configs existentes

O dataset foi gravado a **848×480** (`Scripts_Prometheus_int/full_realsenser_server.py`),
com o depth alinhado ao stream de cor. Mas todos os configs usam:

```yaml
camera_intrinsics: {fx: 600.0, fy: 600.0, cx: 320.0, cy: 240.0}
```

`cx: 320` é o centro de uma imagem de **640** px de largura. Para 848 px o ponto
principal fica em ≈ **424**. São ~104 px de erro, e `depth_to_pointcloud` usa
exatamente isso:

```python
x = (grid_x - cx) * z / fx
```

O efeito é um cisalhamento da nuvem de pontos que cresce com a distância: o
objeto aparece deslocado em X proporcionalmente a quão longe está. O encoder 3D
aprende a compensar dentro do dataset, mas o erro reaparece assim que a cena
muda — o tipo de problema que só se manifesta no robô.

Os valores no config novo (`fx=fy=617, cx=424, cy=240`) vêm do FOV nominal do
D435i a 848×480, e são uma estimativa. Para os reais, rode **no robô**:

```bash
python Scripts_Prometheus_int/print_camera_intrinsics.py
```

### 8.5.2 Os sensores táteis não gravaram

`observation.left_hand_pressure` e `observation.right_hand_pressure` são
**identicamente zero** nos 36 episódios — `min = max = mean = std = 0` nos 33
sensores de cada mão. As colunas existem no parquet, mas não há sinal.

Não quebra nada: com quantis idênticos o normalizador do LeRobot troca o
denominador por `eps=1e-8` (`normalize_processor.py:370`), e o vetor vira uma
constante `-1.0`. Mas o token de pressão passa a carregar uma constante, o que
gasta contexto e dá a falsa impressão de que a fusão tátil está ativa.

Por isso `use_pressure: false` no config da atena. Volte para `true` quando
gravar com o sensor funcionando.

### 8.5.3 O dataset tem uma tarefa só

36 episódios, 15.535 frames, **1 tarefa**: `"pick up the cup"`. O treino roda e
converge, mas não produz comportamento multi-tarefa — é um baseline de
arquitetura, não a entrega final. A seção 3 explica por quê.

---

## 9. Limitações conhecidas

1. **7B é grande.** Inferência em bf16 ocupa ~15 GB. No notebook de 8 GB só roda
   quantizado em 4-bit (~5 GB), e devagar. O servidor de inferência assíncrono
   (`init_lerobot_inference_server.py`) é o caminho: modelo na `hercules`, robô
   consumindo o chunk pela rede.

2. **O head de ação começa do zero.** Diferente do PI05-D, que aproveita o
   expert de ação pré-treinado. Espere precisar de mais steps até o loss
   estabilizar — os primeiros ~500 steps são o head aprendendo a existir.

3. **Prompt em inglês.** O pré-treino robótico do OpenVLA é todo em inglês;
   comandos em português degradam bastante.

4. **`bidirectional_action_attn` depende do repasse de máscara 4D** do
   transformers (`masking_utils.py`, "if the mask is already 4D, return as-is").
   Está verificado na 5.6.1. Se uma atualização futura quebrar isso, o sintoma é
   um erro de shape no attention — e `bidirectional_action_attn: false` é a saída
   imediata.

5. **O carregamento nativo não foi validado contra o checkpoint real.** A
   estrutura de chaves (`vision_backbone.featurizer.*`, `projector.fc*`,
   `language_model.model.*`) segue o Prismatic, mas o download de 7B não foi
   feito neste ambiente. Rode o `--inspect` da seção 5.4 **antes do primeiro
   treino** — leva um minuto e evita descobrir o problema depois de baixar 14 GB.
   O carregamento estrito garante que um prefixo errado vira exceção, não um
   modelo silenciosamente aleatório.

---

## 10. Validação já feita

Executado neste ambiente, com um `LlamaModel` minúsculo no lugar do 7B:

- ✅ O LeRobot resolve `openvladepth` → config, policy e processador, pela cadeia
  real de `factory.py`.
- ✅ Máscara 4D correta: prefixo causal, bloco de queries totalmente conectado,
  prefixo sem acesso às queries, padding bloqueado como chave.
- ✅ A máscara 4D é aceita pelo `LlamaModel` do transformers 5.6.1.
- ✅ Montagem da sequência: `1 BOS + 16 patches + 3 tokens extras + 11 prompt +
  4 queries = 35` — bate com o esperado.
- ✅ Forward → loss → backward, com gradiente chegando em `action_queries`,
  `action_head`, `depth_encoder`, `pressure_proj` e `state_proj`.
- ✅ Grupos do otimizador separados corretamente (22 params @ 5e-4, 27 @ 1e-3).
- ✅ `predict_action_chunk` → `[B, 50, 28]`; `select_action` → `[B, 28]`.
- ✅ Uncertainty gate por MC-dropout produz dispersão não-nula.
- ✅ Validações do config: erro quando `use_depth_3d` não bate com as features,
  aviso quando `override_task` está setado, auto-ajuste do gate.

E **na atena, contra o dataset real** (`pick_up_the_cup_2026-06-09`), ainda com o
Llama minúsculo no lugar do 7B:

- ✅ `check_multitask` lê o dataset local via `root` e detecta a tarefa única.
- ✅ As features do dataset batem com o que a política espera: RGB e depth
  separados corretamente, `observation.state` (28), ação (28) padded para 32.
- ✅ Batch real do dataloader → preprocessor → forward → backward, com imagens de
  480×848: RGB redimensionado para `[B, 6, 224, 224]`, depth mantido em resolução
  nativa (redimensionar destruiria a projeção pinhole).
- ✅ Tokenizer do `openvla-7b` carrega (LlamaTokenizer, vocab 32001, `<PAD>`),
  BOS na posição 0 — que é a premissa do layout da sequência. O prompt do
  dataset ocupa 18 tokens dos 64 reservados.
- ✅ Pós-processamento desnormaliza a ação de volta à escala das juntas.
- ✅ O YAML `openvla_depth_cup_atena.yaml` exato é aceito, e com
  `use_pressure: false` a política nem constrói o `pressure_proj`.
- ✅ `transformers` na atena é 5.6.1 — a mesma versão em que a máscara 4D foi
  verificada.

**Não validado** (exige o download de 14 GB e a GPU): carregamento real dos pesos
do `openvla-7b`, o caminho do LoRA (`peft` não está instalado em nenhuma das duas
máquinas) e a convergência do treino.
