# Gravar várias tarefas e juntar num dataset multi-tarefa

> Como sair de "um dataset, uma tarefa" para **uma rede só que obedece a comandos
> diferentes** — pegar a xícara, colocar a xícara no coador, pegar a chaleira,
> colocar a chaleira.

---

## 0. Leia isto antes de gravar qualquer coisa

Há uma regra no LeRobot que decide toda a ordem do trabalho. Em
`lerobot/datasets/aggregate.py:47`:

```python
def validate_all_metadata(all_metadata):
    fps = all_metadata[0].fps
    robot_type = all_metadata[0].robot_type
    features = all_metadata[0].features
    for meta in all_metadata:
        if fps != meta.fps:       raise ValueError(...)
        if robot_type != ...:     raise ValueError(...)
        if features != ...:       raise ValueError(...)
```

**Para juntar N datasets, os três têm que ser idênticos: `fps`, `robot_type` e o
dicionário de `features` inteiro** — mesmos nomes de câmera, mesmas resoluções,
mesma dimensão de estado e ação.

E não dá para contornar treinando com vários datasets ao mesmo tempo: neste fork
o `MultiLeRobotDataset` está desligado (`lerobot/datasets/factory.py:115` levanta
`NotImplementedError`). O caminho é **juntar em disco antes** e apontar o YAML
para o dataset unificado.

### A consequência prática

> Se você gravar "pegar a xícara" com 28 juntas e 2 câmeras, e depois decidir
> incluir o yaw do tronco (29 juntas) ou uma terceira câmera, **os datasets
> antigos deixam de ser combináveis com os novos**. Ou você regrava tudo, ou
> migra os antigos com `modify_features`/`add_features`.

Por isso a ordem certa é:

```
1. Congelar o schema  (juntas + câmeras + resoluções + fps)
2. Gravar TODAS as tarefas com esse schema
3. Juntar
4. Treinar
```

Você já mencionou querer mais juntas (yaw do tronco), mais câmeras e resolução
diferente para cobrir o FOV do robô. **Faça essas mudanças agora**, na seção 1,
antes de gravar a primeira demo da primeira tarefa.

---

## 1. Congelar o schema

### 1.1 O schema atual

Do `meta/info.json` de `pick_up_the_cup_2026-06-09`:

| feature | shape | origem |
|---|---|---|
| `action` | (28,) | 14 juntas de braço + 14 de mão |
| `observation.state` | (28,) | idem |
| `observation.images.head_camera` | (480, 848, 3) | RealSense D435i, RGB |
| `observation.images.head_camera_depth` | (480, 848, 3) | idem, depth em cinza triplicado |
| `observation.left_hand_pressure` | (33,) | Dex3 esquerda |
| `observation.right_hand_pressure` | (33,) | Dex3 direita |

`fps = 30`, `robot_type = "unitree_g1_dex3"`.

As 28 dimensões vêm de `G1_29_JointArmIndex` (`robot/unitree_g1/g1_utils.py:40`,
índices 15–28) mais 7 juntas por mão Dex3.

### 1.2 Adicionar o yaw do tronco

A junta existe e já está mapeada — `robot/unitree_g1/g1_utils.py:77`:

```python
class G1_29_JointIndex(IntEnum):
    ...
    kWaistYaw   = 12
    kWaistRoll  = 13
    kWaistPitch = 14
    kLeftShoulderPitch = 15    # ← onde G1_29_JointArmIndex começa hoje
```

O que muda ao incluir `kWaistYaw`:

- `action` e `observation.state` passam de **28 → 29**
- a dimensão 28 (a nova, no fim do vetor) é o yaw do tronco
- **nada muda no OpenVLA-Depth**: `max_action_dim: 32` já faz padding, e a
  dimensão real vem de `output_features` do dataset. 28, 29 ou 32 funcionam sem
  tocar no modelo (ver seção 5)

Onde mexer para gravar/executar essa junta:

1. **Leitura** — o `get_observation` de `UnitreeG1Dex3` monta o estado a partir
   de `G1_29_JointArmIndex`. Incluir o índice 12 no vetor.
2. **Escrita** — o `send_action` precisa comandar o motor 12 junto com os de braço.
3. **Teleoperação** — ver 1.4.

> **Cuidado com a ordem.** Se você colocar o yaw no *começo* do vetor, todos os
> índices de junta mudam e a documentação de `loss_per_dim` (o mapa 0–27 do
> README do PI05) deixa de valer. Coloque no **fim** (índice 28) e o mapa antigo
> continua legível.

### 1.3 Mais câmeras e outra resolução

As câmeras são um dicionário em `robot/unitree_g1/unitree_g1_dex3.py:105`:

```python
self.cameras = {
    "head_camera":       ZMQCameraConfig(..., camera_name="head_camera",       width=848, height=480, ...),
    "head_camera_depth": ZMQCameraConfig(..., camera_name="head_camera_depth", width=848, height=480, ...),
    # "d435i_ir_left":   ZMQCameraConfig(...)   ← já existe comentada
}
```

Adicionar uma câmera é acrescentar uma entrada aqui **e** publicá-la no servidor
ZMQ (`Scripts_Prometheus_int/full_realsenser_server.py`). Resoluções diferentes
por câmera são permitidas — cada uma é uma feature independente.

Duas armadilhas:

- **RGB extra é barato para o modelo, caro para o disco.** Cada câmera RGB vira
  256 tokens no prefixo do LLM (seção 5), então 3 câmeras = 768 tokens só de
  imagem. Em disco, cada câmera é um `.mp4` a mais por episódio.
- **Depth extra ainda não é usado.** Hoje o OpenVLA-Depth consome só a primeira
  câmera de profundidade (`depth_images[0]`), e os intrínsecos são um dicionário
  único no YAML. Ver seção 5 para o que falta.

Sobre "mostrar o FOV todo": aumentar a resolução **não** aumenta o campo de
visão — o FOV é ótica, não pixels. Para ver mais cena você precisa de lente mais
aberta ou de uma segunda câmera apontando para outro lado. Aumentar de 848×480
para 1280×720 só te dá mais detalhe no mesmo enquadramento, e o RGB é
redimensionado para 224×224 antes de entrar na torre visual de qualquer forma
(seção 5) — ou seja, resolução alta de RGB é gasto de disco sem ganho para o
modelo. Para o **depth** é diferente: ele entra em resolução nativa na nuvem de
pontos, então ali a resolução importa de verdade.

### 1.4 Yaw do tronco pelo analógico do VR

O analógico já é lido — `teleop/teleop_hand_and_arm.py:316`:

```python
if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
    loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                      -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                      -tele_data.right_ctrl_thumbstickValue[0] * 0.3)
```

Repare no que isso faz hoje: o analógico direito no eixo X vai para o terceiro
argumento de `Move(vx, vy, omega)` — ou seja, **gira o robô inteiro andando**,
via controlador de locomoção. Não é a junta do tronco.

O que você quer ("ele rotaciona o tronco") é outra coisa: mover `kWaistYaw = 12`
com os pés parados. São dois caminhos diferentes e vale escolher conscientemente:

| | gira o corpo (`Move` omega) | gira o tronco (`kWaistYaw`) |
|---|---|---|
| já implementado | ✓ | ✗ |
| pés | andam / pivotam | parados |
| entra no vetor de ação? | **não** (é comando de locomoção) | **sim** (vira a dim 28) |
| a política aprende? | não — fica fora do dataset | sim |

Só o segundo serve para o robô aprender a girar o tronco como parte da tarefa.
O primeiro seria uma ação do operador que a rede nunca vê, e o resultado é uma
política que se confunde quando a base se move sozinha.

Esboço da mudança, para o analógico direito no eixo X virar posição de tronco:

```python
# integra o analógico numa posição-alvo, com limite de curso
waist_yaw_target += -tele_data.right_ctrl_thumbstickValue[0] * WAIST_YAW_RATE * dt
waist_yaw_target = np.clip(waist_yaw_target, -WAIST_YAW_LIMIT, +WAIST_YAW_LIMIT)
# e mandar junto com os braços, no motor G1_29_JointIndex.kWaistYaw
```

Integrar (posição) em vez de usar o valor cru (velocidade) importa: o dataset
guarda **posições de junta**, e o modelo prevê posições. Um analógico mapeado
direto para velocidade produziria um alvo que não corresponde ao que está gravado.

Confirme os limites reais de curso do `kWaistYaw` no G1 antes de soltar — bater
no fim de curso durante uma demo estraga o episódio e pode danificar o robô.

---

## 2. Gravar cada tarefa

### 2.1 Um dataset por tarefa

Grave **um dataset separado por tarefa**, com um nome que diga a tarefa e a data:

```
meu_dataset/
├── pick_up_the_cup_2026-06-20/
├── place_cup_on_coffee_stand_2026-06-20/
├── pick_up_the_kettle_2026-06-21/
└── place_kettle_on_base_2026-06-21/
```

Por que separado e não tudo num só: você pode regravar/descartar uma tarefa sem
mexer nas outras, e a junção é reversível — o dataset unificado é derivado, os
originais continuam sendo a fonte da verdade.

### 2.2 A string `task` é o que faz o multi-tarefa funcionar

Todo o condicionamento por linguagem sai daqui. Em
`policies/openvla_depth/processor_openvla.py`:

```python
prompt = self.prompt_template.format(task=cleaned.lower())
# → "In: What action should the robot take to pick up the cup?\nOut:"
```

Nada mais no modelo sabe qual tarefa está rodando — não há head por tarefa, nem
índice, nem one-hot. Se as strings não distinguirem as tarefas, o multi-tarefa
não existe.

Convenções que fazem diferença:

- **Inglês, imperativo, minúsculas.** O OpenVLA foi pré-treinado com
  `"What action should the robot take to {task}?"` em inglês. Português degrada
  bastante.
- **Descreva a ação e o objeto**, não o número do experimento:

  | ✓ | ✗ |
  |---|---|
  | `pick up the cup` | `tarefa1` |
  | `place the cup on the coffee stand` | `cup_v2` |
  | `pick up the kettle` | `chaleira` |
  | `place the kettle on its base` | `ep_kettle_final` |

- **Varie a redação da mesma tarefa**, se conseguir: metade das demos com
  `pick up the cup`, metade com `grab the cup`. Ensina o modelo a atender ao
  significado em vez de decorar uma string — é o que faz um comando novo, nunca
  visto, ainda funcionar.
- **Verifique depois de gravar.** O `meta/tasks.parquet` de cada dataset tem que
  conter exatamente a string que você espera:

  ```bash
  python -c "import pandas as pd; print(pd.read_parquet('meu_dataset/<TAREFA>/meta/tasks.parquet'))"
  ```

### 2.3 Quantas demos, e o balanceamento

O balanceamento importa mais do que o total. **30 demos de "pegar" e 3 de
"colocar" produzem um modelo que faz "pegar" para qualquer comando** — ele
aprende que o texto raramente muda a resposta certa.

Ponto de partida razoável, dado que 36 demos numa tarefa já deram sinal no PI05:

| tarefas | demos por tarefa | total |
|---|---|---|
| 2 | 30–40 | 60–80 |
| 4 | 30–40 | 120–160 |

Se uma tarefa for mais difícil (colocar exige precisão maior que pegar), dê mais
demos a ela — mas não menos de ~70% do que a tarefa mais representada.

E **varie a cena**: posição inicial do objeto, iluminação, posição do robô. Com
demos idênticas o modelo decora a trajetória e ignora tanto a imagem quanto o
texto.

### 2.4 Separe os episódios de validação por tarefa

Reserve ~17% de **cada** tarefa para validação, não 17% do total tirados de uma
tarefa só. Um `val_dataset` que só tem episódios de "pegar a xícara" mede
generalização dentro de uma tarefa, não entre comandos — que é exatamente o que
você quer medir aqui.

---

## 3. Juntar os datasets

### 3.1 O script

Use `train/build_multitask_dataset.py` (nesta mesma entrega):

```bash
# SEMPRE dry-run primeiro: valida o schema sem escrever nada
python train/build_multitask_dataset.py \
    --datasets meu_dataset/pick_up_the_cup_2026-06-20 \
               meu_dataset/place_cup_on_coffee_stand_2026-06-20 \
               meu_dataset/pick_up_the_kettle_2026-06-21 \
               meu_dataset/place_kettle_on_base_2026-06-21 \
    --output-repo-id local/g1_dex3_multitask \
    --output-dir meu_dataset/g1_dex3_multitask \
    --dry-run

# depois, sem --dry-run
```

O dry-run compara `fps`, `robot_type` e cada feature entre os datasets e imprime
exatamente qual diverge. É bem mais rápido descobrir ali do que depois de copiar
dezenas de GB.

Por baixo ele chama `merge_datasets` (`lerobot/datasets/dataset_tools.py:235`),
que reindexa episódios, concatena os vídeos e unifica a tabela de tarefas.

### 3.2 Espaço em disco

O merge **copia** — não move nem cria link. Você vai precisar do tamanho somado
dos datasets, livre. Na atena o disco está em 98% (192 GB livres), então confira
antes:

```bash
du -sh meu_dataset/*/ ; df -h .
```

### 3.3 Conferir o resultado

```bash
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata as M
m = M('local/g1_dex3_multitask', root='meu_dataset/g1_dex3_multitask')
print('episódios:', m.total_episodes, '| frames:', m.total_frames, '| fps:', m.fps)
print('tarefas:'); print(m.tasks)
"
```

Você tem que ver **uma linha por tarefa**. Se aparecer uma só, as strings de
`task` estavam iguais nos datasets de origem e o multi-tarefa não vai funcionar —
volte à seção 2.2.

---

## 4. Treinar

Aponte `train/config/openvla_depth_multitask.yaml` para o dataset unificado:

```yaml
dataset:
  repo_id: "local/g1_dex3_multitask"
  root: "meu_dataset/g1_dex3_multitask"

val_dataset:
  repo_id: "local/g1_dex3_multitask"
  root: "meu_dataset/g1_dex3_multitask"
  episodes: [...]        # episódios de CADA tarefa

policy:
  require_multitask: true   # aborta se o dataset tiver 1 tarefa só
```

O motor imprime as tarefas encontradas antes de começar e aborta com
`require_multitask: true` se houver menos de duas.

Escale os `steps`: com N tarefas o modelo precisa ver cada comando tantas vezes
quanto veria numa corrida de tarefa única. Com 4 tarefas e ~2 s/step em batch 16,
20.000 steps dão ~11 h.

### O teste que realmente importa

`val_loss` agregado **esconde** o modo de falha típico do multi-tarefa: um modelo
que faz bem a tarefa majoritária e ignora o resto pode ter val_loss ótimo. O
teste decisivo é:

> A partir da **mesma observação inicial**, mudar só o texto do comando e
> verificar que o chunk de ações previsto é diferente.

Se `pick up the cup` e `place the cup on the coffee stand` produzem trajetórias
parecidas com a mesma imagem de entrada, o modelo está ignorando a linguagem —
independentemente do que o loss diga.

---

## 4.5 "Dá para treinar uma tarefa agora e juntar depois?"

Não — e este é o ponto mais contra-intuitivo do documento, então vale insistir.

Existe o `finetune_from_checkpoint.py` nesta branch, que carrega um checkpoint e
continua o treino num dataset novo. É tentador usá-lo assim:

```
treina em "pegar a xícara"  →  fine-tune em "colocar no coador"  →  multi-tarefa?
```

**Isso não produz multi-tarefa.** Produz um modelo que faz "colocar no coador" e
esqueceu "pegar a xícara".

### O motivo

O condicionamento por linguagem só é aprendido se houver **contraste dentro do
treino**.

Com os datasets juntos, um mesmo batch contém amostras de tarefas diferentes.
Duas delas podem ter a imagem quase idêntica — o robô parado na frente da xícara —
e ações completamente diferentes. A única coisa que as distingue é o texto do
prompt. É esse conflito, dentro de um único passo de gradiente, que força o
modelo a ler o comando.

No treino sequencial isso nunca acontece:

| | prompt visto no batch | o que o modelo conclui |
|---|---|---|
| fase 1 | sempre `pick up the cup` | o texto é constante, ignore |
| fase 2 | sempre `place the cup...` | o texto é constante, ignore |

Em nenhum momento dois comandos competem. O modelo aprende a ignorar o texto nas
duas fases, e a fase 2 sobrescreve os pesos da fase 1 — esquecimento catastrófico
em cima de um condicionamento que nunca chegou a existir.

### Quando o fine-tune incremental É a ferramenta certa

- **Mais demos da mesma tarefa** — acrescentar 20 episódios de "pegar a xícara"
  a um modelo já treinado nela.
- **Adaptação de domínio** — treinou no MuJoCo, ajusta no robô real.
- **Acrescentar a 5ª tarefa a um modelo já multi-tarefa** — e mesmo aí você
  precisa misturar dados das 4 antigas no dataset novo (*rehearsal*), senão elas
  degradam. Na prática, refazer o merge com as 5 costuma sair mais barato que
  calibrar a proporção do rehearsal.

### E o treino de tarefa única que já está rodando?

Serve para outra coisa: validar a arquitetura ponta a ponta e dar um baseline de
`val_loss` comparável com o PI05 no mesmo robô. Não é um passo em direção ao
multi-tarefa — é uma medição paralela. Quando os datasets das outras tarefas
existirem, o treino multi-tarefa começa do backbone pré-treinado do OpenVLA de
novo, não desse checkpoint.

---

## 5. O que já é dinâmico, e o que não é

Você pediu encoder/decoder dinâmicos. Parte já está, parte não — e vale saber
qual é qual antes de planejar.

### Já é dinâmico (nenhuma mudança de código)

| | como funciona |
|---|---|
| **Dimensão de ação/estado** | `max_action_dim: 32` faz padding; a dimensão real vem de `output_features` do dataset. 28, 29 (com yaw do tronco) ou 32 funcionam direto. Acima de 32, é só subir `max_action_dim`/`max_state_dim` no YAML. |
| **Número de câmeras RGB** | `_preprocess_images` percorre `config.image_features`; cada câmera RGB vira 256 tokens no prefixo. Adicionar uma câmera ao dataset é suficiente. |
| **Resolução de RGB** | Qualquer resolução é redimensionada para 224×224 (bicúbico) antes da torre visual. |
| **Chunk de ação** | `chunk_size` define as action queries; mudar o horizonte não mexe na arquitetura. |

### Ainda não é dinâmico

| | limitação | o que fazer |
|---|---|---|
| **Múltiplas câmeras de profundidade** | só a primeira é usada (`depth_images[0]`) | juntar as nuvens de pontos, ou um token por câmera |
| **Intrínsecos por câmera** | `camera_intrinsics` é um dicionário único no YAML | virar um dicionário por chave de câmera |
| **Resolução do depth** | entra em resolução nativa na projeção pinhole — os intrínsecos **têm que bater com essa resolução** | ver o aviso da seção 8.5.1 de `OPENVLA_DEPTH.md` |

As duas primeiras só importam quando você tiver de fato uma segunda câmera de
profundidade. A terceira já importa hoje.

---

## 6. Testar no MuJoCo antes de gastar o robô

O simulador já publica pela **mesma interface ZMQ** do servidor RealSense real —
`unitree-g1-mujoco/run_sim.py`:

```python
camera_list = cameras or ["head_camera", "head_camera_depth"]
camera_configs[cam_name] = {"height": 480, "width": 640}
print(f"📷 Cameras: {', '.join(camera_list)} → ZMQ port {camera_port}")
```

Ou seja: o mesmo `init_lerobot_record_v2.py` grava do simulador sem alteração,
porque o robô lê as câmeras de `zmq://<ip>:5555` nos dois casos.

Use o simulador para validar o **pipeline**, não o comportamento:

- a nova junta de tronco entra no vetor de ação com a dimensão certa?
- a câmera nova aparece no `info.json` com o nome e a resolução esperados?
- a string de `task` chega no `tasks.parquet`?
- o merge de duas tarefas de brinquedo passa no `validate_all_metadata`?

Isso fecha o ciclo em minutos, e é onde você quer descobrir que o schema está
errado — não depois de 40 demos reais.

> **Atenção a uma diferença que morde:** o simulador está configurado em
> **640×480** e o robô real grava em **848×480**. Datasets de sim e do robô real
> **não são combináveis** (features diferentes → `validate_all_metadata` recusa).
> Se você pretende co-treinar sim + real, alinhe as resoluções agora. E lembre
> que os intrínsecos da câmera mudam junto: `cx` é ~320 para 640 de largura e
> ~424 para 848.

---

## 7. Ordem de execução sugerida

```
1. Decidir o schema final                       ← inclui yaw do tronco e câmeras
2. Implementar leitura/escrita do kWaistYaw     ← robot/unitree_g1/
3. Mapear o analógico → posição de tronco       ← teleop/teleop_hand_and_arm.py
4. Gravar 2 episódios de brinquedo no MuJoCo    ← valida o schema
5. Merge de brinquedo + treino de 30 steps      ← valida o pipeline inteiro
6. Gravar as 4 tarefas no robô real             ← 30-40 demos cada, balanceadas
7. Merge de verdade + conferir tasks.parquet
8. Treinar com require_multitask: true
9. Teste de troca de comando (seção 4)
```

Os passos 4 e 5 custam uma tarde e evitam a situação de descobrir, com 160 demos
gravadas, que falta uma feature e o merge recusa.

---

## Referências no código

| o quê | onde |
|---|---|
| schema idêntico exigido no merge | `lerobot/datasets/aggregate.py:47` |
| `MultiLeRobotDataset` desligado | `lerobot/datasets/factory.py:115` |
| API de merge | `lerobot/datasets/dataset_tools.py:235` |
| juntas do G1 (`kWaistYaw = 12`) | `robot/unitree_g1/g1_utils.py:77` |
| juntas de braço usadas hoje (15–28) | `robot/unitree_g1/g1_utils.py:40` |
| dicionário de câmeras | `robot/unitree_g1/unitree_g1_dex3.py:105` |
| analógico do VR | `teleop/teleop_hand_and_arm.py:316` |
| câmeras do simulador | `unitree-g1-mujoco/run_sim.py` |
| prompt multi-tarefa | `policies/openvla_depth/processor_openvla.py` |
| checagem de multi-tarefa no treino | `policies/openvla_depth/run_train.py::check_multitask` |
