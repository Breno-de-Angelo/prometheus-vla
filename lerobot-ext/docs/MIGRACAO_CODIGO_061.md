# Migração do código do `lerobot-ext` para a 0.6.1

O que quebrou no nosso código quando o fork subiu para a 0.6.1, o que era a
troca certa, e por quê. Complemento do [INSTALL.md](../../docs/INSTALL.md) (que
cobre o ambiente) e do [PROFUNDIDADE_NATIVA.md](PROFUNDIDADE_NATIVA.md) (que
cobre o formato de profundidade).

Data: 18/08/2026.

---

## 1. Símbolos que mudaram de lugar

| Era | Virou | Onde doía |
|---|---|---|
| `lerobot.datasets.utils.hw_to_dataset_features` | `lerobot.utils.feature_utils.hw_to_dataset_features` | `init_lerobot_record_v2.py` |
| `lerobot.datasets.utils.build_dataset_frame` | `lerobot.utils.feature_utils.build_dataset_frame` | idem |
| `lerobot.datasets.utils.cycle` | `lerobot.utils.utils.cycle` | os 3 `run_train.py` |
| `lerobot.utils.train_utils` | `lerobot.common.train_utils` | os 3 `run_train.py` |
| `lerobot.rl.wandb_utils` | `lerobot.common.wandb_utils` | os 3 `run_train.py` |
| `lerobot.utils.rabc.RABCWeights` | `lerobot.rewards.sarm.rabc.RABCWeights` | os 3 `run_train.py` |
| `lerobot.processor.core` | `lerobot.processor.pipeline` | `pi0_depth`, `openvla_depth` |
| `lerobot.utils.control_utils.init_keyboard_listener` | `lerobot.utils.keyboard_input.init_keyboard_listener` | scripts de teleop/record |

Assinaturas que mudaram junto:

- `hw_to_dataset_features(hw_features, prefix, use_video=True)` — o terceiro
  parâmetro era `use_videos`. Nossos wrappers passaram a repassar
  `*args/**kwargs` em vez de casar assinatura com uma API que ainda se mexe.
- `build_dataset_frame(ds_features, values, prefix)` — `prefix` perdeu o default
  e virou `"observation"` (sem ponto), o `OBS_STR`.

### O detalhe que não é óbvio: trocar no módulo de origem não basta

Quem consome faz `from lerobot.utils.feature_utils import build_dataset_frame`,
ou seja, **copia a referência** para o próprio namespace no momento do import.
Trocar o atributo só no módulo de origem não alcança quem já importou — e o
`init_lerobot_record_v2.py` importa `robot.unitree_g1` no topo, que arrasta meio
LeRobot antes das injeções.

Por isso o `_patch_lerobot()` do `init_lerobot_record_v2.py` reescreve a
referência em **todo módulo `lerobot.*` já carregado**, além do de origem (que
cobre os que ainda vão ser importados).

---

## 2. Configuração de treino

### `eval_freq` sumiu

Na 0.6.1 ele foi partido em dois: `env_eval_freq` (rollout no simulador) e
`eval_steps` (loss em episódios separados, via `eval_split`). Nosso laço de
validação usa `val_dataset`, não `eval_split`, então o knob voltou como **campo
nosso** no `CustomTrainPipelineConfig` dos 3 `run_train.py`, com a semântica de
sempre (0 desliga). Sem isso o draccus recusa o YAML inteiro.

### `dataset.vcodec` sumiu

Virou `dataset.rgb_encoder.vcodec`. Corrigido nos 10 YAMLs de `config/record/`.
Profundidade não passa por aí — tem o `depth_encoder` só dela.

### `use_rabc` sumiu

Migrou para `sample_weighting` (ver `configs/train.py::_migrate_legacy_rabc_fields`).
Nenhum YAML nosso usa RA-BC, então o acesso virou
`getattr(cfg, "use_rabc", False)`. Se for usar, o caminho é o `sample_weighting`
nativo, não ressuscitar o campo.

---

## 3. Índice do sampler — o erro que só aparece com `episodes:` no YAML

```
IndexError: Invalid key: 12800 is out of bounds for size 11069
```

O `EpisodeAwareSampler` numera os quadros pelo índice **absoluto** do dataset
inteiro (12.848 quadros), mas o `__getitem__` recebe índice **relativo** ao
subconjunto selecionado por `episodes:` (11.069). A ponte entre os dois é o
argumento `absolute_to_relative_idx=dataset.absolute_to_relative_idx`, que o
`lerobot_train.py` de fábrica passa e o nosso fork não passava.

Acrescentado nos 6 samplers (treino e validação dos 3 `run_train.py`).

---

## 4. Monkeypatches que viraram comportamento nativo

Os 3 `run_train.py` remendavam o `LeRobotDataset.__getitem__` para duas coisas
que a 0.6.1 faz sozinha, no `datasets/dataset_reader.py`:

- remapear índice absoluto → relativo (o reader resolve);
- aplicar `ColorJitter` só nas câmeras RGB
  (`for cam in camera_keys: if cam in depth_keys: continue`).

Manter o patch não era só inútil: ele chamava `self._ensure_hf_dataset_loaded()`,
método que **não existe mais** — matava o treino no primeiro batch.

Ver também o §3.3 do [PROFUNDIDADE_NATIVA.md](PROFUNDIDADE_NATIVA.md), com os
outros três remendos de profundidade que saíram pelo mesmo motivo.

---

## 5. Transporte ZMQ do robô

`ChannelSubscriber.Read()` do shim ZMQ (`robot/unitree_g1/unitree_sdk2_socket.py`)
não aceitava `timeout`, mas o `unitree_g1_loco.py` chama `Read(timeout=0.1)` —
a assinatura do `ChannelSubscriber` do unitree_sdk2py (DDS). A thread
`_subscribe_motor_state` morria no primeiro ciclo e o `_lowstate` nunca era
preenchido: o log ficava repetindo `Waiting for robot state...` para sempre.

O shim passou a aceitar `timeout` e a **ignorá-lo de propósito**: o `recv` já é
`zmq.NOBLOCK` e volta na hora com `None`, que é exatamente o que o timeout do
lado DDS garante.

---

## 6. Dependência nova: extra `training`

`accelerate` e `wandb` saíram para o extra `training` do LeRobot. Sem ele todo
`run_train.py` morre em `from accelerate import Accelerator`. Registrado no
`install.sh`, no `pyproject.toml` da raiz e na §1.1.1 do
[INSTALL.md](../../docs/INSTALL.md).

Quem montou o env antes: `pip install -e "./lerobot[training]"`.

---

## 7. Como conferir que está tudo de pé

```bash
cd lerobot-ext

# 1. todo import de lerobot resolve? (varre o repo inteiro)
python - <<'EOF'
import ast, importlib, pathlib
falhas = []
for f in pathlib.Path(".").rglob("*.py"):
    if "olds" in f.parts or "__pycache__" in f.parts: continue
    for n in ast.walk(ast.parse(f.read_text())):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("lerobot"):
            m = importlib.import_module(n.module)
            for a in n.names:
                if a.name != "*" and not hasattr(m, a.name):
                    falhas.append(f"{f}:{n.lineno} {n.module}.{a.name}")
print("\n".join(falhas) or "todos resolvem")
EOF

# 2. o treino sobe de verdade? (4 passos, batch pequeno)
HF_HUB_OFFLINE=1 WANDB_MODE=disabled python -m policies.act_depth.run_train \
  --config_path=config/train/actdepth_white_cup_on_dripper.yaml \
  --steps=4 --batch_size=2 --num_workers=0 --output_dir=/tmp/smoke
```
