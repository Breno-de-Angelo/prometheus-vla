# Instalação — prometheus-vla com LeRobot v0.6.1

Migração de **LeRobot 0.4.4 (Python 3.10)** para **LeRobot 0.6.1 (Python 3.12)**.

O ambiente antigo (`conda activate g1`) continua funcionando e **não é tocado** por este
guia. O novo vive em `conda activate prometheus-vla`. Se algo der errado, é só voltar
para o `g1`.

> **Como ler este documento.** O §0 abaixo é o **runbook**: a sequência de comandos, e
> nada além dela. Todo o resto é referência — o *porquê* de cada pin estranho, para
> quando algo quebrar. As correções descritas nos §2 e §3 **já estão commitadas**; você
> não precisa aplicá-las de novo, elas estão ali para explicar por que o código está
> como está.

---

## 0. Caminho rápido — instalação do zero

**Automatizado:**

```bash
cd ~/DEV/prometheus-vla
./install.sh                          # env `prometheus-vla`, com MuJoCo
./install.sh --sem-mujoco             # sem a simulação
./install.sh --env prometheus-teste   # noutro env, para testar sem risco
./install.sh --so-verificar           # confere um env já pronto, sem instalar nada
```

O `install.sh` é exatamente a sequência abaixo, com as checagens que ela merece: para se
a branch do submódulo não existir, avisa se o torch vier sem CUDA, confirma que televuer
e dex_retargeting vieram do repo e não do PyPI, e termina rodando os testes. É
idempotente — reaproveita o env e os pacotes já instalados.

> O `install-g1-0.4.4.sh` ao lado dele é o script **antigo**, do ambiente `g1`
> (LeRobot 0.4.4 / Python 3.10). Continua ali de propósito: é o rollback do §7.

**Manual**, se preferir entender cada passo. Se tudo passar no último comando, o
ambiente está de pé.

```bash
# ── 1. repositório e submódulos ─────────────────────────────────────────────
cd ~/DEV/prometheus-vla
git submodule update --init --recursive
cd lerobot && git checkout prometheus-vla/v0.6.1 && cd ..

# ── 2. interpretador (o conda entra SÓ para isso; os pacotes vêm de pip) ────
conda create -n prometheus-vla python=3.12 -y -c conda-forge --override-channels
conda activate prometheus-vla

# ── 3. torch ANTES de tudo, do índice cu128 ─────────────────────────────────
#    RTX 5070 (Blackwell, sm_120) precisa de CUDA 12.8+. Instalado depois, o
#    resolver puxa um wheel CPU ou cu126 e a GPU some. Ver §4.1.
pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.7,<2.12" "torchvision>=0.22.0,<0.27.0"

# ── 4. SDK da Unitree (não vem pelos extras do lerobot) ─────────────────────
pip install git+https://github.com/unitreerobotics/unitree_sdk2_python.git

# ── 5. lerobot (nosso fork) com os extras ───────────────────────────────────
pip install -e "./lerobot[unitree_g1_dex3,televuer,intelrealsense,pi,dataset]"

# ── 6. os dois forks vendorizados, editable e por último ────────────────────
#    O PyPI tem pacotes com estes nomes, mas NÃO servem — ver §2.7 e §4.4.
pip uninstall -y televuer dex_retargeting
pip install -e ./lerobot-ext/teleop/televuer
pip install -e ./lerobot-ext/teleop/robot_control/dex-retargeting

# ── 7. resto ────────────────────────────────────────────────────────────────
pip install flask pytest
pip install "transformers>=5.4.0,<5.6.0"   # só se algum passo trouxe fora do range

# ── 8. opcional: simulação MuJoCo ───────────────────────────────────────────
#    NÃO use unitree-g1-mujoco/requirements.txt: ele pede opencv-python
#    (não-headless) e reintroduz o conflito de dois cv2 do §2.1.
pip install mujoco loguru msgpack msgpack-numpy matplotlib

# ── 9. conferir ─────────────────────────────────────────────────────────────
cd lerobot-ext && python -m pytest tests/ -q     # esperado: 12 passed
```

Depois disso, a verificação completa (versões, forks certos, IK, retargeting) está no
§5 — vale rodar na primeira instalação.

### Três coisas que mordem, e onde estão explicadas

| Sintoma | Causa | Onde |
|---|---|---|
| `dex_retargeting is required` / dedos parados | veio a 0.5.0 do PyPI em vez do fork 0.4.7 | §2.7 |
| `Retargeting type must be one of [...]` | cwd errado — rode de `lerobot-ext/` | §5.1 |
| vuer reclamando de `aiohttp` | mentira; é o `params-proto` 3.x | §2.2 |

> **Numa máquina nova**, o passo 1 só funciona depois que a branch
> `prometheus-vla/v0.6.1` do submódulo tiver sido enviada para o fork
> (`git push origin prometheus-vla/v0.6.1` de dentro de `lerobot/`). Hoje ela existe
> apenas localmente.

---

## 1. Por que não dá para simplesmente atualizar

A 0.6.1 muda os pisos de várias dependências, e alguns colidem de frente com pins que o
projeto carrega hoje. Estes são os conflitos reais, medidos no `pyproject.toml` de cada
tag e no que está instalado no env `g1`:

| Pacote | Projeto hoje (env `g1`) | LeRobot 0.6.1 exige | Situação |
|---|---|---|---|
| **Python** | 3.10 | `>=3.12` | Bloqueante — exige env novo |
| **numpy** | 1.26.4 (`<2.0.0`) | `>=2.0.0,<2.3.0` | Bloqueante — ver §2.1 |
| **vuer** | 0.0.60 | — (não usa) | Conflito interno — ver §2.2 |
| **transformers** | 5.6.1 | `>=5.4.0,<5.6.0` | Fora do range → downgrade |
| **torch** | 2.10.0 | `>=2.7,<2.12` | OK |
| **datasets** | 4.1.0 | `>=4.8.0,<5.0.0` (extra `dataset`) | Precisa subir |
| **diffusers** | 0.30.0 | `>=0.38.0,<0.40.0` | Precisa subir |
| **huggingface-hub** | `<0.36` | `>=1.6.0,<2.0.0` | Major bump |
| **draccus** | `==0.10.0` | `>=0.11.6,<0.12.0` | Precisa subir |
| **params-proto** | 2.13.2 | — (dep do vuer) | Armadilha — ver §2.2 |
| **opencv** | `opencv-python` 4.11 **e** `-headless` 4.12 | `-headless>=4.9,<4.14` | Dois `cv2` — ver §2.1 |

Dois pontos que não são óbvios e explicam a dor de cabeça atual:

### 1.1 `datasets` saiu do core

Na 0.4.4, `datasets` vinha junto com o LeRobot. Na 0.6.1 ele foi movido para o extra
`dataset`. **Quem usa `LeRobotDataset` precisa instalar `lerobot[dataset]`
explicitamente** — senão o import quebra só na hora de rodar.

### 1.2 Os extras `unitree_g1_dex3` e `televuer` não existem no upstream

Seu `pyproject.toml` pede `lerobot[unitree_g1_dex3,televuer,intelrealsense,pi]`. Desses,
**`unitree_g1_dex3` e `televuer` são invenção do fork** — o upstream tem só `unitree_g1`.
Eles só voltam a existir depois que o rebase do §3 estiver feito. E no upstream a linha
`unitree-sdk2==1.0.1` está **comentada** ("requires specific installation instructions"),
então o SDK continua sendo instalado à parte, via git.

---

## 2. As correções, e por que cada uma existe — **já aplicadas**

> Nada desta seção precisa ser executado: tudo aqui já está commitado no repo.
> É o registro do que foi mexido e do motivo — leia quando algo quebrar, ou
> antes de "limpar" um pin que parece arbitrário. Vários deles não são.

### 2.1 televuer: numpy 2 + opencv headless — **JÁ APLICADO E TESTADO**

`lerobot-ext/teleop/televuer/pyproject.toml` pedia `numpy<2.0.0`. O LeRobot 0.6.1 exige
`numpy>=2.0.0`. Um elimina o outro — e no env `g1` **o televuer venceu**, travando tudo
em numpy 1.26.4.

O televuer é fork nosso e não usa nenhuma API removida no numpy 2 (varredura completa:
`np.float_`, `np.NaN`, `np.Inf`, `np.product`, `np.alltrue`, `np.in1d`, `copy=False`,
`.itemset()`, `.newbyteorder()` — zero ocorrências). O pin foi solto e **validado por
execução** (§2.4).

Junto veio uma segunda correção: o televuer pedia `opencv-python` sem teto, que resolvia
para a **5.0.0.93**, enquanto o lerobot depende de `opencv-python-headless>=4.9,<4.14`.
Os dois pacotes instalam o **mesmo módulo `cv2`** — ter ambos é conflito de binários (o
env `g1` tem exatamente isso: 4.11 e 4.12 lado a lado). O televuer só usa `cvtColor` e
`resize`, nunca `imshow`/`waitKey`, então headless serve e alinha com o lerobot.

O estado atual do arquivo:

```toml
requires-python = ">=3.12"

dependencies = [
    "numpy>=2.0.0,<2.3.0",                  # era "numpy<2.0.0"
    "opencv-python-headless>=4.9.0,<4.14.0", # era "opencv-python" (resolvia p/ 5.x)
    "logging-mp",
    "vuer[all]==0.0.60",                    # manter pinado — ver §2.2
    "params-proto>=2.13.0,<3.0.0",          # armadilha do vuer — ver §2.2
]
```

### 2.2 A armadilha do `params-proto` (erro enganoso do vuer)

Num ambiente limpo, `from vuer import Vuer` falha com:

```
ImportError: cannot import name 'Vuer' from 'vuer'
```

precedido por uma mensagem do vuer dizendo que falta `aiohttp` e mandando instalar
`vuer[all]`. **A mensagem mente.** O `vuer/__init__.py` envolve o import num
`except ImportError` genérico que engole a causa real:

```
vuer/server.py:14: from params_proto import Flag, PrefixProto, Proto
ImportError: cannot import name 'Flag' from 'params_proto'
```

O vuer 0.0.60 declara `params-proto>=2.13.0` **sem teto**, e o params-proto 3.x removeu
`Flag`. Em ambiente novo o pip pega o 3.3.0 e quebra. O env `g1` só funciona porque tem
o 2.13.2 preso lá de antes.

Como o pyproject do vuer vem do PyPI e não dá para editar, o teto foi posto no televuer:
`params-proto>=2.13.0,<3.0.0`.

> Se você algum dia vir o vuer reclamando de aiohttp, **não instale aiohttp** — rode
> `python -c "from vuer.server import Vuer"` para ver o erro verdadeiro.

### 2.3 Corrigir o `vuer` no pyproject raiz — mantenha 0.0.60 — **JÁ APLICADO**

O `pyproject.toml` da raiz pede `vuer>=0.1.1`, mas o televuer pinou `vuer[all]==0.0.60`.
São incompatíveis, e **o televuer venceu**: o env `g1` tem vuer 0.0.60 instalado. Ou
seja, o `>=0.1.1` do pyproject raiz nunca foi satisfeito — é uma declaração morta que só
atrapalha o resolver.

Mantenha o 0.0.60. O televuer importa `WebRTCVideoPlane` e `WebRTCStereoVideoPlane` de
`vuer.schemas`, que são API dessa versão; subir para 0.1.x tem risco real de quebrar o
teleop. Na raiz:

```toml
# pyproject.toml (raiz)
dependencies = [
    "flask>=3.1.2",
    "lerobot[unitree_g1_dex3,televuer,intelrealsense,pi,dataset]",
    "vuer==0.0.60",   # era "vuer>=0.1.1" — nunca foi satisfeito, o televuer pina 0.0.60
]
requires-python = ">=3.12"
```

> Nota: `visualization/visualize_g1_3d.py` também importa `vuer` direto
> (`Vuer`, `VuerSession`, `vuer.schemas.Urdf`). Com 0.0.60 continua como está hoje.

### 2.4 Validação do televuer sob numpy 2 — executada

As correções de §2.1 e §2.2 foram instaladas no env novo e o código numérico foi
exercitado de fato (não só importado), com **numpy 2.2.6 / cv2 4.13.0**:

| O que foi testado | Resultado |
|---|---|
| `safe_mat_update` (válido e singular), `fast_mat_inv`, `safe_rot_update` (25,3,3) | ok |
| `SharedMemory(size=np.prod(shape) * np.uint8().itemsize)` | ok — aceita o `np.int64` que o numpy 2 devolve |
| `cv2.cvtColor` BGR2RGB / GRAY2RGB, `cv2.resize` | ok |
| `Array('d', 16/75/225)` → `(4,4)` / `(25,3)` / `(25,3,3)` | ok |
| `np.concatenate` homogêneo dos landmarks (`tv_wrapper:327`) | ok |
| `from vuer.schemas import WebRTCVideoPlane, WebRTCStereoVideoPlane, …` | ok |
| `TeleVuerWrapper.__init__` tem `wrist_cam` (prova que é o fork) | ok |

O ponto que mais preocupava era o `np.prod`: no numpy 2 ele retorna `np.int64` em vez de
`int`, e `SharedMemory` recebe isso como `size`. Funciona — o CPython converte via
`__index__`.

**O que isto não cobre:** nada que precise do headset. O handshake WebRTC, o streaming de
vídeo para o XR e a captura real de poses de mão só podem ser validados com o hardware.

### 2.5 Atualizar `.python-version` — **JÁ APLICADO**

```bash
echo "3.12" > .python-version
```

### 2.6 O `requirements.txt` precisa ser refeito

Os pins de lá (`numpy<2.0.0`, `datasets==4.1.0`, `diffusers>=0.30.0`, `torch>=2.4.0`)
são da era 0.4.4 e conflitam com a 0.6.1. Depois que a instalação abaixo funcionar, gere
um novo a partir do que ficou de fato instalado (§6).

### 2.7 `dex_retargeting`: use o fork vendorizado, não o PyPI — **JÁ APLICADO E TESTADO**

Esta foi a armadilha mais cara da migração, porque **falhava calada**. O env `g1` tem
`dex_retargeting` **0.4.7**, que **não existe no PyPI** — é o fork vendorizado em
`lerobot-ext/teleop/robot_control/dex-retargeting/` (versão bumpada por silencht),
instalado à mão lá atrás. Como o extra `unitree_g1_vr_teleop` pedia só
`dex-retargeting>=0.1.0`, o env novo pegou a **0.5.0** do PyPI. Ela quebra em dois
lugares:

1. **Parou de re-exportar `RetargetingConfig`** em `dex_retargeting/__init__.py` (agora
   só existe em `dex_retargeting.retargeting_config`). Em
   `robot/unitree_g1/robot_control/hand_retargeting.py` esse import está dentro de um
   `try`, então a quebra virava `HAS_DEX_RETARGETING = False` — o retargeting dos dedos
   simplesmente sumia. Já em `teleop/robot_control/hand_retargeting.py` (a via que o
   `teleop/xr_g1_arm.py` realmente usa) o import é direto, e derrubava a teleoperação
   no import.
2. **Mudou o schema do `RetargetingConfig`**: os três campos
   `target_link_human_indices_{dexpilot,position,vector}` viraram um único
   `target_link_human_indices`. Os nossos `.yml` de mão
   (`assets/unitree_hand/unitree_dex3.yml`) são do formato antigo, e a 0.5.0 responde
   com `TypeError: unexpected keyword argument 'target_link_human_indices_dexpilot'`.

Migrar os `.yml` mexeria na cinemática dos dedos sem como validar sem o robô, então a
decisão foi **ficar na 0.4.7 vendorizada** — mesma política do vuer 0.0.60.

Os imports foram corrigidos assim mesmo, nas 5 cópias do arquivo (`from
dex_retargeting.retargeting_config import RetargetingConfig`): funciona nas duas
versões e tira a dependência de um re-export que já provou ser instável.

O `pyproject.toml` do fork vendorizado carregava os mesmos pins mortos do televuer:

```toml
# lerobot-ext/teleop/robot_control/dex-retargeting/pyproject.toml
"numpy>=2.0.0,<2.3.0",   # era "numpy<2.0.0"
"torch>=2.7,<2.12",      # era "torch==2.3.0" — derrubaria o cu128 da RTX 5070
"nlopt>=2.8.0",          # era "nlopt>=2.6.1,<2.8.0"
```

O `nlopt` merece nota: os wheels até a 2.7.x foram compilados contra o numpy 1, e
importá-los sob numpy 2 estoura *"module compiled using NumPy 1.x cannot be run in
NumPy 2.x"*. Esse erro chegava **disfarçado**, engolido pelo `except ImportError` do
`hand_retargeting`, virando um `"dex_retargeting is required"` que apontava para o
lugar errado. A 2.8+ é compilada com numpy 2.

Instalação (editable, como o televuer):

```bash
pip uninstall -y dex_retargeting
pip install -e ./lerobot-ext/teleop/robot_control/dex-retargeting
```

**Validado por paridade numérica**, não só por import: as mesmas 21 landmarks passadas
pela via real (`ref_value` de 6 vetores + reordenação para a ordem do hardware) produzem
saídas **bit a bit idênticas** no env `g1` (0.4.4 / py3.10) e no `prometheus-vla`
(0.6.1 / py3.12).

### 2.8 Cache do modelo pinocchio: versionado por versão do `pin` — **JÁ APLICADO E TESTADO**

`teleop/robot_control/robot_arm_ik.py` guarda os `pin.Model` num pickle
(`g1_29_model_cache.pkl`) para não reler o URDF a cada subida. **Esse formato não é
estável entre versões do pinocchio**: o cache gravado pelo pin 2.7 (env `g1`) faz o pin
4.1 (env novo) estourar `RuntimeError: input stream error` dentro do `__init__`, antes
de a teleoperação subir.

Apagar o cache resolveria o env novo e **quebraria o `g1`** na volta — o novo pickle não
é legível pelo pin 2.7 —, o que contraria a promessa de rollback do §7. Então o nome do
arquivo passou a carregar a versão: `g1_29_model_cache_pin4_1_0.pkl`. Cada ambiente
escreve e lê o seu, e os dois convivem no mesmo diretório.

Junto veio uma segunda proteção: o carregamento do cache agora é tolerante. Um pickle
ausente, truncado (queda de energia no meio do `dump`) ou de formato alheio só custa a
reconstrução a partir do URDF — lenta, mas correta — em vez de derrubar a sessão por um
arquivo descartável.

**Validado por paridade numérica**: `G1_29_ArmIK.solve_ik` com as mesmas poses de pulso
devolve os mesmos ângulos nos dois ambientes, com diferença máxima de 6.4e-16 rad
(~4e-14 grau) — ou seja, pinocchio 2.7→4.1 e casadi 3.6.7→3.7.2 não mudaram o IK.

### 2.9 Teardown do stream ZMQ: fechar socket com thread dentro do `recv` aborta o processo

Achado ao escrever os testes do §5.2, não pela migração — mas é um crash de verdade, e
aparece justamente quando a câmera cai.

`_SharedZMQStream.stop()` (no nosso fork do lerobot) sinalizava a parada, fazia
`join(timeout=2.0)` e fechava socket e contexto **mesmo se a thread de leitura ainda
estivesse dentro de `recv_multipart`**. Sockets do zmq não são thread-safe: nesse caso a
libzmq não levanta exceção, ela **aborta o processo** — `Fatal Python error: Aborted`,
core dump.

O gatilho era o RCVTIMEO do socket, que valia o `timeout_ms` de configuração — **10 s** no
dex3. Com o servidor de imagem mudo, a thread ficava presa lá dentro por até 10 s, muito
além dos 2 s do join. Ou seja: câmera cai, você encerra a sessão, o processo morre com
core dump em vez de sair limpo.

Duas mudanças em `src/lerobot/cameras/zmq/camera_zmq.py`:

- o RCVTIMEO do laço vira um **poll curto de 200 ms**, que só decide de quanto em quanto
  tempo a thread olha o `stop_event`. A espera por quadro continua sendo a de
  `get_frame`, que tem deadline próprio sobre a `Condition` — nada muda para o chamador;
- o `stop()` passa a esperar de verdade e, se ainda assim a thread não morrer, **não
  fecha** o socket. Vazar um socket até o fim do processo é barato; abortar não.

Efeito colateral mensurável: a suíte de testes caiu de ~22 s para ~11 s, porque o
encerramento de cada câmera deixou de esperar o timeout inteiro.

---

## 3. O rebase do fork sobre a v0.6.1 — **já feito**

> Também é registro. O resultado está na branch `prometheus-vla/v0.6.1` do
> submódulo, e o passo 1 do §0 só faz `git checkout` dela. O que segue é como
> foi feito, para o dia em que a v0.7 sair e a operação tiver de se repetir.

O fork `Breno-de-Angelo/lerobot` está todo em 0.4.4 (todas as branches). Os patches
próprios — dex3, televuer, IK do G1, stream zmq — são **11 commits** tocando 22 arquivos.
Precisam ser replantados em cima da v0.6.1.

A boa notícia: só **8 arquivos** colidem com o que o upstream mexeu no mesmo período.
Os outros 14 aplicam limpo.

**Arquivos que vão dar conflito:**

```
pyproject.toml
src/lerobot/cameras/zmq/camera_zmq.py
src/lerobot/robots/unitree_g1/__init__.py
src/lerobot/robots/unitree_g1/g1_utils.py
src/lerobot/robots/unitree_g1/unitree_sdk2_socket.py
src/lerobot/robots/utils.py
src/lerobot/teleoperators/teleoperator.py
src/lerobot/teleoperators/utils.py
```

**Comandos:**

```bash
cd ~/DEV/prometheus-vla/lerobot

# upstream já foi adicionado; se não estiver:
git remote add upstream https://github.com/huggingface/lerobot
git fetch upstream --tags

BASE=$(git merge-base main origin/sync-zmq-depth)
git checkout -b prometheus-vla/v0.6.1 origin/sync-zmq-depth
git rebase --onto v0.6.1 "$BASE"
```

Resolva os conflitos, `git add <arquivo>`, `git rebase --continue`.
Para desistir a qualquer momento: `git rebase --abort`.

**Atenção especial no `pyproject.toml`**: ao resolver, mantenha os extras
`unitree_g1_dex3` e `televuer` que o fork adiciona, mas rebaseie-os sobre a estrutura
nova da 0.6.1 (que usa o padrão `lerobot[xxx-dep]` em vez de listar pacotes soltos).

Um cuidado semântico que o rebase **não** vai acusar: o upstream renomeou
`robots/unitree_g1/robot_kinematic_processor.py` → `g1_kinematics.py` e reescreveu
`unitree_g1.py` (implementação WBC, PR #2876). Os arquivos do fork não importam esse
módulo diretamente — os imports são de `lerobot.processor`, `g1_utils` e
`teleoperators.teleoperator`, todos ainda existentes — mas vale rodar os smoke tests do
§5 antes de confiar.

---

## 4. Instalação

O ambiente já foi criado com Python 3.12.13:

```bash
conda create -n prometheus-vla python=3.12 -y -c conda-forge --override-channels
```

> Usamos `-c conda-forge --override-channels` de propósito: evita o Terms of Service dos
> canais `repo.anaconda.com` (que têm restrição de uso comercial).

O conda entra só para fornecer o interpretador. Os pacotes vêm todos de pip, para não
misturar dois resolvers no mesmo env.

```bash
conda activate prometheus-vla
cd ~/DEV/prometheus-vla
```

### 4.1 torch primeiro, com CUDA 12.8

Sua GPU é uma **RTX 5070 Laptop (Blackwell, sm_120)**, que precisa de CUDA 12.8+. Instale
o torch antes de tudo, do índice cu128, para o resolver não puxar um wheel CPU ou cu126:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.7,<2.12" "torchvision>=0.22.0,<0.27.0"
```

Confirme antes de seguir:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 4.2 Unitree SDK

Não vem pelos extras do LeRobot (está comentado no upstream):

```bash
pip install git+https://github.com/unitreerobotics/unitree_sdk2_python.git
```

### 4.3 LeRobot (fork rebaseado) com os extras

```bash
pip install -e "./lerobot[unitree_g1_dex3,televuer,intelrealsense,pi,dataset]"
```

Se o rebase do §3 ainda não estiver feito, `unitree_g1_dex3` e `televuer` não existem e
este comando falha. Nesse caso, ou faça o rebase, ou instale temporariamente com os
extras do upstream para validar o resto:

```bash
pip install -e "./lerobot[unitree_g1,intelrealsense,pi,dataset]"
```

### 4.4 televuer — nosso fork, editable, por último

> Já instalado e validado neste env (§2.4). O que segue é o motivo de ele vir por último
> e de precisar do `-e`.

Este passo é o mesmo do `install.sh` antigo e o motivo continua valendo:

O extra `televuer` do LeRobot **não instala** o pacote `televuer` — ele só puxa
vuer/aiortc/opencv. O televuer de upstream (silencht) tem a **mesma versão** (4.0.0) com
conteúdo diferente, e o teleop quebra com
`TypeError: TeleVuerWrapper.__init__() got an unexpected keyword argument 'wrist_cam'`,
porque o painel de câmera de pulso é adição nossa. O `-e` é obrigatório: sem ele, um
`git pull` não atualiza o pacote instalado.

```bash
pip uninstall -y televuer
pip install -e ./lerobot-ext/teleop/televuer
```

### 4.5 Resto do projeto

```bash
pip install flask
```

### 4.5.1 dex_retargeting — o fork vendorizado, editable

Pelo mesmo motivo do televuer: o que está no PyPI (0.5.0) não serve, e a versão que
funciona (0.4.7) só existe dentro do repo. Ver §2.7.

```bash
pip uninstall -y dex_retargeting     # remove a 0.5.0 se o resolver a trouxe
pip install -e ./lerobot-ext/teleop/robot_control/dex-retargeting
```

### 4.6 Fixar o transformers dentro do range

O env `g1` tem transformers 5.6.1, que está **fora** do range da 0.6.1
(`>=5.4.0,<5.6.0`). Se algum passo acima o trouxe errado:

```bash
pip install "transformers>=5.4.0,<5.6.0"
```

---

## 5. Verificação

```bash
conda activate prometheus-vla
cd ~/DEV/prometheus-vla

# versões batem com o esperado?
python - <<'EOF'
import numpy, torch, transformers, lerobot
print("lerobot     ", lerobot.__version__)
print("python      ", __import__("sys").version.split()[0])
print("numpy       ", numpy.__version__, "(esperado: 2.x)")
print("torch       ", torch.__version__, "cuda:", torch.version.cuda, torch.cuda.is_available())
print("transformers", transformers.__version__, "(esperado: >=5.4,<5.6)")
EOF

# o televuer é O NOSSO? (a versão é igual à de upstream e não distingue —
# checa pelo caminho e por um parâmetro que só existe no fork)
python - <<'EOF'
import inspect, sys
from pathlib import Path
import televuer
from televuer import TeleVuerWrapper

caminho = Path(televuer.__file__).resolve()
if "wrist_cam" not in inspect.signature(TeleVuerWrapper.__init__).parameters:
    print(f"ERRO: televuer instalado NAO e o nosso fork -> {caminho}")
    print("Conserte: pip uninstall -y televuer && pip install -e ./lerobot-ext/teleop/televuer")
    sys.exit(1)
print(f"televuer OK (fork, editable) -> {caminho}")
EOF

# os módulos do fork sobreviveram ao rebase?
python - <<'EOF'
from lerobot.robots.unitree_g1.unitree_g1_dex3 import *
from lerobot.teleoperators.televuer import *
from lerobot.cameras.zmq.camera_zmq import *
print("modulos do fork OK")
EOF

# dataset (extra separado na 0.6.1)
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('dataset OK')"

# o dex_retargeting é o fork vendorizado 0.4.7, e não a 0.5.0 do PyPI? (ver §2.7)
python - <<'EOF'
import sys, dex_retargeting
from dex_retargeting.retargeting_config import RetargetingConfig
if "lerobot-ext/teleop/robot_control/dex-retargeting" not in dex_retargeting.__file__:
    print(f"ERRO: dex_retargeting veio do PyPI -> {dex_retargeting.__file__}")
    print("Conserte: pip uninstall -y dex_retargeting && "
          "pip install -e ./lerobot-ext/teleop/robot_control/dex-retargeting")
    sys.exit(1)
print("dex_retargeting OK (fork 0.4.7, editable)")
EOF

# Retargeting das mãos e IK dos braços: rodar DE DENTRO de lerobot-ext (ver §5.1).
# Não basta importar — o que quebrou na migração (§2.7, §2.8) só aparece ao construir.
# O IK na 1ª vez é lento: reconstrói o cache do pinocchio.
cd ~/DEV/prometheus-vla/lerobot-ext && python - <<'EOF'
import sys, numpy as np
sys.path.insert(0, ".")
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

hr = HandRetargeting(HandType.UNITREE_DEX3)
d = np.random.default_rng(0).normal(scale=0.05, size=(21, 3))
ref = d[hr.left_indices[1, :]] - d[hr.left_indices[0, :]]
q = hr.left_retargeting.retarget(ref)[hr.left_dex_retargeting_to_hardware]
assert q.shape == (7,) and np.all(np.isfinite(q)), q
print("retargeting das maos OK ->", np.round(q, 3))

ik = G1_29_ArmIK()
L = np.eye(4); R = np.eye(4)
L[:3, 3] = [0.25, 0.25, 0.1]; R[:3, 3] = [0.25, -0.25, 0.1]
qa, _ = ik.solve_ik(L, R)
assert np.shape(qa) == (14,) and np.all(np.isfinite(qa)), qa
print("IK dos bracos OK ->", np.round(qa, 3))
EOF
cd ~/DEV/prometheus-vla
```

### 5.1 O cwd faz parte da configuração: rode de `lerobot-ext/`

`hand_retargeting.py` e `robot_arm_ik.py` resolvem assets por caminho **relativo ao cwd**
(`assets/unitree_hand/unitree_dex3.yml` e `../assets/g1/g1_body29_hand14.urdf`), então o
diretório de onde você chama decide qual arquivo é lido. O certo é `lerobot-ext/` — é o
que os caminhos assumem, e é onde está o cache `g1_29_model_cache*.pkl`.

Isso importa porque existem **duas árvores de `assets/`** com YAMLs de mão diferentes:

| Arquivo | `type` | Funciona com a 0.4.7? |
|---|---|---|
| `lerobot-ext/assets/unitree_hand/unitree_dex3.yml` | `DexPilot` | sim — é o que a teleop usa |
| `assets/unitree_hand/unitree_dex3.yml` (raiz) | `pinch` | **não** |

Rodar da raiz do repo pega o segundo e falha com
`ValueError: Retargeting type must be one of ['vector', 'position', 'dexpilot']` — `pinch`
é tipo de uma versão mais nova do dex-retargeting. Não é problema de instalação; é cwd
errado.

---

### 5.2 Testes automatizados do caminho das câmeras

```bash
conda activate prometheus-vla
cd ~/DEV/prometheus-vla/lerobot-ext
python -m pytest tests/ -v
```

`tests/test_camera_resiliencia.py` sobe um **servidor ZMQ falso** (mesmo protocolo
`zmq.raw.v1` do servidor de imagem do robô) e exercita o `_read_camera` de verdade,
incluindo os modos de falha que não dá para provocar de propósito com o robô na frente:

| Caso | O que garante |
|---|---|
| leitura normal | quadros avançam |
| servidor emudece dentro da carência | **reusa** o último quadro, não levanta |
| consumidor escreve no quadro recebido | a reserva **não** é corrompida |
| silêncio além de `camera_grace_s` | levanta `TimeoutError` (servidor caído ≠ engasgo) |
| nenhum quadro desde o início | falha de cara (endereço errado) |
| dois engasgos seguidos | a carência **zera** ao voltar quadro novo |

Roda contra as **duas** cópias da lógica (`unitree_g1_loco.py`, em uso, e
`unitree_g1.py`) para as duas não divergirem em silêncio — 12 testes no total.

Foi escrevendo isso que apareceram dois defeitos que nenhum import teria mostrado: a
cópia do quadro só existia no caminho de falha (§2.8 do código: a reserva podia ser
corrompida por quem desenha na imagem) e o crash de teardown do §2.9.

## 6. Congelar o que funcionou

Quando tudo passar, gere o novo `requirements.txt` a partir do estado real:

```bash
conda activate prometheus-vla
pip freeze | grep -vE "^-e |lerobot|televuer" > requirements-lock.txt
```

Mantenha os três editables (`lerobot`, `televuer`, `unitree_sdk2_python`) fora do lock —
eles vêm do repo, não do PyPI.

---

## 7. Rollback

Nada aqui altera o ambiente antigo:

```bash
conda activate g1          # volta ao setup 0.4.4 / Python 3.10
```

Para descartar o env novo por completo:

```bash
conda env remove -n prometheus-vla
```

E no submódulo, para voltar ao ponto de partida:

```bash
cd ~/DEV/prometheus-vla/lerobot
git checkout sync-zmq-depth      # a branch pré-migração (0.4.4), que segue intacta
```

---

## 8. Riscos conhecidos que ainda precisam de validação em hardware

Estes pontos não dá para verificar sem o robô e sem o rebase concluído:

1. **transformers v4 → v5** afeta `lerobot-ext/policies/pi0_depth/modeling_pi05.py`, que
   importa `transformers.models.gemma.modeling_gemma`,
   `transformers.models.paligemma.modeling_paligemma` e
   `transformers.utils.cached_file`. O arquivo já tem camadas de compatibilidade entre
   versões, mas foram escritas para a v4.
2. **huggingface-hub 0.x → 1.x** é major bump; checkpoints e chamadas de download podem
   precisar de ajuste. O CLI `huggingface-cli` virou `hf` (não encontrei uso no projeto).
3. **Reescrita do G1 no upstream** (PR #2876, WBC): `unitree_g1.py` foi bastante
   reescrito e surgiram `gr00t_locomotion.py`, `holosoma_locomotion.py` e
   `run_g1_server.py`. Os arquivos de `lerobot-ext/robot/unitree_g1/` que você tem
   modificados no working tree podem precisar de acerto. O rebase do §3 está feito e
   esses módulos importam sob a 0.6.1, mas isso é import, não comportamento.
4. **Nada do caminho XR foi exercitado**: handshake WebRTC, streaming de vídeo para o
   headset e captura real de poses de mão precisam do Quest/Vision Pro. O que foi
   validado sem hardware é a matemática a jusante — retargeting (§2.7) e IK (§2.8) —,
   com paridade numérica contra o env `g1`.
5. **O caminho ZMQ das câmeras não foi testado contra o robô.** A lógica de tolerância a
   engasgo tem cobertura automatizada contra servidor falso (§5.2), o que fecha os modos
   de falha do *nosso* código. O que continua sem teste é o comportamento da rede real:
   Wi-Fi com perda, latência variável, e os `warmup_s=5, timeout_ms=10000` do dex3 contra
   o servidor de imagem de verdade.

### 8.1 Estado do que ficou fora do caminho principal

- **`mujoco` 3.11.0 instalado e validado** (`pip install mujoco loguru msgpack
  msgpack-numpy matplotlib`). Não instale o `unitree-g1-mujoco/requirements.txt` inteiro:
  ele pede `opencv-python` (não-headless), que reintroduz o conflito de dois `cv2` do
  §2.1. O que foi verificado: os módulos de `unitree-g1-mujoco/sim/` importam, e
  `scene_43dof.xml` carrega e roda 500 passos de física (nq=57, nv=55) com qpos finito.
  **A renderização não foi testada** — precisa de display; nesta máquina, headless, o
  backend `osmesa` não está disponível. A física, que é o que a simulação usa, não
  depende de GL.
- **`G1_ASSETS_DIR` corrigido.** O fallback antigo (`__file__.parent.parent / "assets"`)
  apontava para um diretório inexistente em três das quatro cópias de
  `hand_retargeting.py`, e ninguém exporta a variável — o sintoma era um
  `ValueError: URDF dir ... not exists` acusando um caminho que nunca existiu. Agora,
  sem a variável, sobe-se a árvore procurando um `assets/` que realmente contenha
  `unitree_hand/unitree_dex3.yml`. A variável continua valendo como override.
- **Atenção: há três árvores de `assets/` e duas delas são de outra versão do
  dex-retargeting.** `assets/` (raiz) e `lerobot/src/lerobot/robots/unitree_g1/assets/`
  usam `type: pinch`, que a 0.4.7 rejeita; só `lerobot-ext/assets/` tem o `DexPilot` que
  funciona (§5.1). Não unifiquei porque escolher qual sobrevive muda a cinemática dos
  dedos, e isso precisa do robô para validar.
- **`robot/unitree_g1/robot_control/hand_retargeting.py` é código morto** — nada no repo
  o importa (só o `__init__.py` do próprio pacote). Vale saber que o `retarget_left` dele
  passa as 21 landmarks cruas direto para `retarget()`, enquanto a via viva
  (`teleop/robot_control/`) monta antes os 6 vetores de referência. Se um dia alguém
  ressuscitar essa cópia, é aí que ela vai calar em vez de falhar.
