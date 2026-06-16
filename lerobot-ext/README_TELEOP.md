# Teleoperação VR — G1 + Dex3

Guia para **teleoperar** o Unitree G1 (braços + mãos Dex3) via VR (Quest/Vuer).

> Branch obrigatória: **`Luiz-pi05d`**. Ambiente conda: **`g1`**.

---

## 1. Topologia

A teleop **roda no laptop**; o robô roda os servidores de baixo nível e de câmera.

```
┌─────────────────────┐   VR (Vuer :8012)   ┌──────────────────┐
│  Quest (headset)    │ ◄─────────────────► │                  │
└─────────────────────┘                     │  LAPTOP (cliente)│
                                            │  conda env g1    │
                          ZMQ câmera :5555   │  branch Luiz-    │
┌─────────────────────┐ ◄─────────────────  │  pi05d           │
│  ROBÔ Unitree G1    │                     │  init_lerobot_   │
│  - realsense depth16│ ──── DDS ─────────► │  record.py       │
│  - run_g1_server.py │   (motores/Dex3)    └──────────────────┘
└─────────────────────┘
```

---

## 2. Pré-requisitos (laptop)

```bash
cd ~/I2CA/prometheus-vla
git branch --show-current            # tem que dizer: Luiz-pi05d
conda activate g1
```

Versões que o ambiente `g1` exige (**não atualizar cegamente**):
- `numpy<2` (1.26.4) — NumPy 2.x dá segfault no pinocchio
- `torch==2.3.0` + `torchvision==0.18.0` (casados)
- `dex_retargeting` **0.4.7** do diretório local `lerobot-ext/teleop/robot_control/dex-retargeting/`
- `unitree_sdk2py` do submódulo local `./unitree_sdk2_python/`
- Opcionais: `speech_recognition` (voz) e `pynput` (teclado)

---

## 3. Subir os servidores no robô

### Terminal 1 — câmera depth 16-bit

```bash
ssh unitree@192.168.123.164   # rede cabeada direta notebook ↔ robô
source ~/miniconda3/etc/profile.d/conda.sh && conda activate g1
cd ~/lerobot/src/lerobot/cameras/zmq
python realsense_server_depth16.py --serial 327122071538   # confirme o serial da SUA D435i
```

### Terminal 2 — servidor de baixo nível do G1

```bash
ssh unitree@192.168.123.164   # rede cabeada direta notebook ↔ robô
source ~/miniconda3/etc/profile.d/conda.sh && conda activate g1
cd ~/lerobot/src/lerobot/robots/unitree_g1
python run_g1_server.py                          # espere "bridge running"
```

---

## 4. Rodar — simulação (ensaio de teleop)

> ⚠️ O sim **não** grava depth16 real — use só para validar movimento/controle. Veja `README_RECORD.md §Armadilhas`.

```bash
cd ~/I2CA/prometheus-vla
conda activate g1
python lerobot-ext/init_lerobot_record.py \
    --config_path lerobot-ext/config/record/record_televuer.yaml \
    --sim \
    --quest-ip 192.168.68.61 \
    --quest-adb-ip=192.168.68.51
```

`--sim` sobe o MuJoCo, o servidor Vuer na **:8012** e o publicador ZMQ na **:5555**.
Acesse pelo Quest: `https://<ip-do-laptop>:8012/?ws=wss://<ip-do-laptop>:8012`.

> ⚠️ **Passe sempre `--quest-ip`**. Sem ele o browser do Quest não é aberto → sem sessão VR ativa,
> o teleop não recebe controle. Esse é o motivo nº 1 de "os controles pararam de funcionar".

- **As mãos Dex3 respondem no sim**: o `connect()` sobe a ponte ZMQ↔DDS e controla os dedos. Por padrão as **duas** mãos são controladas; com `--left-arm-limp` a esquerda inteira fica solta.
- **A pressão tátil NÃO existe no sim** — `left/right_hand_pressure` saem zeradas.

---

## 5. Flags da linha de comando

| Flag | O que faz | Default |
|---|---|---|
| `--config_path <yaml>` | Config draccus do record (obrigatório) | — |
| `--sim` | Liga simulação (MuJoCo) | off (robô real) |
| `--quest-ip <ip>` | IP do laptop/servidor Vuer — abre o browser do Quest via ADB | não abre |
| `--quest-adb-ip <ip>` | IP do próprio headset Quest (para ADB) | `192.168.68.51:5555` |
| `--left-arm-limp` | Braço + mão esquerdos com `kp=0` — só o lado direito é teleoperado | off |
| `--debug` | Grava tudo enviado/recebido em `/tmp/g1_debug_io_<ts>.jsonl` | off |
| `--dataset.root=<dir>` | Diretório fixo do dataset (sem ele → auto-timestamp) | auto |

**Mão/braço esquerdo:**
- **Sem `--left-arm-limp` (default):** ambos os lados controlados normalmente.
- **Com `--left-arm-limp`:** lado esquerdo inteiro mole (`kp=0`). Use quando gravar só com o lado direito.

**Exemplos:**
```bash
# Sim, as duas mãos ativas:
python lerobot-ext/init_lerobot_record.py \
    --config_path lerobot-ext/config/record/record_televuer.yaml \
    --sim --quest-ip 192.168.68.61 --quest-adb-ip=192.168.68.51

# Só lado direito + debug-IO:
python lerobot-ext/init_lerobot_record.py \
    --config_path lerobot-ext/config/record/record_televuer.yaml \
    --sim --quest-ip 192.168.68.61 --quest-adb-ip=192.168.68.51 \
    --left-arm-limp --debug
```

---

## 6. Controles (Quest)

Detecção por **borda de subida** (clique, não segurar).

| Botão | Controle | Ação |
|---|---|---|
| **X** | esquerdo | Travar / destravar o seguimento (toggle) |
| **Y** | esquerdo | Encerrar a sessão / gravação |
| **A** | direito | Salvar episódio e começar o próximo |
| **B** | direito | Descartar episódio e regravar |
| **Grip** (squeeze) | esq/dir | Fechar a mão inteira (`input_mode: controller`) |
| **Gatilho** (trigger) | esq/dir | Pinça (`input_mode: controller`) |

**Grip latch:** ao apertar `A`/`B` com a garra direita fechada (squeeze > 0,2), ela **trava** na posição atual e segura o objeto durante o save e o próximo episódio. Para retomar o controle: **solte e re-aperte** o grip. Logs: `🔒 [GRIP LATCH] travada` / `🔓 [GRIP LATCH] Destravado`.

**`input_mode`:**
- `"controller"` (atual): começa **TRAVADO**; X destrava após ~3 s de countdown.
- `"hand"` (hand tracking): começa **DESTRAVADO**.

**Clutch:** ao destravar, o robô ancora a pose atual e segue apenas o **delta** do controle — não salta para a pose absoluta do headset.

### Comandos por voz (pt-BR) e teclado

| Voz | Teclado | Ação |
|---|---|---|
| "salvar" / "gravar" / "próximo" | — | salva episódio, grava o próximo |
| "errei" / "voltar" / "reboot" | — | descarta e regrava |
| "finalizar" / "sair" / "fechar" | — | encerra |
| "pausar" / "congelar" / "travar" | Seta ↓ | modo estátua |
| "continuar" / "destravar" / "play" | Seta ↑ | retoma |
| — | Espaço / `p` | alterna pausa |
