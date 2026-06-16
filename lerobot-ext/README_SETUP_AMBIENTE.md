# Setup do Ambiente — Prometheus VLA (G1 + Dex3)

Guia completo para uma pessoa nova configurar o ambiente do zero e rodar o script de gravação de dados (`init_lerobot_record.py`) no **Unitree G1** com mãos **Dex3**.

---

## Opção A — Instalação automática (recomendado)

Rode o script a partir da **raiz do repositório**:

```bash
cd ~/I2CA/prometheus-vla
bash setup_env.sh           # cria o env "g1"
# ou
bash setup_env.sh g1_env    # cria com nome personalizado
```

O script faz tudo: cria o env conda, instala PyTorch, LeRobot, SDK da Unitree e todas as dependências. Ao final imprime o comando de run. Se tiver algum erro, veja a **Opção B** abaixo para fazer manualmente e identificar o passo que falhou.

---

## Opção B — Instalação manual (passo a passo)

Siga as seções abaixo se quiser controle total ou se o script automático falhar em algum passo.

---

## Pré-requisitos de sistema

- Ubuntu 22.04 ou 24.04 (testado em ambos)
- GPU NVIDIA com CUDA 12.1+ (para treino; o record roda sem GPU)
- Drivers NVIDIA instalados (`nvidia-smi` funciona)
- `git`, `adb` (Android Debug Bridge) disponíveis no PATH

```bash
# Instalar adb se não tiver
sudo apt install android-tools-adb
```

---

## 1. Instalar Miniconda (se não tiver)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Siga as instruções; reinicie o terminal depois
```

---

## 2. Clonar o repositório com submodulos

```bash
git clone --recurse-submodules <URL_DO_REPO> ~/I2CA/prometheus-vla
cd ~/I2CA/prometheus-vla

# Se já clonou sem --recurse-submodules:
git submodule update --init --recursive
```

> Certifique-se de estar na branch correta para gravação de dados:
> ```bash
> git checkout Luiz-pi05d
> ```

---

## 3. Criar o ambiente conda

O ambiente usa **Python 3.10** e algumas dependências científicas pesadas via conda-forge (pinocchio, casadi, proxsuite). **Não usar Python 3.11+** — o pinocchio não tem wheel para versões mais novas.

```bash
conda create -n g1 python=3.10 -c conda-forge -y
conda activate g1
```

### 3.1 Instalar pacotes conda-forge (IK, cinemática)

```bash
conda install -c conda-forge \
    "numpy=1.26.4" \
    "pinocchio=3.1.0" \
    "casadi=3.6.7" \
    "proxsuite=0.7.2" \
    -y
```

> **Por que numpy<2?** O pinocchio 3.1 linka contra NumPy 1.x. NumPy 2.x causa segfault na chamada do IK.

---

## 4. Instalar PyTorch com CUDA 12.1

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

> Se a sua máquina não tiver GPU (só CPU), use:
> ```bash
> pip install torch==2.3.0 torchvision==0.18.0
> ```

---

## 5. Instalar o LeRobot (submódulo local, editable)

O LeRobot usado neste projeto é uma versão **customizada** — está no submódulo `./lerobot`. Instale como editable para que as modificações locais tenham efeito:

```bash
cd ~/I2CA/prometheus-vla
pip install -e "./lerobot[unitree_g1_dex3,televuer,intelrealsense,pi]"
```

Isso instala o LeRobot e seus extras, incluindo `televuer` e `teleimager` (VR e câmera).

---

## 6. Instalar o dex-retargeting (local, editable)

O retargeting das mãos Dex3 é um pacote local dentro do repo:

```bash
pip install -e "lerobot-ext/teleop/robot_control/dex-retargeting"
```

---

## 7. Instalar o SDK da Unitree (submódulo local)

O SDK da Unitree deve ser instalado **do submódulo local** — a versão do PyPI não inclui o `crc_amd64.so` necessário para o robô real:

```bash
pip install ./unitree_sdk2_python
```

---

## 8. Instalar dependências restantes via pip

```bash
pip install \
    diffusers==0.30.0 \
    datasets==4.1.0 \
    draccus==0.10.0 \
    pyzmq \
    opencv-python \
    pyrealsense2 \
    mujoco==3.6.0 \
    av \
    cmake \
    flask \
    pyyaml \
    scipy \
    pandas \
    wandb \
    huggingface-hub \
    accelerate \
    einops \
    safetensors \
    transformers \
    rerun-sdk \
    pynput \
    evdev \
    pyserial \
    pyaudio \
    speechrecognition \
    omniview
```

---

## 9. Verificar a instalação

```bash
python -c "
import torch, lerobot, mujoco, cv2, zmq, numpy as np
from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3
print('torch:', torch.__version__)
print('numpy:', np.__version__)
print('lerobot:', lerobot.__version__)
print('mujoco:', mujoco.__version__)
print('Tudo OK!')
"
```

---

## 10. Rodar o script de gravação

### Modo simulação (sem robô físico — para testar o fluxo)

```bash
cd ~/I2CA/prometheus-vla
conda activate g1

python lerobot-ext/init_lerobot_record.py \
    --config_path lerobot-ext/config/record/record_televuer.yaml \
    --sim \
    --quest-ip 192.168.68.61 \
    --quest-adb-ip 192.168.68.51
```

> Substitua `192.168.68.61` pelo IP **deste laptop** na rede Wi-Fi.
> Substitua `192.168.68.51` pelo IP do **headset Quest** (varia por DHCP).
>
> Para descobrir o IP do laptop: `hostname -I`

### Modo robô real (grava depth 16-bit de verdade)

Antes de rodar no laptop, suba dois servidores **no robô** (em dois terminais SSH separados):

**Terminal 1 — robô (câmera depth 16-bit):**
```bash
ssh unitree@192.168.123.164
conda activate g1
cd ~/lerobot/src/lerobot/cameras/zmq
python realsense_server_depth16.py --serial <SERIAL_DA_CAMERA>
```

**Terminal 2 — robô (servidor de controle G1):**
```bash
ssh unitree@192.168.123.164
conda activate g1
cd ~/lerobot/src/lerobot/robots/unitree_g1
python run_g1_server.py
```

**Terminal 3 — laptop (gravação):**
```bash
cd ~/I2CA/prometheus-vla
conda activate g1

python lerobot-ext/init_lerobot_record.py \
    --config_path lerobot-ext/config/record/record_televuer.yaml \
    --quest-ip 192.168.68.61 \
    --quest-adb-ip 192.168.68.51
```

---

## 11. Flags disponíveis

| Flag | Descrição | Default |
|------|-----------|---------|
| `--config_path <yaml>` | Config do record (obrigatório) | — |
| `--sim` | Usa simulador MuJoCo em vez do robô real | desligado |
| `--quest-ip <ip>` | IP do laptop (para abrir o Vuer no Quest via ADB) | não abre browser |
| `--quest-adb-ip <ip>` | IP do próprio headset Quest na rede | `192.168.68.51` |
| `--left-arm-limp` | Braço e mão esquerda soltos (kp=0) — grava só lado direito | desligado |
| `--debug` | Grava JSONL de envio/recebimento do robô em `/tmp/` | desligado |
| `--dataset.root=<dir>` | Diretório fixo do dataset (sem ele → auto-timestamp) | auto |

---

## 12. Controles do Quest durante a gravação

| Botão | Ação |
|-------|------|
| **X** (esquerdo) | Travar / destravar o seguimento do braço |
| **Y** (esquerdo) | Encerrar a sessão |
| **A** (direito) | Salvar episódio e começar o próximo |
| **B** (direito) | Descartar episódio e regravar |
| **Grip** (squeeze) | Fechar a mão inteira |
| **Gatilho** (trigger) | Pinça (indicador + polegar) |

> Ao iniciar, o robô faz um **soft-start de ~3 s** (rampa da pose atual até a neutra).
> **Aguarde os 3 s antes de apertar X** para destravar.

---

## 13. Onde os datasets ficam salvos

Por padrão, cada run cria uma subpasta com timestamp automático:

```
datasets/G1_Dex3_depth_tactil_dataset/
└── 20260609_143022/        ← timestamp da run
    ├── data/
    ├── meta/
    └── videos/
```

Para salvar num diretório fixo (ex.: para continuar um dataset existente):
```bash
python lerobot-ext/init_lerobot_record.py \
    --config_path lerobot-ext/config/record/record_televuer.yaml \
    --dataset.root=datasets/meu_dataset_fixo \
    --sim
```

---

## 14. Problemas comuns

**`Address already in use` nas portas 5555/8012:**
O script tenta liberar automaticamente no início. Se falhar, mate manualmente:
```bash
kill $(lsof -t -i:5555) 2>/dev/null; kill $(lsof -t -i:8012) 2>/dev/null
```

**`segfault` ao importar pinocchio:**
NumPy está na versão 2.x. Rebaixe:
```bash
pip install "numpy==1.26.4"
```

**`crc_amd64.so` não encontrado (unitree_sdk2py):**
O SDK foi instalado do PyPI em vez do submódulo local. Reinstale:
```bash
pip uninstall unitree-sdk2py -y
pip install ./unitree_sdk2_python
```

**Quest não responde aos controles:**
O browser do Quest não está apontando pro Vuer. Passe `--quest-ip <IP_DO_LAPTOP>` para o script abrir automaticamente via ADB. Veja o guia completo em `_claude_notes/README_QUEST_PROXIMITY.md`.

**Encerre sempre com Ctrl+C**, nunca com `kill -9` — o segundo deixa processos órfãos nas portas.
