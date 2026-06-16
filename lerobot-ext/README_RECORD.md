# Gravação de dataset — G1 + Dex3 (depth 16-bit lossless)

## ⚡ Comandos rápidos

```bash
# ROBÔ (ssh)
ssh unitree@192.168.123.164
cd ~/prometheus-vla && bash lerobot-ext/start_robot.sh        # espere os dois "✅ ATIVO"
```

```bash
# NOTEBOOK
cd ~/I2CA/prometheus-vla/lerobot-ext
python init_lerobot_record.py --config_path config/record/record_televuer.yaml --left-arm-limp   # só braço direito
```

---

Guia para **gravar datasets** no formato LeRobot com profundidade preservada em inteiro (`PNG uint16` em milímetros, lossless) — o formato que o pipeline pi05 (RGB + depth + tátil) espera.

> Branch obrigatória: **`Luiz-pi05d`**. Ambiente conda: **`g1`**.
> Para subir os servidores no robô e configurar a teleop, veja `README_TELEOP.md`.

---

## 1. Por que depth 16-bit

O fluxo antigo salvava a profundidade como vídeo H.264 8-bit — numa cena de mesa isso colapsava tudo em ~8–12 dos 256 níveis. O fluxo atual salva como **PNG uint16 de 1 canal, em milímetros**, lossless.

| | Fluxo antigo | Fluxo atual |
|---|---|---|
| Feature da depth | `video` H.264 8-bit, 3 canais | `image` **PNG uint16, 1 canal** |
| Unidade | 0–2 m comprimido em 8-bit | **milímetros** |
| Perda | lossy | **lossless** |
| Shape | (480, 848, 3) | **(480, 848, 1)** |

> ⚠️ **Robô real** é o único que grava depth16 de verdade. O `--sim` não produz depth16 (ver §Armadilhas).

---

## 2. Rodar — robô real

> ⚠️ Antes de gravar confirme que nenhum outro `realsense_server` está rodando no robô — a porta 5555 fica ocupada.

> ⚠️ O robô faz **soft start** (~3 s de rampa) ao iniciar. Espere os 3 s antes de destravar (botão X). Área livre, robô apoiado, E-stop ao alcance.

### Terminal 1 — Robô: servidores de câmera + controle

Um comando só sobe os dois serviços do robô (ponte DDS↔ZMQ + câmera):

```bash
ssh unitree@192.168.123.164   # rede cabeada direta notebook ↔ robô
cd ~/prometheus-vla
bash lerobot-ext/start_robot.sh                            # espere os dois "✅ ATIVO"
```

O `start_robot.sh` sobe o `Scripts_Prometheus_int/realsense_server.py` — que **já grava depth16** (uint16 em mm, 1 canal; ver §5) — junto com o `run_g1_server.py`.

> Alternativa manual (dois terminais), se quiser separar os serviços. Ative o env `g1` em cada um:
> ```bash
> # câmera depth16
> cd ~/prometheus-vla/lerobot-ext/Scripts_Prometheus_int && python realsense_server.py
> # ponte DDS↔ZMQ (espere "bridge running")
> cd ~/prometheus-vla/lerobot-ext/robot/unitree_g1 && python run_g1_server.py
> ```

### Terminal 2 — Laptop: gravação

```bash
cd ~/I2CA/prometheus-vla/lerobot-ext
python init_lerobot_record.py --config_path config/record/record_televuer.yaml --left-arm-limp
```

> **`--left-arm-limp`** (opcional): solta o braço esquerdo INTEIRO (kp=0 nas juntas 15-21 + mão Dex3) — ele não segue o controle, você empurra fisicamente pra fora do quadro e teleopera **só o lado direito**. Sem a flag, os dois braços são controlados.
>
> O script **ativa o env `g1` sozinho** (se re-executa no Python do env) — não precisa de `conda activate g1` antes. Override do env via `G1_CONDA_ENV`; se a auto-ativação falhar, ative manualmente.
>
> Antes de gravar, ele roda um **pré-flight** (robô, câmera RGB, depth, Quest via USB, depuração WiFi, sensor de proximidade) e **aborta se algo falhar**. Use `--dry-preflight` pra só checar o setup sem gravar. O log verboso da run vai pra `run.log` dentro da pasta do dataset.

Os dois lados **precisam casar**: o `realsense_server.py` (depth16, no robô) ↔ `init_lerobot_record.py` da branch `Luiz-pi05d`. Servidor que não mande uint16, ou record fora da branch, gera **depth corrompida ou quebra por shape**.

---

## 3. Config — `config/record/record_televuer.yaml`

```yaml
robot:
  type: unitree_g1_dex3
  control_mode: "upper_body"
teleop:
  type: xr_g1_arm
  zmq: true
  input_mode: "controller"            # "hand" = hand tracking | "controller" = controles físicos
  display_mode: "ego"
  ee_type: "dex3"
dataset:
  repo_id: lewislf/G1_Dex3_depth_tactil_dataset
  root: "datasets/G1_Dex3_depth_tactil_dataset"   # vira root/<YYYYmmdd_HHMMSS> (auto-timestamp)
  push_to_hub: false
  single_task: "push the cup"         # ⚠️ é "push" (empurrar) — ajuste se a tarefa for "pick up"
  vcodec: "h264"
  reset_time_s: 0                     # sem fase de reset entre episódios
  video_encoding_batch_size: 1        # batch>1 está BUGADO nesta versão (ver §Armadilhas)
play_sounds: false
```

> IP do robô, FPS e portas vêm de `robot/unitree_g1/config_unitree_g1.py` e de `xr_g1_arm.py`.

**Onde o dataset é salvo:** sem `--dataset.root`, o script cria `datasets/G1_Dex3_depth_tactil_dataset/<YYYYmmdd_HHMMSS>/` automaticamente. Para continuar um dataset existente, passe `--dataset.root=<dir>`.

---

## 4. O que `init_lerobot_record.py` faz por baixo

Wrapper de monkey-patches em volta de `lerobot.scripts.lerobot_record.main`:

1. **Libera portas órfãs** 5555 e 8012 antes de subir.
2. **Soft-start:** rampa de 3 s da pose física real até o comando. Reseta a cada episódio salvo.
3. **Tátil:** grava pressão das mãos Dex3 (33-dim por mão) como `observation.left/right_hand_pressure`.
4. **Depth16 (3 patches):**
   - `hw_to_dataset_features` → depth vira `image` `(480,848,1)`, não vídeo.
   - `ZMQCamera.read` → decodifica com `cv2.IMREAD_UNCHANGED` (preserva `uint16`).
   - `image_array_to_pil_image` → salva como PNG modo `I;16`.
5. **Auto-timestamp** do `--dataset.root`.

---

## 5. Formato depth16 — fim a fim

```
D435i z16 (mm)  ──►  raw * depth_scale * 1000  ──►  clip [0, 32767]  ──►  uint16
   (848×480)         (depth_scale ≈ 0.001 m/u)      (sobrevive ao int16)
        │
        ├─► encode PNG lossless  ──► base64 ──► JSON ──► ZMQ PUB :5555
        ▼
   LAPTOP: ZMQCamera.read (IMREAD_UNCHANGED → uint16 (H,W,1))
        ▼
   save PNG modo I;16  ──►  dataset (feature 'image', dtype uint16, mm)
        ▼
   TREINO: depth_to_pointcloud(depth, intrinsics, depth_scale=0.001)   # mm → metros
```

- **Clip em 32767:** o LeRobot decodifica PNG `I;16` como `int16` com sinal → valores > 32767 dariam overflow. 32767 mm ≈ 32 m, além do workspace.

---

## 6. Validar a gravação

A feature de depth tem que aparecer como **`image`** (não `video`):

```bash
python -c "import json,glob; f=sorted(glob.glob('datasets/G1_Dex3_depth_tactil_dataset/*/meta/info.json'))[-1]; print(json.load(open(f))['features']['observation.images.head_camera_depth'])"
# esperado: dtype 'image', shape [480, 848, 1]
```

---

## 7. Pós-gravação — fatiar para treino (14-dim braço direito)

```bash
python lerobot-ext/tools/slice_right_arm_only.py \
    datasets/G1_Dex3_depth_tactil_dataset/<timestamp>
```

Gera `datasets/G1_Dex3_right14_dataset/<timestamp>`. Atualize `dataset.root` no config de treino.

---

## 8. Armadilhas

- **Ctrl+C, não `kill -9`.** `kill -9` deixa órfãos segurando 5555/8012.
- **Servidor + branch têm que casar.** depth16 no robô ↔ `init_lerobot_record.py` da `Luiz-pi05d`.
- **Resolução fixa 848×480.** Resolução diferente quebra o Parquet.
- **`single_task: "push the cup"`** — confirme se bate com a tarefa que está gravando.
- **Reset do copo é manual.** Com `reset_time_s: 0` não há fase de reset automático.
- **`video_encoding_batch_size > 1` está BUGADO.** Crasha com `KeyError: 'videos/.../chunk_index'` no fim da gravação, corrompendo o dataset inteiro. Use `1` (default).
- **Soltar o copo ao salvar (`A`):** resolvido por dois mecanismos:
  1. **Grip latch** (`xr_g1_arm.py`): trava a garra ao apertar `A`/`B` com squeeze > 0,2.
  2. **Heartbeat segura o comando** (`unitree_g1_dex3.py`): durante a pausa de encode (~13 s), re-publica o `q` comandado (não a posição medida), mantendo a força do grip.
  - Alternativa: **salvar por voz** ("salvar"), que não exige soltar o grip.
- **SIM ≠ REAL na depth.** No `--sim`, a depth do MuJoCo sai como `uint8` 3 canais (JPEG lossy) — não é depth16. Use o sim só para validar movimento/teleop.

---

## ⚡ Comandos rápidos

```bash
# ROBÔ (ssh)
ssh unitree@192.168.123.164
cd ~/prometheus-vla && bash lerobot-ext/start_robot.sh        # espere os dois "✅ ATIVO"
```

```bash
# NOTEBOOK
cd ~/I2CA/prometheus-vla/lerobot-ext
python init_lerobot_record.py --config_path config/record/record_televuer.yaml --left-arm-limp   # só braço direito
```
