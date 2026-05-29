# Teleoperação VR + Gravação (record) com depth 16-bit lossless — G1 + Dex3

Guia de ponta a ponta para **teleoperar** o Unitree G1 (braços + mãos Dex3) por VR (Quest/Vuer)
e **gravar** datasets no formato LeRobot com a **profundidade preservada por inteiro**
(`PNG uint16` em milímetros, lossless) — o formato que o pipeline `pi05-D` (RGB + depth + tátil) espera.

> Este documento expande e substitui o `README_depth16.md` (que cobria só o lado da profundidade).
> Branch obrigatória: **`Luiz-pi05d`**. Ambiente conda: **`g1`**.

---

## 1. Por que depth 16-bit

O fluxo antigo salvava a profundidade como **vídeo H.264 8-bit (3 canais)** numa faixa de 0–2 m. Numa cena de
mesa isso colapsava tudo em ~8–12 dos 256 níveis → a depth virava peso morto na fusão multimodal.

O fluxo `Luiz-pi05d` salva a profundidade como **imagem `PNG uint16` de 1 canal, em milímetros**, lossless:
preserva a precisão métrica real da câmera. RGB (`head_camera`) e tátil (`left/right_hand_pressure`) seguem iguais.

| | Fluxo antigo | Fluxo `Luiz-pi05d` (este) |
|---|---|---|
| Feature da depth | `video` H.264 8-bit, 3 canais | `image` **PNG uint16, 1 canal** |
| Unidade | 0–2 m comprimido em 8-bit | **milímetros** (precisão métrica real) |
| Perda | lossy (vídeo + 8-bit) | **lossless** |
| Shape | (480, 640, 3) | **(480, 640, 1)** |

---

## 2. Topologia

A teleop e o record **rodam no laptop**; o robô roda os servidores de baixo nível e de câmera.

```
┌─────────────────────┐   VR (Vuer :8012)   ┌──────────────────┐
│  Quest (headset)    │ ◄─────────────────► │                  │
└─────────────────────┘                     │  LAPTOP (cliente)│
                                            │  conda env g1    │
                          ZMQ câmera :5555   │  branch Luiz-    │
┌─────────────────────┐ ◄─────────────────  │  pi05d           │
│  ROBÔ Unitree G1    │                     │  init_lerobot_   │
│  - realsense depth16│ ──── DDS ─────────► │  record_v2.py    │
│  - run_g1_server.py │   (motores/Dex3)    └──────────────────┘
└─────────────────────┘
```

Há **dois modos**:
- **Robô real** (sem `--sim`): câmera D435i + Dex3 reais. **É o único que grava depth16 de verdade.**
- **Simulação** (`--sim`): MuJoCo. Serve pra ensaiar o fluxo de teleop/controle — **mas a depth NÃO sai
  em depth16** (ver §9 — Armadilhas). Não use o sim pra gerar dataset de profundidade.

---

## 3. Pré-requisitos (laptop)

```bash
cd ~/I2CA/prometheus-vla
git branch --show-current            # tem que dizer: Luiz-pi05d
conda activate g1
```

Versões que o ambiente `g1` exige (**não atualizar cegamente**):
- `numpy<2` (1.26.4) — NumPy 2.x dá segfault no pinocchio
- `torch==2.3.0` + `torchvision==0.18.0` (casados)
- `dex_retargeting` **0.4.7** do diretório local `lerobot-ext/teleop/robot_control/dex-retargeting/`
- `unitree_sdk2py` do submódulo local `./unitree_sdk2_python/` (o do PyPI não traz `crc_amd64.so`)
- Opcionais p/ comandos por voz/teclado: `speech_recognition` (+ microfone + internet) e `pynput`

---

## 4. Rodar — ROBÔ REAL (grava depth16)

Suba **na ordem**: terminais 1 e 2 (robô) primeiro; o terminal 3 (laptop) **só depois** que câmera e
servidor do G1 estiverem no ar.

> ⚠️ Antes de subir a câmera, confirme que **nenhum outro `realsense_server` está rodando** no robô —
> a porta 5555 fica ocupada e a gravação pega a stream errada.

### Terminal 1 — Robô: câmera depth 16-bit

```bash
ssh unitree@<IP_DO_ROBO>                         # confirme o IP da sua rede
source ~/miniconda3/etc/profile.d/conda.sh && conda activate g1
cd ~/lerobot/src/lerobot/cameras/zmq
python realsense_server_depth16.py --serial 327122071538   # confirme o serial da SUA D435i
```

> O `realsense_server_depth16.py` que roda **no robô** é self-contained (faz o encode PNG inline).
> A versão de referência **dentro deste repo** é `lerobot-ext/Scripts_Prometheus_int/realsense_server.py`
> (mesma lógica: z16 → mm → `uint16` → PNG lossless → ZMQ :5555). **Não confunda** com o
> `realsense_server.py` da raiz do repo (antigo, sem depth) nem com `full_realsenser_server.py`.

### Terminal 2 — Robô: servidor de baixo nível do G1

```bash
ssh unitree@<IP_DO_ROBO>
source ~/miniconda3/etc/profile.d/conda.sh && conda activate g1
cd ~/lerobot/src/lerobot/robots/unitree_g1
python run_g1_server.py                          # espere "bridge running"
```

### Terminal 3 — Laptop: gravação

> ⚠️ Ao iniciar, o robô faz **SOFT START** (rampa de ~3 s, `WARMUP_DURATION_S=3.0`) interpolando da
> pose física atual até o comando — para não dar tranco. **Espere os 3 s antes de destravar (botão X).**
> Área livre, robô apoiado, E-stop ao alcance.

```bash
cd ~/I2CA/prometheus-vla/lerobot-ext
conda activate g1
python init_lerobot_record_v2.py --config_path config/record/record_televuer.yaml
```

Os dois lados **precisam casar**: `realsense_server_depth16` (robô) ↔ `init_lerobot_record_v2.py` da
branch `Luiz-pi05d` (laptop). Servidor uint8 padrão, ou record fora da branch, gera **depth corrompida ou
quebra por shape**.

No real, o IP do servidor de imagens default no código é `192.168.123.164` (`xr_g1_arm.py`, quando
`is_simulation=false`); se a sua rede for outra, ajuste o IP do robô / da câmera. O PC precisa estar na
mesma sub-rede do robô (ex.: `ip addr add 192.168.123.<x>/24 ...`).

---

## 5. Rodar — SIMULAÇÃO (ensaio de teleop; **não** grava depth16)

```bash
cd ~/I2CA/prometheus-vla
conda activate g1
python lerobot-ext/init_lerobot_record_v2.py \
    --config_path lerobot-ext/config/record/record_televuer.yaml --sim
```

`--sim` apenas liga `--robot.is_simulation=true` e `--teleop.is_simulation=true`. Sobe o MuJoCo
(`unitree-g1-mujoco`), o servidor Vuer VR na **:8012** e o publicador de imagens ZMQ na **:5555**.
Acesse pelo Quest: `https://<ip-do-laptop>:8012/?ws=wss://<ip-do-laptop>:8012`.

- **As mãos Dex3 não respondem no sim** — `connect()` pula a criação dos publishers DDS da mão; o
  `get_observation` preenche os joints da mão com 0.0. Mãos só no robô real.
- **A depth do sim NÃO é depth16** (ver §9). Use o sim só pra validar movimento/teleop, nunca pra dataset de profundidade.

---

## 6. Controles da teleop (Quest)

Lidos a cada frame em `xr_g1_arm.py`. Detecção por **borda de subida** (clique, não segurar).

| Botão | Controle | Ação |
|---|---|---|
| **X** | esquerdo | **Travar / destravar** o seguimento (toggle). Ao destravar, re-ancora a pose (clutch). |
| **Y** | esquerdo | **Encerrar** a sessão / gravação. |
| **A** | direito | **Salvar** o episódio e já começar a gravar o próximo (sem reset). |
| **B** | direito | **Descartar** o episódio e regravá-lo. |
| **Grip** (squeeze) | esq/dir | **Fechar a mão inteira** (grip completo). Só no `input_mode: controller`. |
| **Gatilho** (trigger) | esq/dir | **Pinça** (indicador + polegar). Só no `input_mode: controller`. |

**`input_mode` (ver config):**
- `"controller"` (atual): começa **TRAVADO**; o botão **X** destrava (depois de um countdown de ~3 s).
  Mão controlada por grip/gatilho.
- `"hand"` (hand tracking): começa **DESTRAVADO** (não há botão X físico); a segurança fica por conta da
  detecção das mãos no VR + countdown de 3 s.

**Clutch (ancoragem relativa):** ao destravar, o robô **não salta** para a pose absoluta do controle —
ele ancora a pose atual e segue apenas o **delta** do movimento do controle. Um **watchdog** re-ancora
sozinho se o loop congelar > 0,5 s (ex.: encoding de vídeo ao salvar), evitando tranco ao retomar.

### Comandos por voz (pt-BR, opcional) e teclado

Só funcionam com `speech_recognition`/microfone/internet (voz) e `pynput` (teclado). Sem essas libs, a
gravação base segue normal — só os atalhos param.

| Voz (pt-BR) | Teclado | Ação |
|---|---|---|
| "salvar" / "gravar" / "próximo" | — | salva episódio, grava o próximo |
| "errei" / "voltar" / "reboot" | — | descarta e regrava |
| "finalizar" / "sair" / "fechar" | — | encerra a gravação |
| "pausar" / "congelar" / "travar" | Seta ↓ | **modo estátua** (congela o robô na última pose) |
| "continuar" / "destravar" / "play" | Seta ↑ | retoma |
| — | Espaço / `p` | alterna pausa |

---

## 7. Config — `config/record/record_televuer.yaml`

```yaml
robot:
  type: unitree_g1_dex3
  control_mode: "upper_body"          # só tronco/braços/mãos
teleop:
  type: xr_g1_arm
  zmq: true
  input_mode: "controller"            # "hand" = hand tracking | "controller" = controles físicos do Quest
  display_mode: "ego"
  ee_type: "dex3"
dataset:
  repo_id: lewislf/G1_Dex3_depth_tactil_dataset
  root: "datasets/G1_Dex3_depth_tactil_dataset"   # base; vira root/<YYYYmmdd_HHMMSS> (auto-timestamp)
  push_to_hub: false                  # NÃO sobe pro HF Hub; fica só local
  single_task: "push the cup"         # ⚠️ é "push" (empurrar) — ajuste se a tarefa for "pick up"
  vcodec: "h264"                      # codec do RGB (a depth NÃO usa vídeo, vai como PNG)
  reset_time_s: 0                     # sem fase de reset entre episódios (salvou → grava o próximo)
  video_encoding_batch_size: 50       # adia o encoding pro fim, não trava o loop a cada save
play_sounds: false
```

> **Câmera, FPS, portas, IP do robô e `is_simulation` não estão neste YAML** — vêm de
> `lerobot-ext/robot/unitree_g1/config_unitree_g1.py` e do código do teleop (`xr_g1_arm.py`), ou da flag `--sim`.

**Onde o dataset é salvo:** se você **não** passar `--dataset.root`, o script cria automaticamente
`datasets/G1_Dex3_depth_tactil_dataset/<YYYYmmdd_HHMMSS>/`. Para escrever num diretório fixo (ou continuar
um), passe `--dataset.root=<dir>` explicitamente.

---

## 8. O que `init_lerobot_record_v2.py` faz por baixo

É um **wrapper de monkey-patches** em volta de `lerobot.scripts.lerobot_record.main`. Em ordem:

1. **Libera portas órfãs** 5555 (ZMQ câmera) e 8012 (Vuer VR) — mata processos que ficaram presos após
   `kill -9`/crash (`_free_stale_ports`, via `ss -tlnp`).
2. **Soft-start (anti-tranco):** rampa de 3 s interpolando da pose física real (lida de
   `robot._lowstate.motor_state[].q` e `_left/_right_hand_state`) até o comando. Reseta a cada episódio salvo.
3. **Tátil:** tira a pressão das mãos Dex3 (33-dim por mão) e a grava como
   `observation.left_hand_pressure` / `observation.right_hand_pressure` (float32, shape 33).
4. **Depth16 (3 patches encadeados):**
   - `hw_to_dataset_features` → `observation.images.head_camera_depth` vira `image` `(480,640,1)`, não vídeo.
   - `ZMQCamera.read` → decodifica a depth com `cv2.IMREAD_UNCHANGED` (preserva `uint16`).
   - `image_array_to_pil_image` → salva `uint16` como PNG modo **`I;16`**.
5. **Auto-timestamp** do `--dataset.root` (§7).

---

## 9. O formato depth16 — fim a fim

```
D435i z16 (mm)  ──►  raw * depth_scale * 1000  ──►  clip [0, 32767]  ──►  uint16
   (640×480)         (depth_scale ≈ 0.001 m/u)      (sobrevive ao int16)
        │
        ├─► encode PNG lossless (cv2.imencode('.png'))  ──► base64 ──► JSON ──► ZMQ PUB :5555
        │      (RGB vai como JPEG quality 80, separado)
        ▼
   LAPTOP: ZMQCamera.read (IMREAD_UNCHANGED → uint16 (H,W,1))
        ▼
   save PNG modo I;16  ──►  dataset (feature 'image', dtype uint16, mm)
        ▼
   TREINO: depth_to_pointcloud(depth, intrinsics, depth_scale=0.001)   # mm → metros (z = mm / 1000)
```

- **Por que clip em 32767 (e não 65535):** o `ToTensor`/PIL do LeRobot decodifica PNG `I;16` como
  `int16` **com sinal** → valores > 32767 dariam overflow/negativo. 32767 mm ≈ 32 m, muito além do workspace.
- **Escala:** o z16 da D435i já vem em mm (`depth_scale` ≈ 0.001 m/unidade). O `depth_scale` aparece
  **duas vezes**: no servidor (m→mm) e no treino (mm→m, hardcoded `0.001` em `pi05_d_injector.py`).

---

## 10. Validar a gravação

Depois de gravar um episódio, a feature de depth tem que estar como **`image`** (não `video`):

```bash
python -c "import json,glob; f=sorted(glob.glob('datasets/G1_Dex3_depth_tactil_dataset/*/meta/info.json'))[-1]; print(json.load(open(f))['features']['observation.images.head_camera_depth'])"
# esperado: dtype 'image', shape [480, 640, 1]
```

Um PNG de depth aberto com PIL deve ter modo **`I;16`** (16-bit) e valores em **milímetros**.

---

## 11. Treino (Atena `hercules@10.9.8.252`)

O treino lê a depth em mm e converte pra metros na back-projection: `pi05_d_injector` chama
`depth_to_pointcloud(..., depth_scale=0.001)`. Config: `config/train/train_cup_depth_tactil_pi05_full.yaml`.
Os patches de treino (`pi05_d_injector` + config) precisam estar na Atena antes de treinar.

---

## 12. Armadilhas

- **Parar com Ctrl+C (SIGINT), não `kill -9`.** `kill -9` deixa subprocessos órfãos segurando 5555/8012 →
  próxima run dá `Address already in use`. O script tenta auto-liberar essas portas no início, mas o
  `_free_stale_ports` mata **qualquer** processo nessas portas (cuidado com processos legítimos ali).
- **Servidor + branch têm que casar.** depth16 no robô ↔ `init_record` da `Luiz-pi05d`. Datasets gravados
  em 8-bit são **incompatíveis** com este fluxo — regrave em 16-bit.
- **Resolução fixa 480×640.** O patch da feature assume esse shape; resolução diferente quebra o Parquet.
- **`single_task: "push the cup"`** — é "empurrar", não "pegar". Confirme se bate com o que você quer treinar.
- **Reset do copo é manual.** Com `reset_time_s: 0` não há fase de reset entre episódios; reposicione o
  copo no home **antes** de cada sequência, fora do fluxo automático.
- **SIM ≠ REAL na depth (importante):** no `--sim`, a depth do MuJoCo **não** sai como `uint16` mm PNG —
  ela é achatada para `uint8` 3 canais (≈ metros × 38, clip 0–255) e transportada como **JPEG lossy**.
  Além disso, `base_sim.py` define `update_render_caches` **duas vezes** e a segunda (a que o Python usa)
  **não** faz a conversão pra mm. Conclusão: **o sim serve pra teleop/controle, não pra gravar depth16**.
  Para alinhar o sim ao real seria preciso publicar a depth como `uint16` mm 1-canal via
  `encode_depth_image` (PNG) e remover o achatamento `uint8` — hoje isso não acontece.
