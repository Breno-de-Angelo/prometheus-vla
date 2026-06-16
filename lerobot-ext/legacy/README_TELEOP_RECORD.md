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
    --config_path lerobot-ext/config/record/record_televuer.yaml \
    --sim \
    --quest-ip 192.168.68.61 \
    --quest-adb-ip=192.168.68.51
```

`--sim` apenas liga `--robot.is_simulation=true` e `--teleop.is_simulation=true`. Sobe o MuJoCo
(`unitree-g1-mujoco`), o servidor Vuer VR na **:8012** e o publicador de imagens ZMQ na **:5555**.
Acesse pelo Quest: `https://<ip-do-laptop>:8012/?ws=wss://<ip-do-laptop>:8012`.

> ⚠️ **Passe o `--quest-ip <ip-do-laptop>`** (ver §Flags). Sem ele o script **não abre/recarrega** a
> página do Vuer no browser do Quest — e se o headset não estiver com uma **sessão VR ativa** apontando
> pro Vuer, o teleop **não recebe dado de controle**: você aperta X/B/A/gatilho/move a mão e **nada
> responde**. Esse é o motivo nº 1 de "os controles pararam de funcionar".

- **As mãos Dex3 respondem no sim** (classe `lerobot-ext`, atual): o `connect()` sobe a ponte ZMQ↔DDS
  (`ponte_mao.py`) e o teleop controla os dedos. **Por padrão as DUAS mãos são controladas** (esquerda
  igual à direita); com `--left-arm-limp` a esquerda inteira (braço + mão) fica solta. ⚠️ Em MuJoCo,
  `kp=0` **não** deixa a junta visivelmente caída (atrito das juntas a segura) — no sim o efeito é a
  esquerda **parar de seguir o controle**; o "cair mole" de verdade só no robô real.
- **A pressão tátil NÃO existe no sim** — `left/right_hand_pressure` saem zeradas; o tátil real só vem do
  robô (porta ZMQ 6002). Não grave dataset de fusão tátil pelo sim.
- **A depth do sim NÃO é depth16** (ver §9). Use o sim só pra validar movimento/teleop, nunca pra dataset de profundidade.

---

## 5.1 Flags da linha de comando (CLI)

Flags **extras** do `init_lerobot_record_v2.py` (extraídas e tratadas no `__main__` **antes** de chamar o
`lerobot_record`; aceitam tanto `--flag valor` quanto `--flag=valor`). Tudo o que não está aqui é repassado
ao draccus (ex.: `--dataset.root=`, `--config_path`).

| Flag | Vale p/ | O que faz | Default |
|---|---|---|---|
| `--config_path <yaml>` | real + sim | Config draccus do record (obrigatório). | — |
| `--sim` | — | Liga simulação (MuJoCo): seta `--robot.is_simulation=true` e `--teleop.is_simulation=true`. Sem ela = robô real. | off (robô real) |
| `--quest-ip <ip>` | real + sim | IP do **laptop/servidor Vuer**. O script usa pra abrir o Vuer no browser do Quest via ADB (`https://<ip>:8012/...`). **Sem ele, o browser do Quest não é aberto** → sem sessão VR ativa, o teleop não recebe controle (ver §5). | não abre o browser |
| `--quest-adb-ip <ip>` | real + sim | IP do **próprio headset Quest** na rede (muda por DHCP). Usado p/ ADB: manter o headset acordado (`prox_close`) e abrir o browser. **Não** é o `--quest-ip`. | `192.168.68.51:5555` |
| `--left-arm-limp` | real + sim | **Lado esquerdo INTEIRO solto** (`kp=0`): braço (juntas 15–21) **e** mão Dex3 (7 motores). A esquerda para de seguir o controle; só o lado **direito** é teleoperado. Sem a flag, as **duas** mãos/braços são controlados normalmente. (env `G1_LEFT_ARM_LIMP=1`) | off (esquerda ativa) |
| `--debug` | real + sim | Grava **tudo** enviado/recebido do robô (send/recv) num JSONL em `/tmp/g1_debug_io_<timestamp>.jsonl`. Analise com `python lerobot-ext/analyze_debug_io.py <arquivo>`. (env `G1_DEBUG_IO=<path>`) | off |
| `--dataset.root=<dir>` | real + sim | Diretório fixo do dataset (repassado ao draccus). Sem ele → auto-timestamp `datasets/.../<YYYYmmdd_HHMMSS>/` (§7). | auto-timestamp |

**Mão/braço esquerdo — resumo do comportamento (importante):**
- **Sem `--left-arm-limp` (default):** mão **e** braço esquerdos controlados normalmente, **simétrico** ao
  lado direito (mesmo `kp`, segue o controle, segura a posição via heartbeat).
- **Com `--left-arm-limp`:** lado esquerdo inteiro com `kp=0` (mole) — braço (em `unitree_g1.py`) e mão
  Dex3 (`connect`/`heartbeat`/`send_action` em `unitree_g1_dex3.py`) param de seguir o controle. Use quando
  for treinar/gravar **só com o lado direito**.

**Exemplos:**
```bash
# Sim, as duas mãos ativas, abrindo o Vuer no Quest e mantendo o headset acordado:
python lerobot-ext/init_lerobot_record_v2.py \
    --config_path lerobot-ext/config/record/record_televuer.yaml \
    --sim --quest-ip 192.168.68.61 --quest-adb-ip=192.168.68.51

# Só lado direito (braço + mão esquerdos soltos) + gravando debug-IO:
python lerobot-ext/init_lerobot_record_v2.py \
    --config_path lerobot-ext/config/record/record_televuer.yaml \
    --sim --quest-ip 192.168.68.61 --quest-adb-ip=192.168.68.51 \
    --left-arm-limp --debug
```

> O guia de como habilitar o ADB-over-wifi no Quest (`adb tcpip 5555`, autorizar o PC, `prox_close` pra
> pendurar no pescoço) está em `_claude_notes/README_QUEST_PROXIMITY.md`.

---

## 6. Controles da teleop (Quest)

Lidos a cada frame em `xr_g1_arm.py`. Detecção por **borda de subida** (clique, não segurar).

| Botão | Controle | Ação |
|---|---|---|
| **X** | esquerdo | **Travar / destravar** o seguimento (toggle). Ao destravar, re-ancora a pose (clutch). |
| **Y** | esquerdo | **Encerrar** a sessão / gravação. |
| **A** | direito | **Salvar** o episódio e já começar a gravar o próximo (sem reset). Trava a garra direita (grip latch). |
| **B** | direito | **Descartar** o episódio e regravá-lo. Trava a garra direita (grip latch). |
| **Grip** (squeeze) | esq/dir | **Fechar a mão inteira** (grip completo). Só no `input_mode: controller`. |
| **Gatilho** (trigger) | esq/dir | **Pinça** (indicador + polegar). Só no `input_mode: controller`. |

**Grip latch (não solta o copo ao salvar):** ao apertar `A`/`B` com a garra direita fechada (squeeze>0,2),
ela **trava** na posição atual e segura o objeto através do save e do próximo episódio — porque pra alcançar
o `A` você relaxa o grip e a mão abriria. Para **retomar o controle ao vivo**: **solte e re-aperte** o grip
até o nível em que travou (tem que soltar primeiro; não destrava no mesmo instante do `A`). Logs:
`🔒 [GRIP LATCH] travada` / `🔓 [GRIP LATCH] Destravado`.

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
  video_encoding_batch_size: 1        # encoda após cada save (default/testado). batch>1 está BUGADO nesta versão (KeyError no save em lote) — ver §12
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
- **`video_encoding_batch_size` > 1 está BUGADO** nesta versão do LeRobot. No fim da gravação, o encode em
  lote crasha com `KeyError: 'videos/observation.images.head_camera/chunk_index'`
  (`lerobot_dataset.py:_save_episode_video`): a detecção de "resume" vê os episódios já gravados (sem
  coluna de vídeo, porque o encode foi adiado) e tenta ler a coluna inexistente → **o dataset inteiro do
  run sai incompleto/corrompido.** Use `video_encoding_batch_size: 1` (default): encoda após cada save; o
  heartbeat segura o robô durante a pausa (~1 s p/ episódios curtos), sem timeout.
- **Soltar o copo ao salvar (`A`).** A mão segue o squeeze **ao vivo** (`right_hand_q = squeeze * RIGHT_TARGET`).
  Pressionar `A` no controle direito costuma exigir relaxar o grip → squeeze cai a 0 → mão abriria → copo cairia.
  **Resolvido por DOIS mecanismos:**
  1. **GRIP LATCH** (`xr_g1_arm.py`): ao apertar `A`/`B` com a garra direita fechada (pico de squeeze recente
     >0,2), ela **trava** no comando atual e segura o copo através do save e do próximo episódio. Retoma o
     controle ao vivo quando você **solta e re-aperta** o grip até o nível travado (precisa soltar antes — não
     destrava no mesmo frame do `A`). Logs `🔒/🔓 [GRIP LATCH]`.
  2. **HEARTBEAT segura o COMANDO** (`unitree_g1_dex3.py:_heartbeat_worker`): durante a pausa de encode (~13s
     com `batch_size: 1`), o teleop não roda e o robô fica só no heartbeat. Ele re-publica o **q comandado**
     das mãos (não a posição medida) — porque re-semear da medida afrouxava a força do grip (o copo bloqueia
     os dedos → erro ~0 → torque ~0) e soltava o copo. Agora mantém a garra firme. Idem o corpo (via `self.msg`).
  Alternativa sem latch: **salvar por VOZ ("salvar")**, que não exige soltar o grip.

  ⚠️ A pausa de ~13s por save é o custo de `video_encoding_batch_size: 1` (encode síncrono). O robô fica
  **seguro e segurando o copo** durante ela (heartbeat). É também um bom momento pra reposicionar o copo.
- **SIM ≠ REAL na depth (importante):** no `--sim`, a depth do MuJoCo **não** sai como `uint16` mm PNG —
  ela é achatada para `uint8` 3 canais (≈ metros × 38, clip 0–255) e transportada como **JPEG lossy**.
  Além disso, `base_sim.py` define `update_render_caches` **duas vezes** e a segunda (a que o Python usa)
  **não** faz a conversão pra mm. Conclusão: **o sim serve pra teleop/controle, não pra gravar depth16**.
  Para alinhar o sim ao real seria preciso publicar a depth como `uint16` mm 1-canal via
  `encode_depth_image` (PNG) e remover o achatamento `uint8` — hoje isso não acontece.
