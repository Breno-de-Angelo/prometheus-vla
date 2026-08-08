# Simulador no seu PC, inferência na atena

> O MuJoCo roda na sua máquina (com janela, para você ver o robô se mexer) e o
> modelo de 7B roda na atena, que tem as A100. Os dois conversam por ZMQ.

Serve para testar se o modelo aprendeu **sem precisar do robô físico** — e sem
precisar de uma GPU grande no seu PC.

---

## 1. Arquitetura

```
     SEU PC                                        ATENA
┌──────────────────┐                        ┌──────────────────┐
│  run_sim.py      │  câmeras   :5555  ───▶ │                  │
│  (MuJoCo+janela) │                        │  init_lerobot_   │
│                  │  lowstate  :6001  ───▶ │  inference_v3.py │
│  run_sim_remote  │  handstate :6002  ───▶ │                  │
│  (ponte DDS↔ZMQ) │                        │  OpenVLA-7B      │
│                  │ ◀──  :6000  lowcmd     │  na GPU          │
│                  │ ◀──  :6003  handcmd    │                  │
└──────────────────┘                        └──────────────────┘
```

São **dois processos no seu PC**: o simulador e a ponte. O simulador fala DDS
localmente (`INTERFACE: "lo"`); a ponte traduz esse DDS para ZMQ na rede.

Do lado da atena nada sabe que é simulação — `remote_sim_ip` faz o
`UnitreeG1Dex3` apontar tanto o canal DDS quanto as câmeras para o seu PC
(`unitree_g1.py:220-225`, `unitree_g1_dex3.py:89-90`).

---

## 2. No seu PC

```bash
cd unitree-g1-mujoco

# terminal 1 — simulador com janela e câmeras publicando
python run_sim.py

# terminal 2 — ponte DDS ↔ ZMQ
python run_sim_remote.py
```

O `run_sim.py` já publica as três câmeras por padrão:

```
📷 Cameras: head_camera 640x480, head_camera_depth 640x480,
            right_wrist_camera 224x224 → ZMQ port 5555
```

Descubra o IP do seu PC na rede (é ele que vai na atena):

```bash
ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v 127.0.0.1
```

E confirme que as portas estão acessíveis de fora:

```bash
ss -tlnp | grep -E '5555|600[0-3]'
```

Se houver firewall, libere 5555 e 6000-6003.

---

## 3. Na atena

```bash
CUDA_VISIBLE_DEVICES=0 python init_lerobot_inference_v3.py \
    --checkpoint=<CAMINHO>/best_val_checkpoint/pretrained_model \
    --task="pick up the white mug and place it to the right" \
    --sim \
    --remote-sim=<IP_DO_SEU_PC> \
    --cam-robot=<IP_DO_SEU_PC> \
    --v
```

`--remote-sim` liga o DDS; `--cam-robot` liga as câmeras. Os dois apontam para o
mesmo IP porque os dois serviços rodam no seu PC.

Antes de rodar, confira qual GPU está livre — as três são compartilhadas:

```bash
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
```

---

## 4. Por que testar aqui e não direto no robô

O simulador dá o que o robô real não dá: **mesma cena inicial toda vez**. Isso é
o que torna possível o teste de condicionamento por linguagem
([INFERENCIA_COMANDO_TEXTO.md](INFERENCIA_COMANDO_TEXTO.md), seção 4) — rodar dois
comandos diferentes a partir de um estado idêntico e comparar as trajetórias. No
robô real a caneca nunca está exatamente no mesmo lugar, então trajetórias
diferentes não provam nada.

E se o modelo fizer algo perigoso, no simulador não quebra nada.

---

## 5. Diferenças entre sim e robô real

Estas ainda existem e importam:

| | MuJoCo | robô real |
|---|---|---|
| `head_camera` | 640×480 | 848×480 |
| `head_camera_depth` | 640×480 | 848×480 |
| `right_wrist_camera` | 224×224 | 224×224 ✓ |
| intrínsecos | fovy 58° do MJCF | D435i, medir com `print_camera_intrinsics.py` |

> **Datasets de sim e de robô real não são combináveis** enquanto as resoluções
> da cabeça diferirem — `aggregate.py:47` compara o dicionário de features
> inteiro. Para co-treinar os dois mundos, alinhe as resoluções primeiro.
>
> A câmera de pulso já nasce alinhada (224×224 dos dois lados), de propósito.

Para **inferência** a diferença é menos grave: o RGB é redimensionado para
224×224 de qualquer jeito. Mas o **depth** entra em resolução nativa na nuvem de
pontos, então os intrínsecos do YAML precisam corresponder à origem que você está
usando — sim ou real, não os dois ao mesmo tempo.

---

## 6. Tronco e teclado, sem VR

Enquanto o VR não volta, o tronco é controlado pelo teclado
(`teleop/unitree_g1/keyboard_g1_arm.py`):

```
TRONCO      , gira p/ esquerda    . gira p/ direita
```

Passo de `waist_speed=0.01` rad por frame, limitado a `±waist_yaw_limit`. A tecla
**integra uma posição** em vez de comandar velocidade — o dataset guarda posições
de junta e o modelo prevê posições; mapear a tecla direto para velocidade
gravaria um alvo que não corresponde ao que foi executado.

O tronco só se move de fato com `use_waist_yaw=True` no config do robô, e no
MuJoCo só com `enable_waist: True` no `config.yaml` — ver
[SCHEMA_G1_V2.md](SCHEMA_G1_V2.md), seção 4.

---

## 6.5 GPU: Intel vs NVIDIA, e o conflito SDL × GLFW

Em notebook híbrido o MuJoCo cai na **Intel integrada**. O ganho de mudar é
grande — medido aqui, render offscreen 640×480 no cenário completo (teclado +
janela do MuJoCo):

| GPU | fps |
|---|---|
| Intel (Mesa ARL) | 57 |
| NVIDIA RTX 5070 | **1141** |

### Como ligar

```bash
PROMETHEUS_FORCE_NVIDIA=1 python init_lerobot_teleoparate_v2.py \
    --config_path=config/teleop/teleop_key_v2.yaml --sim
```

Conferir qual GPU está ativa: `glxinfo -B | grep "OpenGL renderer"`.

### O conflito que travava a janela (já corrigido)

Ligar a NVIDIA parecia quebrar tudo:

```
GLFWError: (65544) b'EGL: Failed to make context current'
ERROR: could not create window
```

Mas a NVIDIA não era a causa. Numa sessão **Wayland**, o SDL do pygame (janela do
teclado) escolhe **x11/XWayland** por padrão, enquanto o visualizador do MuJoCo
(GLFW) usa **Wayland nativo**. Os dois disputam EGL e a janela do MuJoCo morre no
nascimento. A matriz que isolou isso:

| pygame | plataforma | resultado |
|---|---|---|
| sem janela (`SDL=dummy`) | wayland | ✓ janela OK, 1150 fps |
| janela real | SDL padrão (x11) | ✗ EGL failed |
| janela real | `SDL=x11` | ✗ EGL failed |
| janela real | **`SDL=wayland`** | **✓ janela OK, 1105 fps** |

Ou seja: o problema aparecia com ou sem as variáveis da NVIDIA — só ficou visível
quando elas entraram. `SDL_VIDEODRIVER=dummy` também resolve, mas mata o teclado
(sem janela SDL o pygame não captura tecla).

`KeyboardG1Arm.connect()` agora define `SDL_VIDEODRIVER=wayland` sozinho quando
detecta sessão Wayland, e só se você não tiver definido nada. Se precisar
sobrescrever:

```bash
SDL_VIDEODRIVER=x11 python init_lerobot_teleoparate_v2.py ...
```

---

## 7. Problemas comuns

| sintoma | causa |
|---|---|
| atena não recebe `lowstate` | `run_sim_remote.py` não está rodando, ou firewall bloqueando 6001 |
| imagem preta / sem câmera | `run_sim.py` sem `publish_images`, ou `--cam-robot` com IP errado |
| robô parado, sem erro | `--remote-sim` faltando: o DDS não conectou, só as câmeras |
| `could not create window` / `EGL: Failed to make context current` | SDL e GLFW em backends diferentes no Wayland — ver §6.5 |
| simulador muito lento / travado | rodando na Intel; use `PROMETHEUS_FORCE_NVIDIA=1` (§6.5) |
| tronco não gira no MuJoCo | `enable_waist: False` no `config.yaml` (ver SCHEMA_G1_V2.md §4) |
| tronco não gira no robô | `use_waist_yaw=False` no `UnitreeG1Dex3Config` |
| ação com 28 dims quando devia ter 29 | checkpoint treinado no schema v1 — retreine |
