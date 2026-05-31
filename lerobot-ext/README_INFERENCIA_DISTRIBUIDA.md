# Inferência distribuída da VLA (pi05 + RTC) — sim e modelo em máquinas separadas

O **simulador MuJoCo** roda numa máquina com tela; o **modelo VLA** roda na máquina com GPU.
As duas se comunicam por ZMQ. Há também o modo de uma máquina só (headless) no fim.

```
  MÁQUINA DO SIM (env g1)                    MÁQUINA DA GPU (env ms3)
  laptop_sim_host.py            ── ZMQ ──►   init_lerobot_inference_lf.py
   sim + câmeras :5555                        carrega o pi05 na GPU + loop RTC
   run_g1_server :6000-6003 ◄────────────     robô em modo ZMQ (--robot-ip)
```

---

## Pré-requisitos

- **Máquina do sim** — env `g1`: mujoco, unitree_sdk2py, lerobot, zmq, opencv.
- **Máquina da GPU** — env `ms3`: lerobot/pi05 + as deps do sim
  (`pip install "mujoco==3.6.0" pygame glfw "cyclonedds==0.10.2" loguru` e
  `pip install -e ~/Prometheus/Luiz/prometheus-vla/unitree_sdk2_python --no-deps`).
- A máquina da GPU precisa **alcançar o IP** da máquina do sim (`ping`). Pegue o IP com
  `ip addr show tun0` (ou a interface da sua rede).

---

## Terminal 1 — máquina do sim

```bash
cd ~/I2CA/prometheus-vla/unitree-g1-mujoco
python laptop_sim_host.py
```
**O que faz:** sobe o MuJoCo (viewer 3D), publica as câmeras por ZMQ na porta 5555, e lança o
`run_g1_server.py` (ponte DDS↔ZMQ que expõe corpo e mãos nas portas 6000-6003). Abre janelas de
**RGB** e **DEPTH** e imprime a distância mão↔copo.

Opcional: `BAND_Z=0.9 python laptop_sim_host.py` muda a altura em que o robô é segurado (default 1.0).

## Terminal 2 — máquina da GPU

```bash
ssh hercules@10.9.8.252
conda activate /home/hercules/miniconda3/envs/ms3
cd ~/Prometheus/Luiz/prometheus-vla/lerobot-ext
export HF_HOME=/data/huggingface-models HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1
python init_lerobot_inference_lf.py \
  --checkpoint=~/Prometheus/Luiz/prometheus-vla/train/output/groot_pretrain_lf/checkpoints/best/pretrained_model \
  --task="Pick up the cup" --robot-ip=10.8.8.52 \
  --fps=30 --rtc-execution-horizon=10
```
**O que faz:** carrega o checkpoint pi05 na GPU e roda o loop de inferência com Real-Time Chunking;
o robô opera em modo ZMQ conectando na máquina do sim. Sem RTC: `--rtc-enabled=false`.

> `--robot-ip` = IP da máquina onde o sim está rodando (no exemplo, `10.8.8.52`, a interface
> `tun0` da VPN — confira com `ip addr show tun0`).

Está dirigindo quando aparece `Connected to robot` seguido de `[ACTION_QUEUE] Indexes diff...`.

---

## Ajuste ao vivo (sem reiniciar)

Edite `lerobot-ext/g1_tuning.json` na máquina da GPU — o robô em execução aplica na hora:
```json
{ "arm_kp": 100.0, "smoothing_alpha": 0.2, "max_delta": 0.06 }
```
`arm_kp` = rigidez do braço · `smoothing_alpha` = responsividade · `max_delta` = passo máx por ciclo.

## Log de ações

Suba a VLA com `G1_ACTION_LOG=/tmp/g1_action_log.jsonl` no `export`. Cada passo grava, por junta,
o alvo da VLA, o que foi enviado e o observado (JSONL) — útil pra debugar o que o robô recebeu.

## Parar

`Ctrl-C` em cada terminal. Se ficar processo/porta preso:
`for p in 5555 6000 6001 6002 6003; do fuser -k ${p}/tcp; done` (não use `pkill -f` com o nome do
script — casa o próprio shell).

---

## Troubleshooting

| Sintoma | Fix |
|---|---|
| `Waiting for robot state...` infinito | A GPU não recebe o estado. Confira `--robot-ip` e se a máquina do sim está no ar (portas 6001/5555 escutando). |
| `Address already in use` (5555/600x) | Órfão de run anterior: `fuser -k <porta>/tcp`. |
| Braço trêmulo | Baixe `arm_kp` no `g1_tuning.json`. |
| Janela não abre | Precisa de display (`echo $DISPLAY`). |

---

## Modo de uma máquina só (headless, grava MP4)

Sim e modelo na mesma máquina com GPU, sem tela:
```bash
cd ~/Prometheus/Luiz/prometheus-vla/lerobot-ext       # no config.yaml do sim: ENABLE_ONSCREEN: false
MUJOCO_GL=egl HF_HOME=/data/huggingface-models HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
python init_lerobot_inference_lf.py --sim \
  --checkpoint=~/Prometheus/Luiz/prometheus-vla/train/output/groot_pretrain_lf/checkpoints/best/pretrained_model \
  --task="Pick up the cup" --fps=30 --rtc-execution-horizon=10
```
**O que faz:** sobe o sim no mesmo processo (sem viewer, render por EGL) e grava o rollout em
`lerobot-ext/rollout_grasp_<hora>.mp4`.
