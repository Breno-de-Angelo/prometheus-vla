# Inferência Offline / Replay (sem robô) — VLA na Atena + MuJoCo no PC

Roda a VLA pi05 **sem o robô real**: a VLA (na Atena, GPU) enxerga a imagem + o estado
de uma **run real gravada** (um episódio do dataset LeRobot), decide as ações, e um
**MuJoCo** (no teu PC/notebook) visualiza o robô executando essas decisões. Serve pra
**auditar as decisões do modelo** e iterar nos fixes (gate da mão, EH do RTC, etc.) sem
depender do robô físico.

## Como funciona (modo "replay aberto")

```
[ATENA GPU1]  inference_realtime_pi05d_right14.py --robot-ip 127.0.0.1   (VLA intacta)
   ▲ imagem+state (dataset)  via ZMQ 5555/6001/6002          ▼ ações via ZMQ 6000/6003
   └──────────────── túnel SSH reverso ───────────────────────┘
[TEU PC]  unitree-g1-mujoco/offline_sim_host.py
   ├─ serve IMAGEM + STATE de um episódio real do dataset (ZMQ, schema do run_g1_server)
   ├─ recebe os comandos da VLA (não realimenta — replay ABERTO)
   └─ --mujoco: aplica os comandos a um G1 do MuJoCo (via DDS) e grava um vídeo 3ª-pessoa
```

- **Replay ABERTO:** a VLA vê SEMPRE o dataset (imagem+state coerentes da run real); o
  MuJoCo só *visualiza* o que a VLA mandaria. O robô do sim diverge da run real depois de
  alguns frames — isso é esperado (o objetivo é ver as decisões, não fechar o loop).
- **Zero mudança na VLA:** `is_simulation=False` já usa o caminho ZMQ-socket que conecta
  no `--robot-ip`. O `offline_sim_host` fala o mesmo protocolo do `run_g1_server` — a VLA
  não distingue de um robô real.

## Procedimento

### 1. No PC (notebook) — sobe o servidor do dataset (+ MuJoCo)

```bash
cd unitree-g1-mujoco
# só auditar decisões (sem janela do robô; mais leve):
python offline_sim_host.py --dataset-root ../datasets/_merged/G1_Dex3_pick_white_cup_right8_1squeeze \
    --episode 0 --loop

# COM visualização MuJoCo (grava vídeo do robô executando as decisões):
python offline_sim_host.py --dataset-root ../datasets/_merged/G1_Dex3_pick_white_cup_right8_1squeeze \
    --episode 0 --loop --mujoco --video /tmp/offline_viz.mp4
```
O `observation.state` do dataset right8 é 14-dim (7 braço + 7 dedos) — serve pro right14.
Escolha o episódio com `--episode N`. Sem `--loop` ele roda 1 vez e para.

### 2. Túnel SSH reverso (PC → Atena) — encaminha as 5 portas

```bash
ssh -N -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes \
    -R 5555:127.0.0.1:5555 -R 6000:127.0.0.1:6000 -R 6001:127.0.0.1:6001 \
    -R 6002:127.0.0.1:6002 -R 6003:127.0.0.1:6003 hercules@<ATENA>
```
(O PC atrás de NAT não é alcançável direto pela Atena; o túnel reverso resolve. A VLA
na Atena conecta em `127.0.0.1`, que tunela de volta pro servidor no PC.)

### 3. Na Atena — roda a VLA apontando pro túnel

```bash
cd ~/Prometheus/Luiz/prometheus-vla
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python inference_realtime_pi05d_right14.py \
    --robot-ip 127.0.0.1 \
    --checkpoint train_output/cup_pi05_right14_rgb238_lf/checkpoints/last/pretrained_model \
    --fps 30 --hand-kp 0.8 --rtc --live --denoising-steps 5 \
    --rtc-execution-horizon 25 --rtc-max-guidance 1.0 \
    --home-arm-s 0 --open-hand-s 0 --rehome-idle-s 0
```
- `HF_HUB_OFFLINE=1` é **obrigatório**: o tokenizer do paligemma é repo gated; sem isso o
  load falha com 403 (em shell não-interativo sem HF_TOKEN). O paligemma já está no cache.
- `--home-arm-s 0 --open-hand-s 0 --rehome-idle-s 0`: desliga soft-start/re-home (não fazem
  sentido no replay — não há robô pra posicionar; deixa a auditoria limpa, frame a frame).

### 4. O que você vê
- **OmniView** (`--live`, `http://<ATENA>:8013/live.html`): a imagem do dataset + a
  trajetória que a VLA decide + o mapa de atenção. Auditoria ao vivo das decisões.
- **Vídeo MuJoCo** (`--video`): o G1 executando os comandos da VLA, 3ª pessoa.
- **RunRecorder**: `train/log/run_<ts>/chunks.jsonl` (o que a VLA viu + decidiu) pra
  comparar offline com a ação real do dataset.

## Análise offline (auditar as decisões)
O `chunks.jsonl` tem `state` (o que a VLA viu) e `actions[50][14]` (o chunk decidido). A
ação REAL daquele frame está no parquet do dataset (`action`). Comparar a ação decidida vs
a real mostra o efeito dos fixes (gate da mão, EH=25, merge discard). Ver `docs/OFFLINE_INFER_PROGRESS.md`.

## Detalhes técnicos / pegadinhas
- **Portas ZMQ** (bind no PC): 5555 PUB imagem, 6001 PUB state corpo, 6002 PUB state mãos,
  6000 PULL cmd corpo, 6003 PULL cmd mãos. Iguais ao `run_g1_server.py`.
- **Mapeamento dataset→motores**: state[0:7]=braço dir → lowstate.motor_state[22:28];
  state[7:14]=dedos dir → handstate(right).motor_state[0:6].
- **Depth dummy**: a `UnitreeG1Dex3` cria a câmera `head_camera_depth` e lê todas no
  `get_observation`; o servidor publica um depth PNG zerado pra não travar (o right14 é
  RGB-only). Pra pi05-D, publicar o depth real do dataset (TODO).
- **MuJoCo EGL**: o `--mujoco` desliga ENABLE_ONSCREEN/OFFSCREEN do `config.yaml`
  TEMPORARIAMENTE (restaura no fim) pra evitar conflito de 2 contextos GL; renderiza com
  um `mujoco.Renderer` próprio (640×480, o tamanho do framebuffer do modelo) e dirige a
  física com `sim_env.sim_step()` + DDS `rt/lowcmd`.
- **Rede instável**: se a VPN/rede pra Atena cair, o túnel morre (re-suba). O servidor no
  PC e a VLA na Atena sobrevivem independentes; só o túnel precisa voltar.

## Arquivos
- `unitree-g1-mujoco/offline_sim_host.py` — o servidor (novo).
- `inference_realtime_pi05d_right14.py` — a VLA (intacta; só `--robot-ip 127.0.0.1`).
- Reuso: `run_g1_server.py` (schemas), `camera_zmq.py`, `env.py` (make_env).
