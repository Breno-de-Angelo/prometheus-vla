# Deploy pi05 ablations no Unitree G1 + Dex3 (robô real)

## No robô (terminal SSH no robô, 10.9.8.73)

```bash
# 1. Body + DDS server
cd ~/prometheus-vla
python lerobot/src/lerobot/robots/unitree_g1/run_g1_server.py

# 2. Em outro terminal: RealSense ZMQ publisher
python lerobot-ext/Scripts_Prometheus_int/realsense_server.py
```

Mantenha os dois rodando.

## No hercules (10.9.8.232)

### Dry-run primeiro (sem mexer no robô — só inspeciona ações)

```bash
ssh hercules@10.9.8.232
cd ~/prometheus-vla
./run_on_robot.sh pi05_vanilla_cup3
```

O script faz pre-flight (CUDA, HF auth, ping ao robô, porta ZMQ) e roda inferência em `--dry-run`. Logs mostram a primeira ação prevista por chunk.

### Live (envia comandos pro robô)

```bash
./run_on_robot.sh pi05_vanilla_cup3 --live
```

Tem 5s pra Ctrl+C abortar antes de começar a enviar ações.

## Checkpoints disponíveis (best/)

| Nome | Modalidades | Best val_action_mse | Risco |
|---|---|---|---|
| `pi05_vanilla_cup3` | RGB+state | 0.176 | baixo |
| `pi05_depth_cup3` | RGB+state+depth | 0.197 | ⚠️ intrínsecos hardcoded (pode divergir no robô) |
| `pi05_droid` | RGB+state | 0.26+ (em treino) | médio (init Franka) |
| `pi05_libero` | RGB+state | TBD | médio (init Franka) |
| `pi05_vanilla_unitree_toast_then_cup3` | RGB+state | TBD | melhor candidato (pretrain Unitree + finetune cup3) |

## Override de variáveis

```bash
ROBOT_IP=10.9.8.99 TASK="Pick up the cup" FPS=20 ./run_on_robot.sh pi05_vanilla_cup3
```

## Troubleshooting

- **`CUDA unavailable`**: env `g1` não foi ativado, ou nvidia driver.
- **HF user ANONYMOUS**: `huggingface-cli login` (token tem que aceitar `google/paligemma-3b-pt-224`).
- **port 5555 not open**: `realsense_server.py` não está rodando no robô.
- **ping fail**: cabo, switch, ou robô desligado.
- **Predição erra magnitude**: olhar primeiras ações no dry-run; se valores absurdos (>10 rad), checkpoint divergiu — usar outro best/.

## Sobre depth (pi05_depth_cup3)
Os intrínsecos são `fx=fy=600, cx=320, cy=240` e a escala `z = depth_normalized × 2 m`. Isso foi assumido no treino. Antes de rodar live com esse checkpoint, valide com o `realsense-viewer`:
- resolução 640×480 ✓
- range esperado: 0.1 a 2 m
- se sua RealSense publica em mm raw 16-bit, vai precisar normalizar antes (corrigir em `inference_realtime_pi05d.py:90` ou no driver ZMQ).
