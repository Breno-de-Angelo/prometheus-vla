# Setup Pi05-D Distribuído (Mori + Atenas)

## 🎯 Arquitetura
- **Mori (seu notebook)**: Roda MuJoCo simulator, publica imagens via ZMQ (porta 5555), recebe ações via ZMQ (porta 6001)
- **Atenas (A100 80GB)**: Roda pi05-D inference, consome imagens via ZMQ (5555), envia ações via ZMQ (6001)

---

## 📍 IP Local (MORI)
```
192.168.15.111
```

---

## 🖥️ PASSO 1: Mori — Rodar MuJoCo Simulator

**No seu notebook:**

```bash
cd /home/luiz-aumo/I2CA/prometheus-vla

# Ativar conda g1
conda activate g1

# Rodar simulator (deixar rodando)
python unitree-g1-mujoco/run_sim.py
```

**Saída esperada:**
```
📷 Cameras: head_camera, head_camera_depth → ZMQ port 5555
Simulator running. Press Ctrl+C to exit.
Camera images publishing on tcp://localhost:5555
```

✅ **Deixar esse terminal aberto e rodando!**

---

## 🧠 PASSO 2: Atenas — Rodar Pi05-D Inference

**No A100:**

```bash
cd /home/luiz-aumo/I2CA/prometheus-vla

# Ativar conda (ou venv com torch + lerobot)
conda activate g1  # ou seu venv

# Rodar inference (conectando ao ZMQ de MORI)
bash RUN_ATENAS.sh
```

Ou diretamente:
```bash
python lerobot-ext/init_lerobot_inference_pi05d_v2.py \
  --sim \
  --checkpoint=train_output/pi05/checkpoints/best/pretrained_model \
  --cam-robot=192.168.15.111 \
  --port-cam=5555 \
  --task="Pick up the cup" \
  --debug
```

**Saída esperada:**
```
⏳ Carregando PI05-D de: train_output/...
✅ PI05-D carregado com sucesso!
⏳ Conectando ao Unitree G1 (Simulação: True)...
✅ Robô Conectado!
📡 Conectando ao stream em 192.168.15.111:5555...
🚀 INFERÊNCIA PI05-D ATIVA
```

---

## 📋 Checklist

- [ ] Mori: `python unitree-g1-mujoco/run_sim.py` rodando
  - [ ] Publicando imagens na porta 5555
  - [ ] ActionReceiver esperando ações na porta 6001
- [ ] Atenas: conecta via `--cam-robot=<MORI_IP>`
  - [ ] Recebe frames RGB + depth do ZMQ (5555)
  - [ ] Roda pi05-D, genera ações
  - [ ] Envia ações via ZMQ (6001)
- [ ] **Mori vê robot se movendo no MuJoCo viewer (confirma feedback funciona!)**

---

## 🐛 Troubleshooting

### ZMQ connection refused
```
ConnectionRefusedError: [Errno 111] Connection refused
```
→ Mori: `run_sim.py` não tá rodando ou não publicou na porta 5555
→ Solução: Restart `run_sim.py`

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
→ A100 deve ter espaço, mas se acontecer, reduzir batch size ou usar CPU

### Frames não chegam
→ Verificar firewall: porta 5555 aberta?
→ Testar ping: `ping 192.168.15.111`

### Robô não se mexe / ações não chegam
```
[ActionReceiver] Aguardando ações em tcp://127.0.0.1:6001
```
→ Mori: verificar se `run_sim.py` mostra a mensagem acima
→ Atenas: verificar se `[ActionSenderZMQ] Conectado a...` aparece
→ Firewall: porta 6001 aberta entre Atenas e Mori?

---

## 🔧 Customizações

**Mudar task description:**
```bash
--task="Push the cube"
```

**Mudar checkpoint:**
```bash
--checkpoint=train_output/seu_modelo/checkpoints/best/pretrained_model
```

**Modo verbose:**
```bash
--debug
```

---

## 🧪 Testar Comunicação Distribuída

### 1. Verificar publicação de imagens (Mori → Atenas)
```bash
# Terminal no Atenas:
python3 -c "
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect('tcp://192.168.15.111:5555')
sock.setsockopt_string(zmq.SUBSCRIBE, '')
print('Aguardando imagens...')
msg = sock.recv()
print(f'✅ Recebido {len(msg)} bytes')
"
```

### 2. Verificar recepção de ações (Atenas → Mori)
```bash
# Terminal no Mori:
python3 -c "
import zmq, json
ctx = zmq.Context()
sock = ctx.socket(zmq.PULL)
sock.bind('tcp://127.0.0.1:6001')
print('Aguardando ações...')
msg = sock.recv_string()
payload = json.loads(msg)
print(f'✅ Recebido: {payload}')
"

# Em outro terminal no Atenas:
python3 -c "
import zmq, json
ctx = zmq.Context()
sock = ctx.socket(zmq.PUSH)
sock.connect('tcp://192.168.15.111:6001')
payload = {'body_motors': [{'idx': 0, 'q': 0.5, 'kp': 50, 'kd': 1}]}
sock.send_string(json.dumps(payload))
print('✅ Ação enviada')
"
```

---

**Criado em: 2026-05-23**
