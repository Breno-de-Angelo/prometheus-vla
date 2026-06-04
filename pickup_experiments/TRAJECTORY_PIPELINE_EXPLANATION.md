# Explicação do Pipeline de Trajetória e Execução no Robô

## 📊 Resposta sobre Isaac Lab + Explicação dos Scripts

### Sobre Isaac Lab:

Se você tem notebook com GPU integrada → ❌ NÃO funciona
Se você tem desktop com RTX 4080+ → ✅ Funciona localmente
Alternativa: Use NVIDIA Brev (cloud) → ✅ Roda tudo na nuvem

---

## 🤖 Como Calculei a Trajetória e Rodei no Robô

Explicação do **fluxo completo** que tentei:

### **1️⃣ Geração da Trajetória**

`generate_trajectory_ik.py` faz:
- Carrega modelo MuJoCo (scene_29dof.xml)
- Calcula Jacobiana do braço direito
- Usa IK (cinemática inversa) para atingir posição do copo
- Gera waypoints com qpos (configuração de juntas)
- Salva em trajectory_ik.json

**Status:** ❌ Falhou (retornou zeros - alvo provavelmente irrazoável)

---

### **2️⃣ Trajetória Manual (Hardcoded)**

Criei 3 versões:

#### **a) test_simple_elbow.py** ✅ FUNCIONOU
- Só move cotovelo (motor 25) de 0.0 → 1.5
- Teste SUPER simples para validar pipeline
- Log: q_actual = q_target (erro = 0.0000)

#### **b) test_hardcoded_trajectory.py** (5 waypoints)
- Waypoint 0: Posição inicial (shoulder_pitch=0.0)
- Waypoint 1: Braço para frente (shoulder_pitch=0.8)
- Waypoint 2: Descer para pegar (shoulder_pitch=1.2)
- Waypoint 3: Levantar (shoulder_pitch=0.3)
- Waypoint 4: Volta (shoulder_pitch=0.0)

#### **c) test_pick_cup_complete.py** (6 waypoints com mão)
- 1. Inicial
- 2. Abaixar + estender
- 3. Descer mais
- 4. Fechar dedos (índex=0.8, meio=0.8)
- 5. Levantar
- 6. Volta + abrir mão

#### **d) test_pick_cup_aggressive.py** (5 waypoints MUITO forte)
- shoulder_pitch = 1.5 (em vez de 1.0)
- todos_dedos = 1.0 (máximo)
- tempo_por_waypoint = 0.5s (em vez de 0.3s)

---

### **3️⃣ Pipeline de Execução**

```
┌─────────────────────────────────────────────────┐
│ MORI (Meu Notebook)                             │
├─────────────────────────────────────────────────┤
│ run_sim.py                                      │
│ └─ MuJoCo Simulator rodando                     │
│ └─ Carrega G1 + Dex3 + copo                     │
│ └─ ActionReceiver escutando na porta 6001       │
│                                                  │
│ test_pick_cup_*.py                              │
│ └─ Conecta via ZMQ PUSH na porta 6001           │
│ └─ Envia motor commands em JSON                 │
│ └─ ActionReceiver injeta no simulator           │
│ └─ ActionLogger registra tudo em JSONL          │
└─────────────────────────────────────────────────┘
                    ↓ (ZMQ PUSH/PULL)
         JSON com body_motors + right_hand
                    ↓
┌─────────────────────────────────────────────────┐
│ ATENAS (Remota com A100)                        │
├─────────────────────────────────────────────────┤
│ init_lerobot_inference_pi05d_v2.py              │
│ └─ PI05D VLA inference                          │
│ └─ Recebe camera depth via ZMQ 5555             │
│ └─ Gera ações                                   │
│ └─ ActionSenderZMQ envia para Mori 6001         │
└─────────────────────────────────────────────────┘
```

---

### **4️⃣ Scripts-Chave**

`run_sim.py` → Inicia MuJoCo simulator ✅ Funciona

`unitree-g1-mujoco/sim/action_receiver.py` → Recebe comandos ZMQ ✅ Funciona

`unitree-g1-mujoco/sim/action_logger.py` → Registra eventos em JSONL ✅ Funciona

`test_simple_elbow.py` → Teste simples (cotovelo) ✅ Prova pipeline

`test_pick_cup_complete.py` → Trajetória 6-waypoint ⚠️ Executa mas não pega

`test_pick_cup_aggressive.py` → Versão mais agressiva ⚠️ Executa mas não pega

`run_automated_pickup_test.py` → Loop automático + análise ⚠️ Não gera logs novos

`analyze_action_log.py` → Analisa logs JSONL ✅ Funciona

---

### **5️⃣ Problema Identificado**

A pesquisa mostrou que **hardcoded trajectories não funcionam** porque:

**Sem validação de contato** - não sabe se tocou o copo

**Sem force control** - apenas posição pura

**Sem sensores** - não valida se objeto foi agarrado

**Mesh-to-mesh collision** - instável

**Sem fricção torsional** - dedos escorregam
