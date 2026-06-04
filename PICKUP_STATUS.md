# Pickup & Lift Cup - Status Report

**Data:** 2026-05-23  
**Objetivo:** Robô G1 pegar e levantar copo usando controle distribuído (GPU remota → Simulator local)

## ✅ O que funciona

1. **Comunicação ZMQ (porta 6001)** ✓
   - GPU remota consegue enviar comandos ao simulator
   - Payload recebida corretamente com `body_motors` e `right_hand`

2. **Motores do corpo** ✓
   - Shoulder pitch alcança valores alvo (0.0 → 0.6 → 1.0 → 0.3 → 0.0)
   - Elbow responde perfeitamente aos comandos
   - Log mostra: `q_target` = `q_actual` com erro = 0.0000

3. **Motores da mão** ✓
   - Dedos fecham corretamente: q_target vai de 0.0 a 1.0 em waypoint 3
   - Valores de kp/kd variam conforme esperado (kp=5.0 para grasp, kp=10.0 para lift)

4. **Logging e análise** ✓
   - Action logger captura todos os eventos
   - Hand motors aparecem nos bridge_state eventos
   - Análise corrigida para ler hand_motors do array

## ❌ O que não funciona

### Problema Principal: Mão não alcança o copo

**Física do problema:**
- Cup position: `[0.349, -0.046, 0.799]` (Z=79.9cm)
- Thumb position (máxima descida): `[0.253, -0.091, 0.970]` (Z=97.0cm)
- **Gap vertical: 17.1cm** (mão fica 17cm acima do copo)
- **Contatos detectados: 0** (nenhum contato durante grasp)

### Análise de Cinemática Direta

Testes com forward kinematics mostram:
```
Shoulder Pitch | Elbow | Thumb Z | Gap to Cup
      0.6      |  0.8  | 0.888   | -8.9cm
      1.0      |  0.8  | 0.888   | -8.9cm (atual)
      2.0      |  0.8  | 0.888   | -8.9cm
      2.5      |  1.5  | 0.888   | -8.9cm
```

**Descoberta:** Independente dos ângulos, thumb fica sempre em Z≈0.888!

### Possíveis Causas

1. **Limitações geométricas do robô**
   - Arm pode não estender o suficiente para alcançar cup em Z=0.799
   - Base/corpo do robô está posicionado alto demais

2. **Índices de joint incorretos**
   - Joints no qpos array: [30, 33] (corretos para right arm)
   - Actuators: [29, 32] (corretos para motor commands)
   - Mas talvez há outros joints afetando posição final (wrist roll/pitch/yaw?)

3. **Configuração inicial do robô**
   - O simulator pode estar iniciando com configuração não-ideal
   - Corpo/base pode estar posicionado diferente do esperado

## 🔄 Estado da Trajetória

Último teste (`action_log_20260523_212412.jsonl`):

```
Fase     | Shoulder | Elbow | Hand | Cup Z | Status
Init     | 0.00     | 0.30  | 0.0  | 0.799 | -
Approach | 0.60     | 0.80  | 0.0  | 0.799 | Mão ainda aberta
Descend  | 1.00     | 0.80  | 0.0  | 0.799 | Mão aberta, 17cm acima
Grasp    | 1.00     | 0.80  | 1.0  | 0.799 | ⚠️  Mão fecha mas sem contato!
Lift     | 0.30     | 0.80  | 1.0  | 0.799 | ❌ Copo desce (gravidade)
Return   | 0.00     | 0.30  | 0.0  | 0.799 | Volta posição inicial
```

## 🎯 Próximas Ações Necessárias

### A. Investigar Limitação Geométrica
```
1. Verificar posição base do robô no scene
2. Testar IK com vários shoulder_pitch/elbow/wrist values
3. Ver se hand pode descer com wrist joints
4. Considerar inclinar corpo do robô
```

### B. Validar Índices de Joints
```
1. Verificar se outros wrist joints afetam Z
2. Confirmar que scene_43dof.xml não tem constraints
3. Testar movimento de cada joint individualmente
```

### C. Ajustar Waypoints
```
1. Se geom permite: aumentar shoulder_pitch > 1.0
2. Aumentar elbow range
3. Usar wrist joints para ajuste fino de altura
4. Se necessário: mover body ou inclinar robô
```

## 📊 Métricas

| Métrica | Esperado | Atual | Status |
|---------|----------|-------|--------|
| Shoulder pitch max | 1.0 | 1.0 ✓ | OK |
| Elbow max | 0.8 | 0.8 ✓ | OK |
| Hand closure | 1.0 | 1.0 ✓ | OK |
| **Thumb-Cup distance** | **0cm** | **17cm** | ❌ CRÍTICO |
| Cup height change | +10cm | -10cm | ❌ CRÍTICO |
| Contacts detected | >0 | 0 | ❌ CRÍTICO |

## 🔧 Debugging Info

- Config file: `unitree-g1-mujoco/scene_43dof.xml`
- Indices usados:
  - Shoulder Pitch: `qpos[30]`, `actuator[29]`
  - Elbow: `qpos[33]`, `actuator[32]`
- Cup target position: `[0.35, -0.05, 0.80]` (hardcoded em vários scripts)
- Log files: `/tmp/action_log_*.jsonl`

## 💡 Hipóteses para Testar

1. **H1:** Base do robô está muito alta → cup é inalcançável com essa postura
2. **H2:** Índice de wrist está afetando height → testar todos wrist joints
3. **H3:** Scene tem configuração diferente → verificar URDF/XML
4. **H4:** Necessário usar IK inverso ao invés de FK direto
