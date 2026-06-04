# Guia de Otimização - Trajetória Perfeita

Baseado no feedback técnico, aqui estão as 3 correções essenciais para conseguir pegar e levantar o copo com sucesso.

---

## 1️⃣ Corrigir o Script de IK com Levenberg-Marquardt

**Problema:** A pseudo-inversa pura (`np.linalg.pinv`) trava em singularidades e retorna zeros.

**Solução:** Usar damped least squares (Levenberg-Marquardt).

**Arquivo:** `generate_trajectory_ik.py` (já corrigido)

**O que mudou:**

```python
# ANTES (falha perto de singularidades)
jac_pinv = np.linalg.pinv(jac_arm)
dq = jac_pinv @ (error * gain)

# DEPOIS (estável com damping)
damping = 0.01
jjt = jac_arm @ jac_arm.T
damped_inv = np.linalg.inv(jjt + damping * np.eye(3))
dq = jac_arm.T @ damped_inv @ (error * gain)
```

---

## 2️⃣ Ajustar Fricção e Contato no scene_29dof.xml

**Problema:** Mesh-to-mesh collision é instável. O copo escorrega dos dedos.

**Solução:** Aumentar fricção (especialmente torsional) e suavizar o contato.

**Arquivo:** `unitree-g1-mujoco/assets/scene_29dof.xml`

**O que adicionar/mudar:**

Procure pelas tags `<geom>` do copo e dos dedos. Adicione/altere os atributos:

```xml
<!-- COPO -->
<geom name="cup" type="cylinder" size="0.04 0.08" mass="0.1"
      friction="1.5 0.005 0.0001" 
      solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2"/>

<!-- DEDOS (ajuste para cada dedo)
<geom type="mesh" mesh="link_dedo_thumb"
      friction="2.0 0.01 0.0001" 
      solref="0.01 1.0" solimp="0.95 0.99 0.001 0.5 2"/>
```

**Significado dos parâmetros:**

`friction="2.0 0.01 0.0001"`:
- `2.0` = fricção normal (aumenta grip)
- `0.01` = fricção torsional (impede rotação do copo na mão)
- `0.0001` = fricção de rolamento

`solref="0.01 1.0"` e `solimp="0.95 0.99 0.001 0.5 2"`:
- Transformam o contato de "aço duro" para "borracha macia"
- Evita interpenetração violenta e instabilidade

---

## 3️⃣ Usar Ganhos Variáveis (Atuador Complacente)

**Problema:** Ganhos fixos causam penetração de malha e ejeção do copo.

**Solução:** Reduzir `kp` e aumentar `kd` durante o fechamento da mão.

**Arquivo:** `test_pick_cup_optimized.py` (novo script)

**O que mudou:**

```python
# WAYPOINT 3: Fechar a mão
"hand_kp": 5.0,   # Reduzido de 20.0 (atuador macio)
"hand_kd": 3.0,   # Aumentado de 1.0 (mais amortecimento)

# WAYPOINT 4: Levantar (mão firme)
"hand_kp": 10.0,  # Intermediário (manter grip)
"hand_kd": 2.0,
```

**Razão:** Com `kp` baixo, o dedo não tenta "forçar" a posição `1.0` através do copo. Em vez disso, ele fecha gentilmente até tocar e para ali, mantendo uma força estável e segura.

---

## 📋 Checklist de Implementação

- [ ] Executar `generate_trajectory_ik.py` corrigido (damping ativo)
  - Esperado: Retorna waypoints com valores não-zero
  
- [ ] Editar `scene_29dof.xml` com os novos atributos de friction e solref/solimp
  
- [ ] Rodar `test_pick_cup_optimized.py`
  - Esperado: Dedos fecham suavemente, copo é levantado
  
- [ ] Rodar `run_automated_pickup_test.py` com o novo script
  - Esperado: Registra sucesso (dedos > 0.8, movimento descendente-ascendente)

---

## 🚀 Teste Rápido

```bash
# Terminal 1: Simulator
conda activate g1
python unitree-g1-mujoco/run_sim.py

# Terminal 2: Teste otimizado
conda activate g1
cd pickup_experiments
python test_pick_cup_optimized.py
```

**Resultado esperado:** Robô abaixa, fecha mão suavemente, levanta copo.

---

## 📚 Referência Técnica

- **Levenberg-Marquardt (Damped LS):** Estabiliza IK perto de singularidades
- **Fricção Torsional:** Crítica para grasping estável (impedance control)
- **Soft Contact (solimp):** Simula borracha em vez de metal, reduz jitter numérico
- **Compliant Actuator:** Reduzir kp durante contato = controle de força sem sensores

---

**Próxima etapa:** Após validar que consegue pegar e levantar com o hardcoded otimizado, use a trajetória do `generate_trajectory_ik.py` corrigido ou treina com BC/diffusion policies usando `xr_teleoperate`.
