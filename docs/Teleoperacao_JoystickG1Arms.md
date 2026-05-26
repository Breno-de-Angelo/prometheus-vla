# 🎮 Tutorial – Teleoperação JoystickG1Arms + Gravação de Dataset no LeRobot (Unitree G1 Dex3)

Este tutorial explica:

- Como usar a teleoperação JoystickG1Arms

- Como gravar um dataset com lerobot-record

- O que significa cada parâmetro do comando

- Como funciona o controle do joystick

- Como criar a pasta de datasets com permissão 777

## Criando a pasta de datasets (permissão global)

Para permitir que **todos os usuários do sistema** tenham acesso aos datasets:

```bash
sudo mkdir -p /Dados/Prometheus-datasets
sudo chmod -R 777 /Dados
```

### 🔎 **O que isso faz:**

- mkdir -p → cria a pasta mesmo que diretórios anteriores não existam

- chmod -R 777 → dá permissão total (leitura, escrita e execução) para todos os usuários

⚠️ Use 777 apenas em ambiente controlado (laboratório ou desenvolvimento).

## Comando para iniciar a gravação

```bash
lerobot-record \
  --robot.type=unitree_g1_dex3 \
  --robot.is_simulation=true \
  --teleop.type=joystick_g1_arms \
  --dataset.repo_id=Seu_Usuario/g1_pick_kettle \
  --dataset.root=/Dados/Prometheus-datasets/ \
  --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the kettle" \
  --play_sounds=false

```

### Lembrando de iniciar o ambiente vitual g1 ```conda activate g1```

## Explicação de cada parâmetro

### 🤖 ```--robot.type=unitree_g1_dex3```

Define o tipo do robô.

- ```unitree_g1_dex3``` → G1 com mãos Dex3 (29 juntas + mãos articuladas)

---

### 🖥 ```--robot.is_simulation=true```

Define se está usando:

- ```true``` → Simulação

- ```false``` → Robô real

---

### 🎮 ```--teleop.type=joystick_g1_arms```

Define o teleoperador utilizado.

Neste caso:

- Controle via joystick

- Controla apenas os braços e mãos

---

### 📂 ```--dataset.repo_id=Mrwlker/g1_pick_kettle```

Nome do dataset.

Formato:

```
usuario/nome_do_dataset
```

Mesmo que ```push_to_hub=false```, esse nome organiza localmente.

---

### 💾 ```--dataset.root=/Dados/Prometheus-datasets/```

Define onde o dataset será salvo no sistema.

Estrutura criada automaticamente:

```
/Dados/Prometheus-datasets/
   └── Mrwlker/
       └── g1_pick_kettle/
```

---

### ☁️ ```--dataset.push_to_hub=false```

-```true``` → Envia dataset para HuggingFace

-```false``` → Salva apenas localmente

---

### 🎯 ```--dataset.single_task="Pick up the kettle"```

Define a descrição textual da tarefa.

Essa frase será usada como instrução para treino de VLA (Vision-Language-Action).

---

### 🔊 ```--play_sounds=false```

Desativa sons durante gravação.

## Como funciona o Controle JoystickG1Arms

Baseado no código fornecido.

### ⚙️ ***Configuração principal***
```
joystick_id = 0
speed = 0.02
deadzone = 0.1
fps = 60
```

### 🔎 ***Explicação:***

- ```joystick_id``` → ID do controle conectado

- ```speed``` → Sensibilidade de movimento

- ```deadzone``` → Zona morta do analógico

- ```fps``` → Frequência de atualização

---

## Mapeamento Completo do Controle

### 🎯 ***Analógicos***

| Controle | Função padrão | Com LB/RB pressionado |
|----------|---------------|-----------------------|
| Analógico Esquerdo | Ombro esquerdo | Pulso esquerdo |
| Analógico Direito | Ombro direito | Pulso direito |

### 🦾 Braço Esquerdo

### 🎮 Sem pressionar LB:

- LS Y → Shoulder Pitch

- LS X → Shoulder Roll

- D-Pad ↑↓ → Cotovelo

- D-Pad ←→ → Shoulder Yaw

### 🎮 Pressionando LB:

- LS Y → Wrist Pitch

- LS X → Wrist Roll

### 🦾 Braço Direito
### 🎮 Sem pressionar RB:

- RS Y → Shoulder Pitch

- RS X → Shoulder Roll

- Y/A → Cotovelo

- X/B → Shoulder Yaw

### 🎮 Pressionando RB:

- RS Y → Wrist Pitch

- RS X → Wrist Roll

### ✋ Controle das Mãos (Dex3)
| Controle | Função |
|----------|--------|
| LT | Fecha mão esquerda |
| RT | Fecha mão direita |

Se o gatilho passar de 0.0 → mão fecha
Caso contrário → mão aberta

## 🔄 Lógica Interna do Controle

O código:

1. Lê os eixos do joystick

2. Aplica deadzone

3. Multiplica pelo speed

4. Soma na posição atual da junta

5. Envia todas as juntas como RobotAction

```bash
self.body_joints["kLeftShoulderPitch.q"] += ls_y * self.config.speed
```

Ou seja:

```Movimento é incremental, não absoluto.```

### 📊 Estrutura do Dataset Gerado

Cada episódio conterá:

- Observações

- Ações (todas as juntas do braço + mãos)

- Instrução textual

- Timestamp

Formato compatível com treinamento de VLA.

### 🛑 Como parar a gravação

Pressione:

```bash
CTRL + C
```

O dataset será finalizado corretamente.

---

## 🧪 Fluxo Completo de Uso

- 1️⃣ Criar pasta /Dados
- 2️⃣ Dar permissão 777
- 3️⃣ Conectar joystick
- 4️⃣ Rodar comando lerobot-record
- 5️⃣ Executar movimentos
- 6️⃣ Encerrar com CTRL+C

---

### 🧠 Observações Importantes

- Sempre centralize o robô antes de iniciar

- Teste o joystick com:

```bash
jstest /dev/input/js0
```

- Se aparecer erro de joystick:

    - Verifique se o pygame detecta controle

    - Verifique permissões de /dev/input