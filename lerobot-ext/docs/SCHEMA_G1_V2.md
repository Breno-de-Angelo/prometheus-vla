# Schema G1 v2 — yaw do tronco e câmera de pulso

> Mudança de schema do robô: o vetor de ação passa de 28 para 29 dimensões e
> entra uma câmera no pulso direito. Datasets gravados antes e depois disto
> **não podem ser juntados** (`lerobot/datasets/aggregate.py:47`).

---

## 1. O novo schema

| | v1 (atual) | v2 |
|---|---|---|
| ação / estado | 28 | **29** |
| braços | dims 0–13 | dims 0–13 *(inalterado)* |
| **yaw do tronco** | — | **dim 14** |
| mãos Dex3 | dims 14–27 | dims 15–28 |
| câmeras | `head_camera`, `head_camera_depth` | + **`right_wrist_camera`** |

O yaw é declarado por último no `G1_29_JointArmWaistIndex`. O `IntEnum` do Python
itera na ordem de **definição**, não de valor — então os braços mantêm os índices
0–13 que a documentação de `loss_per_dim` já usa, e só as mãos deslocam em 1.

### Ligar

```python
UnitreeG1Dex3Config(
    use_waist_yaw=True,      # 28 → 29 dims
    use_wrist_camera=True,   # + right_wrist_camera 224×224
)
```

Ambos vêm `False` por padrão: sem eles, o comportamento é idêntico ao de antes.

---

## 2. Yaw do tronco

### 2.1 Por que roll e pitch ficam travados

No código anterior, `upper_body` e `high_level` cediam as juntas 12, 13 e 14 ao
WBC com `mode=0, kp=0, kd=0`. Na prática **ninguém assumia**: observado no robô
real, a cintura ficava mole e o tronco balançava sozinho enquanto o G1 andava.

O tratamento agora é assimétrico:

| junta | motor | tratamento |
|---|---|---|
| `kWaistYaw` | 12 | **comandado** — entra no vetor de ação |
| `kWaistRoll` | 13 | **travado** em 0.0 com `waist_lock_kp=100, kd=3` |
| `kWaistPitch` | 14 | **travado** em 0.0 |
| pernas | 0–11 | cedidas ao WBC (ali ele de fato atua) |

Aplicado nos dois caminhos: `unitree_g1.py` (lowlevel, `rt/lowcmd`) e
`unitree_g1_loco.py` (loco, `rt/arm_sdk`).

### 2.2 Limite de curso

`waist_yaw_limit: float = 1.0` (rad). Aplicado em `send_action` nos dois modos,
antes do filtro de suavização. Bater no fim de curso no meio de uma demonstração
estraga o episódio e castiga o motor.

### 2.3 Fonte única da verdade

Antes, cinco métodos repetiam `G1_29_JointArmIndex if control_mode == "upper_body"
else G1_29_JointIndex`. Agora existe `UnitreeG1.body_joint_index`, e acrescentar
uma junta é mexer num lugar só.

| `control_mode` | `use_waist_yaw` | enum | dims |
|---|---|---|---|
| `upper_body` | False | `G1_29_JointArmIndex` | 14 |
| `upper_body` | True | `G1_29_JointArmWaistIndex` | 15 |
| `high_level` | True | `G1_29_JointArmWaistIndex` | 15 |
| `full_body` | qualquer | `G1_29_JointIndex` | 29 |

---

## 3. Câmera de pulso direito

### 3.1 Por que 224×224

O `_preprocess_images` do OpenVLA-Depth redimensiona todo RGB para 224×224 antes
da torre visual. Gravar maior gasta disco e banda sem entregar nada ao modelo.

E há um detalhe que morde: uma fonte 16:9 redimensionada para quadrado fica
**deformada**. No MuJoCo a câmera já é quadrada; se o robô real publicasse 424×240
esticado, treino e inferência veriam geometrias diferentes — erro silencioso, que
não dá exceção nenhuma e só aparece como desempenho ruim.

Por isso o servidor faz **corte central quadrado antes do resize**:

```python
lado = min(h, w)
y0, x0 = (h - lado) // 2, (w - lado) // 2
img = cv2.resize(img[y0:y0+lado, x0:x0+lado], (224, 224), cv2.INTER_AREA)
```

A D435 não tem 224×224 nativo; o menor modo é 424×240, que é o que capturamos.

### 3.2 Onde ela é publicada

| origem | como |
|---|---|
| robô real | `Scripts_Prometheus_int/right_arm_realsense_server.py` (serial `141722078588`), ZMQ `right_wrist_camera` |
| MuJoCo | câmera `right_wrist_camera` no `g1_29dof_with_hand.xml`, presa a `right_wrist_yaw_link` |

Os dois publicam com o **mesmo nome** e o **mesmo tamanho** — é isso que permite
gravar no simulador e no robô com um schema só.

---

## 4. MuJoCo: o ponto de suporte

`config.yaml` passou de `enable_waist: False` para `True` — a faixa elástica sai
do `torso_link` e vai para o `pelvis`.

Não é preferência: a `ElasticBand` aplica um **PD de orientação** no corpo preso
(`unitree_sdk2py_bridge.py:481`):

```python
torque = -self.kp_ang * rotvec - self.kd_ang * ang_vel    # kp_ang = 1000
```

Com a faixa no tronco, esse torque prende a orientação do tronco **no mundo**.
Medido, comandando 0.8 rad de yaw:

| faixa em | junta `waist_yaw` | tronco no mundo |
|---|---|---|
| `torso_link` (antes) | **+1.925 rad (110°)** — 240% do comando | **0.0°** |
| `pelvis` (agora) | +0.431 rad (24.7°) | **+23.2°** |

Ou seja, o problema não era só "o tronco não gira": o atuador empurrava contra a
faixa, quem girava era o pelvis ao contrário, e a junta enrolava até 110° para um
comando de 46°. Como sinal de treino isso seria veneno — o dataset registraria um
ângulo de junta que não corresponde a nenhum movimento real.

---

## 5. Arquivos alterados

| arquivo | mudança |
|---|---|
| `robot/unitree_g1/g1_utils.py` | `G1_29_JointArmWaistIndex`, `G1_WAIST_LOCKED_JOINTS` |
| `robot/unitree_g1/config_unitree_g1.py` | `use_waist_yaw`, `waist_lock_kp/kd`, `waist_yaw_limit` |
| `robot/unitree_g1/unitree_g1.py` | `body_joint_index`; trava de roll/pitch; clamp do yaw; 5 condicionais unificados |
| `robot/unitree_g1/unitree_g1_loco.py` | mesma lógica no caminho do `arm_sdk` |
| `robot/unitree_g1/unitree_g1_dex3.py` | `use_wrist_camera`, `wrist_cam_size`, câmera no dict |
| `Scripts_Prometheus_int/right_arm_realsense_server.py` | 424×240 + corte central → 224×224 |
| `unitree-g1-mujoco/config.yaml` | `enable_waist: True` |
| `unitree-g1-mujoco/assets/g1_29dof_with_hand.xml` | câmera `right_wrist_camera` |
| `unitree-g1-mujoco/run_sim.py` | resolução por câmera; pulso na lista padrão |

---

## 6. O que foi validado, e como

Tudo abaixo rodou nesta máquina:

- ✅ **Ordem do vetor**: os 14 primeiros nomes de `G1_29_JointArmWaistIndex` são
  idênticos aos de `G1_29_JointArmIndex`; `kWaistYaw` é o 15º e tem valor 12.
- ✅ **Seleção de enum** nos 6 cruzamentos de `control_mode` × `use_waist_yaw`,
  com as contagens de dimensão (14 / 15 / 29).
- ✅ **Config do robô**: `use_wrist_camera=True` acrescenta a câmera a 224×224;
  `False` não deixa rastro.
- ✅ **Modelo MuJoCo recarrega** com a câmera nova; ela está presa a
  `right_wrist_yaw_link` e desloca 0.258 m quando o ombro gira 0.9 rad.
- ✅ **Render real** da câmera de pulso em 224×224.
- ✅ **Ponto de suporte**: simulação comparando `torso_link` e `pelvis` (tabela da
  seção 4).
- ✅ **Corte central preserva geometria**: um círculo continua com 113 px de
  largura e 113 de altura depois de 424×240 → 224×224.
- ✅ Todos os arquivos compilam.

**Não validado — exige hardware:** o comportamento real do travamento de roll/pitch
com o WBC ativo, o curso real do `kWaistYaw` no G1, e o serial/foco da câmera de
pulso. Os ganhos `waist_lock_kp=100 / kd=3` são um ponto de partida conservador e
devem ser conferidos com o robô suspenso antes de qualquer demonstração.

---

## 7. O que ainda falta

| item | estado |
|---|---|
| Controle do tronco por teclado (sem VR) | **não feito** |
| Depth em 16 bits de verdade | **não feito** — ver abaixo |
| Sim no PC + inferência na atena | parcial: `run_sim_remote.py` já faz a ponte ZMQ; falta documentar e testar ponta a ponta |

### Sobre o depth

A INJEÇÃO 2 do `init_lerobot_record_v2.py` marca
`dataset_features[...]["info"]["video.is_depth_map"] = True`. Isso é **metadado
cosmético**: o vídeo continua sendo gravado em **h264 `yuv420p`, 8 bits com perda**.
O `full_realsenser_server.py` já corta em 2000 mm e escala para 0–255 antes de
publicar, então a profundidade tem ~7,8 mm de resolução e passa ainda por
compressão com perda.

Ou seja: a flag não faz o LeRobot guardar depth real, e o modelo está aprendendo
com uma nuvem de pontos quantizada. Consertar de verdade significa mudar o formato
de gravação (16 bits sem perda), não a flag — é uma mudança de schema também, e
deve entrar junto com esta se for entrar.
