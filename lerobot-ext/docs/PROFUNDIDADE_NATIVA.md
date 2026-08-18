# Profundidade nativa — do sensor ao modelo

> **O que mudou:** a profundidade deixou de trafegar e de ser gravada como
> imagem cinza de 8 bits e passou a ser **mapa métrico de 1 canal em
> milímetros**, do servidor no robô até a nuvem de pontos dentro da política.
>
> Isto atravessa cinco camadas e **as duas pontas têm que casar**. Mudar uma só
> não dá erro visível: dá dado errado em silêncio.

Data: 18/08/2026. Fecha o item "Depth em 16 bits de verdade — não feito" do
[SCHEMA_G1_V2.md](SCHEMA_G1_V2.md).

---

## 1. Por que mexer

O caminho antigo era:

```
z16 (mm) → clip(0, 2000mm) → ×255/2000 → uint8 → cv2.merge 3 canais → JPEG → h264
```

Três perdas empilhadas: o corte em 2 m, os 256 níveis (7,8 mm por degrau) e
duas compressões com perda. O que chegava na PointNet era uma nuvem de pontos
quantizada grosso, e a política aprendia com ela.

A 0.6.1 do LeRobot sabe guardar profundidade de verdade: uma câmera declarada
com **1 canal** vira mapa de profundidade (`info["is_depth_map"] = True`
automático), gravada em TIFF sem perda e encodada pelo `DepthEncoderConfig` —
HEVC `gray12le` lossless com quantização logarítmica de 12 bits, e o inteiro
lido como **milímetro** (`infer_depth_unit`).

Medido no round-trip: **500 mm → 500,00 | 700 mm → 701,00**. Erro de ~1 mm, que
é a resolução do próprio sensor.

---

## 2. O caminho completo

```
RealSense D435i (z16, alinhada à cor)
   │  full_realsenser_server.py      ← uint16 mm CRU, sem clip, sem escalar
   │  ZMQ multiparte: JSON + buffer binário
   ▼
ZMQCamera (fork do LeRobot)          ← decodifica por payload; devolve (H,W,1) uint16
   │
   ▼
UnitreeG1._cameras_ft                ← declara (H, W, 1) para câmera *_depth
   │
   ▼
hw_to_dataset_features               ← marca is_depth_map=True sozinho
   │
   ▼
LeRobotDataset                       ← TIFF lossless → HEVC gray12le lossless
   │                                   info: {"is_depth_map": true, "depth_unit": "mm"}
   ▼
__getitem__                          ← float32 [1, H, W] em MILÍMETROS
   │
   ▼
depth_to_pointcloud(depth_unit="mm") ← × 0,001 → metros → projeção pinhole
```

### 2.1 Servidor (roda no robô)

`Scripts_Prometheus_int/full_realsenser_server.py`, em
`~/Script_Prometheus_int/` no robô — **não** é a cópia em
`~/prometheus-vla/lerobot-ext/`, que é de abril e não é usada.

A profundidade vai crua em quadro binário próprio da mensagem ZMQ. O RGB
continua base64 JPEG dentro do JSON, sem mudança — é o que mantém a
teleoperação funcionando.

**Por que crua e não PNG de 16 bits:** medido no próprio robô, o PNG custa
**20 ms por quadro**, dois terços do orçamento de 33 ms a 30 fps, com o RGB e o
align da RealSense ainda para pagar. Cru custa zero de CPU e são 814 KB por
quadro (24 MB/s a 30 fps), ~20% de um enlace gigabit — e robô e PC estão os
dois em `eth0` a 1000 Mb/s. Aqui a rede é o recurso de sobra, a CPU não.

O servidor lê o `depth_scale` da câmera e converte para milímetro em vez de
assumir 0,001 — é calibração de fábrica, não garantia.

### 2.2 Transporte (`sim/sensor_utils.py`, 3 cópias idênticas)

- `SensorServer.send_message(data, parts=[...])` manda multiparte. Sem `parts`,
  a mensagem sai com um quadro só, byte a byte igual à antiga.
- `SensorClient.receive_message()` lê multiparte e resolve as imagens que vieram
  em quadro binário já como `np.ndarray`.
- `ImageUtils.encode_raw(img, part)` devolve `(descritor, bytes)`; o descritor
  (`{"part": 1, "dtype": "<u2", "shape": [H, W]}`) viaja no JSON.

**A armadilha que custou caro:** `ZMQ_CONFLATE` é **incompatível com mensagem
multiparte**. Com ele ligado o socket entrega uma parte só — e nem é a
primeira: chega o buffer da profundidade sem o JSON, e o `json.loads` estoura.
O `SensorClient` perdeu o CONFLATE e o efeito dele ("ficar só com o quadro mais
novo") foi reimplementado esvaziando a fila no `receive_message`. O próprio
LeRobot documenta a mesma limitação no `robots/lekiwi/lekiwi_host.py`.

### 2.3 Cliente (fork do LeRobot, `cameras/zmq/camera_zmq.py`)

O `_decode_zmq_images` passou a despachar **por imagem**, não pelo `protocol`
da mensagem: payload `dict` → quadro binário; payload `str` → base64 legado.
É isso que deixa RGB em base64 e profundidade em binário conviverem na mesma
mensagem — sem isso, ou a profundidade perderia os 16 bits, ou a teleoperação
(que fala o formato antigo pelo `sensor_utils.py`) pararia de entender o RGB.

### 2.4 Robô (`robot/unitree_g1/unitree_g1.py` e `unitree_g1_loco.py`)

`_cameras_ft` declara **1 canal** para câmera cujo nome termina em `depth`, 3
para as demais. O número de canais aqui não é papelada: é ele que decide como o
LeRobot grava a câmera.

### 2.5 Configs de gravação

`dataset.vcodec` não existe mais na 0.6.1 e o draccus **recusa o YAML inteiro**
(`The fields vcodec are not valid`). Passou para `dataset.rgb_encoder.vcodec` em
todos os 10 YAMLs de `config/record/`. Profundidade não passa por aí: usa o
`depth_encoder` só dela (HEVC gray12le sem perda).

---

## 3. O lado do modelo

### 3.1 O erro de três ordens de grandeza

`depth_to_pointcloud` fazia `z = tensor * 2.0`, porque o tensor chegava em
[0,1] com 1.0 = 2 m. Com o mapa nativo, o tensor chega em **milímetros**:

```
depth do batch: float32 (B,1,480,848), 10–1787 mm
  depth_unit=mm  →  Z 0,27–1,76 m     ✅ cena de mesa
  fator antigo (× 2.0)  →  Z ≈ 1094 m
```

O fator errado **não quebra nada visivelmente**: o treino roda, a loss cai, e o
modelo aprende com a cena a 1 km de distância. Agora a unidade vem do config
(`depth_unit: mm`) e a função rejeita valor desconhecido.

Corrigido nas **três cópias** do arquivo: `policies/act_depth/depth_encoder.py`,
`policies/pi0_depth/depth_encoder.py` e `train/depth_encoder.py`.

### 3.2 Filtro da nuvem

`0,05 m < z < z_max` (padrão 5 m). O piso porque pixel sem medida volta da
desquantização como o próprio `depth_min` (1 cm), não como zero — sem o piso
viraria uma parede falsa colada na lente. O teto porque a RealSense devolve
pixels saturados (o dataset gravado tem `max` de 65 m) e um punhado deles
domina a escala da nuvem.

### 3.3 Três remendos que viraram dead code

A 0.6.1 passou a fazer nativamente o que era feito na mão. Mantidos, os três
iriam de inúteis a **errados** (aplicam mean/std de 3 canais num mapa de 1):

| Remendo | Onde vivia | Por que saiu |
|---|---|---|
| Reversão da normalização ImageNet | `act_depth`, `pi0_depth`, `openvla_depth` | `datasets/factory.py`: `if key in depth_keys: continue` |
| Override das stats do depth por identidade | `run_train.py` de `act_depth` e `pi0_depth` | idem — e apagava as stats reais em mm, que o visualizador usa |
| Monkeypatch do `__getitem__` | os 3 `run_train.py` | o reader já remapeia índice e já pula depth nas transformações |

O do `__getitem__` não era só inútil: chamava `self._ensure_hf_dataset_loaded()`,
método que não existe mais — **matava o treino no arranque**.

### 3.4 Inferência

Os quatro `init_lerobot_inference_*.py` replicavam a profundidade em 3 canais e
dividiam por 255. Isso **não estourava** — entregava milímetros divididos por
255 a uma política que espera milímetros. Erro silencioso, o pior tipo. Agora
todos usam `_depth_to_tensor(depth)` → `[1, H, W]` float32 em mm.

Os caminhos de replay com vídeo/imagem "fake" leem datasets **antigos** (8 bits,
0–255 ↔ 0–2000 mm) e desfazem a escala para entregar milímetros.

---

## 4. Intrínsecos da câmera

Todos os configs usavam `cx: 320, cy: 240` — o centro de uma imagem de **640 px**
de largura. O dataset é gravado a **848×480**, onde o ponto principal fica em
~424. São ~104 px de erro, e a projeção pinhole (`x = (u - cx)·z/fx`) transforma
isso em cisalhamento da nuvem proporcional à distância: o encoder 3D aprende a
compensar dentro do dataset e o erro reaparece quando a cena muda.

O padrão passou para `fx=fy=617, cx=424, cy=240`, que vem do FOV nominal do
D435i e é **estimativa**. Para os reais, rode **no robô**:

```bash
python Scripts_Prometheus_int/print_camera_intrinsics.py
```

---

## 5. O que foi verificado

Com a câmera real do robô, lendo daqui pelo LeRobot:

```
RGB   : uint8  (480, 848, 3)
DEPTH : uint16 (480, 848, 1)
medidas válidas: 93,5% | min 624 mm | mediana 1218 mm | max 2097 mm
leitura sustentada: 30,3 fps
```

O `max` de 2097 mm é a prova de que o corte em 2000 mm morreu.

Dataset gravado (27 episódios, 12.848 quadros):

```
observation.images.head_camera_depth   video   (480, 848, 1)   is_depth_map: true
info: {"is_depth_map": true, "depth_unit": "mm", "video.codec": "hevc",
       "video.pix_fmt": "gray12le", "video.extra_options": {"x265-params": "lossless=1"}}
```

Round-trip de gravação/leitura: 500 mm → 500,00; 700 mm → 701,00.
Custo de encode: ~28 s por episódio de 60 s no pior caso (ruído puro).

---

## 6. O que ainda não está resolvido

**Os sensores de pressão gravam zero.** `observation.left_hand_pressure` e
`right_hand_pressure` são identicamente zero no `stats.json`
(`min = max = mean = std = 0` nos 33 canais de cada mão). As colunas existem, o
sinal não. Por isso `use_pressure: false` nos configs de treino: ligar só gasta
um token com uma constante e dá a impressão falsa de que a fusão tátil está
ativa. Investigar a porta 6002 do Dex3 antes da próxima gravação.

**Intrínsecos reais.** Ver §4 — os atuais são nominais.

**O servidor de câmera da simulação MuJoCo** (outro repositório) ainda publica
profundidade em 3 canais. Gravar em simulação quebra contra a declaração de
1 canal.

**Episódio 26 do dataset** tem 6 quadros (0,2 s) — tomada abortada. Está fora
das listas de `episodes` dos YAMLs de treino.
