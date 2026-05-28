# 📚 Documentação da Arquitetura: ACT-D (Depth-Augmented ACT com Fusão Tátil)
**Robô:** Unitree G1 | **Efetores:** Dex3 Hands | **Base:** LeRobot v0.4.4

## 1. Visão Geral da Arquitetura
O pipeline original do **ACT (Action Chunking with Transformers)** foi projetado para receber apenas imagens 2D (RGB) e a posição dos motores dos braços. 

Para resolver o problema de tarefas complexas de manipulação (como preensão preditiva de uma xícara) com poucos dados (regime de ~30 demonstrações), nós expandimos a arquitetura para o **ACT-D**. Este sistema é "Multimodal", o que significa que o "cérebro" do robô toma decisões fundindo três sentidos diferentes simultaneamente:
1. **Visão Espacial (RGB 2D)**
2. **Visão Geométrica (Profundidade 3D)**
3. **Percepção Proprioceptiva e Tátil (Motores + Pressão)**

## 2. O Fluxo de Dados (Data Ingestion)
Tudo começa com o arquivo de configuração `train_push_cup.yaml`. Quando o treinamento inicia, o Dataloader do LeRobot lê os arquivos Parquet (`.parquet`) e os vídeos (`.mp4`) gravados durante a sua teleoperação no Vuer VR.

Em vez de jogar tudo em um "liquidificador" numérico, o dataset garante que as grandezas entrem separadas:
* `observation.images.head_camera`: Matriz de pixels coloridos [3, 480, 640].
* `observation.images.head_camera_depth`: Matriz de distância [3, 480, 640].
* `observation.state`: Vetor com os 28 ângulos dos motores dos braços e dedos.
* `observation.left_hand_pressure` / `right_hand_pressure`: Vetores de 33 sensores de força cada.

## 3. O Coração do Sistema: A Injeção Multimodal (Monkey Patch)
Como o LeRobot v0.4.4 não suporta nativamente mapas de profundidade 3D ou sensores de pressão densos, nós aplicamos um *Bypass* Arquitetural usando o script `act_d_injector.py` em conjunto com o `depth_encoder.py`. 

No exato momento em que o modelo ACT é criado na memória do PyTorch, o nosso script intercepta o modelo e altera a sua biologia antes do treinamento começar. Este bypass ocorre em 3 etapas durante o método `forward`:

### A. O Caminho da Visão (ResNet)
A imagem RGB colorida passa normalmente pelo backbone da ResNet original do modelo ACT, que extrai as características visuais em um conjunto de *Tokens Visuais*. Para proteger essa rede de "alucinar", o nosso script remove (com `.pop()`) a imagem de profundidade do pacote antes que a ResNet a veja.

### B. O Caminho da Geometria 3D (PointNet)
A imagem de profundidade que nós "roubamos" é enviada para o módulo `depth_to_pointcloud`.
1. **Reversão Matemática:** O script desfaz o hack do driver ZMQ multiplicando os pixels por `2.0` para recuperar as medidas exatas em metros.
2. **Nuvem de Pontos:** A imagem é projetada no espaço 3D usando a matriz intrínseca da câmera RealSense D435i, criando uma nuvem de coordenadas (X, Y, Z).
3. **PointNet:** Fazemos uma amostragem (subsampling) de 1024 pontos para otimizar a memória da GPU. Essa nuvem de 1024 pontos passa pela nossa rede `PointNetEncoder` (baseada no 3D-CAVLA), que extrai a forma do objeto e gera um **Token 3D** global.

### C. O Caminho Tátil (Dex3)
As matrizes de pressão da mão esquerda (33) e direita (33) são concatenadas num vetor de 66 posições. Este vetor cru passa por uma Rede Neural Perceptron de Múltiplas Camadas (MLP / `nn.Linear`), que "comprime" a sensação de força num **Token de Pressão**.

### D. A Fusão Suprema (Self-Attention do Transformer)
O modelo ACT original gerava apenas um Token de Estado contendo os 28 motores.
O nosso Monkey Patch intercepta esse token e **soma** o *Token 3D* e o *Token de Pressão* a ele. 

Esse "Super-Token Multimodal" é finalmente jogado para dentro da arquitetura Transformer do ACT (Encoder). Graças ao mecanismo de *Self-Attention*, o modelo agora consegue cruzar as informações: ele "entende" que quando os pixels RGB mostram a xícara, a nuvem de pontos 3D diz a que distância ela está, e a pressão tátil confirma se os dedos já a tocaram.

## 4. O Motor de Treinamento (`run_train.py`)
Com a rede neural mutante pronta na memória, o `run_train.py` assume o controle repetindo o seguinte ciclo até atingir os `10000 steps`:

1. **Forward Pass:** O Dataloader entrega um *Batch* (lote de 16 demonstrações) para o modelo. A IA prevê quais seriam as ações (movimentos dos motores) para os próximos instantes (Action Chunking).
2. **Cálculo de Perda (Loss):** O modelo compara o movimento que ele previu com o movimento que você (o humano) realmente fez no VR. A diferença entre os dois é o "Erro" (Loss).
3. **Backpropagation:** Usando o otimizador (AdamW), o PyTorch ajusta os pesos matemáticos da ResNet, da PointNet, da camada de pressão e do Transformer para tentar errar menos na próxima vez.
4. **Avaliação (Prova Surpresa):** A cada 500 passos, o script pausa o treinamento e puxa o `val_dataset` (episódios 8 e 9 que a IA nunca viu). Ele tenta prever os movimentos e calcula o `val_loss`. Isso é crucial para garantir que a IA está aprendendo a *generalizar* a física do mundo, e não apenas "decorando" as imagens.

## 5. Resumo do Fluxo de Execução (O que acontece quando você dá o Play)
1. Você digita: `python init_lerobot_train_v3.py --config_path=config/train/train_push_cup.yaml`
2. O script carrega as ferramentas da versão do LeRobot e localiza o seu dataset no HD.
3. O PyTorch cria a matriz do ACT básico na GPU.
4. O `act_d_injector` entra em cena, acopla a PointNet 3D, a rede de Pressão e altera as rotas de conexão interna.
5. O loop de treinamento (`run_train.py`) inicia as iterações de aprendizagem.
6. A cada 10.000 passos, a "mente" do robô é salva na pasta `train_output/act_d_push_cup/checkpoints`.

Essa infraestrutura coloca a sua pesquisa num patamar de estado da arte, resolvendo gargalos clássicos da robótica (como a falta de propriocepção tátil e a cegueira de profundidade das redes 2D).