# FastWAM-D — FastWAM com profundidade métrica

Extensão do [Fast-WAM](https://arxiv.org/abs/2603.16666) (Yuan et al., 2026) que
entrega ao modelo o **mapa de profundidade métrico** gravado pelo G1, em vez de
só a cor. Desenvolvida no Projeto Prometheus para o Unitree G1 com mãos Dex3-1.

Nada do fork do LeRobot é modificado: tudo aqui é subclasse. O `fastwam` de
origem continua funcionando e rebaseável.

---

## Por que

O FastWAM é um *world action model*: o expert de vídeo é o Wan2.2-TI2V-5B, e o
prior de cena vem de pré-treino em geração de vídeo. Isso dá boa noção de
aparência e de dinâmica, mas a distância continua sendo **inferida** de pistas
monoculares — o mesmo pixel pode ser a xícara a 40 cm ou a parede a 3 m.

O robô já grava profundidade métrica alinhada à câmera de cor (uint16 em
milímetros, mapa nativo do LeRobot 0.6.1). O que faltava era um caminho para
ela entrar no modelo.

**Aviso que não cabe em nota de rodapé:** isto melhora a *percepção* de
distância, não impõe limite nenhum ao que a política comanda. Não é mecanismo
de segurança. O que impede o braço de bater na mesa é o clamp em
`robot/unitree_g1/unitree_g1.py::send_action` (caixa de workspace, teto de
velocidade por junta, corte por torque).

---

## Como a profundidade entra

Três modos, selecionados por `depth_mode` no YAML. Os três existem de
propósito: `off` é o controle do experimento, `token` é a linha de base barata,
`latent` é a condição que se quer testar.

### `latent` (padrão) — correspondência espacial por token

```
depth (uint16 mm)  →  log → [0,1]  →  mosaico das câmeras  →  VAE do Wan  →  latente [B,48,T,h,w]
                                                                                   │
vídeo RGB → VAE do Wan → latente [B,48,T,h,w] ─────────────── concat no canal ─────┘
                                                     │
                                              patch_embedding (Conv3d 96→3072)
```

Cada token do DiT passa a carregar a distância **do próprio pedaço de imagem**
que ele representa.

Quatro decisões que sustentam isso:

**1. A profundidade passa pelo MESMO VAE do vídeo.** Assim o latente sai com a
mesma grade espacial e temporal do RGB, sem projetar um encoder novo nem
acertar fator de downsample na mão. O custo é que o VAE foi treinado em RGB
natural e um mapa de profundidade em cinza é fora da distribuição dele — é
aposta empírica, e é a principal a validar na athena.

**2. Normalização fixa e logarítmica, nunca por quadro.** `depth_min`/`depth_max`
são fixos no config. Normalizar por min/max de cada frame parece inofensivo e
destrói o que interessa: a escala vira relativa e o modelo aprende *contraste*
de profundidade em vez de *distância*. O log é o mesmo motivo pelo qual o
LeRobot grava em log (`datasets/depth_utils.py::quantize_depth`): o erro do
sensor cresce com a distância, e resolução perto vale mais para manipulação.

**3. O mosaico da profundidade espelha o das câmeras.** O FastWAM concatena as
câmeras lado a lado na largura, em ordem alfabética da chave. A profundidade é
montada no mesmo layout: cada câmera ocupa sua fatia, e a fatia recebe a
profundidade dela — ou zeros, se aquela câmera não tiver sensor. A regra que
liga as duas é o nome: `observation.images.head_camera_depth` pertence a
`observation.images.head_camera`. Entrar deslocado seria pior do que não ter
profundidade: o modelo aprenderia uma correspondência espacial falsa.

**4. Canais novos nascem em ZERO.** O `patch_embedding` é uma
`nn.Conv3d(48, 3072, ...)` que vira `nn.Conv3d(96, 3072, ...)`; os 48 primeiros
canais recebem os pesos pré-treinados e os 48 novos, zeros. No passo zero a
saída é **bit a bit a mesma** de antes — o prior de 5B fica intacto — e a
profundidade cresce a partir do treino. Inicializar com ruído (o padrão do
PyTorch) injetaria lixo num modelo já treinado logo no primeiro passo.

### `token` — linha de base

O latente da profundidade é reduzido a um vetor (média espacial/temporal),
projetado em `text_dim` e pendurado como **um** token no contexto da
cross-attention — o mesmo lugar por onde a propriocepção entra
(`wan/modular.py::_append_proprio_to_context`).

Barato e não mexe no `patch_embedding`. Também é bem mais fraco: o modelo
recebe "a cena tem esta geometria", não "este pedaço está a tantos
centímetros". Serve para medir quanto o modo `latent` está de fato ganhando.

### `off` — controle

Roda como o FastWAM de origem, pelo mesmo caminho de código. Sem isso, comparar
com o `fastwam` puro misturaria a diferença do enxerto com qualquer diferença de
configuração.

---

## Armadilhas resolvidas (leia antes de mexer)

**O `_stack_video_from_images` engole tudo.** O FastWAM varre todas as chaves
`observation.images.*` e concatena na largura. Com o mapa de profundidade no
batch, ou o `torch.cat` quebra pelos canais, ou (se alguém declarasse 3 canais)
ele entra como se fosse mais uma câmera de cor. Por isso a profundidade é
retirada do batch antes de chamar o caminho de origem.

**O loader de checkpoint descartaria o `patch_embedding`.** O
`_load_as_safetensor` do FastWAM ignora tensores com shape incompatível e deixa
o parâmetro no valor recém-inicializado. Como ampliamos a entrada de 48 para 96
canais, o shape não bate — e o modelo começaria com a Conv3d de patch **zerada**,
jogando fora o pré-treino de vídeo inteiro, em silêncio. A subclasse
sobrescreve o loader e **amplia** o tensor do checkpoint em vez de descartá-lo.

**O `freeze_video_expert` congelaria os canais novos.** Ele congela os 5B do
expert de vídeo, e os canais de profundidade nascem zerados: congelados,
ficariam zerados para sempre e o modelo treinaria sem nunca enxergar
profundidade. Daí o `depth_train_patch_embedding` (ligado por padrão), que
destrava só a Conv3d de patch.

**O `get_optim_params` não veria o enxerto.** O de origem devolve os parâmetros
do DiT e do encoder de propriocepção; sem sobrescrever, o `patch_embedding`
ampliado nunca receberia atualização.

**Profundidade ausente levanta erro, não vira zeros.** Se o gancho do
`patchify` não achar o latente de profundidade, ele estoura. Concatenar zeros
seria treinar às cegas sem nenhum sinal de que algo está errado.

---

## Como rodar

```bash
cd lerobot-ext
python -m policies.fastwam_depth.run_train \
    --config_path=config/train/fastwamdepth_white_cup_on_dripper.yaml
```

Ablação (mesmo YAML, três corridas):

```bash
--policy.depth_mode=latent   --output_dir=train_output/fwd_latent
--policy.depth_mode=token    --output_dir=train_output/fwd_token
--policy.depth_mode=off      --output_dir=train_output/fwd_off
```

Para partir do checkpoint LIBERO em vez do Wan cru, acrescente
`--policy.path=ZibinDong/fastwam_libero_uncond_2cam224`. Ele foi treinado com
ação de 7 dims; o loader do FastWAM já trata fine-tune entre embodiments
diferentes, descartando só os tensores de forma incompatível (encoder/cabeça de
ação e propriocepção).

**Requisitos de máquina:** o expert de vídeo tem ~5B — só os pesos em bf16
passam de 10 GB. Não roda no notebook de 8 GB; é código para a athena.

---

## O que foi verificado, e onde

Testado nesta máquina (sem GPU para os 5B — são testes de unidade, não de
treino):

- registro da política e resolução pelo `make_policy` (`fastwamdepth` →
  `FastWAMDepthPolicy`);
- `validate_features` aceitando o mapa de 1 canal e mantendo a regra de largura
  do FastWAM só para as câmeras de cor;
- mapeamento métrico: 0 e 10 mm → 0 (sem medida), 300 mm → 0,389,
  1000 mm → 0,651 (confere com o log fechado), ≥5000 mm → 1,0;
- mosaico `[B, 3, T, 224, 448]` com a metade da cabeça preenchida e a do pulso
  em zero;
- ampliação do `patch_embedding` 48 → 96 com **erro máximo 0,0** contra a saída
  original no passo zero (prior intacto), e influência não-nula depois de
  treinar os canais novos;
- corte temporal do latente na inferência (o expert de vídeo só vê o primeiro
  quadro);
- erro levantado quando a profundidade falta;
- ampliação do `patch_embedding` no carregamento de checkpoint.

**Falta rodar na athena:** um treino de verdade. Em especial, se o VAE do Wan
produz latente útil para um mapa de profundidade em cinza (a aposta da decisão
1) só se descobre olhando a curva de loss das três condições da ablação.

---

## Próximo passo natural: prever profundidade futura

O que este enxerto faz é **consumir** profundidade. O passo seguinte é
**prevê-la**: ampliar também o `out_dim` do `Head` (`wan/model.py:268`) e pôr os
canais de profundidade no alvo do `lambda_video`. Aí o modelo tem que aprender
como a geometria da cena evolui, que é a definição operacional de entender
espaço — prever profundidade força geometria de um jeito que consumir não força.

Isso exige destravar parte do expert de vídeo, o que com 23 episódios é convite
ao overfitting. Faz sentido quando o dataset tiver as outras etapas do café.
