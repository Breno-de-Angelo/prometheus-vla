# Prometheus VLA

## Instalar as dependências

1. Clonar repositório com git clone --recurse-submodules https://github.com/Breno-de-Angelo/prometheus-vla (Caso tenha clonado sem --recurse-submodules, rode git submodule update --init --recursive)
2. Instalar uv
3. uv sync

## Visualizar o dataset
No próprio computador:
1. Rode o comando:
    uv run lerobot-dataset-viz --repo-id=Breno-de-Angelo/unitree-g1-dex3-1-pick-kettle-v3 --episode-index=0 --display-compressed-images=true

No computador remoto:
1. Rode o comando:
    uv run lerobot-dataset-viz --repo-id=Breno-de-Angelo/unitree-g1-dex3-1-pick-kettle-v3 --episode-index=0 --display-compressed-images=true --mode=distant
2. Rode no seu computador (é necessário instalar o rerun):
    rerun ws://localhost:9087

## Rodar treinamento
1. Defina os parâmetros do treinamento criando um novo arquivo em train/config/nome_do_treino.yaml
2. Rode o treinamento com o comando:
    CUDA_VISIBLE_DEVICES=1 nohup uv run lerobot-train --config train/config/nome_do_treino.yaml > train/log/nome_do_treino.log 2>&1 &
3. Para verificar o treinamento, use o comando:
    tail -f train/log/nome_do_treino.log


