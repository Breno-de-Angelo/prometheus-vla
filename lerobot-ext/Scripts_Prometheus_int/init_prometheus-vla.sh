#!/bin/bash

# Caminho absoluto para a pasta do seu projeto
PROJECT_DIR=~/prometheus-vla

# 1. Inicializa o Conda usando o ativador exato do seu terminal
source ~/miniconda3/bin/activate

# 2. Ativa o ambiente
conda activate g1

# 3. Função de Segurança (Mata tudo quando você der Ctrl+C)
cleanup() {
    echo -e "\n\n🛑 [Ctrl+C] Pressionado! Encerrando os servidores..."
    # Mata os processos em segundo plano através dos PIDs
    kill $PID_CAM $PID_DEX
    echo "✅ Servidores desligados. Robô liberado."
    exit 0
}

# Prepara a armadilha para o sinal de interrupção (SIGINT)
trap cleanup SIGINT

echo "🤖 Iniciando infraestrutura do Prometheus VLA..."

# 4. Inicia o Servidor da Câmera (RealSense) em background (&)
python $PROJECT_DIR/full_realsenser_server.py &
PID_CAM=$!
echo "   [OK] Servidor RealSense ZMQ (PID: $PID_CAM)"

# Dá um respiro de 1 segundo para a câmera inicializar sem gargalar a USB/CPU
sleep 1

# 5. Inicia o Servidor da Mão (Dex3) em background (&)
python $PROJECT_DIR/dex3_g1_server.py &
PID_DEX=$!
echo "   [OK] Servidor Dex3 Bridge (PID: $PID_DEX)"

echo "-------------------------------------------------------"
echo "🚀 Sistema 100% online! Aguardando conexão do LeRobot."
echo "💡 Pressione [Ctrl + C] para finalizar tudo com segurança."
echo "-------------------------------------------------------"

# O comando 'wait' segura este terminal aberto monitorando os processos de fundo
wait
