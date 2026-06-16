import pandas as pd
import numpy as np
import os

# Caminho do seu arquivo (ajuste se necessário)
caminho_parquet = "meu_dataset/get_the_kettle4/data/chunk-000/file-000.parquet"

def analisar_acoes():
    if not os.path.exists(caminho_parquet):
        print(f"❌ Arquivo não encontrado: {caminho_parquet}")
        return

    print(f"🧐 Analisando ações em: {caminho_parquet}\n")
    
    # Carrega o Parquet
    df = pd.read_parquet(caminho_parquet)
    
    if "action" not in df.columns:
        print("❌ ERRO: A coluna 'action' não existe neste dataset.")
        return

    # Converte a coluna para uma matriz numpy para análise rápida
    # O LeRobot salva a ação como uma lista/array em cada célula
    acoes = np.stack(df["action"].values) # Shape: (Total_Frames, 43)
    
    total_frames = acoes.shape[0]
    num_juntas = acoes.shape[1]
    
    # 1. Checagem Global
    soma_total = np.sum(np.abs(acoes))
    
    print(f"📊 Estatísticas Rápidas:")
    print(f"   - Total de frames analisados: {total_frames}")
    print(f"   - Total de juntas monitoradas: {num_juntas}")
    print("-" * 40)

    if soma_total == 0:
        print("🚨 RESULTADO: O ACTION ESTÁ 100% ZERADO!")
        print("   O robô não recebeu NENHUM comando de movimento durante a gravação.")
        print("   Verifique se a janela do Teleop (Pygame) estava em foco.")
    else:
        print("✅ RESULTADO: FORAM DETECTADOS MOVIMENTOS!")
        
        # 2. Descobrir quais juntas se mexeram
        juntas_com_movimento = []
        for j in range(num_juntas):
            coluna_junta = acoes[:, j]
            if np.max(coluna_junta) != np.min(coluna_junta):
                juntas_com_movimento.append(j)
        
        print(f"   - Juntas que se mexeram: {len(juntas_com_movimento)} de {num_juntas}")
        if juntas_com_movimento:
            print(f"   - Índices das juntas ativas: {juntas_com_movimento}")
            
            # Mostra o valor máximo de movimento detectado para te dar uma ideia da força
            valor_max = np.max(np.abs(acoes))
            print(f"   - Amplitude máxima de movimento: {valor_max:.4f}")

    print("-" * 40)
    print("💡 Dica: Se o 'observation.state' tem dados e o 'action' está zerado,")
    print("   o robô está lendo os sensores, mas o comando do Teleop não está chegando.")

if __name__ == "__main__":
    analisar_acoes()