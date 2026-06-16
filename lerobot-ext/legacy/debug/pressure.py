import zmq
import json
import time

# Configurações do seu G1
ROBOT_IP = "192.168.123.164"
PORT = 6002  # Porta do HANDSTATE definida na sua ponte

def testar_conexao():
    context = zmq.Context()
    sock = context.socket(zmq.SUB)
    
    # Conecta no IP do robô
    addr = f"tcp://{ROBOT_IP}:{PORT}"
    print(f"📡 Conectando em {addr}...")
    sock.connect(addr)
    
    # Inscreve nos tópicos de estado das mãos
    sock.subscribe("") # Ouve tudo que vier na porta 6002

    print("🕵️ Aguardando dados da ponte (Aperte o sensor da mão agora)...")
    
    try:
        while True:
            # Tenta receber com timeout de 2 segundos
            if sock.poll(2000) & zmq.POLLIN:
                payload = sock.recv()
                msg = json.loads(payload.decode("utf-8"))
                
                topic = msg.get("topic", "N/A")
                data = msg.get("data", {})
                side = data.get("side", "N/A")
                
                # AQUI ESTÁ O QUE VOCÊ QUER VER:
                sensors = data.get("press_sensor_state", [])
                
                if sensors:
                    # Pega a pressão do primeiro sensor só para teste rápido
                    p_amostra = sensors[0]["pressure"][:5] 
                    print(f"✅ [{topic}] Mão {side} | Total Áreas: {len(sensors)} | Amostra Pressão: {p_amostra}")
                    
                    # Se você apertar e o valor mudar de 0.0, a ponte está OK!
                    max_p = 0
                    for area in sensors:
                        max_p = max(max_p, max(area["pressure"]))
                    
                    if max_p > 0:
                        print(f"🔥 TOQUE DETECTADO! Pressão Máxima: {max_p:.2f}")
                else:
                    print(f"⚠️ [{topic}] Recebi dados, mas 'press_sensor_state' veio VAZIO.")
            else:
                print("⌛ Sem resposta... A ponte_mao.py está rodando no robô?")
                
    except KeyboardInterrupt:
        print("\nTeste encerrado.")
    finally:
        sock.close()
        context.term()

if __name__ == "__main__":
    testar_conexao()