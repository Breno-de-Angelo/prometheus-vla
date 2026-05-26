import pyaudio
import time
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

NETWORK_INTERFACE = 'eno1' 

def main():
    print("--- Driver de Áudio G1 (Python Puro) ---")
    
    try:
        ChannelFactoryInitialize(0, NETWORK_INTERFACE)
        print(f"[OK] Conexão SDK estabelecida via interface: {NETWORK_INTERFACE}")
    except Exception as e:
        print(f"[ERRO] Falha ao inicializar comunicação: {e}")
        sys.exit(1)

    audio_client = AudioClient()
    audio_client.Init()
    audio_client.SetTimeout(10.0)
    audio_client.SetVolume(100)
    print("[OK] AudioHub Client iniciado e volume no máximo.")

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000 
    CHUNK = 1024
    
    p = pyaudio.PyAudio()

    # CORREÇÃO: Geramos o ID do stream UMA ÚNICA VEZ para toda a transmissão
    session_stream_id = str(time.time_ns())

    def pyaudio_callback(in_data, frame_count, time_info, status):
        try:
            pcm_data = list(in_data)
            # Usamos o mesmo ID fixo para todos os pacotes sequenciais
            audio_client.PlayStream("py_mic", session_stream_id, pcm_data)
        except Exception as e:
            pass
            
        return (None, pyaudio.paContinue)

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        stream_callback=pyaudio_callback)
        
        print("\n🎙️  Capturando voz e enviando para o G1 ao vivo...")
        print("Pressione [Ctrl+C] para encerrar.\n")
        
        while stream.is_active():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] Encerrando pelo usuário.")
    except Exception as e:
        print(f"\n[ERRO] {e}")
    finally:
        print("Parando fluxo de áudio no robô...")
        audio_client.PlayStop("py_mic")
        
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("Finalizado.")

if __name__ == '__main__':
    main()