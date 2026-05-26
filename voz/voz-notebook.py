import pyaudio
import time
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

NETWORK_INTERFACE = 'eno1' 

def main():
    print("--- Transmissor de Áudio do Sistema para o G1 ---")
    
    try:
        ChannelFactoryInitialize(0, NETWORK_INTERFACE)
        print(f"[OK] Conexão estabelecida via {NETWORK_INTERFACE}")
    except Exception as e:
        print(f"[ERRO] Falha ao inicializar comunicação: {e}")
        sys.exit(1)

    audio_client = AudioClient()
    audio_client.Init()
    audio_client.SetTimeout(10.0)
    audio_client.SetVolume(100)
    print("[OK] AudioHub Client iniciado.")

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000 
    CHUNK = 1024
    
    p = pyaudio.PyAudio()

    session_stream_id = str(time.time_ns())

    def pyaudio_callback(in_data, frame_count, time_info, status):
        try:
            pcm_data = list(in_data)
            # Alteramos o app_name para pc_audio para organizar
            audio_client.PlayStream("pc_audio", session_stream_id, pcm_data)
        except Exception as e:
            pass
            
        return (None, pyaudio.paContinue)

    try:
        # Abrimos o canal padrão de entrada do sistema
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        stream_callback=pyaudio_callback)
        
        print("\n🎵 Capturando áudio do PC e enviando para o G1...")
        print("Pressione [Ctrl+C] para encerrar.\n")
        
        while stream.is_active():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] Encerrando pelo usuário.")
    except Exception as e:
        print(f"\n[ERRO] {e}")
    finally:
        audio_client.PlayStop("pc_audio")
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("Finalizado.")

if __name__ == '__main__':
    main()