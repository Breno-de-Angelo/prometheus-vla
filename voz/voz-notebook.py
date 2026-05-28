import pyaudio
import time
import sys
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

NETWORK_INTERFACE = 'eno1' 

def main():
    print("--- Transmissor de Áudio + Equalizador LED para o G1 ---")
    
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
    
    # --- A CORREÇÃO ESTÁ AQUI ---
    # Variável compartilhada para guardar: [Graves, Médios, Agudos]
    freq_data = [0, 0, 0]

    def pyaudio_callback(in_data, frame_count, time_info, status):
        try:
            pcm_data = list(in_data)
            audio_client.PlayStream("pc_audio", session_stream_id, pcm_data)
            
            # --- MÁGICA DO EQUALIZADOR (FFT) ---
            audio_array = np.frombuffer(in_data, dtype=np.int16)
            
            # Aplica a Transformada de Fourier para extrair as frequências
            fft_data = np.abs(np.fft.rfft(audio_array))
            
            # Fatiamos os resultados (Os números dos índices representam as faixas de Hz)
            bass = np.mean(fft_data[1:16])       # Graves (Aprox. 15Hz a 250Hz)
            mids = np.mean(fft_data[16:128])     # Médios (Aprox. 250Hz a 2000Hz)
            treble = np.mean(fft_data[128:512])  # Agudos (Aprox. 2000Hz a 8000Hz)
            
            # Atualiza a variável global
            freq_data[0] = bass
            freq_data[1] = mids
            freq_data[2] = treble
            
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
        
        print("\n🎵 Capturando áudio e controlando LEDs do G1...")
        print("Pressione [Ctrl+C] para encerrar.\n")
        
        last_led_time = time.time()
        
        while stream.is_active():
            time.sleep(0.05) # Dorme rapidinho pra não fritar a CPU
            now = time.time()
            
            # Só atualiza o LED a cada 250ms (Respeitando a regra do SDK > 200ms)
            if (now - last_led_time) >= 0.25:
                bass, mids, treble = freq_data
                
                # NOVO RADAR: Veja qual frequência está dominando!
                print(f"Graves: {int(bass)} | Médios: {int(mids)} | Agudos: {int(treble)}      ", end='\r')
                
                try:
                    total_eq = bass + mids + treble
                    
                    # 1. Limite de Silêncio
                    # Como os números da FFT são muito altos, usamos 50000 como base.
                    # Se o radar mostrar que no silêncio fica maior que isso, aumente esse número!
                    if total_eq < 50000:
                        audio_client.LedControl(0, 0, 20) # Fica um azul quase apagado
                    else:
                        # 2. Descobre qual é a frequência mais alta no momento
                        if bass > mids and bass > treble:
                            # Batida forte / Grave -> Vermelho
                            audio_client.LedControl(255, 0, 0)
                        elif mids > bass and mids > treble:
                            # Vozes e Melodia -> Verde
                            audio_client.LedControl(0, 255, 0)
                        else:
                            # Pratos de Bateria e Eletrônico Agudo -> Ciano
                            audio_client.LedControl(0, 255, 255)
                except:
                    pass 
                
                last_led_time = now

    except KeyboardInterrupt:
        print("\n[INFO] Encerrando pelo usuário.")
    except Exception as e:
        print(f"\n[ERRO] {e}")
    finally:
        print("\nLimpando estado e desligando o som e os LEDs...")
        audio_client.PlayStop("pc_audio")
        
        # Apaga os LEDs na hora de fechar (R=0, G=0, B=0)
        try:
            audio_client.LedControl(0, 0, 0) 
        except:
            pass
            
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        print("Finalizado.")

if __name__ == '__main__':
    main()