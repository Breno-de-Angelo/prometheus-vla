#!/usr/bin/env python
"""
Data Collection Entry Point - HACKER EDITION V9 + VOICE CONTROL
Corrigindo validação de Tuplas e adicionando controle Hands-Free.
"""

import sys
import logging
import numpy as np
import threading
import time

frame_count = 0

# Buffer global de contrabando
buffer_pressao = {"left": np.zeros(33, dtype=np.float32), "right": np.zeros(33, dtype=np.float32)}

try:
    import robot.unitree_g1
    import teleop.unitree_g1
except ImportError as e:
    print(f"\n[IMPORT ERROR]: {e}")
    sys.exit(1)

# =========================================================================
# 💉 INJEÇÃO 1: Rouba a pressão do Robô
# =========================================================================
from robot.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3

original_get_obs = UnitreeG1Dex3.get_observation

def patched_get_observation(self):
    global frame_count
    obs = original_get_obs(self)
    
    if obs is not None:
        frame_count += 1
        
        if "left_hand_pressure" in obs:
            lp = obs.pop("left_hand_pressure")
            rp = obs.pop("right_hand_pressure")
            buffer_pressao["left"] = lp
            buffer_pressao["right"] = rp
            
            if frame_count % 50 == 0:
                max_l = np.max(lp)
                max_r = np.max(rp)
                status = "🟢 SENSOR ATIVO" if (max_l > 0 or max_r > 0) else "⚪ ZERADO (Aguardando Toque)"
                #print(f"[DEBUG] Frame {frame_count} | {status} | Max L: {max_l:.0f} | Max R: {max_r:.0f}")
        #else:
            #if frame_count % 50 == 0:
                #print(f"[ERRO] Frame {frame_count} | 🔴 DRIVER NÃO ENVIOU DADOS DE PRESSÃO!")
                
    return obs

UnitreeG1Dex3.get_observation = patched_get_observation

# =========================================================================
# 💉 INJEÇÃO 2: Editar a Planta Baixa do Parquet (TUPLAS!)
# =========================================================================
import lerobot.datasets.utils
original_hw_to_dataset_features = lerobot.datasets.utils.hw_to_dataset_features

def patched_hw_to_dataset_features(features, feature_type, use_videos):
    dataset_features = original_hw_to_dataset_features(features, feature_type, use_videos)
    
    if "observation.state" in dataset_features:
        print("\n[HACK LEROBOT] 🗜️ Configurando colunas do Parquet para Pressão...")
        
        old_names = dataset_features["observation.state"].get("names", [])
        new_names = [n for n in old_names if "pressure" not in n]
        dataset_features["observation.state"]["names"] = new_names
        dataset_features["observation.state"]["shape"] = (len(new_names),)
        
        dataset_features["observation.left_hand_pressure"] = {
            "dtype": "float32", "shape": (33,), "names": [f"left_hand_pressure_{i}" for i in range(33)]
        }
        dataset_features["observation.right_hand_pressure"] = {
            "dtype": "float32", "shape": (33,), "names": [f"right_hand_pressure_{i}" for i in range(33)]
        }
        
        chave_depth = "observation.images.head_camera_depth"
        if chave_depth in dataset_features:
            print("[HACK LEROBOT] 🚀 Configurando Câmera de Profundidade...")
            if "info" not in dataset_features[chave_depth] or dataset_features[chave_depth]["info"] is None:
                dataset_features[chave_depth]["info"] = {
                    "video.fps": 30, "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.channels": 3, "has_audio": False
                }
            dataset_features[chave_depth]["info"]["video.is_depth_map"] = True
            
    return dataset_features

lerobot.datasets.utils.hw_to_dataset_features = patched_hw_to_dataset_features

# =========================================================================
# 💉 INJEÇÃO 3: Contrabando de volta pro Empacotador
# =========================================================================
original_build_dataset_frame = lerobot.datasets.utils.build_dataset_frame

def patched_build_dataset_frame(features, obs_dict, prefix="observation."):
    lp = buffer_pressao["left"]
    rp = buffer_pressao["right"]
    
    for i in range(33):
        obs_dict[f"left_hand_pressure_{i}"] = float(lp[i])
        obs_dict[f"right_hand_pressure_{i}"] = float(rp[i])
        
    return original_build_dataset_frame(features, obs_dict, prefix)

lerobot.datasets.utils.build_dataset_frame = patched_build_dataset_frame

# =========================================================================
# 🎤 INJEÇÃO 4: Comandos de Voz e Teclado (Setas, Pulo Duplo e PAUSE!)
# =========================================================================
import lerobot.utils.control_utils
import threading
import time

original_init_keyboard = lerobot.utils.control_utils.init_keyboard_listener

global_events = None

def patched_init_keyboard():
    global global_events
    listener, events = original_init_keyboard()
    global_events = events  
    return listener, events

lerobot.utils.control_utils.init_keyboard_listener = patched_init_keyboard

# --- ⌨️ NOVO: Listener de Teclado Paralelo (Setas e Espaço) ---
try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    print("⚠️ Lib 'pynput' não instalada. O atalho de teclado não funcionará. (pip install pynput)")
    pynput_keyboard = None

if pynput_keyboard:
    def on_press(key):
        global robot_paused # Usa a mesma variável da injeção 1
        try:
            # Seta para Baixo (PAUSAR / CONGELAR)
            if key == pynput_keyboard.Key.down:
                if not robot_paused:
                    robot_paused = True
                    print("\n   ⬇️ [TECLADO] Ação: CONGELANDO O ROBÔ NA POSIÇÃO ATUAL! 🧊")

            # Seta para Cima (CONTINUAR / DESTRAVAR)
            elif key == pynput_keyboard.Key.up:
                if robot_paused:
                    robot_paused = False
                    print("\n   ⬆️ [TECLADO] Ação: DESTRAVANDO O ROBÔ! ▶️ (Cuidado com trancos)")

            # Toggle alternativo (Barra de Espaço ou 'P')
            elif key == pynput_keyboard.Key.space or (hasattr(key, 'char') and key.char.lower() == 'p'):
                robot_paused = not robot_paused
                if robot_paused:
                    print("\n   ⌨️ [TECLADO] Ação: CONGELANDO O ROBÔ NA POSIÇÃO ATUAL! 🧊")
                else:
                    print("\n   ⌨️ [TECLADO] Ação: DESTRAVANDO O ROBÔ! ▶️ (Cuidado com trancos)")
                    
        except AttributeError:
            pass

    kb_listener = pynput_keyboard.Listener(on_press=on_press)
    kb_listener.daemon = True
    kb_listener.start()

# --- 🎙️ Função de Voz Original Atualizada ---
def voice_commander_loop():
    global robot_paused # Puxa a variável global de congelamento
    
    try:
        import speech_recognition as sr
    except ImportError:
        print("⚠️ Libs de voz não instaladas. Controle desativado.")
        return

    print("⏳ [VOZ] Aguardando os motores e câmeras iniciarem...")
    
    # Substitua "frame_count" pela variável real de inicialização do seu código, se necessário.
    time.sleep(3) 

    recognizer = sr.Recognizer()
    print("\n🎙️ [VOZ & TECLADO] SISTEMA ATIVO! Comandos:")
    print("   ✅ SALVAR: Voz: 'salvar', 'gravar'")
    print("   ❌ DESCARTAR: Voz: 'errei', 'reboot'")
    print("   🧊 CONGELAR: Voz: 'pausar'    | Teclado: Seta para Baixo (↓)")
    print("   ▶️ DESTRAVAR: Voz: 'continuar' | Teclado: Seta para Cima (↑)")
    print("   🛑 ENCERRAR: Voz: 'sair', 'finalizar'\n")

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        while True:
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
                texto = recognizer.recognize_google(audio, language="pt-BR").lower()

                if global_events is None:
                    continue

                # --- 1. SUCESSO: SALVAR ---
                if any(cmd in texto for cmd in ["gravar", "salvar", "próximo"]):
                    print(f"\n   🗣️ Detectado: '{texto}'")
                    print("   ✅ Ação: Salvando e preparando o próximo...")
                    global_events["exit_early"] = True
                    def auto_skip_to_next():
                        time.sleep(1.0)
                        if global_events: global_events["exit_early"] = True
                    threading.Thread(target=auto_skip_to_next, daemon=True).start()
                
                # --- 2. ERRO: DESCARTAR ---
                elif any(cmd in texto for cmd in ["errei", "reboot", "voltar"]):
                    print(f"\n   🗣️ Detectado: '{texto}'")
                    print("   ❌ Ação: Descartando lixo e recomeçando...")
                    global_events["rerecord_episode"] = True
                    def auto_restart_same():
                        time.sleep(0.5)
                        if global_events: global_events["exit_early"] = True
                        time.sleep(0.5)
                        if global_events: global_events["exit_early"] = True
                    threading.Thread(target=auto_restart_same, daemon=True).start()

                # --- 3. 🧊 CONGELAR O ROBÔ (PAUSE) ---
                elif any(cmd in texto for cmd in ["pausar", "congelar", "travar"]):
                    if not robot_paused:
                        print(f"\n   🗣️ Detectado: '{texto}'")
                        print("   🧊 Ação: CONGELANDO O ROBÔ NA POSIÇÃO ATUAL!")
                        robot_paused = True

                # --- 4. ▶️ DESTRAVAR O ROBÔ (PLAY) ---
                elif any(cmd in texto for cmd in ["continuar", "destravar", "play"]):
                    if robot_paused:
                        print(f"\n   🗣️ Detectado: '{texto}'")
                        print("   ▶️ Ação: DESTRAVANDO O ROBÔ! (Cuidado com trancos)")
                        robot_paused = False
                
                # --- 5. FINALIZAR TUDO ---
                elif any(cmd in texto for cmd in ["finalizar", "sair", "fechar"]):
                    print(f"\n   🗣️ Detectado: '{texto}'")
                    print("   🛑 Ação: Encerrando gravação geral...")
                    global_events["stop_recording"] = True
                    global_events["exit_early"] = True

            except sr.WaitTimeoutError:
                pass 
            except sr.UnknownValueError:
                pass 
            except Exception:
                time.sleep(1)

voice_thread = threading.Thread(target=voice_commander_loop, daemon=True, name="VoiceCommander")
voice_thread.start()
# =========================================================================

# =========================================================================
# 🧊 INJEÇÃO 5: Hack de Congelamento Motor (Pause/Play)
# =========================================================================
robot_paused = False
frozen_action = None

original_send_action = UnitreeG1Dex3.send_action

def patched_send_action(self, action):
    global robot_paused, frozen_action
    
    if robot_paused:
        # 🧊 MODO ESTÁTUA: Ignora o VR e manda o robô segurar a última pose com força
        if frozen_action is not None:
            return original_send_action(self, frozen_action)
        else:
            return original_send_action(self, action)
    else:
        # ▶️ MODO NORMAL: Salva a posição atual e obedece o VR
        frozen_action = {k: v for k, v in action.items()}
        return original_send_action(self, action)

UnitreeG1Dex3.send_action = patched_send_action

# INICIALIZAÇÃO OFICIAL
from lerobot.scripts.lerobot_record import main

class IgnoreFPSWarningFilter(logging.Filter):
    def filter(self, record):
        return "Record loop is running slower" not in record.getMessage()

if __name__ == "__main__":
    cli_args = sys.argv[:]
    
    if "--config_path" not in str(cli_args):
        print("\n[ERRO]: O argumento '--config_path' é obrigatório.")
        sys.exit(1)

    force_sim = "--sim" in cli_args or "--simulation=true" in cli_args
    if "--sim" in sys.argv: sys.argv.remove("--sim")

    if force_sim:
        sys.argv.append("--robot.is_simulation=true")
        sys.argv.append("--teleop.is_simulation=true")
    else:
        sys.argv.append("--robot.is_simulation=false")
        sys.argv.append("--teleop.is_simulation=false")

    logging.getLogger().addFilter(IgnoreFPSWarningFilter())
    logging.getLogger("lerobot").addFilter(IgnoreFPSWarningFilter())

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[SYSTEM]: Gravação finalizada pelo usuário.")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\n[ERRO DE EXECUÇÃO]:")
        traceback.print_exc()
        sys.exit(1)