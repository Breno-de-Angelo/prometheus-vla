#!/usr/bin/env python
"""
Data Collection Entry Point - HACKER EDITION V9 + VOICE CONTROL
Corrigindo validação de Tuplas e adicionando controle Hands-Free.
"""

import os

# Ver a nota longa no init_lerobot_teleoparate_v2.py: o ipopt/BLAS multithread do
# conda-forge faz a IK custar 83 ms em vez de 0,8 ms, e a variável tem que ser
# definida antes do primeiro import que carregue a runtime OpenMP (numpy já basta).
os.environ.setdefault("OMP_NUM_THREADS", "1")

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
# Na 0.6.1 os dois helpers saíram de `lerobot.datasets.utils` e foram para
# `lerobot.utils.feature_utils` — módulo propositalmente leve, para poder ser
# importado sem arrastar o `datasets` do HuggingFace junto.
#
# Trocar o atributo só no módulo de origem NÃO basta: quem consome faz
# `from lerobot.utils.feature_utils import build_dataset_frame`, ou seja, copia
# a referência para o próprio namespace no momento do import. Quem já foi
# importado (o `robot.unitree_g1` lá em cima arrasta meio lerobot) continuaria
# com a função original. `_patch_lerobot` reescreve a referência em todo módulo
# lerobot já carregado; trocar no módulo de origem cobre os que ainda vão ser
# importados — entre eles o `lerobot.scripts.lerobot_record` lá embaixo.
import lerobot.utils.feature_utils as lr_feature_utils
from lerobot.utils.constants import OBS_STR


def _patch_lerobot(nome: str, func_nova, func_velha) -> None:
    setattr(lr_feature_utils, nome, func_nova)
    for mod_nome, mod in list(sys.modules.items()):
        if mod is None or mod is lr_feature_utils or not mod_nome.startswith("lerobot"):
            continue
        if getattr(mod, nome, None) is func_velha:
            setattr(mod, nome, func_nova)


original_hw_to_dataset_features = lr_feature_utils.hw_to_dataset_features


# *args/**kwargs de propósito: o terceiro parâmetro se chama `use_video` na 0.6.1
# (era `use_videos`) e há chamador que passa por nome. Repassar cru evita casar
# assinatura com uma API que ainda está se mexendo.
def patched_hw_to_dataset_features(*args, **kwargs):
    dataset_features = original_hw_to_dataset_features(*args, **kwargs)

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

    # A marcação `video.is_depth_map` da head_camera_depth saiu daqui de propósito.
    # Na 0.6.1 essa flag deixou de ser enfeite de metadado: `meta.depth_keys` a lê e
    # manda a câmera inteira para OUTRO pipeline — quadros gravados como TIFF 16 bits
    # e vídeo encodado pelo `DepthEncoderConfig`, que quantiza profundidade métrica de
    # 1 canal. O nosso `realsense_server.py` publica profundidade já normalizada para
    # cinza uint8 de 3 canais (`cv2.merge([d, d, d])`), que é o que o `_cameras_ft`
    # declara. Ligar a flag mandaria dado de 3 canais para o encoder de 1 canal.
    # Quando a profundidade métrica de verdade for publicada (uint16, 1 canal), o
    # caminho certo é declarar (H, W, 1) no `_cameras_ft`: o próprio
    # `hw_to_dataset_features` marca `info["is_depth_map"] = True` sozinho.

    return dataset_features


_patch_lerobot("hw_to_dataset_features", patched_hw_to_dataset_features, original_hw_to_dataset_features)

# =========================================================================
# 💉 INJEÇÃO 3: Contrabando de volta pro Empacotador
# =========================================================================
original_build_dataset_frame = lr_feature_utils.build_dataset_frame


def patched_build_dataset_frame(ds_features, values, prefix, *args, **kwargs):
    # O `prefix` perdeu o default na 0.6.1 e virou "observation" (sem ponto), o
    # mesmo `OBS_STR` que o record passa. A mesma função também empacota a ação —
    # aí não há pressão nenhuma para contrabandear.
    if prefix == OBS_STR:
        lp = buffer_pressao["left"]
        rp = buffer_pressao["right"]

        for i in range(33):
            values[f"left_hand_pressure_{i}"] = float(lp[i])
            values[f"right_hand_pressure_{i}"] = float(rp[i])

    return original_build_dataset_frame(ds_features, values, prefix, *args, **kwargs)


_patch_lerobot("build_dataset_frame", patched_build_dataset_frame, original_build_dataset_frame)

# =========================================================================
# 🎤 INJEÇÃO 4: Comandos de Voz e Teclado (Setas, Pulo Duplo e PAUSE!)
# =========================================================================
# Ver a nota do init_lerobot_teleoparate_v2.py: na 0.6.1 o `init_keyboard_listener`
# migrou de `lerobot.utils.control_utils` para `lerobot.utils.keyboard_input`. Aqui o
# patch tem efeito de verdade — o `lerobot_record` chama a função —, mas só porque o
# `from lerobot.scripts.lerobot_record import main` lá embaixo vem DEPOIS desta linha.
import lerobot.utils.keyboard_input
import threading
import time

original_init_keyboard = lerobot.utils.keyboard_input.init_keyboard_listener

global_events = None

def patched_init_keyboard():
    global global_events
    listener, events = original_init_keyboard()
    global_events = events  
    return listener, events

lerobot.utils.keyboard_input.init_keyboard_listener = patched_init_keyboard

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