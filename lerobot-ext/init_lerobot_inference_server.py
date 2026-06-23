#!/usr/bin/env python
"""
Inference Server — Roda na Atena (A100)
========================================
Carrega o modelo e fica escutando observações via ZMQ.
Retorna chunks de ação para o cliente (seu PC local).

Protocolo ZMQ: REQ/REP
  Cliente envia:  msgpack({ obs_dict, obs_step, actions_per_chunk, want_attn })
  Servidor retorna: msgpack({ chunk_np: [[float,...]], obs_step: int, attn?: {...} })
  Em caso de erro:  msgpack({ error: str })

  O campo "want_attn" é opcional — clientes antigos que não o enviam
  continuam funcionando normalmente (default False, sem custo extra).
  Quando True, e se o modelo expuser `policy.last_attn_weights`, o
  servidor inclui na resposta:
    attn: {
      "attn_np":   lista aninhada (pesos de atenção do decoder)
      "rgb_frame": lista aninhada uint8 (H, W, 3) — o frame que gerou a inferência
    }

Otimizações nesta versão:
  - Postprocessor vetorizado: chunk inteiro processado em uma chamada (zero loop Python)
  - Print de debug movido para FORA do caminho crítico (não bloqueia o REP socket)
  - Captura de attention é condicional — só roda o trabalho extra (.cpu()/.numpy())
    quando o cliente pediu want_attn, então clientes sem --v-attn não pagam o custo

Uso:
  python inference_server.py --checkpoint=<PATH> [OPÇÕES]

Opções:
  --checkpoint=<PATH>   (obrigatório) Caminho para o pretrained_model
  --port=<INT>          Porta ZMQ (padrão: 5600)
  --host=<IP>           IP para escutar (padrão: 0.0.0.0 — todas as interfaces)
  --debug               Loga cada inferência com tempo e chunk size
  -h, --help            Mostra esta mensagem

Exemplo na Atena:
  python inference_server.py \\
      --checkpoint=train_output/actdepth/best_val_checkpoint/pretrained_model \\
      --port=5600 --debug
"""

import os
import sys
import time
import traceback
import math
import cv2

import numpy as np
import torch

# ── ZMQ + msgpack ──────────────────────────────────────────────────────
try:
    import zmq
    import msgpack
    import msgpack_numpy as m
    m.patch()   # faz msgpack serializar numpy arrays nativamente
except ImportError:
    print("❌ Instale as dependências: pip install pyzmq msgpack msgpack-numpy")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────
# Reutiliza as funções de carregamento do script original
# O servidor precisa estar na mesma pasta que o script de inferência
# ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from init_lerobot_inference_async_v2 import (
        load_policy,
        load_pre_post_processors,
        make_raw_obs,
    )
except ImportError as e:
    print(f"❌ Não consegui importar init_lerobot_inference_async_v2: {e}")
    print("   Coloque inference_server.py na mesma pasta do script de inferência.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────
# Helpers de serialização
# ─────────────────────────────────────────────────────────────────────

def _pack(data: dict) -> bytes:
    return msgpack.packb(data, default=m.encode)

def _unpack(raw: bytes) -> dict:
    return msgpack.unpackb(raw, object_hook=m.decode, raw=False)


# ─────────────────────────────────────────────────────────────────────
# Loop principal do servidor
# ─────────────────────────────────────────────────────────────────────

def run_server(checkpoint_dir: str, host: str, port: int, debug: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ── Carrega modelo ─────────────────────────────────────────────────
    print(f"⏳ Carregando modelo: {checkpoint_dir}")
    policy, policy_type = load_policy(checkpoint_dir, device)
    policy.eval()

    has_depth    = getattr(policy.config, "use_depth_3d", False)
    has_pressure = getattr(policy.config, "use_pressure", False)
    task_str     = "pick up the cup" if policy_type == "pi05depth" else None

    print(f"✅ Modelo carregado — tipo={policy_type} | depth={has_depth} | pressure={has_pressure}")

    preprocessor, postprocessor = load_pre_post_processors(checkpoint_dir, policy)
    print("✅ Pre/Postprocessors carregados")

    # Joint names padrão G1 upper body (28 juntas)
    joint_names = [
        "kLeftShoulderPitch.q",  "kLeftShoulderRoll.q",  "kLeftShoulderYaw.q",
        "kLeftElbow.q",          "kLeftWristRoll.q",      "kLeftWristPitch.q",
        "kLeftWristyaw.q",
        "kRightShoulderPitch.q", "kRightShoulderRoll.q", "kRightShoulderYaw.q",
        "kRightElbow.q",         "kRightWristRoll.q",     "kRightWristPitch.q",
        "kRightWristYaw.q",
        "left_hand_thumb_0_joint.q",  "left_hand_thumb_1_joint.q",
        "left_hand_thumb_2_joint.q",  "left_hand_middle_0_joint.q",
        "left_hand_middle_1_joint.q", "left_hand_index_0_joint.q",
        "left_hand_index_1_joint.q",
        "right_hand_thumb_0_joint.q", "right_hand_thumb_1_joint.q",
        "right_hand_thumb_2_joint.q", "right_hand_index_0_joint.q",
        "right_hand_index_1_joint.q", "right_hand_middle_0_joint.q",
        "right_hand_middle_1_joint.q",
    ]

    # ── ZMQ REP socket ────────────────────────────────────────────────
    ctx    = zmq.Context()
    socket = ctx.socket(zmq.REP)
    addr   = f"tcp://{host}:{port}"
    socket.bind(addr)
    print(f"🔌 Servidor ZMQ escutando em {addr}")
    print("   Aguardando conexão do cliente...\n")

    req_count  = 0
    t_total_ms = 0.0
    _attn_unavailable_warned = False

    try:
        while True:
            # ── Recebe observação ──────────────────────────────────────
            raw = socket.recv()
            msg = _unpack(raw)

            obs               = msg["obs"]           # dict com arrays numpy
            obs_step          = int(msg["obs_step"])
            actions_per_chunk = int(msg.get("actions_per_chunk", 60))
            want_attn         = bool(msg.get("want_attn", False))

            t0 = time.perf_counter()

            try:
                # ── Preprocessor ──────────────────────────────────────
                raw_obs = make_raw_obs(
                    obs, joint_names,
                    has_depth=has_depth,
                    has_pressure=has_pressure,
                    task=task_str,
                )
                batch = preprocessor(raw_obs)
                batch.pop("action", None)
                batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

                if has_pressure:
                    for side in ["left", "right"]:
                        k = f"observation.{side}_hand_pressure"
                        if k in batch and batch[k].dim() == 1:
                            batch[k] = batch[k].unsqueeze(0)

                # ── Inferência ────────────────────────────────────────
                with torch.inference_mode():
                    if hasattr(policy, "predict_action_chunk"):
                        # capture_attn precisa ser pedido explicitamente — nem
                        # toda policy aceita o kwarg. Detecta via inspect para
                        # não quebrar policies que não o conhecem (ex: ACT).
                        import inspect
                        sig = inspect.signature(policy.predict_action_chunk)
                        if want_attn and "capture_attn" in sig.parameters:
                            raw_chunk = policy.predict_action_chunk(batch, capture_attn=True)
                        else:
                            raw_chunk = policy.predict_action_chunk(batch)
                        raw_chunk = raw_chunk[0, :actions_per_chunk, :]

                        # ── Postprocessor VETORIZADO ──────────────────
                        # Processa o chunk inteiro de uma vez em vez de
                        # 60 chamadas em loop — elimina 60 trocas de GIL
                        # e reduz latência de resposta ao cliente.
                        chunk_batch = raw_chunk.unsqueeze(1)  # (T, 1, D)
                        processed = postprocessor(chunk_batch)
                        if isinstance(processed, dict):
                            processed = processed["action"]
                        if processed.dim() == 3:
                            processed = processed.squeeze(1)   # (T, D)
                        chunk_np = list(processed.cpu().numpy())
                    else:
                        action = policy.select_action(batch)
                        action = postprocessor(action)
                        if isinstance(action, dict):
                            action = action["action"]
                        chunk_np = [action.squeeze(0).cpu().numpy()]

                t_inf_ms = (time.perf_counter() - t0) * 1000
                req_count  += 1
                t_total_ms += t_inf_ms

                # ── Monta resposta ─────────────────────────────────────
                # chunk_np como lista de np.ndarray (não .tolist()) — deixa
                # o msgpack_numpy serializar em binário nativo, mais rápido.
                resp = {
                    "chunk_np": list(chunk_np),
                    "obs_step": obs_step,
                }

                # ── Attention map, só se pedido (custo extra opt-in) ───
                # Mesmo padrão usado no script local (init_lerobot_inference_async_v2.py):
                # o modelo expõe `policy.last_attn_weights` após o forward de
                # predict_action_chunk(). O rgb_frame vem da obs ORIGINAL (não
                # do tensor normalizado), igual ao que a AttnMapWindow espera.
                if want_attn:
                    attn_raw = getattr(policy, "last_attn_weights", None)
                    rgb_frame = obs.get("head_camera")

                    if attn_raw is not None and rgb_frame is not None:
                        if isinstance(attn_raw, torch.Tensor):
                            attn_np = attn_raw.detach().cpu().float().numpy()
                        else:
                            attn_np = np.array(attn_raw, dtype=np.float32)

                        # Colapsa heads AQUI (servidor) em vez de mandar todas
                        # as 8 heads pela rede — reduz o payload em 8x. O
                        # cliente só usa max sobre heads mesmo (AttnMapWindow),
                        # então não perdemos nenhuma informação usada.
                        # [B, num_heads, Q, K] → [B, Q, K]
                        if attn_np.ndim == 4:
                            attn_np = attn_np.max(axis=1)

                        rgb_np = np.array(rgb_frame, dtype=np.uint8)
                        attn_meta = getattr(policy, "_last_attn_meta", None)

                        # ── NOVO: Desfazendo o Padding e Recortando a Grade 2D ──
                        if attn_meta is not None and attn_meta.get("n_img_tokens"):
                            suffix_len = attn_meta["suffix_len"]
                            n_cam1 = attn_meta["n_img_tokens"][0]
                            
                            # 1. Isola as queries de ação e as keys da imagem RGB
                            attn_cam1 = attn_np[:, -suffix_len:, :n_cam1]
                            if attn_cam1.ndim == 3:
                                attn_cam1 = attn_cam1[0]  # Remove batch dim: [Actions, Tokens]
                            
                            # 2. Descobre a grade geométrica (ex: 256 tokens -> 16x16)
                            grid_size = int(math.sqrt(n_cam1))
                            h_orig, w_orig = rgb_np.shape[:2]
                            
                            # 3. Matemática reversa do PI05 para achar as tarjas pretas
                            patch_size = 14
                            img_res = grid_size * patch_size
                            ratio = max(w_orig / img_res, h_orig / img_res)
                            
                            resized_h = h_orig / ratio
                            resized_w = w_orig / ratio
                            
                            frac_h = resized_h / img_res
                            frac_w = resized_w / img_res
                            
                            # Acha exatamente onde a imagem real começa e termina dentro do quadrado
                            start_h = int(round(grid_size * (1.0 - frac_h) / 2.0))
                            end_h   = int(round(grid_size * (1.0 + frac_h) / 2.0))
                            start_w = int(round(grid_size * (1.0 - frac_w) / 2.0))
                            end_w   = int(round(grid_size * (1.0 + frac_w) / 2.0))
                            
                            start_h, end_h = max(0, start_h), min(grid_size, end_h)
                            start_w, end_w = max(0, start_w), min(grid_size, end_w)
                            
                            # 4. Recorta a área válida e "estica" de volta
                            actions_count = attn_cam1.shape[0]
                            fixed_attn = np.zeros((actions_count, grid_size * grid_size), dtype=np.float32)
                            
                            for i in range(actions_count):
                                grid = attn_cam1[i].reshape(grid_size, grid_size)
                                crop = grid[start_h:end_h, start_w:end_w]
                                if crop.size > 0:
                                    # O pulo do gato: redimensionar o recorte limpo de volta pro tamanho original
                                    # Isso cancela perfeitamente a distorção quando o cliente for desenhar na tela
                                    crop_resized = cv2.resize(crop, (grid_size, grid_size), interpolation=cv2.INTER_LINEAR)
                                    fixed_attn[i] = crop_resized.flatten()
                            
                            attn_np = fixed_attn
                            
                            # Avisa a interface que É o PI05, mas sem depth!
                            attn_meta["n_img_tokens"] = [grid_size * grid_size]
                            attn_meta["has_depth_tok"] = False
                        else:
                            if attn_np.ndim == 3:
                                attn_np = attn_np[0]

                        # ── CRÍTICO: NÃO usar .tolist() aqui ──────────────────
                        resp["attn"] = {
                            "attn_np":   attn_np,   
                            "rgb_frame": rgb_np,    
                        }

                        if attn_meta is not None:
                            resp["attn"]["meta"] = attn_meta
                    elif not _attn_unavailable_warned:
                        # Avisa UMA vez no terminal do servidor — não quebra
                        # nada, o cliente já trata a ausência do campo "attn"
                        # graciosamente e mostra um aviso próprio.
                        motivo = (
                            "policy.last_attn_weights ausente "
                            "(modelo não expõe pesos de atenção)"
                            if attn_raw is None else
                            "obs['head_camera'] ausente na observação recebida"
                        )
                        print(f"⚠️  Cliente pediu want_attn=True mas {motivo}. "
                              f"Resposta vai sem o campo 'attn'.")
                        _attn_unavailable_warned = True

                # ── Envia resposta ANTES do print ─────────────────────
                # O socket REP deve responder o mais rápido possível.
                # O print (que pode chamar flush/write de terminal e ceder
                # o GIL por alguns ms) vai para DEPOIS do send().
                socket.send(_pack(resp))

                if debug:
                    avg = t_total_ms / req_count
                    attn_tag = "✓" if want_attn and "attn" in resp else ("✗" if want_attn else "-")
                    print(f"🧠 [req #{req_count}] {t_inf_ms:.1f}ms | chunk={len(chunk_np)} | "
                          f"attn={attn_tag} | avg={avg:.1f}ms")

            except Exception as e:
                tb = traceback.format_exc()
                print(f"❌ [servidor] Erro na inferência: {e}\n{tb}")
                socket.send(_pack({"error": str(e)}))

    except KeyboardInterrupt:
        print("\n🛑 Servidor encerrado.")
    finally:
        socket.close()
        ctx.term()
        if req_count > 0:
            print(f"📊 Total: {req_count} inferências | Média: {t_total_ms/req_count:.1f}ms/req")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    if any(f in sys.argv for f in ["-h", "--help"]):
        print(__doc__)
        sys.exit(0)

    checkpoint_dir = None
    host  = "0.0.0.0"
    port  = 5600
    debug = False

    for arg in sys.argv[1:]:
        if arg.startswith("--checkpoint="):
            checkpoint_dir = arg.split("=", 1)[1]
        elif arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])
        elif arg.startswith("--host="):
            host = arg.split("=", 1)[1]
        elif arg == "--debug":
            debug = True

    if checkpoint_dir is None:
        print("❌ ERRO: --checkpoint é obrigatório.")
        sys.exit(1)

    run_server(checkpoint_dir, host, port, debug)


if __name__ == "__main__":
    main()