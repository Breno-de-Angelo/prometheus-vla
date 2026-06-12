#!/usr/bin/env python3
"""
LeRobot Dataset Visualizer & Editor — Backend Server
Versão robusta: não depende de episode_data_index.
Descobre os ranges de frames filtrando por episode_index no próprio dataset.
"""

import json
import sys
import argparse
import traceback
import shutil
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

try:
    import numpy as np
    import torch
    import pandas as pd
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import delete_episodes
except ImportError as e:
    print(f"[ERRO] Dependência faltando: {e}")
    sys.exit(1)

dataset         = None
dataset_root    = None
dataset_repo_id = None

# Cache: ep_idx -> (from_frame, to_frame)  construído uma vez no load
_ep_ranges: dict[int, tuple[int, int]] = {}


def load_dataset(repo_id, root=None):
    global dataset, dataset_root, dataset_repo_id, _ep_ranges
    dataset_repo_id = repo_id

    # ── Validação do caminho físico ───────────────────────────────────────
    if root is not None:
        root_path = Path(root).resolve()
        
        # Cenário A: O 'root' já é a pasta exata do dataset (tem meta/info.json direto)
        if (root_path / "meta" / "info.json").exists():
            dataset_root = str(root_path)
            
        # Cenário B: O 'root' é a pasta pai (ex: "meu_dataset"), procuramos o repo_id dentro
        elif (root_path / repo_id / "meta" / "info.json").exists():
            dataset_root = str(root_path / repo_id)
            
        else:
            print(f"\n[ERRO] Nenhum meta/info.json encontrado.")
            print(f"       Procurei em: {root_path / 'meta' / 'info.json'}")
            print(f"       E também em: {root_path / repo_id / 'meta' / 'info.json'}")
            sys.exit(1)
    else:
        dataset_root = None

    print(f"[INFO] Carregando dataset: {repo_id}  (root={dataset_root})")
    
    # Passamos dataset_root (que agora é GARANTIDAMENTE a pasta final correta)
    dataset = LeRobotDataset(repo_id, root=dataset_root)
    
    n_ep = dataset.meta.total_episodes
    n_fr = dataset.meta.total_frames

    # ── Confirma que leu o arquivo certo ──────────────────────────────────
    try:
        real_path = Path(dataset.root).resolve()
        print(f"[INFO] Caminho real lido: {real_path}")
        info_file = real_path / "meta" / "info.json"
        if info_file.exists():
            with open(info_file) as f:
                info = json.load(f)
            print(f"[INFO] info.json confirma: {info.get('total_episodes')} episódios, fps={info.get('fps')}")
            if info.get("total_episodes") != n_ep:
                print(f"[AVISO] Discrepância! meta diz {n_ep} mas info.json diz {info.get('total_episodes')}")
    except Exception as e:
        print(f"[DEBUG] Não consegui ler info.json para validar: {e}")

    print(f"[INFO] OK — {n_ep} episódios, {n_fr} frames")

    # ── Descobre episode_data_index de onde ele estiver ──────────────────
    _ep_ranges = _build_ep_ranges()
    print(f"[INFO] Ranges calculados para {len(_ep_ranges)} episódios")
    print(f"[DEBUG] Primeiro range: ep0 = {_ep_ranges.get(0)}")


def _build_ep_ranges() -> dict[int, tuple[int, int]]:
    """
    Tenta várias estratégias para obter (from, to) por episódio.
    Retorna dict {ep_idx: (from_abs, to_abs)}.
    """
    # Estratégia 1: atributo episode_data_index no dataset ou meta
    for obj in (dataset, dataset.meta):
        for attr in ("episode_data_index", "episodes_data_index", "_episode_data_index"):
            edi = getattr(obj, attr, None)
            if edi is not None and isinstance(edi, dict) and "from" in edi and "to" in edi:
                print(f"[DEBUG] episode_data_index encontrado em {obj.__class__.__name__}.{attr}")
                result = {}
                for i in range(dataset.meta.total_episodes):
                    f = edi["from"][i]
                    t = edi["to"][i]
                    result[i] = (int(f.item() if hasattr(f,'item') else f),
                                 int(t.item() if hasattr(t,'item') else t))
                return result

    # Estratégia 2: lê episode_index de cada frame do HF dataset subjacente
    print("[DEBUG] episode_data_index não encontrado — varrendo hf_dataset por episode_index...")
    try:
        hf = dataset.hf_dataset  # LeRobot >= 2.x expõe isso
        ep_col = hf["episode_index"]
        result = {}
        for i, ep in enumerate(ep_col):
            ep_int = int(ep)
            if ep_int not in result:
                result[ep_int] = (i, i + 1)
            else:
                result[ep_int] = (result[ep_int][0], i + 1)
        return result
    except Exception as e:
        print(f"[DEBUG] hf_dataset scan falhou: {e}")

    # Estratégia 3: itera dataset diretamente (lento mas infalível)
    print("[DEBUG] Estratégia lenta: iterando dataset completo...")
    result = {}
    n = len(dataset)
    for i in range(n):
        item = dataset[i]
        ep_t = item.get("episode_index", None)
        if ep_t is None:
            continue
        ep_int = int(ep_t.item() if hasattr(ep_t, 'item') else ep_t)
        if ep_int not in result:
            result[ep_int] = (i, i + 1)
        else:
            result[ep_int] = (result[ep_int][0], i + 1)
        if i % 500 == 0:
            print(f"[DEBUG] varrendo frame {i}/{n}...")
    return result


def _ep_from(ep_idx: int) -> int:
    return _ep_ranges.get(ep_idx, (0, 0))[0]

def _ep_to(ep_idx: int) -> int:
    return _ep_ranges.get(ep_idx, (0, 0))[1]

def _ep_length(ep_idx: int) -> int:
    r = _ep_ranges.get(ep_idx)
    return (r[1] - r[0]) if r else 0


def _ep_task(ep_idx: int) -> str:
    meta = dataset.meta
    try:
        row = meta.episodes[ep_idx]   # HF Dataset → dict
        task_ids = row.get("tasks", row.get("task_index", []))
        if isinstance(task_ids, (int, np.integer)):
            task_ids = [int(task_ids)]
        elif not isinstance(task_ids, list):
            task_ids = list(task_ids)
        if task_ids:
            tid = int(task_ids[0])
            if isinstance(meta.tasks, pd.DataFrame):
                df = meta.tasks
                if "task_index" in df.columns:
                    rows = df[df["task_index"] == tid]
                    if not rows.empty:
                        col = "task" if "task" in rows.columns else df.columns[-1]
                        return str(rows.iloc[0][col])
                if tid < len(df):
                    col = "task" if "task" in df.columns else df.columns[-1]
                    return str(df.iloc[tid][col])
    except Exception:
        pass
    return ""


# ── API ──────────────────────────────────────────────────────────────────────

def get_dataset_info():
    if dataset is None:
        return {"error": "Dataset não carregado"}

    features_info = {}
    for k, v in dataset.meta.features.items():
        try:
            feat = dict(v)
        except Exception:
            feat = {"raw": str(v)}
        if "shape" in feat:
            try:
                feat["shape"] = list(feat["shape"])
            except Exception:
                feat["shape"] = str(feat["shape"])
        features_info[k] = feat

    episodes_info = [
        {"index": i, "length": _ep_length(i), "task": _ep_task(i)}
        for i in range(dataset.meta.total_episodes)
    ]

    return {
        "repo_id":        dataset.meta.repo_id,
        "total_episodes": dataset.meta.total_episodes,
        "total_frames":   dataset.meta.total_frames,
        "fps":            dataset.meta.fps,
        "features":       features_info,
        "episodes":       episodes_info,
        "camera_keys":    list(dataset.meta.camera_keys),
    }


def get_episode_data(episode_index: int):
    if dataset is None:
        return {"error": "Dataset não carregado"}

    ep_from = _ep_from(episode_index)
    ep_len  = _ep_length(episode_index)

    action_data = []; state_data = []
    pressure_left = []; pressure_right = []
    timestamps = []

    # Get joint names from info.json for richer labels
    action_names = list(dataset.meta.features.get("action", {}).get("names", []))
    state_names  = list(dataset.meta.features.get("observation.state", {}).get("names", []))

    for i in range(ep_from, ep_from + ep_len):
        item = dataset[i]
        ts = item.get("timestamp", torch.tensor(i / dataset.meta.fps))
        timestamps.append(float(ts))
        if "action" in item:
            action_data.append(item["action"].tolist())
        if "observation.state" in item:
            state_data.append(item["observation.state"].tolist())
        if "observation.left_hand_pressure" in item:
            pressure_left.append(item["observation.left_hand_pressure"].tolist())
        if "observation.right_hand_pressure" in item:
            pressure_right.append(item["observation.right_hand_pressure"].tolist())

    return {
        "episode_index":  episode_index,
        "frame_count":    ep_len,
        "timestamps":     timestamps,
        "action":         action_data,
        "action_names":   action_names,
        "state":          state_data,
        "state_names":    state_names,
        "pressure_left":  pressure_left,
        "pressure_right": pressure_right,
    }


def get_frame_image(episode_index: int, frame_offset: int, camera_key: str):
    if dataset is None:
        return None, "Dataset não carregado"
    frame_abs = _ep_from(episode_index) + frame_offset
    try:
        item = dataset[frame_abs]
        if camera_key not in item:
            return None, f"Câmera '{camera_key}' não encontrada"
        img = item[camera_key]
        if img.dtype == torch.float32:
            img = (img * 255).byte()
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = img.permute(1, 2, 0)
        img = img.numpy().astype(np.uint8)
        import io
        try:
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=85)
            return buf.getvalue(), None
        except ImportError:
            import cv2
            _, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return buf.tobytes(), None
    except Exception as e:
        traceback.print_exc()
        return None, str(e)


# ── HTTP ─────────────────────────────────────────────────────────────────────

class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200); self._cors(); self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        params = parse_qs(parsed.query)
        p = lambda k, d=None: params.get(k, [d])[0]
        try:
            if path == "/api/info":
                self._json(get_dataset_info())
            elif path == "/api/episode":
                self._json(get_episode_data(int(p("index", 0))))
            elif path == "/api/image":
                img, err = get_frame_image(int(p("episode", 0)), int(p("frame", 0)), p("camera", ""))
                if err:
                    self._json({"error": err}, 400)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(img)))
                    self._cors(); self.end_headers()
                    self.wfile.write(img)
            elif path == "/":
                self._serve_html()
            else:
                self._json({"error": "Not found"}, 404)
        except Exception as e:
            traceback.print_exc()
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        if urlparse(self.path).path == "/api/delete_episodes":
            n    = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            idxs = body.get("episode_indices", [])
            try:
                global dataset, _ep_ranges
                delete_episodes(dataset, episode_indices=idxs)
                dataset    = LeRobotDataset(dataset_repo_id, root=dataset_root)
                _ep_ranges = _build_ep_ranges()
                self._json({"success": True,
                            "message": f"Episódios {idxs} deletados",
                            "total_episodes": dataset.meta.total_episodes})
            except Exception as e:
                traceback.print_exc()
                self._json({"error": str(e)}, 500)
        else:
            self._json({"error": "Not found"}, 404)

    def _json(self, data, code=200):
        body = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors(); self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        html = Path(__file__).parent / "lerobot_viz_ui.html"
        if not html.exists():
            self._json({"error": "lerobot_viz_ui.html não encontrado na mesma pasta"}, 404)
            return
        data = html.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._cors(); self.end_headers()
        self.wfile.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root",    default=None)
    ap.add_argument("--port",    type=int, default=7860)
    # Novo parâmetro para clonar o dataset
    ap.add_argument("--clone-to", type=str, default=None, help="Cria uma cópia do dataset antes de abrir")
    args = ap.parse_args()

    # ── LÓGICA DE CLONAGEM ──
    if args.clone_to:
        src_path = Path(args.root).resolve()
        
        # Descobre a pasta de origem exata
        if (src_path / "meta" / "info.json").exists():
            src_dir = src_path
            dest_dir = src_path.parent / args.clone_to
        elif (src_path / args.repo_id / "meta" / "info.json").exists():
            src_dir = src_path / args.repo_id
            dest_dir = src_path / args.clone_to
        else:
            print(f"[ERRO] Não achei o dataset original para clonar em: {src_path}")
            sys.exit(1)
            
        if not dest_dir.exists():
            print(f"[INFO] Criando cópia de segurança: {dest_dir.name}...")
            shutil.copytree(src_dir, dest_dir)
            print("[INFO] Cópia concluída!")
        else:
            print(f"[AVISO] A cópia '{args.clone_to}' já existe. Usando ela direto.")
            
        # Muda o alvo do servidor para carregar a cópia
        args.repo_id = args.clone_to
        args.root = str(dest_dir.parent) 

    # Carrega o dataset (original ou a cópia, se --clone-to foi usado)
    load_dataset(args.repo_id, root=args.root)
    srv = HTTPServer(("0.0.0.0", args.port), APIHandler)
    print(f"\n✅  http://localhost:{args.port}   (Ctrl+C para parar)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()

if __name__ == "__main__":
    main()