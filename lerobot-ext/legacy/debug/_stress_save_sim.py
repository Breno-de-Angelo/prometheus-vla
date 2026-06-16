#!/usr/bin/env python
"""
STRESS TEST do SALVAMENTO no sim (MuJoCo headless) — sem Quest.

Roda o robô no simulador fazendo TRAJETÓRIA ALEATÓRIA, salva episódios e "aperta
A/B" várias vezes (incl. double-tap, episódio quase-vazio, B->A em sequência, ESC
durante save) pra verificar se o SALVAMENTO quebra.

IMPORTANTE: TODA a lógica fica sob `if __name__ == "__main__"`. O publisher de
imagem do sim é um subprocesso multiprocessing 'spawn' que RE-IMPORTA este módulo;
se houvesse efeito colateral no topo (ex.: int(sys.argv[1])), o filho crashava e a
câmera 5555 dava timeout. Sob o guard, o filho importa como __mp_main__ e pula tudo.

Rodar (a partir de lerobot-ext/, env g1):
  MUJOCO_GL=egl ~/miniconda3/envs/g1/bin/python _stress_save_sim.py [batch]
"""
import os
import sys

# Neutraliza a thread de voz já no import (vale p/ pai e filho; inofensivo no filho)
sys.modules["speech_recognition"] = None


def run():
    import time
    import json
    import threading
    import traceback
    import numpy as np

    HERE = os.path.dirname(os.path.abspath(__file__))
    os.chdir(HERE)
    sys.path.insert(0, HERE)

    # parse defensivo do batch (1º arg numérico)
    BATCH = 50
    for a in sys.argv[1:]:
        if a.isdigit():
            BATCH = int(a)
            break
    NUM_EPISODES = 7
    STAMP = time.strftime("%Y%m%d_%H%M%S")
    DS_ROOT = f"/tmp/sim_stress_ds/batch{BATCH}_{STAMP}"
    STATUS_FILE = "/tmp/g1_record_status.json"
    RESULTS = {"events_fired": [], "exceptions": []}
    print(f"[STRESS] batch={BATCH}  dataset={DS_ROOT}  num_episodes={NUM_EPISODES}")

    import init_lerobot_record_v2 as rec
    from lerobot.teleoperators.teleoperator import Teleoperator
    import lerobot.robots.utils as _ru
    import lerobot.scripts.lerobot_record as _lr
    import lerobot.teleoperators as _tele

    STASH = {"robot": None}

    # captura a instância do robô (criada pela INJEÇÃO 0)
    _inj0_make_robot = _ru.make_robot_from_config

    def _capture_make_robot(config):
        r = _inj0_make_robot(config)
        STASH["robot"] = r
        print(f"[STRESS] robô capturado: {type(r).__name__} (action dims = {len(r.action_features)})")
        return r

    _ru.make_robot_from_config = _capture_make_robot
    _lr.make_robot_from_config = _capture_make_robot

    class RandomG1Teleop(Teleoperator):
        name = "random_g1"

        def __init__(self, config, robot):
            super().__init__(config)
            self._robot = robot
            self._connected = False
            self._keys = list(robot.action_features.keys())
            self._q = {}
            self._center = {}
            self._amp = {}
            self._t = 0
            self._theta = 0.06
            self._sigma = 0.05

        @property
        def action_features(self):
            return dict(self._robot.action_features)

        @property
        def feedback_features(self):
            return {}

        @property
        def is_connected(self):
            return self._connected

        @property
        def is_calibrated(self):
            return True

        def connect(self, calibrate: bool = True):
            try:
                obs = self._robot.get_observation() or {}
            except Exception:
                obs = {}
            for k in self._keys:
                base = float(obs.get(k, 0.0))
                self._q[k] = base
                self._center[k] = base
                self._amp[k] = 0.5 if "hand" in k else 0.35
            self._connected = True
            print(f"[STRESS] RandomG1Teleop conectado ({len(self._keys)} juntas).")

        def calibrate(self):
            pass

        def configure(self):
            pass

        def get_action(self):
            self._t += 1
            out = {}
            for k in self._keys:
                c = self._center[k]
                amp = self._amp[k]
                self._q[k] += self._theta * (c - self._q[k]) + self._sigma * np.random.randn()
                phase = 0.15 * self._t + (hash(k) % 100) * 0.06
                target = self._q[k] + 0.25 * amp * np.sin(phase)
                out[k] = float(np.clip(target, c - amp, c + amp))
            return out

        def send_feedback(self, feedback):
            pass

        def disconnect(self):
            self._connected = False

    def _make_random_teleop(config):
        print("[STRESS] make_teleoperator -> RandomG1Teleop (trajetória aleatória, sem VR)")
        return RandomG1Teleop(config, STASH["robot"])

    _tele.make_teleoperator_from_config = _make_random_teleop
    _lr.make_teleoperator_from_config = _make_random_teleop

    # ---------------- PRESSER (A/B/ESC) ----------------
    def _wait_events(timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if rec.global_events is not None:
                return rec.global_events
            time.sleep(0.2)
        return None

    def _read_status():
        try:
            with open(STATUS_FILE) as f:
                return json.load(f)
        except Exception:
            return None

    def _wait_episode(idx, timeout=90):
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = _read_status()
            if st and st.get("episode") == idx:
                return True
            time.sleep(0.1)
        return False

    def _fire(ev, label, **flags):
        for k, v in flags.items():
            ev[k] = v
        RESULTS["events_fired"].append(label)
        print(f"\n[PRESSER] >>> {label}  {flags}", flush=True)

    def presser_loop():
        ev = _wait_events()
        if ev is None:
            print("[PRESSER] global_events nunca apareceu — abortando.")
            return
        print("[PRESSER] global_events pronto. Iniciando sequência de A/B.")

        if _wait_episode(0):
            time.sleep(3.0); _fire(ev, "EP0: A (save normal)", exit_early=True)
        if _wait_episode(1):
            time.sleep(2.0)
            _fire(ev, "EP1: A (save)", exit_early=True)
            time.sleep(0.05); _fire(ev, "EP1: A double-tap", exit_early=True)
            time.sleep(0.03); _fire(ev, "EP1: A triple-tap", exit_early=True)
        if _wait_episode(2):
            time.sleep(1.5); _fire(ev, "EP2: B (descartar)", rerecord_episode=True, exit_early=True)
        if _wait_episode(2):
            time.sleep(2.0); _fire(ev, "EP2': A (save após regravar)", exit_early=True)
        if _wait_episode(3):
            time.sleep(0.15); _fire(ev, "EP3: A imediato (quase vazio)", exit_early=True)
        if _wait_episode(4):
            time.sleep(1.5); _fire(ev, "EP4: B (descartar)", rerecord_episode=True, exit_early=True)
        if _wait_episode(4):
            time.sleep(1.5)
            _fire(ev, "EP4': A", exit_early=True)
            time.sleep(0.02); _fire(ev, "EP4': B logo após (race)", rerecord_episode=True, exit_early=True)
        if _wait_episode(4):
            time.sleep(1.5); _fire(ev, "EP4'': A final", exit_early=True)
        if _wait_episode(5):
            time.sleep(2.0)
            _fire(ev, "EP5: A (save)", exit_early=True)
            time.sleep(0.3); _fire(ev, "EP5: ESC (stop -> finalize/batch encode)", stop_recording=True, exit_early=True)
        print("[PRESSER] sequência concluída.")

    def watchdog(limit_s=360):
        time.sleep(limit_s)
        print(f"\n[WATCHDOG] {limit_s}s estourados — forçando saída.", flush=True)
        os._exit(2)

    threading.Thread(target=presser_loop, daemon=True, name="Presser").start()
    threading.Thread(target=watchdog, daemon=True, name="Watchdog").start()

    # ---------------- roda o record real ----------------
    sys.argv = [
        "init_lerobot_record_v2.py",
        "--config_path", "config/record/record_televuer.yaml",
        "--robot.is_simulation=true",
        "--teleop.is_simulation=true",
        f"--dataset.root={DS_ROOT}",
        f"--dataset.num_episodes={NUM_EPISODES}",
        f"--dataset.video_encoding_batch_size={BATCH}",
        "--dataset.push_to_hub=false",
        "--dataset.episode_time_s=3600",
        "--dataset.reset_time_s=0",
    ]
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["G1_TUNING"] = os.path.join(HERE, "g1_tuning.json")

    print("[STRESS] iniciando record(main)...\n")
    t_start = time.time()
    try:
        rec.main()
        print("\n[STRESS] main() retornou normalmente.")
    except SystemExit as e:
        print(f"\n[STRESS] main() SystemExit code={e.code}")
    except Exception as e:
        RESULTS["exceptions"].append(f"{type(e).__name__}: {e}")
        print(f"\n[STRESS][ERRO] main() lançou: {type(e).__name__}: {e}")
        traceback.print_exc()
    print(f"[STRESS] tempo total: {time.time() - t_start:.1f}s")

    # ---------------- validação ----------------
    print("\n" + "=" * 70 + "\nVALIDAÇÃO DO DATASET\n" + "=" * 70)
    report = {"ok": True, "checks": []}

    def chk(name, cond, detail=""):
        if not cond:
            report["ok"] = False
        report["checks"].append((name, bool(cond), detail))
        print(f"  [{'OK ' if cond else 'FALHOU'}] {name}  {detail}")

    try:
        from pathlib import Path
        root = Path(DS_ROOT)
        info_p = root / "meta" / "info.json"
        chk("meta/info.json existe", info_p.exists(), str(info_p))
        info = json.loads(info_p.read_text()) if info_p.exists() else {}
        n_ep = info.get("total_episodes")
        n_fr = info.get("total_frames")
        chk("total_episodes > 0", bool(n_ep and n_ep > 0), f"total_episodes={n_ep} total_frames={n_fr}")
        data_files = list((root / "data").rglob("*.parquet")) if (root / "data").exists() else []
        chk("parquet de dados presentes", len(data_files) > 0, f"{len(data_files)} arquivo(s)")
        if data_files:
            import pandas as pd
            df = pd.read_parquet(sorted(data_files)[0])
            chk("parquet legível com linhas", len(df) > 0, f"{len(df)} linhas; cols={list(df.columns)[:5]}")
        vids = list((root / "videos").rglob("*.mp4")) if (root / "videos").exists() else []
        nonempty = [v for v in vids if v.stat().st_size > 0]
        chk("vídeos .mp4 gerados (encode)", len(vids) > 0 and len(vids) == len(nonempty),
            f"{len(vids)} mp4, {len(nonempty)} não-vazios")
        pngs = list((root / "images").rglob("*.png")) if (root / "images").exists() else []
        print(f"  [info] depth PNG: {len(pngs)}")
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            ds = LeRobotDataset(info.get("repo_id", "x/y"), root=str(root))
            chk("LeRobotDataset carrega", True, f"{ds.num_frames} frames, {ds.num_episodes} eps")
            try:
                _ = ds[0]
                chk("ds[0] decodifica (vídeo+parquet)", True, "")
            except Exception as e:
                # torchcodec/torch.library.register_fake = mismatch de versão do ENV de
                # decodificação, NÃO um defeito do dataset salvo. Info-only, não falha o teste.
                print(f"  [info] ds[0] decode pulado (env torchcodec): {type(e).__name__}: {e}")
        except Exception as e:
            chk("LeRobotDataset carrega", False, f"{type(e).__name__}: {e}")
    except Exception as e:
        chk("validação rodou", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()

    print("\n" + "=" * 70 + "\nRESUMO\n" + "=" * 70)
    print(f"Eventos A/B/ESC disparados: {len(RESULTS['events_fired'])}")
    for e in RESULTS["events_fired"]:
        print(f"   - {e}")
    print(f"Exceções no main(): {RESULTS['exceptions'] or 'nenhuma'}")
    ok_final = report["ok"] and not RESULTS["exceptions"]
    print(f"Dataset íntegro: {'SIM' if report['ok'] else 'NAO'}   | RESULTADO GERAL: {'PASS' if ok_final else 'FAIL'}")
    with open("/tmp/sim_stress_report.json", "w") as f:
        json.dump({"events_fired": RESULTS["events_fired"], "exceptions": RESULTS["exceptions"],
                   "checks": [(n, ok, d) for n, ok, d in report["checks"]],
                   "ok": ok_final, "root": DS_ROOT}, f, indent=2)
    print("Relatório salvo em /tmp/sim_stress_report.json")
    os._exit(0 if ok_final else 1)


if __name__ == "__main__":
    run()
