#!/usr/bin/env python3
"""
Sweep FIEL (sim real headless) de parametros do copo, mantendo tudo que ja foi validado
(robo aterrado BAND_Z=0.847, mesa topo 0.78, copo em [0.26,0.01], close-frame 85, robo reto).
Varre: escala (tamanho), atrito e massa do copo — as alavancas de "nao escorregar".

Roda SEQUENCIAL (porta 6001) pra ser confiavel: pra cada config gera uma cena temporaria,
sobe o sim real headless, roda o replay, le o log de fisica, mede contatos/elevacao, mata o sim.

Saida: /tmp/sweep_real.json (ranqueado) + print das melhores.

Uso: python pickup_experiments/sweep_real.py
"""
import json, os, re, subprocess, time, signal, glob, math
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
XML = ROOT / "unitree-g1-mujoco/assets/scene_43dof.xml"
ENVBASE = dict(os.environ, BAND_Z="0.847", ONSCREEN_OVERRIDE="false")

# espaco de busca (alavancas anti-escorregamento), demais params fixos (validados)
SCALES = [0.0016, 0.0018, 0.0020]
FRICTIONS = [1.5, 4.0]
MASSES = [0.05, 0.15]
CLOSE_FRAME = 85
CUP_X, CUP_Y = 0.26, 0.01


def make_scene(scale, friction, mass):
    txt = XML.read_text()
    txt = re.sub(r'(<mesh name="cup" file="../cup.stl" scale=")[^"]+(")',
                 rf'\g<1>{scale} {scale} {scale}\g<2>', txt)
    txt = re.sub(r'(<body name="objeto_customizado" pos=")[^"]+(">)',
                 rf'\g<1>{CUP_X} {CUP_Y} 0.82\g<2>', txt)
    txt = re.sub(r'(<geom name="geometria_bloco"[^>]*?friction=")[^"]+(")',
                 rf'\g<1>{friction} 0.005 0.0001\g<2>', txt)
    txt = re.sub(r'(<geom name="geometria_bloco"[^>]*?mass=")[^"]+(")',
                 rf'\g<1>{mass}\g<2>', txt)
    p = XML.parent / f"_sweep_tmp.xml"
    p.write_text(txt)
    return p


def wait_port(timeout=25):
    for _ in range(timeout * 2):
        r = subprocess.run("ss -ltn 2>/dev/null | grep -q 6001", shell=True)
        if r.returncode == 0:
            return True
        time.sleep(0.5)
    return False


def run_one(scale, friction, mass):
    scene = make_scene(scale, friction, mass)
    env = dict(ENVBASE, ROBOT_SCENE_OVERRIDE=str(scene.resolve()), DISPLAY=":1")
    sim = subprocess.Popen(["python", "pickup_experiments/run_sim_visible.py"],
                           cwd=str(ROOT), env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           preexec_fn=os.setsid)
    try:
        if not wait_port():
            return None
        time.sleep(1.5)
        before = set(glob.glob("/tmp/action_log_*.jsonl"))
        subprocess.run(
            ["python", "pickup_experiments/replay_dataset.py", "--data", "right",
             "--ep", "18", "--speed", "0.5", "--torso", "0",
             "--close-frame", str(CLOSE_FRAME)],
            cwd=str(ROOT), env=env, timeout=90,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logs = sorted(glob.glob("/tmp/action_log_*.jsonl"), key=os.path.getmtime)
        log = logs[-1]
        st = [json.loads(l) for l in open(log) if "cup_position" in l]
        c0 = st[0]["cup_position"]; cf = st[-1]["cup_position"]
        zmax = max(s["cup_position"][2] for s in st)
        contacts = max(s.get("num_contacts", 0) for s in st)
        fp = [json.loads(l) for l in open(log) if "finger_positions" in l]
        mind = 9.9
        for s in fp:
            for k, p in s["finger_positions"].items():
                if "right" in k:
                    mind = min(mind, math.dist(p, s["cup_position"]))
        return {
            "scale": scale, "friction": friction, "mass": mass,
            "lift_net_cm": round((cf[2] - c0[2]) * 100, 1),
            "lift_peak_cm": round((zmax - c0[2]) * 100, 1),
            "min_dist_cm": round(mind * 100, 1), "contacts": contacts,
            "held": bool((cf[2] - c0[2]) > 0.04 and cf[2] > 0.5),
        }
    finally:
        try:
            os.killpg(os.getpgid(sim.pid), signal.SIGKILL)
        except Exception:
            pass
        time.sleep(2)


def main():
    results = []
    cfgs = [(s, f, m) for s in SCALES for f in FRICTIONS for m in MASSES]
    print(f"[sweep_real] {len(cfgs)} configs (sequencial, sim real headless)")
    for i, (s, f, m) in enumerate(cfgs):
        r = run_one(s, f, m)
        if r:
            results.append(r)
            print(f"  [{i+1}/{len(cfgs)}] scale={s} fric={f} mass={m} -> "
                  f"held={r['held']} liq={r['lift_net_cm']} pico={r['lift_peak_cm']} "
                  f"min={r['min_dist_cm']} cont={r['contacts']}")
        else:
            print(f"  [{i+1}/{len(cfgs)}] scale={s} fric={f} mass={m} -> FALHA (sim nao subiu)")
        json.dump(results, open("/tmp/sweep_real.json", "w"), indent=2)
    results.sort(key=lambda r: (not r["held"], -r["lift_net_cm"], r["min_dist_cm"]))
    print("\n=== TOP 5 (fiel) ===")
    for r in results[:5]:
        print(f"  held={r['held']} liq={r['lift_net_cm']}cm pico={r['lift_peak_cm']}cm "
              f"min={r['min_dist_cm']}cm cont={r['contacts']} | scale={r['scale']} fric={r['friction']} mass={r['mass']}")
    print("salvo em /tmp/sweep_real.json")


if __name__ == "__main__":
    main()
