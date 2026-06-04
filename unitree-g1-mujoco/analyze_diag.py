#!/usr/bin/env python3
"""Analisa /tmp/sim_val/diag.jsonl gerado pela captura --sim (SIM_VAL_DIAG=1)."""
import json
import sys
from pathlib import Path

P = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sim_val/diag.jsonl")
if not P.exists():
    print("SEM diag:", P); sys.exit(1)
rows = [json.loads(l) for l in P.read_text().splitlines() if l.strip()]
by = {}
for r in rows:
    by.setdefault(r["tag"], []).append(r)

print(f"=== diag.jsonl: {len(rows)} eventos | tags={ {k:len(v) for k,v in by.items()} } ===\n")

# connect
for r in by.get("connect_mujoco", []):
    print(f"[connect MuJoCo] left_kp_init={r.get('left_kp_init')} right_kp_init={r.get('right_kp_init')}")
print()

# SALTO: braço cmd vs cur nos primeiros frames
arm = by.get("arm", [])
if arm:
    print("=== SALTO (braço: cmd vs cur por junta, primeiros frames) ===")
    print("frame | junta | cur -> cmd  (Δ)  kp")
    for r in arm[:8]:
        n = r.get("n")
        for jid, d in sorted(r.get("arm", {}).items(), key=lambda x: int(x[0])):
            cur = d.get("cur"); cmd = d.get("cmd"); kp = d.get("kp")
            delta = (cmd - cur) if (cur is not None and cmd is not None) else None
            mark = "  <== SALTO" if (delta is not None and abs(delta) > 0.25) else ""
            print(f"  {n:4} | {jid:3} | {cur} -> {cmd}  (Δ={round(delta,3) if delta is not None else None})  kp={kp}{mark}")
        print("  -")
    # maior salto observado em qualquer frame/junta
    worst = 0.0; worstinfo = None
    for r in arm:
        for jid, d in r.get("arm", {}).items():
            if d.get("cur") is not None and d.get("cmd") is not None:
                dd = abs(d["cmd"] - d["cur"])
                if dd > worst:
                    worst = dd; worstinfo = (r.get("n"), jid, d["cur"], d["cmd"], d["kp"])
    print(f"\n  >> MAIOR |cmd-cur| = {round(worst,3)} rad em frame={worstinfo[0]} junta={worstinfo[1]} "
          f"(cur={worstinfo[2]} cmd={worstinfo[3]} kp={worstinfo[4]})" if worstinfo else "  (sem dados)")
print()

# MÃO ESQUERDA
hand = by.get("hand", [])
if hand:
    print("=== MÃO ESQUERDA (kp comandado ao longo do tempo) ===")
    haskey = [h.get("has_hand_key") for h in hand]
    print(f"has_hand_key: True={haskey.count(True)} False={haskey.count(False)} (se muito False → send_action dá return antes do bloco da mão)")
    # left_kp ao longo do tempo (motor 0)
    for h in hand[:6] + hand[-3:]:
        lkp = h.get("left_kp"); rkp = h.get("right_kp"); lq = h.get("left_q")
        print(f"  n={h.get('n'):4} has_key={h.get('has_hand_key')} left_kp={lkp} right_kp={rkp}")
    # left ficou mole?
    allzero = all(all(abs(x) < 1e-6 for x in h.get("left_kp", [1])) for h in hand if "left_kp" in h)
    anynonzero = any(any(abs(x) > 1e-6 for x in h.get("left_kp", [])) for h in hand if "left_kp" in h)
    print(f"\n  >> left_kp SEMPRE zero? {allzero} | algum frame com left_kp>0? {anynonzero}")
print("\n=== FIM ===")
