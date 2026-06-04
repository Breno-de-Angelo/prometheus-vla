#!/usr/bin/env python3
"""Analisa o ultimo action_log_*.jsonl: altura do copo, contatos e dist dedo-copo."""
import json, glob, math, os

logs = sorted(glob.glob("/tmp/action_log_*.jsonl"), key=os.path.getmtime)
log = logs[-1]
states = [json.loads(l) for l in open(log) if 'physics_state' in l]
print(f"log: {log}  ({len(states)} estados)")
if not states:
    raise SystemExit("sem physics_state")

z0 = states[0]['cup_position'][2]
zmax = max(s['cup_position'][2] for s in states)
zend = states[-1]['cup_position'][2]
maxcontacts = max(s['num_contacts'] for s in states)
maxhand = max(len(s['hand_cup_contacts']) for s in states)

best = 9.9; bs = None
for s in states:
    c = s['cup_position']
    for k, p in s['finger_positions'].items():
        d = math.dist(p, c)
        if d < best:
            best = d; bs = (s['step'], k.replace('right_hand_', ''), round(d*100, 1))

print(f"cup Z: inicio={z0:.3f}  max={zmax:.3f}  fim={zend:.3f}")
print(f"elevacao liquida (fim-inicio) = {(zend-z0)*100:+.1f}cm | pico = {(zmax-z0)*100:+.1f}cm")
print(f"contatos: max_total={maxcontacts}  max_mao-copo={maxhand}")
print(f"menor dist dedo-copo: {bs} cm")
ok = (zend - z0) > 0.05 and zend > 0.5
print(f">>> {'PEGOU E MANTEVE LEVANTADO ✓' if ok else 'NAO MANTEVE ✗'}")
