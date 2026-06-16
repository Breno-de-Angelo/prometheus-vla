#!/usr/bin/env python3
"""Lê o JSONL do --debug (G1_DEBUG_IO) e resume envio/recebimento do robô.
Uso: python lerobot-ext/analyze_debug_io.py /tmp/g1_debug_io_YYYYMMDD_HHMMSS.jsonl"""
import json
import sys
import numpy as np

p = sys.argv[1] if len(sys.argv) > 1 else None
if not p:
    print("uso: analyze_debug_io.py <arquivo.jsonl>"); sys.exit(1)
rows = [json.loads(l) for l in open(p) if l.strip()]
sends = [r for r in rows if r.get("io") == "send"]
recvs = [r for r in rows if r.get("io") == "recv"]
print(f"=== {p} ===\nframes: send={len(sends)} recv={len(recvs)}\n")

# ---- MÃO ESQUERDA enviada (kp deve ser 0 = mole) ----
if sends:
    lk = np.array([[m[1] for m in r["left_hand"]] for r in sends if "left_hand" in r], dtype=float)
    rk = np.array([[m[1] for m in r["right_hand"]] for r in sends if "right_hand" in r], dtype=float)
    print("MÃO ESQUERDA (kp ENVIADO):")
    print(f"   primeiro frame: {lk[0].tolist() if len(lk) else '-'}")
    print(f"   max kp em qualquer frame/motor: {lk.max() if lk.size else '-'}  (0 = mole; >0 = controlada)")
    print(f"   frames com algum kp>0: {int((lk.max(axis=1) > 0).sum())}/{len(lk)}")
    print("MÃO DIREITA (kp ENVIADO):")
    print(f"   max kp: {rk.max() if rk.size else '-'} (esperado ~0.8-1.0)")
    # tau/mode da esquerda
    lt = np.array([[m[3] for m in r["left_hand"]] for r in sends if "left_hand" in r], dtype=float)
    lmode = [r["left_hand"][0][4] for r in sends if "left_hand" in r]
    print(f"   ESQ tau max={lt.max() if lt.size else '-'} | mode amostra={lmode[0] if lmode else '-'}")

# ---- SALTO do braço (sent vs recebido) ----
if sends and recvs:
    print("\nBRAÇO (q enviado vs medido, por junta — salto = enviado longe do medido):")
    s0 = sends[0].get("arm", {}); r0 = recvs[0].get("arm_state_q", {})
    worst = 0.0; wkey = None
    # casa send[i] com recv[i] aprox
    for s, rc in zip(sends, recvs):
        for k, qk in s.get("arm", {}).items():
            mq = rc.get("arm_state_q", {}).get(k)
            if mq is not None:
                d = abs(qk[0] - mq)
                if d > worst: worst = d; wkey = k
    print(f"   maior |q_enviado - q_medido| = {worst:.3f} rad (junta {wkey})  [<0.1 ok; ~0.4 = salto]")

# ---- TÁTIL recebido (zero ou real?) ----
if recvs:
    def pstat(key):
        a = np.array([r[key] for r in recvs if key in r], dtype=float)
        if a.size == 0: return "ausente"
        return f"{round(100*np.count_nonzero(a)/a.size,1)}% nonzero (min={a.min():.0f} max={a.max():.0f})"
    print("\nTÁTIL/PRESSÃO RECEBIDA:")
    print(f"   left_pressure : {pstat('left_pressure')}")
    print(f"   right_pressure: {pstat('right_pressure')}")
    lq = np.array([r["left_hand_q"] for r in recvs if "left_hand_q" in r], dtype=float)
    if lq.size:
        print(f"   left_hand_q medido: varia {lq.min():.3f}..{lq.max():.3f} (se a esq é mole, tende a ficar parada/cair)")
print("\n=== FIM ===")
