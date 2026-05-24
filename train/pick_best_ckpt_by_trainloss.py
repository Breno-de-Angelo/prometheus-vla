#!/usr/bin/env python3
"""Escolhe o checkpoint com MENOR train loss suavizado de um run sem val_dataset.
O pi05_depth_uni_lf nao tem holdout, entao o lerobot nunca cria o symlink `best`;
train loss e o unico criterio disponivel. Cada linha de log = log_freq steps (exato),
entao reconstruo o step real por indice de linha e tiro a media numa janela em volta
de cada checkpoint salvo (robusto ao ruido do platô).

stdout: SO o nome do dir escolhido (ex: '016000'). stderr: diagnostico (vai pro log)."""
import argparse, os, re, sys

LOG_FREQ = 20  # do yaml: log_freq

def parse_log(log):
    pat = re.compile(r"step:\S+\s+smpl:\S+.*?loss\s+[\d.]+\s+\(([\d.]+)")
    rows, i = [], 0
    with open(log, errors="ignore") as f:
        for line in f:
            if "un_train.py" not in line:
                continue
            m = pat.search(line)
            if not m:
                continue
            i += 1
            rows.append((LOG_FREQ * i, float(m.group(1))))  # (step exato, loss suavizado)
    return rows

def win_loss(rows, step, w):
    vals = [l for (s, l) in rows if step - w <= s <= step + w]
    if vals:
        return sum(vals) / len(vals)
    return min(rows, key=lambda r: abs(r[0] - step))[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--window", type=int, default=600)
    ap.add_argument("--min-step", type=int, default=2000, help="ignora warmup inicial")
    a = ap.parse_args()

    steps = [d for d in sorted(os.listdir(a.ckpt_dir)) if d.isdigit()] if os.path.isdir(a.ckpt_dir) else []
    if not steps:
        print("last"); print("[pick] sem checkpoints numerados -> last", file=sys.stderr); return
    rows = parse_log(a.log) if os.path.exists(a.log) else []
    if not rows:
        print(steps[-1]); print("[pick] sem log -> ultimo (%s)" % steps[-1], file=sys.stderr); return

    cands = [(win_loss(rows, int(d), a.window), d) for d in steps if int(d) >= a.min_step]
    if not cands:
        print(steps[-1]); print("[pick] nenhum ckpt >= min-step -> %s" % steps[-1], file=sys.stderr); return

    cands.sort()
    best_loss, best = cands[0]
    print("[pick] loss suavizado (janela +/-%d) por ckpt:" % a.window, file=sys.stderr)
    for loss, d in sorted(cands, key=lambda c: int(c[1])):
        print("   %s  %.4f%s" % (d, loss, "   <== BEST" if d == best else ""), file=sys.stderr)
    print("[pick] BEST = %s (loss %.4f)" % (best, best_loss), file=sys.stderr)
    print(best)  # stdout: so o nome do dir escolhido

if __name__ == "__main__":
    main()
