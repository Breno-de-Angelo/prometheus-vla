#!/usr/bin/env python3
"""Recuperação forense do RGB de um dataset G1 cujo vídeo consolidado ficou
invertido/quebrado e cujos PNGs RGB foram embaralhados por retentativas
(clear_episode_buffer(delete_images=False) reusou índices de frame).

NÃO modifica o dataset original (lê read-only; escreve só em --out).

Método (validado em 2026-06-06):
  1. ANCHOR estado: casa observation.state[braço+mão dir] do parquet contra
     debug_logs/episode_*.csv obs.*.q (resíduo ~0, contraste >1e6 quando trava).
     Localiza o good-take na timeline de gravação -> debug offset.
  2. ANCHOR brilho: o good-take (debug cam.head_camera.mean) é casado por
     correlação-normalizada (NCC) contra o brilho dos PNGs RGB da pasta do ep.
     Acha o offset do good-take dentro dos PNGs sobreviventes.
  3. Se ambos travam (state res<0.005 E NCC>=THR), extrai os N PNGs e encoda
     vídeo forward. Senão marca o ep como RGB PERDIDO (frames sobrescritos).

Saída: <out>/videos_recuperados/epN.mp4 (recuperados) + MANIFEST.json.

Uso:
    python tools/recover_rgb_forensic.py --root <dataset> --out <dir_novo> [--thr 0.97]
"""
import argparse, glob, io, json, os, subprocess, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ARM = ['obs.kRightShoulderPitch.q','obs.kRightShoulderRoll.q','obs.kRightShoulderYaw.q',
       'obs.kRightElbow.q','obs.kRightWristRoll.q','obs.kRightWristPitch.q','obs.kRightWristYaw.q']
HAND = ['obs.right_hand_thumb_0_joint.q','obs.right_hand_thumb_1_joint.q','obs.right_hand_thumb_2_joint.q',
        'obs.right_hand_index_0_joint.q','obs.right_hand_index_1_joint.q','obs.right_hand_middle_0_joint.q','obs.right_hand_middle_1_joint.q']
ACT_DIMS = list(range(7, 14)) + list(range(21, 28))   # braço dir (7-13) + mão dir (21-27)


def png_means(pdir):
    fs = sorted(glob.glob(pdir + '/frame-*.png'))
    return np.array([float(np.asarray(Image.open(f).convert('L').resize((64, 36))).mean()) for f in fs]), fs


def ncc(target, series):
    N = len(target); t = target - target.mean(); tn = np.linalg.norm(t) + 1e-9
    if len(series) < N: return -1, -1
    cc = []
    for o in range(len(series) - N + 1):
        w = series[o:o + N]; w = w - w.mean()
        cc.append(float(np.dot(w, t) / ((np.linalg.norm(w) + 1e-9) * tn)))
    cc = np.array(cc); o = int(np.argmax(cc)); return o, float(cc[o])


def encode(frame_paths, out_mp4, fps):
    tmp = Path(tempfile.mkdtemp(prefix='rrgb_'))
    for i, fp in enumerate(frame_paths):
        os.symlink(os.path.abspath(fp), tmp / f'f-{i:06d}.png')
    cmd = ['ffmpeg', '-y', '-framerate', str(fps), '-i', str(tmp / 'f-%06d.png'),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'fast', str(out_mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for f in tmp.glob('*'): f.unlink()
    tmp.rmdir()
    if r.returncode != 0: raise RuntimeError(r.stderr[-300:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--thr', type=float, default=0.97, help='limiar NCC de brilho p/ aceitar recuperação')
    ap.add_argument('--fps', type=int, default=30)
    args = ap.parse_args()

    D = Path(args.root).resolve()
    OUT = Path(args.out).resolve(); (OUT / 'videos_recuperados').mkdir(parents=True, exist_ok=True)
    vkey = 'observation.images.head_camera'

    df = pd.concat([pd.read_parquet(f, columns=['episode_index', 'frame_index', 'observation.state'])
                    for f in sorted(glob.glob(str(D / 'data/chunk-000/file-*.parquet')))], ignore_index=True)
    eps = sorted(int(e) for e in df['episode_index'].unique())

    manifest = {'recovered': [], 'lost': [], 'state_fail': []}
    for EP in eps:
        sub = df[df['episode_index'] == EP].sort_values('frame_index')
        Sp = np.stack(sub['observation.state'].to_numpy()).astype(float)[:, ACT_DIMS]; N = len(Sp)
        dl = D / f'debug_logs/episode_{EP:06d}.csv'
        if not dl.exists(): manifest['state_fail'].append(EP); continue
        dbg = pd.read_csv(dl)
        Db = np.nan_to_num(dbg[ARM + HAND].to_numpy().astype(float))
        if len(Db) < N: manifest['state_fail'].append(EP); continue
        errs = np.array([np.mean((Db[o:o + N] - Sp) ** 2) for o in range(len(Db) - N + 1)])
        so = int(np.argmin(errs)); res = float(errs[so]); med = float(np.median(errs))
        if not (res < 0.005 and med / max(res, 1e-9) > 50):
            manifest['state_fail'].append(EP); continue
        cam = dbg['cam.head_camera.mean'].to_numpy().astype(float); T = cam[so:so + N]
        pdir = str(D / f'images/{vkey}/episode-{EP:06d}')
        Rp, fs = png_means(pdir)
        po, corr = ncc(T, Rp)
        if corr >= args.thr:
            frames = fs[po:po + N]
            encode(frames, OUT / 'videos_recuperados' / f'ep{EP}.mp4', args.fps)
            manifest['recovered'].append({'ep': EP, 'N': N, 'debug_off': so, 'png_off': po, 'corr': round(corr, 4)})
        else:
            manifest['lost'].append({'ep': EP, 'best_corr': round(corr, 4)})

    (OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    print(f"recuperados: {len(manifest['recovered'])} | perdidos: {len(manifest['lost'])} | state_fail: {len(manifest['state_fail'])}")
    print('recuperados:', [r['ep'] for r in manifest['recovered']])
    print('out:', OUT)


if __name__ == '__main__':
    main()
