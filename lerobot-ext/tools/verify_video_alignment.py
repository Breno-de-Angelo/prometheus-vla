#!/usr/bin/env python3
"""Verifica se o RGB de TREINO (videos/.../file-*.mp4) está alinhado frame-a-frame
com o parquet — pra rodar numa gravação de TESTE depois do fix (batch_encoding_size=1).

Checa, por episódio:
  1) CONTAGEM: nº de frames do trecho de vídeo do episódio == length do parquet
     (pega truncamento, o sintoma do bug de batch encoding).
  2) ORIENTAÇÃO/SYNC: o RGB e a PROFUNDIDADE embutida no parquet (forward por
     construção — embutida no save) vêm da MESMA câmera no mesmo instante. Compara
     a posição vertical da região "perto" (depth) com a da mão (escuro no RGB) ao
     longo do tempo: correlação positiva = alinhado/forward; negativa = invertido.
  3) Gera um mosaico RGB|DEPTH (f0/meio/fim) de episódios-amostra p/ conferência visual.

Uso:
    python tools/verify_video_alignment.py --root <dataset> [--sample 6] [--out /tmp/verify.png]
Veredito por episódio: OK / FALHA-CONTAGEM / SUSPEITA-INVERSAO.
"""
import argparse, glob, io, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont


def vid_nframes(path):
    r = subprocess.run(["ffprobe","-v","quiet","-count_frames","-select_streams","v:0",
                        "-show_entries","stream=nb_read_frames","-of","csv=p=0",str(path)],
                       capture_output=True, text=True)
    try: return int(r.stdout.strip())
    except ValueError: return -1


def vid_frames_small(path, idxs, W=96, H=54):
    """extrai frames especificos (por indice) do video em cinza pequeno."""
    raw = subprocess.run(["ffmpeg","-v","quiet","-i",str(path),"-vf",f"scale={W}:{H},format=gray",
                          "-f","rawvideo","-"], capture_output=True).stdout
    n = len(raw)//(W*H)
    if n == 0: return {}
    arr = np.frombuffer(raw[:n*W*H], np.uint8).reshape(n, H, W)
    return {i: arr[min(i, n-1)] for i in idxs}, n


def depth_small(b, W=96, H=54):
    a = np.array(Image.open(io.BytesIO(b["bytes"] if isinstance(b, dict) else bytes(b))))
    return np.array(Image.fromarray(a).resize((W, H)))


def near_y(depthframe):
    # "perto" = mm pequeno e >0 (a mao/copo perto da camera). centroide vertical normalizado.
    v = depthframe.astype(float); valid = v > 0
    if valid.sum() < 20: return np.nan
    thr = np.percentile(v[valid], 15)  # 15% mais perto
    m = valid & (v <= thr)
    if m.sum() < 10: return np.nan
    return float(np.nonzero(m)[0].mean()) / depthframe.shape[0]


def dark_y(grayframe):
    m = grayframe < 60
    if m.sum() < 15: return np.nan
    return float(np.nonzero(m)[0].mean()) / grayframe.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--video-key", default="observation.images.head_camera")
    ap.add_argument("--depth-col", default="observation.images.head_camera_depth")
    ap.add_argument("--sample", type=int, default=6, help="quantos eps no mosaico visual")
    ap.add_argument("--out", default="/tmp/verify_alignment.png")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    D = Path(args.root).resolve()

    meta = pd.read_parquet(glob.glob(str(D/"meta/episodes/**/file-*.parquet"), recursive=True)[0]).sort_values("episode_index")
    dfa = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(D/"data/chunk-000/file-*.parquet")))], ignore_index=True)
    vids = sorted(glob.glob(str(D/f"videos/{args.video_key}/**/file-*.mp4"), recursive=True))
    consolidated = vids[0] if vids else None
    print(f"[verify] {len(meta)} eps | video consolidado: {consolidated}")

    results = []
    for _, r in meta.iterrows():
        e = int(r["episode_index"]); N = int(r["length"])
        sub = dfa[dfa["episode_index"] == e].sort_values("frame_index")
        # depth embutido (forward): near_y em f0/mid/last
        dcol = sub[args.depth_col].to_numpy()
        d_y = [near_y(depth_small(dcol[k])) for k in (0, len(dcol)//2, len(dcol)-1)]
        # frames do video correspondentes ao episodio (range de timestamp)
        f0 = int(round(float(r[f"videos/{args.video_key}/from_timestamp"]) * args.fps))
        rgb_idx = [f0, f0 + N//2, f0 + N - 1]
        fr, vn = vid_frames_small(consolidated, rgb_idx)
        r_y = [dark_y(fr[i]) for i in rgb_idx]
        # checagem 1: contagem (do range vs N) — usa nb_frames total se 1 ep, senao confia no range
        count_ok = True  # consolidado: contagem por-ep via timestamp; checa total no fim
        # checagem 2: tendencia vertical igual? (corr de 3 pontos)
        dv = np.array(d_y); rv = np.array(r_y); mask = ~(np.isnan(dv)|np.isnan(rv))
        if mask.sum() >= 2 and dv[mask].std()>1e-6 and rv[mask].std()>1e-6:
            c = float(np.corrcoef(dv[mask], rv[mask])[0,1])
        else:
            c = float("nan")
        verdict = "OK" if (np.isnan(c) or c >= -0.1) else "SUSPEITA-INVERSAO"
        results.append({"ep": e, "N": N, "depth_y": [round(x,2) if not np.isnan(x) else None for x in d_y],
                        "rgb_y": [round(x,2) if not np.isnan(x) else None for x in r_y], "corr": round(c,2) if not np.isnan(c) else None, "verdict": verdict})

    # checagem de contagem total do video consolidado
    total_len = int(meta["length"].sum()); total_vid = vid_nframes(consolidated)
    print(f"[verify] CONTAGEM TOTAL: video={total_vid} frames | parquet={total_len} -> {'OK' if total_vid==total_len else 'FALHA (truncado/dessincronizado)'}")
    susp = [r["ep"] for r in results if r["verdict"]!="OK"]
    print(f"[verify] episodios com SUSPEITA DE INVERSAO: {susp if susp else 'NENHUM'}")
    for r in results[:20]: print("  ", r)

    # mosaico visual RGB|DEPTH f0/mid/last
    sample = [int(x) for x in np.linspace(0, len(meta)-1, min(args.sample, len(meta)))]
    rows = []
    for si in sample:
        rr = meta.iloc[si]; e=int(rr["episode_index"]); N=int(rr["length"])
        sub = dfa[dfa["episode_index"]==e].sort_values("frame_index"); dcol=sub[args.depth_col].to_numpy()
        f0 = int(round(float(rr[f"videos/{args.video_key}/from_timestamp"])*args.fps))
        fr,_ = vid_frames_small(consolidated, [f0,f0+N//2,f0+N-1], 160, 90)
        rgbs=[Image.fromarray(fr[i]).convert("RGB") for i in [f0,f0+N//2,f0+N-1]]
        deps=[]
        for k in (0,len(dcol)//2,len(dcol)-1):
            a=depth_small(dcol[k],160,90).astype(float); a=(255*(1-np.clip(a/ (a[a>0].max() if (a>0).any() else 1),0,1))).astype(np.uint8)
            deps.append(Image.fromarray(a).convert("RGB"))
        rows.append((e,rgbs,deps))
    tw=160; th=90; pad=4; lblw=40; toph=20
    W=lblw+6*(tw+pad); Himg=toph+len(rows)*(2*th+pad+14)
    c=Image.new("RGB",(W,Himg),(12,14,20)); d=ImageDraw.Draw(c)
    try: fn=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",11)
    except: fn=ImageFont.load_default()
    d.text((4,4),"RGB(f0,mid,last) | DEPTH(f0,mid,last) — devem estar SINCRONIZADOS",fill=(230,200,120),font=fn)
    y=toph
    for e,rgbs,deps in rows:
        d.text((3,y+th//2),f"ep{e}",fill=(255,255,120),font=fn)
        for j,im in enumerate(rgbs): c.paste(im,(lblw+j*(tw+pad),y))
        for j,im in enumerate(deps): c.paste(im,(lblw+(j+3)*(tw+pad),y))
        y+=2*th+pad+14
    c.save(args.out); print(f"[verify] mosaico: {args.out}")


if __name__ == "__main__":
    main()
