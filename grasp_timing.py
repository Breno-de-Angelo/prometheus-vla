# Mede o adiantamento do fechamento da mão (modelo vs demo) em todos os eps de val,
# e se o braço JÁ tinha chegado (deslocamento GT entre t0 e t1). Seedado.
import sys, json, argparse
import numpy as np, torch
sys.path.insert(0,"."); sys.path.insert(0,"lerobot-ext"); sys.path.insert(0,"lerobot-ext/train")
import probe_saliency as PS
ap=argparse.ArgumentParser()
ap.add_argument("--ckpt",required=True); ap.add_argument("--repo-id",required=True); ap.add_argument("--root",required=True)
ap.add_argument("--eps",default="214-237"); ap.add_argument("--seed",type=int,default=1234)
ap.add_argument("--out",default="/tmp/grasp_timing.json")
a=ap.parse_args()
lo,hi=[int(x) for x in a.eps.split("-")]; EPS=list(range(lo,hi+1))
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
pol=PS.PI05Policy.from_pretrained(a.ckpt,strict=False).to(dev).eval()
pre,_=PS.make_pre_post_processors(policy_cfg=pol.config,pretrained_path=a.ckpt)
def cross(x,th=0.5):
    idx=np.where(np.asarray(x)>=th)[0]; return int(idx[0]) if len(idx) else -1
res=[]
for ep in EPS:
    ds=PS.LeRobotDataset(a.repo_id,root=a.root,episodes=[ep],video_backend="pyav")
    fps=int(getattr(ds.meta,"fps",30) or 30)
    acts=np.asarray(ds.hf_dataset.select_columns(["action"])["action"],dtype=np.float32)
    gt_sq=acts[:,7]; gt_j=acts[:,:7]
    t1=cross(gt_sq)
    if t1<0:  # ep sem pega -> pula a inferência
        res.append(dict(ep=ep,grasp=False)); print(f"ep {ep}: SEM pega (pula)"); continue
    pred_sq=[]
    for i in range(len(ds)):
        b=PS.make_batch(ds[i],pol,dev)
        with torch.no_grad():
            torch.manual_seed(a.seed)
            c=pol.predict_action_chunk(pre(PS.clone(b)))[0,0].detach().float().cpu().numpy()
        pred_sq.append(PS.pred_squeeze(c,"8dim"))
    pred_sq=np.array(pred_sq)
    t0=cross(pred_sq)
    off=(t1-t0) if (t0>=0) else None
    disp_max=disp_mean=None
    if t0>=0 and t1>t0:
        d=np.abs(gt_j[t1]-gt_j[t0]); disp_max=float(d.max()); disp_mean=float(d.mean())
    amp=float((gt_j.max(0)-gt_j.min(0)).max())
    res.append(dict(ep=ep,grasp=True,t0=t0,t1=t1,off_frames=off,
                    off_s=(None if off is None else round(off/fps,3)),
                    disp_max=None if disp_max is None else round(disp_max,4),
                    disp_mean=None if disp_mean is None else round(disp_mean,4),
                    amp_max=round(amp,4)))
    print(f"ep {ep}: t0={t0} t1={t1} off={off}f disp_max={disp_max}")
json.dump(res,open(a.out,"w"),indent=1)
# resumo
g=[r for r in res if r.get("grasp") and r.get("off_frames") is not None]
offs=[r["off_frames"] for r in g]; dmax=[r["disp_max"] for r in g if r["disp_max"] is not None]
print("\n=== RESUMO ===")
print(f"eps com pega: {len(g)}  | adianta em media {np.mean(offs):.1f} frames ({np.mean(offs)/30:.2f}s), mediana {np.median(offs):.0f}")
print(f"adianta (negativo=atrasa) min/max: {min(offs)}/{max(offs)} frames")
prem=[r for r in g if r['disp_max'] and r['disp_max']>0.2]
print(f"deslocamento do braço em t0 (o quanto ainda ia mexer): media_max={np.mean(dmax):.3f} rad")
print(f"eps com fechamento PREMATURO (braço ainda longe, disp_max>0.2rad): {len(prem)} -> {[r['ep'] for r in prem]}")
print(f"[ok] {a.out}")
