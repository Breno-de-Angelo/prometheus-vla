# Testa a origem do jitter da trajetória: ruído do flow-matching (não-seedado) vs seedado.
# Por frame do ep 214: prevê a 1ª ação seedando o ruído com o MESMO seed (remove o ruído,
# sobra só a variação por observação). Compara com o não-seedado (do manifest).
import sys, json, argparse
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "lerobot-ext"); sys.path.insert(0, "lerobot-ext/train")
import probe_saliency as PS
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--repo-id", required=True)
ap.add_argument("--root", required=True); ap.add_argument("--episode", type=int, default=214)
ap.add_argument("--out", default="/tmp/jitter_seeded.json")
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pol = PS.PI05Policy.from_pretrained(a.ckpt, strict=False).to(dev).eval()
pre, _ = PS.make_pre_post_processors(policy_cfg=pol.config, pretrained_path=a.ckpt)
ds = PS.LeRobotDataset(a.repo_id, root=a.root, episodes=[a.episode], video_backend="pyav")
st = ds.meta.stats
q01 = np.asarray(st["action"]["q01"], np.float32)[:7]; q99 = np.asarray(st["action"]["q99"], np.float32)[:7]
def un(n7): return (np.asarray(n7, np.float32)+1)/2*(q99-q01)+q01
seeded=[]
for i in range(len(ds)):
    b = PS.make_batch(ds[i], pol, dev)
    with torch.no_grad():
        torch.manual_seed(1234)  # MESMO ruído todo frame -> sobra só a observação
        c = pol.predict_action_chunk(pre(PS.clone(b)))[0,0].detach().float().cpu().numpy()[:8]
    seeded.append([round(float(x),4) for x in un(c[:7])])
json.dump({"seeded_pred_j": seeded}, open(a.out,"w"))
print("[ok]", a.out, len(seeded), "frames")
