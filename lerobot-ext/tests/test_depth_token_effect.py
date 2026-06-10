"""FASE 1 (auditoria): o token de depth injetado influencia a saída do modelo?

Roda predict_action_chunk duas vezes com a MESMA observação e o MESMO ruído do
flow matching: (a) depth real do dataset, (b) depth zerado (nuvem vazia → token
constante do PointNet). Se as máscaras do prefixo estivessem marcando o token
como padding, a atenção o ignoraria e as duas saídas seriam idênticas.

Também imprime os valores reais de pad_mask/att_mask por segmento do prefixo
(imagem / linguagem / depth) capturados em runtime.

O checkpoint do run depth fica na Atena; aqui usamos o modelo com pesos
ALEATÓRIOS na variante pequena (gemma_300m), o que é suficiente para o
diagnóstico de máscara: com pesos aleatórios, tokens de depth diferentes só
produzem ações diferentes se a atenção realmente enxerga o token.

Uso (laptop, env g1, a partir da raiz do repo — pesos aleatórios):
    conda run -n g1 python lerobot-ext/tests/test_depth_token_effect.py

Uso (Atena, env ms3, GPU 0 — checkpoint REAL do run depth):
    CUDA_VISIBLE_DEVICES=0 python lerobot-ext/tests/test_depth_token_effect.py \
        --checkpoint train_output/cup_pi05_right14_depth_lf/checkpoints/005000/pretrained_model

No modo --checkpoint, os intrínsecos usados são os MESMOS do treino (default
fx=600/cx=320 do injector, sabidamente nominais de 640x480) — de propósito:
o teste mede se o modelo treinado usa o token, com a geometria que ele viu.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "lerobot-ext"))  # para `train.pi05_depth_injector`

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from train.pi05_depth_injector import inject_pi05_depth

RGB_KEY = "observation.images.head_camera"
DEPTH_KEY = "observation.images.head_camera_depth"
DATASET_ROOT = REPO / "lerobot-ext/datasets/G1_Dex3_right14_dataset/20260608_205432"

# Intrínsecos nominais D435 @848x480 (provisórios — ver FASE 3 da auditoria)
INTRINSICS = {"fx": 425.0, "fy": 425.0, "cx": 424.0, "cy": 240.0}


def build_policy():
    # gemma_300m como VLM não fecha as dims do projector de imagem (2048 vs 1024);
    # usamos o gemma_2b real, construído direto em bf16 para caber na RAM do laptop.
    cfg = PI05Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        chunk_size=10,
        n_action_steps=10,
        max_state_dim=32,
        max_action_dim=32,
        input_features={
            RGB_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 848)),
            DEPTH_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(1, 480, 848)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
    )
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    try:
        policy = PI05Policy(cfg)
    finally:
        torch.set_default_dtype(torch.float32)  # PointNet do injector fica em fp32
    return policy


def load_real_depth():
    """Um frame real de depth (mm) do dataset right14; fallback sintético se indisponível."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id="local/right14", root=str(DATASET_ROOT), video_backend="pyav")
        depth = ds[100][DEPTH_KEY].float()  # [1, 480, 848] em milímetros
        src = f"dataset {DATASET_ROOT.name} (frame 100)"
    except Exception as e:  # noqa: BLE001
        print(f"[aviso] dataset indisponível ({e}); usando depth sintético 600±100mm")
        torch.manual_seed(1)
        depth = (600.0 + 100.0 * torch.randn(1, 480, 848)).clamp(100.0, 2000.0)
        src = "sintético"
    return depth.unsqueeze(0) if depth.dim() == 3 else depth, src


def _sanitize_checkpoint_config(ckpt_dir: str) -> str:
    """Remove do config.json campos que a versão atual do PI05Config não aceita
    (skew de versão do lerobot, ex.: tokenizer_name). Copia para um dir temp com
    symlink para o model.safetensors — o checkpoint original fica intocado."""
    import dataclasses
    import json
    import os
    import tempfile

    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    cfg_path = Path(ckpt_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    valid = {f.name for f in dataclasses.fields(PI05Config)} | {"type"}
    extra = set(cfg) - valid
    if not extra:
        return ckpt_dir
    print(f"[sanitize] removendo campos desconhecidos do config.json: {sorted(extra)}")
    tmp = Path(tempfile.mkdtemp(prefix="ckpt_sane_"))
    (tmp / "config.json").write_text(json.dumps({k: v for k, v in cfg.items() if k in valid}))
    for f in os.listdir(ckpt_dir):
        if f != "config.json":
            os.symlink(str(Path(ckpt_dir).resolve() / f), tmp / f)
    return str(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="dir pretrained_model do run depth (Atena); sem ele, pesos aleatórios")
    args = ap.parse_args()

    if args.checkpoint:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = _sanitize_checkpoint_config(args.checkpoint)
        policy = PI05Policy.from_pretrained(ckpt, strict=False).to(device).eval()
        # intrínsecos da ERA DO TREINO deste checkpoint (o antigo default 640x480,
        # sabidamente errado p/ 848x480 — corrigido na FASE 3, mas o teste mede o
        # modelo como ele foi treinado) + pesos treinados do PointNet
        train_era_intrinsics = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}
        inject_pi05_depth(policy, device, camera_intrinsics=train_era_intrinsics,
                          load_injected_from=args.checkpoint, depth_scale=0.001)
        print(f"[modo] checkpoint real: {args.checkpoint} (device={device})")
    else:
        device = "cpu"
        policy = build_policy().to(device).eval()
        inject_pi05_depth(policy, device, camera_intrinsics=INTRINSICS, depth_scale=0.001)
        # FASE 5 zero-inicializa a fc2 do PointNet (token = 0 até treinar) — o que
        # tornaria este teste de máscara um falso negativo no modo random. Para
        # medir se a ATENÇÃO enxerga o token, re-randomizamos só a fc2 aqui.
        torch.manual_seed(3)
        torch.nn.init.normal_(policy.pointnet.fc2.weight, std=0.02)
        torch.nn.init.normal_(policy.pointnet.fc2.bias, std=0.02)
        print("[modo] pesos aleatórios (laptop; fc2 re-randomizada p/ driblar o zero-init)")

    # Espião nas máscaras finais do prefixo (depois do patch do injector)
    rec = {}
    patched_embed_prefix = policy.model.embed_prefix

    def spy(images, img_masks, tokens, masks):
        embs, pad, att = patched_embed_prefix(images, img_masks, tokens, masks)
        rec["pad"], rec["att"], rec["len"] = pad[0].tolist(), att[0].tolist(), embs.shape[1]
        rec["n_lang"] = tokens.shape[1]
        return embs, pad, att

    policy.model.embed_prefix = spy

    dtype = next(policy.model.parameters()).dtype
    torch.manual_seed(2)
    rgb = torch.rand(1, 3, 480, 848, dtype=dtype, device=device)
    n_lang = 48
    batch_base = {
        RGB_KEY: rgb,
        OBS_LANGUAGE_TOKENS: torch.randint(1, 100, (1, n_lang), device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, n_lang, dtype=torch.bool, device=device),
    }
    noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim,
                        dtype=dtype, device=device)

    depth_real, src = load_real_depth()
    depth_real = depth_real.to(device)
    print(f"[depth real] {src}: mediana={depth_real[depth_real > 0].median():.0f}mm")

    # Seed antes de CADA chamada: o subsample da nuvem (torch.randperm em
    # depth_to_pointcloud) é estocástico e mudaria o token entre chamadas.
    with torch.no_grad():
        torch.manual_seed(123)
        a1 = policy.predict_action_chunk({**batch_base, DEPTH_KEY: depth_real.clone()},
                                         noise=noise.clone(), num_steps=2)
        torch.manual_seed(123)
        a1b = policy.predict_action_chunk({**batch_base, DEPTH_KEY: depth_real.clone()},
                                          noise=noise.clone(), num_steps=2)
        torch.manual_seed(123)
        a2 = policy.predict_action_chunk({**batch_base, DEPTH_KEY: torch.zeros_like(depth_real)},
                                         noise=noise.clone(), num_steps=2)

    # ---- Relatório das máscaras (último token = depth) ----
    n_total, n_lang = rec["len"], rec["n_lang"]
    n_img = n_total - n_lang - 1
    pad, att = rec["pad"], rec["att"]
    seg = lambda v, a, b: sorted(set(v[a:b]))  # noqa: E731
    print(f"\n[máscaras do prefixo] total={n_total} tokens (img={n_img}, lang={n_lang}, depth=1)")
    print(f"  imagem : pad={seg(pad, 0, n_img)} att={seg(att, 0, n_img)}")
    print(f"  língua : pad={seg(pad, n_img, n_img + n_lang)} att={seg(att, n_img, n_img + n_lang)}")
    print(f"  depth  : pad=[{pad[-1]}] att=[{att[-1]}]")

    # ---- Diffs ----
    det = (a1 - a1b).abs().max().item()
    maxdiff = (a1 - a2).abs().max().item()
    mse = ((a1 - a2) ** 2).mean().item()
    print(f"\n[determinismo] mesmo input 2x: max abs diff = {det:.3e} (esperado 0)")
    print(f"[efeito depth] real vs zerado : max abs diff = {maxdiff:.3e}, MSE = {mse:.3e}")

    assert det == 0.0, f"forward não-determinístico (diff={det}); teste inválido"
    if maxdiff < 1e-6:
        print("\n❌ FALHA: token de depth IGNORADO pela atenção — verificar pad/att masks acima.")
        sys.exit(1)
    print("\n✅ OK: o token de depth influencia o chunk de ação (pad=True, att=0 — "
          "mesmo bloco bidirecional dos tokens de imagem).")


if __name__ == "__main__":
    main()
