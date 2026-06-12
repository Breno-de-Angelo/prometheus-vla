#!/usr/bin/env python
"""
Fine-tune Entry Point — Carrega checkpoint existente e treina com novo dataset.

USO:
    python finetune_from_checkpoint.py \
        --checkpoint_path=train_output/pick_up_the_cup_depth-260610/best_val_checkpoint \
        --config_path=configs/finetune_new_task.yaml

O YAML de fine-tune precisa ter pelo menos:
    dataset:
      repo_id: "seu/novo-dataset"
    training:
      num_epochs: 20        # menos épocas para fine-tune
      lr: 1e-5              # lr menor que o treinamento original
"""

import sys
import os
import shutil
import yaml
import argparse

# Garante que o diretório atual está no path (para importar 'policies')
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def display_help():
    print("\n" + "=" * 70)
    print("FINE-TUNE A PARTIR DE CHECKPOINT (actdepth / pi05depth / genérico)")
    print("=" * 70)
    print("USO:")
    print("  python finetune_from_checkpoint.py \\")
    print("      --checkpoint_path=<PASTA_DO_CHECKPOINT> \\")
    print("      --config_path=<YAML_DO_FINE_TUNE>\n")
    print("ARGUMENTOS:")
    print("  --checkpoint_path   Pasta que contém pretrained_model/ e training_state/")
    print("                      Ex: train_output/meu_modelo/best_val_checkpoint")
    print("  --config_path       YAML com dataset novo e hiperparâmetros de fine-tune")
    print("  --freeze_encoder    (opcional) Congela o encoder visual — treina só a cabeça")
    print("  --output_dir        (opcional) Onde salvar o novo treino (default: train_output/finetune_<timestamp>)")
    print("=" * 70 + "\n")


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_policy_type(config_path: str) -> str:
    cfg = load_yaml(config_path)
    return cfg.get("policy", {}).get("type", "")


def resolve_pretrained_model_path(checkpoint_path: str) -> str:
    """
    Aceita tanto a pasta raiz do checkpoint quanto a subpasta pretrained_model/.
    Retorna sempre o caminho para pretrained_model/.
    """
    pretrained = os.path.join(checkpoint_path, "pretrained_model")
    if os.path.isdir(pretrained):
        return pretrained
    # talvez o usuário já apontou direto para pretrained_model/
    if os.path.isfile(os.path.join(checkpoint_path, "model.safetensors")):
        return checkpoint_path
    raise FileNotFoundError(
        f"Não encontrei pretrained_model/ dentro de '{checkpoint_path}'.\n"
        f"Estrutura esperada:\n"
        f"  {checkpoint_path}/\n"
        f"    pretrained_model/\n"
        f"      model.safetensors\n"
        f"      config.json\n"
        f"      train_config.json\n"
    )


def merge_configs(base_train_config: dict, finetune_yaml: dict) -> dict:
    """
    Mescla o train_config.json original com as overrides do YAML de fine-tune.
    O YAML de fine-tune tem prioridade em tudo que ele definir.
    """
    import copy
    merged = copy.deepcopy(base_train_config)

    def deep_merge(base: dict, override: dict):
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                deep_merge(base[k], v)
            else:
                base[k] = v

    deep_merge(merged, finetune_yaml)
    return merged


def patch_config_for_finetune(merged: dict, pretrained_model_path: str,
                               output_dir: str, freeze_encoder: bool) -> dict:
    """Injeta campos obrigatórios para o fine-tune funcionar corretamente."""
    merged.setdefault("training", {})
    merged.setdefault("policy", {})

    # Aponta para o modelo pré-treinado
    merged["policy"]["pretrained_path"] = pretrained_model_path

    # Diretório de saída do fine-tune
    merged["training"]["output_dir"] = output_dir

    # Flag de fine-tune para os motores que suportam
    merged["training"]["finetune_from_pretrained"] = True

    if freeze_encoder:
        merged["training"]["freeze_encoder"] = True
        print("[INFO]: Encoder congelado — apenas a cabeça de ação será treinada.")

    return merged


def save_merged_config(merged: dict, output_dir: str) -> str:
    """Salva o config mesclado como YAML temporário para passar ao motor de treino."""
    os.makedirs(output_dir, exist_ok=True)
    tmp_config_path = os.path.join(output_dir, "finetune_merged_config.yaml")
    with open(tmp_config_path, "w") as f:
        yaml.dump(merged, f, allow_unicode=True, default_flow_style=False)
    print(f"[INFO]: Config mesclado salvo em: {tmp_config_path}")
    return tmp_config_path


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    # ── Parse de argumentos ──
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--config_path",     type=str, default=None)
    parser.add_argument("--output_dir",      type=str, default=None)
    parser.add_argument("--freeze_encoder",  action="store_true")
    parser.add_argument("-h", "--help",      action="store_true")
    args, unknown_args = parser.parse_known_args()

    if args.help:
        display_help()
        return 0

    # ── Validações básicas ──
    if not args.checkpoint_path:
        print("[ERRO]: --checkpoint_path é obrigatório. Use -h para ajuda.")
        return 1
    if not args.config_path:
        print("[ERRO]: --config_path é obrigatório. Use -h para ajuda.")
        return 1
    if not os.path.isdir(args.checkpoint_path):
        print(f"[ERRO]: Checkpoint não encontrado: {args.checkpoint_path}")
        return 1
    if not os.path.isfile(args.config_path):
        print(f"[ERRO]: Config YAML não encontrado: {args.config_path}")
        return 1

    # ── Resolve o caminho do modelo pré-treinado ──
    try:
        pretrained_model_path = resolve_pretrained_model_path(args.checkpoint_path)
    except FileNotFoundError as e:
        print(f"[ERRO]: {e}")
        return 1

    print(f"[INFO]: Modelo base: {pretrained_model_path}")

    # ── Carrega o train_config.json original do checkpoint ──
    original_train_config_path = os.path.join(pretrained_model_path, "train_config.json")
    if os.path.isfile(original_train_config_path):
        import json
        with open(original_train_config_path) as f:
            base_config = json.load(f)
        print("[INFO]: train_config.json original carregado do checkpoint.")
    else:
        print("[AVISO]: train_config.json não encontrado no checkpoint — usando config do YAML como base.")
        base_config = {}

    # ── Carrega o YAML de fine-tune ──
    finetune_yaml = load_yaml(args.config_path)

    # ── Mescla configs (YAML de fine-tune tem prioridade) ──
    merged = merge_configs(base_config, finetune_yaml)

    # ── Define output_dir ──
    if args.output_dir:
        output_dir = args.output_dir
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        task_name = finetune_yaml.get("task_name", "finetune")
        output_dir = os.path.join("train_output", f"{task_name}-ft-{timestamp}")

    print(f"[INFO]: Saída do fine-tune: {output_dir}")

    # ── Injeta campos de fine-tune no config mesclado ──
    merged = patch_config_for_finetune(
        merged,
        pretrained_model_path=pretrained_model_path,
        output_dir=output_dir,
        freeze_encoder=args.freeze_encoder,
    )

    # ── Salva config mesclado como YAML temporário ──
    tmp_config_path = save_merged_config(merged, output_dir)

    # ── Detecta o tipo de política ──
    policy_type = merged.get("policy", {}).get("type", "")
    print(f"[INFO]: policy.type = '{policy_type}'")

    # ── Registra todas as políticas ──
    try:
        import policies
    except ImportError as e:
        print(f"[ERRO]: Falha ao registrar políticas: {e}")
        return 1

    # ── Seleciona o motor de treino ──
    if policy_type == "actdepth":
        from policies.act_depth.run_train import main as run_train_main
        print("[INFO]: Usando motor ACT-D para fine-tune...")

    elif policy_type == "pi05depth":
        from policies.pi0_depth.run_train import main as run_train_main
        print("[INFO]: Usando motor PI05-Depth para fine-tune...")

    else:
        print(f"[AVISO]: Tipo '{policy_type}' sem motor dedicado — usando motor genérico LeRobot.")
        from lerobot.scripts.lerobot_train import main as run_train_main

    # ── Injeta o config temporário como argumento do motor ──
    sys.argv = [sys.argv[0], f"--config_path={tmp_config_path}"] + unknown_args

    print("\n" + "=" * 60)
    print("INICIANDO FINE-TUNE")
    print(f"  Modelo base : {pretrained_model_path}")
    print(f"  Novo dataset: {merged.get('dataset', {}).get('repo_id', '(não definido)')}")
    print(f"  Saída       : {output_dir}")
    print("=" * 60 + "\n")

    return run_train_main()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[SISTEMA]: Fine-tune encerrado pelo usuário.")
        sys.exit(0)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
