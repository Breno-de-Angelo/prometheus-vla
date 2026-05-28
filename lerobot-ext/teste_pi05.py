#!/usr/bin/env python3

from transformers import (
    PaliGemmaForConditionalGeneration,
)
from transformers.models.auto import CONFIG_MAPPING

def print_children(module, prefix=""):
    for name, child in module.named_children():
        print(f"{prefix}.{name} -> {type(child).__name__}")

def main():
    cfg = CONFIG_MAPPING["paligemma"]()
    model = PaliGemmaForConditionalGeneration(cfg)

    print("\nSubmódulos de model:\n")

    print_children(model.model)

    print("\nAtributos relacionados a visão:\n")

    for attr in dir(model.model):
        if (
            "vision" in attr.lower()
            or "siglip" in attr.lower()
            or "image" in attr.lower()
        ):
            print(attr)

if __name__ == "__main__":
    main()