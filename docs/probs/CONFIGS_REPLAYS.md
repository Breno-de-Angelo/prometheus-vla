# Configs dos replays interativos (ep 214)

**Escrito:** 2026-06-14 21:40 (-03)
**O que é:** registro das configurações usadas nos dois replays interativos de 1 episódio (não-seedado × seedado), pra reproduzir/comparar. Os HTMLs ficam em `docs/probs/`.

---

## Comum aos dois

| item | valor |
|---|---|
| **Modelo** | valfix (`y32omum0`) — checkpoint `best` @ **step 7000** (val_action_mse 0.0855, val_loss 0.105) |
| **Caminho** | `train_output/cup_pi05_right8_armstate7_valfix_lf/checkpoints/best/pretrained_model` (TRX50) |
| **Dataset** | `lewislf/G1_Dex3_pick_white_cup_right8_1squeeze_armstate7` |
| **Episódio** | **214** (val) — 153 frames @ 30 fps, pega clara (squeeze 0→1) |
| **Ação** | 8 dims (7 braço rad + 1 squeeze) · `mode=8dim` |
| **Inferência** | `predict_action_chunk` por frame · `chunk_size=50` · `num_inference_steps=10` (denoising) |
| **Atenção** | real, via `attn_recorder` (action expert → 256 tokens de imagem → 16×16), recortada à faixa do `resize_with_pad` |
| **GPU** | TRX50 GPU1 (`CUDA_VISIBLE_DEVICES=1`), env `ms3`, `OMP_NUM_THREADS=8` |
| **Gravador** | `gen_episode_replay.py` · player `build_replay_html.py` |

> **Nota sobre o replay × deploy:** aqui o chunk é re-previsto em **cada frame** (super-amostra, só pra visualizar). No deploy real **não** é assim: RTC re-planeja a cada ~20-30 passos (`--rtc-execution-horizon=20`, `--rtc-refill=30`) + a rampa do host interpola a 250 Hz e clipa velocidade.

---

## Replay 1 — NÃO-seedado  ·  `replay_ep214_valfix_NAOseedado.html`

- **Ruído do flow-matching:** **fresco a cada frame** (`--seed` não setado). Cada previsão sorteia um ruído independente.
- **Efeito:** a trajetória "calculada" (verde) **chacoalha** em volta do GT — jitter de alta frequência.
- **Jitter medido** (roughness = |2ª diferença| média): **0.0436 rad** (todas as juntas), 0.0510 na shoulder_pitch.
- **Comando:**
  ```bash
  python gen_episode_replay.py --ckpt <best> --repo-id <...armstate7> --root <...armstate7> \
    --episode 214 --mode 8dim --label valfix --outdir /tmp/replay_valfix
  ```

## Replay 2 — SEEDADO  ·  `replay_ep214_valfix_SEEDADO.html`

- **Ruído do flow-matching:** **fixo** (`--seed 1234`, mesmo ruído todo frame). Sobra só a variação por observação.
- **Efeito:** a verde fica **~10× mais lisa**, quase colada na tendência do GT.
- **Jitter medido:** **0.0047 rad** — **89% menor** que o não-seedado (GT real = 0.0008).
- **Erro de rastreio** quase igual (0.0200 vs 0.0229 do não-seedado) → seedar **não custa precisão**.
- **Comando:** idem acima **+** `--seed 1234` e `--label valfix_seedado --outdir /tmp/replay_valfix_seeded`.

---

## Leitura
O jitter da verde é **~89% ruído do flow-matching** (não-seedado), não erro do modelo. No robô real o RTC + a rampa de 250 Hz já filtram isso. A variância "honesta" que sobra (seedado 0.0047 vs GT 0.0008) é o alvo da **EMA (run 3)** + mais dados.
