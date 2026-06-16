#!/usr/bin/env python3
"""Gravador NÃO-bloqueante da inferência REAL no robô — pra replay offline + probes.

O loop de controle só ENFILEIRA (numpy + dict); uma thread de fundo escreve em disco
(jpg/png + jsonl), então a gravação nunca derruba o FPS. Tudo best-effort: qualquer
falha é engolida e jamais interrompe a inferência.

Layout de <out_dir>:
  chunks.jsonl  - 1 linha por INFERÊNCIA: chunk previsto + observation.state + refs de imagem
  frames.jsonl  - 1 linha por AÇÃO executada (o "real" enviado ao robô)
  rgb/NNNNNN.jpg   depth/NNNNNN.png   attn/NNNNNN.jpg
  meta.json     - checkpoint, args, fps, modo (right8/armstate7)

chunks.jsonl é compatível com `offline_sim_host.py --replay-decisions`
('actions' = chunk previsto [chunk, dim]; 'state_raw' = observation.state).
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger("run_recorder")


def _tolist(x):
    """Tensor/array/list -> lista de floats (ou None)."""
    if x is None:
        return None
    if hasattr(x, "detach"):                       # torch.Tensor
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.astype(float).tolist()
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return x


class RunRecorder:
    """Grava decisões (1/inferência) e frames executados (1/ação) de forma assíncrona."""

    def __init__(self, out_dir, meta=None, queue_max=2048):
        self.dir = Path(out_dir)
        for sub in ("rgb", "depth", "attn"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        if meta is not None:
            try:
                (self.dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
            except Exception:
                pass
        self._q: queue.Queue = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self._dropped = 0
        self._n_dec = 0
        self._n_frm = 0
        self._ch = open(self.dir / "chunks.jsonl", "w")
        self._fr = open(self.dir / "frames.jsonl", "w")
        self._t = threading.Thread(target=self._writer, daemon=True)
        self._t.start()
        logger.info("[record] gravando em %s", self.dir)

    # ---------- chamadas no loop de controle: SÓ enfileiram (rápido, sem I/O) ----------
    def record_decision(self, idx, t, state_raw, chunk, action_mode="right14",
                        rgb=None, depth=None, attn=None, infer_ms=None, extra=None):
        rec = {"idx": int(idx), "t": float(t), "state_raw": _tolist(state_raw),
               "actions": _tolist(chunk), "action_mode": action_mode}
        if infer_ms is not None:
            rec["infer_ms"] = float(infer_ms)
        if extra:
            rec.update(extra)
        self._put(("dec", rec,
                   None if rgb is None else np.asarray(rgb).copy(),
                   None if depth is None else np.asarray(depth).copy(),
                   None if attn is None else np.asarray(attn).copy()))

    def record_frame(self, frame, t, action, grasp=None):
        rec = {"frame": int(frame), "t": float(t),
               "action": {k: round(float(v), 5) for k, v in (action or {}).items()}}
        if grasp is not None:
            rec["grasp"] = grasp
        self._put(("frm", rec, None, None, None))

    def _put(self, item):
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self._dropped += 1                      # nunca bloqueia o loop de controle

    # ---------- thread de fundo: serializa + escreve em disco ----------
    def _writer(self):
        try:
            import cv2
        except Exception as e:
            logger.warning("[record] cv2 indisponível, sem imagens: %s", e)
            cv2 = None
        while not (self._stop.is_set() and self._q.empty()):
            try:
                kind, rec, rgb, depth, attn = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if kind == "dec":
                    i = rec["idx"]
                    if cv2 is not None and rgb is not None:
                        img = np.asarray(rgb)
                        if img.ndim == 3 and img.shape[2] == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        if cv2.imwrite(str(self.dir / "rgb" / f"{i:06d}.jpg"), img,
                                       [cv2.IMWRITE_JPEG_QUALITY, 90]):
                            rec["rgb"] = f"rgb/{i:06d}.jpg"
                    if cv2 is not None and depth is not None:
                        d = np.asarray(depth)
                        if d.dtype != np.uint16:
                            d = np.clip(d, 0, 65535).astype(np.uint16)
                        if cv2.imwrite(str(self.dir / "depth" / f"{i:06d}.png"), d):
                            rec["depth"] = f"depth/{i:06d}.png"
                    if cv2 is not None and attn is not None:
                        a = np.asarray(attn)
                        if a.dtype != np.uint8:
                            a = np.clip(a * 255.0, 0, 255).astype(np.uint8) if float(a.max() or 0) <= 1.0 else a.astype(np.uint8)
                        if cv2.imwrite(str(self.dir / "attn" / f"{i:06d}.jpg"), a,
                                       [cv2.IMWRITE_JPEG_QUALITY, 90]):
                            rec["attn"] = f"attn/{i:06d}.jpg"
                    self._ch.write(json.dumps(rec) + "\n")
                    self._n_dec += 1
                else:
                    self._fr.write(json.dumps(rec) + "\n")
                    self._n_frm += 1
            except Exception as e:
                logger.debug("[record] write falhou: %s", e)

    def close(self):
        self._stop.set()
        try:
            self._t.join(timeout=8.0)
        except Exception:
            pass
        for f in (self._ch, self._fr):
            try:
                f.flush()
                f.close()
            except Exception:
                pass
        logger.info("[record] fim: %d decisões, %d frames, %d drops -> %s",
                    self._n_dec, self._n_frm, self._dropped, self.dir)
