#!/usr/bin/env python3
"""
Viewer de episódios do dataset LeRobot — app Tkinter nativo.

Uso:
    python lerobot-ext/tools/episode_viewer.py <pasta_sessao>
    python lerobot-ext/tools/episode_viewer.py datasets/G1_Dex3_depth_tactil_dataset/20260603_110712

Controles:
    SPACE       — pausa/resume global
    ← →         — seek ±1s em todos
    R           — reinicia todos
    clique num thumbnail → abre popup com o episódio em tamanho real
"""

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


# ── config ──────────────────────────────────────────────
THUMB_W    = 320
THUMB_H    = 240
MAX_COLS   = 4
FPS        = 30
FRAME_MS   = int(1000 / FPS)


# ── helpers ─────────────────────────────────────────────

def load_meta(session_dir: Path) -> pd.DataFrame:
    files = sorted((session_dir / "meta" / "episodes" / "chunk-000").glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("episode_index").reset_index(drop=True)


def find_video(session_dir: Path, chunk: int, fidx: int) -> Path:
    cam = sorted((session_dir / "videos").iterdir())[0]
    return cam / f"chunk-{chunk:03d}" / f"file-{fidx:03d}.mp4"


def frame_to_photo(frame: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


# ── EpisodePlayer ────────────────────────────────────────

class EpisodePlayer:
    """Gerencia a posição de reprodução de um episódio e seu VideoCapture."""

    def __init__(self, ep_idx: int, file_idx, t_start: float, t_end: float, cap):
        self.ep_idx  = ep_idx
        self.file_idx = file_idx
        self.t_start = t_start
        self.t_end   = t_end
        self.cap     = cap
        self.pos     = 0.0   # 0.0 – 1.0

    @property
    def duration(self) -> float:
        return max(0.001, self.t_end - self.t_start)

    def current_time(self) -> float:
        return self.pos * self.duration

    def get_frame(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        t = self.t_start + self.pos * self.duration
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = self.cap.read()
        return frame if ok else None

    def advance(self, dt: float):
        self.pos = min(1.0, self.pos + dt / self.duration)
        if self.pos >= 1.0:
            self.pos = 0.0   # loop

    def seek(self, dt: float):
        self.pos = max(0.0, min(1.0, self.pos + dt / self.duration))

    def reset(self):
        self.pos = 0.0


# ── App principal ────────────────────────────────────────

class EpisodeViewerApp(tk.Tk):

    def __init__(self, session_dir: Path):
        super().__init__()
        self.title(f"Episode Viewer — {session_dir.name}")
        self.configure(bg="#1e1e1e")

        print("Carregando metadados...")
        eps  = load_meta(session_dir)
        n    = len(eps)
        print(f"  {n} episódios")

        # abre caps
        caps: dict[int, cv2.VideoCapture | None] = {}
        for _, row in eps.iterrows():
            fi = row["videos/observation.images.head_camera/file_index"]
            ci = row["videos/observation.images.head_camera/chunk_index"]
            if pd.isna(fi) or pd.isna(ci):
                continue
            fi, ci = int(fi), int(ci)
            if fi not in caps:
                vpath = find_video(session_dir, ci, fi)
                caps[fi] = cv2.VideoCapture(str(vpath)) if vpath.exists() else None
                if caps[fi] is None:
                    print(f"[WARN] sem vídeo: {vpath}")

        self.players: list[EpisodePlayer] = []
        for _, row in eps.iterrows():
            fi_raw = row["videos/observation.images.head_camera/file_index"]
            fi     = int(fi_raw) if not pd.isna(fi_raw) else None
            ts     = row["videos/observation.images.head_camera/from_timestamp"]
            te     = row["videos/observation.images.head_camera/to_timestamp"]
            self.players.append(EpisodePlayer(
                ep_idx   = int(row["episode_index"]),
                file_idx = fi,
                t_start  = float(ts) if not pd.isna(ts) else 0.0,
                t_end    = float(te) if not pd.isna(te) else 0.0,
                cap      = caps.get(fi) if fi is not None else None,
            ))

        self.paused   = False
        self.n_cols   = min(MAX_COLS, n)
        self.n_rows   = (n + self.n_cols - 1) // self.n_cols
        self._photos  = {}   # mantém referências para não serem garbage-collected
        self._lock    = threading.Lock()

        self._build_ui(n)
        self.bind("<space>",  lambda _: self._toggle_pause())
        self.bind("<Left>",   lambda _: self._seek_all(-1.0))
        self.bind("<Right>",  lambda _: self._seek_all(+1.0))
        self.bind("r",        lambda _: self._reset_all())
        self.bind("R",        lambda _: self._reset_all())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._running = True
        self._schedule_tick()
        print("Viewer aberto. SPACE=pausa | ←→=seek | R=reiniciar | clique=zoom")

    # ── UI ──────────────────────────────────────────────

    def _build_ui(self, n: int):
        # barra superior de controles
        ctrl = tk.Frame(self, bg="#2d2d2d")
        ctrl.pack(fill="x", padx=6, pady=4)

        btn_kw = dict(bg="#444", fg="white", relief="flat", padx=8, pady=3,
                      activebackground="#666", font=("Helvetica", 11))
        tk.Button(ctrl, text="⏮ Reiniciar", command=self._reset_all,  **btn_kw).pack(side="left", padx=2)
        tk.Button(ctrl, text="◀ -1s",       command=lambda: self._seek_all(-1), **btn_kw).pack(side="left", padx=2)
        self._btn_pause = tk.Button(ctrl, text="⏸ Pausar", command=self._toggle_pause, **btn_kw)
        self._btn_pause.pack(side="left", padx=2)
        tk.Button(ctrl, text="▶ +1s",       command=lambda: self._seek_all(+1),  **btn_kw).pack(side="left", padx=2)

        self._lbl_status = tk.Label(ctrl, text="", bg="#2d2d2d", fg="#aaa",
                                    font=("Helvetica", 10))
        self._lbl_status.pack(side="right", padx=8)

        # grid de thumbnails dentro de canvas com scrollbar
        outer = tk.Frame(self, bg="#1e1e1e")
        outer.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(outer, bg="#1e1e1e", highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._grid_frame = tk.Frame(self._canvas, bg="#1e1e1e")
        self._canvas_window = self._canvas.create_window((0, 0), window=self._grid_frame, anchor="nw")

        self._grid_frame.bind("<Configure>", self._on_frame_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<MouseWheel>", lambda e: self._canvas.yview_scroll(-1*(e.delta//120), "units"))
        self._canvas.bind("<Button-4>",   lambda e: self._canvas.yview_scroll(-1, "units"))
        self._canvas.bind("<Button-5>",   lambda e: self._canvas.yview_scroll( 1, "units"))

        # células
        self._cells: list[dict] = []
        for i, p in enumerate(self.players):
            r, c = divmod(i, self.n_cols)
            cell = tk.Frame(self._grid_frame, bg="#1e1e1e", padx=2, pady=2)
            cell.grid(row=r, column=c, sticky="nsew")

            lbl_img = tk.Label(cell, bg="#111", cursor="hand2",
                               width=THUMB_W, height=THUMB_H)
            lbl_img.pack()
            lbl_img.bind("<Button-1>", lambda e, idx=i: self._open_zoom(idx))

            lbl_txt = tk.Label(cell, text=f"EP {p.ep_idx}", bg="#1e1e1e",
                               fg="#ccc", font=("Helvetica", 9))
            lbl_txt.pack()

            self._cells.append({"img": lbl_img, "txt": lbl_txt})

        for c in range(self.n_cols):
            self._grid_frame.columnconfigure(c, weight=1)

    def _on_frame_configure(self, _):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self._canvas.itemconfig(self._canvas_window, width=e.width)

    # ── controles ───────────────────────────────────────

    def _toggle_pause(self):
        self.paused = not self.paused
        self._btn_pause.config(text="▶ Continuar" if self.paused else "⏸ Pausar")

    def _seek_all(self, dt: float):
        for p in self.players:
            p.seek(dt)

    def _reset_all(self):
        for p in self.players:
            p.reset()

    # ── loop de frames ───────────────────────────────────

    def _schedule_tick(self):
        if self._running:
            self.after(FRAME_MS, self._tick)

    def _tick(self):
        if not self._running:
            return

        playing = sum(1 for p in self.players if p.cap is not None)
        self._lbl_status.config(
            text=f"{'⏸' if self.paused else '▶'}  {playing}/{len(self.players)} com vídeo"
        )

        for i, p in enumerate(self.players):
            frame = p.get_frame()
            if frame is not None:
                photo = frame_to_photo(frame, THUMB_W, THUMB_H)
                self._photos[i] = photo
                self._cells[i]["img"].config(image=photo)

            dur = p.duration
            t   = p.current_time()
            self._cells[i]["txt"].config(
                text=f"EP {p.ep_idx}  {t:.1f}s / {dur:.1f}s"
                     + ("  [sem vídeo]" if p.cap is None else "")
            )

            if not self.paused and p.cap is not None:
                p.advance(FRAME_MS / 1000.0)

        self._schedule_tick()

    # ── zoom popup ───────────────────────────────────────

    def _open_zoom(self, idx: int):
        p = self.players[idx]
        if p.cap is None:
            return

        win = tk.Toplevel(self)
        win.title(f"EP {p.ep_idx}")
        win.configure(bg="#111")

        lbl = tk.Label(win, bg="#111")
        lbl.pack(fill="both", expand=True)

        info = tk.Label(win, text="", bg="#111", fg="#0f0", font=("Courier", 10))
        info.pack()

        zoom_photo = {}
        zoom_running = [True]
        zoom_paused  = [False]

        def zoom_tick():
            if not zoom_running[0]:
                return
            frame = p.get_frame()
            if frame is not None:
                h = win.winfo_height() - 30 or 480
                w = win.winfo_width()  or 640
                photo = frame_to_photo(frame, w, h)
                zoom_photo["p"] = photo
                lbl.config(image=photo)
            info.config(text=f"{'⏸' if zoom_paused[0] else '▶'}  EP {p.ep_idx}  {p.current_time():.1f}s / {p.duration:.1f}s  [SPACE=pausa | ←→=seek]")
            if not zoom_paused[0]:
                p.advance(FRAME_MS / 1000.0)
            win.after(FRAME_MS, zoom_tick)

        def on_close():
            zoom_running[0] = False
            win.destroy()

        win.bind("<space>",  lambda _: zoom_paused.__setitem__(0, not zoom_paused[0]))
        win.bind("<Left>",   lambda _: p.seek(-1.0))
        win.bind("<Right>",  lambda _: p.seek(+1.0))
        win.bind("<Escape>", lambda _: on_close())
        win.protocol("WM_DELETE_WINDOW", on_close)
        win.geometry("800x600")
        zoom_tick()

    # ── shutdown ─────────────────────────────────────────

    def _on_close(self):
        self._running = False
        self.destroy()


# ── entry point ──────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python episode_viewer.py <pasta_sessao>")
        sys.exit(1)

    app = EpisodeViewerApp(Path(sys.argv[1]))
    app.mainloop()
