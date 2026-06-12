#!/usr/bin/env python3
"""
playback_log.py — Reproduz o log de inferência em tempo real com animação matplotlib.

Layout (cabe numa tela 1080p ou maior):
  ┌─────────────────────────────────────────┐
  │  Heatmap geral (todas as 28 juntas)     │  ← faixa fina no topo
  ├──────────────┬──────────────────────────┤
  │ Braço Esq   │  Braço Dir               │
  ├──────────────┼──────────────────────────┤
  │ Mão Esq     │  Mão Dir                 │
  └──────────────┴──────────────────────────┘
  Status: step | % | velocidade | controles

Controles:
  [SPACE]    Pausa / Resume
  [← →]      Recua / Avança 50 steps
  [1-5]      Velocidade: 0.25× 0.5× 1× 2× 4×
  [Q / ESC]  Sai

Uso:
  python playback_log.py <log.txt> [OPÇÕES]

Opções:
  --fps=<INT>       FPS da animação (padrão: 30)
  --speed=<FLOAT>   Velocidade inicial (padrão: 1.0)
  --window=<INT>    Steps visíveis na janela deslizante (padrão: 120)
  --size=<W>x<H>    Tamanho da figura em polegadas (padrão: 15x8)
  --joints=<lista>  Índices de juntas separados por vírgula (padrão: todas)
  -h / --help       Mostra esta mensagem

Exemplos:
  python playback_log.py logs/log_20260608_192117.txt
  python playback_log.py logs/log_20260608_192117.txt --speed=2 --size=13x7
  python playback_log.py logs/log_20260608_192117.txt --window=60 --fps=60
"""

import sys
import os
import re
import time
import threading
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D


# ─────────────────────────────────────────────────────────────────────────────
JOINT_GROUPS = [
    ("Braço Esq",  list(range(0,  7)),  "#4fc3f7"),
    ("Braço Dir",  list(range(7,  14)), "#81c784"),
    ("Mão Esq",    list(range(14, 21)), "#ffb74d"),
    ("Mão Dir",    list(range(21, 28)), "#ce93d8"),
]
SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
SPEED_LABELS = ["0.25x", "0.5x", "1x", "2x", "4x"]


# ─────────────────────────────────────────────────────────────────────────────
def load_log(path):
    meta = {}
    joint_names = []
    executed, exec_steps, exec_ts = [], [], []
    network, net_obs = defaultdict(list), {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                if "Juntas:" in line:
                    joint_names = [j.strip() for j in line.split("Juntas:")[1].split("|")]
                elif "Política:" in line:
                    meta["policy"] = line.split("Política:")[1].strip()
                elif "chunk=" in line:
                    m = re.search(r"chunk=(\d+)", line)
                    if m:
                        meta["chunk"] = int(m.group(1))
                continue
            if line in ("---BEGIN---", "---END---", ""):
                continue
            parts = line.split("\t")
            if not parts:
                continue
            tag = parts[0]
            if tag == "EXECUTED" and len(parts) >= 4:
                exec_steps.append(int(parts[1]))
                exec_ts.append(float(parts[2]))
                executed.append([float(v) for v in parts[3:]])
            elif tag == "NETWORK" and len(parts) >= 5:
                chunk_id = int(parts[1])
                si = int(parts[2])
                obs_step = int(parts[3])
                network[chunk_id].append((si, [float(v) for v in parts[4:]]))
                net_obs[chunk_id] = obs_step

    executed_np = np.array(executed) if executed else np.zeros((0, max(len(joint_names), 1)))
    net_np = {}
    for cid, rows in network.items():
        rows.sort(key=lambda x: x[0])
        net_np[cid] = np.array([r[1] for r in rows])

    return meta, executed_np, exec_steps, exec_ts, net_np, net_obs, joint_names


# ─────────────────────────────────────────────────────────────────────────────
class PlayerState:
    def __init__(self, n_steps, initial_speed=1.0, fps=30, window=120):
        self.n_steps    = n_steps
        self.current    = 0
        self.paused     = False
        self.speed_idx  = min(range(len(SPEEDS)),
                              key=lambda i: abs(SPEEDS[i] - initial_speed))
        self.fps        = fps
        self.window     = window
        self._lock      = threading.Lock()
        self._last_tick = time.perf_counter()

    @property
    def speed(self):
        return SPEEDS[self.speed_idx]

    def tick(self):
        now = time.perf_counter()
        with self._lock:
            if self.paused:
                self._last_tick = now
                return
            dt = now - self._last_tick
            self._last_tick = now
            self.current = min(self.n_steps - 1,
                               int(self.current + dt * self.fps * self.speed))

    def seek(self, delta):
        with self._lock:
            self.current = max(0, min(self.n_steps - 1, self.current + delta))
            self._last_tick = time.perf_counter()

    def toggle_pause(self):
        with self._lock:
            self.paused = not self.paused
            self._last_tick = time.perf_counter()

    def set_speed(self, idx):
        with self._lock:
            self.speed_idx = max(0, min(len(SPEEDS) - 1, idx))


# ─────────────────────────────────────────────────────────────────────────────
def short_name(i, joint_names):
    if not joint_names or i >= len(joint_names):
        return f"j{i}"
    n = joint_names[i]
    n = n.replace(".q", "").replace("kLeft", "L.").replace("kRight", "R.")
    n = n.replace("left_hand_", "LH.").replace("right_hand_", "RH.")
    return n[:13]


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    log_file = None
    fps = 30
    speed = 1.0
    window = 120
    joint_filter = None
    fig_w, fig_h = 15, 8   # polegadas — cabe bem em 1080p

    for arg in sys.argv[1:]:
        if arg in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        elif arg.startswith("--fps="):
            fps = int(arg.split("=", 1)[1])
        elif arg.startswith("--speed="):
            speed = float(arg.split("=", 1)[1])
        elif arg.startswith("--window="):
            window = int(arg.split("=", 1)[1])
        elif arg.startswith("--size="):
            wh = arg.split("=", 1)[1].lower().replace(",", "x").split("x")
            fig_w, fig_h = float(wh[0]), float(wh[1])
        elif arg.startswith("--joints="):
            joint_filter = [int(x) for x in arg.split("=", 1)[1].split(",")]
        elif not arg.startswith("--"):
            log_file = arg

    if log_file is None:
        print("ERRO: informe o arquivo de log.\nUso: python playback_log.py <log.txt>")
        sys.exit(1)

    return log_file, fps, speed, window, joint_filter, fig_w, fig_h


# ─────────────────────────────────────────────────────────────────────────────
def main():
    log_file, anim_fps, init_speed, window_size, joint_filter, fig_w, fig_h = parse_args()

    print(f"Carregando: {log_file}")
    meta, executed, exec_steps, exec_ts, net_np, net_obs, joint_names = load_log(log_file)

    if executed.shape[0] == 0:
        print("ERRO: log sem dados EXECUTED.")
        sys.exit(1)

    n_steps  = executed.shape[0]
    n_joints = executed.shape[1]
    steps_arr = np.array(exec_steps)

    # Filtra grupos
    if joint_filter is not None:
        groups = [(g, [i for i in idx if i in joint_filter], c)
                  for g, idx, c in JOINT_GROUPS]
        groups = [(g, i, c) for g, i, c in groups if i]
        if not groups:
            groups = [("Selecionadas", joint_filter, "#4fc3f7")]
    else:
        groups = [(g, [i for i in idx if i < n_joints], c)
                  for g, idx, c in JOINT_GROUPS]
        groups = [(g, i, c) for g, i, c in groups if i]

    state = PlayerState(n_steps, init_speed, anim_fps, window_size)

    # Pré-computa grid NETWORK
    print("Pre-computando grid da rede...")
    max_step_val = max(exec_steps) if exec_steps else 0
    net_grid  = np.full((max_step_val + 1, n_joints), np.nan)
    net_count = np.zeros((max_step_val + 1, n_joints))
    for chunk_id, chunk_arr in net_np.items():
        obs_s = net_obs.get(chunk_id, 0)
        for si, row in enumerate(chunk_arr):
            s = obs_s + si
            if s <= max_step_val and len(row) >= n_joints:
                net_grid[s] += row[:n_joints]
                net_count[s] += 1
    mask = net_count > 0
    net_grid[mask] = net_grid[mask] / net_count[mask]
    net_aligned = np.full((n_steps, n_joints), np.nan)
    for i, s in enumerate(exec_steps):
        if s <= max_step_val:
            net_aligned[i] = net_grid[s]

    print(f"   {n_steps} steps | {n_joints} juntas | {len(net_np)} chunks")
    print(f"   Janela: {fig_w:.0f}x{fig_h:.0f} pol  (use --size=WxH para ajustar)")

    # ── Layout compacto ────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0d0d1a")
    fig.canvas.manager.set_window_title(f"Playback — {os.path.basename(log_file)}")

    # GridSpec: linha 0 = heatmap (pequeno), linhas 1-2 = grupos 2×2
    # height_ratios: heatmap ocupa 18% da altura, cada linha de grupos 41%
    n_group_rows = (len(groups) + 1) // 2          # ceil(n_groups / 2)
    h_ratios = [1] + [2.3] * n_group_rows
    gs = gridspec.GridSpec(
        1 + n_group_rows, 2,
        figure=fig,
        height_ratios=h_ratios,
        hspace=0.52,
        wspace=0.32,
        top=0.91,
        bottom=0.06,
        left=0.06,
        right=0.97,
    )

    # Heatmap ocupa as 2 colunas da linha 0
    ax_heat = fig.add_subplot(gs[0, :])

    # Grupos em grid 2 colunas
    ax_groups = []
    for gi in range(len(groups)):
        row = 1 + gi // 2
        col = gi % 2
        ax_groups.append(fig.add_subplot(gs[row, col]))

    # Estilo base
    for ax in [ax_heat] + ax_groups:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="gray", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    # ── Heatmap ────────────────────────────────────────────────────────
    heat_data = np.full((n_joints, window_size), np.nan)
    vmin_heat = np.nanmin(executed) if executed.size else -1.5
    vmax_heat = np.nanmax(executed) if executed.size else  1.5
    im_heat = ax_heat.imshow(
        heat_data, aspect="auto", origin="lower", cmap="plasma",
        vmin=vmin_heat, vmax=vmax_heat, interpolation="nearest"
    )
    ax_heat.set_title("Heatmap — todas as juntas (janela deslizante)",
                      color="white", fontsize=8, pad=3)
    ax_heat.set_ylabel("Junta", color="gray", fontsize=7)
    yticks = list(range(0, n_joints, max(1, n_joints // 8)))
    ax_heat.set_yticks(yticks)
    ax_heat.set_yticklabels([short_name(i, joint_names) for i in yticks], fontsize=6)
    ax_heat.set_xticklabels([])
    heat_cursor = ax_heat.axvline(x=window_size - 1, color="#ef5350", linewidth=1.2)

    # Barra de cor discreta no heatmap
    cbar = fig.colorbar(im_heat, ax=ax_heat, orientation="vertical",
                        fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=6, colors="gray")

    # ── Grupos de juntas ───────────────────────────────────────────────
    group_lines_exec, group_lines_net, vlines_groups = [], [], []

    for ax, (gname, gidx, gcol) in zip(ax_groups, groups):
        ax.set_title(gname, color="white", fontsize=8, pad=3)
        ax.set_xlabel("Step", color="gray", fontsize=7)
        ax.set_ylabel("rad", color="gray", fontsize=7)

        lexec, lnet = [], []
        for ji in gidx:
            ln_e, = ax.plot([], [], color=gcol, alpha=0.92, linewidth=1.0,
                            label=short_name(ji, joint_names))
            ln_n, = ax.plot([], [], color=gcol, alpha=0.28, linewidth=0.7,
                            linestyle="--")
            lexec.append(ln_e)
            lnet.append(ln_n)

        ax.legend(fontsize=5.2, loc="upper right", framealpha=0.2,
                  labelcolor="white", facecolor="#111", ncol=2,
                  borderpad=0.3, handlelength=1.0)
        vl = ax.axvline(x=0, color="#ef5350", linewidth=1.0)
        vlines_groups.append(vl)
        group_lines_exec.append(lexec)
        group_lines_net.append(lnet)

    # Legenda global EXECUTED / NETWORK
    legend_handles = [
        Line2D([0], [0], color="#ddd", linewidth=1.4, label="EXECUTED (robo)"),
        Line2D([0], [0], color="#ddd", linewidth=0.7, linestyle="--",
               alpha=0.5, label="NETWORK (previsao)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=7.5, facecolor="#111", labelcolor="white",
               framealpha=0.35, bbox_to_anchor=(0.5, 0.0))

    # Título / status (linha de texto acima do heatmap)
    title_text = fig.suptitle("", color="white", fontsize=9, y=0.985)

    # ── Teclas ─────────────────────────────────────────────────────────
    def on_key(event):
        if event.key in ("q", "escape"):
            plt.close("all")
            sys.exit(0)
        elif event.key == " ":
            state.toggle_pause()
        elif event.key == "left":
            state.seek(-50)
        elif event.key == "right":
            state.seek(+50)
        elif event.key in [str(i + 1) for i in range(len(SPEEDS))]:
            state.set_speed(int(event.key) - 1)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ── Animação ────────────────────────────────────────────────────────
    def update(_frame):
        state.tick()
        cur = state.current

        w_start = max(0, cur - window_size)
        w_end   = min(n_steps, cur + 1)
        win_idx  = slice(w_start, w_end)
        win_steps = steps_arr[win_idx]

        # Heatmap
        seg = executed[win_idx].T
        pad = window_size - seg.shape[1]
        if pad > 0:
            seg = np.concatenate([np.full((n_joints, pad), np.nan), seg], axis=1)
        im_heat.set_data(seg)
        cursor_x = min(window_size - 1, cur - w_start)
        heat_cursor.set_xdata([cursor_x, cursor_x])

        # Grupos
        for ax, (gname, gidx, gcol), lexec, lnet, vl in zip(
                ax_groups, groups, group_lines_exec, group_lines_net, vlines_groups):

            if len(win_steps) == 0:
                continue

            for ji, le, ln in zip(gidx, lexec, lnet):
                le.set_data(win_steps, executed[win_idx, ji])
                ln.set_data(win_steps, net_aligned[win_idx, ji])

            # Escala Y automática com margem
            all_v = np.concatenate([
                executed[win_idx][:, gidx].flatten(),
                net_aligned[win_idx][:, gidx].flatten()
            ])
            valid = all_v[~np.isnan(all_v)]
            if valid.size > 0:
                vlo, vhi = valid.min(), valid.max()
                mg = max((vhi - vlo) * 0.12, 0.05)
                ax.set_ylim(vlo - mg, vhi + mg)

            ax.set_xlim(
                int(win_steps[0]) if len(win_steps) else 0,
                int(win_steps[-1]) + 1 if len(win_steps) else window_size
            )
            vl.set_xdata([steps_arr[cur], steps_arr[cur]])

        # Status no supertítulo
        spd_lbl = SPEED_LABELS[state.speed_idx]
        pause_lbl = "|| PAUSADO" if state.paused else f"> {spd_lbl}"
        pct = cur / max(n_steps - 1, 1) * 100
        fname = os.path.basename(log_file)
        title_text.set_text(
            f"{fname}  |  Step {steps_arr[cur]}/{steps_arr[-1]}  ({pct:.0f}%)"
            f"  |  {pause_lbl}"
            f"  |  [SPACE] Pausa  [< >] +-50  [1-5] Vel  [Q] Sair"
        )

        if cur >= n_steps - 1 and not state.paused:
            state.toggle_pause()

        return ([im_heat, heat_cursor, title_text]
                + [ln for ll in group_lines_exec + group_lines_net for ln in ll]
                + vlines_groups)

    interval_ms = max(16, int(1000 / anim_fps))
    _anim = FuncAnimation(fig, update, interval=interval_ms,
                          blit=False, cache_frame_data=False)

    print("Playback iniciado!")
    print("  [SPACE] Pausa  [< >] +-50 steps  [1-5] Velocidade  [Q] Sair")
    plt.show()


if __name__ == "__main__":
    main()