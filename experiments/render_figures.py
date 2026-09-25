"""README figures: what three memories hold after 600 frames at the same budget."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

from sihtimuisti import race
from sihtimuisti.world import make_world

# ordinal one-hue ramp: level 0 (sharp) darkest -> level 5 (2x2) lightest
LEVEL_COLORS = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#86b6ef"]
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#d9d9d6"
SEED, BUDGET = 1, 8
ROWS = [("thumbnails_L3", "best thumbnail (8x8, every frame it can fit)"),
        ("fade_recall", "fade + recall (blur with age)"),
        ("rd", "rd (least-error octave drop\nor neighbour merge)")]


def main():
    world = make_world(SEED, 600)
    sched = race.build_schedule(world, SEED)
    frng = np.random.default_rng(SEED * 1000 + 1)
    frames = [world.frame(t, frng) for t in range(world.T)]
    mems = {p: race.run(SEED, BUDGET, p, world=world, schedule=sched, frames=frames)["memory"]
            for p, _ in ROWS}

    # --- figure 1: timeline -------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=150)
    changes = [t for t in range(1, world.T) if world.configs[t] != world.configs[t - 1]]
    y = len(ROWS)
    for t in changes:
        ax.plot([t, t], [y + 0.15, y + 0.75], color=MUTED, lw=0.8)
    ax.text(-8, y + 0.45, "scene changes", ha="right", va="center", color=INK, fontsize=9)
    for r, (pol, label) in enumerate(ROWS):
        yy = len(ROWS) - 1 - r
        for it in mems[pol].items:
            x0, x1 = it.tmin, it.tmax + 1
            ax.add_patch(Rectangle((x0, yy + 0.1), max(x1 - x0, 1), 0.8,
                                   facecolor=LEVEL_COLORS[it.level], edgecolor="white", lw=0.4))
        n = len(mems[pol].items)
        ax.text(-8, yy + 0.5, f"{label}\n{n} items", ha="right", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, world.T)
    ax.set_ylim(0, len(ROWS) + 0.9)
    ax.set_yticks([])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("frame", color=MUTED, fontsize=9)
    res = ["64x64", "32x32", "16x16", "8x8", "4x4", "2x2"]
    ax.legend(handles=[Patch(color=LEVEL_COLORS[i], label=res[i]) for i in range(6)],
              title="resolution kept", ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              frameon=False, fontsize=8, title_fontsize=8)
    ax.set_title(f"What each memory holds after 600 frames, same budget (8 frames' worth), world {SEED}.\n"
                 "Each block is one stored item: width = frames it covers, shade = detail kept.",
                 fontsize=10, color=INK, loc="left")
    fig.subplots_adjust(left=0.25, right=0.99, top=0.82, bottom=0.3)
    fig.savefig("results/fig_timeline.png")
    plt.close(fig)

    # --- figure 2: the rd memory, as pictures ------------------------------
    items = mems["rd"].items
    cols = 10
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.3), dpi=150)
    for ax, it in zip(axes.ravel(), items):
        ax.imshow(it.view(), cmap="gray", vmin=0.1, vmax=0.9, interpolation="nearest")
        ax.set_title(f"t {it.tmin}-{it.tmax}\n{res[it.level]}", fontsize=6, color=INK)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("The rd memory's contents after 600 frames (8 frames' budget): "
                 "each tile is one item (frames covered, resolution kept)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig("results/fig_rd_memory.png")
    plt.close(fig)
    print("wrote results/fig_timeline.png, results/fig_rd_memory.png")


if __name__ == "__main__":
    main()
