"""Make the Part 1 report figures (figs/) from the result files in ../results/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "mathtext.fontset": "stix"})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential ramp 100->700
CMAP = LinearSegmentedColormap.from_list("blue_seq", BLUES)
SERIES = {  # colour follows the entity; line style = method, marker = learning-rate regime (secondary encoding)
    "LoRA-best":   dict(color="#2a78d6", ls="-",  marker="o", label=r"LoRA, tuned $\eta$"),
    "LoRA+-best":  dict(color="#eb6834", ls="--", marker="o", label=r"LoRA+, tuned $(\eta_A,\lambda)$"),
    "LoRA-lowlr":  dict(color="#1baf7a", ls="-",  marker="s", label=r"LoRA, $\eta=10^{-4}$"),
    "LoRA+-lowlr": dict(color="#eda100", ls="--", marker="s", label=r"LoRA+, $\eta_A=10^{-4}, \lambda=16$"),
}

def load(name):
    return [json.loads(l) for l in open(os.path.join(R, name)) if l.strip()]

# ------------------------------------------------------------------ Figure 1: heat maps (seed 42 grid)
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.32})
for ax, task, title in [(axes[0], "textvqa", "TextVQA (zero-shot 32.8)"), (axes[1], "vqav2", "VQAv2 (zero-shot 45.3)")]:
    recs = load(f"grid_{task}.jsonl")
    etas = sorted({r["eta_a"] for r in recs}); lams = sorted({r["ratio"] for r in recs})
    M = np.full((len(lams), len(etas)), np.nan)
    for r in recs:
        M[lams.index(r["ratio"]), etas.index(r["eta_a"])] = r["acc"]
    lo, hi = np.nanmin(M), np.nanmax(M)
    im = ax.imshow(M, origin="lower", cmap=CMAP, aspect="auto", vmin=lo, vmax=hi)
    for i in range(len(lams)):
        for j in range(len(etas)):
            v = M[i, j]
            dark = (v - lo) / (hi - lo) > 0.55
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.8, color="white" if dark else INK)
    best = max(recs, key=lambda r: r["acc"]); bl = max([r for r in recs if r["ratio"] == 1], key=lambda r: r["acc"])
    for b, ls, lw in [(bl, (0, (2, 1.2)), 1.3), (best, "-", 1.6)]:
        ax.add_patch(Rectangle((etas.index(b["eta_a"]) - .46, lams.index(b["ratio"]) - .46), .92, .92,
                               fill=False, ec="#eb6834", lw=lw, ls=ls))
    ax.set_xticks(range(len(etas)), [f"{e*1e4:g}" for e in etas])
    ax.set_yticks(range(len(lams)), [f"{l:g}" for l in lams])
    ax.set_xlabel(r"$\eta_A$ ($\times 10^{-4}$)"); ax.set_ylabel(r"$\lambda=\eta_B/\eta_A$")
    ax.set_title(title, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=6.5)
    cb.set_label("VQA accuracy (%)", fontsize=7, color=MUTED)
fig.savefig(os.path.join(OUT, "fig1_heatmaps.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_heatmaps.png"), dpi=220, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ Figure 2: accuracy during training (3 seeds)
fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.25), gridspec_kw={"wspace": 0.22})
for ax, task, title in [(axes[0], "textvqa", "TextVQA"), (axes[1], "vqav2", "VQAv2")]:
    zrows = [json.loads(l) for l in open(os.path.join(R, f"preds_zeroshot_{task}.jsonl"))]
    z200 = 100 * np.mean([r["score"] for r in zrows[:200]])       # same 200 questions as the curve
    recs = load(f"seeds_{task}.jsonl")
    for name, st in SERIES.items():
        rs = [r for r in recs if r["name"] == name]
        if not rs:
            continue
        steps = [0] + [s for s, _ in rs[0]["curve"]]
        A = np.array([[z200] + [a for _, a in r["curve"]] for r in rs])
        m, sd = A.mean(0), A.std(0, ddof=1)
        ax.fill_between(steps, m - sd, m + sd, color=st["color"], alpha=0.13, lw=0)
        ax.plot(steps, m, color=st["color"], ls=st["ls"], lw=1.5, marker=st["marker"], ms=3.6,
                mec="white", mew=0.6, label=st["label"])
    ax.axvline(125, color=MUTED, lw=0.6, ls=":")
    ax.text(127, ax.get_ylim()[0] + 0.03 * np.ptp(ax.get_ylim()), "epoch 2", fontsize=6.5, color=MUTED)
    ax.set_title(title, color=INK); ax.set_xlabel("training step")
    ax.set_xticks([0, 20, 40, 70, 125, 190]); ax.set_xticks([10], minor=True)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("VQA accuracy (%), 200 eval q.")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.2)
fig.savefig(os.path.join(OUT, "fig2_curves.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig2_curves.png"), dpi=220, bbox_inches="tight")
plt.close(fig)
print("figures written to", OUT)
