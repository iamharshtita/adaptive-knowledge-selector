#!/usr/bin/env python3
"""
RL Training Progress Visualisation

Generates a multi-panel dashboard showing how the DQN agent learns
to route queries to correct knowledge sources over the training run.

Panels:
  1. Reward progression (rolling average) & loss curve
  2. Source-selection heatmap (query type × source, per training phase)
  3. Correctness matrix (how often each query type is routed correctly)
  4. Epsilon decay schedule

Usage:
    python scripts/visualise_training.py                          # default log
    python scripts/visualise_training.py --log path/to/log.json   # custom log
"""

import argparse
import json
import os
import sys

import numpy as np

# Use Agg backend so the script works headless (no display required)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── Affinity ground-truth (mirrors reward_evaluator._SOURCE_AFFINITY) ──
_AFFINITY = {
    "TextKnowledgeSource":   ["research", "general"],
    "KnowledgeGraphSource":  ["interaction"],
    "ToolAPISource":         ["dosage", "label", "interaction"],
    "LLMSource":             ["concept", "general"],
    "PDFKnowledgeSource":    ["document", "research", "concept"],
}

SOURCE_ORDER = [
    "TextKnowledgeSource",
    "KnowledgeGraphSource",
    "ToolAPISource",
    "LLMSource",
    "PDFKnowledgeSource",
]
SOURCE_SHORT = {
    "TextKnowledgeSource":   "Text (PubMed)",
    "KnowledgeGraphSource":  "KG (Hetionet)",
    "ToolAPISource":         "ToolAPI",
    "LLMSource":             "LLM (Groq)",
    "PDFKnowledgeSource":    "PDF",
}
QTYPE_ORDER = ["interaction", "dosage", "concept", "research", "document", "label", "general"]


def is_correct(qtype, source):
    """Return True if the source is in the affinity set for the query type."""
    return qtype in _AFFINITY.get(source, [])


def rolling(arr, window=20):
    """Simple rolling mean."""
    out = np.empty(len(arr))
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = np.mean(arr[start:i+1])
    return out


def build_dashboard(log_path: str, out_path: str):
    with open(log_path) as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    rewards = np.array(data.get("rewards", [r["reward"] for r in episodes]))
    losses = np.array(data.get("losses", []))

    n = len(episodes)
    if n == 0:
        print("No episode data found in log.")
        sys.exit(1)

    # ── Derived arrays ──
    epsilons = np.array([e["epsilon"] for e in episodes])
    qtypes   = [e["query_type"] for e in episodes]
    sources  = [e["source_chosen"] for e in episodes]

    # Split training into 3 phases: early / mid / late
    n3 = n // 3
    phases = ["Early (explore)", "Mid (learn)", "Late (exploit)"]
    phase_ranges = [(0, n3), (n3, 2*n3), (2*n3, n)]

    # ── Build per-phase heatmaps ──
    # For each phase: count[qtype][source]
    phase_counts = []
    for start, end in phase_ranges:
        mat = np.zeros((len(QTYPE_ORDER), len(SOURCE_ORDER)))
        for i in range(start, end):
            qi = QTYPE_ORDER.index(qtypes[i]) if qtypes[i] in QTYPE_ORDER else -1
            si = SOURCE_ORDER.index(sources[i]) if sources[i] in SOURCE_ORDER else -1
            if qi >= 0 and si >= 0:
                mat[qi, si] += 1
        # Normalise per row (each qtype sums to 1)
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        mat_norm = mat / row_sums
        phase_counts.append(mat_norm)

    # ── Correctness over time (rolling window) ──
    correct_arr = np.array([1.0 if is_correct(qt, sr) else 0.0
                            for qt, sr in zip(qtypes, sources)])
    correctness_rolling = rolling(correct_arr, window=30)

    # ── Per-qtype correctness curve ──
    qtype_correct = {}
    for qt in QTYPE_ORDER:
        mask = [i for i, q in enumerate(qtypes) if q == qt]
        if mask:
            vals = correct_arr[mask]
            qtype_correct[qt] = rolling(vals, window=max(5, len(vals)//10))
        else:
            qtype_correct[qt] = np.array([])

    # ══════════════════════════════════════════════════════════════════
    #  PLOT
    # ══════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(22, 18), facecolor="#0d1117")
    fig.suptitle("RL Agent Training Progress — Adaptive Knowledge Selector",
                 fontsize=20, fontweight="bold", color="white", y=0.98)

    gs = gridspec.GridSpec(3, 3, hspace=0.38, wspace=0.30,
                           left=0.06, right=0.96, top=0.93, bottom=0.05)

    dark_bg = "#0d1117"
    panel_bg = "#161b22"
    grid_c = "#21262d"
    text_c = "#c9d1d9"

    # ── Custom colormaps ──
    cmap_heat = LinearSegmentedColormap.from_list(
        "routing",
        ["#0d1117", "#1f6feb", "#58a6ff", "#79c0ff", "#a5d6ff"],
    )
    cmap_correct = LinearSegmentedColormap.from_list(
        "correct",
        ["#f85149", "#d29922", "#3fb950"],
    )

    def style_ax(ax, title):
        ax.set_facecolor(panel_bg)
        ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(colors=text_c, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(grid_c)

    # ──────────── Panel 1: Reward + Loss ────────────
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1, "① Reward Progression & Loss")
    ax1.plot(rewards, alpha=0.15, color="#58a6ff", linewidth=0.5, label="_raw")
    ax1.plot(rolling(rewards, 30), color="#58a6ff", linewidth=2, label="Avg Reward (30-ep)")
    ax1.axhline(y=0, color="#f85149", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Episode", color=text_c)
    ax1.set_ylabel("Reward", color="#58a6ff")
    ax1.set_ylim(-0.5, 1.15)
    ax1.grid(True, color=grid_c, alpha=0.3)

    if len(losses) > 0:
        ax1b = ax1.twinx()
        loss_x = np.linspace(0, n, len(losses))
        ax1b.plot(loss_x, losses, color="#f0883e", alpha=0.5, linewidth=1, label="Loss")
        ax1b.set_ylabel("Loss (MSE)", color="#f0883e")
        ax1b.tick_params(colors="#f0883e", labelsize=9)
        for spine in ax1b.spines.values():
            spine.set_color(grid_c)

    ax1.legend(loc="lower right", facecolor=panel_bg, edgecolor=grid_c,
               labelcolor=text_c, fontsize=9)

    # ──────────── Panel 2: Epsilon decay ────────────
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2, "② Exploration Decay (ε)")
    ax2.fill_between(range(n), epsilons, alpha=0.3, color="#a371f7")
    ax2.plot(epsilons, color="#a371f7", linewidth=2)
    ax2.set_xlabel("Episode", color=text_c)
    ax2.set_ylabel("Epsilon", color="#a371f7")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, color=grid_c, alpha=0.3)
    # Annotate phases
    for i, (s, e) in enumerate(phase_ranges):
        mid = (s + e) / 2
        ax2.axvspan(s, e, alpha=0.06, color=["#f85149", "#d29922", "#3fb950"][i])
        ax2.text(mid, 0.95, phases[i], ha="center", va="top",
                 fontsize=8, color=text_c, alpha=0.7)

    # ──────────── Panels 3-5: Source selection heatmaps per phase ────────────
    for pi in range(3):
        ax = fig.add_subplot(gs[1, pi])
        style_ax(ax, f"③ Routing: {phases[pi]}")
        mat = phase_counts[pi]

        im = ax.imshow(mat, aspect="auto", cmap=cmap_heat, vmin=0, vmax=1)

        # Overlay text
        for qi in range(len(QTYPE_ORDER)):
            for si in range(len(SOURCE_ORDER)):
                val = mat[qi, si]
                # Highlight correct cells
                is_corr = is_correct(QTYPE_ORDER[qi], SOURCE_ORDER[si])
                border = "★" if is_corr and val > 0.3 else ""
                color = "#3fb950" if is_corr and val > 0.3 else text_c
                ax.text(si, qi, f"{val:.0%}{border}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold" if border else "normal")

        ax.set_xticks(range(len(SOURCE_ORDER)))
        ax.set_xticklabels([SOURCE_SHORT[s] for s in SOURCE_ORDER], rotation=35,
                           ha="right", fontsize=8)
        ax.set_yticks(range(len(QTYPE_ORDER)))
        ax.set_yticklabels(QTYPE_ORDER, fontsize=9)

    # ──────────── Panel 6: Correctness over time ────────────
    ax6 = fig.add_subplot(gs[2, :2])
    style_ax(ax6, "④ Routing Accuracy Over Time (rolling 30-ep window)")
    ax6.plot(correctness_rolling, color="#3fb950", linewidth=2.5, label="Overall")
    ax6.fill_between(range(n), correctness_rolling, alpha=0.15, color="#3fb950")

    # Per query-type lines
    qtype_colors = {
        "interaction": "#f85149", "dosage": "#f0883e", "concept": "#d29922",
        "research": "#58a6ff", "document": "#a371f7", "label": "#79c0ff",
        "general": "#8b949e",
    }
    for qt in QTYPE_ORDER:
        if len(qtype_correct.get(qt, [])) > 0:
            # Plot at the episode indices where this qtype appears
            mask = [i for i, q in enumerate(qtypes) if q == qt]
            if mask:
                ax6.plot(mask, qtype_correct[qt][:len(mask)],
                         color=qtype_colors.get(qt, text_c),
                         linewidth=1.2, alpha=0.6, label=qt)

    ax6.set_xlabel("Episode", color=text_c)
    ax6.set_ylabel("Correct Routing %", color="#3fb950")
    ax6.set_ylim(-0.05, 1.1)
    ax6.grid(True, color=grid_c, alpha=0.3)
    ax6.legend(loc="lower right", facecolor=panel_bg, edgecolor=grid_c,
               labelcolor=text_c, fontsize=8, ncol=3)

    # ──────────── Panel 7: Final source preference matrix ────────────
    ax7 = fig.add_subplot(gs[2, 2])
    style_ax(ax7, "⑤ Final Routing Preference")

    # Use late-phase data
    mat_final = phase_counts[2]
    im7 = ax7.imshow(mat_final, aspect="auto", cmap=cmap_correct, vmin=0, vmax=1)
    for qi in range(len(QTYPE_ORDER)):
        for si in range(len(SOURCE_ORDER)):
            val = mat_final[qi, si]
            is_corr = is_correct(QTYPE_ORDER[qi], SOURCE_ORDER[si])
            txt = f"{val:.0%}"
            if is_corr and val > 0.3:
                txt += " ✓"
            elif not is_corr and val > 0.3:
                txt += " ✗"
            color = "white" if val > 0.5 else text_c
            ax7.text(si, qi, txt, ha="center", va="center",
                     fontsize=8, color=color, fontweight="bold")

    ax7.set_xticks(range(len(SOURCE_ORDER)))
    ax7.set_xticklabels([SOURCE_SHORT[s] for s in SOURCE_ORDER], rotation=35,
                        ha="right", fontsize=8)
    ax7.set_yticks(range(len(QTYPE_ORDER)))
    ax7.set_yticklabels(QTYPE_ORDER, fontsize=9)

    # ── Colour bar ──
    cbar = fig.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=text_c, labelsize=8)
    cbar.set_label("Selection Frequency", color=text_c, fontsize=9)

    # ── Save ──
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor=dark_bg)
    plt.close(fig)
    print(f"\n✅ Dashboard saved to: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise RL training progress")
    parser.add_argument("--log", default="data/rl_selector/training_log.json",
                        help="Path to training_log.json")
    parser.add_argument("--out", default="visualizations/training_dashboard.png",
                        help="Path for output image")
    args = parser.parse_args()
    build_dashboard(args.log, args.out)
