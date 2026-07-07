"""
Publication figure of the SAE-LEWIS pipeline.

Two panels:
  (a) self-supervised training-data generation (corruption v4)
  (b) architecture and inference (tagger → editor → ranker, iterative)

Outputs figures/pipeline.{pdf,svg,png}. Pure matplotlib — tweak the
coordinates below and re-run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

# ---------------------------------------------------------------------------
# palette: role → face color
C_DATA = "#F0F0F0"      # data / text objects
C_FROZEN = "#D6E4F5"    # frozen pretrained components
C_TRAIN = "#FBE0C3"     # trained components
C_GATE = "#DFF0D8"      # quality gates / checks
C_EDGE = "#555555"
C_LOOP = "#8A5A00"      # iterative-refinement feedback

FS = 7.0                # base font size
FS_S = 6.2              # small font size
FS_T = 8.0              # panel title font size


def box(ax, x, y, w, h, text, fc, fs=FS, lw=0.8):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       linewidth=lw, edgecolor=C_EDGE, facecolor=fc,
                       mutation_scale=1.2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p0, p1, color=C_EDGE, lw=0.9, ls="-",
          connectionstyle="arc3,rad=0.0", shrinkA=1.5, shrinkB=1.5):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8,
                        linewidth=lw, color=color, linestyle=ls,
                        connectionstyle=connectionstyle,
                        shrinkA=shrinkA, shrinkB=shrinkB, zorder=3)
    ax.add_patch(a)


def polyarrow(ax, pts, color=C_EDGE, lw=0.9, ls="-"):
    """Multi-segment connector; arrowhead on the final segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls,
            solid_capstyle="round", zorder=2)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls,
          shrinkA=0.0)


def edge(b, side, frac=0.5):
    """Point on a box edge; b = (x, y, w, h)."""
    x, y, w, h = b
    return {
        "l": (x, y + h * frac),
        "r": (x + w, y + h * frac),
        "t": (x + w * frac, y + h),
        "b": (x + w * frac, y),
    }[side]


# ---------------------------------------------------------------------------
def panel_a(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")
    ax.text(1, 33.5, "(a) Self-supervised training-data generation",
            fontsize=FS_T, fontweight="bold", ha="left", va="top")

    x_src = box(ax, 1, 13, 11.5, 8, "Clean\nsentence $X$\n(Dolma)", C_DATA)

    lex = box(ax, 18, 20, 22, 9,
              "Lexical ops\nMLM REPL / INS / DEL\n(ModernBERT)", C_FROZEN)
    tra = box(ax, 18, 6, 22, 10.5,
              "Grammatical transforms\n17 UD/UniMorph families\n"
              "(VOICE, INV, MOD, NEG, …)\n+ round-trip check", C_DATA)

    gate = box(ax, 45.5, 12, 16.5, 11,
               "Quality gates\nsymmetric SLOR\n(causal Gemma)\n"
               "SAE top-$K$ shift", C_GATE)

    pair = box(ax, 67, 14, 13.5, 8,
               "Pairs $(X, X')$\nboth\ndirections", C_DATA)

    cond = box(ax, 42, 0.8, 30, 7.4,
               "Conditioning: SAE local diff\n(Gemma Scope L12, 16k) top-$k$\n"
               "− grammaticality blocklist", C_FROZEN, fs=FS_S)

    rec = box(ax, 85.5, 8, 13.5, 15.5,
              "Training record\n$X'$ tokens\ngold tags\n"
              "$z_{amp}, z_{sup}$", C_DATA)

    arrow(ax, edge(x_src, "r", 0.75), edge(lex, "l"),
          connectionstyle="arc3,rad=-0.15")
    arrow(ax, edge(x_src, "r", 0.25), edge(tra, "l"),
          connectionstyle="arc3,rad=0.15")
    arrow(ax, edge(lex, "r"), edge(gate, "l", 0.75),
          connectionstyle="arc3,rad=-0.12")
    arrow(ax, edge(tra, "r"), edge(gate, "l", 0.25),
          connectionstyle="arc3,rad=0.12")
    arrow(ax, edge(gate, "r"), edge(pair, "l"))
    arrow(ax, edge(pair, "b", 0.4), edge(cond, "t", 0.65),
          connectionstyle="arc3,rad=0.2")
    arrow(ax, edge(pair, "r"), edge(rec, "l", 0.8),
          connectionstyle="arc3,rad=-0.1")
    arrow(ax, edge(cond, "r"), edge(rec, "b", 0.35),
          connectionstyle="arc3,rad=0.12")

    ax.text(82, 24.2, "token diff → gold tags", fontsize=FS_S,
            ha="center", style="italic", color="#333333")


def panel_b(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")
    ax.text(1, 45.5, "(b) Architecture and inference", fontsize=FS_T,
            fontweight="bold", ha="left", va="top")

    # conditioning row -------------------------------------------------------
    zbox = box(ax, 1, 36, 17, 8,
               "Conditioning\n$z_{amp}, z_{sup} \\in \\mathbb{R}^{16384}$\n"
               "(sparse SAE features)", C_DATA, fs=FS_S)
    proj = box(ax, 22, 36, 19, 8,
               "$\\mathrm{Proj}_A = W_{dec} + BA$\n"
               "$W_{dec}$: frozen SAE decoder\n$BA$: rank-32, trained",
               C_TRAIN, fs=FS_S)
    pref = box(ax, 45, 36, 21, 8,
               "prefix $[\\mathrm{INT}_{amp},\\,\\mathrm{INT}_{sup}]$\n"
               "prepended to the\nresidual stream", C_DATA, fs=FS_S)
    arrow(ax, edge(zbox, "r"), edge(proj, "l"))
    arrow(ax, edge(proj, "r"), edge(pref, "l"))

    # conditioning bus into both models
    bus = [(55.5, 36), (55.5, 32), (16.5, 32), (16.5, 7)]
    xs, ys = zip(*bus)
    ax.plot(xs, ys, color=C_EDGE, lw=0.9, zorder=1)
    arrow(ax, (30, 32), (30, 28.5), shrinkA=0.0)          # → tagger top
    arrow(ax, (16.5, 7), (20, 7), shrinkA=0.0)            # → editor left

    # row 1: tag -------------------------------------------------------------
    xin = box(ax, 1, 21, 12, 7.5, "Input\ntext $x$", C_DATA)
    tagger = box(ax, 20, 18, 20, 10.5,
                 "Tagger\nLLM2Vec bi-encoder\n(Gemma-2-2B + LoRA)\n"
                 "op$_3$: KEEP/REPL/DEL + INS", C_TRAIN, fs=FS_S)
    plans = box(ax, 46, 19.5, 16, 7.5,
                "Edit plans ($\\tau$ levels)\n+ identity\ncandidate",
                C_DATA, fs=FS_S)
    tmpl = box(ax, 66, 19.5, 14, 7.5,
               "Template\n$x\\;[\\mathrm{SEP}]\\;\\tilde{x}$\n"
               "(MASK/INS slots)", C_DATA, fs=FS_S)
    arrow(ax, edge(xin, "r"), edge(tagger, "l", 0.65))
    arrow(ax, edge(tagger, "r", 0.5), edge(plans, "l"))
    arrow(ax, edge(plans, "r"), edge(tmpl, "l"))

    # row 2: fill + rank ------------------------------------------------------
    editor = box(ax, 20, 2, 20, 10.5,
                 "Editor\nLLM2Vec bi-encoder\n(Gemma-2-2B + LoRA)\n"
                 "fills MASK / INS slots", C_TRAIN, fs=FS_S)
    cands = box(ax, 46, 3, 16, 8,
                "Candidates\n+ fill top-$k$\nvariants", C_DATA, fs=FS_S)
    rank = box(ax, 66, 2, 17, 11,
               "Ranker\n$\\alpha$ sae_align $+ \\beta\\,\\Delta$fluency\n"
               "$+ \\gamma$ content $- \\eta$|INS|\n+ fluency gate",
               C_GATE, fs=FS_S)
    out = box(ax, 87, 4.5, 11, 6.5, "Output\n$y$", C_DATA)

    # wrap: template → editor (new row)
    arrow(ax, edge(tmpl, "b", 0.35), edge(editor, "t", 0.9),
          connectionstyle="arc3,rad=0.12")
    arrow(ax, edge(editor, "r", 0.5), edge(cands, "l"))
    arrow(ax, edge(cands, "r"), edge(rank, "l", 0.6))
    arrow(ax, edge(rank, "r", 0.5), edge(out, "l"))

    # identity candidate straight into the ranker
    arrow(ax, edge(plans, "b", 0.25), edge(rank, "t", 0.25),
          connectionstyle="arc3,rad=0.1", ls=(0, (2, 1.5)))
    ax.text(64.5, 15.6, "identity", fontsize=FS_S, ha="center",
            style="italic", color="#333333")

    # iterative refinement feedback along the bottom margin
    polyarrow(ax, [(92.5, 4.5), (92.5, 0.7), (7, 0.7), (7, 21)],
              color=C_LOOP, ls=(0, (4, 2)))
    ax.text(53, 0.95, "iterative refinement (≤ $P$ passes)",
            fontsize=FS_S, ha="center", va="bottom", style="italic",
            color=C_LOOP)


def main():
    fig = plt.figure(figsize=(7.2, 5.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[34, 46],
                          hspace=0.04, left=0.01, right=0.99,
                          top=0.995, bottom=0.05)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    panel_a(ax_a)
    panel_b(ax_b)

    legend = [
        Patch(facecolor=C_FROZEN, edgecolor=C_EDGE, label="frozen pretrained"),
        Patch(facecolor=C_TRAIN, edgecolor=C_EDGE,
              label="trained (LoRA / heads / $BA$)"),
        Patch(facecolor=C_GATE, edgecolor=C_EDGE, label="gates & selection"),
        Patch(facecolor=C_DATA, edgecolor=C_EDGE, label="data / text"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=FS_S,
               frameon=False, handlelength=1.4, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.0))

    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out_dir / f"pipeline.{ext}",
                    dpi=300 if ext == "png" else None)
    print(f"[figure] wrote {out_dir}/pipeline.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
