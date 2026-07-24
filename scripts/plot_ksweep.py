"""Plot exact vs intervention count k (final protocol, 04 S9v).

Reads runs/tables/ksweep_final.json (built by run_ksweep_final.sh) and
writes runs/tables/ksweep_final.png: net exact (solid) and true/random
(faint) for both directions over log2 k.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.load(open("runs/tables/ksweep_final.json"))
ks = [r["k"] for r in rows]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for d, color, label in (("abl", "tab:blue", "ablation"),
                        ("enh", "tab:red", "enhancement")):
    ax.plot(ks, [r[f"{d}_net"] for r in rows], "-o", color=color,
            label=f"{label} net")
    ax.plot(ks, [r[f"{d}_true"] for r in rows], "--", color=color,
            alpha=0.35, label=f"{label} true")
    ax.plot(ks, [r[f"{d}_rand"] for r in rows], ":", color=color,
            alpha=0.35, label=f"{label} random")
ax.set_xscale("log", base=2)
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.set_xlabel("intervention count k (top-k latents of the feature spec)")
ax.set_ylabel("exact (eval-500)")
ax.set_title("Exact vs intervention count — adopted config (L12)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig("runs/tables/ksweep_final.png", dpi=200)
print("wrote runs/tables/ksweep_final.png")
