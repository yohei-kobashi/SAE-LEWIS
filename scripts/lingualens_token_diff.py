"""
Sample LinguaLens minimal pairs and report token-level diff distribution.

For each (sentence1, sentence2) pair we tokenize with the Gemma tokenizer
used by the SAE-LEWIS pipeline (google/gemma-2-2b by default) and compute:

  * len(tok1), len(tok2), |Δlen|
  * EDIT(tok1, tok2) -- token-level Levenshtein (insert/delete/substitute,
    all cost 1). This corresponds to the minimum N a SAE-LEWIS compound
    corruption would need to transform sentence1 into sentence2.
  * |set(tok1) △ set(tok2)| -- set-level symmetric difference (order-
    insensitive, ignores duplicates). Coarser but useful when many edits
    are local rearrangements.

Output: a Markdown report with per-language and overall percentiles plus
a histogram of edit distances. Designed to be checked into the repo
without going into README.md.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer


def token_edit_distance(a: List[int], b: List[int]) -> int:
    """Classical Levenshtein with unit costs (insert/delete/substitute)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def percentiles(values: List[float], qs: List[float]) -> List[float]:
    if not values:
        return [float("nan")] * len(qs)
    s = sorted(values)
    out: List[float] = []
    for q in qs:
        if q <= 0:
            out.append(float(s[0]))
            continue
        if q >= 1:
            out.append(float(s[-1]))
            continue
        pos = q * (len(s) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        frac = pos - lo
        out.append(s[lo] * (1 - frac) + s[hi] * frac)
    return out


def histogram(values: List[int], buckets: List[Tuple[int, int, str]]) -> Counter:
    out: Counter = Counter()
    for v in values:
        placed = False
        for lo, hi, label in buckets:
            if lo <= v <= hi:
                out[label] += 1
                placed = True
                break
        if not placed:
            out["other"] += 1
    return out


def summarise(records: List[Dict]) -> Dict:
    if not records:
        return {
            "n": 0, "len1_p50": None, "len2_p50": None,
            "delta_len_p50": None, "edit_p50": None, "set_diff_p50": None,
        }
    len1 = [r["len1"] for r in records]
    len2 = [r["len2"] for r in records]
    dlen = [abs(r["len1"] - r["len2"]) for r in records]
    edit = [r["edit"] for r in records]
    setd = [r["set_diff"] for r in records]
    qs = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    p_len1 = percentiles(len1, qs)
    p_len2 = percentiles(len2, qs)
    p_dlen = percentiles(dlen, qs)
    p_edit = percentiles(edit, qs)
    p_setd = percentiles(setd, qs)
    return {
        "n": len(records),
        "len1_mean": sum(len1) / len(len1),
        "len2_mean": sum(len2) / len(len2),
        "edit_mean": sum(edit) / len(edit),
        "set_diff_mean": sum(setd) / len(setd),
        "len1_pcts": p_len1,
        "len2_pcts": p_len2,
        "delta_len_pcts": p_dlen,
        "edit_pcts": p_edit,
        "set_diff_pcts": p_setd,
        # Bucketed histogram of edit distance
        "edit_hist": dict(histogram(edit, [
            (0, 0, "=0"),
            (1, 1, "=1"),
            (2, 2, "=2"),
            (3, 3, "=3"),
            (4, 5, "4-5"),
            (6, 10, "6-10"),
            (11, 20, "11-20"),
            (21, 50, "21-50"),
            (51, 10_000, "51+"),
        ])),
    }


def render_md(
    args, by_lang: Dict[str, Dict], overall: Dict,
    feature_summary: List[Tuple[str, Dict]],
) -> str:
    qs_labels = ["p05", "p25", "p50", "p75", "p90", "p95", "p99"]
    out: List[str] = []
    out.append("# LinguaLens minimal-pair token-diff distribution\n")
    out.append(
        f"Dataset: `THU-KEG/LinguaLens-Data` (split=train, total 7251 pairs).  "
        f"Tokenizer: `{args.tokenizer}`.  Sample size: {args.sample_size} "
        f"(seed={args.seed}).\n"
    )
    out.append(
        "**EDIT** = token-level Levenshtein distance (insert/delete/substitute, "
        "all cost 1) — the minimum N a SAE-LEWIS compound corruption would "
        "need to transform `sentence1` into `sentence2`.  **set_diff** = "
        "`|set(tok1) △ set(tok2)|` — order-insensitive, dedup-counted token "
        "set symmetric difference.\n"
    )

    # Overall
    out.append("\n## Overall\n")
    out.append(_render_summary_table(overall, qs_labels))
    out.append(
        "\n**Edit-distance histogram (overall):**\n\n"
        "| edit | count | % |\n"
        "|------|-------|---|\n"
    )
    n_total = overall["n"]
    for label in ("=0", "=1", "=2", "=3", "4-5", "6-10",
                  "11-20", "21-50", "51+", "other"):
        c = overall["edit_hist"].get(label, 0)
        pct = (100.0 * c / n_total) if n_total else 0.0
        out.append(f"| {label} | {c} | {pct:.1f}% |\n")

    # Per language
    out.append("\n## Per language\n")
    for lang, summ in sorted(by_lang.items()):
        out.append(f"\n### {lang}  (n = {summ['n']})\n")
        out.append(_render_summary_table(summ, qs_labels))
        out.append(
            "\nEdit histogram: "
            + ", ".join(
                f"{label}={summ['edit_hist'].get(label, 0)}"
                for label in ("=0", "=1", "=2", "=3", "4-5",
                              "6-10", "11-20", "21-50", "51+")
            )
            + "\n"
        )

    # Top features (by edit-distance median, to flag what's hardest)
    out.append("\n## Top features by edit-distance p50 (largest first)\n")
    out.append("| feature | n | EDIT p50 | EDIT p90 | set_diff p50 | mean len1 | mean len2 |\n")
    out.append("|---------|---|----------|----------|--------------|-----------|-----------|\n")
    rows = sorted(feature_summary, key=lambda kv: kv[1]["edit_pcts"][2], reverse=True)
    for name, s in rows[:25]:
        out.append(
            f"| {name} | {s['n']} | {s['edit_pcts'][2]:.1f} | "
            f"{s['edit_pcts'][4]:.1f} | {s['set_diff_pcts'][2]:.1f} | "
            f"{s['len1_mean']:.1f} | {s['len2_mean']:.1f} |\n"
        )
    out.append("\n## Bottom features by edit-distance p50 (smallest first)\n")
    out.append("| feature | n | EDIT p50 | EDIT p90 | set_diff p50 | mean len1 | mean len2 |\n")
    out.append("|---------|---|----------|----------|--------------|-----------|-----------|\n")
    for name, s in rows[-25:][::-1]:
        out.append(
            f"| {name} | {s['n']} | {s['edit_pcts'][2]:.1f} | "
            f"{s['edit_pcts'][4]:.1f} | {s['set_diff_pcts'][2]:.1f} | "
            f"{s['len1_mean']:.1f} | {s['len2_mean']:.1f} |\n"
        )

    out.append(
        "\n## Notes\n\n"
        "* Gemma is a subword (SentencePiece) tokenizer.  For Chinese the "
        "Gemma tokenizer falls through to roughly character-level "
        "segmentation, which inflates token counts and edit distance "
        "relative to a Chinese-native tokenizer.  Treat the English and "
        "Chinese rows separately when comparing to the SAE-LEWIS "
        "compound-corruption N distribution (which is currently calibrated "
        "on English Dolma only).\n"
        "* The edit distance is computed over the FULL tokenization "
        "including BOS but excluding nothing else — i.e. exactly the "
        "number of token-level operations the SAE-LEWIS editor would have "
        "to emit if `sentence1` were corrupted into `sentence2`.\n"
        "* Re-run with `--sample-size 7251` to use the full dataset, or "
        "`--language English` / `--language Chinese` to slice.\n"
    )
    return "".join(out)


def _render_summary_table(summ: Dict, qs_labels: List[str]) -> str:
    out: List[str] = []
    out.append("\n| metric | mean | " + " | ".join(qs_labels) + " |\n")
    out.append("|--------|------|" + "|".join("-----" for _ in qs_labels) + "|\n")
    out.append(
        f"| `len(tok1)`   | {summ['len1_mean']:.2f} | "
        + " | ".join(f"{v:.1f}" for v in summ["len1_pcts"])
        + " |\n"
    )
    out.append(
        f"| `len(tok2)`   | {summ['len2_mean']:.2f} | "
        + " | ".join(f"{v:.1f}" for v in summ["len2_pcts"])
        + " |\n"
    )
    out.append(
        f"| `|Δlen|`      | -   | "
        + " | ".join(f"{v:.1f}" for v in summ["delta_len_pcts"])
        + " |\n"
    )
    out.append(
        f"| `EDIT`        | {summ['edit_mean']:.2f} | "
        + " | ".join(f"{v:.1f}" for v in summ["edit_pcts"])
        + " |\n"
    )
    out.append(
        f"| `set_diff`    | {summ['set_diff_mean']:.2f} | "
        + " | ".join(f"{v:.1f}" for v in summ["set_diff_pcts"])
        + " |\n"
    )
    return "".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="google/gemma-2-2b",
                   help="HF id of the tokenizer to use (matches the LLM "
                        "in the SAE-LEWIS pipeline).")
    p.add_argument("--sample-size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--language", default=None,
                   help="Filter to a single language (English | Chinese). "
                        "None = use both.")
    p.add_argument("--out-md", default="runs/lingualens_token_diff.md")
    p.add_argument("--out-jsonl", default=None,
                   help="Optional path to write per-pair raw records as JSONL.")
    args = p.parse_args()

    print(f"[lingualens] loading dataset")
    ds = load_dataset("THU-KEG/LinguaLens-Data", split="train")
    print(f"[lingualens] total examples: {len(ds)}")
    if args.language:
        ds = ds.filter(lambda r: r["language"] == args.language)
        print(f"[lingualens] after language={args.language} filter: {len(ds)}")

    rng = random.Random(args.seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    sample_n = min(args.sample_size, len(idx))
    chosen = idx[:sample_n]
    print(f"[lingualens] sampling {sample_n} pairs (seed={args.seed})")

    print(f"[lingualens] loading tokenizer {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    records: List[Dict] = []
    by_lang: Dict[str, List[Dict]] = defaultdict(list)
    by_feat: Dict[str, List[Dict]] = defaultdict(list)
    for i, k in enumerate(chosen):
        ex = ds[int(k)]
        ids1 = tok(ex["sentence1"], add_special_tokens=True)["input_ids"]
        ids2 = tok(ex["sentence2"], add_special_tokens=True)["input_ids"]
        ed = token_edit_distance(ids1, ids2)
        sd = len(set(ids1) ^ set(ids2))
        rec = {
            "language": ex["language"],
            "feature":  ex["feature"],
            "len1":     len(ids1),
            "len2":     len(ids2),
            "edit":     ed,
            "set_diff": sd,
        }
        records.append(rec)
        by_lang[ex["language"]].append(rec)
        by_feat[ex["feature"]].append(rec)
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1} / {sample_n}")

    overall = summarise(records)
    by_lang_summ = {lang: summarise(rs) for lang, rs in by_lang.items()}
    feature_summary = [
        (name, summarise(rs)) for name, rs in by_feat.items() if len(rs) >= 5
    ]

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(args, by_lang_summ, overall, feature_summary))
    print(f"[lingualens] wrote {out_md}")

    if args.out_jsonl:
        out_jl = Path(args.out_jsonl)
        out_jl.parent.mkdir(parents=True, exist_ok=True)
        with out_jl.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[lingualens] wrote {out_jl}")

    # Brief console summary
    print()
    print(f"Overall (n={overall['n']}):")
    print(f"  EDIT       p50={overall['edit_pcts'][2]:.1f}  "
          f"p75={overall['edit_pcts'][3]:.1f}  "
          f"p90={overall['edit_pcts'][4]:.1f}  "
          f"p95={overall['edit_pcts'][5]:.1f}")
    print(f"  set_diff   p50={overall['set_diff_pcts'][2]:.1f}  "
          f"p75={overall['set_diff_pcts'][3]:.1f}  "
          f"p90={overall['set_diff_pcts'][4]:.1f}")
    print(f"  len1       p50={overall['len1_pcts'][2]:.1f}  "
          f"len2 p50={overall['len2_pcts'][2]:.1f}")
    print(f"  edit hist (overall): {overall['edit_hist']}")


if __name__ == "__main__":
    main()
