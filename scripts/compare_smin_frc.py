#!/usr/bin/env python3
"""S_min 安定核 × FRC top-3 比較(🔵免許規則・書ける形(ii)の測定器)。

prune_spec.py --effector steer の出力(事例ごとの最小介入集合 S_min)から、
現象ごとの**事例横断の安定核**(出現率 ≥ core-frac の特徴)を抽出し、
LinguaLens の FRC top-3 と直接比較する。

安定核の意味(PAPER_OUTLINE 🔵): S_min には feature方向+事例内容が混在
するが、同一現象の事例間で**共通に生き残る**特徴は現象側、事例ごとに
入れ替わる特徴は内容側 — という操作的分解。安定核 = 「因果検証済みの、
featureに対応するactivations」の候補であり、これと FRC top-3 の本数・
中身を比べて初めて「対応する活性はtop-3より多い/別物」が書ける。

CPU-only(recordsを読むだけ)。B-2 の直後に同セッションで走る。
    python scripts/compare_smin_frc.py \
        --smin-records runs/prod_gemma_v6/prune_spec_steer/records.jsonl \
        --out runs/tables/smin_vs_frc
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smin-records", required=True)
    p.add_argument("--frc-json", default="runs/frc/identified_l12_16k_r3.json")
    p.add_argument("--auroc-json",
                   default="runs/auroc/identified_l12_16k_r1.json")
    p.add_argument("--core-frac", type=float, default=0.5,
                   help="安定核 = 現象のS_min集合の≥この割合に出現する特徴")
    p.add_argument("--min-pairs", type=int, default=3)
    p.add_argument("--out", default="runs/tables/smin_vs_frc")
    return p.parse_args()


def main():
    args = parse_args()
    rows = [json.loads(l) for l in open(args.smin_records) if l.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"[smin-frc] {len(ok)} pruned pairs (of {len(rows)})")

    frc = {}
    if Path(args.frc_json).exists():
        raw = json.loads(Path(args.frc_json).read_text())
        frc = {ph: [int(f) for f, _ in lst][:3] for ph, lst in raw.items()}
    auroc1 = {}
    if Path(args.auroc_json).exists():
        raw = json.loads(Path(args.auroc_json).read_text())
        auroc1 = {ph: int(lst[0][0]) for ph, lst in raw.items() if lst}

    by_ph = defaultdict(list)
    labels = {}
    for r in ok:
        s = frozenset(int(e["f"]) for e in r["s_min"])
        by_ph[r["feature"]].append(s)
        for e in r["s_min"]:
            if e.get("label"):
                labels[int(e["f"])] = e["label"][:45]

    L = [f"# S_min 安定核 × FRC top-3(core-frac {args.core_frac:g}, "
         f"min-pairs {args.min_pairs})",
         "",
         "安定核 = 因果検証済みの feature対応集合の候補(事例間で共通に"
         "生き残る介入ハンドル)。FRC3列 = LinguaLensの現象同定 top-3。",
         "",
         "| phenomenon | pairs | mean\\|S_min\\| | union | Jaccard | "
         "\\|core\\| | core∩FRC3 | FRC3出現率 | AUROC1出現率 | core特徴 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    n_core_any, n_core_frc, core_sizes = 0, 0, []
    for ph, sets in sorted(by_ph.items(), key=lambda kv: -len(kv[1])):
        n = len(sets)
        if n < args.min_pairs:
            continue
        cnt = defaultdict(int)
        for s in sets:
            for f in s:
                cnt[f] += 1
        core = {f for f, c in cnt.items() if c / n >= args.core_frac}
        union = set().union(*sets)
        js = [len(a & b) / len(a | b)
              for i, a in enumerate(sets) for b in sets[i + 1:]]
        f3 = frc.get(ph, [])
        # FRC3出現率: FRC top-3 の各特徴が S_min に現れる頻度の平均
        f3_rate = (st.mean([cnt.get(f, 0) / n for f in f3]) if f3
                   else float("nan"))
        a1 = auroc1.get(ph)
        a1_rate = (cnt.get(a1, 0) / n) if a1 is not None else float("nan")
        inter = core & set(f3)
        if core:
            n_core_any += 1
            core_sizes.append(len(core))
        if inter:
            n_core_frc += 1
        core_txt = ", ".join(
            f"{f}({cnt[f]}/{n}){(' ' + labels[f]) if f in labels else ''}"
            for f in sorted(core, key=lambda x: -cnt[x])[:4]) or "—"
        L.append(f"| {ph} | {n} | "
                 f"{st.mean([len(s) for s in sets]):.1f} | {len(union)} | "
                 f"{st.mean(js):.2f} | {len(core)} | {len(inter)} | "
                 f"{f3_rate:.2f} | {a1_rate:.2f} | {core_txt} |")

    n_ph = sum(1 for s in by_ph.values() if len(s) >= args.min_pairs)
    L += ["", "## 集計", "",
          f"- 対象現象: {n_ph}(pairs ≥ {args.min_pairs})",
          f"- 安定核が非空: {n_core_any}/{n_ph}",
          f"- 安定核がFRC top-3と交わる: {n_core_frc}/{n_ph}",
          (f"- 安定核サイズ: median {st.median(core_sizes):g} / "
           f"mean {st.mean(core_sizes):.1f}(vs FRC の 3)"
           if core_sizes else "- 安定核なし"),
          "",
          "読み(免許規則・形(ii)): 安定核が非空でFRC3と交わらない → "
          "因果的に有効な対応集合はFRCの同定と別物。核がFRC3を含みつつ"
          "大きい → 対応集合はtop-3より広い。核が空(低Jaccard) → 事例"
          "あたり少数だが固定の対応集合は存在しない(現象レベル選択の"
          "失敗の機構的説明)。いずれの場合もFRC3出現率の列が「FRCの3本は"
          "因果ハンドルとしてどれだけ再利用されているか」を直接測る。"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[smin-frc] wrote {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
