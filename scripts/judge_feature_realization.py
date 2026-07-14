"""
LinguaLens-basis evaluation: LLM-judged feature realization rate (FRR).

LinguaLens evaluates interventions NOT by exact match but by an LLM
judge (GPT-4o in the paper, §4.2.2) that compares feature PROMINENCE
between the intervened and unmodified outputs; success = prominence
moved in the commanded direction, with a random-feature control group.
This script adapts that protocol to the minimal-pair editing regime and
runs it over EXISTING records.jsonl files (probe / B2 / pipeline all
store output text), so no generation is repeated:

  gold direction  : judge(src vs tgt)  — which side exhibits the pair's
                    `feature` more (LinguaLens-Data column). Cached in
                    --gold-cache and shared across systems. Pairs the
                    judge calls equal are excluded from FRR.
  system judgment : judge(src vs output). Copies (output == src) score
                    "equal" without a judge call.
  FRR             = P(system direction == gold direction) over pairs
                    with a clear gold direction.

Judge backends: "hf:<model_id>" (local, greedy) or "openai:<model>"
(needs OPENAI_API_KEY; the paper used GPT-4o — use that for the final
table, a local judge for pilot reads). A/B presentation order is
randomized per call (seeded) to cancel position bias.

Usage (miyabi):
    python scripts/judge_feature_realization.py \
        --records runs/prod_gemma_v6/editflow_s3/probe500/records.jsonl \
        --mode thr0.1 --condition true \
        --gold-cache runs/frr/gold_gemma9b.jsonl \
        --judge hf:google/gemma-2-9b-it \
        --out runs/frr/ef_s3_thr01.jsonl
    # pipeline records have no mode key:
    python scripts/judge_feature_realization.py \
        --records runs/prod_gemma_v6/eval_lingualens_final/records.jsonl \
        --mode "" --condition true ...
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

PROMPT = """You are evaluating linguistic feature prominence in sentences.

Linguistic feature: {feature}

Sentence A: {a}
Sentence B: {b}

Which sentence more strongly exhibits the linguistic feature above?
Answer with exactly one letter: "A", "B", or "C" if they exhibit it equally."""


class HFJudge:
    def __init__(self, model_id: str, device: str, dtype: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}[dtype]
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dt).to(device).eval()
        self.device = device
        self.torch = torch

    def __call__(self, prompt: str) -> str:
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False)
        enc = self.tok(text, return_tensors="pt",
                       add_special_tokens=False).to(self.device)
        with self.torch.no_grad():
            gen = self.model.generate(
                **enc, max_new_tokens=6, do_sample=False,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(gen[0, enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)


class OpenAIJudge:
    def __init__(self, model: str):
        import os
        import urllib.request
        self.model = model
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise SystemExit("openai judge needs OPENAI_API_KEY")
        self.urllib = urllib.request
        # GPT-5 / o-series require max_completion_tokens and may reject
        # temperature; GPT-4-era models keep the legacy params.
        self.modern = not self.model.startswith(("gpt-4", "gpt-3"))

    def _call_once(self, body: dict) -> str:
        req = self.urllib.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}",
                     "Content-Type": "application/json"})
        with self.urllib.urlopen(req, timeout=180) as r:
            return json.load(r)["choices"][0]["message"]["content"]

    def __call__(self, prompt: str) -> str:
        body = {"model": self.model,
                "messages": [{"role": "user", "content": prompt}]}
        if self.modern:
            # headroom for models that spend tokens before answering
            body["max_completion_tokens"] = 64
        else:
            body["temperature"] = 0
            body["max_tokens"] = 4
        import urllib.error
        try:
            return self._call_once(body)
        except urllib.error.HTTPError as e:
            if e.code == 400 and not self.modern:
                # legacy params rejected — retry with the modern schema
                body.pop("temperature", None)
                body.pop("max_tokens", None)
                body["max_completion_tokens"] = 64
                return self._call_once(body)
            raise


def parse_answer(text: str):
    t = text.strip().strip('"\'`* ').upper()
    # prefer a letter at the very start ("B", "B.", "B) because ...") —
    # avoids the article-"A" false positive in explanatory outputs
    if t[:1] in ("A", "B", "C") and (len(t) == 1 or not t[1].isalpha()):
        return t[0]
    m = re.search(r"\bANSWER\s*(?:IS|:)?\s*([ABC])\b", t)
    if m:
        return m.group(1)
    # last resort: LAST standalone letter — verbose judges put the verdict
    # at the end, and this avoids the mid-text article-"A" false positive
    ms = re.findall(r"\b([ABC])\b", t)
    return ms[-1] if ms else None


def compare(judge, feature: str, x: str, y: str, rng) -> str:
    """Judge which of x/y exhibits `feature` more.
    Returns 'x' / 'y' / 'equal' (presentation order randomized)."""
    flip = rng.random() < 0.5
    a, b = (y, x) if flip else (x, y)
    ans = parse_answer(judge(PROMPT.format(feature=feature, a=a, b=b)))
    if ans == "C" or ans is None:
        return "equal"
    picked_a = ans == "A"
    if flip:
        return "y" if picked_a else "x"
    return "x" if picked_a else "y"


def bname(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4-8" if n <= 8 else "9+"


def norm(s: str) -> str:
    return " ".join(s.split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--mode", default="thr0.1",
                   help="decode-mode key inside outputs[condition]; "
                        "empty string for pipeline-format records "
                        "(text sits directly under the condition)")
    p.add_argument("--condition", default="true")
    p.add_argument("--gold-cache", required=True,
                   help="jsonl cache of gold (src vs tgt) judgments — "
                        "shared across systems FOR THE SAME JUDGE")
    p.add_argument("--n-ops-ref", default="",
                   help="records.jsonl whose n_ops is joined by idx for "
                        "the bucket breakdown (pipeline records don't "
                        "store n_ops — point this at the EF probe500 "
                        "records)")
    p.add_argument("--judge", default="hf:google/gemma-2-9b-it")
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-dtype", default="bfloat16")
    args = p.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)

    with open(args.records) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    print(f"[frr] {len(recs)} records from {args.records}")
    nref = {}
    if args.n_ops_ref:
        with open(args.n_ops_ref) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    nref[int(r["idx"])] = int(r.get("n_ops", 1))
        print(f"[frr] n_ops joined from {args.n_ops_ref} ({len(nref)})")

    if args.judge.startswith("hf:"):
        judge = HFJudge(args.judge[3:], args.device, args.llm_dtype)
    elif args.judge.startswith("openai:"):
        judge = OpenAIJudge(args.judge[len("openai:"):])
    else:
        raise SystemExit(f"unknown judge spec {args.judge!r}")
    print(f"[frr] judge: {args.judge}")

    gold_path = Path(args.gold_cache)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold = {}
    if gold_path.exists():
        with open(gold_path) as f:
            for line in f:
                if line.strip():
                    g = json.loads(line)
                    gold[int(g["idx"])] = g["gold"]
        print(f"[frr] gold cache: {len(gold)} judgments")
    gf = open(gold_path, "a")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    j = json.loads(line)
                    done[int(j["idx"])] = j
        print(f"[frr] RESUME: {len(done)} judged")
    of = open(out_path, "a")

    rows = []
    for i, rec in enumerate(recs):
        k = int(rec["idx"])
        if k in done:
            rows.append(done[k])
            continue
        ex = ds[k]
        feature = ex.get("feature") or ex.get("categories") or "?"
        # probe/B2 records use src/tgt; pipeline records use source/target
        src = rec.get("src") or rec.get("source")
        tgt = rec.get("tgt") or rec.get("target")
        if src is None or tgt is None:
            raise SystemExit(f"record idx {k} has no src/source field")
        o = rec["outputs"].get(args.condition)
        if o is None:
            continue
        node = o if not args.mode else o.get(args.mode)
        if not isinstance(node, dict) or "text" not in node:
            continue          # mode absent on this record (mixed sweeps)
        out_text = node["text"]

        rng = random.Random(args.seed * 1000003 + k)
        if k not in gold:
            g = compare(judge, feature, src, tgt, rng)   # 'x'=src,'y'=tgt
            gold[k] = {"x": "src", "y": "tgt",
                       "equal": "equal"}[g]
            gf.write(json.dumps({"idx": k, "gold": gold[k],
                                 "feature": feature}) + "\n")
            gf.flush()
        if norm(out_text) == norm(src):
            sysj = "equal"                                # copy
        else:
            s = compare(judge, feature, src, out_text, rng)
            sysj = {"x": "src", "y": "out", "equal": "equal"}[s]
        # gold 'tgt' means the TARGET side exhibits the feature more →
        # realization = the OUTPUT side exhibits it more; gold 'src'
        # means editing should REDUCE prominence → realization = judge
        # still picks src over output.
        if gold[k] == "equal":
            realized = None
        elif gold[k] == "tgt":
            realized = sysj == "out"
        else:
            realized = sysj == "src" and norm(out_text) != norm(src)
        row = {"idx": k, "n_ops": rec.get("n_ops") or nref.get(k, 1),
               "feature": feature, "gold": gold[k], "sys": sysj,
               "copy": float(norm(out_text) == norm(src)),
               "realized": realized}
        rows.append(row)
        of.write(json.dumps(row, ensure_ascii=False) + "\n")
        of.flush()
        if (i + 1) % 25 == 0:
            print(f"[frr] {i + 1}/{len(recs)}")
    of.close()
    gf.close()

    scored = [r for r in rows if r["realized"] is not None]
    n_eq = len(rows) - len(scored)
    frr = (sum(1 for r in scored if r["realized"]) / len(scored)
           if scored else float("nan"))
    print(f"\n[frr] records={len(rows)} gold-equal-excluded={n_eq} "
          f"scored={len(scored)}")
    print(f"[frr] FRR = {frr:.4f}   copy={sum(r['copy'] for r in rows) / max(1, len(rows)):.4f}")
    byb = defaultdict(list)
    for r in scored:
        byb[bname(int(r["n_ops"]))].append(r)
    for b in ("1", "2-3", "4-8", "9+"):
        rs = byb.get(b, [])
        if rs:
            print(f"[frr]   {b}: {sum(1 for r in rs if r['realized']) / len(rs):.4f} (n={len(rs)})")


if __name__ == "__main__":
    main()
