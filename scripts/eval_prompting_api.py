"""Reference prompting row with a frontier API model as the editor
(user 2026-07-25): same A3' edit-instruction protocol as the
prompting_edit arm in eval_ef_bare.py, but the rewrite is produced by an
OpenAI chat model (default gpt-5.6-luna) instead of frozen
gemma-2-2b-it. Reference value only — not an intervention arm.

Reads the existing prompting_edit records.jsonl so the eval pairs are
IDENTICAL to the published row; re-draws the random-condition feature
with a fixed per-pair seed (statistically equivalent to the prng draw).

Usage (prepost, needs OPENAI_API_KEY):
    python scripts/eval_prompting_api.py \
        --records runs/prod_gemma_v6/a3prime_edit/records.jsonl \
        --a3-prompts runs/a3_prompts/steering_prompts.json \
        --output-dir runs/prod_gemma_v6/api_prompting_l12
    (add --reverse-pairs with the amp-direction records)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class ApiEditor:
    """Chat-completions caller with modern params (max_completion_tokens,
    no temperature — gpt-5.x models reject/ignore the legacy fields)."""

    def __init__(self, model: str, max_tokens: int = 256):
        self.model = model
        self.max_tokens = max_tokens
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise SystemExit("needs OPENAI_API_KEY")

    def __call__(self, prompt: str) -> str:
        body = {"model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": self.max_tokens}
        for attempt in range(4):
            try:
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Authorization": f"Bearer {self.key}",
                             "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    return json.load(r)["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if attempt < 3:        # incl. sporadic 400s (observed
                    time.sleep(5 * (attempt + 1))   # once in 565 calls)
                    continue
                print(f"[api-prompting] giving up: HTTP {e.code} "
                      f"{e.read()[:300]}", flush=True)
                return ""              # scored as non-match
            except (urllib.error.URLError, ConnectionError, TimeoutError,
                    OSError):
                if attempt < 3:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
        return ""                      # network retries exhausted


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True,
                   help="existing prompting_edit records.jsonl (pair list)")
    p.add_argument("--a3-prompts", required=True,
                   help="steering_prompts.json (feature vocabulary for "
                        "the random condition)")
    p.add_argument("--model", default="gpt-5.6-luna")
    p.add_argument("--reverse-pairs", action="store_true",
                   help="enhancement direction (verb 'add')")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    a3 = json.loads(Path(args.a3_prompts).read_text())
    feats = sorted(a3)
    recs = [json.loads(l) for l in open(args.records) if l.strip()]
    verb = "add" if args.reverse_pairs else "remove"

    cache_path = out / f"api_cache_{args.model}.jsonl"
    cache = {}
    if cache_path.exists():
        for l in open(cache_path):
            c = json.loads(l)
            cache[(c["idx"], c["cond"])] = c["text"]
    cache_f = open(cache_path, "a")

    # reasoning models think inside the completion budget — 256 gave
    # ~2-4% empty outputs (07-25); 1024 clears it
    call = ApiEditor(args.model, max_tokens=1024)

    def instr_for(feat, cond, idx):
        if cond == "random":
            others = [f2 for f2 in feats
                      if f2.replace("_", " ") != feat.replace("_", " ")]
            rng = random.Random(args.seed * 1_000_003 + idx)
            feat = others[rng.randrange(len(others))]
        return (f"Rewrite the input sentence to {verb} any "
                f"{feat.replace('_', ' ')}. Output only the rewritten "
                f"sentence.")

    jobs = []
    for r in recs:
        for cond in ("true", "random"):
            if (r["idx"], cond) in cache:
                continue
            jobs.append((r, cond))
    print(f"[api-prompting] {len(recs)} pairs, {len(jobs)} calls to run "
          f"({len(cache)} cached), model={args.model}, verb={verb}")

    def run_one(job):
        r, cond = job
        prompt = (instr_for(r["feature"], cond, int(r["idx"]))
                  + "\n\nInput: " + r["src"])
        text = call(prompt).split("\n")[0].strip()
        return r, cond, text

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r, cond, text in pool.map(run_one, jobs):
            cache[(r["idx"], cond)] = text
            if text:                   # empty = API failure; leave
                cache_f.write(json.dumps(   # uncached so a rerun retries
                    {"idx": r["idx"], "cond": cond, "text": text}) + "\n")
                cache_f.flush()
            done += 1
            if done % 100 == 0:
                print(f"[api-prompting] {done}/{len(jobs)}")
    cache_f.close()

    # ---- score + report (same exact/copy convention as eval_ef_bare) ----
    rows = {}
    out_recs = open(out / "records.jsonl", "w")
    for cond in ("true", "random"):
        n = ex = cp = ne = 0
        for r in recs:
            text = cache.get((r["idx"], cond))
            if text is None:
                continue
            n += 1
            ex += text.strip() == r["tgt"].strip()
            cp += text.strip() == r["src"].strip()
            ne += text.strip() == r["src"].strip()
        rows[cond] = (n, ex / n, cp / n)
    for r in recs:
        out_recs.write(json.dumps(
            {"idx": r["idx"], "src": r["src"], "tgt": r["tgt"],
             "feature": r["feature"],
             "outputs": {c: {"text": cache.get((r["idx"], c))}
                         for c in ("true", "random")}}) + "\n")
    out_recs.close()

    dirn = "enh" if args.reverse_pairs else "abl"
    lines = [f"# API prompting reference — {args.model} ({dirn})", "",
             f"pairs: {len(recs)} | verb: {verb} | seed: {args.seed}", "",
             "| cond | n | exact | copy |", "|---|---|---|---|"]
    for cond in ("true", "random"):
        n, e, c = rows[cond]
        lines.append(f"| {cond} | {n} | {e:.4f} | {c:.4f} |")
    net = rows["true"][1] - rows["random"][1]
    lines += ["", f"net (true - random): **{net:.4f}**"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
