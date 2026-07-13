"""
Fetch Neuronpedia auto-interp explanations for a Gemma Scope SAE and
write a compact {feature_index: description} JSON (B2 baseline input).

The old /api/explanation/export endpoint is gone; the datasets now live
on S3 as gzipped JSONL batches:
  https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/
      v1/{model}/{sae}/explanations/batch-{i}.jsonl.gz

Usage:
    python scripts/fetch_sae_explanations.py \
        --out runs/np_explanations/gemma-2-2b_12-res-16k.json
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import urllib.request
from pathlib import Path

BASE = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma-2-2b")
    p.add_argument("--sae", default="12-gemmascope-res-16k")
    p.add_argument("--out", required=True)
    p.add_argument("--max-batches", type=int, default=64)
    args = p.parse_args()

    out: dict = {}
    n_batch = 0
    for i in range(args.max_batches):
        url = f"{BASE}/{args.model}/{args.sae}/explanations/batch-{i}.jsonl.gz"
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                break                      # past the last batch
            raise
        n_batch += 1
        with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                idx = str(rec["index"])
                desc = (rec.get("description") or "").strip()
                # first explanation per feature wins (batches may repeat)
                if desc and idx not in out:
                    out[idx] = desc
        print(f"[np-fetch] batch-{i}: total {len(out)} features")
    if not out:
        raise SystemExit(f"no explanations found under "
                         f"{BASE}/{args.model}/{args.sae}/explanations/")
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False))
    print(f"[np-fetch] wrote {path} ({len(out)} features, "
          f"{n_batch} batches)")


if __name__ == "__main__":
    main()
