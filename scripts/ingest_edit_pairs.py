"""
Ingest natural edit-pair corpora into the corruption-cache format (v6-C).

The gold-template probe (README §13.6) located the OOD failure in the
editor's feature→CONTENT mapping: synthetic corruption teaches it only
over the corruption distribution. This script converts REAL human edit
pairs into ordinary cache records so they train the same models with no
pipeline change, reusing corruption.py's own machinery end-to-end
(build_pair_sample alignment → symmetric SLOR gate → SAE-diff gate →
blocklist-masked conditioning top-k).

Sources (HF):
  paws      sentence1/sentence2, labeled_final, label==1 only — high-
            overlap paraphrases (word-order + lexical swaps): context-
            underdetermined pairs where only the conditioning decides.
            Both directions emitted.
  coedit    grammarly/coedit — real human edits across gec / neutralize
            / simplification / paraphrase / coherence / clarity; the
            instruction prefix is stripped at the first ': '. Direction:
            src→tgt always; tgt→src ONLY for paraphrase (for the other
            tasks the reverse would supervise producing errors /
            degradations as gold content). (wiki_atomic_edits was the
            original plan but its upstream GCS bucket is gone — 403.)

Records carry bucket="natural_edit" and t_family="PAWS"/"COEDIT", so
the LOFO load-time filters (--exclude-families/--only-families) and the
per-family eval breakdown work on them unchanged — the pilot's control
run trains on the SAME merged cache with these families excluded.

Usage (miyabi):
    python scripts/ingest_edit_pairs.py \
        --source paws --max-pairs 15000 \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --unigram-cache runs/prod_gemma_v4/corruption/unigram.json \
        --blocklist runs/blocklist/blocklist.npy \
        --out-dir runs/prod_gemma_v4/natural_edits_paws
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import corruption                                                  # noqa: E402
from corruption import (                                           # noqa: E402
    Stage, build_pair_sample, finalize_sample, load_unigram,
)
from model import SAEFeatureExtractor, load_causal_gemma           # noqa: E402

SOURCES = {
    "paws": {"t_family": "PAWS"},
    "coedit": {"t_family": "COEDIT"},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=sorted(SOURCES))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--unigram-cache", required=True,
                   help="unigram.json from the existing corruption cache "
                        "(SLOR baseline must match the training cache).")
    p.add_argument("--max-pairs", type=int, default=20000,
                   help="Source pairs to ingest (records ≈ 2x with "
                        "--directions both).")
    p.add_argument("--directions", choices=["auto", "both", "forward"],
                   default="auto",
                   help="auto: per-pair policy (paws + coedit-paraphrase "
                        "both directions, other coedit tasks forward "
                        "only); both/forward: force.")
    p.add_argument("--coedit-tasks", default="all",
                   help="'all' or comma list of coedit task names (gec, "
                        "neutralize, simplification, paraphrase, "
                        "coherence, clarity).")
    p.add_argument("--shard-size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)

    # SAE / models (corruption.py defaults)
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", choices=["jumprelu", "topk"],
                   default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)
    p.add_argument("--k-train", type=int, default=64)
    p.add_argument("--device", default="cuda")

    # Pair filters
    p.add_argument("--sent-min-tokens", type=int, default=5)
    p.add_argument("--sent-max-tokens", type=int, default=64)
    p.add_argument("--max-n", type=int, default=8,
                   help="Drop pairs whose alignment has more than this "
                        "many edit ops (matches the pipeline's edit scale).")

    # Gates — names must match what corruption.finalize_sample reads.
    p.add_argument("--transform-slor-delta", type=float, default=1.5,
                   help="Symmetric fluency gate (both sides are human "
                        "text, so the transform-style gate applies).")
    p.add_argument("--slor-drop-per-op", type=float, default=0.10,
                   help="Unused by the transform-style gate; present "
                        "because gate_thresholds() reads it.")
    p.add_argument("--sae-min-topk-size", type=int, default=10)
    p.add_argument("--sae-min-topk-change", type=int, default=1)
    p.add_argument("--cond-scope", choices=["local", "global"],
                   default="local")
    p.add_argument("--blocklist", default="")
    return p.parse_args()


def iter_pairs(source: str, seed: int, coedit_tasks: str = "all"):
    """Yield (src_text, tgt_text, t_type, reversible)."""
    from datasets import load_dataset
    if source == "paws":
        # Namespaced id: newer huggingface_hub rejects the canonical
        # namespace-less "paws".
        ds = load_dataset("google-research-datasets/paws", "labeled_final",
                          split="train")
        ds = ds.shuffle(seed=seed)
        for r in ds:
            if int(r["label"]) == 1:
                yield r["sentence1"], r["sentence2"], "NE:PAWS", True
    else:
        wanted = (None if coedit_tasks.strip() == "all"
                  else {t.strip() for t in coedit_tasks.split(",")})
        ds = load_dataset("grammarly/coedit", split="train")
        ds = ds.shuffle(seed=seed)
        for r in ds:
            task = r["task"]
            if wanted is not None and task not in wanted:
                continue
            parts = r["src"].split(": ", 1)
            if len(parts) != 2:              # no instruction prefix found
                continue
            a = parts[1]
            yield a, r["tgt"], f"NE:COEDIT-{task.upper()}", \
                task == "paraphrase"


def main():
    args = parse_args()
    args.calibration_mode = False
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = SOURCES[args.source]

    print(f"[ingest] loading tokenizer + causal Gemma from {args.llm2vec_dir}")
    causal_llm, gemma_tok = load_causal_gemma(args.llm2vec_dir)
    causal_llm = causal_llm.to(args.device)
    print("[ingest] loading SAE extractor")
    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer, sae_type=args.sae_type, sae_k=args.sae_k,
    ).to(args.device).eval()
    # Same forward truncation as corruption.py: the SAE reads
    # hidden_states[layer_idx], so layers above it are dead weight.
    _inner = extractor.llm
    if hasattr(_inner, "layers"):
        _n_keep = int(extractor.layer_idx) + 1
        if _n_keep < len(_inner.layers):
            _inner.layers = _inner.layers[:_n_keep]
            _inner.config.num_hidden_layers = _n_keep

    args._blocklist = None
    if args.blocklist:
        _bl = np.load(args.blocklist)
        args._blocklist = torch.as_tensor(np.asarray(_bl, dtype=np.int64))
        print(f"[ingest] blocklist: {len(_bl)} features masked")

    unigram_log, unigram_log_unk = load_unigram(Path(args.unigram_cache))
    print(f"[ingest] unigram baseline: {len(unigram_log)} tokens "
          f"({args.unigram_cache})")

    stage = Stage(
        extractor=extractor, causal_llm=causal_llm, mlm=None,
        gemma_tok=gemma_tok, spacy_nlp=None,
        mask_id=int(gemma_tok.mask_token_id),
        ins_id=int(gemma_tok.convert_tokens_to_ids("[INS]")),
        del_id=int(gemma_tok.convert_tokens_to_ids("[DEL]")),
        pad_id=int(gemma_tok.pad_token_id),
        device=args.device, k_train=int(args.k_train),
        unigram_log=unigram_log, unigram_log_unk=unigram_log_unk,
    )

    written = 0
    shard_idx = 0
    cur = None
    reasons: Counter = Counter()

    def open_shard():
        nonlocal cur, shard_idx
        if cur is not None:
            cur.close()
        path = out_dir / f"shard-{shard_idx:05d}.jsonl.gz"
        cur = gzip.open(path, "wt", encoding="utf-8")
        shard_idx += 1

    open_shard()
    pairs_used = 0
    pair_iter = iter_pairs(args.source, args.seed, args.coedit_tasks)
    for pair_i, (a, b, tt, reversible) in enumerate(pair_iter):
        if pairs_used >= args.max_pairs:
            break
        a, b = (a or "").strip(), (b or "").strip()
        if not a or not b or a.casefold() == b.casefold():
            reasons["identical_or_empty"] += 1
            continue
        n_a = len(gemma_tok(a, add_special_tokens=False).input_ids)
        n_b = len(gemma_tok(b, add_special_tokens=False).input_ids)
        if not (args.sent_min_tokens <= n_a <= args.sent_max_tokens
                and args.sent_min_tokens <= n_b <= args.sent_max_tokens):
            reasons["length"] += 1
            continue

        directions = [(b, a, tt)]                         # edit a → b
        if args.directions == "both" or (args.directions == "auto"
                                         and reversible):
            directions.append((a, b, tt + "/rev"))
        wrote_pair = 0
        for clean, corr, tt in directions:
            ps, why = build_pair_sample(stage, clean, corr)
            if ps is None:
                reasons[f"build_{why}"] += 1
                continue
            if not (1 <= int(ps.get("N_total", 0)) <= args.max_n):
                reasons["n_out_of_range"] += 1
                continue
            pf, why = finalize_sample(
                stage, ps, source_id=f"{args.source}:{pair_i}", args=args,
                calibration_writer=None, transform=True)
            if pf is None:
                reasons[f"gate_{why}"] += 1
                continue
            pf["bucket"] = "natural_edit"
            pf["subset_kind"] = "full"
            pf["n_parent"] = int(pf.get("N_total", 1))
            pf["t_type"] = tt
            pf["t_family"] = src["t_family"]
            cur.write(json.dumps(pf, ensure_ascii=False) + "\n")
            written += 1
            wrote_pair += 1
            if written % args.shard_size == 0:
                open_shard()
        if wrote_pair:
            pairs_used += 1
        if (pair_i + 1) % 1000 == 0:
            print(f"[ingest] scanned={pair_i + 1} used={pairs_used} "
                  f"records={written}")

    cur.close()
    # Drop a trailing empty shard.
    last = out_dir / f"shard-{shard_idx - 1:05d}.jsonl.gz"
    if written % args.shard_size == 0 and last.exists() and written > 0:
        last.unlink()

    meta = {
        "d_sae": int(extractor.d_sae),
        "source": args.source,
        "t_family": src["t_family"],
        "pairs_used": pairs_used,
        "records": written,
        "directions": args.directions,
        "gates": {"transform_slor_delta": args.transform_slor_delta,
                  "sae_min_topk_change": args.sae_min_topk_change},
        "reject_reasons": dict(reasons),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ingest] done: {written} records from {pairs_used} pairs "
          f"→ {out_dir}")
    print(f"[ingest] rejects: {dict(reasons)}")


if __name__ == "__main__":
    main()
