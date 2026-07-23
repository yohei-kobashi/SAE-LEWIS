"""T2 (user-approved 2026-07-23): IN-CONTEXT z sidecar for the training
corruption cache — the training-side counterpart of improvement (7).

The editor trains on per-pair specs measured on BARE sentences (base
gemma-2-2b) but is evaluated with in-context feature specs measured
inside the repeat prompt on gemma-2-2b-it. This script re-measures every
cached row's z_X_topk / z_X_prime_topk at the SAME operating point as
(7): each sentence embedded in the chat-templated repeat prompt, edit-span
token positions located inside the prompt, max-pool + blocklist + top-64.
All other record fields pass through unchanged so CorruptionDataset and
train_ef_editor read the new cache dirs as-is.

Fallback: if the sentence tokens cannot be located inside the prompt
(find_subseq fails even after the leading-token drop), the ORIGINAL z
entries are kept (counted; expected <<1%).

Resume: per shard (tmp-write + rename = atomic).

Usage (GPU):
    python scripts/make_z_sidecar_ctx.py \
        --cache-dir runs/prod_gemma_v4/corruption_z_l12 \
        --out-dir   runs/prod_gemma_v4/corruption_zctx_l12 \
        --layer 12
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModel, AutoTokenizer   # noqa: E402

from intervener import (REPEAT_PROMPT, chat_prompt_ids,  # noqa: E402
                        find_subseq)
from model import load_sae                               # noqa: E402

LAYER_CFG = {
    4:  ("layer_4/width_16k/average_l0_60/params.npz",
         "runs/blocklist_l4/blocklist.npy"),
    12: ("layer_12/width_16k/average_l0_82/params.npz",
         "runs/blocklist/blocklist.npy"),
    20: ("layer_20/width_16k/average_l0_71/params.npz",
         "runs/blocklist_l20/blocklist.npy"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True,
                   help="existing z sidecar dir (records keep x_token_ids)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--it-model", default="google/gemma-2-2b-it")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--cond-topk", type=int, default=64)
    p.add_argument("--batch-seqs", type=int, default=24)
    p.add_argument("--max-len", type=int, default=192)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def edit_positions(a: List[int], b: List[int]):
    pa, pb = set(), set()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        pa.update(range(i1, max(i1 + 1, i2)))
        pb.update(range(j1, max(j1 + 1, j2)))
    return sorted(p for p in pa if p < len(a)), \
        sorted(p for p in pb if p < len(b))


def main():
    args = parse_args()
    L = args.layer
    sae_path, blk_path = LAYER_CFG[L]
    device = args.device

    tok = AutoTokenizer.from_pretrained(args.it_model)
    llm = AutoModel.from_pretrained(
        args.it_model, torch_dtype=torch.bfloat16).to(device).eval()
    llm.requires_grad_(False)
    sae = load_sae("jumprelu", args.sae_repo, sae_path).to(device).eval()
    blk = torch.as_tensor(np.load(blk_path).astype(np.int64), device=device)

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((cache_dir / "meta.json").read_text())
    meta["z_sidecar_ctx"] = {
        "it_model": args.it_model, "sae_path": sae_path, "layer": L,
        "context": "repeat prompt (chat template, src-first)",
        "source_cache": str(cache_dir), "cond_topk": args.cond_topk}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    @torch.no_grad()
    def batch_z(prompt_ids: List[List[int]]):
        T = max(len(s) for s in prompt_ids)
        ids = torch.zeros(len(prompt_ids), T, dtype=torch.long,
                          device=device)
        mask = torch.zeros(len(prompt_ids), T, dtype=torch.long,
                           device=device)
        for i, s in enumerate(prompt_ids):
            ids[i, :len(s)] = torch.tensor(s, device=device)
            mask[i, :len(s)] = 1
        out = llm(input_ids=ids, attention_mask=mask,
                  output_hidden_states=True, use_cache=False)
        h = out.hidden_states[L + 1]
        zs = []
        for i, s in enumerate(prompt_ids):
            z = sae.encode(h[i, :len(s)].to(sae.W_enc.dtype))
            zs.append(z.float())
        return zs

    def pooled_topk(z, pos):
        dense = (z[pos].max(dim=0).values if pos else
                 z.max(dim=0).values).clone()
        dense[blk] = 0.0
        k = min(args.cond_topk, dense.numel())
        vals, idx = dense.topk(k)
        keep = vals > 0
        return [{"f": int(f), "v": round(float(v), 4)}
                for f, v in zip(idx[keep].tolist(), vals[keep].tolist())]

    def prep_side(token_ids: List[int]):
        """-> (prompt_ids, lo, off, needle) or None if not locatable."""
        text = tok.decode([t for t in token_ids], skip_special_tokens=True)
        text = text.strip()
        if not text:
            return None
        pids = chat_prompt_ids(tok, REPEAT_PROMPT.format(src=text))
        if len(pids) > args.max_len + 40:
            pids = pids[:args.max_len + 40]
        needle = tok(text, add_special_tokens=False).input_ids
        off = 0
        lo = find_subseq(pids, needle)
        if lo is None and len(needle) > 1:
            lo = find_subseq(pids, needle[1:])
            if lo is not None:
                off = 1
        if lo is None:
            return None
        return pids, lo, off, needle

    shards = sorted(cache_dir.glob("shard-*.jsonl.gz"))
    print(f"[ctx-sidecar] {len(shards)} shards, L{L}, "
          f"context=repeat prompt on {args.it_model}", flush=True)
    n_done = n_fb = 0
    for shard in shards:
        outp = out_dir / shard.name
        if outp.exists():
            print(f"[ctx-sidecar] skip {shard.name} (done)", flush=True)
            continue
        recs = []
        try:
            with gzip.open(shard, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            recs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except (OSError, EOFError, gzip.BadGzipFile):
            print(f"[ctx-sidecar] {shard.name} truncated — prefix only",
                  flush=True)
        w = gzip.open(str(outp) + ".tmp", "wt", encoding="utf-8")
        half = max(1, args.batch_seqs // 2)
        for i0 in range(0, len(recs), half):
            chunk = recs[i0:i0 + half]
            plan, prompts = [], []
            for r in chunk:
                a = list(map(int, r["x_token_ids"]))[:args.max_len]
                b = list(map(int, r["x_prime_token_ids"]))[:args.max_len]
                sa, sb = prep_side(a), prep_side(b)
                if sa is None or sb is None:
                    plan.append((r, None))
                    continue
                plan.append((r, (sa, sb, len(prompts))))
                prompts += [sa[0], sb[0]]
            zs = batch_z(prompts) if prompts else []
            for r, info in plan:
                r2 = dict(r)
                if info is None:
                    n_fb += 1          # keep original bare-context z
                else:
                    (pa_ids, lo_a, off_a, na), (pb_ids, lo_b, off_b, nb), \
                        base = info[0], info[1], info[2]
                    ea, eb = edit_positions(na, nb)
                    pos_a = [lo_a + (i - off_a) for i in ea if i >= off_a
                             and lo_a + (i - off_a) < lo_a + len(na) - off_a]
                    pos_b = [lo_b + (i - off_b) for i in eb if i >= off_b
                             and lo_b + (i - off_b) < lo_b + len(nb) - off_b]
                    za, zb = zs[base], zs[base + 1]
                    if not pos_a:
                        pos_a = list(range(lo_a, lo_a + len(na) - off_a))
                    if not pos_b:
                        pos_b = list(range(lo_b, lo_b + len(nb) - off_b))
                    pos_a = [p for p in pos_a if p < za.shape[0]]
                    pos_b = [p for p in pos_b if p < zb.shape[0]]
                    r2["z_X_topk"] = pooled_topk(za, pos_a)
                    r2["z_X_prime_topk"] = pooled_topk(zb, pos_b)
                w.write(json.dumps(r2, ensure_ascii=False) + "\n")
            n_done += len(chunk)
        w.close()
        Path(str(outp) + ".tmp").rename(outp)
        print(f"[ctx-sidecar] {shard.name}: total {n_done} "
              f"(fallback {n_fb})", flush=True)
    print(f"[ctx-sidecar] DONE — {n_done} records, {n_fb} fallbacks")
    print("==================== CTX-SIDECAR-DONE ====================")


if __name__ == "__main__":
    main()
