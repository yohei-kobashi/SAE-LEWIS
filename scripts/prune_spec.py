"""
P-O: per-instance minimal conditioning set — "WHICH activations carry THIS
minimal pair?"

Motivation (user, 2026-07-17): the point of using an SAE is to identify the
activations that correspond to the feature. Our findings so far say there is
no single activation per minimal pair — the command is a combination — but
that does not mean the k=32 set is irreducible PER INSTANCE. M0 refuted
GLOBAL narrowing (k32->k8 degrades exact and FRR together); this measures
INSTANCE narrowing: on pairs the editor already solves exactly, prune the
32-feature spec to the smallest subset that still yields the exact edit.

The hypothesis this tests, and the mechanistic story it would complete:
    |S_min| is SMALL per instance, but WHICH features it contains varies
    across instances of the same phenomenon.
If so, that is the explanation for the whole phenomenon-level failure family
(P-B collapse, P-J's weak r=3 / null r=1, AxBench's causally-empty single
latent): three features per instance are enough, but no FIXED three serve a
phenomenon — aggregation averages away instance identity.

Method (on pairs where the full-spec decode is exact; minimality is only
well-defined where the command succeeded):
  1. order the spec's features by |delta| descending (amp and sup merged);
  2. binary-search the smallest magnitude-prefix whose decode is still
     exact (~log2(32) decodes; endpoint verified, not assumed);
  3. backward-eliminate the survivors smallest-magnitude-first (~p* more
     decodes) -> S_min, verified exact once more at the end.
Decode = the champion configuration exactly (thr0.1, 48 steps, lens λ=1,
same suppress list), via scripts.editflow_probe.decode_flow.

Output: per-pair S_min with Neuronpedia labels + a report with
  - the |S_min| distribution (median vs the k=32 default),
  - per-phenomenon: mean |S_min|, the UNION size across its pruned pairs,
    within-phenomenon pairwise Jaccard of S_min, top recurring features.
  union >> |S_min| with low Jaccard = the aggregation story, measured.

Usage (interact-g):
    python scripts/prune_spec.py \
        --editflow-ckpt runs/prod_gemma_v6/editflow_s3/editflow-final.pt \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --exact-from runs/prod_gemma_v6/ksweep500/records.jsonl:k32 \
        --explanations runs/np_explanations/gemma-2-2b_12-res-16k.json \
        --output-dir runs/prod_gemma_v6/prune_spec --max-pairs 60

--effector steer (P-O INTERVENTION version, 2026-07-17 reframing): same
pruning algorithm, but the effector is B3's steering intervention — dvec =
za@W_dec − zs@W_dec added at layers[12] of gemma-2-2b-it (hook + rewrite
copied from eval_clamp_baseline.py, greedy so deterministic). This makes
S_min a CAUSAL minimal set: the smallest activation subset whose
INTERVENTION still executes the exact minimal-pair edit. Under the 🔵
reframing this is the primary instrument (localization spectrum of the
causal claim); the EF version above is its information-side counterpart.
    python scripts/prune_spec.py --effector steer \
        --llm2vec-dir runs/mcgill_gemma_repro_3k/final \
        --exact-from runs/prod_gemma_v6/steer_baseline500/records.jsonl:steer0.5 \
        --k-amp 64 --k-sup 64 \
        --output-dir runs/prod_gemma_v6/prune_spec_steer --max-pairs 60
(--k-amp/--k-sup 64 = the defaults the steer baseline ran with; the spec
must be rebuilt identically or step 0's reproduce guard rejects the pairs.)
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from editflow import load_editflow_from_checkpoint            # noqa: E402
from eval_lingualens import (                                 # noqa: E402
    diff_intervention, edit_char_ranges, local_pool_topk, sae_z_with_offsets,
)
from model import SAEFeatureExtractor, load_sae_w_dec         # noqa: E402
from scripts.editflow_probe import decode_flow                # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--effector", choices=["ef", "steer"], default="ef",
                   help="ef = conditioning decode (information-side); "
                        "steer = B3 intervention rewrite (causal-side)")
    p.add_argument("--it-model", default="google/gemma-2-2b-it")
    p.add_argument("--steer-alpha", type=float, default=0.5)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--editflow-ckpt", default="",
                   help="required for --effector ef")
    p.add_argument("--llm2vec-dir", required=True)
    p.add_argument("--exact-from", required=True,
                   help="records.jsonl:mode — pairs whose `mode` output was "
                        "exact define the pruning population")
    p.add_argument("--llm", default="google/gemma-2-2b")
    p.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--sae-path",
                   default="layer_12/width_16k/average_l0_82/params.npz")
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--sae-type", default="jumprelu")
    p.add_argument("--sae-k", type=int, default=None)
    p.add_argument("--dataset", default="THU-KEG/LinguaLens-Data")
    p.add_argument("--language", default="English")
    p.add_argument("--sample-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--blocklist", default="runs/blocklist/blocklist.npy")
    p.add_argument("--pool-topk", type=int, default=64)
    p.add_argument("--k-amp", type=int, default=32)
    p.add_argument("--k-sup", type=int, default=32)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--thr", type=float, default=0.1)
    p.add_argument("--w-max", type=float, default=20.0)
    p.add_argument("--max-ops-per-step", type=int, default=8)
    p.add_argument("--max-grow", type=int, default=24)
    p.add_argument("--steer-lambda", type=float, default=1.0)
    p.add_argument("--max-pairs", type=int, default=60,
                   help="pruning costs ~15-20 decodes/pair; subsample")
    p.add_argument("--explanations", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def spec_from_entries(entries, d_sae):
    """entries: list of (fid, side, mag); side 'amp'/'sup'."""
    za = torch.zeros(d_sae)
    zs = torch.zeros(d_sae)
    for fid, side, mag in entries:
        (za if side == "amp" else zs)[fid] = mag
    return za, zs


def main():
    args = parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- pairs where the full spec already wins ---------------------------
    path, mode = args.exact_from.rsplit(":", 1)
    exact_idx = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            node = (r.get("outputs", {}).get("true") or {}).get(mode)
            if isinstance(node, dict) and float(node.get("exact", 0)) >= 1.0:
                exact_idx.append(int(r["idx"]))
    print(f"[prune] {len(exact_idx)} exact-hit pairs in {path} ({mode})")

    ds = load_dataset(args.dataset, split="train")
    if args.language and args.language.lower() != "all":
        ds = ds.filter(lambda r: r["language"] == args.language)
    rng = np.random.default_rng(args.seed)
    sample = set(rng.choice(len(ds), size=min(args.sample_size, len(ds)),
                            replace=False).tolist())
    pool = sorted(set(exact_idx) & sample)
    if args.max_pairs and len(pool) > args.max_pairs:
        pool = list(np.random.default_rng(args.seed + 7).choice(
            pool, size=args.max_pairs, replace=False))
        pool.sort()
    print(f"[prune] pruning {len(pool)} pairs (~15-20 decodes each)")

    extractor = SAEFeatureExtractor(
        llm_name=args.llm, sae_repo=args.sae_repo, sae_path=args.sae_path,
        sae_layer=args.sae_layer, sae_type=args.sae_type, sae_k=args.sae_k,
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.llm2vec_dir)
    suppress = [tokenizer.mask_token_id,
                tokenizer.convert_tokens_to_ids("[INS]"),
                tokenizer.convert_tokens_to_ids("[SEP]"),
                tokenizer.convert_tokens_to_ids("[DEL]"),
                tokenizer.bos_token_id, tokenizer.eos_token_id,
                tokenizer.pad_token_id]
    suppress = sorted({int(s) for s in suppress if s is not None})
    dtype = torch.bfloat16
    w_dec = load_sae_w_dec(args.sae_repo, args.sae_path).to(args.device)
    if args.effector == "steer":
        # B3's effector, copied from eval_clamp_baseline.py: SteerHook on
        # layers[sae_layer] of the -it model, neutral Rewrite prompt,
        # greedy (deterministic — pruning needs stable decodes).
        from transformers import AutoModelForCausalLM
        from scripts.eval_clamp_baseline import (PROMPT, SteerHook,
                                                 extract_sentence)
        it_tok = AutoTokenizer.from_pretrained(args.it_model)
        it_model = AutoModelForCausalLM.from_pretrained(
            args.it_model, torch_dtype=dtype).to(args.device).eval()
        hook = SteerHook()
        it_model.model.layers[args.sae_layer].register_forward_hook(hook)
        model, head_w = None, None

        @torch.no_grad()
        def steer_rewrite(src: str) -> str:
            text_in = it_tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(src=src)}],
                add_generation_prompt=True, tokenize=False)
            enc = it_tok(text_in, return_tensors="pt",
                         add_special_tokens=False).to(args.device)
            gen = it_model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=it_tok.pad_token_id or it_tok.eos_token_id)
            return extract_sentence(
                it_tok.decode(gen[0, enc["input_ids"].shape[1]:],
                              skip_special_tokens=True), src)
    else:
        if not args.editflow_ckpt:
            raise SystemExit("--editflow-ckpt is required for --effector ef")
        model = load_editflow_from_checkpoint(
            args.llm2vec_dir, args.editflow_ckpt, dtype=dtype,
        ).to(args.device).eval()
        head_w = model.lm_head.weight.detach().float().to(args.device)
    blk = None
    if args.blocklist and Path(args.blocklist).exists():
        blk = torch.as_tensor(np.asarray(np.load(args.blocklist),
                                         dtype=np.int64))
    expl = {}
    if args.explanations and Path(args.explanations).exists():
        expl = json.loads(Path(args.explanations).read_text())

    def lens_bias(za_v, zs_v):
        d = (za_v.to(args.device) - zs_v.to(args.device)) @ w_dec
        lb = head_w @ d
        s = lb.std()
        return None if float(s) < 1e-6 else args.steer_lambda * lb / (s + 1e-8)

    n_decodes = 0

    def decode_with(entries, src_ids, src, d_sae, k):
        nonlocal n_decodes
        n_decodes += 1
        za, zs = spec_from_entries(entries, d_sae)
        if args.effector == "steer":
            # dvec construction verbatim from eval_clamp_baseline (B3)
            W = w_dec.float()
            hook.dvec = (za.to(args.device).float() @ W
                         - zs.to(args.device).float() @ W)
            hook.alpha = float(args.steer_alpha)
            hook.pos_mask = None
            hook.enabled = True
            out = steer_rewrite(src)
            hook.enabled = False
            return out
        srng = random.Random(args.seed * 1000003 + int(k))
        out_ids = decode_flow(
            model, src_ids, za.to(args.device), zs.to(args.device),
            steps=args.steps, device=args.device, mode="thr",
            thr_frac=args.thr, w_max=args.w_max,
            lens_bias=lens_bias(za, zs), cfg_scale=1.0,
            max_ops_per_step=args.max_ops_per_step, max_grow=args.max_grow,
            suppress_ids=suppress, rng=srng)
        return tokenizer.decode(out_ids, skip_special_tokens=True)

    norm = lambda s: " ".join(s.split())                       # noqa: E731
    results, part = [], out_dir / "records.partial.jsonl"
    done = set()
    if part.exists():
        with open(part) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add(int(r["idx"]))
                    results.append(r)
        print(f"[prune] RESUME: {len(done)} pairs done")
    pf = open(part, "a")

    for k in pool:
        if k in done:
            continue
        ex = ds[int(k)]
        src, tgt = ex["sentence1"], ex["sentence2"]
        feature = ex.get("feature") or "?"
        src_ids = tokenizer(src, add_special_tokens=True).input_ids
        tgt_norm = norm(tgt)

        with torch.no_grad():
            s_off, z_s = sae_z_with_offsets(extractor, src, args.device)
            t_off, z_t = sae_z_with_offsets(extractor, tgt, args.device)
            om_s = [tuple(o) for o in tokenizer(
                src, add_special_tokens=True,
                return_offsets_mapping=True)["offset_mapping"]]
            om_t = [tuple(o) for o in tokenizer(
                tgt, add_special_tokens=True,
                return_offsets_mapping=True)["offset_mapping"]]
            opcodes = difflib.SequenceMatcher(
                None, src_ids,
                tokenizer(tgt, add_special_tokens=True).input_ids,
                autojunk=False).get_opcodes()
            sr, tr = edit_char_ranges(opcodes, om_s, om_t)
            z_src = local_pool_topk(z_s, s_off, sr, args.pool_topk, blk)
            z_tgt = local_pool_topk(z_t, t_off, tr, args.pool_topk, blk)
        za_t, zs_t = diff_intervention(z_src, z_tgt, args.k_amp, args.k_sup)
        d_sae = za_t.shape[0]

        entries = ([(int(i), "amp", float(za_t[i]))
                    for i in torch.nonzero(za_t > 0).flatten()]
                   + [(int(i), "sup", float(zs_t[i]))
                      for i in torch.nonzero(zs_t > 0).flatten()])
        entries.sort(key=lambda e: -e[2])          # |delta| descending
        n_full = len(entries)

        # 0) reproduce the exact hit with the full spec (env-drift guard)
        if norm(decode_with(entries, src_ids, src, d_sae, k)) != tgt_norm:
            row = {"idx": int(k), "feature": feature, "n_full": n_full,
                   "status": "full-spec no longer exact (skipped)"}
            results.append(row)
            pf.write(json.dumps(row) + "\n")
            pf.flush()
            continue

        # 1) minimal magnitude-prefix via binary search (verified endpoint)
        lo, hi = 1, n_full
        while lo < hi:
            mid = (lo + hi) // 2
            if norm(decode_with(entries[:mid], src_ids, src, d_sae, k)) \
                    == tgt_norm:
                hi = mid
            else:
                lo = mid + 1
        prefix = entries[:lo]
        p_star = lo

        # 2) backward elimination, smallest magnitude first
        s_min = list(prefix)
        for e in sorted(prefix, key=lambda x: x[2]):
            if len(s_min) == 1:
                break
            trial = [x for x in s_min if x is not e]
            if norm(decode_with(trial, src_ids, src, d_sae, k)) == tgt_norm:
                s_min = trial
        # 3) final verification
        assert norm(decode_with(s_min, src_ids, src, d_sae, k)) == tgt_norm

        row = {"idx": int(k), "feature": feature, "n_full": n_full,
               "p_prefix": p_star, "n_min": len(s_min), "status": "ok",
               "s_min": [{"f": f, "side": sd, "mag": round(m, 4),
                          "label": expl.get(str(f), "")[:80]}
                         for f, sd, m in s_min]}
        results.append(row)
        pf.write(json.dumps(row, ensure_ascii=False) + "\n")
        pf.flush()
        print(f"[prune] idx {k} ({feature}): {n_full} -> prefix {p_star} "
              f"-> S_min {len(s_min)}")
    pf.close()

    ok = [r for r in results if r.get("status") == "ok"]
    with open(out_dir / "records.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    L = ["# P-O: per-instance minimal conditioning sets", "",
         f"pairs pruned: {len(ok)} (of {len(results)} attempted); effector = "
         f"{args.effector}"
         + (f" (steer alpha {args.steer_alpha:g}, greedy rewrite)"
            if args.effector == "steer" else
            f" (champion thr{args.thr:g}, {args.steps} steps)")
         + f"; total decodes {n_decodes}", ""]
    if ok:
        import statistics as st
        sizes = [r["n_min"] for r in ok]
        L += [f"**|S_min|: median {st.median(sizes):g}, mean "
              f"{st.mean(sizes):.1f}, min {min(sizes)}, max {max(sizes)} — "
              f"vs the k=32 default (mean n_full "
              f"{st.mean([r['n_full'] for r in ok]):.1f}).**", ""]
        by_ph = defaultdict(list)
        for r in ok:
            by_ph[r["feature"]].append(r)
        L += ["| phenomenon | pairs | mean \\|S_min\\| | union | mean "
              "pairwise Jaccard | top recurring feature |",
              "|---|---|---|---|---|---|"]
        for ph, rows in sorted(by_ph.items(), key=lambda kv: -len(kv[1])):
            if len(rows) < 2:
                continue
            sets = [frozenset(e["f"] for e in r["s_min"]) for r in rows]
            union = set().union(*sets)
            js = [len(a & b) / len(a | b)
                  for i, a in enumerate(sets) for b in sets[i + 1:]]
            cnt = defaultdict(int)
            for s_ in sets:
                for f_ in s_:
                    cnt[f_] += 1
            top_f, top_n = max(cnt.items(), key=lambda kv: kv[1])
            lab = expl.get(str(top_f), "")[:50]
            L.append(f"| {ph} | {len(rows)} | "
                     f"{st.mean([r['n_min'] for r in rows]):.1f} | "
                     f"{len(union)} | {st.mean(js):.2f} | "
                     f"{top_f} ({top_n}/{len(rows)}) {lab} |")
        L += ["", "Reading: small |S_min| with union >> |S_min| and low "
              "Jaccard = the command is small PER INSTANCE but instance-"
              "specific — the mechanistic explanation for why phenomenon-"
              "level selections (FRC r=3, AUROC r=1) fail: no fixed small "
              "set serves a phenomenon. High Jaccard phenomena are the "
              "opposite: a stable causal core exists and could seed a "
              "causally-validated dictionary (better than FRC/AUROC)."]
    report = "\n".join(L)
    print("\n" + report)
    (out_dir / "report.md").write_text(report + "\n")
    print(f"\n[prune] wrote {out_dir}/report.md")


if __name__ == "__main__":
    main()
