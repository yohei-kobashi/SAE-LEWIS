"""Debug the AxBench L20 all-zero result: is the generation off-topic
because of the steering, or broken even WITHOUT any hook?

Generates concept 0's first 3 instructions under:
  (a) no hook at all
  (b) sae steering factor 0.2 (the sweep's minimum, which won selection
      everywhere in the zero run)
  (c) sae steering factor 1.0
and prints everything. Greedy AND temp-1.0 sampling for (a) to separate
sampling weirdness from prompt weirdness.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import load_sae
from scripts.eval_axbench_repro_gen import (AdditionHook, CANONICAL_16K,
                                            stable_hash)

AX = Path("third_party/axbench")     # run from the repo root
CFG = "prod_2b_l20_v1"
LAYER = 20
DEVICE = "cuda"


def main():
    ax = AX
    cfg_dir = ax / "axbench" / "concept500" / CFG
    meta_rows = [json.loads(l) for l in
                 open(cfg_dir / "generate" / "metadata.jsonl")]
    concept = meta_rows[0]["concept"]
    vanilla = int(meta_rows[0]["ref"].split("/")[-1])
    alpaca = pd.read_json(ax / "alpaca_eval.json")
    instr = alpaca.sample(10, random_state=0)["instruction"].tolist()[:3]
    stage_a = json.loads(
        (Path("runs/axbench_repro") / f"stage_a_{CFG}.json").read_text())
    max_act = float(stage_a["0"]["max_act_vanilla"])
    print(f"concept: {concept}\nvanilla latent: {vanilla} "
          f"max_act: {max_act:.1f}")

    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        torch_dtype=torch.bfloat16).to(DEVICE).eval()
    sae = load_sae("jumprelu", "google/gemma-scope-2b-pt-res",
                   CANONICAL_16K[LAYER], sae_k=None).to(DEVICE).eval()
    hook = AdditionHook(sae.W_dec)
    model.model.layers[LAYER].register_forward_hook(hook)
    print(f"W_dec[vanilla] norm: {float(sae.W_dec[vanilla].norm()):.3f}")

    def chat_wrap(instruction):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=True, add_generation_prompt=True)[1:]
        return tok.decode(ids)

    tok.padding_side = "left"

    @torch.no_grad()
    def gen(prompts, sample, seed=0):
        enc = tok(prompts, return_tensors="pt",
                  padding=True).to(DEVICE)
        torch.manual_seed(seed)
        out = model.generate(
            **enc, max_new_tokens=96, do_sample=sample,
            temperature=1.0 if sample else None,
            pad_token_id=tok.pad_token_id or tok.eos_token_id)
        n_in = enc["input_ids"].shape[1]
        return [tok.decode(o[n_in:], skip_special_tokens=True) for o in out]

    wrapped = [chat_wrap(q) for q in instr]
    for tag, enabled, factor, sample in (
            ("NOHOOK-greedy", False, 0.0, False),
            ("NOHOOK-temp1", False, 0.0, True),
            ("SAE-f0.2-temp1", True, 0.2, True),
            ("SAE-f1.0-temp1", True, 1.0, True)):
        hook.enabled = enabled
        if enabled:
            hook.vec = factor * max_act * sae.W_dec[vanilla].to(DEVICE)
        seed = stable_hash("sae", 0, factor, 0) % (2**31)
        texts = gen(wrapped, sample, seed)
        print(f"\n========== {tag} ==========")
        for q, t in zip(instr, texts):
            print(f"[Q] {q[:70]}")
            print(f"[A] {t[:250]!r}\n")


if __name__ == "__main__":
    main()
