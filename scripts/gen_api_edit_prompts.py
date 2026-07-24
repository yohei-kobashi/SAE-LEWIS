"""A3'-luna prep (user 2026-07-25): have a frontier API model (default
gpt-5.6-luna) WRITE the per-feature edit instructions that are then fed
to frozen gemma-2-2b-it via eval_ef_bare --arms prompting_edit
--a3-edit-prompts. This upgrades the prompt-author of the prompting
baseline (AxBench uses gpt-4o-mini as the author); the executor stays
gemma, so the row remains a prompting baseline on the target model.

Pure stdlib + API — runs anywhere with OPENAI_API_KEY:
    python scripts/gen_api_edit_prompts.py \
        --features runs/a3_prompts/steering_prompts.json \
        --output runs/a3_prompts/edit_prompts_luna.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from eval_prompting_api import ApiEditor                        # noqa: E402

META = """You are writing an instruction for a small language model \
(gemma-2-2b-it, 2B parameters).

The instruction will be shown to the small model followed by a line \
"Input: <sentence>". The small model must rewrite that single input \
sentence to {VERB} the linguistic feature "{CONCEPT}", changing as \
little else as possible.

Write the most effective instruction you can for a 2B model: keep it \
short and imperative; you may add a one-line definition of the feature \
and at most two brief in-context examples if that helps. The \
instruction MUST end by telling the model to output only the rewritten \
sentence and nothing else.

Return only the instruction, without any additional text."""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True,
                   help="json whose keys are the feature names "
                        "(e.g. steering_prompts.json)")
    p.add_argument("--model", default="gpt-5.6-luna")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    feats = sorted(json.loads(Path(args.features).read_text()))
    out_path = Path(args.output)
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    # reasoning models spend completion budget on thinking — 512 left
    # 86/198 entries empty (observed 07-25); 4096 clears it
    call = ApiEditor(args.model, max_tokens=4096)

    for i, feat in enumerate(feats):
        cur = done.setdefault(feat, {})
        for dkey, verb in (("abl", "remove"), ("enh", "add")):
            if cur.get(dkey):
                continue
            prompt = (META.replace("{VERB}", verb)
                          .replace("{CONCEPT}", feat.replace("_", " ")))
            cur[dkey] = call(prompt).strip()
            out_path.write_text(json.dumps(done, indent=1))
        if (i + 1) % 20 == 0:
            print(f"[gen-edit-prompts] {i + 1}/{len(feats)}")
    print(f"[gen-edit-prompts] wrote {out_path} ({len(done)} features)")


if __name__ == "__main__":
    main()
