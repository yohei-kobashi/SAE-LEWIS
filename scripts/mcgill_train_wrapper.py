"""
Wrapper for McGill's run_mntp.py / run_simcse.py that adds auto-resume.

Why this exists:

  McGill's training scripts call `trainer.train()` with no arguments.
  HF Trainer does NOT auto-read `TrainingArguments.resume_from_checkpoint`
  in `train()` — only the explicit kwarg counts. So even if our config
  JSON sets resume_from_checkpoint=True, an interrupted run restarts
  from step 0 next invocation. Confirmed empirically with our
  Sheared-LLaMA SimCSE re-run (log showed loss=0.12 at step 50 —
  fresh, not resumed from checkpoint-1000).

What this wrapper does:

  1. Monkey-patches Trainer.train BEFORE importing McGill's script,
     so the patch propagates to MNTPTrainer / SimCSETrainer subclasses
     through method-resolution-order (as long as they don't override
     `train` themselves; they don't in the vendored code we've seen).
  2. Inside the patched train, if resume_from_checkpoint is None,
     scans output_dir for checkpoint-* subdirs. If one exists, sets
     resume_from_checkpoint=True so HF Trainer auto-detects the latest.
  3. Delegates to the McGill script via runpy so the entire code path
     runs as if invoked directly with `python experiments/run_XXX.py`.

Usage (from train_mcgill_llm2vec.sh):
  cd $VENDOR_DIR
  python $REPO_ROOT/scripts/mcgill_train_wrapper.py mntp   $STAGE_CFG
  python $REPO_ROOT/scripts/mcgill_train_wrapper.py simcse $STAGE_CFG
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendored" / "mcgill_llm2vec"


def _install_auto_resume_patch():
    """Wrap Trainer so it auto-resumes when checkpoint-N/ exists.

    Two patches:

      * Trainer.train — if the caller didn't pass resume_from_checkpoint,
        set it to True whenever a checkpoint-*/ dir already exists under
        args.output_dir. HF Trainer then picks the latest and calls
        _load_from_checkpoint.

      * Trainer._load_from_checkpoint — HF Trainer 4.44's implementation
        doesn't recognise peft-only checkpoints (adapter_model.safetensors
        + adapter_config.json with nothing else); it falls through to
        load_sharded_checkpoint which errors with "Can't find a
        checkpoint index". We catch that specific failure and manually
        load the adapter into whichever inner module actually exposes
        load_adapter (base_model on PeftModel, .model on our LLM2Vec-style
        wrappers, or the top-level model itself).
    """
    from transformers import Trainer

    _original_train = Trainer.train
    _original_load_from_checkpoint = Trainer._load_from_checkpoint

    def _peft_fallback_load(target_dir, model):
        """Try every plausible path from `model` down to a load_adapter-capable
        submodule and use it. Returns True on success, False otherwise."""
        candidates = []
        seen = set()
        for m in (model,
                  getattr(model, "model", None),
                  getattr(model, "base_model", None),
                  getattr(getattr(model, "model", None), "model", None)):
            if m is None or id(m) in seen:
                continue
            seen.add(id(m))
            candidates.append(m)

        for cand in candidates:
            if not hasattr(cand, "load_adapter"):
                continue
            adapter_name = (getattr(cand, "active_adapter", None)
                            or "default")
            try:
                cand.load_adapter(target_dir, adapter_name=adapter_name)
                print(f"[wrapper]   loaded adapter via "
                      f"{type(cand).__name__}.load_adapter"
                      f"(adapter={adapter_name!r})", flush=True)
                return True
            except Exception as ex:
                print(f"[wrapper]   {type(cand).__name__}.load_adapter "
                      f"failed: {ex}", flush=True)
                continue
        return False

    def _patched_load_from_checkpoint(self, resume_from_checkpoint, model=None):
        try:
            return _original_load_from_checkpoint(
                self, resume_from_checkpoint, model=model
            )
        except ValueError as e:
            msg = str(e)
            if ("Can't find a checkpoint index" not in msg
                    and "Can't find a valid checkpoint" not in msg):
                raise
            # We're in the peft-only checkpoint fallback path.
            print(f"[wrapper] HF Trainer full-model resume failed:", flush=True)
            print(f"[wrapper]   {msg}", flush=True)
            print(f"[wrapper] falling back to peft adapter load from "
                  f"{resume_from_checkpoint}", flush=True)
            target = model if model is not None else self.model
            if _peft_fallback_load(resume_from_checkpoint, target):
                return  # HF Trainer will handle optimizer/scheduler/rng next
            # Nothing to try — re-raise the original.
            raise

    def _patched_train(self, resume_from_checkpoint=None, *args, **kwargs):
        if resume_from_checkpoint is None or resume_from_checkpoint is False:
            out_dir = getattr(self.args, "output_dir", None)
            if out_dir and os.path.isdir(out_dir):
                for name in sorted(os.listdir(out_dir)):
                    if name.startswith("checkpoint-") and (
                        os.path.isdir(os.path.join(out_dir, name))
                    ):
                        resume_from_checkpoint = True
                        print(
                            f"[wrapper] auto-resume ENABLED: found "
                            f"checkpoint(s) under {out_dir}, HF Trainer "
                            f"will pick the latest",
                            flush=True,
                        )
                        break
                else:
                    print(
                        f"[wrapper] no checkpoint-*/ under {out_dir}, "
                        f"starting fresh",
                        flush=True,
                    )
        return _original_train(
            self, resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs
        )

    Trainer.train = _patched_train
    Trainer._load_from_checkpoint = _patched_load_from_checkpoint
    print("[wrapper] Trainer patched (auto-resume + peft-only fallback)",
          flush=True)


def main():
    if len(sys.argv) < 3:
        print("usage: mcgill_train_wrapper.py {mntp|simcse} CONFIG.json",
              file=sys.stderr)
        sys.exit(2)

    stage = sys.argv[1]
    config_path = sys.argv[2]

    if stage not in {"mntp", "simcse"}:
        print(f"unknown stage: {stage} (expected 'mntp' or 'simcse')",
              file=sys.stderr)
        sys.exit(2)

    script = VENDOR_DIR / "experiments" / f"run_{stage}.py"
    if not script.exists():
        print(f"vendored script not found: {script}", file=sys.stderr)
        sys.exit(2)

    # Ensure the vendored `llm2vec` package resolves first (its
    # experiments/ scripts do `from llm2vec.models import ...`).
    sys.path.insert(0, str(VENDOR_DIR))

    _install_auto_resume_patch()

    # McGill's script uses `sys.argv[1]` as the config file path.
    sys.argv = [str(script), config_path]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
