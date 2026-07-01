"""
Diagnose a McGill LoRA adapter checkpoint.

Given a path to an adapter dir (contains adapter_config.json +
adapter_model.safetensors), print:

- List of files in the dir with sizes
- adapter_config.json contents (esp. target_modules and modules_to_save)
- Norm / max / mean_abs stats for lora_A and lora_B tensors
- Verdict per target module: is lora_B still at zero-init?
- Top layers with largest lora_B norms (where the adapter learned the most)

Use it when the bridge script reports FATAL: MNTP/SimCSE merge produced
no weight change — this tells us whether the adapter file itself is
zero, or whether the bridge is looking at the wrong module.

Usage:
    python scripts/diagnose_mcgill_adapter.py runs/mcgill_sheared_repro/mntp/checkpoint-1000
    python scripts/diagnose_mcgill_adapter.py runs/mcgill_sheared_repro/simcse/checkpoint-1000
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch


def _fmt_size(n: int) -> str:
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:6.1f}{unit}"
        n /= 1024
    return f"{n:6.1f}T"


def _load_tensors(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    if path.suffix == ".bin":
        return torch.load(path, map_location="cpu", weights_only=True)
    raise ValueError(f"unknown adapter file suffix: {path}")


def _target_module_of(key: str) -> str:
    for tm in ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj",
               "fc1", "fc2", "lm_head", "embed_tokens"):
        if tm in key:
            return tm
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir")
    ap.add_argument("--zero-thresh", type=float, default=1e-8,
                    help="max_abs below this counts as 'still at zero init'")
    args = ap.parse_args()

    d = Path(args.adapter_dir)
    if not d.is_dir():
        print(f"FATAL: not a directory: {d}", file=sys.stderr)
        sys.exit(2)

    print(f"=== {d} ===")

    # ---- 1. File listing ----
    print("\n[files]")
    for f in sorted(d.iterdir()):
        if f.is_file():
            print(f"  {_fmt_size(f.stat().st_size)}  {f.name}")
        else:
            print(f"  <dir>    {f.name}/")

    # ---- 2. adapter_config.json ----
    ac = d / "adapter_config.json"
    if not ac.exists():
        print("\n[FATAL] no adapter_config.json — this is not a peft adapter dir",
              file=sys.stderr)
        sys.exit(3)

    cfg = json.loads(ac.read_text())
    print("\n[adapter_config.json]")
    for key in ("base_model_name_or_path", "peft_type", "r", "lora_alpha",
                "lora_dropout", "bias", "target_modules", "modules_to_save",
                "task_type"):
        if key in cfg:
            print(f"  {key}: {cfg[key]}")

    target_modules = cfg.get("target_modules") or []
    if isinstance(target_modules, str):
        target_modules = [target_modules]

    # ---- 3. adapter_model.{safetensors,bin} ----
    weight_path = None
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        if (d / name).exists():
            weight_path = d / name
            break
    if weight_path is None:
        print("\n[FATAL] no adapter_model.{safetensors,bin} file", file=sys.stderr)
        sys.exit(4)

    print(f"\n[weights] loading {weight_path.name}")
    tensors = _load_tensors(weight_path)
    print(f"  total tensor entries: {len(tensors)}")

    # ---- 4. Categorize + stats ----
    by_kind = defaultdict(list)  # kind -> [(key, tensor)]
    for k, v in tensors.items():
        if "lora_A" in k:
            by_kind["lora_A"].append((k, v))
        elif "lora_B" in k:
            by_kind["lora_B"].append((k, v))
        elif "modules_to_save" in k or "original_module" in k:
            by_kind["modules_to_save"].append((k, v))
        else:
            by_kind["other"].append((k, v))

    print("\n[per-kind counts]")
    for kind, items in by_kind.items():
        print(f"  {kind}: {len(items)}")

    def stats(t: torch.Tensor) -> tuple[float, float, float]:
        t = t.float()
        return (t.norm().item(), t.abs().max().item(), t.abs().mean().item())

    # ---- 5. lora_B is the key one (init to 0; should be non-0 after training) ----
    if by_kind["lora_B"]:
        print(f"\n[lora_B]  (must be non-zero after training)")
        by_module = defaultdict(list)  # target_module -> list of (key, max_abs)
        for k, v in by_kind["lora_B"]:
            n, mx, mn = stats(v)
            tm = _target_module_of(k)
            by_module[tm].append((k, n, mx))

        print(f"  {'module':<15} {'#tensors':>10} {'#nonzero':>10} {'max_norm':>12} {'max_max_abs':>12}")
        for tm in sorted(by_module):
            entries = by_module[tm]
            nonzero = sum(1 for _, _, mx in entries if mx > args.zero_thresh)
            max_norm = max(n for _, n, _ in entries)
            max_mx = max(mx for _, _, mx in entries)
            marker = " <-- ZERO!" if nonzero == 0 else ""
            print(f"  {tm:<15} {len(entries):>10} {nonzero:>10} {max_norm:>12.4e} {max_mx:>12.4e}{marker}")

        # ---- 6. Top 5 layers with biggest LoRA_B ----
        all_b = [(k, *stats(v)) for k, v in by_kind["lora_B"]]
        top = sorted(all_b, key=lambda r: -r[1])[:5]
        print(f"\n[top-5 lora_B by norm]")
        for k, n, mx, mn in top:
            print(f"  {n:.4e}  max_abs={mx:.4e}  {k}")

    # ---- 7. lora_A (init random, should be small but non-zero) ----
    if by_kind["lora_A"]:
        print(f"\n[lora_A]  (init random, always non-zero)")
        all_a = [stats(v) for _, v in by_kind["lora_A"]]
        norms = [n for n, _, _ in all_a]
        print(f"  mean norm: {sum(norms)/len(norms):.4e}")
        print(f"  max  norm: {max(norms):.4e}")

    # ---- 8. modules_to_save (if any) — these carry full-rank updates ----
    if by_kind["modules_to_save"]:
        print(f"\n[modules_to_save]  (full-rank trained modules e.g. lm_head)")
        for k, v in by_kind["modules_to_save"][:5]:
            n, mx, mn = stats(v)
            print(f"  {k}: shape={tuple(v.shape)} norm={n:.4e} max_abs={mx:.4e}")

    # ---- 9. Verdict ----
    print("\n[verdict]")
    any_lora_b_learned = any(
        stats(v)[1] > args.zero_thresh for _, v in by_kind["lora_B"]
    )
    if not any_lora_b_learned and by_kind["lora_B"]:
        print("  ✗ lora_B is ENTIRELY at zero initialization — the saved adapter")
        print("    has NO trained delta anywhere. Root cause is upstream in")
        print("    how the trainer saves the adapter, or how peft is configured.")
    elif by_kind["lora_B"]:
        print(f"  ✓ Some lora_B has been trained (adapter is not empty).")
        # q_proj-specific check because that's what mcgill_merge_and_expand.py samples
        q_learned = any(
            stats(v)[1] > args.zero_thresh
            for k, v in by_kind["lora_B"] if "q_proj" in k
        )
        if not q_learned:
            print("  ⚠ But q_proj lora_B is at zero — the bridge only probes q_proj,")
            print("    so it declares 'zero delta' even though other modules learned.")
            print("    → Fix: bridge should probe a module that IS a target,")
            print("      or scan multiple modules.")
        else:
            print("  ✓ q_proj lora_B has learned — merge should show a non-zero delta.")
    else:
        print("  ⚠ No lora_B tensors found at all. This is not a standard peft LoRA.")


if __name__ == "__main__":
    main()
