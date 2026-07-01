"""
Level-1 diagnostic: is our MNTP training's stop_after_n_steps enough?

Reads `trainer_state.json` from a McGill MNTP output dir (either
the top-level `output_dir` or one of its `checkpoint-N/` subdirs)
and prints the eval_loss / eval_accuracy trajectory across all
recorded eval events. Then computes a simple verdict:

  - "still-climbing" — recent eval-accuracy deltas are >= 40% of
    the early deltas, so training is far from saturated. Longer
    stop_after_n_steps would very likely help.

  - "slowing"       — recent deltas are 15–40% of the early deltas.
    Diminishing returns; 2-3x more steps might squeeze another
    couple of accuracy points but the ROI is dropping.

  - "plateau"       — recent deltas are < 15% of early deltas or
    trending toward zero. Extra steps unlikely to pay for
    themselves; current run captures most of the achievable MLM
    accuracy.

Usage:
    python scripts/analyze_mntp_eval_curve.py runs/mcgill_sheared_repro/mntp
    python scripts/analyze_mntp_eval_curve.py \\
        runs/mcgill_gemma_repro_3k/mntp/checkpoint-1000

To sanity-check against the *_3k SimCSE-only comparison, pass the MNTP
dir for both the 1k and 3k runs and eyeball whether the accuracy curves
converge or one is markedly higher at step 1000.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _find_trainer_state(root: Path) -> Path:
    """Locate the most informative trainer_state.json under `root`.

    Priority order:
      1. `root/trainer_state.json` (the top-level save at end of training —
         contains the full log_history including everything below).
      2. Highest-N `root/checkpoint-N/trainer_state.json` — mid-training
         snapshots have the history up to that step.
    """
    top = root / "trainer_state.json"
    if top.exists():
        return top

    best_n, best = -1, None
    for cd in root.glob("checkpoint-*"):
        if not (cd / "trainer_state.json").exists():
            continue
        n_str = cd.name.split("-", 1)[1]
        if not n_str.isdigit():
            continue
        n = int(n_str)
        if n > best_n:
            best_n, best = n, cd / "trainer_state.json"
    if best is None:
        raise SystemExit(f"[analyze] no trainer_state.json under {root}")
    return best


def _extract_eval_events(state: dict) -> list[dict]:
    """Return list of {step, eval_loss, eval_accuracy} for each eval event."""
    events = []
    for entry in state.get("log_history", []):
        if "eval_loss" not in entry:
            continue
        step = entry.get("step", entry.get("global_step"))
        events.append({
            "step": step,
            "eval_loss": entry["eval_loss"],
            "eval_accuracy": entry.get("eval_accuracy"),
        })
    events.sort(key=lambda e: e["step"])
    return events


def _verdict(events: list[dict], key: str) -> tuple[str, dict]:
    """Compare early vs late eval deltas for `key` (e.g. 'eval_accuracy')."""
    if len(events) < 4:
        return "too-few-events", {}
    vals = [e[key] for e in events if e.get(key) is not None]
    if len(vals) < 4:
        return "no-metric", {}

    n = len(vals)
    third = max(1, n // 3)

    # Early gain: from event 0 to event `third-1`.
    early_delta = vals[third - 1] - vals[0]
    # Late gain: from event -(third) to -1.
    late_delta = vals[-1] - vals[-third]

    # Guard against decreases (shouldn't happen for accuracy, could for loss).
    early_mag = abs(early_delta)
    late_mag = abs(late_delta)
    if early_mag == 0:
        ratio = float("inf") if late_mag > 0 else 0.0
    else:
        ratio = late_mag / early_mag

    info = {
        "n_events": n,
        "third": third,
        "early_from": vals[0],
        "early_to": vals[third - 1],
        "early_delta": early_delta,
        "late_from": vals[-third],
        "late_to": vals[-1],
        "late_delta": late_delta,
        "late_over_early": ratio,
    }
    if ratio >= 0.40:
        label = "still-climbing"
    elif ratio >= 0.15:
        label = "slowing"
    else:
        label = "plateau"
    return label, info


def _asciibar(x: float, lo: float, hi: float, width: int = 40) -> str:
    if hi == lo:
        return "|"
    frac = (x - lo) / (hi - lo)
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "|" + "#" * filled + " " * (width - filled) + "|"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir",
                    help="MNTP output dir (contains trainer_state.json OR "
                         "checkpoint-N/ dirs each with one).")
    ap.add_argument("--metric", default="eval_accuracy",
                    choices=["eval_accuracy", "eval_loss"],
                    help="Which metric to base the verdict on.")
    args = ap.parse_args()

    root = Path(args.run_dir).resolve()
    if not root.exists():
        raise SystemExit(f"[analyze] {root} does not exist")

    ts_path = _find_trainer_state(root)
    print(f"[analyze] source: {ts_path.relative_to(root.parent) if ts_path.is_relative_to(root.parent) else ts_path}")
    state = json.loads(ts_path.read_text())
    events = _extract_eval_events(state)
    if not events:
        raise SystemExit("[analyze] no eval_* entries in log_history")

    print(f"[analyze] {len(events)} eval event(s), "
          f"step range {events[0]['step']} … {events[-1]['step']}")
    print()

    # ---- Table + ASCII bar ---------------------------------------------
    losses = [e["eval_loss"] for e in events]
    accs = [e["eval_accuracy"] for e in events if e["eval_accuracy"] is not None]
    lo_l, hi_l = min(losses), max(losses)
    lo_a, hi_a = (min(accs), max(accs)) if accs else (0.0, 1.0)

    hdr = f"{'step':>6} | {'eval_loss':>9}  {'Δ':>7} | " \
          f"{'eval_acc':>8}  {'Δ':>7} | {'accuracy bar (min→max)':<44}"
    print(hdr)
    print("-" * len(hdr))
    prev_loss = None
    prev_acc = None
    for e in events:
        step = e["step"]
        loss = e["eval_loss"]
        acc = e["eval_accuracy"]
        dloss = f"{loss - prev_loss:+.3f}" if prev_loss is not None else "  —  "
        dacc = (f"{acc - prev_acc:+.3f}"
                if (acc is not None and prev_acc is not None) else "  —  ")
        acc_str = f"{acc:.4f}" if acc is not None else "   —   "
        bar = _asciibar(acc, lo_a, hi_a) if acc is not None else "|" + " " * 40 + "|"
        print(f"{step:>6} | {loss:>9.4f}  {dloss:>7} | "
              f"{acc_str:>8}  {dacc:>7} | {bar}")
        prev_loss = loss
        prev_acc = acc

    # ---- Verdict --------------------------------------------------------
    print()
    label, info = _verdict(events, args.metric)
    print(f"[analyze] verdict (based on {args.metric}): {label}")
    if info:
        print(f"[analyze]   early split : first {info['third']} eval events")
        print(f"[analyze]     value     : {info['early_from']:.4f} → {info['early_to']:.4f}   "
              f"Δ = {info['early_delta']:+.4f}")
        print(f"[analyze]   late split  : last  {info['third']} eval events")
        print(f"[analyze]     value     : {info['late_from']:.4f} → {info['late_to']:.4f}   "
              f"Δ = {info['late_delta']:+.4f}")
        print(f"[analyze]   late / early: {info['late_over_early']:.2f}")

    # ---- Recommendation ------------------------------------------------
    print()
    if label == "still-climbing":
        print("[analyze] RECOMMEND: increase stop_after_n_steps 3–5×. Training")
        print("           has not converged — the same LR + data give clear")
        print("           room to improve MNTP quality.")
    elif label == "slowing":
        print("[analyze] RECOMMEND: increase stop_after_n_steps 2–3× if you")
        print("           need max MNTP quality; skip the extra spend if the")
        print("           downstream metric (STS-B, probing F1) is already at")
        print("           the target.")
    elif label == "plateau":
        print("[analyze] RECOMMEND: keep current stop_after_n_steps. Extra")
        print("           MLM training is unlikely to move downstream metrics.")
        print("           Investigate SimCSE, data, or seed variance instead.")
    else:
        print(f"[analyze] verdict inconclusive ({label}); need more eval events.")


if __name__ == "__main__":
    main()
