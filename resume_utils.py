"""
Shared progress + resume utilities for SAE-LEWIS training scripts.

Each step-based training script (train_tagger, train_editor_phaseA,
train_length_head) saves a model checkpoint as `<prefix>-step{N}.pt` at
every `--save-steps`. We sidecar a `<prefix>-step{N}.state.pt` file
with the optimizer + scheduler + step counter + RNG state so the loop
can pick up exactly where it left off.

Resume is on by default in every script that imports this. Pass
`--no-resume` to ignore existing checkpoints and start fresh.

The progress bar (tqdm) is wrapped around `range(start_step, max_steps)`
so the wall-clock estimate is correct after resume.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


_CKPT_RE = re.compile(r"^(?P<prefix>.+)-step(?P<step>\d+)\.pt$")


def find_latest_ckpt(out_dir: Path, prefix: str) -> Optional[Tuple[Path, int]]:
    """Locate the highest-step checkpoint matching `<prefix>-step{N}.pt`.

    Returns (path, N) or None if no checkpoint exists.
    """
    best: Optional[Tuple[Path, int]] = None
    for p in out_dir.glob(f"{prefix}-step*.pt"):
        m = _CKPT_RE.match(p.name)
        if not m or m.group("prefix") != prefix:
            continue
        n = int(m.group("step"))
        if best is None or n > best[1]:
            best = (p, n)
    return best


def state_path_for(ckpt_path: Path) -> Path:
    """Sidecar state path used by save_train_state / load_train_state."""
    return ckpt_path.with_suffix(".state.pt")


def save_train_state(
    ckpt_path: Path,
    optim: torch.optim.Optimizer,
    sched,
    step: int,
    extra: Optional[Dict] = None,
) -> None:
    """Save optimizer / scheduler / step + RNG to `<ckpt>.state.pt`.

    Call this right AFTER the model's own `save(ckpt_path)`.
    """
    state = {
        "optim": optim.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "step":  int(step),
        "torch_rng_state":   torch.get_rng_state(),
        "numpy_rng_state":   np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if extra:
        for k, v in extra.items():
            state[k] = v
    torch.save(state, state_path_for(ckpt_path))


def load_train_state(
    ckpt_path: Path,
    optim: torch.optim.Optimizer,
    sched,
    device: str = "cpu",
) -> Optional[int]:
    """Restore optimizer / scheduler / step + RNG from sidecar state file.

    Returns the saved step counter, or None if the sidecar is missing
    (caller should fall back to model-only resume in that case).
    """
    sp = state_path_for(ckpt_path)
    if not sp.exists():
        return None
    state = torch.load(sp, map_location=device, weights_only=False)
    try:
        optim.load_state_dict(state["optim"])
    except (KeyError, ValueError):
        pass
    if sched is not None and state.get("sched") is not None:
        try:
            sched.load_state_dict(state["sched"])
        except (KeyError, ValueError):
            pass
    if "torch_rng_state" in state:
        try:
            torch.set_rng_state(state["torch_rng_state"])
        except (RuntimeError, TypeError):
            pass
    if "numpy_rng_state" in state:
        try:
            np.random.set_state(state["numpy_rng_state"])
        except (TypeError, ValueError):
            pass
    if torch.cuda.is_available() and "cuda_rng_state_all" in state:
        try:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        except (RuntimeError, TypeError):
            pass
    return int(state.get("step", 0))


def add_resume_args(p) -> None:
    """Attach the standard `--resume / --no-resume` toggle to an argparse parser.

    Resume is the DEFAULT (matches the user-facing convention). Pass
    `--no-resume` to start over.
    """
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="Default. Resume from the latest checkpoint in "
                        "--output-dir if one exists.")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Ignore any existing checkpoint and start fresh.")
