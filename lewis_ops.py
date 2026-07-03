"""
LEWIS-style op application for SAE-LEWIS (v2 — LEWIS-faithful two-tag scheme).

Following LEWIS (Reid & Zhong 2021, §2.1), the tagger produces TWO tags per
source token:

    1. `ins_before` — a binary indicator: should a phrase be inserted to the
       LEFT of this token?  Insertion is a property of the token BOUNDARY,
       not of the token itself (the token is otherwise kept/replaced/deleted
       as usual), so it does not compete with KEEP in a softmax.
    2. `op3`        — the non-insertion operation for the token itself:
       KEEP / REPL / DEL (3-class).

This replaces the v1 4-class scheme, where the INS tag was assigned to the
gap-adjacent token and therefore structurally collided with KEEP (the tagged
token's content is unchanged), which showed up as INS F1 = 0 on held-out
evaluation.

Editor-input semantics (LEWIS-faithful):

    KEEP : retain the original token in the editor template; expect identity
           at the output.
    REPL : replace the original token by [MASK]; the editor predicts the
           replacement.
    DEL  : the token is REMOVED from the editor template — deletion is the
           tagger's decision alone, exactly as in LEWIS where the BART
           generator never sees deleted tokens. The editor does not emit a
           [DEL] marker (the v1 [DEL]-output pathway is gone).
    ins_before : one or more [INS] slots are inserted at the boundary; the
           editor predicts the inserted tokens. Slot counts are set by
           template enumeration at inference (expand_ins_gap).

The corruption CACHE still stores the v1 4-class `tagger_gold`
(KEEP/REPL/INS/DEL with INS on the gap-adjacent token); `split_cache_tags`
converts it losslessly to the two-tag scheme (an INS cache tag always sits on
an otherwise-KEEP token — corruption.py rejects INS gaps adjacent to
REPL/DEL spans), so no cache regeneration is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Cache-side (v1) op ids — the corruption cache stores these in tagger_gold.
# ---------------------------------------------------------------------------
OP_KEEP = 0
OP_REPL = 1
OP_INS = 2
OP_DEL = 3
NUM_OPS = 4
OP_NAMES = ["KEEP", "REPL", "INS", "DEL"]

# ---------------------------------------------------------------------------
# Tagger-side (v2) op ids — 3-class head plus a separate binary insert head.
# ---------------------------------------------------------------------------
OP3_KEEP = 0
OP3_REPL = 1
OP3_DEL = 2
NUM_OPS3 = 3
OP3_NAMES = ["KEEP", "REPL", "DEL"]

# cache 4-class id → tagger 3-class id (INS maps to KEEP: the gap-adjacent
# token itself is unchanged; the insertion is carried by ins_before).
_CACHE_TO_OP3 = {OP_KEEP: OP3_KEEP, OP_REPL: OP3_REPL,
                 OP_INS: OP3_KEEP, OP_DEL: OP3_DEL}


def op_name(op: int) -> str:
    return OP_NAMES[int(op)]


def op3_name(op: int) -> str:
    return OP3_NAMES[int(op)]


def split_cache_tags(tags: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert cached 4-class tagger_gold to the two-tag scheme.

    Returns (op3, ins_before), both len(tags) arrays. Entries < 0 (ignore
    index, e.g. -100 padding) are passed through unchanged in op3 and map to
    -100 in ins_before as well.
    """
    op3 = np.full(len(tags), -100, dtype=np.int64)
    ins = np.full(len(tags), -100, dtype=np.int64)
    for i, t in enumerate(tags):
        t = int(t)
        if t < 0:
            continue
        op3[i] = _CACHE_TO_OP3[t]
        ins[i] = 1 if t == OP_INS else 0
    return op3, ins


@dataclass
class EditorInputs:
    """Deterministic editor-template artifacts derived from the tagger's
    two-tag output.

    Attributes
    ----------
    input_ids : np.ndarray (T_out,) int32
        The editor template x'_c — original tokens for KEEP, [MASK] for
        REPL, [INS] for insertion slots. DEL tokens are absent.
    op_per_pos : np.ndarray (T_out,) int8
        Op per template position: OP_KEEP / OP_REPL / OP_INS (cache ids;
        OP_DEL never appears — deleted tokens are not emitted).
    source_pos : np.ndarray (T_out,) int32
        Index into the source `token_ids` for each template position; -1
        for [INS] slots (no source token).
    ins_gaps : list of (start, length)
        Start position (in T_out) and slot count for each INS gap. Used by
        template enumeration at inference.
    """

    input_ids: np.ndarray
    op_per_pos: np.ndarray
    source_pos: np.ndarray
    ins_gaps: List[Tuple[int, int]]


def apply_ops_for_editor(
    token_ids: Sequence[int],
    op3: Sequence[int],
    ins_before: Sequence[int],
    mask_token_id: int,
    ins_token_id: int,
) -> EditorInputs:
    """Build the editor template x'_c from the two-tag output.

    `op3` and `ins_before` have the SAME LENGTH AS `token_ids`. Per source
    position i:
      1. if ins_before[i]: open a 1-slot INS gap (emit one [INS]);
         enumeration resizes it later.
      2. then emit per op3[i]:
           KEEP → token_ids[i];  REPL → [MASK];  DEL → nothing.
    """
    if not (len(token_ids) == len(op3) == len(ins_before)):
        raise ValueError(
            f"length mismatch: token_ids={len(token_ids)} op3={len(op3)} "
            f"ins_before={len(ins_before)}"
        )

    out_ids: List[int] = []
    out_ops: List[int] = []
    out_src: List[int] = []
    ins_gaps: List[Tuple[int, int]] = []

    for i in range(len(token_ids)):
        if int(ins_before[i]) > 0:
            ins_gaps.append((len(out_ids), 1))
            out_ids.append(ins_token_id)
            out_ops.append(OP_INS)
            out_src.append(-1)
        op = int(op3[i])
        if op == OP3_KEEP:
            out_ids.append(int(token_ids[i]))
            out_ops.append(OP_KEEP)
            out_src.append(i)
        elif op == OP3_REPL:
            out_ids.append(mask_token_id)
            out_ops.append(OP_REPL)
            out_src.append(i)
        elif op == OP3_DEL:
            pass  # deleted tokens never enter the template (LEWIS-faithful)
        else:
            raise ValueError(f"unknown op3 id {op} at position {i}")

    return EditorInputs(
        input_ids=np.asarray(out_ids, dtype=np.int32),
        op_per_pos=np.asarray(out_ops, dtype=np.int8),
        source_pos=np.asarray(out_src, dtype=np.int32),
        ins_gaps=ins_gaps,
    )


def expand_ins_gap(
    inputs: EditorInputs,
    gap_idx: int,
    new_slot_count: int,
    ins_token_id: int,
) -> EditorInputs:
    """Return a new EditorInputs with the `gap_idx`-th INS gap resized to
    `new_slot_count` slots. Used by template enumeration at inference.
    """
    start, length = inputs.ins_gaps[gap_idx]
    if new_slot_count == length:
        return inputs

    new_input_ids = np.concatenate([
        inputs.input_ids[:start],
        np.full(new_slot_count, ins_token_id, dtype=np.int32),
        inputs.input_ids[start + length:],
    ])
    new_op_per_pos = np.concatenate([
        inputs.op_per_pos[:start],
        np.full(new_slot_count, OP_INS, dtype=np.int8),
        inputs.op_per_pos[start + length:],
    ])
    new_source_pos = np.concatenate([
        inputs.source_pos[:start],
        np.full(new_slot_count, -1, dtype=np.int32),
        inputs.source_pos[start + length:],
    ])

    delta = new_slot_count - length
    new_gaps: List[Tuple[int, int]] = []
    for i, (s, l) in enumerate(inputs.ins_gaps):
        if i == gap_idx:
            new_gaps.append((s, new_slot_count))
        elif s > start:
            new_gaps.append((s + delta, l))
        else:
            new_gaps.append((s, l))

    return EditorInputs(
        input_ids=new_input_ids,
        op_per_pos=new_op_per_pos,
        source_pos=new_source_pos,
        ins_gaps=new_gaps,
    )


def decode_editor_output(
    input_ids: np.ndarray,
    argmax_ids: np.ndarray,
    op_per_pos: np.ndarray,
) -> List[int]:
    """Reduce editor (template_ids, argmax over the template segment) to the
    final token list.

    KEEP : output = template token (identity enforced regardless of argmax)
    REPL : output = argmax
    INS  : output = argmax

    Deletion needs no handling here: DEL tokens were removed from the
    template by `apply_ops_for_editor`.
    """
    out: List[int] = []
    for pos in range(len(input_ids)):
        op = int(op_per_pos[pos])
        if op == OP_KEEP:
            out.append(int(input_ids[pos]))
        else:  # OP_REPL / OP_INS
            out.append(int(argmax_ids[pos]))
    return out


def stats_ins_run_lengths(ops: Sequence[int]) -> List[int]:
    """Return a list of INS-run lengths (consecutive OP_INS) in a cache-id
    op sequence (used for cache-side statistics)."""
    out: List[int] = []
    i, n = 0, len(ops)
    while i < n:
        if int(ops[i]) == OP_INS:
            j = i
            while j < n and int(ops[j]) == OP_INS:
                j += 1
            out.append(j - i)
            i = j
        else:
            i += 1
    return out
