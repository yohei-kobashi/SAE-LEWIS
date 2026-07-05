"""
v4 linguistically-typed transformation proposers (README §6.2.10-to-be).

Each proposer inspects a spaCy parse and returns grammatical↔grammatical
variant sentences (full text — gold is built downstream by token-level
diff, corruption.pair_to_gold, so no span bookkeeping is needed here).
Families are indexed on UniMorph/UD dimensions; BLiMP's 12 phenomenon
categories map onto them as ops (alternations), coordination requirements
(agreement), applicability constraints (islands: the wh proposers only
ever extract matrix arguments of simple clauses), or item-swap sub-rules
(NPI ↔ negation).

Quality contract per proposal:
  * conservative applicability predicate on the dependency parse;
  * `invertible=True` families are ROUND-TRIP checked by the caller:
    some reverse-direction proposal on T(X) must reproduce X verbatim;
  * non-invertible families (information-losing: WH, VALENCY, ELLIPSIS)
    carry structural sanity checks inside the proposer;
  * the symmetric SLOR gate + SAE gates downstream are the last line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from lemminflect import getInflection


@dataclass
class Proposal:
    t_type: str        # "FAMILY:direction"
    family: str
    out_text: str
    invertible: bool


NOM2ACC = {"i": "me", "he": "him", "she": "her", "we": "us", "they": "them"}
ACC2NOM = {v: k for k, v in NOM2ACC.items()}
REFLEX = {"i": "myself", "you": "yourself", "he": "himself", "she": "herself",
          "it": "itself", "we": "ourselves", "they": "themselves"}
ACC_REFLEX = {"me": "myself", "you": "yourself", "him": "himself",
              "her": "herself", "it": "itself", "us": "ourselves",
              "them": "themselves"}
REFLEX_ACC = {v: k for k, v in ACC_REFLEX.items()}
MODALS = ["can", "may", "must", "should", "might", "could"]
ERGATIVES = {"break", "open", "close", "melt", "sink", "boil", "dry",
             "freeze", "shatter", "crack", "burn", "bend", "snap", "move",
             "turn", "shake", "improve", "change", "increase", "decrease"}
TOUGH_ADJS = {"hard", "easy", "difficult", "tough", "impossible", "simple"}
NPI_SWAP = {"some": "any", "someone": "anyone", "something": "anything",
            "somebody": "anybody", "somewhere": "anywhere"}
NPI_UNSWAP = {v: k for k, v in NPI_SWAP.items()}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _span(tok):
    """(char_start, char_end) of a token's subtree (contiguous only)."""
    toks = sorted(tok.subtree, key=lambda t: t.i)
    if toks[-1].i - toks[0].i != len(toks) - 1:
        return None
    return toks[0].idx, toks[-1].idx + len(toks[-1].text)


def _core_np_span(head):
    """Char span of det/amod/compound/poss + head — the core NP without
    prep/relative attachments (used where those must stay in place)."""
    toks = [c for c in head.children
            if c.dep_ in ("det", "amod", "compound", "poss", "nummod")
            and c.i < head.i] + [head]
    toks = sorted(toks, key=lambda t: t.i)
    if toks[-1].i - toks[0].i != len(toks) - 1:
        return None
    return toks[0].idx, toks[-1].idx + len(toks[-1].text)


def _splice(text: str, edits: List[tuple]) -> str:
    """Apply non-overlapping (cs, ce, new) edits."""
    for cs, ce, new in sorted(edits, key=lambda e: -e[0]):
        text = text[:cs] + new + text[ce:]
    return text


def _tok_edit(tok, new: str) -> tuple:
    return (tok.idx, tok.idx + len(tok.text), new)


def _inflect(lemma: str, tag: str) -> Optional[str]:
    forms = getInflection(lemma, tag)
    return forms[0] if forms else None


def _subj_number(subj) -> str:
    if subj is None:
        return "Sing"
    n = subj.morph.get("Number")
    if n:
        return n[0]
    if subj.lower_ in ("they", "we", "you"):
        return "Plur"
    return "Sing"


def _subj_of(root):
    for c in root.children:
        if c.dep_ in ("nsubj", "nsubjpass"):
            return c
    return None


def _fin_tag_for(subj, past: bool) -> str:
    if past:
        return "VBD"
    if subj is not None and (subj.lower_ in ("i", "you", "we", "they")
                             or _subj_number(subj) == "Plur"):
        return "VBP"
    return "VBZ"


def _be_form(past: bool, subj) -> str:
    plur = subj is not None and (_subj_number(subj) == "Plur"
                                 or subj.lower_ in ("we", "they", "you"))
    if past:
        return "were" if plur else "was"
    if subj is not None and subj.lower_ == "i":
        return "am"
    return "are" if plur else "is"


def _do_form(past: bool, subj) -> str:
    if past:
        return "did"
    return "do" if (subj is not None and (subj.lower_ in ("i", "you")
                    or _subj_number(subj) == "Plur")) else "does"


def _decap(s: str, keep: bool) -> str:
    return s if (keep or not s) else s[0].lower() + s[1:]


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _keep_cap(tok) -> bool:
    return tok.pos_ == "PROPN" or tok.text == "I"


def _simple_root(doc):
    """Finite lexical-verb ROOT with no aux/cop children; None otherwise."""
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ != "VERB":
        return None
    if root.tag_ not in ("VBD", "VBZ", "VBP"):
        return None
    if any(c.dep_ in ("aux", "auxpass", "neg") for c in root.children):
        return None
    return root


def _sent_ok(doc) -> bool:
    return len(list(doc.sents)) == 1 and doc.text.strip().endswith(".")


# ---------------------------------------------------------------------------
# v4a families
# ---------------------------------------------------------------------------
def propose_tense(doc, text, rng) -> List[Proposal]:
    out = []
    root = _simple_root(doc)
    if root is not None and _sent_ok(doc):
        subj = _subj_of(root)
        if root.tag_ == "VBD":
            f = _inflect(root.lemma_, _fin_tag_for(subj, past=False))
            if f:
                out.append(Proposal("TENSE:PAST->PRES", "TENSE",
                                    _splice(text, [_tok_edit(root, f)]), True))
            b = _inflect(root.lemma_, "VB")
            if b:
                out.append(Proposal("TENSE:PAST->FUT", "TENSE",
                                    _splice(text, [_tok_edit(root, f"will {b}")]),
                                    True))
        elif root.tag_ in ("VBZ", "VBP"):
            f = _inflect(root.lemma_, "VBD")
            if f:
                out.append(Proposal("TENSE:PRES->PAST", "TENSE",
                                    _splice(text, [_tok_edit(root, f)]), True))
    # will + VB  →  finite
    for t in doc:
        if t.dep_ == "ROOT" and t.tag_ == "VB":
            auxs = [c for c in t.children if c.dep_ == "aux"]
            if len(auxs) == 1 and auxs[0].lemma_ == "will" and _sent_ok(doc):
                subj = _subj_of(t)
                for past, name in ((False, "FUT->PRES"), (True, "FUT->PAST")):
                    f = _inflect(t.lemma_, _fin_tag_for(subj, past))
                    if f:
                        cs, ce = auxs[0].idx, t.idx + len(t.text)
                        out.append(Proposal(f"TENSE:{name}", "TENSE",
                                            _splice(text, [(cs, ce, f)]), True))
    return out


def propose_aspect(doc, text, rng) -> List[Proposal]:
    out = []
    root = _simple_root(doc)
    if root is not None and root.tag_ == "VBD" and _sent_ok(doc):
        subj = _subj_of(root)
        g = _inflect(root.lemma_, "VBG")
        n = _inflect(root.lemma_, "VBN")
        if g:
            out.append(Proposal("ASPECT:SIMPLE->PROG", "ASPECT", _splice(
                text, [_tok_edit(root, f"{_be_form(True, subj)} {g}")]), True))
        if n:
            hv = "have" if _fin_tag_for(subj, False) == "VBP" else "has"
            out.append(Proposal("ASPECT:SIMPLE->PERF", "ASPECT", _splice(
                text, [_tok_edit(root, f"{hv} {n}")]), True))
    # reverse: be+VBG / have+VBN  →  simple past
    for t in doc:
        if t.dep_ != "ROOT":
            continue
        auxs = [c for c in t.children if c.dep_ == "aux"]
        if (t.tag_ == "VBG" and len(auxs) == 1
                and auxs[0].lemma_ == "be" and auxs[0].tag_ in ("VBD",)):
            f = _inflect(t.lemma_, "VBD")
            if f and _sent_ok(doc):
                cs, ce = auxs[0].idx, t.idx + len(t.text)
                out.append(Proposal("ASPECT:PROG->SIMPLE", "ASPECT",
                                    _splice(text, [(cs, ce, f)]), True))
        if (t.tag_ == "VBN" and len(auxs) == 1 and auxs[0].lemma_ == "have"
                and auxs[0].tag_ in ("VBZ", "VBP")):
            f = _inflect(t.lemma_, "VBD")
            if f and _sent_ok(doc):
                cs, ce = auxs[0].idx, t.idx + len(t.text)
                out.append(Proposal("ASPECT:PERF->SIMPLE", "ASPECT",
                                    _splice(text, [(cs, ce, f)]), True))
    return out


def propose_modality(doc, text, rng) -> List[Proposal]:
    out = []
    root = _simple_root(doc)
    if root is not None and root.lemma_ != "be" and _sent_ok(doc):
        b = _inflect(root.lemma_, "VB")
        if b:
            m = MODALS[int(rng.randrange(len(MODALS)))]
            out.append(Proposal(f"MOD:+{m}", "MODALITY",
                                _splice(text, [_tok_edit(root, f"{m} {b}")]),
                                True))
    for t in doc:
        if t.dep_ == "ROOT" and t.tag_ == "VB":
            auxs = [c for c in t.children if c.dep_ == "aux"]
            if len(auxs) == 1 and auxs[0].tag_ == "MD" \
                    and auxs[0].lower_ in MODALS and _sent_ok(doc):
                subj = _subj_of(t)
                for past in (False, True):     # both tenses → roundtrip can
                    f = _inflect(t.lemma_, _fin_tag_for(subj, past))
                    if f:                      # recover +MOD on past inputs
                        cs, ce = auxs[0].idx, t.idx + len(t.text)
                        out.append(Proposal(
                            f"MOD:-{auxs[0].lower_}", "MODALITY",
                            _splice(text, [(cs, ce, f)]), True))
                others = [m for m in MODALS if m != auxs[0].lower_]
                m2 = others[int(rng.randrange(len(others)))]
                out.append(Proposal(f"MOD:{auxs[0].lower_}->{m2}", "MODALITY",
                                    _splice(text, [_tok_edit(auxs[0], m2)]),
                                    True))
    return out


def propose_number(doc, text, rng) -> List[Proposal]:
    out = []
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or not _sent_ok(doc):
        return out
    subj = _subj_of(root)
    if subj is None or subj.pos_ != "NOUN":
        return out
    if any(c.dep_ == "nummod" or c.pos_ == "NUM" for c in subj.children):
        return out
    det = next((c for c in subj.children if c.dep_ == "det"), None)
    num = _subj_number(subj)
    to_plur = (num == "Sing")
    if det is not None and det.lower_ in ("a", "an", "every", "each") and to_plur:
        return out                     # a/every + plural is ill-formed
    noun_tag = "NNS" if to_plur else "NN"
    new_noun = _inflect(subj.lemma_, noun_tag)
    if not new_noun or new_noun == subj.text:
        return out
    edits = [_tok_edit(subj, new_noun)]
    if det is not None:
        dmap = ({"this": "these", "that": "those"} if to_plur
                else {"these": "this", "those": "that"})
        if det.lower_ in dmap:
            edits.append(_tok_edit(det, dmap[det.lower_]))
        elif det.lower_ not in ("the",) and det.dep_ != "poss":
            if det.lower_ in ("a", "an") and not to_plur:
                pass
            elif det.lower_ not in ("a", "an"):
                return out
    # verb agreement site
    vt = root if root.tag_ in ("VBZ", "VBP") else None
    if vt is None:
        fin_aux = [c for c in root.children
                   if c.dep_ in ("aux", "auxpass") and c.tag_ in ("VBZ", "VBP")]
        vt = fin_aux[0] if len(fin_aux) == 1 else None
    if root.tag_ not in ("VBD",) and vt is None:
        if root.tag_ not in ("VBZ", "VBP", "VBD"):
            return out
    if vt is not None:
        tag = "VBP" if to_plur else "VBZ"
        nv = _inflect(vt.lemma_, tag)
        if not nv:
            return out
        edits.append(_tok_edit(vt, nv))
    name = "NUMBER:SG->PL" if to_plur else "NUMBER:PL->SG"
    out.append(Proposal(name, "NUMBER", _splice(text, edits), True))
    return out


def propose_degree(doc, text, rng) -> List[Proposal]:
    out = []
    if not _sent_ok(doc):
        return out
    for t in doc:
        if t.pos_ != "ADJ":
            continue
        if t.tag_ == "JJ" and len(t.text) <= 7 and t.text.isalpha():
            for tag, name in (("JJR", "DEG:POS->CMP"), ("JJS", "DEG:POS->SUP")):
                f = _inflect(t.lemma_, tag)
                if f and f != t.text and " " not in f:
                    out.append(Proposal(name, "DEGREE",
                                        _splice(text, [_tok_edit(t, f)]), True))
        elif t.tag_ in ("JJR", "JJS"):
            f = _inflect(t.lemma_, "JJ")
            if f and f != t.text:
                name = "DEG:CMP->POS" if t.tag_ == "JJR" else "DEG:SUP->POS"
                out.append(Proposal(name, "DEGREE",
                                    _splice(text, [_tok_edit(t, f)]), True))
        if t.tag_ == "JJ":
            advs = [c for c in t.children if c.dep_ == "advmod"]
            if not advs:
                out.append(Proposal("DEG:+very", "DEGREE", _splice(
                    text, [(t.idx, t.idx, "very ")]), True))
            elif len(advs) == 1 and advs[0].lower_ == "very":
                a = advs[0]
                out.append(Proposal("DEG:-very", "DEGREE", _splice(
                    text, [(a.idx, t.idx, "")]), True))
    return out[:4]


def _npi_edits(root, to_any: bool) -> List[tuple]:
    table = NPI_SWAP if to_any else NPI_UNSWAP
    edits = []
    for t in root.subtree:
        if t.i > root.i and t.lower_ in table:
            new = table[t.lower_]
            edits.append(_tok_edit(t, _cap(new) if t.text[0].isupper() else new))
    return edits


def propose_negation(doc, text, rng) -> List[Proposal]:
    out = []
    root = _simple_root(doc)
    if root is not None and _sent_ok(doc):
        subj = _subj_of(root)
        if root.lemma_ == "be":
            edits = [(root.idx + len(root.text), root.idx + len(root.text),
                      " not")] + _npi_edits(root, True)
            out.append(Proposal("NEG:+", "NEGATION", _splice(text, edits), True))
        else:
            b = _inflect(root.lemma_, "VB")
            if b:
                d = _do_form(root.tag_ == "VBD", subj)
                edits = [_tok_edit(root, f"{d} not {b}")] + _npi_edits(root, True)
                out.append(Proposal("NEG:+", "NEGATION", _splice(text, edits),
                                    True))
    # removal
    for t in doc:
        if t.dep_ != "ROOT" or not _sent_ok(doc):
            continue
        negs = [c for c in t.children if c.dep_ == "neg" and c.lower_ == "not"]
        if len(negs) != 1:
            continue
        neg = negs[0]
        auxs = [c for c in t.children if c.dep_ == "aux" and c.lemma_ == "do"]
        subj = _subj_of(t)
        if t.tag_ == "VB" and len(auxs) == 1:
            f = _inflect(t.lemma_, _fin_tag_for(subj, auxs[0].tag_ == "VBD"))
            if f:
                edits = [(auxs[0].idx, t.idx + len(t.text), f)] + \
                    _npi_edits(t, False)
                out.append(Proposal("NEG:-", "NEGATION", _splice(text, edits),
                                    True))
        elif t.lemma_ == "be" and t.tag_ in ("VBD", "VBZ", "VBP"):
            edits = [(t.idx + len(t.text), neg.idx + len(neg.text) -
                      (len(t.text) + 1) + t.idx + len(t.text) + 1, "")]
            edits = [(t.idx + len(t.text), neg.idx + len(neg.text), "")] + \
                _npi_edits(t, False)
            out.append(Proposal("NEG:-", "NEGATION", _splice(text, edits), True))
    return out


def propose_detquant(doc, text, rng) -> List[Proposal]:
    out = []
    if not _sent_ok(doc):
        return out
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    for t in doc:
        if t.dep_ == "det" and t.lower_ in ("every", "each"):
            new = "each" if t.lower_ == "every" else "every"
            out.append(Proposal(f"DETQ:{t.lower_}->{new}", "DETQUANT", _splice(
                text, [_tok_edit(t, _cap(new) if t.text[0].isupper() else new)]),
                True))
    # all + PL subj + VBP  <->  every + SG subj + VBZ
    if root is not None:
        subj = _subj_of(root)
        if subj is not None and subj.pos_ == "NOUN" and root.tag_ in ("VBP", "VBZ"):
            det = next((c for c in subj.children if c.dep_ == "det"), None)
            if det is not None and det.lower_ == "all" and root.tag_ == "VBP":
                sg = _inflect(subj.lemma_, "NN")
                v = _inflect(root.lemma_, "VBZ")
                if sg and v:
                    new_det = "Every" if det.text[0].isupper() else "every"
                    out.append(Proposal("DETQ:ALL->EVERY", "DETQUANT", _splice(
                        text, [_tok_edit(det, new_det), _tok_edit(subj, sg),
                               _tok_edit(root, v)]), True))
            if det is not None and det.lower_ == "every" and root.tag_ == "VBZ":
                pl = _inflect(subj.lemma_, "NNS")
                v = _inflect(root.lemma_, "VBP")
                if pl and v:
                    new_det = "All" if det.text[0].isupper() else "all"
                    out.append(Proposal("DETQ:EVERY->ALL", "DETQUANT", _splice(
                        text, [_tok_edit(det, new_det), _tok_edit(subj, pl),
                               _tok_edit(root, v)]), True))
    return out


def propose_anaphor(doc, text, rng) -> List[Proposal]:
    out = []
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or not _sent_ok(doc):
        return out
    subj = _subj_of(root)
    if subj is None or subj.pos_ != "PRON":
        return out
    for c in root.children:
        if c.dep_ != "dobj" or c.pos_ != "PRON":
            continue
        refl = REFLEX.get(subj.lower_)
        if c.lower_ in ACC_REFLEX and refl and ACC_REFLEX[c.lower_] == refl:
            out.append(Proposal("ANA:+REFL", "ANAPHOR",
                                _splice(text, [_tok_edit(c, refl)]), True))
        if c.lower_ in REFLEX_ACC and refl == c.lower_:
            out.append(Proposal("ANA:-REFL", "ANAPHOR", _splice(
                text, [_tok_edit(c, REFLEX_ACC[c.lower_])]), True))
    return out


# ---------------------------------------------------------------------------
# v4c families
# ---------------------------------------------------------------------------
def _svo(doc):
    """(root, subj, obj) of a simple transitive declarative, else None.
    Island discipline: only MATRIX arguments of a mono-clausal sentence are
    ever touched (no ccomp/xcomp/advcl/relcl anywhere)."""
    root = _simple_root(doc)
    if root is None or not _sent_ok(doc):
        return None
    if any(t.dep_ in ("ccomp", "xcomp", "advcl", "relcl", "csubj", "conj")
           for t in doc):
        return None
    subj = _subj_of(root)
    obj = next((c for c in root.children if c.dep_ == "dobj"), None)
    if subj is None or obj is None:
        return None
    ss, so = _span(subj), _span(obj)
    if ss is None or so is None:
        return None
    if not (ss[1] <= root.idx and root.idx + len(root.text) <= so[0]):
        return None
    if ss[0] != doc[0].idx:
        return None
    return root, subj, obj, ss, so


def propose_voice(doc, text, rng) -> List[Proposal]:
    out = []
    svo = _svo(doc)
    if svo is not None:
        root, subj, obj, ss, so = svo
        if root.lemma_ != "be":
            vbn = _inflect(root.lemma_, "VBN")
            if vbn:
                subj_txt = text[ss[0]:ss[1]]
                obj_txt = text[so[0]:so[1]]
                if subj.pos_ == "PRON" and subj.lower_ in NOM2ACC:
                    subj_txt = NOM2ACC[subj.lower_]
                else:
                    subj_txt = _decap(subj_txt, _keep_cap(doc[0]))
                if obj.pos_ == "PRON" and obj.lower_ in ACC2NOM:
                    obj_txt = _cap(ACC2NOM[obj.lower_])
                else:
                    obj_txt = _cap(obj_txt)
                be = _be_form(root.tag_ == "VBD", obj)
                T = (obj_txt + " " + be + " " + vbn + " by " + subj_txt
                     + text[so[1]:])
                out.append(Proposal("VOICE:ACT->PASS", "VOICE", T, True))
    # passive -> active
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is not None and root.tag_ == "VBN" and _sent_ok(doc):
        aux = [c for c in root.children if c.dep_ in ("auxpass", "aux")
               and c.lemma_ == "be" and c.tag_ in ("VBD", "VBZ", "VBP")]
        spass = next((c for c in root.children if c.dep_ == "nsubjpass"), None)
        agents = [c for c in root.children if c.dep_ == "agent"]
        if len(aux) == 1 and spass is not None and len(agents) == 1:
            pobj = next((c for c in agents[0].children if c.dep_ == "pobj"),
                        None)
            sp, ap = _span(spass), (None if pobj is None else _span(pobj))
            if pobj is not None and sp is not None and ap is not None \
                    and sp[0] == doc[0].idx:
                ag_txt = text[ap[0]:ap[1]]
                if pobj.pos_ == "PRON" and pobj.lower_ in ACC2NOM:
                    ag_txt = _cap(ACC2NOM[pobj.lower_])
                else:
                    ag_txt = _cap(ag_txt)
                ob_txt = text[sp[0]:sp[1]]
                if spass.pos_ == "PRON" and spass.lower_ in NOM2ACC:
                    ob_txt = NOM2ACC[spass.lower_]
                else:
                    ob_txt = _decap(ob_txt, _keep_cap(doc[0]))
                v = _inflect(root.lemma_,
                             _fin_tag_for(pobj, aux[0].tag_ == "VBD"))
                if v:
                    T = ag_txt + " " + v + " " + ob_txt + text[ap[1]:]
                    out.append(Proposal("VOICE:PASS->ACT", "VOICE", T, True))
    return out


def propose_interrog(doc, text, rng) -> List[Proposal]:
    out = []
    root = _simple_root(doc)
    # +YN via do-support
    if root is not None and root.lemma_ != "be" and _sent_ok(doc):
        subj = _subj_of(root)
        ss = None if subj is None else _span(subj)
        if ss is not None and ss[0] == doc[0].idx:
            b = _inflect(root.lemma_, "VB")
            if b:
                d = _cap(_do_form(root.tag_ == "VBD", subj))
                body = _splice(text, [_tok_edit(root, b)])
                body = _decap(body, _keep_cap(doc[0]))
                T = d + " " + body[:-1].rstrip() + "?"
                out.append(Proposal("Q:+YN", "INTERROG", T, True))
    # -YN: "Did subj VB ...?"
    if text.endswith("?"):
        r2 = next((t for t in doc if t.dep_ == "ROOT"), None)
        if r2 is not None and r2.tag_ == "VB":
            auxs = [c for c in r2.children if c.dep_ == "aux"
                    and c.lemma_ == "do" and c.i == 0]
            subj = _subj_of(r2)
            if len(auxs) == 1 and subj is not None:
                f = _inflect(r2.lemma_,
                             _fin_tag_for(subj, auxs[0].tag_ == "VBD"))
                ss = _span(subj)
                if f and ss is not None:
                    body = _splice(text, [_tok_edit(r2, f)])
                    aux_end = auxs[0].idx + len(auxs[0].text) + 1
                    body = body[aux_end:]
                    T = _cap(body)[:-1].rstrip() + "."
                    out.append(Proposal("Q:-YN", "INTERROG", T, True))
    # wh questions (information-losing; matrix arguments only — islands
    # are unreachable by construction)
    svo = _svo(doc)
    if svo is not None:
        root, subj, obj, ss, so = svo
        wh_s = "Who" if subj.pos_ in ("PROPN", "PRON") else "What"
        T = wh_s + text[ss[1]:]
        if T.endswith("."):
            T = T[:-1] + "?"
            out.append(Proposal("Q:+WHSUBJ", "INTERROG", T, False))
        b = _inflect(root.lemma_, "VB")
        if b and root.lemma_ != "be":
            wh_o = "Who" if obj.pos_ in ("PROPN", "PRON") else "What"
            d = _do_form(root.tag_ == "VBD", subj)
            mid = _splice(text[ss[0]:so[0]], [
                (root.idx - ss[0], root.idx - ss[0] + len(root.text), b)])
            mid = _decap(mid, _keep_cap(doc[0])).rstrip()
            tail = text[so[1]:].rstrip()
            tail = tail[:-1] if tail.endswith(".") else tail
            T = f"{wh_o} {d} {mid}{tail}?"
            out.append(Proposal("Q:+WHOBJ", "INTERROG", T, False))
    return out


def propose_existential(doc, text, rng) -> List[Proposal]:
    out = []
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.lemma_ != "be" or not _sent_ok(doc):
        return out
    if root.tag_ not in ("VBD", "VBZ", "VBP"):
        return out
    expl = next((c for c in root.children if c.dep_ == "expl"), None)
    if expl is None:
        subj = _subj_of(root)
        if subj is None or subj.pos_ != "NOUN":
            return out
        det = next((c for c in subj.children if c.dep_ == "det"), None)
        if det is None or det.lower_ not in ("a", "an", "some"):
            return out
        ss = _core_np_span(subj)
        if ss is None or ss[0] != doc[0].idx or root.idx < ss[1]:
            return out
        rest = text[root.idx + len(root.text):]
        if len(rest.strip()) <= 1:
            return out
        T = ("There " + root.text + " " + _decap(text[ss[0]:ss[1]], False)
             + rest)
        out.append(Proposal("EXIST:+THERE", "EXISTENTIAL", T, True))
    else:
        subj = next((c for c in root.children
                     if c.dep_ in ("attr", "nsubj") and c.i > root.i), None)
        if subj is None or expl.i != 0:
            return out
        ss = _core_np_span(subj)
        if ss is None:
            return out
        T = (_cap(text[ss[0]:ss[1]]) + " " + root.text + text[ss[1]:])
        out.append(Proposal("EXIST:-THERE", "EXISTENTIAL", T, True))
    return out


def propose_valency(doc, text, rng) -> List[Proposal]:
    out = []
    svo = _svo(doc)
    if svo is None:
        return out
    root, subj, obj, ss, so = svo
    if root.lemma_ not in ERGATIVES:
        return out
    v = _inflect(root.lemma_,
                 _fin_tag_for(obj, root.tag_ == "VBD"))
    if not v:
        return out
    obj_txt = text[so[0]:so[1]]
    if obj.pos_ == "PRON" and obj.lower_ in ACC2NOM:
        obj_txt = ACC2NOM[obj.lower_]
    T = _cap(obj_txt) + " " + v + text[so[1]:]
    out.append(Proposal("VAL:CAUS->INCH", "VALENCY", T, False))
    return out


def propose_tough(doc, text, rng) -> List[Proposal]:
    out = []
    if not _sent_ok(doc):
        return out
    m = re.match(r"^It (is|was) (\w+) to (\w+) (.+)\.$", text)
    if m and m.group(2).lower() in TOUGH_ADJS:
        T = f"{_cap(m.group(4))} {m.group(1)} {m.group(2)} to {m.group(3)}."
        out.append(Proposal("TOUGH:IT->RAISED", "TOUGH", T, True))
    m2 = re.match(r"^(.+) (is|was) (\w+) to (\w+)\.$", text)
    if m2 and m2.group(3).lower() in TOUGH_ADJS and not text.startswith("It "):
        T = (f"It {m2.group(2)} {m2.group(3)} to {m2.group(4)} "
             f"{_decap(m2.group(1), _keep_cap(doc[0]))}.")
        out.append(Proposal("TOUGH:RAISED->IT", "TOUGH", T, True))
    return out


def propose_ellipsis(doc, text, rng) -> List[Proposal]:
    out = []
    if not _sent_ok(doc):
        return out
    nouns = [t for t in doc if t.pos_ == "NOUN"
             and any(c.dep_ == "nummod" for c in t.children)]
    if len(nouns) == 2 and nouns[0].lemma_ == nouns[1].lemma_ \
            and any(t.lower_ == "and" for t in doc):
        second = nouns[1]
        num = next(c for c in second.children if c.dep_ == "nummod")
        cs = num.idx + len(num.text)
        ce = second.idx + len(second.text)
        out.append(Proposal("ELL:+NBAR", "ELLIPSIS",
                            _splice(text, [(cs, ce, "")]), False))
    return out


def propose_cleft(doc, text, rng) -> List[Proposal]:
    out = []
    svo = _svo(doc)
    if svo is not None:
        root, subj, obj, ss, so = svo
        if subj.pos_ in ("PROPN",):
            be = "was" if root.tag_ == "VBD" else "is"
            T = f"It {be} {text[ss[0]:ss[1]]} who{text[ss[1]:]}"
            out.append(Proposal("CLEFT:+IT", "CLEFT", T, True))
    m = re.match(r"^It (is|was) (.+?) who (.+)$", text)
    if m and _sent_ok(doc):
        T = f"{_cap(m.group(2))} {m.group(3)}"
        out.append(Proposal("CLEFT:-IT", "CLEFT", T, True))
    return out


def propose_inversion(doc, text, rng) -> List[Proposal]:
    out = []
    m = re.match(r"^If (\w+) (had|were) (.+?), (.+)$", text)
    if m:
        T = f"{_cap(m.group(2))} {m.group(1)} {m.group(3)}, {m.group(4)}"
        out.append(Proposal("INV:IF->AUX", "INVERSION", T, True))
    m2 = re.match(r"^(Had|Were) (\w+) (.+?), (.+)$", text)
    if m2:
        T = f"If {m2.group(2)} {m2.group(1).lower()} {m2.group(3)}, {m2.group(4)}"
        out.append(Proposal("INV:AUX->IF", "INVERSION", T, True))
    return out


def propose_splitjoin(doc, text, rng) -> List[Proposal]:
    out = []
    sents = list(doc.sents)
    if len(sents) == 1 and ", and " in text and text.endswith("."):
        left, right = text.split(", and ", 1)
        if len(left.split()) >= 3 and len(right.split()) >= 3:
            r_doc_has_verb = any(t.pos_ in ("VERB", "AUX")
                                 for t in doc if t.idx > text.index(", and "))
            l_has_verb = any(t.pos_ in ("VERB", "AUX")
                             for t in doc if t.idx < text.index(", and "))
            if r_doc_has_verb and l_has_verb:
                T = left + ". " + _cap(right)
                out.append(Proposal("SPLIT:,and->.", "SPLITJOIN", T, True))
    if len(sents) == 2:
        a, b = sents[0].text.strip(), sents[1].text.strip()
        if a.endswith(".") and b.endswith("."):
            T = a[:-1] + ", and " + _decap(b, _keep_cap(sents[1][0]))
            out.append(Proposal("JOIN:.->,and", "SPLITJOIN", T, True))
    return out


# ---------------------------------------------------------------------------
FAMILIES: Dict[str, Callable] = {
    # v4a
    "TENSE": propose_tense,
    "ASPECT": propose_aspect,
    "MODALITY": propose_modality,
    "NUMBER": propose_number,
    "DEGREE": propose_degree,
    "NEGATION": propose_negation,
    "DETQUANT": propose_detquant,
    "ANAPHOR": propose_anaphor,
    # v4c
    "VOICE": propose_voice,
    "INTERROG": propose_interrog,
    "EXISTENTIAL": propose_existential,
    "VALENCY": propose_valency,
    "TOUGH": propose_tough,
    "ELLIPSIS": propose_ellipsis,
    "CLEFT": propose_cleft,
    "INVERSION": propose_inversion,
    "SPLITJOIN": propose_splitjoin,
}


def propose_transforms(nlp, text: str, rng,
                       families: Optional[List[str]] = None) -> List[Proposal]:
    doc = nlp(text)
    out: List[Proposal] = []
    for name, fn in FAMILIES.items():
        if families and name not in families:
            continue
        try:
            out.extend(fn(doc, text, rng))
        except Exception:
            continue                     # a proposer bug must never kill a worker
    return [p for p in out if p.out_text != text and p.out_text.strip()]


def roundtrip_ok(nlp, prop: Proposal, x_text: str, rng) -> bool:
    """For invertible families: some same-family proposal on T(X) must
    reproduce X verbatim."""
    if not prop.invertible:
        return True
    try:
        back = propose_transforms(nlp, prop.out_text, rng,
                                  families=[prop.family])
    except Exception:
        return False
    return any(b.out_text == x_text for b in back)
