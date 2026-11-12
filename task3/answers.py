import ast
import os
import re
from functools import lru_cache

NO_ANSWER = "NO_ANSWER"
TIMEOUT = None if os.name == "nt" else 5
LETTERS = "ABCDEFGHIJKLMNOP"


def last_boxed(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    j = i
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    if depth:
        return None
    out = text[i : j - 1].strip()
    return out or None


def _quick(s):
    s = s.strip().rstrip(".").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "").replace("$", "").replace("^\\circ", "").replace("^{\\circ}", "")
    return re.sub(r"\s+", "", s)


@lru_cache(maxsize=None)
def _parse(s):
    from math_verify import parse

    try:
        return parse("$" + s + "$", parsing_timeout=TIMEOUT)
    except Exception:
        return None


def math_equal(a, b):
    from math_verify import verify

    if _quick(a) == _quick(b):
        return True
    pa, pb = _parse(a), _parse(b)
    if not pa or not pb:
        return False
    try:
        return bool(verify(pa, pb, timeout_seconds=TIMEOUT)) or bool(verify(pb, pa, timeout_seconds=TIMEOUT))
    except Exception:
        return False


def group_math(answers):
    reps, labels, canon = [], [], {}
    for a in answers:
        if a is None or not str(a).strip():
            labels.append(0)
            continue
        a = str(a).strip()
        placed = None
        for gid, rep in reps:
            if math_equal(a, rep):
                placed = gid
                break
        if placed is None:
            placed = len(reps) + 1
            reps.append((placed, a))
        labels.append(placed)
    canon = {0: NO_ANSWER, **{g: r for g, r in reps}}
    return labels, canon


def grade_math_groups(canon, gold):
    return {g: (g != 0 and math_equal(rep, gold)) for g, rep in canon.items()}


def _number(s):
    s = str(s).replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def _norm_text(s):
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" .")


def canonical_vista(extracted, meta):
    if extracted is None or not str(extracted).strip():
        return NO_ANSWER
    s = str(extracted).strip()
    qtype, atype = meta["question_type"], meta["answer_type"]
    if qtype == "multi_choice":
        choices = list(meta["choices"])
        m = re.fullmatch(r"\(?([A-Za-z])\)?[.:]?", s)
        if m and m.group(1).upper() in LETTERS[: len(choices)]:
            return m.group(1).upper()
        norm = [_norm_text(c) for c in choices]
        if _norm_text(s) in norm:
            return LETTERS[norm.index(_norm_text(s))]
        m = re.match(r"\(?([A-Za-z])\)\s", s)
        if m and m.group(1).upper() in LETTERS[: len(choices)]:
            return m.group(1).upper()
        return "RAW:" + _norm_text(s)
    if atype == "integer":
        v = _number(s)
        return "RAW:" + _norm_text(s) if v is None else str(int(round(v)))
    if atype == "float":
        v = _number(s)
        if v is None:
            return "RAW:" + _norm_text(s)
        return str(round(v, int(meta["precision"] or 2)))
    if atype == "list":
        try:
            return str(ast.literal_eval(s))
        except Exception:
            return "RAW:" + _norm_text(s).replace(" ", "")
    return _norm_text(s)


def gold_vista(meta):
    gold = str(meta["answer"])
    if meta["question_type"] == "multi_choice":
        norm = [_norm_text(c) for c in meta["choices"]]
        if _norm_text(gold) in norm:
            return LETTERS[norm.index(_norm_text(gold))]
        return gold.strip().upper()
    return canonical_vista(gold, meta)
