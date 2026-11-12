from multiprocessing import Pool

import numpy as np
from scipy.stats import rankdata

from common.log import progress
from task3.config import BUDGETS, N_ORDERS

BASE_METHODS = ["first", "sc", "logp", "selfcert", "selfcert_rwc"]
VLM_METHODS = ["vision", "vision_rwc", "visconf", "visconf_rwc", "visconf_token", "visconf_token_rwc"]
_DATA = None


def methods_for(vlm):
    return BASE_METHODS + (VLM_METHODS if vlm else [])


def quality_rwc(vlm):
    return "visconf_rwc" if vlm else "selfcert_rwc"


def vote(groups, weights):
    ug, first = np.unique(groups, return_index=True)
    totals = np.bincount(groups, weights=weights, minlength=int(groups.max()) + 1)[ug]
    best = totals.max()
    tied = np.where(totals >= best - 1e-12)[0]
    pick = tied[np.argmin(first[tied])]
    ordered = np.sort(totals)[::-1]
    second = ordered[1] if len(ordered) > 1 else 0.0
    return int(ug[pick]), int(first[pick]), float(best), float(second), float(totals.sum())


def rwc_weights(score):
    n = len(score)
    if n == 1:
        return np.ones(1)
    ranks = rankdata(-score, method="average")
    return (n - ranks) / (n - 1)


def visconf_scores(d, q, rolls):
    segs = [d.kl[d.offs[q, r] : d.offs[q, r] + d.tokens[q, r]] for r in rolls]
    pooled = np.concatenate(segs)
    n = len(rolls)
    C, Zt = np.zeros(n), np.zeros(n)
    if pooled.size == 0:
        return C, Zt
    lo, hi = np.percentile(pooled, [5, 95])
    den = max(hi - lo, 1e-8)
    for i, (seg, r) in enumerate(zip(segs, rolls)):
        if seg.size == 0:
            continue
        c = np.clip((seg - lo) / den, 0.0, 1.0)
        C[i] = c.mean()
        if d.vlm:
            Zt[i] = (c * d.vtok[d.offs[q, r] : d.offs[q, r] + d.tokens[q, r]]).mean()
    return C, Zt


def eval_prefix(d, q, rolls):
    g = d.groups[q, rolls]
    c = d.correct[q, rolls]
    n = len(rolls)
    scores = {"logp": d.L[q, rolls], "selfcert": d.K[q, rolls]}
    if d.vlm:
        C, Zt = visconf_scores(d, q, rolls)
        scores["vision"] = d.V[q, rolls]
        scores["visconf"] = scores["vision"] * C
        scores["visconf_token"] = Zt
    pos, aux = {"first": 0}, {}
    _, p, top, second, total = vote(g, np.ones(n))
    pos["sc"] = p
    cnt = np.bincount(g)
    ordered = np.sort(cnt)[::-1]
    aux["sc_top_count"] = int(ordered[0])
    aux["sc_second_count"] = int(ordered[1]) if len(ordered) > 1 else 0
    aux["sc_group"] = int(g[p])
    for name, s in scores.items():
        pos[name] = int(np.argmax(s))
    for name in ("selfcert", "vision", "visconf", "visconf_token"):
        if name not in scores:
            continue
        gw, pw, best, sec, tot = vote(g, rwc_weights(scores[name]))
        pos[f"{name}_rwc"] = pw
        aux[f"{name}_rwc_group"] = gw
        aux[f"{name}_rwc_margin"] = (best - sec) / tot if tot > 0 else 0.0
    return pos, aux, g, c


def _run_question(q):
    d = _DATA
    methods = methods_for(d.vlm)
    B = len(BUDGETS)
    res = {m: np.zeros((N_ORDERS, B), dtype=bool) for m in methods}
    grp = {m: np.zeros((N_ORDERS, B), dtype=np.int16) for m in methods}
    sel = {m: np.zeros((N_ORDERS, B), dtype=np.int8) for m in methods}
    cost = np.zeros((N_ORDERS, B))
    cost_first = np.zeros(N_ORDERS)
    oracle = np.zeros((N_ORDERS, B), dtype=bool)
    distinct = np.zeros((N_ORDERS, B), dtype=np.int16)
    aux_keys = ["sc_top_count", "sc_second_count", "sc_group", "quality_group", "quality_margin"]
    aux = {k: np.zeros((N_ORDERS, B)) for k in aux_keys}
    qr = quality_rwc(d.vlm)
    for o in range(N_ORDERS):
        order = d.orders[q, o].astype(int)
        for bi, n in enumerate(BUDGETS):
            rolls = order[:n]
            pos, a, g, c = eval_prefix(d, q, rolls)
            for m in methods:
                p = pos[m]
                res[m][o, bi] = c[p]
                grp[m][o, bi] = g[p]
                sel[m][o, bi] = p
            cost[o, bi] = d.tokens[q, rolls].sum()
            cost_first[o] = d.tokens[q, order[0]]
            oracle[o, bi] = c.any()
            distinct[o, bi] = len(np.unique(g))
            for k in ("sc_top_count", "sc_second_count", "sc_group"):
                aux[k][o, bi] = a[k]
            aux["quality_group"][o, bi] = a[f"{qr}_group"]
            aux["quality_margin"][o, bi] = a[f"{qr}_margin"]
    return res, grp, sel, cost, cost_first, oracle, distinct, aux


def run_all(d, workers=1):
    global _DATA
    _DATA = d
    methods = methods_for(d.vlm)
    Q, B = d.n_questions, len(BUDGETS)
    out = {
        "correct": {m: np.zeros((Q, N_ORDERS, B), dtype=bool) for m in methods},
        "group": {m: np.zeros((Q, N_ORDERS, B), dtype=np.int16) for m in methods},
        "selected_pos": {m: np.zeros((Q, N_ORDERS, B), dtype=np.int8) for m in methods},
        "cost": np.zeros((Q, N_ORDERS, B)),
        "cost_first": np.zeros((Q, N_ORDERS)),
        "oracle": np.zeros((Q, N_ORDERS, B), dtype=bool),
        "distinct": np.zeros((Q, N_ORDERS, B), dtype=np.int16),
        "aux": {k: np.zeros((Q, N_ORDERS, B)) for k in ["sc_top_count", "sc_second_count", "sc_group", "quality_group", "quality_margin"]},
    }
    bar = progress(Q, f"selection {d.name}", unit="q")
    it = Pool(workers).imap(_run_question, range(Q), chunksize=4) if workers > 1 else map(_run_question, range(Q))
    for q, (res, grp, sel, cost, cf, orc, dis, aux) in enumerate(it):
        for m in methods:
            out["correct"][m][q], out["group"][m][q], out["selected_pos"][m][q] = res[m], grp[m], sel[m]
        out["cost"][q], out["cost_first"][q], out["oracle"][q], out["distinct"][q] = cost, cf, orc, dis
        for k in aux:
            out["aux"][k][q] = aux[k]
        bar.update(1)
    bar.close()
    return out


def method_cost(out, method):
    if method == "first":
        return np.repeat(out["cost_first"][:, :, None], len(BUDGETS), axis=2)
    return out["cost"]
