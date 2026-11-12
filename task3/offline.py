import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common.utils import save_json
from task3.config import BOOT, BUDGETS, OUT, SEED
from task3.selection import methods_for, method_cost, visconf_scores

DIFFS = [
    ("sc", "first"), ("logp", "first"), ("selfcert", "first"), ("selfcert_rwc", "selfcert"), ("selfcert_rwc", "sc"),
    ("vision", "first"), ("vision_rwc", "vision"), ("visconf", "first"), ("visconf_rwc", "visconf"),
    ("visconf_rwc", "sc"), ("visconf", "selfcert"), ("visconf_rwc", "selfcert_rwc"),
    ("visconf_token", "visconf"), ("visconf_token_rwc", "visconf_rwc"),
]


def per_question_acc(out, method):
    return out["correct"][method].mean(1)


def bootstrap_diff(a, b, rng):
    diff = a - b
    idx = rng.integers(0, len(diff), size=(BOOT, len(diff)))
    means = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def curves(d, out):
    rows = []
    for split, idx in (("dev", d.dev), ("eval", d.eval)):
        cover = out["oracle"][idx].mean(1)
        distinct = out["distinct"][idx].mean(1)
        for m in methods_for(d.vlm):
            acc = per_question_acc(out, m)[idx]
            cost = method_cost(out, m)[idx].mean(1)
            for bi, n in enumerate(BUDGETS):
                rows.append(
                    {
                        "dataset": d.name, "split": split, "method": m, "N": n,
                        "accuracy": acc[:, bi].mean(), "coverage": cover[:, bi].mean(), "distinct_groups": distinct[:, bi].mean(),
                        "tokens": cost[:, bi].mean(),
                        "no_correct_candidate": 1 - cover[:, bi].mean(),
                        "correct_missed": (cover[:, bi] - acc[:, bi]).mean(),
                    }
                )
    return pd.DataFrame(rows)


def diff_table(d, out):
    rng = np.random.default_rng(SEED)
    rows = []
    methods = set(methods_for(d.vlm))
    for a, b in DIFFS:
        if a not in methods or b not in methods:
            continue
        qa, qb = per_question_acc(out, a)[d.eval], per_question_acc(out, b)[d.eval]
        for bi, n in enumerate(BUDGETS):
            if n == 1:
                continue
            m, lo, hi = bootstrap_diff(qa[:, bi], qb[:, bi], rng)
            rows.append({"dataset": d.name, "a": a, "b": b, "N": n, "diff_pp": 100 * m, "lo_pp": 100 * lo, "hi_pp": 100 * hi})
    return pd.DataFrame(rows)


def long_frame(d, out):
    frames = []
    qid = np.array(d.qids)
    for m in methods_for(d.vlm):
        Q, O, B = out["correct"][m].shape
        qi, oi, bi = np.meshgrid(np.arange(Q), np.arange(O), np.arange(B), indexing="ij")
        cost = method_cost(out, m)
        sel_roll = np.take_along_axis(d.orders[:, :, None, :].astype(np.int64).repeat(B, 2), out["selected_pos"][m][..., None].astype(np.int64), axis=3)[..., 0]
        frames.append(
            pd.DataFrame(
                {
                    "dataset": d.name, "qid": qid[qi.ravel()], "split": np.where(np.isin(qi.ravel(), d.dev), "dev", "eval"),
                    "order": oi.ravel(), "N": np.array(BUDGETS)[bi.ravel()], "method": m,
                    "selected_rollout": sel_roll.ravel(), "selected_group": out["group"][m].ravel(), "correct": out["correct"][m].ravel(),
                    "tokens": cost.ravel(), "oracle": out["oracle"].ravel(), "distinct_groups": out["distinct"].ravel(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return spearmanr(x, y).statistic


def signal_analysis(d, log):
    rng = np.random.default_rng(SEED)
    n = d.n_questions
    feats = {"length_tokens": d.tokens.astype(float), "mean_logp": d.L, "self_certainty": d.K}
    if d.vlm:
        feats["vision_engagement"] = d.V
        Z, Zt, C = np.zeros_like(d.V), np.zeros_like(d.V), np.zeros_like(d.V)
        for q in range(n):
            c, zt = visconf_scores(d, q, np.arange(32))
            C[q], Zt[q], Z[q] = c, zt, d.V[q] * c
        feats["visconf"], feats["visconf_token"] = Z, Zt
        feats["calibrated_confidence"] = C
    feats["shortest (neg. length)"] = -d.tokens.astype(float)
    rows = []
    for name, f in feats.items():
        rho, qdiff = [], []
        for q in d.eval:
            c = d.correct[q].astype(float)
            mixed = 0 < c.sum() < 32
            if not mixed:
                continue
            rho.append(_spearman(f[q], c))
            order = np.argsort(f[q], kind="stable")
            qdiff.append(c[order[-8:]].mean() - c[order[:8]].mean())
        rho, qdiff = np.array(rho), np.array(qdiff)
        idx = rng.integers(0, len(qdiff), size=(BOOT, len(qdiff)))
        sel = np.array([d.correct[q, int(np.argmax(f[q]))] for q in d.eval])
        rows.append(
            {
                "dataset": d.name, "signal": name, "mixed_questions": len(qdiff),
                "spearman_with_correct": float(np.nanmean(rho)),
                "top_minus_bottom_quartile": float(qdiff.mean()),
                "quartile_lo": float(np.percentile(qdiff[idx].mean(1), 2.5)),
                "quartile_hi": float(np.percentile(qdiff[idx].mean(1), 97.5)),
                "top1_selection_accuracy_n32": float(sel.mean()),
            }
        )
    save_json(rows, OUT / f"signals_{d.name}.json")
    pd.DataFrame(rows).to_csv(OUT / f"signals_{d.name}.csv", index=False)
    if d.vlm:
        conf_vision(d, log)
    return rows


def conf_vision(d, log):
    per_roll = np.full((d.n_questions, 32), np.nan)
    gap = []
    for q in range(d.n_questions):
        segs = [d.kl[d.offs[q, r] : d.offs[q, r] + d.tokens[q, r]] for r in range(32)]
        pooled = np.concatenate(segs)
        lo, hi = np.percentile(pooled, [5, 95])
        den = max(hi - lo, 1e-8)
        for r in range(32):
            if d.tokens[q, r] < 3:
                continue
            c = np.clip((segs[r] - lo) / den, 0, 1)
            v = d.vtok[d.offs[q, r] : d.offs[q, r] + d.tokens[q, r]]
            if c.std() > 0 and v.std() > 0:
                per_roll[q, r] = pearsonr(c, v).statistic
            if q in set(d.eval[:50].tolist()):
                cov = float(((c - c.mean()) * (v - v.mean())).mean())
                gap.append(abs(float((c * v).mean() - d.V[q, r] * c.mean()) - cov))
    qmean = np.nanmean(per_roll[d.eval], axis=1)
    rho = []
    for q in d.eval:
        c, _ = visconf_scores(d, q, np.arange(32))
        rho.append(_spearman(c, d.V[q]))
    summary = {
        "mean_per_rollout_corr": float(np.nanmean(per_roll[d.eval])),
        "median_question_corr": float(np.nanmedian(qmean)),
        "fraction_questions_negative": float(np.nanmean(qmean < 0)),
        "fraction_rollouts_negative": float(np.nanmean(per_roll[d.eval] < 0)),
        "rollout_level_spearman_C_vs_V_mean": float(np.nanmean(rho)),
        "identity_max_abs_error_Ztoken_minus_Z_vs_cov": float(np.max(gap)),
    }
    np.save(OUT / "conf_vision_corr.npy", per_roll)
    save_json(summary, OUT / "conf_vision_summary.json")
    log.info("confidence-vision: %s", summary)


def save_groups(d):
    rows = []
    for q in range(d.n_questions):
        rows.append(
            {
                "qid": d.qids[q],
                "gold": d.gold[q],
                "groups": {int(g): v for g, v in d.canon[q].items()},
                "rollout_extracted": d.extracted[q],
                "rollout_group": d.groups[q].tolist(),
                "rollout_correct": d.correct[q].tolist(),
                "rollout_tokens": d.tokens[q].tolist(),
            }
        )
    save_json(rows, OUT / f"answer_groups_{d.name}.json")


def disagreement_examples(d, out):
    qr = "visconf_rwc" if d.vlm else "selfcert_rwc"
    cases = {"sc_right_quality_wrong": [], "quality_right_sc_wrong": [], "correct_available_all_missed": []}
    for q in d.eval:
        sc, ql = bool(out["correct"]["sc"][q, 0, -1]), bool(out["correct"][qr][q, 0, -1])
        any_ok = bool(d.correct[q].any())
        key = "sc_right_quality_wrong" if sc and not ql else "quality_right_sc_wrong" if ql and not sc else "correct_available_all_missed" if any_ok and not sc and not ql else None
        if key and len(cases[key]) < 3:
            order = d.orders[q, 0].astype(int)
            cases[key].append(
                {
                    "qid": d.qids[q], "gold": d.gold[q], "pool_correct_fraction": float(d.correct[q].mean()),
                    "sc_answer": d.canon[q][int(out["group"]["sc"][q, 0, -1])], f"{qr}_answer": d.canon[q][int(out["group"][qr][q, 0, -1])],
                    "answer_counts": {d.canon[q][int(g)]: int((d.groups[q] == g).sum()) for g in np.unique(d.groups[q])},
                    "first_order_rollouts": order[:8].tolist(),
                }
            )
    save_json({"rule": "first held-out questions (permuted order), order 0, N=32, by disagreement type", **cases}, OUT / f"selection_examples_{d.name}.json")


def trajectories(d, out, log, k=3):
    if not d.vlm:
        return
    picks = {}
    for q in d.eval:
        c = d.correct[q]
        z = np.zeros(32)
        cc, _ = visconf_scores(d, q, np.arange(32))
        z = d.V[q] * cc
        top = int(np.argmax(z))
        sc_correct = bool(out["correct"]["sc"][q, :, -1].mean() >= 0.5)
        vc_correct = bool(c[top])
        tag = None
        if sc_correct and vc_correct:
            tag = "agree_correct"
        elif sc_correct and not vc_correct:
            tag = "visconf_wrong_sc_right"
        elif vc_correct and not sc_correct:
            tag = "visconf_right_sc_wrong"
        if tag and tag not in picks:
            picks[tag] = int(q)
        if len(picks) == 3:
            break
    series = {}
    for tag, q in picks.items():
        cc, _ = visconf_scores(d, q, np.arange(32))
        z = d.V[q] * cc
        for role, r in (("top_visconf", int(np.argmax(z))), ("bottom_visconf", int(np.argmin(z)))):
            seg = slice(d.offs[q, r], d.offs[q, r] + d.tokens[q, r])
            kl = d.kl[seg]
            pooled = np.concatenate([d.kl[d.offs[q, j] : d.offs[q, j] + d.tokens[q, j]] for j in range(32)])
            lo, hi = np.percentile(pooled, [5, 95])
            series[f"{tag}|{role}"] = {
                "qid": d.qids[q], "rollout": r, "correct": bool(d.correct[q, r]), "answer": d.extracted[q][r],
                "c": np.clip((kl - lo) / max(hi - lo, 1e-8), 0, 1).tolist(), "v": d.vtok[seg].tolist(), "kl": kl.tolist(),
            }
    save_json({"rule": "first eval questions (permuted order) whose fixed-32 self-consistency and top-VisConf rollout are both correct, SC-right/VisConf-wrong, and VisConf-right/SC-wrong", "series": series}, OUT / "trajectories.json")
    log.info("trajectories saved: %s", list(picks))
