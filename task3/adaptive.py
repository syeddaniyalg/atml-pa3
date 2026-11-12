import numpy as np
import pandas as pd

from common.utils import save_json
from task3.config import BOOT, BUDGETS, DELTA_GRID, OUT, SEED, STOP_POINTS, TAU_GRID

B32 = len(BUDGETS) - 1
STOP_IDX = [BUDGETS.index(n) for n in STOP_POINTS]


def simulate(out, tau=None, delta=None):
    aux = out["aux"]
    Q, O, _ = out["cost"].shape
    stop = np.full((Q, O), B32)
    if tau is not None:
        n = np.array(BUDGETS)
        cond = (aux["sc_top_count"] / n >= tau - 1e-12) & (aux["sc_top_count"] > aux["sc_second_count"]) & (aux["sc_group"] != 0)
        if delta is not None:
            cond &= (aux["sc_group"] == aux["quality_group"]) & (aux["quality_margin"] >= delta - 1e-12)
        for bi in reversed(STOP_IDX):
            stop = np.where(cond[:, :, bi], bi, stop)
    correct = np.take_along_axis(out["correct"]["sc"], stop[:, :, None], 2)[..., 0]
    cost = np.take_along_axis(out["cost"], stop[:, :, None], 2)[..., 0]
    rollouts = np.array(BUDGETS)[stop]
    return {"stop": stop, "correct": correct, "cost": cost, "rollouts": rollouts}


def summarize(sim, idx):
    return {
        "accuracy": float(sim["correct"][idx].mean()),
        "tokens": float(sim["cost"][idx].mean()),
        "rollouts": float(sim["rollouts"][idx].mean()),
    }


def grid(out, idx):
    rows = [{"family": "fixed32", "tau": np.nan, "delta": np.nan, **summarize(simulate(out), idx)}]
    for tau in TAU_GRID:
        rows.append({"family": "agreement", "tau": tau, "delta": np.nan, **summarize(simulate(out, tau), idx)})
        for delta in DELTA_GRID:
            rows.append({"family": "quality", "tau": tau, "delta": delta, **summarize(simulate(out, tau, delta), idx)})
    return pd.DataFrame(rows)


def select(dev_grid):
    base = dev_grid[dev_grid.family == "fixed32"].iloc[0]
    chosen = {}
    for family in ("agreement", "quality"):
        sub = pd.concat([dev_grid[dev_grid.family == family], dev_grid[dev_grid.family == "fixed32"]])
        ok = sub[sub.accuracy >= base.accuracy - 0.01 - 1e-12].copy()
        ok["tau_key"] = ok.tau.fillna(9.0)
        ok["delta_key"] = ok.delta.fillna(9.0)
        best = ok.sort_values(["tokens", "accuracy", "tau_key", "delta_key"], ascending=[True, False, True, True]).iloc[0]
        chosen[family] = {"tau": None if np.isnan(best.tau) else float(best.tau), "delta": None if np.isnan(best.delta) else float(best.delta), "dev_accuracy": float(best.accuracy), "dev_tokens": float(best.tokens), "fallback_fixed32": bool(best.family == "fixed32")}
    return chosen


def boot_ci(per_q_a, per_q_b, rng):
    diff = per_q_a - per_q_b
    idx = rng.integers(0, len(diff), size=(BOOT, len(diff)))
    m = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def run(d, out, log):
    rng = np.random.default_rng(SEED)
    dev_grid = grid(out, d.dev)
    dev_grid.insert(0, "dataset", d.name)
    dev_grid.to_csv(OUT / f"adaptive_dev_grid_{d.name}.csv", index=False)
    chosen = select(dev_grid)
    save_json(chosen, OUT / f"adaptive_selected_{d.name}.json")
    log.info("%s adaptive selected on dev: %s", d.name, chosen)

    fixed = simulate(out)
    sims = {"fixed32": fixed}
    for family, cfg in chosen.items():
        sims[family] = simulate(out, cfg["tau"], cfg["delta"]) if cfg["tau"] is not None else fixed
    rows, dist_rows, fail_rows = [], [], []
    ev = d.eval
    for name, sim in sims.items():
        rows.append({"dataset": d.name, "policy": name, **summarize(sim, ev)})
        for bi, n in enumerate(BUDGETS):
            if n in (*STOP_POINTS, 32):
                dist_rows.append({"dataset": d.name, "policy": name, "stop_at": n, "fraction": float((sim["stop"][ev] == bi).mean())})
    for bi, n in enumerate(BUDGETS):
        sc = {"correct": out["correct"]["sc"][:, :, bi], "cost": out["cost"][:, :, bi], "rollouts": np.full(out["cost"].shape[:2], n)}
        rows.append({"dataset": d.name, "policy": f"fixed_sc_N{n}", **summarize(sc, ev)})
    for name in ("agreement", "quality"):
        qa = sims[name]["correct"][ev].mean(1)
        qb = fixed["correct"][ev].mean(1)
        mean, lo, hi = boot_ci(qa, qb, rng)
        rows.append({"dataset": d.name, "policy": f"{name}_minus_fixed32", "accuracy": mean, "accuracy_lo": lo, "accuracy_hi": hi, "tokens": float((sims[name]["cost"][ev] - fixed["cost"][ev]).mean()), "rollouts": float((sims[name]["rollouts"][ev] - 32).mean())})
    pool_frac = d.correct.mean(1)
    for name in ("agreement", "quality"):
        sim = sims[name]
        early = sim["stop"] < B32
        wrong_early = early & ~sim["correct"]
        recovered = wrong_early & out["correct"]["sc"][:, :, B32]
        mask = np.zeros_like(early)
        mask[ev] = True
        fail_rows.append(
            {
                "dataset": d.name, "policy": name,
                "early_stop_rate": float(early[ev].mean()),
                "incorrect_early_stop_rate": float(wrong_early[ev].mean()),
                "precision_of_early_stops": float(sim["correct"][early & mask].mean()) if (early & mask).any() else np.nan,
                "incorrect_early_but_fixed32_correct": float(recovered[ev].mean()),
                "incorrect_early_and_fixed32_incorrect": float((wrong_early & ~out["correct"]["sc"][:, :, B32])[ev].mean()),
            }
        )
        for lo_f, hi_f in ((0, 0.34), (0.34, 0.67), (0.67, 1.01)):
            sel = np.isin(np.arange(len(pool_frac)), ev) & (pool_frac >= lo_f) & (pool_frac < hi_f)
            if sel.any():
                fail_rows.append({"dataset": d.name, "policy": name, "pool_correct_fraction_bin": f"[{lo_f:.2f},{min(hi_f, 1):.2f})", "questions": int(sel.sum()), "accuracy": float(sim["correct"][sel].mean()), "mean_rollouts": float(sim["rollouts"][sel].mean()), "fixed32_accuracy": float(fixed["correct"][sel].mean())})
    pd.DataFrame(rows).to_csv(OUT / f"adaptive_heldout_{d.name}.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(OUT / f"adaptive_stop_distribution_{d.name}.csv", index=False)
    pd.DataFrame(fail_rows).to_csv(OUT / f"adaptive_failures_{d.name}.csv", index=False)

    cases = []
    sim = sims["quality"]
    for q in ev:
        for o in range(sim["stop"].shape[1]):
            if sim["stop"][q, o] < B32 and not sim["correct"][q, o] and out["correct"]["sc"][q, o, B32]:
                cases.append({"qid": d.qids[q], "order": int(o), "stopped_at": int(sim["rollouts"][q, o]), "pool_correct_fraction": float(pool_frac[q])})
                break
        if len(cases) >= 5:
            break
    save_json({"rule": "first held-out questions (permuted order) with a quality-aware early stop that is incorrect while fixed-32 self-consistency is correct", "cases": cases}, OUT / f"adaptive_premature_{d.name}.json")
    return chosen
