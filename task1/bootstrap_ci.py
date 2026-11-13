"""Paired bootstrap CIs (over benchmark examples) for Task 1 differences.

Reads the saved per-example outputs in results/task1/eval/* and writes
results/task1/tables/bootstrap_ci.{csv,md}.  Run: python -m task1.bootstrap_ci
"""
import json

import numpy as np
import pandas as pd

from common.utils import read_jsonl
from task1.config import OUT, SEED

BOOT = 2000
METRICS = {"GSM8K": ("gsm8k", None), "HellaSwag": ("hellaswag", "acc_norm"), "BoolQ": ("boolq", "acc"), "ARC-E": ("arc_easy", "acc_norm")}
VS_BASE = ["fullft_mid", "fullft_final", "l2sp_final", "lora4_final", "lora16_final", "oplora16_final"]
CONTROLLED = [
    ("rank", "lora16_final", "lora4_final"),
    ("explicit regularization", "l2sp_final", "fullft_final"),
    ("orthogonal protection", "oplora16_final", "lora16_final"),
    ("full vs low-rank", "lora16_final", "fullft_final"),
    ("full FT progression", "fullft_final", "fullft_mid"),
]


def load(cond, task, key):
    d = OUT / "eval" / cond
    if task == "gsm8k":
        return {r["doc_id"]: float(r["correct_flexible"]) for r in read_jsonl(d / "gsm8k_samples.jsonl")}
    return {r["doc_id"]: float(r[key]) for r in read_jsonl(d / "likelihood_samples.jsonl") if r["task"] == task}


def paired_ci(a, b, rng):
    ids = sorted(a)
    diff = np.array([a[i] - b[i] for i in ids])
    idx = rng.integers(0, len(diff), size=(BOOT, len(diff)))
    m = diff[idx].mean(1)
    return 100 * diff.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


def main():
    rng = np.random.default_rng(SEED)
    cache = {}

    def get(cond, metric):
        if (cond, metric) not in cache:
            cache[(cond, metric)] = load(cond, *METRICS[metric])
        return cache[(cond, metric)]

    rows = []
    for name, a, b in [("vs baseline", c, "baseline") for c in VS_BASE] + CONTROLLED:
        for metric in METRICS:
            d, lo, hi = paired_ci(get(a, metric), get(b, metric), rng)
            rows.append({"comparison": name if name != "vs baseline" else f"{a} vs baseline", "a": a, "b": b, "metric": metric,
                         "diff_pp": round(d, 2), "lo_pp": round(lo, 2), "hi_pp": round(hi, 2), "significant": not (lo <= 0 <= hi)})
    df = pd.DataFrame(rows)
    tab = OUT / "tables"
    df.to_csv(tab / "bootstrap_ci.csv", index=False)
    lines = ["| comparison | a | b | metric | diff pp | 95% CI | significant |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['comparison']} | {r['a']} | {r['b']} | {r['metric']} | {r['diff_pp']} | [{r['lo_pp']}, {r['hi_pp']}] | {r['significant']} |")
    (tab / "bootstrap_ci.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows ({BOOT} resamples, seed {SEED})")


if __name__ == "__main__":
    main()
