import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.utils import load_json, read_jsonl, save_json
from task1.config import OUT

CONDS = ["baseline", "fullft_mid", "fullft_final", "l2sp_final", "lora4_final", "lora16_final", "oplora16_final"]
LABELS = {
    "baseline": "Baseline",
    "fullft_mid": "Full FT (mid)",
    "fullft_final": "Full FT",
    "l2sp_final": "Full FT + L2-SP",
    "lora4_final": "LoRA-4",
    "lora16_final": "LoRA-16",
    "oplora16_final": "OPLoRA-16",
}
RUNS = ["fullft", "l2sp", "lora4", "lora16", "oplora16"]
RUN_LABELS = {"fullft": "Full FT", "l2sp": "Full FT + L2-SP", "lora4": "LoRA-4", "lora16": "LoRA-16", "oplora16": "OPLoRA-16"}
PRIMARY = {
    "gsm8k_flexible_em": "GSM8K",
    "hellaswag_acc_norm": "HellaSwag",
    "boolq_acc": "BoolQ",
    "arc_easy_acc_norm": "ARC-E",
}
FIG = OUT / "figures"
TAB = OUT / "tables"


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _md(df, path):
    TAB.mkdir(parents=True, exist_ok=True)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    (TAB / path).write_text("\n".join(lines), encoding="utf-8")


def load_scores():
    scores = {}
    for c in CONDS:
        p = OUT / "eval" / c / "scores.json"
        if p.exists():
            scores[c] = load_json(p)
    return scores


def benchmark_table(scores):
    rows = []
    base = scores.get("baseline")
    for c in CONDS:
        if c not in scores:
            continue
        s = scores[c]
        row = {"condition": LABELS[c]}
        for key, name in PRIMARY.items():
            row[name] = round(100 * s[key], 2)
            if base is not None:
                row[f"{name} Δpp"] = round(100 * (s[key] - base[key]), 2)
        for key in ("hellaswag_acc", "arc_easy_acc"):
            row[key] = round(100 * s[key], 2)
        row["GSM8K strict"] = round(100 * s["gsm8k_strict_em"], 2)
        row["GSM8K extractable %"] = round(100 * s["gsm8k_extractable_rate"], 1)
        row["GSM8K truncated %"] = round(100 * s["gsm8k_truncated_rate"], 1)
        row["mean retention Δpp"] = (
            round(np.mean([100 * (s[k] - base[k]) for k in list(PRIMARY)[1:]]), 2) if base is not None else None
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "benchmarks.csv", index=False)
    _md(df, "benchmarks.md")
    return df


COMPARISONS = [
    ("rank", "lora16_final", "lora4_final"),
    ("explicit regularization", "l2sp_final", "fullft_final"),
    ("orthogonal protection", "oplora16_final", "lora16_final"),
    ("full vs low-rank", "lora16_final", "fullft_final"),
    ("full FT progression", "fullft_final", "fullft_mid"),
]


def comparison_table(scores):
    rows = []
    for name, a, b in COMPARISONS:
        if a not in scores or b not in scores:
            continue
        row = {"comparison": name, "a": LABELS[a], "b": LABELS[b]}
        for key, short in PRIMARY.items():
            row[f"{short} (a-b) pp"] = round(100 * (scores[a][key] - scores[b][key]), 2)
        row["GSM8K strict (a-b) pp"] = round(100 * (scores[a]["gsm8k_strict_em"] - scores[b]["gsm8k_strict_em"]), 2)
        row["GSM8K extractable (a-b) pp"] = round(100 * (scores[a]["gsm8k_extractable_rate"] - scores[b]["gsm8k_extractable_rate"]), 2)
        row["GSM8K truncated (a-b) pp"] = round(100 * (scores[a]["gsm8k_truncated_rate"] - scores[b]["gsm8k_truncated_rate"]), 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "controlled_comparisons.csv", index=False)
    _md(df, "controlled_comparisons.md")
    return df


def plot_losses():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for m in RUNS:
        p = OUT / "runs" / m / "train_log.jsonl"
        if not p.exists():
            continue
        df = pd.DataFrame(read_jsonl(p))
        smooth = df["sft_loss"].rolling(15, min_periods=1).mean()
        axes[0].plot(df["update"], smooth, label=RUN_LABELS[m])
        axes[1].plot(df["update"], df["lr"], label=RUN_LABELS[m])
    axes[0].set(xlabel="optimizer update", ylabel="SFT loss (15-update mean)", title="Training loss")
    axes[1].set(xlabel="optimizer update", ylabel="learning rate", title="Learning-rate schedule")
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    _save(fig, "loss_and_lr")


def plot_l2sp():
    p = OUT / "runs" / "l2sp" / "train_log.jsonl"
    if not p.exists():
        return
    df = pd.DataFrame(read_jsonl(p))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(df["update"], df["sft_loss"].rolling(15, min_periods=1).mean(), label="SFT")
    axes[0].plot(df["update"], df["total_loss"].rolling(15, min_periods=1).mean(), label="SFT + λΩ")
    axes[0].set(xlabel="optimizer update", ylabel="loss", title="L2-SP objective")
    axes[0].legend(frameon=False)
    axes[1].plot(df["update"], df["omega"], label="Ω")
    axes[1].plot(df["update"], df["lam_omega"], label="λΩ")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="optimizer update", title="Penalty")
    axes[1].legend(frameon=False)
    names, vals = [], []
    for m in ("fullft", "l2sp"):
        s = OUT / "runs" / m / "train_summary.json"
        if s.exists():
            names.append(RUN_LABELS[m])
            vals.append(load_json(s)["displacement_final"])
    axes[2].bar(names, vals, color=["#4c72b0", "#dd8452"][: len(vals)])
    axes[2].set(ylabel="‖θ − θ0‖₂", title="Final displacement")
    for a in axes:
        a.grid(alpha=0.3)
    _save(fig, "l2sp")


def plot_benchmarks(df):
    delta_cols = [f"{n} Δpp" for n in PRIMARY.values()]
    sub = df[df["condition"] != "Baseline"]
    x = np.arange(len(sub))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, col in enumerate(delta_cols):
        ax.bar(x + (i - 1.5) * w, sub[col], w, label=col.replace(" Δpp", ""))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["condition"], rotation=20)
    ax.set_ylabel("change from baseline (pp)")
    ax.legend(frameon=False, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "benchmark_deltas")

    fig, ax = plt.subplots(figsize=(5, 4))
    for _, r in sub.iterrows():
        ax.scatter(r["GSM8K Δpp"], r["mean retention Δpp"], s=50)
        ax.annotate(r["condition"], (r["GSM8K Δpp"], r["mean retention Δpp"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="GSM8K change (pp)", ylabel="mean change on other benchmarks (pp)", title="Adaptation vs retention")
    ax.grid(alpha=0.3)
    _save(fig, "tradeoff")

    prog = df[df["condition"].isin(["Baseline", "Full FT (mid)", "Full FT"])]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for n in PRIMARY.values():
        ax.plot(["start", "mid", "final"][: len(prog)], prog[n], marker="o", label=n)
    ax.set(ylabel="score (%)", title="Full FT progression")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "fullft_progression")


def resources_table():
    rows = []
    for m in RUNS:
        p = OUT / "runs" / m / "train_summary.json"
        if not p.exists():
            continue
        s = load_json(p)
        rows.append(
            {
                "method": RUN_LABELS[m],
                "trainable params (M)": round(s["trainable_params"] / 1e6, 3),
                "train time (h)": round(s["train_seconds"] / 3600, 3),
                "peak GPU mem (GB)": round(s["peak_mem_gb"], 2),
                "weights/adapter size (MB)": round(s.get("weights_bytes", s.get("adapter_bytes", 0)) / 2**20, 1),
                "recovery ckpt (GB)": round(s["recovery_ckpt_bytes"] / 2**30, 2),
                "skipped updates": s["skipped_updates"],
                "overhead": _overhead(s),
            }
        )
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "resources.csv", index=False)
    _md(df, "resources.md")
    return df


def _overhead(s):
    parts = []
    if "reference_bytes" in s:
        parts.append(f"ref {s['reference_bytes'] / 2**30:.2f} GB (CPU)")
    if s.get("penalty_seconds") and s["spec"]["lam"] > 0:
        parts.append(f"penalty {s['penalty_seconds'] / 60:.1f} min")
    if "basis_bytes" in s:
        parts.append(f"bases {s['basis_bytes'] / 2**20:.0f} MB, svd {s['preprocess_seconds']:.0f}s, export {s.get('export_seconds', 0):.1f}s")
    return "; ".join(parts)


def displacement_table():
    rows = []
    for m in ("fullft", "l2sp"):
        p = OUT / "runs" / m / "train_summary.json"
        if p.exists():
            s = load_json(p)
            rows.append(
                {
                    "method": RUN_LABELS[m],
                    "λ": s["spec"]["lam"],
                    "‖θ−θ0‖ mid": s.get("displacement_mid"),
                    "‖θ−θ0‖ final": s["displacement_final"],
                    "final SFT loss": s["final"].get("sft_loss"),
                    "final λΩ": s["final"].get("lam_omega"),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "displacement.csv", index=False)
    return df


def plot_diagnostics():
    p = OUT / "diagnostics.json"
    if not p.exists():
        return None
    rows = load_json(p)
    df = pd.DataFrame(rows).drop(columns=["R_math", "R_nonmath"])
    df["name"] = [("Full FT (mid)" if (m, s) == ("fullft", "mid") else RUN_LABELS[m]) for m, s in zip(df.method, df.stage)]
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "diagnostics.csv", index=False)
    order = list(dict.fromkeys(df["name"]))
    layers = sorted(df["layer"].unique())
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    w = 0.8 / len(layers)
    metrics = [("rho_k", "subspace interference ρ_k"), ("kappa", "relative update magnitude κ"), ("R_math_median", "median R(z), math"), ("R_nonmath_median", "median R(z), non-math")]
    for ax, (col, title) in zip(axes, metrics):
        for i, l in enumerate(layers):
            vals = [df[(df["name"] == n) & (df["layer"] == l)][col].mean() for n in order]
            ax.bar(np.arange(len(order)) + (i - (len(layers) - 1) / 2) * w, vals, w, label=f"layer {l}")
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    _save(fig, "diagnostics")
    return df


def _repetition(text, n=4):
    toks = text.split()
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def _json_ok(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try:
        json.loads(t)
        return True
    except Exception:
        return False


def _template_ok(prompt, text):
    names = re.findall(r"^\s*(\w+)\s*=", prompt, flags=re.M)
    return bool(names) and all(re.search(rf"\b{n}\s*=", text) for n in names)


def _math_markers(text):
    return len(re.findall(r"\\boxed|\\frac|\$\$|####|step[- ]by[- ]step|the answer is|\\\(", text, flags=re.I))


def qualitative_subset():
    rows = []
    for c in CONDS:
        p = OUT / "eval" / c / "qualitative.json"
        if not p.exists():
            continue
        data = load_json(p)
        plain = [q for q in data if q["format"] == "plain"]
        picks = {plain[0]["base"], plain[-1]["base"]}
        for q in data:
            if q["base"] not in picks:
                continue
            fmt = q["format"]
            rows.append(
                {
                    "condition": LABELS[c],
                    "question": q["base"],
                    "format": fmt,
                    "n_tokens": q["n_tokens"],
                    "truncated": q["truncated"],
                    "format_ok": _json_ok(q["response"]) if fmt == "json" else (_template_ok(q["prompt"], q["response"]) if fmt == "template" else None),
                    "repetition_4gram": round(_repetition(q["response"]), 3),
                    "math_markers": _math_markers(q["response"]),
                    "response": q["response"],
                    "annotation": "",
                }
            )
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "qualitative_annotation_worksheet.csv", index=False)
    lines = []
    for (question, fmt), group in df.groupby(["question", "format"], sort=False):
        lines.append(f"## {question} / {fmt}\n")
        for _, r in group.iterrows():
            body = r["response"].strip().replace("\n", " ")[:400]
            lines.append(f"**{r['condition']}** (tokens={r['n_tokens']}, truncated={r['truncated']}, format_ok={r['format_ok']}, math_markers={r['math_markers']}): {body}\n")
    (TAB / "qualitative_subset.md").write_text("\n".join(lines), encoding="utf-8")
    summary = (
        df.groupby("condition", sort=False)
        .agg(truncated=("truncated", "sum"), mean_tokens=("n_tokens", "mean"), math_markers=("math_markers", "sum"), mean_rep=("repetition_4gram", "mean"))
        .round(3)
    )
    summary.to_csv(TAB / "qualitative_summary.csv")
    return df


def gsm8k_style_table(scores):
    rows = []
    for c in CONDS:
        if c in scores:
            s = scores[c]
            rows.append(
                {
                    "condition": LABELS[c],
                    "flexible EM": round(100 * s["gsm8k_flexible_em"], 2),
                    "strict EM": round(100 * s["gsm8k_strict_em"], 2),
                    "extractable %": round(100 * s["gsm8k_extractable_rate"], 1),
                    "truncated %": round(100 * s["gsm8k_truncated_rate"], 1),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "gsm8k_style.csv", index=False)
    return df


def main():
    scores = load_scores()
    TAB.mkdir(parents=True, exist_ok=True)
    if scores:
        df = benchmark_table(scores)
        plot_benchmarks(df)
        gsm8k_style_table(scores)
        comparison_table(scores)
    plot_losses()
    plot_l2sp()
    resources_table()
    displacement_table()
    plot_diagnostics()
    qualitative_subset()
    save_json({"conditions_found": list(scores)}, OUT / "analysis_manifest.json")


if __name__ == "__main__":
    main()
