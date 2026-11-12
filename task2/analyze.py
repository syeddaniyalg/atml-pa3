import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common.utils import load_json, read_jsonl, save_json
from task2.config import OUT, STAGES
from task2.features import THUMBS

FIG = OUT / "figures"
TAB = OUT / "tables"
CONDS = ["correct", "complementary", "mismatched", "none"]
SUITES = {"grounding": ["presence", "counting"], "natural": ["yes/no", "number", "other"]}


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _ci(m):
    return f"{100 * m['value']:.1f} [{100 * m['lo']:.1f}, {100 * m['hi']:.1f}]"


def _ci_raw(m):
    return f"{m['value']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


def learning_curves(tokens):
    run = OUT / f"run_{tokens}"
    train = pd.DataFrame(read_jsonl(run / "train_log.jsonl"))
    dev = pd.DataFrame(read_jsonl(run / "dev_metrics.jsonl"))
    bounds = np.cumsum([s.updates for s in STAGES])[:-1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    ax = axes[0, 0]
    ax.plot(train["update"], train["loss"].rolling(25, min_periods=1).mean())
    ax.set(title="Training loss (25-update mean)", xlabel="update", ylabel="loss")
    axes[0, 1].plot(dev["update"], dev["caption_loss"], marker="o")
    axes[0, 1].set(title="Dev caption loss", xlabel="update")
    for suite, ax in zip(("grounding", "natural"), (axes[0, 2], axes[1, 0])):
        for c in ("correct", "complementary", "none"):
            ax.plot(dev["update"], dev[f"{suite}_nll_{c}"], marker="o", label=c)
        ax.set(title=f"Dev answer NLL ({suite})", xlabel="update")
        ax.legend(frameon=False, fontsize=8)
    for suite in ("grounding", "natural"):
        axes[1, 1].plot(dev["update"], dev[f"{suite}_swap_penalty"], marker="o", label=suite)
    axes[1, 1].axhline(0, color="k", lw=0.6)
    axes[1, 1].set(title="Image-swap penalty ΔNLL", xlabel="update")
    axes[1, 1].legend(frameon=False)
    for suite in ("grounding", "natural"):
        d = dev.dropna(subset=[f"{suite}_acc_correct"])
        axes[1, 2].plot(d["update"], d[f"{suite}_acc_correct"], marker="o", label=suite)
    axes[1, 2].set(title="Dev generation accuracy (correct image)", xlabel="update")
    axes[1, 2].legend(frameon=False)
    for a in axes.flat:
        for b in bounds:
            a.axvline(b, color="gray", ls="--", lw=0.8)
        a.grid(alpha=0.3)
    _save(fig, f"learning_curves_{tokens}")


def final_tables(tokens):
    res = load_json(OUT / f"run_{tokens}" / "final_metrics.json")
    TAB.mkdir(parents=True, exist_ok=True)
    for suite, subs in SUITES.items():
        rows = []
        for c in CONDS:
            m = res["metrics"][suite][c]
            row = {"condition": c, "accuracy %": _ci(m["accuracy"])}
            for s in subs:
                row[f"{s} %"] = _ci(m[f"accuracy_{s}"]) if f"accuracy_{s}" in m else ""
            row["pair acc %"] = _ci(m["pair_accuracy"])
            row["identical %"] = _ci(m["identical_answer_rate"])
            row["empty %"] = _ci(m["empty_rate"])
            row["follows partner %"] = _ci(m["follows_partner_rate"])
            row["answer NLL"] = _ci_raw(m["answer_nll"])
            row["sec/answer"] = round(m["generation_seconds_per_answer"], 4)
            rows.append(row)
        pd.DataFrame(rows).to_csv(TAB / f"final_{suite}_{tokens}.csv", index=False)
    rows = []
    for suite in SUITES:
        for k, m in res["advantages"][suite].items():
            rows.append({"suite": suite, "contrast": k, "value": _ci_raw(m)})
    pd.DataFrame(rows).to_csv(TAB / f"advantages_{tokens}.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, suite in zip(axes, SUITES):
        vals = [res["metrics"][suite][c]["accuracy"] for c in CONDS]
        y = np.array([100 * v["value"] for v in vals])
        err = np.array([[100 * (v["value"] - v["lo"]) for v in vals], [100 * (v["hi"] - v["value"]) for v in vals]])
        ax.bar(CONDS, y, yerr=err, capsize=3, color=["#4c72b0", "#dd8452", "#55a868", "#8172b3"])
        ax.set(title=f"{suite} accuracy", ylabel="%")
        ax.grid(axis="y", alpha=0.3)
    _save(fig, f"final_accuracy_{tokens}")
    return res


def resources_table():
    rows = []
    feat = OUT / "feature_cache_resources.json"
    for tokens in (196, 49):
        p = OUT / f"run_{tokens}" / "train_resources.json"
        f = OUT / f"run_{tokens}" / "final_metrics.json"
        if not p.exists():
            continue
        r = load_json(p)
        for name, s in r["stages"].items():
            rows.append(
                {
                    "tokens": tokens,
                    "stage": name,
                    "updates": s["updates"],
                    "train time (min)": round(s["seconds_excluding_dev_and_ckpt"] / 60, 2),
                    "peak mem (GB)": round(s["peak_mem_gb"], 2),
                    "trainable connector": s["trainable"]["connector"],
                    "trainable LoRA": s["trainable"]["lora"],
                    "weights (MB)": round(s["weights_bytes"] / 2**20, 1),
                }
            )
        if f.exists():
            e = load_json(f)
            rows.append({"tokens": tokens, "stage": "final eval", "train time (min)": round(e["eval_seconds"] / 60, 2), "peak mem (GB)": round(e["eval_peak_mem_gb"], 2)})
    if feat.exists():
        c = load_json(feat)
        rows.append({"tokens": "-", "stage": "feature cache", "train time (min)": round((c["download_decode_seconds"] + c["encoder_seconds"]) / 60, 2), "peak mem (GB)": round(c["peak_mem_gb"], 2), "weights (MB)": round(c["feature_bytes"] / 2**20, 1)})
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "resources.csv", index=False)
    return df


def compression_table():
    if not (OUT / "run_49" / "final_metrics.json").exists():
        return
    rows = []
    for tokens in (196, 49):
        res = load_json(OUT / f"run_{tokens}" / "final_metrics.json")
        for suite in SUITES:
            m = res["metrics"][suite]["correct"]
            adv = res["advantages"][suite]["accuracy_correct_minus_complementary"]
            rows.append({"tokens": tokens, "suite": suite, "accuracy %": _ci(m["accuracy"]), "pair acc %": _ci(m["pair_accuracy"]), "advantage vs swap (pp)": _ci(adv), "sec/answer": round(m["generation_seconds_per_answer"], 4)})
    pd.DataFrame(rows).to_csv(TAB / "compression_196_vs_49.csv", index=False)


def qualitative(tokens, per_bucket=2):
    run = OUT / f"run_{tokens}"
    picks = {}
    rows = {}
    for suite in SUITES:
        pred = pd.DataFrame(read_jsonl(run / f"final_predictions_{suite}.jsonl"))
        piv = {c: pred[pred.condition == c].set_index(["pair_id", "slot"]) for c in CONDS}
        pair_ids = sorted(pred.pair_id.unique())
        buckets = {"image_driven": [], "ignores_image": [], "wrong_with_correct_image": []}
        for pid in pair_ids:
            cor = [piv["correct"].loc[(pid, s)] for s in (0, 1)]
            com = [piv["complementary"].loc[(pid, s)] for s in (0, 1)]
            if all(r["exact"] == 1 for r in cor) and all(r["follows_partner"] == 1 for r in com):
                buckets["image_driven"].append(pid)
            elif all(r["exact"] == 1 for r in cor) and com[0]["prediction"] == cor[0]["prediction"] and com[1]["prediction"] == cor[1]["prediction"]:
                buckets["ignores_image"].append(pid)
            elif any(r["exact"] == 0 for r in cor):
                buckets["wrong_with_correct_image"].append(pid)
        picks[suite] = {k: v[:per_bucket] for k, v in buckets.items()}
        rows[suite] = {c: piv[c] for c in CONDS}
    save_json({"rule": "per suite and bucket, the first pairs in sorted pair-id order; image_driven: both correct-image answers match references and both swapped-image answers equal the partner's reference; ignores_image: both correct and swapped predictions identical; wrong_with_correct_image: at least one incorrect answer with the original image", "picks": picks}, TAB / f"qualitative_picks_{tokens}.json")
    examples = [(s, b, pid) for s, bk in picks.items() for b, pids in bk.items() for pid in pids]
    if not examples:
        return
    cols = 4
    fig, axes = plt.subplots(len(examples), cols, figsize=(3.2 * cols, 3.4 * len(examples)))
    axes = np.atleast_2d(axes)
    for r, (suite, bucket, pid) in enumerate(examples):
        piv = rows[suite]
        slot_rows = [piv["correct"].loc[(pid, s)] for s in (0, 1)]
        for s in (0, 1):
            ax = axes[r, s]
            key = slot_rows[s]["image"].replace(":", "_")
            path = THUMBS / f"{key}.jpg"
            if path.exists():
                ax.imshow(Image.open(path))
            ax.axis("off")
            ref = slot_rows[s]["target"]
            ax.set_title(f"ref: {ref}\ncorrect: {piv['correct'].loc[(pid, s)]['prediction']} | swap: {piv['complementary'].loc[(pid, s)]['prediction']}\nnone: {piv['none'].loc[(pid, s)]['prediction']}", fontsize=7)
        axes[r, 2].axis("off")
        axes[r, 3].axis("off")
        axes[r, 2].text(0, 0.5, f"{suite} / {bucket}\n{slot_rows[0]['question']}", fontsize=7, wrap=True, va="center")
    _save(fig, f"qualitative_{tokens}")


def main():
    for tokens in (196, 49):
        if (OUT / f"run_{tokens}" / "final_metrics.json").exists():
            learning_curves(tokens)
            final_tables(tokens)
            qualitative(tokens)
    resources_table()
    compression_table()


if __name__ == "__main__":
    main()
