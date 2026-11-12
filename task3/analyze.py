import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.utils import load_json
from task3.config import OUT

FIG = OUT / "figures"
NAMES = {
    "first": "FirstAns", "sc": "Self-consistency", "logp": "Max mean log-prob", "selfcert": "Max self-certainty",
    "selfcert_rwc": "Self-cert. + RWC", "vision": "Vision only", "vision_rwc": "Vision + RWC", "visconf": "VisConf",
    "visconf_rwc": "VisConf + RWC", "visconf_token": "Tokenwise fusion", "visconf_token_rwc": "Tokenwise + RWC",
}


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def accuracy_curves(ds):
    df = pd.read_csv(OUT / f"curves_{ds}.csv")
    ev = df[df.split == "eval"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for m, g in ev.groupby("method", sort=False):
        axes[0].plot(g.N, 100 * g.accuracy, marker="o", label=NAMES.get(m, m))
        axes[1].plot(g.tokens, 100 * g.accuracy, marker="o", label=NAMES.get(m, m))
    cov = ev[ev.method == "sc"]
    axes[0].plot(cov.N, 100 * cov.coverage, "k--", label="Oracle coverage")
    axes[1].plot(cov.tokens, 100 * cov.coverage, "k--", label="Oracle coverage")
    axes[0].set(xscale="log", xlabel="candidates N", ylabel="accuracy (%)", title=f"{ds}: accuracy vs N")
    axes[0].set_xticks(cov.N)
    axes[0].set_xticklabels(cov.N)
    axes[1].set(xscale="log", xlabel="mean generated tokens", title="accuracy vs token cost")
    axes[0].legend(frameon=False, fontsize=7)
    axes[2].plot(cov.N, cov.distinct_groups, marker="o")
    axes[2].set(xscale="log", xlabel="candidates N", ylabel="distinct answer groups", title="answer diversity")
    axes[2].set_xticks(cov.N)
    axes[2].set_xticklabels(cov.N)
    for a in axes:
        a.grid(alpha=0.3)
    _save(fig, f"accuracy_curves_{ds}")

    sel = [m for m in ("first", "sc", "selfcert_rwc", "visconf_rwc") if m in set(ev.method)]
    fig, axes = plt.subplots(1, len(sel), figsize=(3.6 * len(sel), 3.4), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), sel):
        g = ev[ev.method == m]
        correct = g.accuracy.values
        ax.bar(range(len(g)), 100 * correct, label="selected correct")
        ax.bar(range(len(g)), 100 * g.correct_missed.values, bottom=100 * correct, label="correct available, missed")
        ax.bar(range(len(g)), 100 * g.no_correct_candidate.values, bottom=100 * (correct + g.correct_missed.values), label="no correct candidate")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g.N)
        ax.set(title=NAMES[m], xlabel="N")
    np.atleast_1d(axes)[0].set_ylabel("% of questions")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=7, loc="lower right")
    _save(fig, f"decomposition_{ds}")


def signals(ds):
    p = OUT / f"signals_{ds}.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    y = 100 * df.top_minus_bottom_quartile
    err = np.array([100 * (df.top_minus_bottom_quartile - df.quartile_lo), 100 * (df.quartile_hi - df.top_minus_bottom_quartile)])
    ax.barh(df.signal, y, xerr=err, capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="correctness: top − bottom quartile (pp)", title=f"{ds}: within-question signal strength")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, f"signals_{ds}")


def conf_vision():
    p = OUT / "conf_vision_corr.npy"
    if not p.exists():
        return
    arr = np.load(p)
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    ax.hist(arr[~np.isnan(arr)], bins=50)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="per-rollout corr(c_t, v_t)", ylabel="rollouts", title="Confidence–vision coupling")
    _save(fig, "conf_vision")


def trajectories():
    p = OUT / "trajectories.json"
    if not p.exists():
        return
    series = load_json(p)["series"]
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 2.2 * len(series)), sharex=False)
    for ax, (name, s) in zip(np.atleast_1d(axes), series.items()):
        k = 15
        sm = lambda x: np.convolve(x, np.ones(k) / k, mode="same")
        ax.plot(sm(s["c"]), label="calibrated confidence c")
        ax.plot(sm(s["v"]), label="visual engagement v")
        ax.set_title(f"{name}  q={s['qid']} correct={s['correct']} ans={s['answer']}", fontsize=8)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
    _save(fig, "trajectories")


def live():
    p = OUT / "live_strategies.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (col, title) in zip(axes, (("accuracy", "accuracy"), ("output_tokens_per_q", "output tokens / question"), ("seconds_per_q", "seconds / question"))):
        ax.bar(df.strategy, df[col])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    _save(fig, "live_strategies")


def adaptive(ds):
    p = OUT / f"adaptive_heldout_{ds}.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fixed = df[df.policy.str.startswith("fixed_sc_N")]
    pol = df[df.policy.isin(["agreement", "quality", "fixed32"])]
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.plot(fixed.tokens, 100 * fixed.accuracy, marker="o", label="fixed-N self-consistency")
    for _, r in pol.iterrows():
        ax.scatter(r.tokens, 100 * r.accuracy, s=60, marker="*", label=r.policy)
    ax.set(xscale="log", xlabel="mean generated tokens", ylabel="accuracy (%)", title=f"{ds}: adaptive vs fixed")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.3)
    _save(fig, f"adaptive_{ds}")


def main():
    for ds in ("math", "mathvista"):
        if (OUT / f"curves_{ds}.csv").exists():
            accuracy_curves(ds)
            signals(ds)
            adaptive(ds)
    conf_vision()
    trajectories()
    live()


if __name__ == "__main__":
    main()
