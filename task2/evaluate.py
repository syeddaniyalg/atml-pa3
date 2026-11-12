import time

import numpy as np
import torch

from common.log import progress
from common.utils import Jsonl, load_json, run_batched, save_json
from task2.dataset import build_eval_items, caption_examples
from task2.vqa_norm import exact_match, normalize, soft_accuracy

CONDITIONS = ("correct", "complementary", "mismatched", "none")
SUBTYPES = {"grounding": ("presence", "counting"), "natural": ("yes/no", "number", "other")}


class FeatureStore:
    def __init__(self, feats, index, device):
        self.feats, self.index, self.device = feats, index, device

    def get(self, keys):
        rows = [self.index[k] for k in keys]
        arr = np.stack([self.feats[r] for r in rows])
        return torch.from_numpy(arr).to(self.device)


def suite_pairs(splits, part, suite):
    if suite == "grounding":
        return splits["presence"][part] + splits["counting"][part]
    return splits["natural"][part]


def condition_keys(items, condition):
    if condition == "correct":
        return [i["image"] for i in items]
    if condition == "complementary":
        return [i["partner"] for i in items]
    if condition == "mismatched":
        return [i["mismatched"] for i in items]
    return None


@torch.no_grad()
def answer_nll(model, store, items, condition, bs=16):
    keys = condition_keys(items, condition)

    def step(s, e):
        chunk = items[s:e]
        feats = store.get(keys[s:e]) if keys is not None else None
        targets = [model.target_ids(i["target"], "short") for i in chunk]
        with torch.autocast("cuda", dtype=torch.float16, enabled=model.device.type == "cuda"):
            return model.example_loss(feats, [i["question"] for i in chunk], targets, drop_end=True).float().cpu().tolist()

    return run_batched(step, len(items), bs, label="answer_nll")


@torch.no_grad()
def generate_all(model, store, items, condition, bs, max_new):
    keys = condition_keys(items, condition)
    t0 = time.perf_counter()

    def step(s, e):
        feats = store.get(keys[s:e]) if keys is not None else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=model.device.type == "cuda"):
            return model.generate(feats, [i["question"] for i in items[s:e]], max_new)

    preds = run_batched(step, len(items), bs, label="generate")
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    return preds, time.perf_counter() - t0


def score_items(items, preds, suite):
    rows = []
    for it, pred in zip(items, preds):
        norm = normalize(pred)
        rows.append(
            {
                "score": soft_accuracy(pred, it["refs"]) if suite == "natural" else exact_match(pred, it["target"]),
                "exact": float(norm == normalize(it["target"])),
                "empty": float(pred.strip() == ""),
                "follows_partner": float(norm == normalize(it["partner_target"])),
                "norm": norm,
            }
        )
    return rows


def dev_eval(model, store, splits, cfg, with_generation=True):
    model.eval()
    res = {}
    caps = caption_examples(splits["caption"]["dev"], "val")
    losses = []
    with torch.no_grad():
        def cap_step(s, e):
            chunk = caps[s:e]
            feats = store.get([c.image for c in chunk])
            tg = [model.target_ids(c.target, "caption") for c in chunk]
            with torch.autocast("cuda", dtype=torch.float16, enabled=model.device.type == "cuda"):
                return model.example_loss(feats, [c.question for c in chunk], tg).float().cpu().tolist()

        losses = run_batched(cap_step, len(caps), 16, label="dev_caption")
    res["caption_loss"] = float(np.mean(losses))
    for suite in ("grounding", "natural"):
        items = build_eval_items(suite_pairs(splits, "dev", suite))
        nll = {c: np.array(answer_nll(model, store, items, c)) for c in ("correct", "complementary", "none")}
        for c, v in nll.items():
            res[f"{suite}_nll_{c}"] = float(v.mean())
        res[f"{suite}_swap_penalty"] = float((nll["complementary"] - nll["correct"]).mean())
        res[f"{suite}_noimg_penalty"] = float((nll["none"] - nll["correct"]).mean())
        if with_generation:
            preds, _ = generate_all(model, store, items, "correct", cfg.gen_bs, cfg.max_new)
            res[f"{suite}_acc_correct"] = float(np.mean([r["score"] for r in score_items(items, preds, suite)]))
    return res


def bootstrap(values, boot, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(boot, len(values)))
    means = values[idx].mean(1)
    return {"value": float(values.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5))}


def final_eval(model, store, splits, cfg, out_dir, log):
    model.eval()
    metrics, advantages, all_rows = {}, {}, {}
    for suite in ("grounding", "natural"):
        pairs = suite_pairs(splits, "test", suite)
        items = build_eval_items(pairs)
        pair_ids = [p["id"] for p in pairs]
        pos = {pid: i for i, pid in enumerate(pair_ids)}
        n_pairs = len(pairs)
        per_pair = {}
        suite_metrics = {}
        writer = Jsonl(out_dir / f"final_predictions_{suite}.jsonl", "w")
        bar = progress(len(CONDITIONS), f"final {suite}", unit="cond")
        for cond in CONDITIONS:
            cpath = out_dir / "eval_cache" / f"{suite}_{cond}.json"
            cpath.parent.mkdir(parents=True, exist_ok=True)
            if cpath.exists():
                cached = load_json(cpath)
                preds, secs, nll = cached["preds"], cached["secs"], cached["nll"]
                log.info("[%s/%s] reused from cache", suite, cond)
            else:
                preds, secs = generate_all(model, store, items, cond, cfg.gen_bs, cfg.max_new)
                nll = answer_nll(model, store, items, cond)
                save_json({"preds": preds, "secs": secs, "nll": nll}, cpath)
            scored = score_items(items, preds, suite)
            for it, pred, sc, nl in zip(items, preds, scored, nll):
                writer.write(suite=suite, condition=cond, pair_id=it["pair_id"], slot=it["slot"], subtype=it["subtype"], question=it["question"], image=it["image"], partner=it["partner"], mismatched=it["mismatched"], target=it["target"], partner_target=it["partner_target"], refs=it["refs"], prediction=pred, nll=nl, **{k: v for k, v in sc.items() if k != "norm"})
            arr = {k: np.zeros((n_pairs, 2)) for k in ("score", "exact", "empty", "follows_partner", "nll")}
            norms = [[None, None] for _ in range(n_pairs)]
            for it, sc, nl in zip(items, scored, nll):
                r = pos[it["pair_id"]]
                for k in ("score", "exact", "empty", "follows_partner"):
                    arr[k][r, it["slot"]] = sc[k]
                arr["nll"][r, it["slot"]] = nl
                norms[r][it["slot"]] = sc["norm"]
            pair_vals = {
                "accuracy": arr["score"].mean(1),
                "pair_accuracy": (arr["exact"].min(1)),
                "identical_answer_rate": np.array([float(a == b) for a, b in norms]),
                "empty_rate": arr["empty"].mean(1),
                "follows_partner_rate": arr["follows_partner"].mean(1),
                "answer_nll": arr["nll"].mean(1),
            }
            per_pair[cond] = pair_vals
            sub_of_pair = np.array([items[2 * r]["subtype"] for r in range(n_pairs)])
            cm = {k: bootstrap(v, cfg.bootstrap, 6304) for k, v in pair_vals.items()}
            for st in SUBTYPES[suite]:
                mask = sub_of_pair == st
                if mask.any():
                    cm[f"accuracy_{st}"] = bootstrap(pair_vals["accuracy"][mask], cfg.bootstrap, 6304)
                    cm[f"pair_accuracy_{st}"] = bootstrap(pair_vals["pair_accuracy"][mask], cfg.bootstrap, 6304)
            cm["generation_seconds_per_answer"] = secs / len(items)
            cm["n_pairs"], cm["n_questions"] = n_pairs, len(items)
            suite_metrics[cond] = cm
            bar.update(1)
            log.info("[%s/%s] acc=%.3f pair=%.3f identical=%.3f empty=%.3f nll=%.3f", suite, cond, cm["accuracy"]["value"], cm["pair_accuracy"]["value"], cm["identical_answer_rate"]["value"], cm["empty_rate"]["value"], cm["answer_nll"]["value"])
        bar.close()
        writer.close()
        adv = {}
        for other in ("complementary", "mismatched", "none"):
            adv[f"accuracy_correct_minus_{other}"] = bootstrap(per_pair["correct"]["accuracy"] - per_pair[other]["accuracy"], cfg.bootstrap, 6304)
            adv[f"nll_{other}_minus_correct"] = bootstrap(per_pair[other]["answer_nll"] - per_pair["correct"]["answer_nll"], cfg.bootstrap, 6304)
        metrics[suite], advantages[suite] = suite_metrics, adv
        all_rows[suite] = per_pair
    return {"metrics": metrics, "advantages": advantages}
