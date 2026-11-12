import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.log import fmt_metrics, get_logger, line, progress
from common.utils import Jsonl, env_info, peak_mem_gb, reset_peak, save_json, seed_all, set_child_env
from task3.answers import grade_math_groups, group_math, last_boxed, math_equal
from task3.config import INSTRUCTION, LIVE_MODEL, LIVE_REVISION, N_LIVE, OUT, REVIEW, SEED, SYSTEM

SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)


def load_questions(limit=None):
    import pyarrow.parquet as pq
    from task3.config import download_data
    from task3.data import ensure_splits

    ids = ensure_splits("math")[0]["live"][: limit or N_LIVE]

    t = pq.read_table(download_data() / "qwen25-math" / "generations.parquet", columns=["sample_id", "rollout_id", "question_text", "gold_answer"]).to_pandas()
    t = t[t.rollout_id == 0].set_index("sample_id")
    return [{"qid": q, "problem": t.loc[q, "question_text"], "gold": t.loc[q, "gold_answer"]} for q in ids]


class Generator:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(LIVE_MODEL, revision=LIVE_REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(LIVE_MODEL, revision=LIVE_REVISION, torch_dtype=torch.float16).cuda().eval()
        self.eos = [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")]

    @torch.no_grad()
    def __call__(self, messages, max_new, seed):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to("cuda")
        seed_all(seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(**enc, max_new_tokens=max_new, pad_token_id=self.tok.eos_token_id, eos_token_id=self.eos, **SAMPLING)
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        new = out[0, enc["input_ids"].shape[1] :].tolist()
        ended = any(t in self.eos for t in new)
        if ended:
            new = new[: min(new.index(t) for t in self.eos if t in new)]
        return {
            "prompt": text,
            "response": self.tok.decode(new, skip_special_tokens=True),
            "in_tokens": enc["input_ids"].shape[1],
            "out_tokens": len(new),
            "seconds": secs,
            "truncated": not ended,
            "seed": seed,
            "max_new": max_new,
        }


def user_message(problem):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": problem + "\n\n" + INSTRUCTION}]


def run(limit=None):
    log = get_logger("task3-live", OUT / "logs" / "live.log")
    log.info("env: %s", env_info())
    questions = load_questions(limit)
    gen = Generator()
    gen(user_message("1+1?"), 8, SEED)
    records = OUT / "live_records.jsonl"
    finished = {json.loads(x)["qid"] for x in open(records, encoding="utf-8")} if records.exists() else set()
    writer = Jsonl(records, "a")
    bar = progress(len(questions), "live MATH", unit="q")
    totals = {"long": 0.0, "parallel": 0.0, "refine": 0.0}
    for qi, q in enumerate(questions):
        if q["qid"] in finished:
            bar.update(1)
            continue
        base = SEED * 1000 + qi * 10
        msgs = user_message(q["problem"])
        reset_peak()
        long = gen(msgs, 3072, base)
        parallel = [gen(msgs, 1024, base + 1 + k) for k in range(3)]
        initial = parallel[0]
        rev_msgs = msgs + [{"role": "assistant", "content": initial["response"]}, {"role": "user", "content": REVIEW}]
        revised = gen(rev_msgs, 2048, base + 5)
        answers = {"long": last_boxed(long["response"]), "parallel": [last_boxed(p["response"]) for p in parallel], "initial": last_boxed(initial["response"]), "revised": last_boxed(revised["response"])}
        labels, canon = group_math(answers["parallel"])
        ok = grade_math_groups(canon, q["gold"])
        counts = np.bincount(labels)
        first = {g: labels.index(g) for g in set(labels)}
        best = max(set(labels), key=lambda g: (counts[g], -first[g]))
        sc_correct = bool(ok[best])
        correct = {
            "long": bool(answers["long"] and math_equal(answers["long"], q["gold"])),
            "parallel_sc": sc_correct,
            "initial": bool(answers["initial"] and math_equal(answers["initial"], q["gold"])),
            "refined": bool(answers["revised"] and math_equal(answers["revised"], q["gold"])),
        }
        row = {"qid": q["qid"], "gold": q["gold"], "answers": answers, "correct": correct, "long": long, "parallel": parallel, "revision": revised, "revision_prompt_tokens": revised["in_tokens"], "peak_mem_gb": peak_mem_gb()}
        writer.write(**row)
        totals["long"] += long["seconds"]
        totals["parallel"] += sum(p["seconds"] for p in parallel)
        totals["refine"] += initial["seconds"] + revised["seconds"]
        shown = {k: v for k, v in {"long": int(correct["long"]), "sc3": int(sc_correct), "refined": int(correct["refined"]), "t_long": totals["long"], "t_par": totals["parallel"], "t_ref": totals["refine"], "mem": peak_mem_gb()}.items()}
        bar.set_postfix(fmt_metrics(shown), refresh=False)
        bar.update(1)
        log.info("[%d/%d] %s", qi + 1, len(questions), line(shown))
    bar.close()
    writer.close()
    summarize()


def summarize():
    rows = [json.loads(x) for x in open(OUT / "live_records.jsonl", encoding="utf-8")]
    n = len(rows)

    def strat(name):
        if name == "long":
            calls = lambda r: [r["long"]]
            correct = lambda r: r["correct"]["long"]
        elif name == "parallel":
            calls = lambda r: r["parallel"]
            correct = lambda r: r["correct"]["parallel_sc"]
        else:
            calls = lambda r: [r["parallel"][0], r["revision"]]
            correct = lambda r: r["correct"]["refined"]
        acc = np.mean([correct(r) for r in rows])
        cs = [c for r in rows for c in calls(r)]
        ans_key = {"long": lambda r: r["answers"]["long"], "parallel": lambda r: r["answers"]["parallel"], "refine": lambda r: r["answers"]["revised"]}[name]
        if name == "parallel":
            extract = np.mean([a is not None for r in rows for a in ans_key(r)])
        else:
            extract = np.mean([ans_key(r) is not None for r in rows])
        return {
            "strategy": name,
            "accuracy": float(acc),
            "input_tokens_per_q": sum(c["in_tokens"] for c in cs) / n,
            "output_tokens_per_q": sum(c["out_tokens"] for c in cs) / n,
            "seconds_per_q": sum(c["seconds"] for c in cs) / n,
            "truncation_rate_per_response": float(np.mean([c["truncated"] for c in cs])),
            "extraction_rate": float(extract),
            "peak_mem_gb": float(max(r["peak_mem_gb"] for r in rows)),
        }

    table = pd.DataFrame([strat(s) for s in ("long", "parallel", "refine")])
    rng = np.random.default_rng(SEED)
    boot = {}
    c_long = np.array([r["correct"]["long"] for r in rows], float)
    c_par = np.array([r["correct"]["parallel_sc"] for r in rows], float)
    c_ref = np.array([r["correct"]["refined"] for r in rows], float)
    for name, a, b in (("parallel_minus_long", c_par, c_long), ("refine_minus_long", c_ref, c_long), ("refine_minus_parallel", c_ref, c_par)):
        idx = rng.integers(0, n, size=(1000, n))
        m = (a - b)[idx].mean(1)
        boot[name] = {"diff": float((a - b).mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}
    w2c = sum(1 for r in rows if not r["correct"]["initial"] and r["correct"]["refined"])
    c2w = sum(1 for r in rows if r["correct"]["initial"] and not r["correct"]["refined"])
    trans = {
        "wrong_to_correct": w2c,
        "correct_to_wrong": c2w,
        "correct_to_correct": sum(1 for r in rows if r["correct"]["initial"] and r["correct"]["refined"]),
        "wrong_to_wrong": sum(1 for r in rows if not r["correct"]["initial"] and not r["correct"]["refined"]),
        "answer_changed": sum(1 for r in rows if r["answers"]["initial"] != r["answers"]["revised"]),
        "initial_accuracy": float(np.mean([r["correct"]["initial"] for r in rows])),
        "revision_prompt_tokens_mean": float(np.mean([r["revision_prompt_tokens"] for r in rows])),
    }
    ex = {"wrong_to_correct": [], "correct_to_wrong": []}
    for r in rows:
        key = "wrong_to_correct" if (not r["correct"]["initial"] and r["correct"]["refined"]) else "correct_to_wrong" if (r["correct"]["initial"] and not r["correct"]["refined"]) else None
        if key and len(ex[key]) < 3:
            ex[key].append({"qid": r["qid"], "gold": r["gold"], "initial_answer": r["answers"]["initial"], "revised_answer": r["answers"]["revised"], "revision_excerpt": r["revision"]["response"][:500]})
    save_json({"rule": "first questions in the live-subset order whose refinement changed correctness", **ex}, OUT / "revision_examples.json")
    table.to_csv(OUT / "live_strategies.csv", index=False)
    save_json({"bootstrap_accuracy_differences": boot, "transitions": trans, "questions": n}, OUT / "live_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    set_child_env()
    if args.summary_only:
        summarize()
    else:
        run(args.limit or None)
