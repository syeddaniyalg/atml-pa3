import time

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM

from common.utils import Jsonl, free_cuda, load_json, save_json
from task1.config import OUT, SEED
from task1.qualitative import run_qualitative


class QwenLM(HFLM):
    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )


def _make_lm(model, tok, bs):
    return QwenLM(pretrained=model, tokenizer=tok, batch_size=bs, max_length=4096)


def _oom_backoff(run, bs, log, label):
    while True:
        try:
            return run(bs)
        except torch.OutOfMemoryError:
            free_cuda()
            if bs <= 1:
                raise
            bs = max(1, bs // 2)
            log.warning("[%s] OOM, retrying with batch size %d", label, bs)


def _simple_eval(lm, tasks, cfg, chat, gen_kwargs=None):
    return lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        apply_chat_template=chat,
        gen_kwargs=gen_kwargs,
        limit=cfg.eval_limit or None,
        log_samples=True,
        bootstrap_iters=0,
        random_seed=SEED,
        numpy_random_seed=SEED,
        torch_random_seed=SEED,
        fewshot_random_seed=SEED,
    )


def _gsm8k_rows(samples, tok, max_new):
    rows = {}
    for s in samples:
        row = rows.setdefault(s["doc_id"], {"doc_id": s["doc_id"], "target": str(s["target"])})
        resp = s["resps"][0][0] if isinstance(s["resps"][0], list) else s["resps"][0]
        row["response"] = resp
        pred = s["filtered_resps"][0]
        pred = pred[0] if isinstance(pred, list) else pred
        key = "flexible" if s["filter"] == "flexible-extract" else "strict"
        row[key] = pred
        row[f"correct_{key}"] = bool(s.get("exact_match", 0))
    for row in rows.values():
        n = len(tok(row["response"], add_special_tokens=False).input_ids)
        row["n_tokens"] = n
        row["truncated"] = n >= max_new - 1
        row["extracted"] = row.get("flexible") not in (None, "[invalid]", "")
    return list(rows.values())


def evaluate_condition(model, tok, cfg, cond, log):
    out = OUT / "eval" / cond
    out.mkdir(parents=True, exist_ok=True)
    if (out / "scores.json").exists():
        log.info("[%s] evaluation already complete, skipping", cond)
        return load_json(out / "scores.json")
    model.eval()
    model.config.use_cache = True
    start = time.perf_counter()
    scores = {}

    partial = out / "gsm8k_partial.json"
    if partial.exists() and (out / "gsm8k_samples.jsonl").exists():
        saved = load_json(partial)
        scores.update(saved["scores"])
        n_gsm = saved["n"]
        log.info("[%s] gsm8k reused from earlier run: flexible=%.4f", cond, scores["gsm8k_flexible_em"])
    else:
        log.info("[%s] gsm8k_cot (generation)", cond)
        gen = _oom_backoff(
            lambda bs: _simple_eval(_make_lm(model, tok, bs), ["gsm8k_cot"], cfg, True, f"max_gen_toks={cfg.gsm_max_new},do_sample=False"),
            cfg.gsm_bs,
            log,
            cond,
        )
        g = gen["results"]["gsm8k_cot"]
        scores["gsm8k_flexible_em"] = float(g["exact_match,flexible-extract"])
        scores["gsm8k_strict_em"] = float(g["exact_match,strict-match"])
        rows = _gsm8k_rows(gen["samples"]["gsm8k_cot"], tok, cfg.gsm_max_new)
        scores["gsm8k_extractable_rate"] = sum(r["extracted"] for r in rows) / len(rows)
        scores["gsm8k_truncated_rate"] = sum(r["truncated"] for r in rows) / len(rows)
        w = Jsonl(out / "gsm8k_samples.jsonl", "w")
        for r in rows:
            w.write(**r)
        w.close()
        log.info("[%s] gsm8k flexible=%.4f strict=%.4f", cond, scores["gsm8k_flexible_em"], scores["gsm8k_strict_em"])
        n_gsm = len(rows)
        save_json({"scores": dict(scores), "n": n_gsm}, partial)
    free_cuda()

    log.info("[%s] hellaswag / boolq / arc_easy (likelihood)", cond)
    ll = _oom_backoff(lambda bs: _simple_eval(_make_lm(model, tok, bs), ["hellaswag", "boolq", "arc_easy"], cfg, False), cfg.ll_bs, log, cond)
    for task in ("hellaswag", "boolq", "arc_easy"):
        r = ll["results"][task]
        scores[f"{task}_acc"] = float(r["acc,none"])
        if "acc_norm,none" in r:
            scores[f"{task}_acc_norm"] = float(r["acc_norm,none"])
    w = Jsonl(out / "likelihood_samples.jsonl", "w")
    for task in ("hellaswag", "boolq", "arc_easy"):
        for s in ll["samples"][task]:
            w.write(task=task, doc_id=s["doc_id"], target=s["target"], acc=s.get("acc"), acc_norm=s.get("acc_norm"))
    w.close()
    scores["n"] = {
        "gsm8k": n_gsm,
        **{t: len([s for s in ll["samples"][t]]) for t in ("hellaswag", "boolq", "arc_easy")},
    }
    scores["eval_seconds_benchmarks"] = time.perf_counter() - start
    log.info("[%s] scores: %s", cond, {k: round(v, 4) for k, v in scores.items() if isinstance(v, float)})

    t1 = time.perf_counter()
    run_qualitative(model, tok, cfg, out / "qualitative.json", log)
    scores["eval_seconds_qualitative"] = time.perf_counter() - t1
    save_json(scores, out / "scores.json")
    free_cuda()
    return scores


def load_eval_model_from_state(cfg, state_path, device):
    from task1.models import load_base

    model = load_base(cfg, torch.float16, device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    return model
