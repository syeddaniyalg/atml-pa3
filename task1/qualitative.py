from pathlib import Path

import torch
import yaml

from common.log import progress
from common.utils import save_json
from task1.models import chat_prompt

QUESTIONS = Path(__file__).parent / "first_plot_questions.yaml"


def load_questions():
    items = yaml.safe_load(open(QUESTIONS, encoding="utf-8"))
    out = []
    for i, q in enumerate(items):
        qid = q["id"]
        if qid.endswith("_json"):
            fmt, base = "json", qid[: -len("_json")]
        elif qid.endswith("_template"):
            fmt, base = "template", qid[: -len("_template")]
        else:
            fmt, base = "plain", qid
        out.append(
            {
                "order": i,
                "id": qid,
                "base": base,
                "format": fmt,
                "system": q.get("system"),
                "prompt": q["paraphrases"][0],
            }
        )
    return out


@torch.no_grad()
def run_qualitative(model, tok, cfg, path, log):
    questions = load_questions()
    side = tok.padding_side
    tok.padding_side = "left"
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    bar = progress(len(questions), "qualitative", unit="q")
    results = []
    for i in range(0, len(questions), cfg.qual_bs):
        group = questions[i : i + cfg.qual_bs]
        texts = [chat_prompt(tok, q["prompt"], q["system"]) for q in group]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=cfg.qual_max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=True,
            pad_token_id=tok.pad_token_id,
            eos_token_id=stop,
        )
        new = out[:, enc["input_ids"].shape[1] :]
        for q, ids in zip(group, new):
            ids = ids.tolist()
            ended = any(t in stop for t in ids)
            if ended:
                ids = ids[: min(ids.index(t) for t in stop if t in ids)]
            results.append(
                {
                    **q,
                    "response": tok.decode(ids, skip_special_tokens=True),
                    "n_tokens": len(ids),
                    "truncated": not ended,
                }
            )
        bar.update(len(group))
    bar.close()
    tok.padding_side = side
    save_json(results, path)
    log.info("qualitative: %d responses, %d truncated", len(results), sum(r["truncated"] for r in results))
    return results
