import argparse
import re

import torch
from datasets import load_dataset

from common.log import get_logger, progress
from common.paths import SPLITS
from common.utils import save_json
from task1.config import DATA, OUT, make_cfg
from task1.models import chat_prompt, load_tokenizer


def normalize(text):
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def encode_example(tok, question, response):
    prompt = tok(chat_prompt(tok, question), add_special_tokens=False).input_ids
    resp = tok(response + "<|im_end|>", add_special_tokens=False).input_ids
    return prompt, resp


def build(cfg, log):
    tok = load_tokenizer()
    gsm_test = load_dataset("openai/gsm8k", "main", split="test")
    banned = {normalize(q) for q in gsm_test["question"]}
    stream = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    items, ids = [], []
    stats = {"empty": 0, "too_long": 0, "gsm8k_overlap": 0, "scanned": 0}
    bar = progress(cfg.n_examples, "metamath", unit="ex")
    for idx, row in enumerate(stream):
        stats["scanned"] += 1
        q, r = (row.get("query") or "").strip(), (row.get("response") or "").strip()
        if not q or not r:
            stats["empty"] += 1
            continue
        if normalize(q) in banned or normalize(row.get("original_question") or "") in banned:
            stats["gsm8k_overlap"] += 1
            continue
        prompt, resp = encode_example(tok, q, r)
        if len(prompt) + len(resp) > cfg.max_len:
            stats["too_long"] += 1
            continue
        items.append((prompt + resp, len(prompt)))
        ids.append(idx)
        bar.update(1)
        if len(items) % 500 == 0:
            bar.set_postfix(scanned=stats["scanned"], overlap=stats["gsm8k_overlap"], long=stats["too_long"])
        if len(items) >= cfg.n_examples:
            break
    bar.close()
    lengths = [len(x[0]) for x in items]
    stats.update(kept=len(items), mean_len=sum(lengths) / len(lengths), max_len=max(lengths))
    DATA.mkdir(parents=True, exist_ok=True)
    name = "train_smoke.pt" if cfg.smoke else "train.pt"
    torch.save({"items": items, "ids": ids}, DATA / name)
    if not cfg.smoke:
        save_json({"source": "meta-math/MetaMathQA", "indices": ids}, SPLITS / "task1_train_ids.json")
        save_json(stats, OUT / "data_stats.json")
    log.info("data ready: %s", stats)
    return items


def load_train(cfg):
    name = "train_smoke.pt" if cfg.smoke else "train.pt"
    path = DATA / name
    if not path.exists():
        build(cfg, get_logger("data"))
    return torch.load(path)["items"]


def collate(items, pad_id):
    length = max(len(ids) for ids, _ in items)
    input_ids = torch.full((len(items), length), pad_id, dtype=torch.long)
    attn = torch.zeros((len(items), length), dtype=torch.long)
    labels = torch.full((len(items), length), -100, dtype=torch.long)
    for i, (ids, npmt) in enumerate(items):
        t = torch.tensor(ids)
        input_ids[i, : len(ids)] = t
        attn[i, : len(ids)] = 1
        labels[i, npmt : len(ids)] = t[npmt:]
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = make_cfg(smoke=args.smoke)
    build(cfg, get_logger("data"))
