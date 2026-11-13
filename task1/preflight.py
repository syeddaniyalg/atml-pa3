import argparse
import time
from dataclasses import replace

import torch

from common.log import get_logger
from common.utils import env_info, free_cuda, get_device, peak_mem_gb, reset_peak, set_child_env
from task1.config import OUT, make_cfg
from task1.evaluate import _make_lm, _simple_eval
from task1.models import load_base, load_tokenizer

STRESS_LENGTHS = (1024, 2048)


def stress_forward(model, device, bs, length, vocab):
    ids = torch.randint(0, vocab, (bs, length), device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(input_ids=ids).logits
        torch.log_softmax(logits.float(), dim=-1).sum().item()
    del ids, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    set_child_env()
    cfg = make_cfg(smoke=args.smoke, tiny=args.tiny)
    log = get_logger("preflight", OUT / "logs" / "preflight.log")
    log.info("env: %s", env_info())
    device = get_device()
    tok = load_tokenizer()
    model = load_base(cfg, torch.float16, device).eval()
    vocab = model.config.vocab_size

    for length in STRESS_LENGTHS:
        bs = cfg.ll_bs
        while True:
            reset_peak()
            try:
                stress_forward(model, device, bs, length, vocab)
                log.info("stress forward ok: batch %d x %d tokens, peak %.2f GB", bs, length, peak_mem_gb())
                break
            except torch.OutOfMemoryError:
                free_cuda()
                if bs <= 1:
                    raise SystemExit(f"PREFLIGHT FAILED: out of memory even at batch 1, length {length}")
                bs //= 2
                log.warning("OOM at length %d, trying batch %d", length, bs)
        free_cuda()

    t0 = time.perf_counter()
    small = replace(cfg, eval_limit=16)
    out = _simple_eval(_make_lm(model, tok, cfg.ll_bs), ["hellaswag", "boolq", "arc_easy"], small, False)
    log.info("likelihood smoke ok: %s (%.0fs)", {t: round(float(out["results"][t]["acc,none"]), 3) for t in ("hellaswag", "boolq", "arc_easy")}, time.perf_counter() - t0)
    free_cuda()

    t0 = time.perf_counter()
    tiny = replace(cfg, eval_limit=4)
    out = _simple_eval(_make_lm(model, tok, cfg.gsm_bs), ["gsm8k_cot"], tiny, True, "max_gen_toks=64,do_sample=False")
    log.info("generation smoke ok (%.0fs): flexible=%s", time.perf_counter() - t0, out["results"]["gsm8k_cot"].get("exact_match,flexible-extract"))
    free_cuda()
    log.info("PREFLIGHT OK")


if __name__ == "__main__":
    main()
