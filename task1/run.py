import argparse
import time

import torch

from common.lora import export_factors, lora_modules, merge_factors
from common.paths import ckpt_root
from common.utils import env_info, free_cuda, get_device, path_size, save_json, save_run_config, set_child_env
from common.log import get_logger
from task1.config import METHODS, OUT, make_cfg
from task1.data import collate, load_train
from task1.evaluate import evaluate_condition, load_eval_model_from_state
from task1.models import load_base, load_tokenizer
from task1.train import train


@torch.no_grad()
def logits_of(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()


def export_adapter(method, model, cfg, tok, device, log):
    run_dir = OUT / "runs" / method
    timer = time.perf_counter()
    exported = export_factors(model)
    torch.save(exported, run_dir / "adapter.pt")
    export_seconds = time.perf_counter() - timer
    mods = lora_modules(model)
    deltas = {
        f"L{l}": mods[f"model.layers.{l}.mlp.down_proj"].delta().detach().cpu() for l in cfg.diag_layers
    }
    torch.save(deltas, run_dir / "delta_down_proj_final.pt")

    model.eval()
    items = load_train(cfg)[:4]
    batch = collate(items, tok.pad_token_id)
    unmerged = logits_of(model, batch, device)
    merged = load_base(cfg, torch.float16, device)
    merge_factors(merged, exported)
    merged.eval()
    merged_logits = logits_of(merged, batch, device)
    mask = batch["attention_mask"].bool().to(device)
    diff = (unmerged - merged_logits)[mask].abs()
    check = {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "argmax_agreement": (unmerged.argmax(-1) == merged_logits.argmax(-1))[mask].float().mean().item(),
    }
    log.info("merged vs unmerged logits: %s", check)
    return merged, {"adapter_bytes": path_size(run_dir / "adapter.pt"), "export_seconds": export_seconds, "merge_check": check}


def run_method(method, cfg, resume, log):
    tok = load_tokenizer()
    device = get_device()
    model, result = train(method, cfg, log, resume)
    spec = METHODS[method]
    if spec["kind"] == "full":
        model.eval()
        model.gradient_checkpointing_disable()
        if method == "fullft":
            mid_path = OUT / "runs" / "fullft" / "model_mid_fp16.pt"
            if not mid_path.exists():
                mid_path = ckpt_root() / "fullft_mid_fp16.pt"
            mid = load_eval_model_from_state(cfg, mid_path, device)
            evaluate_condition(mid, tok, cfg, "fullft_mid", log)
            del mid
            free_cuda()
        model = model.half()
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, OUT / "runs" / method / "model_final_fp16.pt")
        final = model
    else:
        final, extra = export_adapter(method, model, cfg, tok, device, log)
        result.update(extra)
        del model
        free_cuda()
    save_json(result, OUT / "runs" / method / "train_summary.json")
    evaluate_condition(final, tok, cfg, f"{method}_final", log)
    del final
    free_cuda()
    save_json({"done": True}, OUT / "runs" / method / "done.json")


def run_baseline(cfg, log):
    tok = load_tokenizer()
    model = load_base(cfg, torch.float16, get_device())
    evaluate_condition(model, tok, cfg, "baseline", log)
    save_json({"done": True}, OUT / "runs" / "baseline" / "done.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=[*METHODS, "baseline"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    set_child_env()
    cfg = make_cfg(smoke=args.smoke, tiny=args.tiny)
    log = get_logger(args.method, OUT / "logs" / f"{args.method}.log")
    log.info("env: %s", env_info())
    save_run_config(OUT / "runs" / args.method / "config.json", cfg=cfg, method=METHODS.get(args.method), seed=6304)
    if args.method == "baseline":
        run_baseline(cfg, log)
    else:
        run_method(args.method, cfg, args.resume, log)
