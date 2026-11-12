import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common.log import fmt_metrics, line, progress
from common.lora import LLM_TARGETS, basis_bytes, compute_bases, inject_lora
from common.paths import ckpt_root
from common.utils import Jsonl, Timer, count_params, free_cuda, get_device, load_json, path_size, peak_mem_gb, reset_peak, save_json, seed_all
from task1.config import METHODS, OUT, SEED
from task1.data import collate, load_train
from task1.models import load_base, load_tokenizer


def lr_lambda(cfg):
    def fn(step):
        if step < cfg.warmup:
            return (step + 1) / cfg.warmup
        frac = (step - cfg.warmup) / max(1, cfg.updates - cfg.warmup)
        return 0.5 * (1 + math.cos(math.pi * frac))

    return fn


def _ce(h, y, w):
    return F.cross_entropy(F.linear(h, w).float(), y, reduction="sum")


def sft_loss_sum(model, batch, chunk=1024):
    hidden = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    labels = batch["labels"][:, 1:]
    sel = labels != -100
    h = hidden[:, :-1][sel]
    y = labels[sel]
    w = model.lm_head.weight
    total = torch.zeros((), device=h.device, dtype=torch.float32)
    for i in range(0, h.size(0), chunk):
        total = total + checkpoint(_ce, h[i : i + chunk], y[i : i + chunk], w, use_reentrant=False)
    return total, y.numel()


def build_model(spec, cfg, device, log):
    info = {}
    if spec["kind"] == "full":
        model = load_base(cfg, torch.float32, device)
    else:
        model = load_base(cfg, torch.float16, device)
        for p in model.parameters():
            p.requires_grad_(False)
        bases = None
        if spec["kind"] == "oplora":
            timer = Timer()
            with timer():
                bases = compute_bases(model, LLM_TARGETS, cfg.protect_k)
            info["basis_bytes"] = basis_bytes(bases)
            info["preprocess_seconds"] = timer.total
            log.info("oplora bases: %.1f MB in %.1fs", info["basis_bytes"] / 2**20, timer.total)
        inject_lora(model, LLM_TARGETS, spec["r"], spec["alpha"], cfg.dropout, SEED, bases)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    return model, info


def make_optimizer(model, spec, cfg):
    decay, plain = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else plain).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=spec["lr"], betas=cfg.betas, eps=cfg.eps, fused=torch.cuda.is_available())


def l2sp_step(params, ref, lam, device):
    omega = torch.zeros((), device=device, dtype=torch.float32)
    for p, r in zip(params, ref):
        diff = p.detach() - r.to(device, non_blocking=True)
        omega += 0.5 * diff.float().pow(2).sum()
        if lam > 0 and p.grad is not None:
            p.grad.add_(diff, alpha=lam)
    return omega.item()


def displacement(params, ref, device):
    total = torch.zeros((), device=device, dtype=torch.float64)
    for p, r in zip(params, ref):
        total += (p.detach() - r.to(device)).double().pow(2).sum()
    return total.sqrt().item()


def down_proj_delta(model, ref, names, layers):
    lookup = {n: i for i, n in enumerate(names)}
    params = dict(model.named_parameters())
    out = {}
    for layer in layers:
        key = f"model.layers.{layer}.mlp.down_proj.weight"
        out[f"L{layer}"] = params[key].detach().float().cpu() - ref[lookup[key]].float()
    return out


def rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def set_rng_state(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def save_recovery(method, step, model, spec, opt, sched, scaler, cfg):
    directory = ckpt_root() / method
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "data_position": step * cfg.batch,
        "model": model.state_dict() if spec["kind"] == "full" else trainable_state(model),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": rng_state(),
    }
    path = directory / f"u{step}.pt"
    torch.save(state, path)
    for old in sorted(directory.glob("u*.pt"), key=lambda p: int(p.stem[1:]))[: -cfg.keep_ckpt]:
        old.unlink()
    return path


def latest_recovery(method):
    directory = ckpt_root() / method
    files = sorted(directory.glob("u*.pt"), key=lambda p: int(p.stem[1:])) if directory.exists() else []
    return files[-1] if files else None


def train(method, cfg, log, resume=False):
    spec = METHODS[method]
    run_dir = OUT / "runs" / method
    run_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    use_amp = device.type == "cuda"
    seed_all(SEED)
    tok = load_tokenizer()
    data = load_train(cfg)
    assert len(data) >= cfg.updates * cfg.batch, "not enough training examples"

    model, info = build_model(spec, cfg, device, log)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(model, spec, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    trainable = count_params(model, trainable_only=True)
    total = count_params(model)
    log.info("method=%s trainable=%.2fM total=%.2fM", method, trainable / 1e6, total / 1e6)

    names = [n for n, _ in model.named_parameters()]
    ref = None
    ref_timer = Timer()
    if spec["kind"] == "full":
        with ref_timer():
            ref = [p.detach().cpu().clone() for p in params]
            if use_amp:
                ref = [r.pin_memory() for r in ref]
        info["reference_bytes"] = sum(r.numel() * 4 for r in ref)
        info["reference_build_seconds"] = ref_timer.total

    start = 0
    if resume:
        path = latest_recovery(method)
        if path is not None:
            state = torch.load(path, map_location="cpu", weights_only=False)
            if spec["kind"] == "full":
                model.load_state_dict(state["model"])
            else:
                model.load_state_dict(state["model"], strict=False)
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            scaler.load_state_dict(state["scaler"])
            set_rng_state(state["rng"])
            start = state["step"]
            log.info("resumed from %s at update %d", path, start)

    if start >= cfg.updates and (run_dir / "train_summary.json").exists():
        log.info("training already finished, reusing summary")
        return model, load_json(run_dir / "train_summary.json")

    logger = Jsonl(run_dir / "train_log.jsonl", "a" if start else "w")
    pen_timer, ckpt_timer = Timer(), Timer()
    mid_info, recovery_bytes = {}, 0
    reset_peak()
    t_train = time.perf_counter()
    train_pause = 0.0
    bar = progress(cfg.updates, method, initial=start, unit="upd")
    last = {}
    for step in range(start, cfg.updates):
        t0 = time.perf_counter()
        items = data[step * cfg.batch : (step + 1) * cfg.batch]
        micro = [collate(items[i : i + cfg.micro], tok.pad_token_id) for i in range(0, cfg.batch, cfg.micro)]
        ntok = sum(int(m["labels"][:, 1:].ne(-100).sum()) for m in micro)
        sft_sum = 0.0
        for mb in micro:
            mb = {k: v.to(device, non_blocking=True) for k, v in mb.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                lsum, _ = sft_loss_sum(model, mb)
            scaler.scale(lsum / ntok).backward()
            sft_sum += lsum.item()
        scaler.unscale_(opt)
        row = {}
        if spec["kind"] == "full" and spec["lam"] > 0:
            with pen_timer():
                omega = l2sp_step(params, ref, spec["lam"], device)
            row.update(omega=omega, lam_omega=spec["lam"] * omega, total_loss=sft_sum / ntok + spec["lam"] * omega)
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.clip).item()
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        lr = opt.param_groups[0]["lr"]
        sched.step()
        opt.zero_grad(set_to_none=True)
        dt = time.perf_counter() - t0
        done = step + 1
        row.update(
            update=done,
            examples=done * cfg.batch,
            sft_loss=sft_sum / ntok,
            lr=lr,
            grad_norm=gnorm,
            loss_scale=scaler.get_scale(),
            skipped=bool(skipped),
            tokens=ntok,
            step_seconds=dt,
            tok_per_s=ntok / dt,
            mem_gb=peak_mem_gb(),
        )
        logger.write(**row)
        last = row
        shown = {k: row[k] for k in ("sft_loss", "lr", "grad_norm", "tok_per_s", "mem_gb") if k in row}
        if "lam_omega" in row:
            shown["lam_omega"] = row["lam_omega"]
        if skipped:
            shown["skipped"] = 1
            log.info("update %d skipped (non-finite grads), scale -> %.0f", done, scaler.get_scale())
        bar.set_postfix(fmt_metrics(shown), refresh=False)
        bar.update(1)
        if done % cfg.log_every == 0 or done == cfg.updates:
            log.info("[%d/%d] %s", done, cfg.updates, line(shown))

        if done in cfg.ckpt_updates:
            pause = time.perf_counter()
            with ckpt_timer():
                if done == cfg.mid_update and spec["kind"] == "full":
                    mid_info["displacement"] = displacement(params, ref, device)
                    torch.save(down_proj_delta(model, ref, names, cfg.diag_layers), run_dir / "delta_down_proj_mid.pt")
                    if method == "fullft":
                        half = {k: v.detach().half().cpu() for k, v in model.state_dict().items()}
                        torch.save(half, ckpt_root() / "fullft_mid_fp16.pt")
                        del half
                path = save_recovery(method, done, model, spec, opt, sched, scaler, cfg)
                recovery_bytes = path_size(path)
            train_pause += time.perf_counter() - pause
            log.info("recovery checkpoint u%d saved (%.2f GB)", done, recovery_bytes / 2**30)
    bar.close()
    train_seconds = time.perf_counter() - t_train - train_pause
    peak = peak_mem_gb()

    skipped_total = sum(1 for r in _read_log(run_dir) if r.get("skipped"))
    result = {
        "method": method,
        "spec": spec,
        "trainable_params": trainable,
        "total_params": total,
        "train_seconds": train_seconds,
        "checkpoint_seconds": ckpt_timer.total,
        "peak_mem_gb": peak,
        "recovery_ckpt_bytes": recovery_bytes,
        "skipped_updates": skipped_total,
        "final": last,
        **info,
    }
    if spec["kind"] == "full":
        result["displacement_mid"] = mid_info.get("displacement")
        result["displacement_final"] = displacement(params, ref, device)
        result["penalty_seconds"] = pen_timer.total
        torch.save(down_proj_delta(model, ref, names, cfg.diag_layers), run_dir / "delta_down_proj_final.pt")
        weights_path = ckpt_root() / f"{method}_model_final.pt"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        result["weights_bytes"] = path_size(weights_path)
        weights_path.unlink()
    del opt, sched, scaler, ref
    free_cuda()
    save_json(result, run_dir / "train_summary.json")
    return model, result


def _read_log(run_dir):
    from common.utils import read_jsonl

    return read_jsonl(run_dir / "train_log.jsonl")
