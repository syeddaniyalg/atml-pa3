import math
import random
import time

import numpy as np
import torch

from common.log import fmt_metrics, line, progress
from common.paths import ckpt_root
from common.utils import Jsonl, Timer, get_device, load_json, path_size, peak_mem_gb, reset_peak, save_json, seed_all
from task2.config import OUT, SEED, STAGES
from task2.data import load_splits
from task2.dataset import MixSampler
from task2.evaluate import FeatureStore, dev_eval
from task2.features import load_features
from task2.model import VLM


def lr_lambda(updates, warmup_frac, final_frac):
    warm = max(1, round(warmup_frac * updates))

    def fn(step):
        if step < warm:
            return (step + 1) / warm
        frac = (step - warm) / max(1, updates - warm)
        return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * frac))

    return fn


def make_optimizer(model, stage, cfg):
    groups = []
    for params, lr in ((model.connector_params(), stage.conn_lr), (model.lora_params() if stage.lora_lr else [], stage.lora_lr)):
        decay = [p for p in params if p.ndim >= 2]
        plain = [p for p in params if p.ndim < 2]
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": cfg.weight_decay})
        if plain:
            groups.append({"params": plain, "lr": lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), fused=torch.cuda.is_available())


def trainable_state(model):
    state = {"connector": model.connector.state_dict(), "scale": model.scale.detach().cpu()}
    state["lora"] = {n: p.detach().cpu() for n, p in model.lm.named_parameters() if "lora_" in n}
    return state


def load_trainable(model, state):
    model.connector.load_state_dict(state["connector"])
    model.scale.copy_(state["scale"].to(model.device))
    params = dict(model.lm.named_parameters())
    for n, v in state["lora"].items():
        params[n].data.copy_(v.to(model.device))


def calibrate(model, store, splits, cfg):
    from task2.dataset import caption_examples

    examples = caption_examples(splits["caption"]["train"])[:256:4]
    feats = store.get([e.image for e in examples])
    with torch.no_grad():
        raw = model.connector(feats.float())
        conn_norm = raw.norm(dim=-1).mean().item()
        ids = torch.cat([model.target_ids(e.target, "caption") for e in examples])
        text_norm = model.embed(ids).float().norm(dim=-1).mean().item()
    scale = text_norm / conn_norm
    model.scale.fill_(scale)
    info = {"rule": "scale = mean L2 norm of text-token embeddings / mean L2 norm of initial connector outputs, measured on 64 training captions", "text_norm": text_norm, "connector_norm_before": conn_norm, "scale": scale}
    save_json(info, OUT / f"run_{cfg.tokens}" / "calibration.json")
    return info


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng_state(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def train(cfg, log, resume=False):
    device = get_device()
    run = OUT / f"run_{cfg.tokens}"
    run.mkdir(parents=True, exist_ok=True)
    ck_dir = ckpt_root() / f"task2_{cfg.tokens}"
    ck_dir.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    splits = load_splits(cfg, log)
    feats, index = load_features()
    store = FeatureStore(feats, index, device)
    model = VLM(cfg, device)
    sampler = MixSampler(splits)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    boundaries = np.cumsum([0, *cfg.stage_updates])
    calib = calibrate(model, store, splits, cfg) if not (resume and (ck_dir / "last.pt").exists()) else None
    log.info("calibration: %s", calib)
    sanity = model.sanity(store.get([e for e in list(index)[:2]]))
    start_global, state = 0, None
    if resume and (ck_dir / "last.pt").exists():
        state = torch.load(ck_dir / "last.pt", map_location="cpu", weights_only=False)
        start_global = state["global"]
        log.info("resuming at update %d", start_global)

    state_stage = None
    if state is not None:
        state_stage = state["stage"]
        if state_stage >= 1:
            model.enable_lora()
        load_trainable(model, state["model"])
        sampler.load(state["sampler"])
        set_rng_state(state["rng"])
        scaler.load_state_dict(state["scaler"])

    dev_log = Jsonl(run / "dev_metrics.jsonl", "a" if start_global else "w")
    train_log = Jsonl(run / "train_log.jsonl", "a" if start_global else "w")
    resources = {"stages": {}}
    if not start_global:
        dev = dev_eval(model, store, splits, cfg)
        dev_log.write(update=0, stage="init", **dev)
        log.info("dev@0: %s", line(dev))

    total = int(boundaries[-1])
    bar = progress(total, f"vlm{cfg.tokens}", initial=start_global, unit="upd")
    ck_timer = Timer()
    last_dev = {}
    for si, stage in enumerate(STAGES):
        lo, hi = int(boundaries[si]), int(boundaries[si + 1])
        if start_global >= hi:
            continue
        if si >= 1 and not model.lora_on:
            model.enable_lora()
        stage_updates = hi - lo
        opt = make_optimizer(model, stage, cfg)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(stage_updates, cfg.warmup_frac, cfg.final_lr_frac))
        done_in_stage = max(0, start_global - lo)
        if state is not None and state_stage == si:
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            state = None
        trainable = model.trainable_summary()
        log.info("stage %s: %d updates, trainable=%s", stage.name, stage_updates, trainable)
        reset_peak()
        t_stage = time.perf_counter()
        pause = 0.0
        counts_total = {}
        model.train()
        for u in range(done_in_stage, stage_updates):
            t0 = time.perf_counter()
            examples, counts = sampler.batch(stage, cfg.batch)
            loss_val = 0.0
            for m in range(0, cfg.batch, cfg.micro):
                chunk = examples[m : m + cfg.micro]
                feats_b = store.get([e.image for e in chunk])
                tg = [model.target_ids(e.target, e.kind if e.kind == "caption" else "short") for e in chunk]
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    per_ex = model.example_loss(feats_b, [e.question for e in chunk], tg)
                loss = per_ex.sum() / cfg.batch
                scaler.scale(loss).backward()
                loss_val += loss.item()
            scaler.unscale_(opt)
            params = [p for g in opt.param_groups for p in g["params"]]
            if u == 0 and si == 0 and start_global == 0:
                sanity["connector_grad_norm_first_update"] = float(torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model.connector.parameters() if p.grad is not None)))
                sanity["backbone_grads_none"] = all(p.grad is None for n, p in model.lm.named_parameters() if "lora_" not in n)
                save_json(sanity, run / "sanity.json")
                log.info("sanity: %s", sanity)
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.clip).item()
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped = scaler.get_scale() < before
            lr = opt.param_groups[0]["lr"]
            sched.step()
            opt.zero_grad(set_to_none=True)
            g = lo + u + 1
            dt = time.perf_counter() - t0
            for k, v in counts.items():
                counts_total[k] = counts_total.get(k, 0) + v
            train_log.write(update=g, stage=stage.name, loss=loss_val, lr=lr, grad_norm=gnorm, skipped=bool(skipped), step_seconds=dt, mem_gb=peak_mem_gb(), **{f"draws_{k}": v for k, v in counts.items()})
            shown = {"stage": stage.name, "loss": loss_val, "lr": lr, "gnorm": gnorm, "mem": peak_mem_gb(), **{k: v for k, v in last_dev.items()}}
            bar.set_postfix(fmt_metrics(shown), refresh=False)
            bar.update(1)
            if g % cfg.log_every == 0 or g == total:
                log.info("[%d/%d] %s", g, total, line({k: shown[k] for k in ("stage", "loss", "lr", "gnorm", "mem")}))

            if g % cfg.dev_every == 0 or g in cfg.dev_points or g == boundaries[si + 1]:
                t1 = time.perf_counter()
                dev = dev_eval(model, store, splits, cfg)
                model.train()
                dev_log.write(update=g, stage=stage.name, **dev)
                last_dev = {"cap": dev["caption_loss"], "g_swap": dev["grounding_swap_penalty"], "n_swap": dev["natural_swap_penalty"]}
                log.info("dev@%d: %s", g, line(dev))
                pause += time.perf_counter() - t1
            if g % cfg.ckpt_every == 0:
                t1 = time.perf_counter()
                with ck_timer():
                    torch.save(
                        {"global": g, "stage": si, "model": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "sampler": sampler.state(), "rng": rng_state(), "stage_progress": u + 1},
                        ck_dir / "last.pt",
                    )
                pause += time.perf_counter() - t1
        torch.save(trainable_state(model), run / f"weights_stage{si + 1}.pt")
        resources["stages"][stage.name] = {
            "updates": stage_updates,
            "seconds_excluding_dev_and_ckpt": time.perf_counter() - t_stage - pause,
            "peak_mem_gb": peak_mem_gb(),
            "trainable": trainable,
            "draw_counts": counts_total,
            "weights_bytes": path_size(run / f"weights_stage{si + 1}.pt"),
        }
        del opt, sched
    bar.close()
    res_path = run / "train_resources.json"
    if res_path.exists():
        previous = load_json(res_path)
        previous["stages"].update(resources["stages"])
        resources["stages"] = previous["stages"]
        resources["checkpoint_seconds"] = ck_timer.total + previous.get("checkpoint_seconds", 0.0)
    else:
        resources["checkpoint_seconds"] = ck_timer.total
    resources["recovery_ckpt_bytes"] = path_size(ck_dir / "last.pt") if (ck_dir / "last.pt").exists() else 0
    resources["total_train_seconds"] = sum(s["seconds_excluding_dev_and_ckpt"] for s in resources["stages"].values())
    save_json(resources, res_path)
    dev_log.close()
    train_log.close()
    return model, store, splits, resources
