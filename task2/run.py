import argparse
import time


from common.launcher import Job, run_jobs
from common.log import get_logger
from common.utils import env_info, free_cuda, peak_mem_gb, reset_peak, save_json, save_run_config, set_child_env
from task2 import analyze
from task2.config import OUT, STAGES, make_cfg
from task2.data import load_splits
from task2.evaluate import final_eval
from task2.features import cache_features
from task2.train import train


def prepare(cfg, log):
    load_splits(cfg, log)
    cache_features(cfg, log)


def run_one(cfg, log, resume):
    save_run_config(OUT / f"run_{cfg.tokens}" / "config.json", cfg=cfg, stages=list(STAGES), seed=6304)
    model, store, splits, resources = train(cfg, log, resume)
    run = OUT / f"run_{cfg.tokens}"
    reset_peak()
    start = time.perf_counter()
    result = final_eval(model, store, splits, cfg, run, log)
    result["eval_seconds"] = time.perf_counter() - start
    result["eval_peak_mem_gb"] = peak_mem_gb()
    result["tokens"] = cfg.tokens
    save_json(result, run / "final_metrics.json")
    save_json({"done": True}, run / "done.json")
    del model
    free_cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prep", "train", "all", "analyze"], default="all")
    ap.add_argument("--tokens", type=int, default=196)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--optional", action="store_true")
    ap.add_argument("--gpus", default="0,1")
    args = ap.parse_args()
    set_child_env()
    cfg = make_cfg(args.tokens, smoke=args.smoke)
    log = get_logger(f"task2-{args.mode}-{args.tokens}", OUT / "logs" / f"{args.mode}_{args.tokens}.log")
    log.info("env: %s", env_info())
    if args.mode == "prep":
        prepare(cfg, log)
    elif args.mode == "train":
        run_one(cfg, log, args.resume)
    elif args.mode == "analyze":
        analyze.main()
    else:
        prepare(cfg, log)
        flags = [f for f, on in (("--smoke", args.smoke), ("--resume", args.resume)) if on]
        jobs = [Job("vlm196", ["-m", "task2.run", "--mode", "train", "--tokens", "196", *flags], done=str(OUT / "run_196" / "done.json"))]
        if args.optional:
            jobs.append(Job("vlm49", ["-m", "task2.run", "--mode", "train", "--tokens", "49", *flags], done=str(OUT / "run_49" / "done.json")))
        run_jobs(jobs, gpus=[int(g) for g in args.gpus.split(",")])
        analyze.main()


if __name__ == "__main__":
    main()
