import argparse
import os
import sys
import threading

from common.launcher import Job, run_jobs
from common.log import get_logger
from common.utils import env_info, save_json, save_run_config, set_child_env
from task3 import adaptive, analyze, offline, selection
from task3 import config as t3config
from task3.config import OUT, download_data
from task3.data import ensure_splits, load


def offline_stage(workers, log):
    download_data()
    for name in ("math", "mathvista"):
        d = load(name, log)
        out = selection.run_all(d, workers)
        offline.curves(d, out).to_csv(OUT / f"curves_{name}.csv", index=False)
        offline.diff_table(d, out).to_csv(OUT / f"differences_{name}.csv", index=False)
        offline.long_frame(d, out).to_parquet(OUT / f"selection_results_{name}.parquet", index=False)
        offline.signal_analysis(d, log)
        offline.save_groups(d)
        offline.disagreement_examples(d, out)
        offline.trajectories(d, out, log)
        adaptive.run(d, out, log)
        log.info("%s offline stage complete", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["offline", "live", "analyze", "all"], default="all")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--live-limit", type=int, default=0)
    args = ap.parse_args()
    set_child_env()
    log = get_logger("task3", OUT / "logs" / "task3.log")
    log.info("env: %s", env_info())
    save_run_config(OUT / "config.json", constants={k: v for k, v in vars(t3config).items() if k.isupper() and not k.startswith("_")}, workers=args.workers)
    if args.stage == "offline":
        offline_stage(args.workers, log)
    elif args.stage == "live":
        from task3 import live

        live.run(args.live_limit or None)
    elif args.stage == "analyze":
        analyze.main()
    else:
        download_data()
        ensure_splits("math")
        jobs = [Job("live", ["-m", "task3.live", *(["--limit", str(args.live_limit)] if args.live_limit else [])], done=str(OUT / "live_summary.json"))]
        runner = threading.Thread(target=run_jobs, args=(jobs, [args.gpu]))
        runner.start()
        offline_stage(args.workers, log)
        runner.join()
        analyze.main()
        save_json({"complete": True}, OUT / "done.json")


if __name__ == "__main__":
    sys.exit(main())
