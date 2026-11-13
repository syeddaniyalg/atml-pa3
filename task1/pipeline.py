import argparse
import subprocess
import sys

from common.launcher import Job, run_jobs
from common.log import get_logger
from common.utils import env_info, set_child_env
from task1 import analyze
from task1.config import OUT, make_cfg
from task1.data import load_train
from task1.diagnostics import run_diagnostics, verify_oplora

ORDER = ["fullft", "l2sp", "lora16", "oplora16", "lora4", "baseline"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=ORDER)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-final", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()
    set_child_env()
    cfg = make_cfg(smoke=args.smoke, tiny=args.tiny)
    log = get_logger("task1", OUT / "logs" / "pipeline.log")
    log.info("env: %s", env_info())
    load_train(cfg)

    flags = [f for f, on in (("--smoke", args.smoke), ("--tiny", args.tiny), ("--resume", args.resume)) if on]
    if not args.skip_preflight:
        log.info("running preflight (memory stress and a tiny evaluation) before any training")
        code = subprocess.run([sys.executable, "-m", "task1.preflight", *[f for f in flags if f != "--resume"]]).returncode
        if code != 0:
            raise SystemExit("preflight failed, nothing was started; see results/task1/logs/preflight.log")
    jobs = [
        Job(m, ["-m", "task1.run", "--method", m, *flags], done=str(OUT / "runs" / m / "done.json"))
        for m in ORDER
        if m in args.methods
    ]
    run_jobs(jobs, gpus=[int(g) for g in args.gpus.split(",")])
    if args.skip_final:
        return
    run_diagnostics(cfg, log)
    if (OUT / "runs" / "oplora16" / "adapter.pt").exists():
        verify_oplora(cfg, log)
    analyze.main()
    log.info("task1 complete; results in %s", OUT)


if __name__ == "__main__":
    main()
