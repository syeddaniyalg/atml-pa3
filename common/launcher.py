import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from common.paths import ROOT


@dataclass
class Job:
    name: str
    args: list
    done: str = ""
    env: dict = field(default_factory=dict)


def _pump(proc, prefix, lock):
    buf = ""
    stream = proc.stdout
    while True:
        chunk = stream.read1(4096)
        if not chunk:
            break
        buf += chunk.decode("utf-8", errors="replace")
        parts = re.split(r"[\r\n]", buf)
        buf = parts.pop()
        with lock:
            for part in parts:
                if part.strip():
                    print(f"{prefix} {part}", flush=True)
    if buf.strip():
        with lock:
            print(f"{prefix} {buf}", flush=True)


def run_jobs(jobs, gpus=(0, 1), interval=20):
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    lock = threading.Lock()
    codes = {}

    def worker(gpu):
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            if job.done and Path(job.done).exists():
                with lock:
                    print(f"[{job.name}|gpu{gpu}] skipped, found {job.done}", flush=True)
                codes[job.name] = 0
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            env["PA3_TQDM_INTERVAL"] = str(interval)
            env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            env.update(job.env)
            start = time.time()
            with lock:
                print(f"[{job.name}|gpu{gpu}] start: {' '.join(map(str, job.args))}", flush=True)
            proc = subprocess.Popen(
                [sys.executable, *map(str, job.args)],
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            _pump(proc, f"[{job.name}|gpu{gpu}]", lock)
            code = proc.wait()
            codes[job.name] = code
            with lock:
                print(f"[{job.name}|gpu{gpu}] exit={code} in {(time.time() - start) / 60:.1f} min", flush=True)

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = {k: v for k, v in codes.items() if v != 0}
    if failed:
        raise RuntimeError(f"jobs failed: {failed}")
    return codes
