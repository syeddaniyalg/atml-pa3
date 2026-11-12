import gc
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Jsonl:
    def __init__(self, path, mode="a"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, mode, encoding="utf-8")

    def write(self, **row):
        self.f.write(json.dumps(row, default=_default, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def peak_mem_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**30
    return 0.0


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def count_params(model, trainable_only=False):
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        if trainable_only and not p.requires_grad:
            continue
        total += p.numel()
    return total


def path_size(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Timer:
    def __init__(self):
        self.total = 0.0

    @contextmanager
    def __call__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.total += time.perf_counter() - start


def env_info():
    info = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    for name in ("transformers", "lm_eval", "math_verify", "datasets"):
        try:
            info[name] = getattr(__import__(name), "__version__", "unknown")
        except Exception:
            pass
    return info


def save_run_config(path, **objs):
    from dataclasses import asdict, is_dataclass
    from importlib import metadata

    out = {"env": env_info(), "versions": {}}
    for pkg in ("transformers", "lm_eval", "math-verify", "datasets", "torch", "peft"):
        try:
            out["versions"][pkg] = metadata.version(pkg)
        except Exception:
            pass
    for key, obj in objs.items():
        if is_dataclass(obj):
            out[key] = asdict(obj)
        elif isinstance(obj, (list, tuple)):
            out[key] = [asdict(o) if is_dataclass(o) else o for o in obj]
        else:
            out[key] = obj
    save_json(out, path)


def set_child_env():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
