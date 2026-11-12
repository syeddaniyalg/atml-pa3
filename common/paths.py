import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("PA3_RESULTS", ROOT / "results"))
CACHE = Path(os.environ.get("PA3_CACHE", ROOT / "cache"))
SPLITS = ROOT / "splits"


def ckpt_root():
    default = "/kaggle/temp/ckpt" if os.path.isdir("/kaggle/temp") else str(ROOT / "ckpt")
    return Path(os.environ.get("PA3_CKPT", default))
