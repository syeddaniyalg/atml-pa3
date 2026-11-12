import os
from pathlib import Path

from common.paths import CACHE, RESULTS

SEED = 6304
OUT = RESULTS / "task3"
WORK = CACHE / "task3"
BUDGETS = (1, 2, 4, 8, 16, 32)
N_ORDERS = 10
N_ROLLOUTS = 32
N_DEV = 200
N_LIVE = 50
STOP_POINTS = (4, 8, 16)
TAU_GRID = (0.5, 0.625, 0.75, 0.875, 1.0)
DELTA_GRID = (0.0, 0.1, 0.25, 0.5)
BOOT = 1000
DRIVE_FOLDER = "1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23"
LIVE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LIVE_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SYSTEM = "You are a helpful assistant."
INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVIEW = "Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \\boxed{}."

DATASETS = {
    "math": {"folder": "qwen25-math", "vlm": False},
    "mathvista": {"folder": "qwen25vl-mathvista", "vlm": True},
}


def find_data():
    candidates = [os.environ.get("PA3_T3_DATA", ""), str(WORK / "ATML-PA3-Data"), "/kaggle/working/ATML-PA3-Data", "/kaggle/input/atml-pa3-data/ATML-PA3-Data", "/kaggle/input/atml-pa3-data"]
    for c in candidates:
        if c and (Path(c) / "qwen25-math" / "generations.parquet").exists():
            return Path(c)
    root = Path("/kaggle/input")
    if root.exists():
        for p in root.rglob("qwen25-math"):
            if (p / "generations.parquet").exists():
                return p.parent
    return None


def download_data():
    found = find_data()
    if found is not None:
        return found
    import gdown

    WORK.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(id=DRIVE_FOLDER, output=str(WORK / "ATML-PA3-Data"), quiet=False, use_cookies=False)
    found = find_data()
    if found is None:
        raise FileNotFoundError("ATML-PA3-Data not found; set PA3_T3_DATA")
    return found
