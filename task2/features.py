import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch
from PIL import Image
from transformers import SiglipVisionModel

from common.log import get_logger, progress
from common.utils import free_cuda, get_device, load_json, peak_mem_gb, reset_peak, save_json
from task2.config import COCO_IMG, DATA, OUT, VISION, make_cfg
from task2.data import image_keys, load_splits

FEATS = DATA / "feats.npy"
INDEX = DATA / "feats_index.json"
THUMBS = OUT / "thumbs"


def url_of(key):
    split, iid = key.split(":")
    return COCO_IMG.format(split=split, iid=int(iid))


def load_image(key, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(url_of(key), timeout=60)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def preprocess(img):
    arr = np.asarray(img.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)


def cache_features(cfg, log):
    splits = load_splits(cfg, log)
    keys = image_keys(splits)
    if FEATS.exists() and INDEX.exists() and len(load_json(INDEX)) == len(keys):
        log.info("features cached: %d images", len(keys))
        return
    device = get_device()
    model = SiglipVisionModel.from_pretrained(VISION, torch_dtype=torch.float16).to(device).eval()
    progress_file = FEATS.with_suffix(".progress")
    resume_at = 0
    out = None
    if FEATS.exists() and progress_file.exists():
        out = np.lib.format.open_memmap(FEATS, mode="r+")
        if out.shape[0] == len(keys):
            resume_at = int(progress_file.read_text().strip() or 0)
            log.info("resuming feature cache at image %d of %d", resume_at, len(keys))
        else:
            out = None
    if out is None:
        out = np.lib.format.open_memmap(FEATS, mode="w+", dtype=np.float16, shape=(len(keys), 196, 768))
    THUMBS.mkdir(parents=True, exist_ok=True)
    reset_peak()
    t_dl = t_gpu = 0.0
    bs = 64
    bar = progress(len(keys), "siglip features", unit="img", initial=resume_at)
    pool = ThreadPoolExecutor(cfg.n_img_workers)

    def fetch(key):
        img = load_image(key)
        if key.startswith("val:"):
            img.resize((224, 224)).save(THUMBS / f"{key.replace(':', '_')}.jpg", quality=85)
        return preprocess(img)

    for start in range(resume_at, len(keys), bs):
        chunk = keys[start : start + bs]
        t0 = time.perf_counter()
        batch = np.stack(list(pool.map(fetch, chunk)))
        t_dl += time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.no_grad():
            feats = model(pixel_values=torch.from_numpy(batch).to(device, torch.float16)).last_hidden_state
        out[start : start + len(chunk)] = feats.cpu().numpy()
        if (start // bs) % 20 == 19:
            out.flush()
            progress_file.write_text(str(start + len(chunk)))
        t_gpu += time.perf_counter() - t0
        bar.update(len(chunk))
        bar.set_postfix(dl_s=f"{t_dl:.0f}", gpu_s=f"{t_gpu:.0f}")
    bar.close()
    pool.shutdown()
    out.flush()
    save_json({k: i for i, k in enumerate(keys)}, INDEX)
    save_json(
        {"images": len(keys), "download_decode_seconds": t_dl, "encoder_seconds": t_gpu, "peak_mem_gb": peak_mem_gb(), "feature_shape": [196, 768], "feature_bytes": int(FEATS.stat().st_size)},
        OUT / "feature_cache_resources.json",
    )
    del model
    free_cuda()
    log.info("cached %d images in %.0fs (download %.0fs, encoder %.0fs)", len(keys), t_dl + t_gpu, t_dl, t_gpu)


def load_features():
    return np.load(FEATS, mmap_mode="r"), load_json(INDEX)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cache_features(make_cfg(smoke=args.smoke), get_logger("task2-feats", OUT / "logs" / "features.log"))
