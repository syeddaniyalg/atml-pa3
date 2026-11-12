import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from common.log import progress


def fetch(url, dest, retries=4):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.rename(dest)
            return dest
        except Exception:
            if attempt == retries - 1:
                raise
    return dest


def fetch_big(url, dest, desc):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar = progress(total, desc, unit="B")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        bar.close()
    tmp.rename(dest)
    return dest


def fetch_many(pairs, workers, desc):
    bar = progress(len(pairs), desc, unit="file")

    def job(p):
        out = fetch(*p)
        bar.update(1)
        return out

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(job, pairs))
    bar.close()
    return results


def unzip(path, out_dir):
    out_dir = Path(out_dir)
    with zipfile.ZipFile(path) as z:
        z.extractall(out_dir)
    return out_dir
