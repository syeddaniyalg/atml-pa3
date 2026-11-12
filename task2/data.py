import argparse
import zipfile
from collections import Counter, defaultdict

import numpy as np

from common.download import fetch_big
from common.log import get_logger
from common.paths import SPLITS
from common.utils import load_json, save_json
from task2.config import COCO_ANN, DATA, OUT, SEED, VQA_BASE, VQA_FILES, make_cfg
from task2.vqa_norm import normalize

RAW = DATA / "raw"
PLURALS = {"person": "people", "knife": "knives", "mouse": "mice", "skis": "skis", "scissors": "scissors", "sheep": "sheep", "broccoli": "broccoli"}
MIN_POS_AREA = 0.02
MIN_COUNT_AREA = 0.005


def plural(name):
    words = name.split()
    last = words[-1]
    if name in PLURALS:
        return PLURALS[name]
    if last in PLURALS:
        words[-1] = PLURALS[last]
    elif last.endswith(("s", "x", "z", "ch", "sh")):
        words[-1] = last + "es"
    else:
        words[-1] = last + "s"
    return " ".join(words)


def article(name):
    return "an" if name[0] in "aeiou" else "a"


def download_raw(log):
    ann_zip = fetch_big(COCO_ANN, RAW / "annotations_trainval2014.zip", "coco annotations")
    with zipfile.ZipFile(ann_zip) as z:
        for member in z.namelist():
            base = member.split("/")[-1]
            if base.startswith(("captions_", "instances_")) and base.endswith("2014.json") and not (RAW / base).exists():
                with z.open(member) as src, open(RAW / base, "wb") as dst:
                    dst.write(src.read())
    for key, fname in VQA_FILES.items():
        path = fetch_big(VQA_BASE + fname, RAW / fname, fname)
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                base = member.split("/")[-1]
                if base.endswith(".json") and not (RAW / base).exists():
                    with z.open(member) as src, open(RAW / base, "wb") as dst:
                        dst.write(src.read())
    log.info("raw annotations ready in %s", RAW)


def coco_summary(split):
    cache = DATA / f"coco_{split}_summary.json"
    if cache.exists():
        return load_json(cache)
    inst = load_json(RAW / f"instances_{split}2014.json")
    caps = load_json(RAW / f"captions_{split}2014.json")
    sizes = {i["id"]: (i["width"], i["height"]) for i in inst["images"]}
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    per = defaultdict(lambda: defaultdict(lambda: {"n": 0, "crowd": 0, "areas": []}))
    for a in inst["annotations"]:
        w, h = sizes[a["image_id"]]
        d = per[a["image_id"]][a["category_id"]]
        if a["iscrowd"]:
            d["crowd"] += 1
        else:
            d["n"] += 1
            d["areas"].append(a["area"] / (w * h))
    objects = {}
    for iid, cd in per.items():
        objects[str(iid)] = {
            str(c): {"n": v["n"], "crowd": v["crowd"], "min_area": min(v["areas"]) if v["areas"] else 0.0, "max_area": max(v["areas"]) if v["areas"] else 0.0}
            for c, v in cd.items()
        }
    captions = defaultdict(list)
    for a in sorted(caps["annotations"], key=lambda x: x["id"]):
        captions[str(a["image_id"])].append(a["caption"].strip())
    out = {"cats": {str(k): v for k, v in cats.items()}, "images": sorted(i["id"] for i in inst["images"]), "objects": objects, "captions": captions}
    save_json(out, cache)
    return out


class Grounding:
    def __init__(self, summary, allowed):
        self.cats = summary["cats"]
        objects = summary["objects"]
        self.pos = defaultdict(list)
        self.neg = defaultdict(list)
        self.cnt = defaultdict(lambda: defaultdict(list))
        for iid in allowed:
            obj = objects.get(str(iid), {})
            for c in self.cats:
                o = obj.get(c)
                if o is None:
                    self.neg[c].append(iid)
                    continue
                if o["n"] >= 1 and o["max_area"] >= MIN_POS_AREA:
                    self.pos[c].append(iid)
                if o["crowd"] == 0 and 1 <= o["n"] <= 4 and o["min_area"] >= MIN_COUNT_AREA:
                    self.cnt[c][o["n"]].append(iid)


def _free(items, used):
    return [i for i in items if i not in used] if used is not None else items


def make_presence(g, n, rng, used=None, tag="presence"):
    pairs = []
    cats = [c for c in g.cats if g.pos[c] and g.neg[c]]
    tries = 0
    while len(pairs) < n and tries < n * 200:
        tries += 1
        c = cats[rng.integers(len(cats))]
        pos, neg = _free(g.pos[c], used), _free(g.neg[c], used)
        if not pos or not neg:
            continue
        a, b = int(pos[rng.integers(len(pos))]), int(neg[rng.integers(len(neg))])
        name = g.cats[c]
        pairs.append(
            {
                "id": f"{tag}_{len(pairs)}",
                "type": "presence",
                "category": name,
                "question": f"Does this image contain {article(name)} {name}?",
                "images": [a, b],
                "answers": ["yes", "no"],
            }
        )
        if used is not None:
            used.update((a, b))
    return pairs


def make_counting(g, n, rng, used=None, tag="counting"):
    pairs = []
    cats = [c for c in g.cats if len([k for k in g.cnt[c] if g.cnt[c][k]]) >= 2]
    tries = 0
    while len(pairs) < n and tries < n * 200:
        tries += 1
        c = cats[rng.integers(len(cats))]
        ks = [k for k in g.cnt[c] if _free(g.cnt[c][k], used)]
        if len(ks) < 2:
            continue
        k1, k2 = rng.choice(ks, size=2, replace=False)
        p1, p2 = _free(g.cnt[c][int(k1)], used), _free(g.cnt[c][int(k2)], used)
        a, b = int(p1[rng.integers(len(p1))]), int(p2[rng.integers(len(p2))])
        name = g.cats[c]
        pairs.append(
            {
                "id": f"{tag}_{len(pairs)}",
                "type": "counting",
                "category": name,
                "question": f"How many {plural(name)} can you see?",
                "images": [a, b],
                "answers": [str(int(k1)), str(int(k2))],
            }
        )
        if used is not None:
            used.update((a, b))
    return pairs


def load_vqa(split):
    key = "train" if split == "train" else "val"
    qs = load_json(RAW / f"v2_OpenEnded_mscoco_{key}2014_questions.json")["questions"]
    ans = load_json(RAW / f"v2_mscoco_{key}2014_annotations.json")["annotations"]
    pairs = load_json(RAW / f"v2_mscoco_{key}2014_complementary_pairs.json")
    return qs, ans, pairs


def natural_pool(split):
    qs, anns, pairs = load_vqa(split)
    qmap = {q["question_id"]: q for q in qs}
    amap = {a["question_id"]: a for a in anns}
    pool = defaultdict(list)

    def consensus(a):
        norm = [normalize(x["answer"]) for x in a["answers"]]
        ans, cnt = Counter(norm).most_common(1)[0]
        return ans, cnt

    for q1, q2 in pairs:
        if q1 not in amap or q2 not in amap:
            continue
        a1, a2 = amap[q1], amap[q2]
        if a1["answer_type"] != a2["answer_type"] or a1["image_id"] == a2["image_id"]:
            continue
        c1, n1 = consensus(a1)
        c2, n2 = consensus(a2)
        if n1 < 8 or n2 < 8 or c1 == c2:
            continue
        targets = []
        for a, c in ((a1, c1), (a2, c2)):
            targets.append(next(x["answer"] for x in a["answers"] if normalize(x["answer"]) == c))
        pool[a1["answer_type"]].append(
            {
                "id": f"natural_{q1}_{q2}",
                "type": "natural",
                "answer_type": a1["answer_type"],
                "question": qmap[q1]["question"].strip(),
                "images": [a1["image_id"], a2["image_id"]],
                "qids": [q1, q2],
                "answers": targets,
                "consensus": [c1, c2],
                "refs": [[x["answer"] for x in a1["answers"]], [x["answer"] for x in a2["answers"]]],
            }
        )
    return pool


def pick_natural(pool, counts, rng, used=None, tag="natural"):
    out = []
    for answer_type, n in zip(("yes/no", "number", "other"), counts):
        cands = pool[answer_type]
        order = rng.permutation(len(cands))
        taken = 0
        for i in order:
            if taken >= n:
                break
            p = cands[int(i)]
            if used is not None and any(im in used for im in p["images"]):
                continue
            out.append(p)
            taken += 1
            if used is not None:
                used.update(p["images"])
        if taken < n:
            raise RuntimeError(f"not enough natural pairs for {answer_type}: {taken}/{n}")
    return out


def keyed(split, pairs):
    return [{**p, "images": [f"{split}:{i}" for i in p["images"]], "split": split} for p in pairs]


def isolation_report(splits):
    sets = {
        "caption_dev": {f"val:{x['image_id']}" for x in splits["caption"]["dev"]},
        "presence_dev": {i for p in splits["presence"]["dev"] for i in p["images"]},
        "presence_test": {i for p in splits["presence"]["test"] for i in p["images"]},
        "counting_dev": {i for p in splits["counting"]["dev"] for i in p["images"]},
        "counting_test": {i for p in splits["counting"]["test"] for i in p["images"]},
        "natural_dev": {i for p in splits["natural"]["dev"] for i in p["images"]},
        "natural_test": {i for p in splits["natural"]["test"] for i in p["images"]},
    }
    train = {f"train:{x['image_id']}" for x in splits["caption"]["train"]}
    train |= {i for key in ("presence", "counting") for p in splits[key]["train_pool"] for i in p["images"]}
    train |= {i for p in splits["natural"]["train"] for i in p["images"]}
    names = list(sets)
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlaps[f"{a}&{b}"] = len(sets[a] & sets[b])
        overlaps[f"{a}&train"] = len(sets[a] & train)
    assert all(v == 0 for v in overlaps.values()), overlaps
    return {"sizes": {k: len(v) for k, v in sets.items()}, "train_images": len(train), "overlaps": overlaps}


def build(cfg, log):
    rng = np.random.default_rng(SEED)
    download_raw(log)
    tr, va = coco_summary("train"), coco_summary("val")

    train_ids = [i for i in tr["images"] if len(tr["captions"].get(str(i), [])) >= cfg.captions_per_image]
    cap_images = sorted(int(x) for x in rng.choice(train_ids, cfg.n_train_images, replace=False))
    cap_train = [{"image_id": i, "captions": tr["captions"][str(i)][: cfg.captions_per_image]} for i in cap_images]
    g_train = Grounding(tr, cap_images)
    pres_pool = keyed("train", make_presence(g_train, cfg.pool_presence, rng, None, "presence_train"))
    cnt_pool = keyed("train", make_counting(g_train, cfg.pool_counting, rng, None, "counting_train"))
    log.info("train pools: presence=%d counting=%d", len(pres_pool), len(cnt_pool))

    nat_train = keyed("train", pick_natural(natural_pool("train"), cfg.natural_train, rng))
    nat_val_pool = natural_pool("val")
    used = set()
    nat_test = keyed("val", pick_natural(nat_val_pool, cfg.natural_test, rng, used, "natural_test"))
    nat_dev = keyed("val", pick_natural(nat_val_pool, cfg.natural_dev, rng, used, "natural_dev"))

    g_val = Grounding(va, [i for i in va["images"] if i not in used])
    pres_test = keyed("val", make_presence(g_val, cfg.grounding_test_pairs, rng, used, "presence_test"))
    cnt_test = keyed("val", make_counting(g_val, cfg.grounding_test_pairs, rng, used, "counting_test"))
    pres_dev = keyed("val", make_presence(g_val, cfg.grounding_dev_pairs, rng, used, "presence_dev"))
    cnt_dev = keyed("val", make_counting(g_val, cfg.grounding_dev_pairs, rng, used, "counting_dev"))
    for lst, need in ((pres_test, cfg.grounding_test_pairs), (cnt_test, cfg.grounding_test_pairs), (pres_dev, cfg.grounding_dev_pairs), (cnt_dev, cfg.grounding_dev_pairs)):
        assert len(lst) == need, "grounding pair shortfall"

    val_cap = [i for i in va["images"] if i not in used and va["captions"].get(str(i))]
    dev_cap_ids = sorted(int(x) for x in rng.choice(val_cap, cfg.n_dev_caption, replace=False))
    cap_dev = [{"image_id": i, "caption": va["captions"][str(i)][0]} for i in dev_cap_ids]

    splits = {
        "seed": SEED,
        "caption": {"train": cap_train, "dev": cap_dev},
        "presence": {"train_pool": pres_pool, "dev": pres_dev, "test": pres_test},
        "counting": {"train_pool": cnt_pool, "dev": cnt_dev, "test": cnt_test},
        "natural": {"train": nat_train, "dev": nat_dev, "test": nat_test},
    }
    report = isolation_report(splits)
    log.info("isolation ok: %s", report["sizes"])
    DATA.mkdir(parents=True, exist_ok=True)
    save_json(splits, DATA / "splits.json")
    save_json(report, OUT / "split_report.json")
    if not cfg.smoke:
        save_json(splits, SPLITS / "task2_splits.json")
    return splits


def load_splits(cfg, log=None):
    path = DATA / "splits.json"
    if not path.exists():
        build(cfg, log or get_logger("data"))
    return load_json(path)


def image_keys(splits):
    keys = {f"train:{x['image_id']}" for x in splits["caption"]["train"]}
    keys |= {f"val:{x['image_id']}" for x in splits["caption"]["dev"]}
    for name in ("presence", "counting"):
        for part in splits[name].values():
            keys |= {i for p in part for i in p["images"]}
    for part in splits["natural"].values():
        keys |= {i for p in part for i in p["images"]}
    return sorted(keys)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = make_cfg(smoke=args.smoke)
    build(cfg, get_logger("task2-data"))
