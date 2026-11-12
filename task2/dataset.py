import random
from dataclasses import dataclass, field

import numpy as np

from task2.config import SEED

CAPTION_Q = "Briefly describe the image."
SHORT = " Answer the question using a single word or phrase."


@dataclass
class Example:
    image: str
    question: str
    target: str
    kind: str
    pair_id: str = ""
    refs: list = field(default_factory=list)


def caption_examples(rows, split="train"):
    out = []
    for r in rows:
        caps = r["captions"] if "captions" in r else [r["caption"]]
        for c in caps:
            out.append(Example(f"{split}:{r['image_id']}", CAPTION_Q, c.strip(), "caption"))
    return out


def pair_examples(pairs, kind=None):
    out = []
    for p in pairs:
        refs = p.get("refs", [[], []])
        for i in range(2):
            out.append(
                Example(p["images"][i], p["question"] + SHORT, p["answers"][i], kind or p["type"], p["id"], refs[i])
            )
    return out


class MixSampler:
    def __init__(self, splits):
        self.captions = caption_examples(splits["caption"]["train"])
        self.pools = {
            "presence": splits["presence"]["train_pool"],
            "counting": splits["counting"]["train_pool"],
            "natural": splits["natural"]["train"],
        }
        self.rng = random.Random(SEED)
        self.order = []
        self.pos = 0

    def state(self):
        return {"rng": self.rng.getstate(), "order": self.order, "pos": self.pos}

    def load(self, state):
        self.rng.setstate(state["rng"])
        self.order, self.pos = state["order"], state["pos"]

    def caption_stream(self, n):
        out = []
        while len(out) < n:
            if self.pos >= len(self.order):
                self.order = list(range(len(self.captions)))
                self.rng.shuffle(self.order)
                self.pos = 0
            take = min(n - len(out), len(self.order) - self.pos)
            out += [self.captions[i] for i in self.order[self.pos : self.pos + take]]
            self.pos += take
        return out

    def draw(self, mix, draws):
        kinds, probs = zip(*mix)
        examples, counts = [], {k: 0 for k in kinds}
        for _ in range(draws):
            r, acc, kind = self.rng.random(), 0.0, kinds[-1]
            for k, p in zip(kinds, probs):
                acc += p
                if r < acc:
                    kind = k
                    break
            counts[kind] += 1
            if kind == "caption":
                examples += self.rng.sample(self.captions, 2)
            else:
                pair = self.rng.choice(self.pools[kind])
                examples += pair_examples([pair], kind)
        return examples, counts

    def batch(self, stage, size):
        if stage.name == "caption":
            return self.caption_stream(size), {"caption": size}
        return self.draw(stage.mix, size // 2)


def build_eval_items(pairs, seed=SEED):
    rng = np.random.default_rng(seed)
    all_images = [i for p in pairs for i in p["images"]]
    items = []
    for p in pairs:
        pool = [i for i in all_images if i not in p["images"]]
        refs = p.get("refs", [[], []])
        for i in (0, 1):
            items.append(
                {
                    "pair_id": p["id"],
                    "slot": i,
                    "type": p["type"],
                    "subtype": p.get("answer_type", p["type"]),
                    "question": p["question"] + SHORT,
                    "image": p["images"][i],
                    "partner": p["images"][1 - i],
                    "mismatched": pool[int(rng.integers(len(pool)))],
                    "target": p["answers"][i],
                    "partner_target": p["answers"][1 - i],
                    "refs": refs[i],
                }
            )
    return items
