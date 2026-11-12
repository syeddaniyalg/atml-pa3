from dataclasses import dataclass, replace

from common.paths import CACHE, RESULTS

LLM = "Qwen/Qwen3-0.6B"
VISION = "google/siglip-base-patch16-224"
SEED = 6304
OUT = RESULTS / "task2"
DATA = CACHE / "task2"

COCO_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
COCO_IMG = "http://images.cocodataset.org/{split}2014/COCO_{split}2014_{iid:012d}.jpg"
VQA_BASE = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/"
VQA_FILES = {
    "q_train": "v2_Questions_Train_mscoco.zip",
    "q_val": "v2_Questions_Val_mscoco.zip",
    "a_train": "v2_Annotations_Train_mscoco.zip",
    "a_val": "v2_Annotations_Val_mscoco.zip",
    "c_train": "v2_Complementary_Pairs_Train_mscoco.zip",
    "c_val": "v2_Complementary_Pairs_Val_mscoco.zip",
}


@dataclass(frozen=True)
class Stage:
    name: str
    updates: int
    conn_lr: float
    lora_lr: float
    mix: tuple


STAGES = (
    Stage("caption", 2250, 1e-3, 0.0, (("caption", 1.0),)),
    Stage("vit", 1500, 1e-4, 2e-5, (("presence", 0.5), ("counting", 0.25), ("caption", 0.25))),
    Stage("natural", 750, 5e-5, 1e-5, (("natural", 0.7), ("presence", 0.1), ("counting", 0.1), ("caption", 0.1))),
)


@dataclass(frozen=True)
class Cfg:
    tokens: int = 196
    micro: int = 8
    accum: int = 2
    n_train_images: int = 6000
    captions_per_image: int = 3
    n_dev_caption: int = 96
    pool_presence: int = 1200
    pool_counting: int = 1200
    grounding_dev_pairs: int = 12
    grounding_test_pairs: int = 32
    natural_train: tuple = (640, 256, 640)
    natural_dev: tuple = (12, 8, 12)
    natural_test: tuple = (36, 24, 36)
    weight_decay: float = 0.01
    clip: float = 1.0
    warmup_frac: float = 0.05
    final_lr_frac: float = 0.1
    q_cap: int = 80
    cap_cap: int = 64
    ans_cap: int = 16
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    ckpt_every: int = 250
    dev_every: int = 500
    dev_points: tuple = (2250, 3750, 4500)
    gen_bs: int = 8
    max_new: int = 16
    bootstrap: int = 4000
    stage_updates: tuple = (2250, 1500, 750)
    log_every: int = 25
    smoke: bool = False
    tiny: bool = False
    n_img_workers: int = 24

    @property
    def batch(self):
        return self.micro * self.accum

    @property
    def total_updates(self):
        return sum(self.stage_updates)


def make_cfg(tokens=196, smoke=False, tiny=False):
    cfg = Cfg(tokens=tokens)
    if smoke:
        cfg = replace(
            cfg,
            n_train_images=48,
            n_dev_caption=8,
            pool_presence=16,
            pool_counting=16,
            grounding_dev_pairs=2,
            grounding_test_pairs=2,
            natural_train=(8, 4, 8),
            natural_dev=(2, 1, 2),
            natural_test=(2, 2, 2),
            stage_updates=(3, 3, 2),
            ckpt_every=2,
            dev_every=3,
            dev_points=(3, 6, 8),
            bootstrap=50,
            log_every=1,
            smoke=True,
            n_img_workers=8,
        )
    if tiny:
        cfg = replace(cfg, tiny=True)
    return cfg
