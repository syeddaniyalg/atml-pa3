from dataclasses import dataclass, replace

from common.paths import CACHE, RESULTS

MODEL = "Qwen/Qwen3-0.6B"
SEED = 6304
OUT = RESULTS / "task1"
DATA = CACHE / "task1"

METHODS = {
    "fullft": dict(kind="full", lr=2e-5, lam=0.0),
    "l2sp": dict(kind="full", lr=2e-5, lam=1e-3),
    "lora4": dict(kind="lora", r=4, alpha=8, lr=2e-4),
    "lora16": dict(kind="lora", r=16, alpha=32, lr=2e-4),
    "oplora16": dict(kind="oplora", r=16, alpha=32, k=128, lr=2e-4),
}

EVAL_STAGES = {
    "fullft": ("baseline", "mid", "final"),
    "l2sp": ("final",),
    "lora4": ("final",),
    "lora16": ("final",),
    "oplora16": ("final",),
}


@dataclass(frozen=True)
class Cfg:
    n_examples: int = 40000
    max_len: int = 2048
    micro: int = 2
    accum: int = 16
    updates: int = 1250
    warmup: int = 38
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    clip: float = 1.0
    dropout: float = 0.05
    ckpt_updates: tuple = (312, 625, 937, 1250)
    mid_update: int = 625
    diag_layers: tuple = (0, 13, 27)
    protect_k: int = 128
    probe_n: int = 32
    eval_limit: int = 0
    gsm_bs: int = 48
    ll_bs: int = 8
    gsm_max_new: int = 1024
    qual_max_new: int = 512
    qual_bs: int = 8
    log_every: int = 10
    tiny: bool = False
    smoke: bool = False
    keep_ckpt: int = 1

    @property
    def batch(self):
        return self.micro * self.accum


def make_cfg(smoke=False, tiny=False):
    cfg = Cfg()
    if smoke:
        cfg = replace(
            cfg,
            n_examples=128,
            updates=4,
            warmup=1,
            ckpt_updates=(1, 2, 3, 4),
            mid_update=2,
            eval_limit=8,
            gsm_max_new=48,
            qual_max_new=24,
            log_every=1,
            smoke=True,
            gsm_bs=8,
            ll_bs=8,
        )
    if tiny:
        cfg = replace(cfg, tiny=True, diag_layers=(0, 1, 3), protect_k=16)
    return cfg
