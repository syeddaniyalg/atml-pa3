import numpy as np
import torch
from datasets import load_dataset

from common.paths import SPLITS
from common.utils import get_device, load_json, save_json
from task1.config import MODEL, OUT, SEED
from task1.models import chat_prompt, load_base, load_tokenizer

CONDITIONS = [
    ("fullft", "mid"),
    ("fullft", "final"),
    ("l2sp", "final"),
    ("lora4", "final"),
    ("lora16", "final"),
    ("oplora16", "final"),
]


def probe_set(cfg):
    path = SPLITS / "task1_probe_ids.json"
    if path.exists() and not cfg.smoke:
        ids = load_json(path)
    else:
        rng = np.random.default_rng(SEED)
        gsm = load_dataset("openai/gsm8k", "main", split="test")
        hs = load_dataset("Rowan/hellaswag", split="validation")
        ids = {
            "gsm8k": sorted(rng.choice(len(gsm), cfg.probe_n, replace=False).tolist()),
            "hellaswag": sorted(rng.choice(len(hs), cfg.probe_n, replace=False).tolist()),
        }
        if not cfg.smoke:
            save_json(ids, path)
    return ids


def _hellaswag_prompt(doc):
    def clean(text):
        text = text.strip().replace(" [title]", ". ")
        import re

        text = re.sub(r"\[.*?\]", "", text)
        return text.replace("  ", " ")

    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    return clean(doc["activity_label"] + ": " + ctx)


def probe_prompts(cfg, tok):
    ids = probe_set(cfg)
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    hs = load_dataset("Rowan/hellaswag", split="validation")
    prompts = []
    for i in ids["gsm8k"]:
        prompts.append(("math", i, chat_prompt(tok, "Q: " + gsm[i]["question"] + "\nA:")))
    for i in ids["hellaswag"]:
        prompts.append(("nonmath", i, _hellaswag_prompt(hs[i])))
    return prompts


@torch.no_grad()
def cache_activations(cfg, tok, device):
    model = load_base(cfg, torch.float32, device).eval()
    layers = cfg.diag_layers
    store = {l: [] for l in layers}
    hooks = []
    state = {}

    def make_hook(l):
        def hook(module, args):
            store[l].append(args[0][0, state["last"]].detach().float().cpu())

        return hook

    for l in layers:
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(make_hook(l)))
    prompts = probe_prompts(cfg, tok)
    kinds = []
    for kind, idx, text in prompts:
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        state["last"] = int(enc["attention_mask"][0].nonzero().max())
        model(**enc)
        kinds.append((kind, idx))
    for h in hooks:
        h.remove()
    z = {l: torch.stack(store[l]) for l in layers}
    del model
    return z, kinds


def base_down_proj(cfg, device):
    model = load_base(cfg, torch.float32, device)
    out = {l: model.model.layers[l].mlp.down_proj.weight.detach().clone() for l in cfg.diag_layers}
    del model
    return out


def run_diagnostics(cfg, log):
    device = get_device()
    tok = load_tokenizer()
    w0 = base_down_proj(cfg, device)
    z, kinds = cache_activations(cfg, tok, device)
    is_math = torch.tensor([k == "math" for k, _ in kinds])
    k = cfg.protect_k
    bases = {}
    for l, w in w0.items():
        u, _, _ = torch.linalg.svd(w, full_matrices=False)
        bases[l] = u[:, : min(k, u.shape[1])]
    rows = []
    for method, stage in CONDITIONS:
        path = OUT / "runs" / method / f"delta_down_proj_{stage}.pt"
        if not path.exists():
            log.info("missing %s", path)
            continue
        deltas = torch.load(path, map_location=device)
        for l in cfg.diag_layers:
            dw = deltas[f"L{l}"].to(device).float()
            uk = bases[l]
            rho = (uk @ (uk.T @ dw)).pow(2).sum() / dw.pow(2).sum().clamp_min(1e-30)
            kappa = dw.norm() / w0[l].norm()
            zl = z[l].to(device)
            ratio = (zl @ dw.T).norm(dim=1) / ((zl @ w0[l].T).norm(dim=1) + 1e-8)
            rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "layer": l,
                    "rho_k": rho.item(),
                    "kappa": kappa.item(),
                    "R_math_mean": ratio[is_math.to(device)].mean().item(),
                    "R_math_median": ratio[is_math.to(device)].median().item(),
                    "R_nonmath_mean": ratio[~is_math.to(device)].mean().item(),
                    "R_nonmath_median": ratio[~is_math.to(device)].median().item(),
                    "R_math": ratio[is_math.to(device)].cpu().tolist(),
                    "R_nonmath": ratio[~is_math.to(device)].cpu().tolist(),
                }
            )
    save_json(rows, OUT / "diagnostics.json")
    save_json({"model": MODEL, "protect_k": k, "layers": list(cfg.diag_layers), "n_probe": len(kinds)}, OUT / "diagnostics_meta.json")
    log.info("diagnostics: %d rows", len(rows))
    return rows


def verify_oplora(cfg, log):
    path = OUT / "runs" / "oplora16" / "adapter.pt"
    if not path.exists():
        return None
    device = get_device()
    exported = torch.load(path, map_location="cpu")
    model = load_base(cfg, torch.float32, device)
    names = list(exported["factors"])
    sample = [n for n in names if ".layers.0." in n or ".layers.13." in n]
    bases = {}
    for n in sample:
        w = dict(model.named_modules())[n].weight.detach().float()
        u, _, vh = torch.linalg.svd(w, full_matrices=False)
        kk = min(cfg.protect_k, *w.shape)
        bases[n] = (u[:, :kk], vh[:kk].T)
    out = {}
    for n in sample:
        a, b = exported["factors"][n]
        uk, vk = bases[n]
        out[n] = {
            "norm_UkT_B": (uk.T @ b.to(device)).norm().item(),
            "norm_A_Vk": (a.to(device) @ vk).norm().item(),
            "norm_B": b.norm().item(),
            "norm_A": a.norm().item(),
        }
    save_json(out, OUT / "oplora_constraint_check.json")
    worst = max(max(v["norm_UkT_B"] / max(v["norm_B"], 1e-12), v["norm_A_Vk"] / max(v["norm_A"], 1e-12)) for v in out.values())
    log.info("oplora constraint check: worst relative leakage %.2e", worst)
    return out
