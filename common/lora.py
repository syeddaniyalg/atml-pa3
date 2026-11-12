import math
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

LLM_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base, name, r, alpha, dropout, seed, uk=None, vk=None):
        super().__init__()
        self.base = base
        self.name = name
        self.r = r
        self.scale = alpha / r
        din, dout = base.in_features, base.out_features
        gen = torch.Generator().manual_seed((zlib.crc32(name.encode()) + seed) % (2**31))
        bound = 1.0 / math.sqrt(din)
        device = base.weight.device
        self.lora_A = nn.Parameter(((torch.rand(r, din, generator=gen) * 2 - 1) * bound).to(device))
        self.lora_B = nn.Parameter(torch.zeros(dout, r, device=device))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("uk", uk, persistent=False)
        self.register_buffer("vk", vk, persistent=False)

    @property
    def projected(self):
        return self.uk is not None

    def factors(self):
        with torch.autocast(device_type="cuda", enabled=False):
            a, b = self.lora_A.float(), self.lora_B.float()
            if self.projected:
                b = b - self.uk @ (self.uk.T @ b)
                a = a - (a @ self.vk) @ self.vk.T
        return a, b

    def delta(self):
        a, b = self.factors()
        return self.scale * (b @ a)

    def forward(self, x):
        out = self.base(x)
        a, b = self.factors()
        h = F.linear(self.dropout(x), a.to(x.dtype))
        h = F.linear(h, b.to(x.dtype))
        return out + h.to(out.dtype) * self.scale


def decoder_linears(model, targets=LLM_TARGETS):
    found = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets and ".layers." in name:
            found.append(name)
    return found


def _set_module(model, name, new):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def inject_lora(model, targets, r, alpha, dropout, seed, bases=None):
    names = decoder_linears(model, targets)
    for name in names:
        base = dict(model.named_modules())[name]
        uk = vk = None
        if bases is not None:
            uk, vk = bases[name]
        _set_module(model, name, LoRALinear(base, name, r, alpha, dropout, seed, uk, vk))
    return names


def lora_modules(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def compute_bases(model, targets, k):
    bases = {}
    for name in decoder_linears(model, targets):
        w = dict(model.named_modules())[name].weight.detach().float()
        kk = min(k, w.shape[0], w.shape[1])
        u, _, vh = torch.linalg.svd(w, full_matrices=False)
        bases[name] = (u[:, :kk].contiguous(), vh[:kk].T.contiguous())
    return bases


def basis_bytes(bases):
    return sum(u.numel() * 4 + v.numel() * 4 for u, v in bases.values())


def export_factors(model):
    out = {}
    for name, m in lora_modules(model).items():
        a, b = m.factors()
        out[name] = (a.detach().cpu(), b.detach().cpu())
    scale = next(iter(lora_modules(model).values())).scale
    return {"factors": out, "scale": scale}


@torch.no_grad()
def merge_factors(model, exported):
    modules = dict(model.named_modules())
    scale = exported["scale"]
    for name, (a, b) in exported["factors"].items():
        lin = modules[name]
        delta = scale * (b.to(lin.weight.device).float() @ a.to(lin.weight.device).float())
        lin.weight.copy_((lin.weight.float() + delta).to(lin.weight.dtype))
    return model
