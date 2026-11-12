import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.lora import inject_lora
from task2.config import LLM, SEED

PREFIX = "<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Connector(nn.Module):
    def __init__(self, din=768, dmid=1024, dout=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, dmid), nn.GELU(), nn.Linear(dmid, dout))

    def forward(self, x):
        return self.net(x)


def pool_tokens(feats, tokens):
    if tokens == 196:
        return feats
    b, n, d = feats.shape
    side = int(n**0.5)
    grid = feats.view(b, side, side, d).permute(0, 3, 1, 2)
    grid = F.avg_pool2d(grid, 2)
    return grid.flatten(2).transpose(1, 2)


class VLM(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(LLM)
        self.lm = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.float16).to(device)
        self.lm.config.use_cache = False
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.connector = Connector().to(device)
        self.register_buffer("scale", torch.ones((), device=device))
        self.embed = self.lm.get_input_embeddings()
        self.end_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pre_ids = torch.tensor(self.tok(PREFIX, add_special_tokens=False).input_ids, device=device)
        self.lora_on = False
        self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self._question_cache = {}

    def enable_lora(self):
        inject_lora(
            self.lm,
            ("q_proj", "k_proj", "v_proj", "o_proj"),
            self.cfg.lora_r,
            self.cfg.lora_alpha,
            self.cfg.lora_dropout,
            SEED,
        )
        self.lora_on = True

    def lora_params(self):
        return [p for n, p in self.lm.named_parameters() if "lora_" in n]

    def connector_params(self):
        return list(self.connector.parameters())

    def question_ids(self, question):
        ids = self._question_cache.get(question)
        if ids is None:
            q = self.tok(question, add_special_tokens=False).input_ids[: self.cfg.q_cap]
            tail = self.tok(SUFFIX, add_special_tokens=False).input_ids
            ids = torch.tensor(q + tail, device=self.device)
            self._question_cache[question] = ids
        return ids

    def target_ids(self, text, kind):
        cap = self.cfg.cap_cap if kind == "caption" else self.cfg.ans_cap
        ids = self.tok(text, add_special_tokens=False).input_ids[:cap]
        return torch.tensor(ids + [self.end_id], device=self.device)

    def visual(self, feats):
        feats = pool_tokens(feats.to(self.device).float(), self.cfg.tokens)
        return self.connector(feats) * self.scale

    def build(self, vis, questions, targets=None):
        embeds, labels, lengths = [], [], []
        for b, q in enumerate(questions):
            parts = [self.embed(self.pre_ids)]
            lab = [torch.full((len(self.pre_ids),), -100, device=self.device)]
            if vis is not None:
                parts.append(vis[b].to(torch.float16))
                lab.append(torch.full((vis.shape[1],), -100, device=self.device))
            qi = self.question_ids(q)
            parts.append(self.embed(qi))
            lab.append(torch.full((len(qi),), -100, device=self.device))
            if targets is not None:
                parts.append(self.embed(targets[b]))
                lab.append(targets[b])
            embeds.append(torch.cat(parts))
            labels.append(torch.cat(lab))
            lengths.append(len(embeds[-1]))
        return embeds, labels, lengths

    def pad_right(self, embeds, labels):
        width = max(e.shape[0] for e in embeds)
        d = embeds[0].shape[1]
        x = torch.zeros(len(embeds), width, d, dtype=torch.float16, device=self.device)
        y = torch.full((len(embeds), width), -100, dtype=torch.long, device=self.device)
        m = torch.zeros(len(embeds), width, dtype=torch.long, device=self.device)
        for i, (e, l) in enumerate(zip(embeds, labels)):
            x[i, : e.shape[0]] = e
            y[i, : l.shape[0]] = l
            m[i, : e.shape[0]] = 1
        return x, y, m

    def token_nll(self, feats, questions, targets):
        vis = self.visual(feats) if feats is not None else None
        embeds, labels, _ = self.build(vis, questions, targets)
        x, y, m = self.pad_right(embeds, labels)
        hidden = self.lm.model(inputs_embeds=x, attention_mask=m, use_cache=False).last_hidden_state
        tgt = y[:, 1:]
        sel = tgt != -100
        logits = F.linear(hidden[:, :-1][sel], self.lm.lm_head.weight).float()
        nll = F.cross_entropy(logits, tgt[sel], reduction="none")
        rows = torch.arange(len(questions), device=self.device)[:, None].expand_as(tgt)[sel]
        return nll, rows, tgt[sel]

    def example_loss(self, feats, questions, targets, drop_end=False):
        nll, rows, toks = self.token_nll(feats, questions, targets)
        if drop_end:
            keep = toks != self.end_id
            nll, rows = nll[keep], rows[keep]
        n = len(questions)
        total = torch.zeros(n, device=self.device).index_add_(0, rows, nll)
        count = torch.zeros(n, device=self.device).index_add_(0, rows, torch.ones_like(nll)).clamp_min(1)
        return total / count

    @torch.no_grad()
    def generate(self, feats, questions, max_new=16):
        vis = self.visual(feats) if feats is not None else None
        embeds, _, lengths = self.build(vis, questions)
        width = max(lengths)
        d = embeds[0].shape[1]
        n = len(embeds)
        x = torch.zeros(n, width, d, dtype=torch.float16, device=self.device)
        mask = torch.zeros(n, width, dtype=torch.long, device=self.device)
        for i, e in enumerate(embeds):
            x[i, width - e.shape[0] :] = e
            mask[i, width - e.shape[0] :] = 1
        pos = (mask.cumsum(1) - 1).clamp_min(0)
        out = self.lm(inputs_embeds=x, attention_mask=mask, position_ids=pos, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1)
        done = torch.zeros(n, dtype=torch.bool, device=self.device)
        seqs = [[] for _ in range(n)]
        last_pos = pos[:, -1:]
        for _ in range(max_new):
            for i in range(n):
                if not done[i] and nxt[i].item() != self.end_id:
                    seqs[i].append(nxt[i].item())
            done |= nxt == self.end_id
            if done.all():
                break
            mask = torch.cat([mask, torch.ones(n, 1, dtype=torch.long, device=self.device)], 1)
            last_pos = last_pos + 1
            out = self.lm(input_ids=nxt[:, None], attention_mask=mask, position_ids=last_pos, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
        return [self.tok.decode(s, skip_special_tokens=True).strip() for s in seqs]

    def trainable_summary(self):
        return {
            "connector": sum(p.numel() for p in self.connector.parameters() if p.requires_grad),
            "lora": sum(p.numel() for p in self.lora_params() if p.requires_grad),
        }

    def sanity(self, feats):
        shapes = {"feature": list(feats.shape[1:]), "connector_out": list(self.visual(feats[:1]).shape[1:])}
        frozen_ok = all(not p.requires_grad for n, p in self.lm.named_parameters() if "lora_" not in n)
        return {"shapes": shapes, "backbone_frozen": frozen_ok}
