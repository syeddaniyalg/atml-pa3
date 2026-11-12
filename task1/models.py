import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from task1.config import MODEL, SEED


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base(cfg, dtype=torch.float16, device="cpu"):
    if cfg.tiny:
        conf = AutoConfig.from_pretrained(MODEL)
        conf.hidden_size = 64
        conf.intermediate_size = 128
        conf.num_hidden_layers = 4
        conf.num_attention_heads = 4
        conf.num_key_value_heads = 2
        conf.head_dim = 16
        conf.max_window_layers = 4
        conf.layer_types = ["full_attention"] * 4 if hasattr(conf, "layer_types") else None
        if conf.layer_types is None:
            del conf.layer_types
        gen_state = torch.random.get_rng_state()
        torch.manual_seed(SEED)
        model = AutoModelForCausalLM.from_config(conf, torch_dtype=torch.float32)
        torch.random.set_rng_state(gen_state)
        model = model.to(dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype)
    model.config.use_cache = False
    return model.to(device)


def chat_prompt(tok, user, system=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
