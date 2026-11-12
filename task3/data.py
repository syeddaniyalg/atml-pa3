import pickle
from dataclasses import dataclass, field
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common.log import get_logger, progress
from common.paths import SPLITS
from common.utils import save_json
from task3.answers import NO_ANSWER, canonical_vista, gold_vista, group_math, grade_math_groups
from task3.config import DATASETS, N_DEV, N_LIVE, N_ORDERS, N_ROLLOUTS, OUT, SEED, WORK, download_data


@dataclass
class Data:
    name: str
    vlm: bool
    qids: list
    question: list
    gold: list
    tokens: np.ndarray
    capped: np.ndarray
    extracted: list
    groups: np.ndarray
    correct: np.ndarray
    canon: list
    L: np.ndarray
    K: np.ndarray
    V: np.ndarray
    offs: np.ndarray
    kl: np.ndarray
    logp: np.ndarray
    vtok: np.ndarray
    texts: np.ndarray
    orders: np.ndarray
    dev: np.ndarray
    eval: np.ndarray
    split_ids: dict = field(default_factory=dict)

    @property
    def n_questions(self):
        return len(self.qids)


def _flat(col):
    arr = col.combine_chunks()
    return arr.offsets.to_numpy(), arr.values.to_numpy(zero_copy_only=False)


def _gather(offsets, values, order):
    lengths = (offsets[1:] - offsets[:-1])[order]
    new_off = np.concatenate([[0], np.cumsum(lengths)])
    idx = np.concatenate([np.arange(offsets[o], offsets[o + 1]) for o in order])
    return new_off[:-1], lengths, values[idx]


def _segment_mean(flat, starts, lengths):
    cs = np.concatenate([[0.0], np.cumsum(flat, dtype=np.float64)])
    return (cs[starts + lengths] - cs[starts]) / np.maximum(lengths, 1)


def _group_job(args):
    answers, gold = args
    labels, canon = group_math(answers)
    ok = grade_math_groups(canon, gold)
    return labels, canon, ok


def _vista_meta():
    from huggingface_hub import HfApi, hf_hub_download

    files = [f for f in HfApi().list_repo_files("AI4Math/MathVista", repo_type="dataset") if f.startswith("data/testmini")]
    path = hf_hub_download("AI4Math/MathVista", files[0], repo_type="dataset")
    t = pq.read_table(path, columns=["pid", "choices", "answer", "answer_type", "precision", "question_type"]).to_pandas()
    return {str(r.pid): {"choices": list(r.choices) if r.choices is not None else [], "answer": r.answer, "answer_type": r.answer_type, "precision": r.precision, "question_type": r.question_type} for r in t.itertuples()}


def make_splits(qids, name):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(qids))
    ordered = [qids[i] for i in perm]
    dev, held = ordered[:N_DEV], ordered[N_DEV:]
    ids = {"dev": dev, "eval": held}
    if name == "math":
        ids["live"] = held[:N_LIVE]
    orng = np.random.default_rng(SEED)
    orders = np.stack([[orng.permutation(N_ROLLOUTS) for _ in range(N_ORDERS)] for _ in qids]).astype(np.int8)
    return ids, orders


def ensure_splits(name):
    root = download_data() / DATASETS[name]["folder"]
    sid = pq.read_table(root / "generations.parquet", columns=["sample_id"])["sample_id"].to_pandas()
    qids = sorted(sid.unique())
    ids, orders = make_splits(qids, name)
    save_json(ids, SPLITS / f"task3_{name}_split_ids.json")
    np.savez_compressed(SPLITS / f"task3_{name}_orders.npz", orders=orders, qids=np.array(qids))
    return ids, orders, qids


def build(name, log, workers=4):
    root = download_data() / DATASETS[name]["folder"]
    vlm = DATASETS[name]["vlm"]
    gen = pq.read_table(root / "generations.parquet")
    prob = pq.read_table(root / "probability_metrics.parquet")
    att = pq.read_table(root / "attention_metrics.parquet") if vlm else None

    def order_of(table):
        sid = table["sample_id"].to_pandas()
        codes, uniques = pd.factorize(sid, sort=True)
        rid = table["rollout_id"].to_numpy()
        return np.lexsort((rid, codes)), list(uniques)

    og, qids = order_of(gen)
    op, qids_p = order_of(prob)
    assert qids == qids_p and len(qids) * N_ROLLOUTS == gen.num_rows
    sid_g = gen["sample_id"].to_numpy(zero_copy_only=False)[og]
    sid_p = prob["sample_id"].to_numpy(zero_copy_only=False)[op]
    assert (sid_g == sid_p).all() and (gen["rollout_id"].to_numpy()[og] == prob["rollout_id"].to_numpy()[op]).all()

    n_ids = pc.list_value_length(gen["generated_token_ids"]).to_numpy()
    tokens = gen["n_generated_text_tokens"].to_numpy()
    po, pv = _flat(prob["logp"])
    ko, kv = _flat(prob["kl_u_p"])
    starts, lengths, logp = _gather(po, pv, op)
    _, klen, kl = _gather(ko, kv, op)
    assert (lengths == klen).all() and (lengths == n_ids[og]).all() and (lengths == tokens[og]).all()
    report = {"rows": gen.num_rows, "questions": len(qids), "tokens_total": int(lengths.sum()), "alignment": "logp/kl/token-id lengths equal for all rollouts"}

    vtok = np.zeros(0, dtype=np.float32)
    V = np.zeros(len(og))
    if vlm:
        oa, _ = order_of(att)
        assert (att["sample_id"].to_numpy(zero_copy_only=False)[oa] == sid_g).all()
        parts = {}
        for key in ("img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"):
            off, val = _flat(att[key])
            s, ln, v = _gather(off, val, oa)
            assert (ln == lengths).all()
            parts[key] = v.astype(np.float64)
        img, prm, gtx = parts["img_attn_total"], parts["prompt_text_attn_total"], parts["generated_text_attn_total"]
        vtok = (img / np.maximum(img + prm + gtx, 1e-12)).astype(np.float32)
        V = _segment_mean(vtok, starts, lengths)
        report["attention_alignment"] = "attention list lengths equal generated token counts"

    L = _segment_mean(logp, starts, lengths)
    K = _segment_mean(kl, starts, lengths)

    q_text = gen["question_text"].to_numpy(zero_copy_only=False)[og].reshape(-1, N_ROLLOUTS)
    gold = gen["gold_answer"].to_numpy(zero_copy_only=False)[og].reshape(-1, N_ROLLOUTS)[:, 0].tolist()
    ext = gen["extracted_answer"].to_pylist()
    ext = [ext[i] for i in og]
    ext_q = [ext[i * N_ROLLOUTS : (i + 1) * N_ROLLOUTS] for i in range(len(qids))]
    texts = gen["generated_text"].to_numpy(zero_copy_only=False)[og].reshape(-1, N_ROLLOUTS)
    capped = gen["hit_max_new_tokens"].to_numpy(zero_copy_only=False)[og].reshape(-1, N_ROLLOUTS)

    cache = WORK / f"answers_{name}.pkl"
    if cache.exists():
        groups, canon, correct = pickle.load(open(cache, "rb"))
    else:
        groups = np.zeros((len(qids), N_ROLLOUTS), dtype=np.int16)
        correct = np.zeros((len(qids), N_ROLLOUTS), dtype=bool)
        canon = []
        if name == "math":
            with Pool(workers) as pool:
                bar = progress(len(qids), "answer groups", unit="q")
                for qi, (labels, cn, ok) in enumerate(pool.imap(_group_job, zip(ext_q, gold), chunksize=8)):
                    groups[qi] = labels
                    canon.append(cn)
                    correct[qi] = [ok[l] for l in labels]
                    bar.update(1)
                bar.close()
        else:
            meta = _vista_meta()
            mismatch = 0
            for qi, qid in enumerate(qids):
                m = meta[qid]
                g = gold_vista(m)
                mismatch += int(str(m["answer"]).strip() != str(gold[qi]).strip())
                forms = [canonical_vista(a, m) for a in ext_q[qi]]
                ids, cn = {NO_ANSWER: 0}, {0: NO_ANSWER}
                for f in forms:
                    if f not in ids:
                        ids[f] = len(ids)
                        cn[ids[f]] = f
                groups[qi] = [ids[f] for f in forms]
                correct[qi] = [(f == g and f != NO_ANSWER) for f in forms]
                canon.append(cn)
            report["gold_mismatch_with_metadata"] = mismatch
        WORK.mkdir(parents=True, exist_ok=True)
        pickle.dump((groups, canon, correct), open(cache, "wb"))

    ids, orders, split_qids = ensure_splits(name)
    assert split_qids == qids
    pos = {q: i for i, q in enumerate(qids)}
    report.update(accuracy_first_rollout=float(correct[:, 0].mean()), accuracy_all_rollouts=float(correct.mean()), no_answer_rate=float((groups == 0).mean()), capped_rate=float(capped.mean()))
    save_json(report, OUT / f"data_report_{name}.json")
    log.info("%s: %s", name, report)
    return Data(
        name=name, vlm=vlm, qids=qids, question=q_text[:, 0].tolist(), gold=gold, tokens=lengths.reshape(-1, N_ROLLOUTS), capped=capped, extracted=ext_q,
        groups=groups, correct=correct, canon=canon, L=L.reshape(-1, N_ROLLOUTS), K=K.reshape(-1, N_ROLLOUTS), V=V.reshape(-1, N_ROLLOUTS),
        offs=starts.reshape(-1, N_ROLLOUTS), kl=kl, logp=logp, vtok=vtok, texts=texts, orders=orders,
        dev=np.array([pos[q] for q in ids["dev"]]), eval=np.array([pos[q] for q in ids["eval"]]), split_ids=ids,
    )


def load(name, log=None):
    return build(name, log or get_logger(f"task3-{name}"))
