"""Train the set-conditioned CMU scorer from counterfactual utility labels.

Rows: {qid, q, sel: [already-selected passages], cand, delta}
Within each (qid, sel) group the highest-delta candidate is the positive and
every other candidate is a negative. The scorer sees (q + sel, cand), so it
learns complementarity: a passage is good because of what the set is missing,
not because it is globally relevant.

--pointwise drops the selected set from the input (same target, no set
conditioning) and is the ablation that isolates the contribution.

--infer-answer additionally feeds the gold answer at training time only; it is
used to check that the labels are learnable at all.
"""
import argparse, json, os, random, sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.cmu_fmt import text_a, text_b  # noqa: E402


def read(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class PairDS(Dataset):
    def __init__(self, pairs, tok, ml, pointwise):
        self.p, self.t, self.ml, self.pw = pairs, tok, ml, pointwise

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        q, sel, pos, neg = self.p[i]
        a = q if self.pw else text_a(q, sel)
        kw = dict(truncation=True, max_length=self.ml, padding="max_length",
                  return_tensors="pt")
        bp = self.t(a, text_b(pos), **kw)
        bn = self.t(a, text_b(neg), **kw)
        return ({k: v.squeeze(0) for k, v in bp.items()},
                {k: v.squeeze(0) for k, v in bn.items()})


def build_pairs(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["qid"], tuple(r["sel"])), []).append(r)
    pairs = []
    for (_qid, sel), items in groups.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: -x["delta"])
        pos = items[0]
        for neg in items[1:]:
            pairs.append((pos["q"], list(sel), pos["cand"], neg["cand"]))
    return pairs


def split_by_qid(pairs):
    qids = sorted({p[0] for p in pairs})
    rnd = random.Random(0)
    rnd.shuffle(qids)
    cut = max(1, int(len(qids) * 0.1))
    dev_q = set(qids[:cut])
    dev = [p for p in pairs if p[0] in dev_q]
    tr = [p for p in pairs if p[0] not in dev_q]
    return tr, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "train_labels.jsonl"))
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "model"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--pointwise", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=0)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    rows = read(a.data)
    pairs = build_pairs(rows)
    tr, dev = split_by_qid(pairs)
    random.Random(0).shuffle(tr)
    if a.max_pairs and len(tr) > a.max_pairs:
        tr = tr[:a.max_pairs]
    print(f"rows={len(rows)} pairs={len(pairs)} train={len(tr)} dev={len(dev)} "
          f"pointwise={a.pointwise}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    mdl = AutoModelForSequenceClassification.from_pretrained(a.model)
    opt = torch.optim.AdamW(mdl.parameters(), lr=a.lr)
    dl = DataLoader(PairDS(tr, tok, a.max_len, a.pointwise), batch_size=a.batch, shuffle=True)
    ddl = DataLoader(PairDS(dev, tok, a.max_len, a.pointwise), batch_size=a.batch)

    for ep in range(a.epochs):
        mdl.train()
        tot = cnt = 0
        for bp, bn in dl:
            opt.zero_grad()
            sp = mdl(**bp).logits.view(-1)
            sn = mdl(**bn).logits.view(-1)
            loss = torch.nn.functional.relu(a.margin - (sp - sn)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()
            tot += loss.item(); cnt += 1
            if cnt % 100 == 0:
                print(f"  ep{ep+1} {cnt}/{len(dl)} loss={tot/cnt:.4f}", flush=True)
        mdl.eval()
        correct = total = 0
        with torch.inference_mode():
            for bp, bn in ddl:
                sp = mdl(**bp).logits.view(-1)
                sn = mdl(**bn).logits.view(-1)
                correct += int((sp > sn).sum()); total += sp.numel()
        print(f"ep{ep+1} train_loss={tot/max(cnt,1):.4f} dev_pair_acc={correct/max(total,1):.3f}",
              flush=True)

    os.makedirs(a.out, exist_ok=True)
    mdl.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
