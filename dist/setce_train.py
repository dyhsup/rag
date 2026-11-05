"""Train a set-level matcher: score(question, {d1,...,dk}) against a set built
from concatenated passages. Contrastive negatives are topically near-identical
sets with one slot replaced.
"""
import argparse, json, os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAXD = 450


def read(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class DS(Dataset):
    def __init__(self, recs, tok, ml):
        self.r, self.t, self.ml = recs, tok, ml

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        r = self.r[i]
        b = self.t(r["q"], (r["d1"][:MAXD] + " || " + r["d2"][:MAXD]),
                   truncation=True, max_length=self.ml, padding="max_length",
                   return_tensors="pt")
        return {k: v.squeeze(0) for k, v in b.items()}, torch.tensor(float(r["label"]))


def auc(s, y):
    p = [a for a, b in zip(s, y) if b]; q = [a for a, b in zip(s, y) if not b]
    if not p or not q:
        return float("nan")
    return sum(1 for a in p for b in q if a > b) / (len(p) * len(q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/reqce/dist/setce/train.jsonl")
    ap.add_argument("--model", default="/models/ce-msmarco")
    ap.add_argument("--out", default="/root/reqce/model_set")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=384)
    a = ap.parse_args()
    recs = read(a.data)
    qids = sorted({r["qid"].split("#")[0] for r in recs})
    import random
    random.Random(0).shuffle(qids)
    dev = set(qids[:int(len(qids) * 0.1)])
    tr = [r for r in recs if r["qid"].split("#")[0] not in dev]
    va = [r for r in recs if r["qid"].split("#")[0] in dev]
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"train={len(tr)} val={len(va)} device={dev_}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    mdl = AutoModelForSequenceClassification.from_pretrained(a.model).to(dev_)
    opt = torch.optim.AdamW(mdl.parameters(), lr=a.lr)
    dl = DataLoader(DS(tr, tok, a.max_len), batch_size=a.batch, shuffle=True)
    for ep in range(a.epochs):
        mdl.train(); tot = 0; cnt = 0
        for b, y in dl:
            b = {k: v.to(dev_) for k, v in b.items()}; y = y.to(dev_)
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(mdl(**b).logits.view(-1), y)
            loss.backward(); opt.step()
            tot += loss.item(); cnt += 1
            if cnt % 50 == 0:
                print(f"  ep{ep+1} {cnt}/{len(dl)} loss={tot/cnt:.4f}", flush=True)
        mdl.eval(); sc = []
        with torch.inference_mode():
            for i in range(0, len(va), 64):
                c = va[i:i + 64]
                bb = tok([r["q"] for r in c], [r["d1"][:MAXD] + " || " + r["d2"][:MAXD] for r in c],
                         truncation=True, max_length=a.max_len, padding=True, return_tensors="pt").to(dev_)
                sc.extend(mdl(**bb).logits.view(-1).tolist())
        print(f"ep{ep+1} val AUC={auc(sc, [r['label'] for r in va]):.3f}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    mdl.save_pretrained(a.out); tok.save_pretrained(a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
