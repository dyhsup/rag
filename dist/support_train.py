"""Train a matcher on counterfactual deletion: the full passage must outrank the
passage with its load-bearing sentence removed, while staying close to the
passage with a control sentence removed."""
import argparse, json, os, random
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def drop(sents, i):
    return " ".join(s for j, s in enumerate(sents) if j != i)


class DS(Dataset):
    def __init__(self, recs, tok, ml):
        self.r, self.t, self.ml = recs, tok, ml

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        r = self.r[i]
        mk = lambda txt: self.t(r["q"], txt, truncation=True, max_length=self.ml,
                                padding="max_length", return_tensors="pt")
        return ({k: v.squeeze(0) for k, v in mk(r["text"]).items()},
                {k: v.squeeze(0) for k, v in mk(drop(r["sents"], r["sup"])).items()},
                {k: v.squeeze(0) for k, v in mk(drop(r["sents"], r["ctrl"])).items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/reqce/dist/support/train.jsonl")
    ap.add_argument("--model", default="/models/ce-msmarco")
    ap.add_argument("--out", default="/root/reqce/model_support")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--margin", type=float, default=0.5)
    a = ap.parse_args()
    recs = read(a.data)
    qids = sorted({r["qid"] for r in recs}); random.Random(0).shuffle(qids)
    dev = set(qids[:max(1, len(qids) // 10)])
    tr = [r for r in recs if r["qid"] not in dev]
    va = [r for r in recs if r["qid"] in dev]
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"train={len(tr)} val={len(va)} device={dev_}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    mdl = AutoModelForSequenceClassification.from_pretrained(a.model).to(dev_)
    opt = torch.optim.AdamW(mdl.parameters(), lr=a.lr)
    dl = DataLoader(DS(tr, tok, a.max_len), batch_size=a.batch, shuffle=True)
    for ep in range(a.epochs):
        mdl.train(); tot = 0; cnt = 0
        for bf, bs, bc in dl:
            bf = {k: v.to(dev_) for k, v in bf.items()}
            bs = {k: v.to(dev_) for k, v in bs.items()}
            bc = {k: v.to(dev_) for k, v in bc.items()}
            opt.zero_grad()
            sf = mdl(**bf).logits.view(-1)
            ss = mdl(**bs).logits.view(-1)
            sc = mdl(**bc).logits.view(-1)
            loss = (torch.nn.functional.relu(a.margin - (sf - ss)).mean()
                    + 0.5 * torch.nn.functional.relu((sf - sc) - 0.1).abs().mean())
            loss.backward(); opt.step()
            tot += loss.item(); cnt += 1
            if cnt % 50 == 0:
                print(f"  ep{ep+1} {cnt}/{len(dl)} loss={tot/cnt:.4f}", flush=True)
        mdl.eval(); corr = 0; tot_n = 0
        with torch.inference_mode():
            for i in range(0, len(va), 32):
                c = va[i:i + 32]
                tk = lambda k, txt: tok([r[k] for r in c], txt, truncation=True,
                                        max_length=a.max_len, padding=True,
                                        return_tensors="pt").to(dev_)
                sf = mdl(**tk("q", [r["text"] for r in c])).logits.view(-1)
                ss = mdl(**tk("q", [drop(r["sents"], r["sup"]) for r in c])).logits.view(-1)
                corr += int((sf > ss).sum()); tot_n += sf.numel()
        print(f"ep{ep+1} deletion_acc={corr/max(tot_n,1):.3f}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    mdl.save_pretrained(a.out); tok.save_pretrained(a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
