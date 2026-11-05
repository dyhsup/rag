"""Summarize domain-CE runs vs zero-shot CE and IRCoT."""
import json
import os

from . import eval as ev

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


def load(p):
    if not os.path.exists(p):
        return {}
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def main():
    rows = []
    for tag in ["musique200", "wiki2_200", "hotpot200", "squad_200", "trivia_200"]:
        r = load(os.path.join(RES, f"e13_select_{tag}.jsonl"))
        v = load(os.path.join(RES, f"g8_{tag}_vanilla_ce.jsonl"))
        i = load(os.path.join(RES, f"g8_{tag}_ircot_ce.jsonl"))
        rz = load(os.path.join(RES, f"e13_select_{tag}_zeroshot.jsonl"))
        iz = load(os.path.join(RES, f"g8_{tag}_ircot_ce_zeroshot.jsonl"))
        ks = [k for k in r if k in i]
        if not ks:
            continue
        em = lambda x: ev.mean([x[k]["em"] for k in ks])
        d = ev.paired_bootstrap([r[k]["em"] for k in ks], [i[k]["em"] for k in ks])
        row = {"tag": tag, "n": len(ks), "vanilla_ce": em(v) if v else None,
               "ircot_ce": em(i), "rbg": em(r), "rbg_minus_ircot": d,
               "ircot_zeroshot": em(iz) if iz else None,
               "rbg_zeroshot": em(rz) if rz else None}
        rows.append(row)
        print(f"{tag:12s} n={len(ks):3d} vanilla_ce={row['vanilla_ce']:.3f} ircot_ce={row['ircot_ce']:.3f} "
              f"RBG={row['rbg']:.3f}  d={d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]")
    json.dump(rows, open(os.path.join(RES, "domain_ce_summary.json"), "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
