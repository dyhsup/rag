"""G0: does the binding advantage grow with hop depth?"""
import json
import os

from . import eval as ev
from .musique import load

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


def read(p):
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def main():
    qs = {q["qid"]: q for q in load(os.path.join(ROOT, "data", "raw", "musique200.jsonl"))}
    files = {"vanilla": "e4_musique_vanilla.jsonl", "ircot": "e4_musique_ircot.jsonl",
             "selfask": "e4_musique_selfask.jsonl", "ours_auto": "e4_musique_ours_auto.jsonl",
             "ours_gold": "e4_musique_ours_gold.jsonl", "ours_bm25": "e4_musique_ours_bm25_auto.jsonl"}
    methods = {m: read(os.path.join(RES, f)) for m, f in files.items()}
    out = {"musique": {}, "hotpot": {}}
    print("=== MuSiQue by hop count ===")
    print("| hops | n | vanilla | IRCoT | ours | ours-IRCoT |")
    for h in [2, 3, 4]:
        ids = [k for k in methods["ours_auto"] if k in qs and qs[k]["n_hops"] == h]
        if not ids:
            continue
        em = {m: [methods[m][k]["em"] for k in ids if k in methods[m]] for m in methods}
        ids2 = [k for k in ids if all(k in methods[m] for m in ["ours_auto", "ircot"])]
        d = ev.paired_bootstrap([methods["ours_auto"][k]["em"] for k in ids2],
                                [methods["ircot"][k]["em"] for k in ids2])
        out["musique"][h] = {"n": len(ids), "em": {m: ev.mean(v) for m, v in em.items()},
                             "delta_ours_ircot": d}
        print(f"| {h} | {len(ids)} | {ev.mean(em['vanilla']):.3f} | {ev.mean(em['ircot']):.3f} | "
              f"{ev.mean(em['ours_auto']):.3f} | {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] |")

    hp = {q["qid"]: q for q in (json.loads(l) for l in open(os.path.join(ROOT, "data", "raw", "hotpot200.jsonl"), encoding="utf-8"))}
    hm = {m: read(os.path.join(RES, f"e4_hotpot_{m}.jsonl")) for m in ["vanilla", "ircot", "ours_auto"]}
    print("\n=== HotpotQA by level / type ===")
    print("| slice | n | vanilla | IRCoT | ours | ours-IRCoT |")
    for field in ["level", "type"]:
        vals = sorted({hp[k][field] for k in hm["ours_auto"] if k in hp})
        for v in vals:
            ids = [k for k in hm["ours_auto"] if k in hp and hp[k][field] == v and k in hm["ircot"]]
            if len(ids) < 10:
                continue
            em = {m: ev.mean([hm[m][k]["em"] for k in ids]) for m in hm}
            d = ev.paired_bootstrap([hm["ours_auto"][k]["em"] for k in ids],
                                    [hm["ircot"][k]["em"] for k in ids])
            out["hotpot"][f"{field}:{v}"] = {"n": len(ids), "em": em, "delta": d}
            print(f"| {field}={v} | {len(ids)} | {em['vanilla']:.3f} | {em['ircot']:.3f} | "
                  f"{em['ours_auto']:.3f} | {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] |")
    json.dump(out, open(os.path.join(RES, "g0_subgroup.json"), "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
