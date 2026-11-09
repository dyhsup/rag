# Results index and protocol policy

## Final mainline

Use `final_manifest.json` as the protocol index. The main claims are:

1. query-slot causal intervention: whole question versus bound requirement under fixed plan/reader/reranker/budget;
2. binding restores candidate reachability on later hops;
3. coverage decomposes empirically into candidate reachability and selection efficiency;
4. the method is effective mainly when the binding gap is large and the evidence budget covers all requirements;
5. matching-side and binding-quality-control alternatives have documented negative or small results.

## Legacy / exploratory

The following reports predate the final protocol or represent single configurations and must not be mixed into a single comparison table without their full settings:

- `e4_report.md`
- `e4_report_musique.md`
- `e4_report_hotpot.md`
- `e24_role_ot_smoke.json`
- `e24_role_ot_v2.json`
- `e26_fact_smoke_v4.json`
- `e26_fact_smoke_v5.json`

When citing an endpoint number, record dataset, split, candidate pool, retriever, CE checkpoint, query slot, plan/binding source, budget, reader and stop policy.
