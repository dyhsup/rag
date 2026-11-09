# Final mainline

## Claim

For multi-hop RAG, the dominant query–evidence failure in the studied setting is often an unbound query slot: later-hop requirements lack the intermediate entity introduced by earlier hops. Binding that variable restores candidate reachability. The remaining loss is empirically decomposed into candidate reachability and selection efficiency.

## Method

CovBind/RBG-Exec executes a typed requirement program:

1. compile the question into dependent requirements;
2. bind previous answers into later requirements;
3. retrieve and rerank with the instantiated requirement as the query slot;
4. select evidence per requirement and stop on verifiable coverage or budget exhaustion;
5. use a typed decision layer for final-hop versus synthesis cases.

The contribution is explicit execution semantics and causal diagnosis, not a new matcher architecture.

## Frozen primary evidence

- Query-slot intervention: MuSiQue 400, whole-question EM .170 versus bound-requirement EM .425; paired Δ=.255, bootstrap CI [.210,.303].
- Hop recall: unbound dense .915/.615/.673/.538; bound dense .970/.935/.921/.923.
- Candidate reachability: BM25 .610; MiniLM .760; BGE-small .765; HyDE+MiniLM .770.
- Pointwise selection efficiency remains .47–.56; coverage selection approaches the candidate ceiling.
- Endpoint under domain CE: MuSiQue 200 CovBind .440 vs IRCoT .315; MuSiQue 400 .425 vs .302; 2Wiki .620 vs .615; Hotpot .680 vs .665; MHR .530 vs .465.
- Applicability requires a large binding gap B and sufficient evidence budget k≥m.

## Boundaries

- Effects are strongest on MuSiQue and are not universal across datasets.
- Plan and binding errors propagate; the six tested surface quality-control mechanisms did not solve this.
- The reachability × selection-efficiency decomposition is empirical, not a parameterized theorem.
- Legacy single-configuration reports must not be combined with final endpoint tables without their protocol fields.
- E24/E26 structured matcher experiments are exploratory engineering negatives, not part of the main algorithm claim.
