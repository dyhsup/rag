# E24 Role-OT-style structured grounding screening

## Protocol and scope

- Dataset: MuSiQue validation, first 100 questions
- Plan/binding: **gold plan and gold binding**; this is an oracle-binding screening, not an end-to-end predicted-binding result
- Candidate pool: the 20 passages attached to each question
- Budget: k=2
- Baseline: requirement-conditioned CE `dist/reqce/model_hop`
- Reader: off for retrieval screening
- Structural score: title+passage entity overlap, relation-family overlap, local sentence support, and answer-type compatibility; it does not read final answers or gold labels
- Role-OT is only a name for this structured-grounding implementation family; this prototype is a deterministic heuristic and does not implement a learned OT optimizer.

## Result

| method | coverage@2 | full-evidence rate |
|---|---:|---:|
| requirement CE (`seq_ce`) | 0.7308 | 0.4200 |
| structural grounding (`role_ot`) | 0.5025 | 0.2200 |
| structural coverage selector | 0.4733 | 0.1600 |
| CE + structural score | **0.7358** | 0.4300 |

Paired bootstrap over the 100 questions, relative to `seq_ce`:

| method | coverage delta | 95% bootstrap interval |
|---|---:|---:|
| structural grounding | -0.2283 | [-0.2792, -0.1783] |
| structural coverage selector | -0.2575 | [-0.3125, -0.2058] |
| CE + structure | +0.0050 | [0.0000, +0.0150] |

## Decision

The prototype does not pass the pre-registered threshold of `+3pp coverage`, and the hybrid gains only 0.5pp. Stop expanding this heuristic version.

The negative result is narrower than “structured grounding fails”: it shows that a hand-written relation vocabulary plus surface entity overlap is substantially weaker than the trained requirement CE. It also exposes protocol requirements for the next audit: use the same title/passage input for baseline and matcher, report 2-hop with k=m separately, record source/checkpoint hashes, and use predicted bindings for any end-to-end claim.

## Next action

At most one independent fairness audit is justified: train a genuinely structure-aware matcher or fact extractor on `musique1000[400:1000]`, tune only on `[200:300]`, and evaluate on disjoint `musique200[:200]` or `musique400[:400]`. Add same-question cross-hop, same-entity wrong-relation, and same-relation wrong-entity hard negatives. Kill immediately if 2-hop/k=2 coverage is below `+3pp` or paired CI crosses zero, or EM is below `+2pp`. If this audit also fails, keep the main paper centered on query-slot causal localization and reachability/selection-efficiency diagnostics.
