# E26 Structured fact grounding smoke test

## Protocol

- MuSiQue validation, first 20 questions
- gold plan + gold binding
- per-question 20-passage pool
- budget `k=m` (each question receives one selected passage per hop)
- baseline: existing requirement CE `model_hop`
- structure parser: deterministic explicit relation templates; LLM parser was unavailable during the run and the code used the documented offline fallback
- no gold answer, `is_gold`, or `para_idx` is used for selection

## Result

| method | hop coverage | segment coverage | full-hop coverage |
|---|---:|---:|---:|
| structured fact grounding | 0.2292 | 0.2292 | 0.0000 |
| requirement CE (remote/default run) | 0.8083 | 0.8083 | 0.6500 |
| requirement CE (local `dist/reqce/model_hop`) | **0.8375** | **0.8375** | **0.7000** |

The local checkpoint is the fair baseline for the latest run. Average extracted facts per question: 13.0.

## Interpretation

This smoke test is a parser/engineering baseline, not a method conclusion. The structured selector is far below the trained requirement CE. The run also shows that a naïve relation-template extractor over raw Wikipedia passages creates many spurious `played` facts from generic “as” constructions. It should not be expanded or presented as a successful structured-grounding algorithm.

The next serious experiment, if pursued, needs a separately evaluated fact extractor or trained structure-aware matcher, with held-out hard negatives and a parser-availability log. The innovation claim remains the requirement-variable/role/direction grounding objective; this run only tests a weak implementation.
