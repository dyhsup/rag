# Cross-dataset protocol audit

| dataset | hop labels | slot counterfactual | reachability curve | reader/EM | status |
|---|---|---|---|---|---|
| MuSiQue | full per-hop support | full 200/400 | BM25/MiniLM/BGE/HyDE | full | primary causal dataset |
| 2Wiki | reconstructed evidence hops | 50-question gold-binding/top20 | 200-question whole-query BM25 | domain-CE endpoint | mechanism replication + reachability |
| HotpotQA | no canonical hop decomposition in cache | not run as hop counterfactual | 200-question whole-query BM25 | domain-CE endpoint | reachability/control only |
| MultiHopRAG | flat evidence list in cache | not run as hop counterfactual | 200-question whole-query BM25 | news-domain endpoint | reachability/control only |

## Interpretation policy

Only MuSiQue supports the full 6.1–6.5 suite with native hop labels. 2Wiki supports a reconstructed gold-binding mechanism test. HotpotQA and MultiHopRAG support passage-level reachability and endpoint/reader controls, not a native hop-level causal claim. These distinctions are included to avoid presenting non-equivalent protocols as one pooled result.
