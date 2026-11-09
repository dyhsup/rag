# Unified method comparison

> Rows are grouped by dataset but not pooled across protocols. Domain-CE CovBind and zero-shot baseline rows are explicitly labeled.

| dataset | method | n | EM | F1 | protocol |
|---|---|---:|---:|---:|---|
| MuSiQue-200 | Vanilla | 200 | 0.175 | 0.281 | zero-shot CE/BM25/standard reader |
| MuSiQue-200 | IRCoT | 200 | 0.330 | 0.434 | zero-shot CE/BM25/IRCoT reader |
| MuSiQue-200 | Self-Ask | 200 | 0.075 | 0.163 | zero-shot CE/BM25/Self-Ask |
| MuSiQue-200 | HyDE | 200 | 0.205 | 0.313 | zero-shot CE/BM25/HyDE |
| MuSiQue-200 | CovBind | 200 | 0.440 | 0.574 | domain CE/bound query/CoT reader |
| MuSiQue-200 | CovBind-whole-query | 200 | 0.170 | 0.286 | domain CE/whole query/CoT reader |
| MuSiQue-400 | IRCoT | 400 | 0.255 | 0.358 | zero-shot CE/BM25/IRCoT reader |
| MuSiQue-400 | CovBind | 400 | 0.425 | 0.559 | domain CE/bound query/CoT reader |
| MuSiQue-400 | CovBind-whole-query | 400 | 0.170 | 0.275 | domain CE/whole query/CoT reader |
| 2Wiki-200 | IRCoT | 200 | 0.535 | 0.626 | zero-shot CE/BM25/IRCoT reader |
| 2Wiki-200 | CovBind | 200 | 0.620 | 0.718 | domain CE/bound query/CoT reader |
| Hotpot-200 | Vanilla | 200 | 0.540 | 0.643 | zero-shot CE/BM25/standard reader |
| Hotpot-200 | IRCoT | 200 | 0.660 | 0.779 | zero-shot CE/BM25/IRCoT reader |
| Hotpot-200 | CovBind | 200 | 0.680 | 0.802 | domain CE/bound query/CoT reader |
| MHR-200 | IRCoT | 200 | 0.465 | 0.538 | zero-shot CE/BM25/IRCoT reader |
