# H3/Qwen3.8 八端点端到端扩展实验

## 服务

- Host: `10.206.122.203`
- Ports: `8000–8007`
- Model: `/root/paddlejob/workspace/env_run/output/models/Qwen3.8-27B`
- Client: `src/llm.py` round-robin across all eight endpoints
- Separate cache: `cache/llm_h3.sqlite`

## 50-question exploratory results

| dataset | rewrite EM/F1 | HyDE EM/F1 | decomposition EM/F1 |
|---|---:|---:|---:|
| MuSiQue | .160/.289 | .300/.435 | .160/.265 |
| 2Wiki | .420/.495 | .360/.412 | .460/.517 |
| HotpotQA | .580/.700 | .640/.755 | .620/.722 |
| MultiHopRAG | .580/.630 | .560/.610 | .540/.618 |

These are 50-question exploratory runs, not replacements for the 200/400-question main tables. The same H3/Qwen3.8 model is used for generation and reading, and all rows use the same per-question BM25 pool with the same reader template.

## Artifacts

- `results/h3_e2e_*_summary.json`
- `results/h3_e2e_*_fix_summary.json`
- per-question JSONL files with generated query/pseudo-passage/decomposition metadata
- `src/extended_e2e_methods.py`

## Note

The first decomposition run had a prompt-brace formatting bug (`KeyError("queries")`); it was fixed by escaping JSON braces and rerun under `*_fix`. Only fixed decomposition results are valid.
