# E3 automatic binding: error propagation (MuSiQue)

- questions: 200

## retrieval recall@50 by hop and binding mode

| hop | none | decomp only | oracle binding | predicted binding |
|---|---|---|---|---|
| 0 | 0.795 | 0.945 | 0.945 | 0.945 |
| 1 | 0.425 | 0.555 | 0.940 | 0.790 |
| 2 | 0.554 | 0.604 | 0.891 | 0.723 |
| 3 | 0.256 | 0.333 | 0.897 | 0.359 |

## hop answer EM (reader quality + binding)

| hop | decomp only | oracle binding | predicted binding |
|---|---|---|---|
| 0 | 0.520 | 0.520 | 0.520 |
| 1 | 0.125 | 0.385 | 0.300 |
| 2 | 0.158 | 0.386 | 0.198 |
| 3 | 0.026 | 0.333 | 0.026 |

## end-to-end (final answer vs gold)

| mode | EM | F1 |
|---|---|---|
| none | 0.050 | 0.134 |
| decomp | 0.105 | 0.172 |
| oracle | 0.365 | 0.489 |
| pred | 0.195 | 0.267 |

