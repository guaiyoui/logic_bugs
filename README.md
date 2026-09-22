# logic_bugs

Research playground on two tightly related questions:

1. **DBMS logic-bug oracles** (`absoracle/`) — metamorphic/consistency
   oracles that catch wrong-answer bugs in query engines.
2. **Guaranteed join-size bounds** (`boundlab/lpbench/`) — pessimistic
   cardinality estimation as an LP over entropy variables, extended with
   new statistic families. The bound side doubles as a *soundness oracle*:
   any optimizer estimate outside `[LB, UB]` is provably wrong.

## boundlab/lpbench — LpBound reimplementation + pairwise join counts

Clean-room reimplementation of the LpBound linear program
(SIGMOD 2025 best paper): `2^n` entropy variables `h(U)`, elementary
Shannon inequalities, and statistics constraints

```
(1/p)·h(U) + h(V|U) <= log2 ||deg_R(V|U)||_p ,   p in {1..10, inf}
```

for full and simple degree sequences, plus selection predicates.

**New statistic** (the experiment): exact pairwise join counts

```
h(A_i ∪ A_j) <= log2 |R_i ⋈ R_j|
```

— one scalar per connected atom pair. Norms are permutation-blind; this
captures cross-relation degree *alignment* they cannot see.

Headline results (see `boundlab/lpbench/REPORT.md` for the full write-up):

| case | truth | LpBound q-err | +pairs q-err |
|---|---|---|---|
| J2, anti-correlated ranks | 23,228 | 256.7 | **1.00** |
| J2, disjoint supports | 0 | 5.9M | **exact** |
| star3, anti-correlated | 5.97M | 1542.7 | **7.8** |
| cit-Patents p2 (directed) | 3.89M | 4.81 | **1.00** |
| cit-Patents tri (directed) | 0 | 14.5M | **3.9M** |

Also implemented:

- **k-hop marginal ladder**: `triples=True` adds exact 3-atom join counts
  (synthetic 4-chain: 2997x -> 1065x (+pairs) -> 181x (+triples)).
- **Provable lower bounds** (`lower_bound`): spine/star inclusion-exclusion
  and triangle degree-sum bounds -> two-sided `[LB, UB]` intervals. On
  `j3/anticorr` the interval is [23.1K, 5.9M] around truth 28.6K.
- **LP-dual proof certificates** (`certify`): extracts the active
  inequalities + dual multipliers, reconstructing the bound as a
  PANDA-style proof sequence (verified by strong duality).

Reproduce:

```bash
cd boundlab/lpbench
python run.py synth   # synthetic relational
python run.py j3      # 3-chain
python run.py j4      # 4-chain + triples + certificate demo
python run.py graph   # cit-Patents directed/undirected motifs
```

## absoracle — DBMS logic-bug oracles

- `data_homo` — data-transformation homomorphisms (result must transform
  accordingly)
- `eqcoh` — cross-module equivalence coherence
- `dml_image` — DML as image transformation
- `env_sweep` — environment/flag sweep consistency

Campaign runner: `./run_campaign.sh [env|dml|homo]`. Logs of assert-build
findings in `logs/`, per-run output in `results/`.
