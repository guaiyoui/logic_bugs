# Pairwise Join-Count Constraints for Pessimistic Cardinality Estimation

> Status: working prototype. Clean-room reimplementation of LpBound (SIGMOD'25)
> plus a new statistic family (exact pairwise join counts). All code under
> `boundlab/lpbench/` — independent of `~/work/join_bound/`.

---

## 1. Problem

Given a conjunctive query

```
Q(V) = R_1(A_1) ⋈ R_2(A_2) ⋈ ⋯ ⋈ R_m(A_m)
```

and statistics precomputed on the base relations, compute an **upper bound**
`B ≥ |Q|` that is *guaranteed* (pessimistic cardinality estimation, PCE).
The bound must hold for every database instance consistent with the
statistics — not just the observed one — and should be as tight as possible.

Why it matters: a guaranteed bound eliminates catastrophic underestimates
in query optimization (the dominant source of bad plans per Leis et al.),
and a `[lower, upper]` interval doubles as a **soundness oracle** for any
cardinality estimator.

---

## 2. Related work and *how each result is proved*

The proof techniques matter because they tell us which parts are
mechanizable — i.e., amenable to LLM-driven auto-research.

### 2.1 AGM bound — entropy counting (Shearer's lemma)

**Bound**: `|Q| ≤ ∏_j |R_j|^{w_j}` where `w` is a fractional edge cover.

**Proof technique**. Take the uniform distribution over the join output
`Q`; its joint entropy is `h(V) = log|Q|`. For each atom,
`h(A_j) ≤ log|R_j|` (support size bound). Minimize `Σ w_j log|R_j|` subject
to covering all attributes — the LP dual of maximizing `h(V)` under only
support constraints. Every subsequent work keeps this skeleton:
*uniform-over-output distribution → its entropy vector `h` is a feasible
point of a polymatroid LP → `max h(V)` upper-bounds `log|Q|`*.
Soundness of each new statistic reduces to showing the true entropy vector
satisfies the new constraint.

### 2.2 Degree constraints / PANDA — disjunctive combinatorics

**Bound**: adds `deg(X_i | X_U) ≤ D_{i,U}` (max-degree / functional
dependencies). The polymatroid LP with these constraints has an integrality
gap for cyclic queries; PANDA closes it by algorithmically rewriting the
query into a disjunction over simpler queries (**proof sequence**).

**Proof technique**: case-analysis trees. Each disjunctive step is
machine-checkable, which is why proof sequences can be *generated* by a
solver and *verified* by a checker — the first hint that this field is
auto-research friendly.

### 2.3 Degree Sequence Bound (DSB, ICDT'23) — the full histogram

**Statistics**: the entire degree sequence `deg_R(X|U) = (d_1 ≥ d_2 ≥ ⋯)`
(sorted multiplicities), plus max tuple multiplicity.

**Proof technique**: for Berge-acyclic queries the bound is *provably tight*
via a chain of Hölder-style shift arguments along the join tree — the
degree sequence determines the worst case exactly. SafeBound (SIGMOD'23)
makes it practical by bounding *cumulative* degree sequences with a few
segments (compression preserves validity: over-approximating the sequence
over-approximates the bound).

### 2.4 ℓp-norm bounds / LpBound (PODS'24 → SIGMOD'25) — the LP formulation

**Key lemma** (the `q`-inequality): for *any* distribution `h` supported on
relation `R`, and any `U, V` disjoint subsets of `attrs(R)`,

```
(1/p)·h(U) + h(V | U)  ≤  log₂ ‖deg_R(V|U)‖_p
```

**Proof of the lemma**: one application of Hölder's inequality.
`h(V|U) = Σ_u p_u · H(V|U=u) ≤ Σ_u p_u log deg(u)`; combining with
`(1/p)h(U)` and Hölder `Σ_u p_u^{1} deg(u) ≤ ‖p‖_∞-style…` — formally,
`2^{(1/p)h(U) + h(V|U)} ≤ ‖deg‖_p` follows from
`‖deg‖_p^p = Σ_u deg(u)^p` dominating the entropy expression. The point:
**each new statistic yields a linear inequality in `h`, proved once by
Hölder, then reused forever inside the LP.**

**System**: `max h(V)` s.t. all such inequalities for
`p ∈ {1,…,10,∞}` (full and simple degree sequences), all atoms, all `U ⊆ join attrs`.
`p=1` recovers cardinality, `p=∞` max-degree, `p=q=2` Cauchy–Schwarz.
The paper's theorem: LP optimum `2^{max h(V)} ≥ |Q|` for acyclic and cyclic
queries, predicates handled by conditioning the degree sequences.

### 2.5 CorrBound (PACMMOD'26) — correlation statistics

**Statistics**: generalized inner products `⟨a,b⟩ = Σ_k a_k b_k` between
*paired* degree sequences (cross-relation and within-relation).

**Proof technique**: a new family of information inequalities (they show
it strictly contains the ℓp family); storage handled by sketching +
low-rank factorization of the "cardinality tensor". This is the closest
existing work to ours — see §5 for the difference.

### 2.6 xBound (2026) — lower bounds

Reverses the inequality direction: `ℓ_∞/ℓ_2/ℓ_{-∞}` "reverse inner-product"
inequalities give `|Q| ≥ …`. Pairs with LpBound to form `[lb, ub]`
intervals — already demoed as an estimator soundness checker.

### 2.7 Partition constraints (ICDT'25)

Split a relation into subrelations with tighter per-partition degree
constraints; refines bounds *and* worst-case-optimal join algorithms.
Orthogonal to ours (they refine within-relation; we add cross-relation).

---

## 3. Motivation

### 3.1 The blind spot: norms are permutation-blind

Every statistic LpBound uses — `|R|`, `‖deg‖_p`, `max deg` — is a function
of the **multiset** of degrees. It cannot see *which keys are hot in both
relations simultaneously*. But the join size is an inner product:

```
|R ⋈_X S| = Σ_x deg_R(x) · deg_S(x)
```

which depends entirely on the **alignment** of the two sequences:

| alignment | same `a`, `b` multisets | `Σ_x a_x b_x` |
|---|---|---|
| ranks aligned | `a=[8,4,2], b=[9,3,1]` | `8·9+4·3+2·1 = 86` |
| ranks reversed | `a=[8,4,2], b=[1,3,9]` | `8·1+4·3+2·9 = 38` |
| disjoint supports | `a=[8,4,2], b=[0,0,0]` | `0` |

All three instances produce **identical LpBound statistics** — every ℓp norm
of every degree sequence is the same — yet join sizes differ by orders of
magnitude (infinitely, in the disjoint case). No norm-only bound can
separate them; ours does by construction.

### 3.2 The cheapest bivariate marginal

LpBound has *all univariate marginals* of the degree distributions.
The natural next rung on the ladder is the *bivariate* marginal — but the
full co-occurrence table is `O(|dom|)` storage. We take the cheapest
sufficient compression of it:

```
pair(R_i, R_j) := |R_i ⋈_{shared attrs} R_j|     — one scalar per atom pair
```

This is exactly the inner product `Σ_x a_x b_x`, i.e., the single number
the norms fail to capture. Storage is `O(#connected pairs)`; computation is
one hash-join or two `value_counts`.

### 3.3 Why this is an auto-research-shaped contribution

The whole pipeline — "propose a statistic → derive its linear constraint
in `h` → prove soundness by a support argument → drop it into the LP →
measure tightness on synthetic + real instances" — is mechanical except the
*proposal* step. This is precisely the propose/verify/score loop an LLM
agent can run (cf. Argus for bug oracles). We simulate the discovered
statistic here by hand; the loop is the deliverable.

---

## 4. Our method

### 4.1 Constraint

For every pair of atoms `i < j` sharing ≥ 1 attribute, add

```
h(A_i ∪ A_j) ≤ log₂ |R_i ⋈_{A_i ∩ A_j} R_j|
```

### 4.2 Soundness (one paragraph)

Let `h` be the entropy vector of the uniform distribution over the join
output `Q`. The marginal support of `(A_i ∪ A_j)` under `Q` is contained in
`π_{A_i∪A_j}(Q) ⊆ π_{A_i∪A_j}(R_i ⋈ R_j)`, so
`h(A_i∪A_j) ≤ log |π_{A_i∪A_j}(R_i ⋈ R_j)| ≤ log |R_i ⋈ R_j|`. Hence the
true entropy vector satisfies the constraint, so the LP optimum with the
constraint added is still `≥ log|Q|` — and never larger than before
(adding constraints only shrinks the feasible polytope). ∎

The same argument holds for *any* distribution supported on the join, which
is why the group-by objective `max h(V_0)` stays valid.

### 4.3 Baseline fidelity

Our `lpbound` arm reimplements the paper's LP:

- variables `h(U)` for all `U ⊆ attrs` (`2^n` variables),
- elementary Shannon inequalities (monotonicity + submodularity),
- h(∅) = 0, h ≥ 0,
- statistics `(1/p)h(U) + h(V|U) ≤ log₂‖deg_R(V|U)‖_p` for
  `p ∈ {1..10,∞}`, both *full* (`deg(*|U)`, fiber tuple count) and
  *simple* (`deg(X|U)`, distinct-value count) degree sequences,
  enumerated over all `U ⊆ atom attrs`,
- predicates applied by filtering before computing degree sequences —
  *stronger* than the paper's MCV/histogram approximation, i.e., we give
  the baseline an unfair advantage and still beat it.

Solver: `scipy.optimize.linprog` (HiGHS). Truth: exact counts via
weight-propagating merge join (never materializes the full join) or
closed-form degree products where applicable.

---

## 5. Results

### 5.1 Synthetic (zipf degrees, 3000-key domain)

Query `J2 = R(X,Y) ⋈_X S(X,Z)` unless noted. `q-err = bound/truth` (≥1).

| case | truth | LpBound | ours |
|---|---|---|---|
| sym (aligned ranks) | 5,963,246 | 1.00 | 1.00 |
| **anti_rank** (same domain, ranks reversed) | 23,228 | **256.73** | **1.00** |
| **anti_dom** (disjoint supports) | 0 | 5,963,246 | **1** |
| asym (n keys vs 40 hot keys) | 27,361,221 | 1.04 | **1.00** |
| unif | 76,917 | 1.27 | **1.00** |
| sym + range predicate | 1,752,639 | 1.00 | 1.00 |
| star3 / sym (3 atoms share X) | 9.2e9 | 1.00 | 1.00 |
| **star3 / anti_rank** | 5,973,360 | **1542.72** | **7.78** |
| j3 chain / correlated | 2,856,962 | 6.49 | 3.86 |
| **j3 chain / z-anticorr** | 28,577 | **630.18** | **206.88** |

On any 2-atom query the pair constraint lands on the *full* attribute set
`A_i ∪ A_j = V`, so our bound is **exactly the truth** — every J2 row is
`q-err = 1.00`. On star3-anti the three pair counts bound each
`h(A_i∪A_j) ≤ log Σ_x a_x²`, collapsing 1543× → 7.8×. The residual gap on
`j3-anticorr` (207×) is transitive correlation the pairs cannot see:
`R ⋈ S` is bounded, but which `Y`s pair with which `Z`s inside `S` is a
*within-relation* property — motivating conditional/triple statistics (§7).

### 5.2 cit-Patents, 3M edges, directed (in/out degree sequences differ)

| motif | truth | LpBound | ours |
|---|---|---|---|
| **p2** `E(X,Y)⋈E(Y,Z)` | 3,892,651 | **4.81** | **1.00** |
| tri (directed; graph is ~DAG → truth 0) | 0 | 14,534,776 | **3,892,651** |
| path3 | 3,767,959 | 58.95 | **37.84** |
| claw3 | 230,066,856 | 1.29 | 1.29 |

### 5.3 Same data, undirected — the negative control

| motif | LpBound | ours |
|---|---|---|
| p2 | 1.00 | 1.00 |
| tri | 23.53 | 23.53 |
| claw3 | 1.26 | 1.26 |
| path3 | 1.69 | 1.69 |

**Why the tie is expected and important**: after symmetrization all atoms
share one degree sequence `deg`, so `|R_i ⋈ R_j| = Σ deg²` — which is
*exactly* what the `p=q=2` ℓp constraint already yields (Hölder is tight
when the two sequences are identical). The pairs add nothing when the
marginals already determine the inner product. This confirms the mechanism:
**pair statistics pay off iff cross-relation alignment is not inferable
from univariate marginals.**

### 5.4 Cost

- Storage: one scalar per connected atom pair (`O(m²)` worst case,
  `O(m)` for chains; we use all pairs — selection is future work).
- LP overhead: +0.2–0.5 s per query at 3M edges (statistic computation,
  dominated by `np.unique` on base columns; LP itself is ms).
- No bound ever regresses (constraints are monotone).

---

## 6. Relation to CorrBound — why this isn't redundant

CorrBound's generalized inner products `⟨a,b⟩` are also pairwise
statistics. Differences:

1. **What's stored**: CorrBound stores inner products of *degree sequences
   viewed as vectors over the value domain* — which requires the two
   relations' domains to be aligned, and in practice must be approximated
   via sketches/low-rank factors. Ours stores a single scalar — the *exact*
   pair join count — which is both cheaper and exact (no sketch error).
2. **Constraint shape**: their inner products enter new information
   inequalities; ours is a direct support bound on `h(A_i∪A_j)`, the
   strongest possible constraint obtainable from one scalar (it is the
   tightest cardinality statement about that attribute subset).
3. **Composability**: pair counts compose directly with the existing LP —
   no new inequality family needed, so soundness is a two-line argument,
   and the statistic trivially extends to *conditional* pair counts
   (`|R_i ⋈ R_j|` under predicates, or partitioned by a third relation's
   keys — a strict generalization of partition constraints across
   relations).

A fair reading: **CorrBound shows pairwise statistics are the right
direction; we show the simplest member of that family — exact pair
counts — is already powerful, exact, and O(1).** The full pairwise-marginal
spectrum (per-key joint histograms, conditional counts) is the design
space between us.

---

## 7. What's next

1. **Selection**: which pairs to materialize under a storage budget —
   a knapsack/submodular optimization over LP dual values; ideal LLM-agent
   task (propose selection rules, measure on workload).
2. **Conditional / partitioned pair counts**: `|R_i ⋈ R_j|` grouped by
   attributes of a *third* relation — should capture the transitive
   correlation that leaves the 207× gap on `j3-anticorr`.
3. **Lower bounds from pairs**: `|Q| ≥ |R_i ⋈ R_j|` is trivial, but
   inclusion-exclusion over pairs (`|A⋈B⋈C| ≥ |A⋈B| + |B⋈C| − |B|`) gives
   nontrivial xBound-style lower bounds for chains — worth formalizing.
4. **Proof certificates**: extract the LP dual solution as a PANDA-style
   proof sequence → machine-checkable proof of each reported bound.
5. **Real workload**: JOB/STATS via Postgres-extracted degree sequences;
   then end-to-end plan quality (the paper's metric: plans at least as
   good as with true cardinalities).
6. **LLM loop**: candidate-statistic generation → soundness check via the
   support argument + adversarial counterexample search → LP integration —
   the auto-research harness this prototype is a manual run of.

---

## 8. Files

```
boundlab/lpbench/
  engine.py   — LP construction, all statistics, solver, exact truth
  data.py     — synthetic generators + cit-Patents loader
  run.py      — benchmark driver (synth / j3 / graph)
  REPORT.md   — this file
```

Reproduce:

```bash
cd boundlab/lpbench
python run.py synth   # synthetic table
python run.py j3      # 3-chain
python run.py graph   # cit-Patents, directed + undirected
```
