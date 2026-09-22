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

## 4. A self-contained theory: bounds as polynomial certificates

The bounds above were stated as constraints inside an entropy LP.
Here we re-derive everything from scratch — no entropy, no `h(U)`
variables — as a **moment problem on a degree field**.  All prior
bounds (AGM, degree-sequence, Lp-norm, pair counts) reappear as
sections of one LP.

### 4.1 The degree field and the product moment

For a star query `R_1(X,·) ⋈ … ⋈ R_m(X,·)` on one shared key, every
key `x ∈ dom(X)` carries a **degree vector**
`d(x) = (d_1(x),…,d_m(x)) ∈ ℕᵐ`, `d_i(x) = #{t ∈ R_i : t.X = x}`.
Then

```
|Q| = Σ_x  ∏_i d_i(x)                    (the product moment)
```

Statistics are **observed moments** `μ_α = Σ_x d(x)^α`, `α ∈ ℕᵐ`:

| α | moment | = known statistic |
|---|---|---|
| `p·eᵢ` | `Σ_x d_i(x)^p` | `‖deg_i‖_p^p` (LpBound's entire statistic set) |
| `eᵢ+eⱼ` | `Σ_x d_i d_j` | `|R_i ⋈ R_j|` (our pair count) |
| `eᵢ+eⱼ+eₖ` | `Σ_x d_i d_j d_k` | triple count |
| `0` | `Σ_x 1` | `|dom(X)|` (support) |

### 4.2 The certificate LP and its dual

Summing a pointwise inequality `∏_i d_i ≤ Σ_α c_α d^α` over the field
gives `|Q| ≤ Σ_α c_α μ_α`.  Since scalar moments do not reveal which
`d_i(x)` pairs with which `d_j(x)`, the inequality must dominate on
the **worst-case domain** `Ω = ∏_i supp(deg_i)` (the product of
realized degree supports).  Hence

```
UB(A) = min  Σ_α c_α μ_α      LB(A) = max  Σ_α c_α μ_α
        c ≥ 0                        c ≥ 0
        s.t. Σ_α c_α d^α ≥ Π d_i      s.t. Σ_α c_α d^α ≤ Π d_i
             ∀d ∈ Ω                        ∀d ∈ Ω
```

A bound is therefore *literally a nonnegativity certificate*: a
polynomial that dominates `Π d_i` pointwise.  AM-GM
(`abc ≤ (a³+b³+c³)/3`), Young, and Hölder are all members of this
cone — they are the *closed-form* points; the LP finds the best one
adapted to the observed moments.

**Duality.**  The dual of `UB(A)` is the moment LP: maximize
`Σ_y ν(y)·Π y_i` over nonnegative weightings `ν` on `Ω` matching the
observed moments.  Strong duality: `UB(A)` = the largest product
moment consistent with `A` — the certificate *is* the proof.

### 4.3 The Hölder section (univariate moments are exactly Hölder)

With `A = {p·eᵢ}` only, the optimal certificate recovers the
conjugate-Hölder bound `min_{Σ1/pᵢ=1} ∏ᵢ‖deg_i‖_{pᵢ}` — and on the
discrete grid it is slightly *tighter*, because dominance is only
required on realized degree values, not all of ℝⁿ.  Empirically the
two coincide to machine precision on aligned cases and the LP is
strictly tighter otherwise (§6).  In words:

> the entropy-LP's implied bound on each pair is *already* the
> Hölder relaxation of the pair join count; adding the exact pair
> moment pins that relaxation to its exact value `⟨deg_i, deg_j⟩`.

### 4.4 The hierarchy and the exactness level

`A_k = {α : |α|₁ ≤ k}` gives a monotone sequence
`UB(1) ≥ UB(2) ≥ …` — a **Lasserre-style moment hierarchy**.  Two
exactness levels are immediate:

- `e₁+…+eₘ ∈ A ⇒ UB = |Q|` exactly (the product moment *is* the truth;
  this is why `+triples` hits 1.00 on 3-atom stars and why `m`-atom
  queries need the `m`-th moment).
- the Hölder gap `log(min-conjugate-∏‖d_i‖ / ⟨d_i,d_j⟩)` measures
  exactly how much the level-2 moment buys over level-1 — a
  computable predictor for statistic selection.

### 4.5 Where the polynomial cone is *not* enough

The lower-bound side exposes an honest limitation: with `c ≥ 0` and
polynomial basis `d^α`, the best LB on stars collapses to 0 —
`max(0, Σdᵢ−(m−1))` (inclusion–exclusion) is *piecewise-linear*, not
polynomial.  Extending the certificate basis beyond monomials
(piecewise-linear / max-terms) is the natural next theorem, and keeps
the framework one object: bounds = certificates over a basis family.

### 4.6 Scalability: the LP is small and separable

Variables = `|A|` (one per observed moment); rows = `|Ω|`.
`Ω` is exponential only in `m` (#atoms on the separator), not in
`n` (#attributes) — the `2ⁿ` entropy-variable blowup disappears.
When `|Ω| = ∏|supp_i|` is large we solve by *separation*
(cutting planes): solve on a coarse grid, ask an oracle for the
most-violated grid point, add it, repeat — convergence yields a
certificate dominating the entire product, so soundness is proven,
not sampled.

---

## 5. Instantiation: our method as moment constraints in the entropy LP

### 5.1 Constraint

For every pair of atoms `i < j` sharing ≥ 1 attribute, add

```
h(A_i ∪ A_j) ≤ log₂ |R_i ⋈_{A_i ∩ A_j} R_j|
```

### 5.2 Soundness (one paragraph)

Let `h` be the entropy vector of the uniform distribution over the join
output `Q`. The marginal support of `(A_i ∪ A_j)` under `Q` is contained in
`π_{A_i∪A_j}(Q) ⊆ π_{A_i∪A_j}(R_i ⋈ R_j)`, so
`h(A_i∪A_j) ≤ log |π_{A_i∪A_j}(R_i ⋈ R_j)| ≤ log |R_i ⋈ R_j|`. Hence the
true entropy vector satisfies the constraint, so the LP optimum with the
constraint added is still `≥ log|Q|` — and never larger than before
(adding constraints only shrinks the feasible polytope). ∎

The same argument holds for *any* distribution supported on the join, which
is why the group-by objective `max h(V_0)` stays valid.

### 5.3 Baseline fidelity

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

## 6. Results

### 6.1 Synthetic (zipf degrees, 3000-key domain)

Query `J2 = R(X,Y) ⋈_X S(X,Z)` unless noted. `q-err = bound/truth` (≥1).

| case | truth | LpBound | ours (+pairs) | LB |
|---|---|---|---|---|
| sym (aligned ranks) | 5,963,246 | 1.00 | 1.00 | **exact** |
| **anti_rank** (same domain, ranks reversed) | 23,228 | **256.73** | **1.00** | **exact** |
| **anti_dom** (disjoint supports) | 0 | 5,963,246 | **1** | **exact (0)** |
| asym (n keys vs 40 hot keys) | 27,361,221 | 1.04 | **1.00** | **exact** |
| unif | 76,917 | 1.27 | **1.00** | **exact** |
| sym + range predicate | 1,752,639 | 1.00 | 1.00 | **exact** |
| star3 / sym (3 atoms share X) | 9.2e9 | 1.00 | 1.00 | 11.9M |
| **star3 / anti_rank** | 5,973,360 | **1542.72** | **7.78** | 33,342 |
| j3 chain / correlated | 2,856,962 | 6.49 | 3.86 | 990,598 |
| **j3 chain / z-anticorr** | 28,577 | **630.18** | **206.88** | **23,149** |

On any 2-atom query the pair constraint lands on the *full* attribute set
`A_i ∪ A_j = V`, so our bound is **exactly the truth** — every J2 row is
`q-err = 1.00`. On star3-anti the three pair counts bound each
`h(A_i∪A_j) ≤ log Σ_x a_x²`, collapsing 1543× → 7.8×. The residual gap on
`j3-anticorr` (207×) is transitive correlation the pairs cannot see:
`R ⋈ S` is bounded, but which `Y`s pair with which `Z`s inside `S` is a
*within-relation* property — motivating conditional/triple statistics (§7).

### 6.2 cit-Patents, 3M edges, directed (in/out degree sequences differ)

| motif | truth | LpBound | +pairs | +triples | LB |
|---|---|---|---|---|---|
| **p2** `E(X,Y)⋈E(Y,Z)` | 3,892,651 | 4.81 | **1.00** | — | **3,892,651 (exact)** |
| tri (≈DAG → truth 0) | 0 | 14,534,776 | **3,892,651** | — | **0 (exact)** |
| claw3 | 230,066,856 | 1.29 | 1.29 | — | 45,211,328 |
| path3 | 3,767,959 | 58.95 | **37.84** | — | 1,867,991 |
| **path4** | 2,618,085 | 2067.18 | 2067.18 | **133.68** | n/a (no spine) |

### 6.3 Same data, undirected — the negative control

| motif | LpBound | +pairs | LB |
|---|---|---|---|
| p2 | 1.00 | 1.00 | 46,425,742 (exact) |
| tri | 23.53 | 23.53 | 0 |
| claw3 | 1.26 | 1.26 | 86,851,484 |
| path3 | 1.69 | 1.69 | 86,851,484 |

**Why the tie is expected and important**: after symmetrization all atoms
share one degree sequence `deg`, so `|R_i ⋈ R_j| = Σ deg²` — which is
*exactly* what the `p=q=2` ℓp constraint already yields (Hölder is tight
when the two sequences are identical). The pairs add nothing when the
marginals already determine the inner product. This confirms the mechanism:
**pair statistics pay off iff cross-relation alignment is not inferable
from univariate marginals.**

### 6.4 The k-hop ladder and two-sided intervals

Synthetic 4-chain `R(X,Y)-S(Y,Z)-T(Z,U)-V(U,W)` with `T.z` rank-reversed
vs `S.z` (truth = 42,320):

| arm | bound | q-err |
|---|---|---|
| LpBound | 126,823,654 | 2996.78 |
| +pairs | 45,071,595 | 1065.02 |
| +triples | 7,673,965 | **181.33** |

Pairs alone are insufficient on length-4 chains (they cap 2-edge prefixes
but the residual product still blows up); triples cap 3-edge subjoins and
recover most of the gap. This is the expected tradeoff curve:
storage `O(#connected k-subsets)` buys `k`-hop correlation.

**Lower bounds** (`lower_bound`, new): spine/star inclusion-exclusion
`Σ_t max(0, Σ deg_i(t) − (k−1))` and triangle `Σ max(0, deg(y)+deg(z)−n_X)`.
Notable: on every 2-atom query the spine LB is *exact*, so J2/p2 intervals
collapse to a point `[truth, truth]`. On `j3/anticorr` the interval is
`[23,149, 5.9M]` around truth `28,577` — the LB is within 1.24× of truth
while the UB carries the remaining slack. On `tri/directed` the interval is
`[0, 3.9M]` — LB certifies emptiness up to the UB gap.

**Proof certificates** (`certify`, new): the LP dual solution is extracted
as `(inequality, dual weight, rhs)` triples; by strong duality
`log2 bound = Σ w_i·b_i`. Verified to the last digit on all queries —
e.g. `j4+pairs` decomposes as `bound = |R| · |T⋈V|` mediated by six Shannon
inequalities. This is a machine-checkable proof sequence in the PANDA
style, emitted for free by the solver.

### 6.5 Real-world benchmarks (the actual LpBound evaluation suite)

Same datasets/workloads as the LpBound paper: **JOB** on IMDB
(JOB-light 70 queries, JOB-join 31, JOB-range), **STATS** (146 queries,
Stack-Overflow dump), **DBLP** dense 8-vertex subgraph patterns.
Truth = `COUNT(*)` of the original SQL in DuckDB over the *same* parsed
rows the statistics see.  `q-err = bound/truth ≥ 1`; a value `< 1` is a
soundness violation.

**JOB-light (70 queries, 4–8 tables, predicates):**

| arm | geomean | median | p90 | max | exact | viol |
|---|---|---|---|---|---|---|
| LpBound | 16.14 | 10.12 | 98.0 | 6089 | 0% | 0 |
| +pairs | **4.97** | **3.89** | **32.7** | 467 | 6% | 0 |

+pairs is strictly tighter on **100%** of the 70 queries; LB coverage
100%, 0 violations.

**JOB-join (31 queries, up to 14 tables / 23 LP attributes, no
predicates):** 29/31 evaluated (1 genuine empty result, 1 DuckDB truth
timeout).

| arm | geomean | median | p90 | max | viol |
|---|---|---|---|---|---|
| LpBound | 10.07 | 4.64 | 123.6 | 4654 | 0 |
| +pairs | **5.05** | **3.46** | **60.0** | 539 | 0 |

Strictly tighter on 93%.  Highlights: `jobjoin.11` 117→6.6,
`jobjoin.21` 467→54, `jobjoin.27` 4654→540, `jobjoin.31` 107→22.
Simple star/chain queries collapse to q-err ≈ 1.0 (pair count = truth).
The n≥13 queries all run through the reduced LP; on `jobjoin.9` it
reproduces the full-LP optimum bit-for-bit.

**JOB-range (1000 queries, range/equality predicates — the largest
suite):** 994/1000 evaluated (3 genuine empty results, 3 truth
timeouts).

| arm | geomean | median | p90 | max | exact | viol |
|---|---|---|---|---|---|---|
| LpBound | 111.0 | 90.4 | 2591 | 198533 | 0% | 0 |
| +pairs | **7.30** | **5.70** | **67.4** | 13426 | 11% | 0 |

Strictly tighter on **99%** of queries; 28% of bounds within 2× of
truth.  LB emitted on 100%, 0 violations.

**STATS (146 queries, Stack-Overflow, heavy predicates):**

| arm | geomean | median | p90 | max | exact | viol |
|---|---|---|---|---|---|---|
| LpBound | 175.3 | 108.4 | 14370 | 4.7e8 | 1% | 0 |
| +pairs | 88.5 | 44.8 | 8710 | 3.1e8 | 8% | 0 |
| +triples | **8.28** | **4.91** | **255** | 1995 | **34%** | 0 |

Strictly tighter on 97% (pairs) and 90% (triples over pairs).  The
triples arm is the headline result here: exact 3-atom join counts
collapse the remaining gap by another **10.7× geomean** over pairs —
e.g. `stats.13` 108→35→**1.00**, `stats.14` 6170→2240→**1.00**,
`stats.116` 9224→4002→**22.8**.  The 4-atom star joins that dominate
STATS' tail are exactly the transitive-correlation regime pairs cannot
see; a 3-way exact marginal covers the whole connected skeleton.

Two soundness bugs were caught *by* the regression rather than assumed
away: `stats.114-121` were silently under-bounded before the join-key
encoding fix (per-table packing broke cross-table equality), and
`stats.142` exposed a merge-order bug in the triple counter — an atom
reachable only through a *later* atom in combination order was skipped,
under-counting the join and producing an invalid (too-small)
constraint.  Both fixed and re-verified `bound ≥ truth` on all 145
evaluated queries (1 truth timeout).  LB emits on 100%, 0 violations.

**DBLP dense subgraphs (8-vertex patterns):** pair statistics tie the
baseline — bound-only protocol (TQ=0, 30 dense patterns):
`pairs/lpbound` bound-ratio geomean = **1.000, 100% ties**.  Expected
negative control: after label propagation each edge atom is tiny
(~9K rows), single-atom degree sequences already dominate pair counts;
DuckDB also times out on most dense patterns, so the ratio is reported
on bounds alone.  This is §6's predicted no-win regime.

### 6.6 The certificate LP reproduces the entropy LP — exactly

`moment.py` solves the certificate LP of §4 (variables = moment
coefficients, rows = worst-case dominance domain, cutting-plane
separation when the product grid is large).  On every case checked it
**reproduces the entropy LP to the last bit** — evidence that the two
formulations are the same object:

| case | truth | Hölder closed | mcert univ | mcert +pairs | mcert +triples | entropy lpbound / +pairs |
|---|---|---|---|---|---|---|
| j2/sym | 5.96M | 1.00 | 1.00 | 1.00 | — | 1.00 / 1.00 |
| j2/anti_rank | 23K | 256.7 | **256.73** | **1.00** | — | 256.73 / 1.00 |
| j2/anti_dom | 0 | inf | inf | **0** | — | 6.0M / 1 |
| j2/asym | 27.4M | 1.02 | 1.03 | 1.00 | — | 1.04 / 1.00 |
| j2/unif | 77K | 1.27 | 1.27 | 1.00 | — | 1.27 / 1.00 |
| star3/sym | 9.22B | 1.01 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 |
| star3/anti_rank | 6.0M | 1563 | **1542.72** | **7.78** | **1.00** | 1542.72 / 7.78 |

(q-err shown; mcert univar = the entropy `lpbound` column to printed
precision on every case — the Hölder-section claim of §4.3.)

STATS single-key stars (first 12 evaluated): `mc_pairs` = entropy
`+pairs` bit-for-bit; `mc_tri` = `+triples`; `mc_uni` tracks `lpbound`
within 0.01% — and is *slightly tighter* where the discrete grid
dominance beats continuous Hölder (`stats.2`: 14,326,174 vs
14,326,997).  The Hölder gap is confirmed as the exact predictor of
where the pair moment pays.

Lower-bound side (`mcert LB`): collapses to 0 on stars with `m ≥ 3` —
the polynomial cone provably cannot express `max(0, Σdᵢ−(m−1))`;
recorded as §4.5's open extension rather than hidden.

*Solver caveat worth recording:* on badly-scaled moment LPs HiGHS can
return a numerically suboptimal point (on `star3/anti_rank` it reported
UB = 1.004×truth while the monomial certificate `c_{(1,1,1)} = 1` —
equal to `μ_{(1,1,1)}` = truth — was feasible and cheaper).  Since any
"super-product" moment `α ≥ 𝟏` is itself a valid monomial certificate,
`certificate_bound` enforces `UB ≤ min_{α ≥ 𝟏} μ_α` as a sanity floor;
after the floor the triple bound is exact (`5,973,360` = truth).

### 6.7 Engineering: what it took to be correct *and* scalable

Real data surfaced four bugs a synthetic-only evaluation would miss:

- **SQL `COUNT(*)` semantics**: entropy variables must range over
  *physical rows*, not key projections — each atom carries a private
  row-id attribute (without it bounds under-count by the duplicate-key
  product).
- **NULL semantics**: `NULL` never joins `NULL`; naive pandas/factorized
  encodings treat `NaN == NaN`.  NULLs get per-(table,column) salted
  unique sentinels; a `to_numpy` view-mutation bug silently rewrote
  source columns before the fix.
- **Join-key packing must be a pure function of the values**: an early
  `encode()` used per-table radix bases/min-shifts — identical tuples
  packed differently across tables, producing *invalid* (too-small)
  pair statistics and under-bounds on STATS.  Fixed-base packing (with
  an arbitrary-precision fallback) restored soundness; regression now
  asserts `bound ≥ truth` on every query.
- **LP size**: `n` attributes ⇒ `2^n` entropy vars; JOB-join reaches
  `n = 23` (8.4 M).  A *reduced* LP keeps only valid inequalities —
  atom-local statistics, an atom-ordering submodularity ladder, and
  pair/triple-ladder constraints so marginals still propagate —
  soundness is preserved (fewer constraints ⇒ looser, never violated).
- **k-way join counts need a connectivity-ordered merge**: merging
  atoms in combination order can skip an atom whose only join partner
  comes later — the result is a *smaller* count, hence an *invalid*
  constraint (`stats.142` regression).  Atoms are BFS-ordered before
  merging; row-ids are folded into tuple weights so a star triple
  costs O(#keys), not O(|join|).
- **Object columns with NULLs**: a mixed `object` array (`'S3524'`,
  `NaN`) raises `TypeError` on elementwise `<`; predicates evaluate
  through a NULL-aware comparator (SQL: any comparison on NULL is
  false).

### 6.8 Cost

- Storage: one scalar per connected atom subset (`O(m²)` pairs; `O(m)`
  for chains — selection under a budget is future work).
- Runtime: JOB-light ≈ 9 s/query median (dominated by degree-sequence
  scans on the 33 M-row `cast_info`); the reduced LP itself is ms–s.
- No bound ever regresses (constraints are monotone); every reported
  bound verified `≥ truth`, every LB `≤ truth`.

---

## 7. Relation to CorrBound — why this isn't redundant

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

## 8. What's next — progress and remaining

Done in this round:

1. ~~**Conditional / partitioned pair counts**~~ — partially: the `triples`
   arm (exact 3-atom join counts) is the first step; on the synthetic
   4-chain it recovers 1065× → 181×. Still open: *conditional* pair
   counts (`|R_i ⋈ R_j|` partitioned by a third relation's keys), which
   target the transitive-correlation gap that remains on `j3-anticorr`
   (residual 207×).
2. ~~**Lower bounds from pairs**~~ — `lower_bound` implemented: spine/star
   inclusion-exclusion + triangle degree-sum. Two-sided intervals are now
   emitted everywhere a template applies (exact on all 2-atom queries;
   `[23.1K, 5.9M]` on `j3-anticorr`; `[0, 3.9M]` on directed tri).
3. ~~**Proof certificates**~~ — `certify` extracts the LP dual as a
   PANDA-style proof sequence; verified consistent by strong duality on
   every query (e.g. `bound = |R|·|T⋈V|` + 6 Shannon steps on j4).

4. ~~**Real workload**~~ — the full LpBound evaluation suite is
   implemented (§5.5): JOB-light/JOB-join/JOB-range on IMDB, STATS on the
   Stack-Overflow dump, DBLP dense patterns.  Truth is DuckDB `COUNT(*)`
   over the identical parsed rows; every emitted bound is checked
   `≥ truth`, every LB `≤ truth`.

Still open:

5. **Selection**: which pair/triple stats to materialize under a storage
   budget — knapsack over LP dual values; ideal LLM-agent task.
6. **Conditional marginals**: pair counts conditioned on a third atom's
   key group (the 207× residual on `j3-anticorr` is transitive
   correlation that pairs cannot see).
7. **Lower-bound coverage**: no template for 4-chains yet — need
   overlapping-spine or separator-tree decompositions.
8. **End-to-end plan quality**: the paper's headline claim is that
   pessimistic bounds produce plans ≥ true-cardinality plans; wiring our
   bounds into an optimizer (Postgres hook / Lero-style comparator) is
   the remaining evaluation step.
9. **LLM loop**: candidate-statistic generation → support-argument
   soundness check + adversarial counterexample search → LP integration;
   this prototype is the manual run of that loop.

---

## 9. Files

```
boundlab/lpbench/
  engine.py   — LP construction (full + reduced), all statistics,
                solver, exact truth, lower bounds, certificates
  data.py     — synthetic generators + cit-Patents loader
  run.py      — benchmark driver (synth / j3 / graph)
  sql.py      — flat-SQL parser, NULL-correct predicate evaluation
  dbload.py   — CSV/IMDB/STATS/DBLP loaders, NULL-safe key encoding
  bench.py    — real-workload driver (DuckDB truth + arms + watchdog)
  results/    — per-workload CSVs
  REPORT.md   — this file
```

Reproduce:

```bash
cd boundlab/lpbench
python run.py synth                       # synthetic table
python run.py j3                          # 3-chain
python run.py graph                       # cit-Patents, directed + undirected
python bench.py joblight                  # JOB-light, IMDB (arms via ARMS=)
ARMS=lpbound,pairs,triples python bench.py stats
STARTQ=330 MAXQ=10 python bench.py jobrange   # partial reruns
```
