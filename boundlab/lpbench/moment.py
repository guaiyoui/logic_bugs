"""Self-contained moment-theoretic join-size bounds.

Framework (no entropy, no h(U) variables):

  Star query  R_1(X,..) ⋈ ... ⋈ R_m(X,..)  on one shared key X.
  Every key x carries a degree vector  d(x) = (d_1(x),...,d_m(x)),
  d_i(x) = #{t in R_i : t.X = x}.  Then

      |Q| = sum_x  prod_i d_i(x)                     (product moment)

  Statistics = observed moments  mu_a = sum_x prod_i d_i(x)^{a_i}
  for alpha in an observed set A:
      alpha = p*e_i            ->  ||deg_i||_p^p   (LpBound's statistics)
      alpha = e_i + e_j        ->  |R_i ⋈ R_j|   (pair join counts)
      alpha = e_i+e_j+e_k      ->  triple counts
      alpha = 0                ->  |dom(X)|      (support size)

  CERTIFICATE LP (dual form).  Since summing a pointwise inequality
      prod_i d_i  <=  sum_a c_a d^a        for every *realized* vector d
  over the field gives  |Q| <= sum_a c_a mu_a ,  the tightest bound is

      UB = min_c  sum_a c_a mu_a
           s.t.   prod_i d_i  <=  sum_a c_a d^a    for all d in D

  where D = set of distinct realized degree vectors (each weighted by
  its multiplicity).  The lower-bound version flips both inequalities:

      LB = max_c  sum_a c_a mu_a
           s.t.   sum_a c_a d^a <= prod_i d_i     for all d in D

  Rows of the LP = distinct realized degree vectors;  columns = moments.
  AGM-type / degree-sequence / pair-count / partition bounds all appear
  as different observed-moment sets A of this one LP.
"""
import itertools
import numpy as np
from scipy.optimize import linprog


# ------------------------------------------------------------- degree field
def degree_field(degseqs):
    """degseqs: list of (keys_i, deg_i) arrays, keys already encoded in a
    common domain (engine.encode'ed columns of the shared attribute).

    Returns (D, mult): D = (K, m) distinct degree vectors, mult = (K,)
    multiplicity = #keys carrying that vector.  Keys absent from atom i
    get d_i = 0 (they contribute 0 to the product anyway but still
    appear in single-variable moments)."""
    m = len(degseqs)
    # gather union of keys with per-atom degree columns
    all_keys = np.unique(np.concatenate([k for k, _ in degseqs]))
    cols = []
    for keys, deg in degseqs:
        pos = np.searchsorted(all_keys, keys)
        v = np.zeros(len(all_keys), dtype=np.float64)
        v[pos] = deg
        cols.append(v)
    F = np.stack(cols, axis=1)                    # (#keys, m)
    D, mult = np.unique(F, axis=0, return_counts=True)
    return D, mult.astype(np.float64)


# ---------------------------------------------------------------- moments
def moment_matrix(D, alphas):
    """M[k, a] = prod_i D[k, i] ** alphas[a][i]."""
    K, m = D.shape
    M = np.ones((K, len(alphas)))
    for j, a in enumerate(alphas):
        a = np.asarray(a)
        for i in range(m):
            if a[i]:
                M[:, j] *= D[:, i] ** a[i]
    return M


def observed_moments(D, mult, alphas):
    """mu_a = sum_k mult[k] * prod D[k]^alphas[a]."""
    return moment_matrix(D, alphas).T @ mult


def product_values(D):
    """p_k = prod_i D[k,i]  (the per-vector contribution to |Q|)."""
    return D.prod(axis=1)


# ---------------------------------------------------------- certificate LP
def product_domain(degseqs, cap=200_000):
    """Worst-case dominance domain: the cartesian product of per-atom
    realized degree supports (with 0 for absent keys).  Scalar moments
    do not reveal which d_i(x) pairs with which d_j(x), so a valid
    certificate must dominate on EVERY combination in the product.
    For large products the caller should use separation via
    `certificate_bound_separated` instead of materializing this."""
    vals = [np.unique(np.concatenate([[0.0], d.astype(np.float64)]))
            for _, d in degseqs]
    n = int(np.prod([len(v) for v in vals]))
    if n > cap:
        return None, vals
    grids = np.meshgrid(*vals, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1), vals


def _min_slack(vals, c, alphas, side, block=400_000):
    """Separation oracle: scan the full product grid (blocked over ALL
    axes, so a single block never exceeds `block` rows) for the point
    minimizing the certificate slack.
    upper: slack = M(d).c - prod(d); violation iff min < 0.
    Returns (min_slack, argmin_point)."""
    m = len(vals)
    # per-axis chunk counts so prod(chunks_len) <= block
    budgets = [1] * m
    rem = block
    for i in range(m):
        budgets[i] = max(1, min(len(vals[i]), rem))
        rem = max(1, rem // budgets[i])
    ranges = []
    for i in range(m):
        edges = np.linspace(0, len(vals[i]),
                            int(np.ceil(len(vals[i]) / budgets[i])) + 1
                            ).astype(int)
        ranges.append([vals[i][a:b] for a, b in zip(edges[:-1],
                                                  edges[1:])])
    best, best_pt = np.inf, None
    for combo in itertools.product(*ranges):
        grids = np.meshgrid(*combo, indexing="ij")
        dg = np.stack([g.ravel() for g in grids], axis=1)
        Md = moment_matrix(dg, alphas) @ c
        slack = Md - product_values(dg)
        if side == "lower":
            slack = -slack
        j = np.argmin(slack)
        if slack[j] < best:
            best, best_pt = float(slack[j]), dg[j]
    return best, best_pt


def certificate_bound_separated(D, mult, alphas, degseqs, side="upper",
                                tol=1e-9, maxit=200, solver_retry=0):
    """Certificate LP over the full worst-case product domain via
    cutting planes: solve on a small domain, ask the oracle for the
    most-violated grid point, add it, repeat.  Converges with an
    optimality proof (final certificate dominates the ENTIRE product).
    `solver_retry` > 0 perturbs the LP numerics (tolerance/presolve)
    to work around transient HiGHS failures on degenerate LPs."""
    Om, vals = product_domain(degseqs)
    if Om is not None:                     # small enough: solve directly
        return certificate_bound(D, mult, alphas, side, omega=Om,
                                 retry=solver_retry)
    # seed domain: realized field + axis midpoint + axis extremes
    m = len(vals)
    seed = [np.asarray(D, dtype=np.float64)]
    for i in range(m):
        mid = np.zeros((1, m))
        mid[0, i] = vals[i][(len(vals[i]) - 1) // 2]
        seed.append(mid)
        ext = np.zeros((1, m))
        ext[0, i] = vals[i][-1]
        seed.append(ext)
    Om_cur = np.unique(np.concatenate(seed), axis=0)
    for _ in range(maxit):
        b, c, msg = certificate_bound(D, mult, alphas, side,
                                      omega=Om_cur, retry=solver_retry)
        if b is None:
            return None, None, msg
        slack, pt = _min_slack(vals, c, alphas, side)
        if slack >= -tol * max(1.0, abs(b)):
            return b, c, alphas              # dominates all of Omega
        Om_cur = np.unique(np.concatenate([Om_cur, pt[None]]), axis=0)
    return b, c, "separation did not converge"


def certificate_bound(D, mult, alphas, side="upper", omega=None,
                      retry=0):
    """Solve the certificate LP.

    Moments mu are computed on the realized field (D, mult); the
    dominance constraints run over `omega` (defaults to D itself --
    pass product_domain(degseqs) for the worst-case domain).

    upper: min c.mu  s.t.  M_O c >= p_O   (c free)
    lower: max c.mu  s.t.  M_O c <= p_O
    Returns (bound, coeffs, alphas).
    """
    Om = D if omega is None else omega
    M = moment_matrix(Om, alphas)
    p = product_values(Om)
    mu = moment_matrix(D, alphas).T @ mult
    # column scaling: moments differ by 30+ orders of magnitude; solve in
    # scaled variables c'_a = c_a * s_a so all matrix entries lie in [0,1]
    s = M.max(axis=0)
    s[s == 0] = 1.0
    Ms = M / s
    mus = mu / s
    # nonnegative certificates only: every classical inequality
    # (AM-GM/Young/Hoelder) lives in this cone, and free coefficients
    # invite numeric unboundedness misdetection in HiGHS.
    from scipy.sparse import csr_matrix
    opts = {"primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9}
    if retry == 1:
        opts["presolve"] = False
    elif retry >= 2:
        opts = {"primal_feasibility_tolerance": 1e-7,
                "dual_feasibility_tolerance": 1e-7}
    Asp = csr_matrix(Ms)
    if side == "upper":
        res = linprog(mus, A_ub=-Asp, b_ub=-p, bounds=(0, None),
                      method="highs", options=opts)
    else:
        res = linprog(-mus, A_ub=Asp, b_ub=p, bounds=(0, None),
                      method="highs", options=opts)
    if res.status != 0:
        return None, None, res.message
    val = float(np.dot(mus, res.x))
    # sanity floor: any "super-product" moment (alpha_i >= 1 for all i)
    # is itself a feasible monomial certificate  prod d_i <= d^alpha,
    # so UB <= min over such mu_alpha -- guards against HiGHS
    # returning a numerically suboptimal point on badly-scaled LPs.
    if side == "upper":
        sp = [j for j, a in enumerate(alphas) if all(x >= 1 for x in a)]
        if sp:
            val = min(val, float(np.min(mu[sp])))
    else:
        sp = [j for j, a in enumerate(alphas) if all(x >= 1 for x in a)]
        if sp:
            val = max(val, 0.0)
    return val, res.x / s, alphas


# ----------------------------------------------------------- moment sets
def univariate_alphas(m, ps):
    """{p*e_i}: the degree-norm (LpBound-style) moment set."""
    return [tuple(0 if j != i else p for j in range(m))
            for i in range(m) for p in ps]


def pair_alphas(m):
    """{e_i + e_j}: exact pair join counts."""
    return [tuple(1 if j in (a, b) else 0 for j in range(m))
            for a, b in itertools.combinations(range(m), 2)]


def triple_alphas(m):
    return [tuple(1 if j in t else 0 for j in range(m))
            for t in itertools.combinations(range(m), 3)]


def with_support(alphas, m):
    """Prepend alpha = 0 (support-size observation)."""
    return [tuple(0 for _ in range(m))] + list(alphas)


# -------------------------------------------------------- Hoelder section
def holder_bound(degseqs, ps=None):
    """Closed form: min over conjugate exponents of prod ||deg_i||_{p_i}.
    Scans a weight-simplex grid, so conjugates are *continuous*
    (p_i = 1/w_i need not be integer) — unlike the certificate LP whose
    moment set is an integer-exponent lattice (§4.3)."""
    m = len(degseqs)

    def eval_exp(ws):
        out = 1.0
        for i, w in enumerate(ws):
            p = 1.0 / w
            pn = np.sum(degseqs[i][1].astype(np.float64) ** p) ** (1.0 / p)
            out *= pn
        return out

    # grid over weight simplex (w_i = 1/p_i), coarse but enough to
    # illustrate the section's value
    best = np.inf
    grid = [g / 10.0 for g in range(1, 10)]
    for ws in itertools.product(grid, repeat=m):
        if abs(sum(ws) - 1.0) > 1e-9:
            continue
        best = min(best, eval_exp(ws))
    return best
