"""lpbench engine: clean-room LpBound baseline + ours (pairwise marginals).

Query model: full conjunctive query over set-valued relations
    Q(V) = R1(A1) |x| R2(A2) |x| ... |x| Rm(Am)
plus optional selection predicates (equality / range, conjunction)
and optional group-by attribute subset.

Tables are dicts {col_name: int64 np.ndarray}.  An *atom* binds a table
to logical attribute names: {"t": table_name, "c": {attr: col_name},
"pred": optional callable(rows)->mask}.  Self-joins are just several
atoms on the same table with different column maps.

LpBound statistics (SIGMOD'25, Zhang-Mayer-Khamis-Olteanu-Suciu):
  for each atom j, each conditioning set U ⊆ joinattrs(j):
      full  deg_j(*|U)(u) = |sigma_{U=u} R_j|            (fiber size)
      simple deg_j(X|U)(u) = |pi_X sigma_{U=u} R_j|     (distinct X)
  each yields |P| constraints   h(U+V) - (1-1/p) h(U) <= log2 ||deg||_p
  (p=1 gives cardinality, p=inf the max-degree bound.)

Ours adds, for each pair of atoms sharing >= 1 attribute:
      h(A_i u A_j) <= log2 |R_i |x|_shared R_j|          (exact pair marginal)
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.optimize import linprog

PSET = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf]


# ---------------------------------------------------------------- encoding
def encode(cols):
    """Pack int64 columns into one int64 key; falls back to dense rank
    codes via lexicographic unique if the packed value would overflow."""
    if len(cols[0]) == 0:
        return np.zeros(0, dtype=np.int64)
    k = np.zeros(len(cols[0]), dtype=np.int64)
    hi = 1
    for c in cols:
        m = int(c.max()) + 1
        if hi > np.iinfo(np.int64).max // max(m, 1):
            return np.unique(np.stack(cols, 1), axis=0,
                             return_inverse=True)[1].astype(np.int64)
        k = k * m + c
        hi *= m
    return k


def degseq(rows, u_cols, x_col=None):
    """Degree sequence (sorted desc).
    x_col=None -> full: fiber tuple counts per u.
    x_col=c    -> simple: #distinct c-values per u."""
    ku = encode([rows[c] for c in u_cols]) if u_cols else np.zeros(len(next(iter(rows.values()))), dtype=np.int64)
    if x_col is None:
        _, cnt = np.unique(ku, return_counts=True)
    else:
        ux = encode([ku, rows[x_col]])
        _, first = np.unique(ux, return_index=True)
        _, cnt = np.unique(ku[np.sort(first)], return_counts=True)
    return np.sort(cnt)[::-1]


def log2_norms(d):
    """{p: log2 ||d||_p} computed in log space."""
    out = {}
    ld = np.log2(d.astype(np.float64))
    mx = ld.max() if len(d) else 0.0
    for p in PSET:
        if p == 1:
            out[p] = np.log2(d.sum())
        elif p == np.inf:
            out[p] = mx
        else:
            out[p] = mx + np.log2((2.0 ** ((ld - mx) * p)).sum()) / p
    return out


def pair_join_size(rows_i, ci, rows_j, cj, shared_i, shared_j):
    """Exact |pi_{shared} join| count = sum_k a_k * b_k."""
    ki = encode([rows_i[c] for c in shared_i])
    kj = encode([rows_j[c] for c in shared_j])
    vi, ci_ = np.unique(ki, return_counts=True)
    vj, cj_ = np.unique(kj, return_counts=True)
    common, ii, jj = np.intersect1d(vi, vj, assume_unique=True,
                                  return_indices=True)
    return int((ci_[ii] * cj_[jj]).sum())


# ------------------------------------------------------------------- query
class Atom:
    def __init__(self, table, colmap, pred=None):
        self.table = table          # name into db
        self.c = dict(colmap)       # attr -> column name
        self.pred = pred            # rows(dict) -> bool mask

    @property
    def attrs(self):
        return set(self.c)


class Query:
    def __init__(self, atoms, group_attrs=None):
        self.atoms = atoms
        self.attrs = sorted(set().union(*[a.attrs for a in atoms]))
        self.group_attrs = sorted(group_attrs) if group_attrs else list(self.attrs)


def filtered_rows(db, atom):
    rows = db[atom.table]
    if atom.pred is None:
        return rows
    m = atom.pred(rows)
    return {c: v[m] for c, v in rows.items()}


# ------------------------------------------------------------------ LP build
def shannon_constraints(n):
    """Monotonicity + elementary submodularities (span the Shannon cone)."""
    nvar = 1 << n
    rows = []
    for S in range(nvar):
        for v in range(n):
            if not (S >> v) & 1:
                r = np.zeros(nvar); r[S] = 1; r[S | (1 << v)] = -1
                rows.append(r)
    for a, b in itertools.combinations(range(n), 2):
        ab = (1 << a) | (1 << b)
        for S in range(nvar):
            if S & ab:
                continue
            r = np.zeros(nvar)
            r[S | ab] = 1; r[S] = 1
            r[S | (1 << a)] = -1; r[S | (1 << b)] = -1
            rows.append(r)
    return np.array(rows)


def _build_lp(query, db, pairs=False, triples=False):
    """Build (A, b, desc, c, M, nvar). desc[i] describes row i of A_ub."""
    attrs = query.attrs
    idx = {a: i for i, a in enumerate(attrs)}
    n = len(attrs)
    nvar = 1 << n

    def M(names):
        m = 0
        for a in names:
            m |= 1 << idx[a]
        return m

    A = [shannon_constraints(n)]
    b = [np.zeros(len(A[0]))]
    desc = ["shannon"] * len(A[0])
    h0 = np.zeros(nvar); h0[0] = 1.0          # h(empty) <= 0
    A.append(h0.reshape(1, -1)); b.append(np.zeros(1)); desc.append("h(0)=0")
    n_stat = 0
    empty = False

    for ai_, atom in enumerate(query.atoms):
        rows = filtered_rows(db, atom)
        if len(next(iter(rows.values()))) == 0:
            empty = True
            break
        J = atom.attrs                       # join attrs of this atom
        seqs = []
        for r in range(len(J) + 1):
            for U in itertools.combinations(sorted(J), r):
                U = set(U)
                seqs.append((U, None, degseq(rows, [atom.c[u] for u in U])))
                for X in sorted(J - U):
                    seqs.append((U, X, degseq(rows, [atom.c[u] for u in U],
                                              atom.c[X])))
        for U, X, d in seqs:
            target = M(U | ({X} if X else J))
            tag = (f"{atom.table}[{ai_}].deg"
                   f"({'*' if X is None else X}|{''.join(sorted(U))})")
            for p, ln in log2_norms(d).items():
                row = np.zeros(nvar)
                row[target] = 1.0
                if p != np.inf:
                    row[M(U)] = -(1.0 - 1.0 / p)
                else:
                    row[M(U)] = -1.0
                A.append(row.reshape(1, -1)); b.append(np.array([ln]))
                desc.append(f"{tag} p={p}")
                n_stat += 1

    if empty:
        return None, None, desc, None, M, nvar

    def multi_count(atoms_):
        if len(atoms_) == 2:
            ai, aj = atoms_
            sh = sorted(ai.attrs & aj.attrs)
            return pair_join_size(
                filtered_rows(db, ai), ai.c, filtered_rows(db, aj), aj.c,
                [ai.c[a] for a in sh], [aj.c[a] for a in sh])
        return _true_count_merge(Query(list(atoms_)), db)

    def _connected(atoms_):
        """Connected via shared attributes (chain counts too)."""
        seen = set(atoms_[0].attrs)
        rest = list(atoms_[1:])
        while rest:
            nxt = [a_ for a_ in rest if a_.attrs & seen]
            if not nxt:
                return False
            for a_ in nxt:
                seen |= a_.attrs
                rest.remove(a_)
        return True

    if pairs or triples:
        sizes = ([2] if pairs else []) + ([3] if triples else [])
        for k in sizes:
            for combo in itertools.combinations(range(len(query.atoms)), k):
                atoms = [query.atoms[i] for i in combo]
                if not _connected(atoms):
                    continue
                cnt = multi_count(atoms)
                row = np.zeros(nvar)
                row[M(set().union(*[a_.attrs for a_ in atoms]))] = 1.0
                A.append(row.reshape(1, -1))
                b.append(np.array([np.log2(max(cnt, 1))]))
                desc.append(f"k{k}:{'+'.join(a.table for a in atoms)}")
                n_stat += 1

    c = np.zeros(nvar)
    c[M(query.group_attrs)] = -1.0
    return np.vstack(A), np.concatenate(b), desc, c, M, nvar, n_stat


def bound(query, db, pairs=False, triples=False):
    """Return (bound, lp_opt_bits, n_stat_rows)."""
    out = _build_lp(query, db, pairs=pairs, triples=triples)
    if out[0] is None:
        return 0.0, -np.inf, 0
    A, b, desc, c, M, nvar, n_stat = out
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    return 2.0 ** (-res.fun), -res.fun, n_stat


def certify(query, db, pairs=False, triples=False, tol=1e-7):
    """Extract a PANDA-style proof certificate: the LP dual solution.

    Returns (bound, certificate) where certificate is a list of
    (description, dual_weight, rhs_bits).  By LP duality
        log2(bound) = sum_i dual_weight_i * rhs_i
    and every listed row is a named valid inequality, so the bound is
    reproducible as a nonnegative combination of verified facts.
    """
    out = _build_lp(query, db, pairs=pairs, triples=triples)
    if out[0] is None:
        return 0.0, []
    A, b, desc, c, M, nvar, n_stat = out
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    # HiGHS reports marginals <= 0 for "<=" rows of a minimization;
    # dual weights are -marginals >= 0 and satisfy
    #     -res.fun == sum_i w_i * b_i        (LP strong duality)
    lam = -np.asarray(res.ineqlin.marginals)
    cert = []
    for i, l in enumerate(lam):
        if l > tol:
            cert.append((desc[i] if i < len(desc) else f"row{i}",
                         float(l), float(b[i])))
    lhs = sum(l_ * rhs for _, l_, rhs in cert)
    return 2.0 ** (-res.fun), cert, float(-res.fun), float(lhs)


# ------------------------------------------------------------ lower bounds
def encode_cols(rows, atom, attrs_):
    return encode([rows[atom.c[a]] for a in attrs_])


def lower_bound(query, db):
    """Provable lower bound on |Q| via inclusion-exclusion over degrees.

    Templates:
      * spine/star: some atom S such that non-spine atoms share attrs
        only with S.  |Q| = sum_{t in S} prod_o deg_o(t[shared_o]) and
        prod_i a_i >= sum_i a_i - (k-1) for integers a_i >= 1.
      * triangle (3 atoms, pairwise single shared attr): per middle tuple
        (y,z), the closing set is an intersection of two key sets, so
        |A_y cap B_z| >= |A_y| + |B_z| - n_X.
    Returns int, or None when no template applies.
    """
    atoms, m = query.atoms, len(query.atoms)

    # triangle: every pair shares exactly one attr, three distinct attrs
    if m == 3:
        sh = [atoms[i].attrs & atoms[j].attrs
              for i, j in itertools.combinations(range(3), 2)]
        if all(len(s) == 1 for s in sh) and len(set.union(*sh)) == 3:
            return _triangle_lb(atoms, db)

    # spine/star detection: attrs shared by two non-spine atoms must be
    # spine attrs (then a spine tuple fixes all join conditions)
    for si in range(m):
        S_attrs = atoms[si].attrs
        ok = all(
            (atoms[i].attrs & atoms[j].attrs) <= S_attrs or si in (i, j)
            for i, j in itertools.combinations(range(m), 2))
        ok = ok and all(atoms[si].attrs & atoms[k].attrs
                        for k in range(m) if k != si)
        if ok:
            return _spine_lb(atoms, si, db)
    return None


def _spine_lb(atoms, si, db):
    S = atoms[si]
    others = [atoms[k] for k in range(len(atoms)) if k != si]
    srows = filtered_rows(db, S)
    n = len(next(iter(srows.values())))
    sums = np.zeros(n, dtype=np.int64)
    allpos = np.ones(n, dtype=bool)
    for o in others:
        sha = sorted(S.attrs & o.attrs)
        orows = filtered_rows(db, o)
        ov, oc = np.unique(encode_cols(orows, o, sha), return_counts=True)
        deg_map = dict(zip(ov.tolist(), oc.tolist()))
        sk = encode([srows[S.c[a]] for a in sha])
        d = np.fromiter((deg_map.get(int(v), 0) for v in sk),
                        dtype=np.int64, count=n)
        sums += d
        allpos &= d > 0
    contrib = np.maximum(0, sums - (len(others) - 1))
    return int(contrib[allpos].sum())


def _triangle_lb(atoms, db):
    """|tri| >= sum_{(y,z) in E2} max(0, deg_1(y) + deg_3(z) - n_X)
    where deg_1(y) = |pi_X sigma_{Y=y} E1| etc. and n_X bounds the
    closing-attribute domain."""
    e1, e2, e3 = atoms
    sh12 = next(iter(e1.attrs & e2.attrs))          # Y
    sh23 = next(iter(e2.attrs & e3.attrs))          # Z
    sh13 = next(iter(e1.attrs & e3.attrs))          # X
    r1, r2, r3 = (filtered_rows(db, a) for a in atoms)
    # candidate x's per y (resp. per z): rows of e1 with Y=y contribute
    # distinct x's; under set semantics row count == distinct count.
    d1 = dict(zip(*map(lambda a: a.tolist(),
                       np.unique(r1[e1.c[sh12]], return_counts=True))))
    d3 = dict(zip(*map(lambda a: a.tolist(),
                       np.unique(r3[e3.c[sh23]], return_counts=True))))
    nX = len(np.union1d(np.unique(r1[e1.c[sh13]]),
                        np.unique(r3[e3.c[sh13]])))
    return int(sum(max(0, d1.get(int(y), 0) + d3.get(int(z), 0) - nX)
                   for y, z in zip(r2[e2.c[sh12]], r2[e2.c[sh23]])))


# --------------------------------------------------------------- true count
def _shared_key_truth(query, db):
    """If every pair of atoms shares exactly the same single attribute K,
    truth = sum_k prod_j deg_j(k) -- no materialization needed."""
    keyattr = None
    for i, j in itertools.combinations(range(len(query.atoms)), 2):
        sh = query.atoms[i].attrs & query.atoms[j].attrs
        if len(sh) != 1:
            return None
        if keyattr is None:
            keyattr = next(iter(sh))
        elif next(iter(sh)) != keyattr:
            return None
    if keyattr is None:
        return None
    acc = None
    for atom in query.atoms:
        kcol = filtered_rows(db, atom)[atom.c[keyattr]]
        v, c = np.unique(kcol, return_counts=True)
        m = dict(zip(v.tolist(), c.tolist()))
        if acc is None:
            acc = m
        else:
            acc = {k: acc[k] * m[k] for k in acc.keys() & m.keys()}
    return int(sum(acc.values()))


def true_count(query, db):
    fast = _shared_key_truth(query, db)
    if fast is not None and set(query.group_attrs) == set(query.attrs):
        return fast
    return _true_count_merge(query, db)


def _true_count_merge(query, db):
    """Exact join size (bag semantics) via weight-propagating merge join.

    Intermediate state = dict attr -> np.ndarray plus weight '_w'; each
    step joins on shared attrs and re-contracts duplicate tuples into
    weights, so memory stays bounded by #distinct intermediate tuples.
    """
    atoms = query.atoms
    cur = {a: filtered_rows(db, atoms[0])[atoms[0].c[a]].copy()
           for a in atoms[0].attrs}
    cur["_w"] = np.ones(len(next(iter(cur.values()))), dtype=np.int64)
    cur_attrs = set(atoms[0].attrs)
    cur = _contract(cur, cur_attrs)
    for atom in atoms[1:]:
        rows = filtered_rows(db, atom)
        rcols = {a: rows[atom.c[a]] for a in atom.attrs}
        shared = cur_attrs & atom.attrs
        if not shared:
            continue
        kl = encode([cur[s] for s in shared])
        kr = encode([rcols[s] for s in shared])
        # per-key positions on each side
        lk, li = np.unique(kl, return_inverse=True)
        rk, ri = np.unique(kr, return_inverse=True)
        common, il, ir = np.intersect1d(lk, rk, assume_unique=True,
                                      return_indices=True)
        # left row index -> position within its key group
        lorder = np.argsort(li, kind="stable")
        li_s = li[lorder]
        loff = np.searchsorted(li_s, np.arange(len(lk) + 1))
        rorder = np.argsort(ri, kind="stable")
        ri_s = ri[rorder]
        roff = np.searchsorted(ri_s, np.arange(len(rk) + 1))
        new_attrs = sorted(cur_attrs | atom.attrs)
        out = {a: [] for a in new_attrs}
        out["_w"] = []
        for c in range(len(common)):
            L = lorder[loff[il[c]]:loff[il[c] + 1]]
            Rr = rorder[roff[ir[c]]:roff[ir[c] + 1]]
            li_r = np.repeat(L, len(Rr))
            rj_r = np.tile(Rr, len(L))
            for a in cur_attrs:
                out[a].append(cur[a][li_r])
            for a in atom.attrs - cur_attrs:
                out[a].append(rcols[a][rj_r])
            out["_w"].append(cur["_w"][li_r])
        cur = {a: np.concatenate(v) if v else np.array([], dtype=np.int64)
               for a, v in out.items()}
        cur_attrs = set(new_attrs)
        if len(cur["_w"]) == 0:
            return 0
        cur = _contract(cur, cur_attrs)
    if set(query.group_attrs) != set(query.attrs):
        key = encode([cur[a] for a in query.group_attrs])
        return int(len(np.unique(key)))
    return int(cur["_w"].sum())


def _contract(cur, attrs):
    key = encode([cur[a] for a in sorted(attrs)])
    uk, first, inv, cnt = np.unique(key, return_index=True,
                                  return_inverse=True, return_counts=True)
    w = np.zeros(len(uk), dtype=np.int64)
    np.add.at(w, inv, cur["_w"])
    out = {a: cur[a][first] for a in sorted(attrs)}
    out["_w"] = w
    return out
