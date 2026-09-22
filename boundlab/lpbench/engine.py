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
# Packing must be a pure function of the column VALUES (consistent across
# tables -- these keys feed join intersections).  Per-table bases/mins
# silently break that: same tuple -> different packed keys -> wrong joins.
# We therefore use a fixed radix/offset; out-of-range or wide tuples go
# through arbitrary-precision Python ints with the SAME convention.
_EBASE = np.int64(1) << 31          # digit base
_EOFF = np.int64(1) << 30           # maps [-2^30, 2^30) -> [0, 2^31)


def _encode_obj(cols):
    """Arbitrary-precision pack, same fixed-base convention (rare path)."""
    ko = (cols[0] + _EOFF).astype(object)
    for c in cols[1:]:
        ko = ko * int(_EBASE) + (c + _EOFF).astype(object)
    return ko


def encode(cols):
    """Pack int64 columns into one key, consistent ACROSS tables."""
    if len(cols[0]) == 0:
        return np.zeros(0, dtype=np.int64)
    for c in cols:
        lo, hi_ = int(c.min()), int(c.max())
        if lo < -int(_EOFF) or hi_ >= int(_EBASE) - int(_EOFF):
            return _encode_obj(cols)
    k = np.zeros(len(cols[0]), dtype=np.int64)
    hi = 1
    for c in cols:
        if hi > np.iinfo(np.int64).max // int(_EBASE):
            return _encode_obj(cols)
        k = k * _EBASE + (c + _EOFF)
        hi *= int(_EBASE)
    return k


def _enc(rows, cols, ctx=None):
    """encode() with per-query cache."""
    if ctx is None:
        return encode([rows[c] for c in cols])
    key = (id(rows), tuple(cols))
    ec = ctx.setdefault("enc", {})
    if key not in ec:
        ec[key] = encode([rows[c] for c in cols])
    return ec[key]


def _dense_col(rows, col, ctx=None):
    """Dense rank codes of a column (cached per column in ctx).
    Local to one table -- only used for within-table grouping."""
    if ctx is None:
        return np.unique(rows[col], return_inverse=True)[1].astype(np.int64)
    key = ("dense", id(rows), col)
    dc = ctx.setdefault("enc", {}).get(key)
    if dc is None:
        dc = np.unique(rows[col], return_inverse=True)[1].astype(np.int64)
        ctx["enc"][key] = dc
    return dc


def _deg_key(rows, cols, ctx=None):
    """Pack columns into one int64 group key, LOCAL to this row set.
    Columns are dense-ranked first (cheap + cached), then pairwise-packed.
    Only used for within-table grouping, so codes need not be stable
    across tables."""
    if not cols:
        return np.zeros(len(next(iter(rows.values()))), dtype=np.int64)
    codes = [_dense_col(rows, c, ctx) for c in cols]
    while len(codes) > 1:
        nxt = []
        for i in range(0, len(codes), 2):
            a = codes[i]
            if i + 1 == len(codes):
                nxt.append(a)
                break
            b = codes[i + 1]
            mb = int(b.max()) + 1
            if int(a.max()) > np.iinfo(np.int64).max // max(mb, 1):
                # pathological: both near-unique columns; group by lexsort
                order = np.lexsort((b, a))
                sa, sb = a[order], b[order]
                new = np.ones(len(a), bool)
                new[1:] |= (sa[1:] != sa[:-1]) | (sb[1:] != sb[:-1])
                g = np.empty(len(a), np.int64)
                g[order] = np.cumsum(new) - 1
                nxt.append(g)
            else:
                nxt.append(a * mb + b)
        codes = nxt
    return codes[0]


def degseq(rows, u_cols, x_col=None, ctx=None):
    """Degree sequence (sorted desc).
    x_col=None -> full: fiber tuple counts per u.
    x_col=c    -> simple: #distinct c-values per u."""
    if ctx is not None:
        key = ("deg", id(rows), tuple(u_cols), x_col)
        got = ctx.setdefault("enc", {}).get(key)
        if got is not None:
            return got
    ku = (_deg_key(rows, u_cols, ctx) if u_cols
          else np.zeros(len(next(iter(rows.values()))), dtype=np.int64))
    if x_col is None:
        _, cnt = np.unique(ku, return_counts=True)
    else:
        ux = _deg_key(rows, list(u_cols) + [x_col], ctx)
        _, first = np.unique(ux, return_index=True)
        _, cnt = np.unique(ku[np.sort(first)], return_counts=True)
    out = np.sort(cnt)[::-1]
    if ctx is not None:
        ctx["enc"][key] = out
    return out


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


def pair_join_size(rows_i, ci, rows_j, cj, shared_i, shared_j, ctx=None):
    """Exact |pi_{shared} join| count = sum_k a_k * b_k."""

    def vc(rows, cols):
        if ctx is None:
            return np.unique(
                encode([rows[c] for c in cols]), return_counts=True)
        key = ("vc", id(rows), tuple(cols))
        if key not in ctx.setdefault("enc", {}):
            ctx["enc"][key] = np.unique(
                _enc(rows, cols, ctx), return_counts=True)
        return ctx["enc"][key]

    vi, ci_ = vc(rows_i, shared_i)
    vj, cj_ = vc(rows_j, shared_j)
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


def filtered_rows(db, atom, ctx=None):
    """Predicate-filtered rows plus a synthetic __rowid__ column that
    makes each physical row a distinct entity (COUNT(*) semantics)."""
    if ctx is not None:
        key = (atom.table, tuple(sorted(atom.c.items())), id(atom.pred))
        rc = ctx.setdefault("rows", {})
        if key in rc:
            return rc[key]
    rows = dict(db[atom.table])
    n = len(next(iter(rows.values())))
    rows["__rowid__"] = np.arange(n)
    if atom.pred is not None:
        m = atom.pred(rows)
        rows = {c: v[m] for c, v in rows.items()}
    if ctx is not None:
        ctx["rows"][key] = rows
    return rows


# ------------------------------------------------------------------ LP build
def shannon_constraints(n):
    """Monotonicity + elementary submodularities as sparse (cols, coefs)."""
    nvar = 1 << n
    for S in range(nvar):
        for v in range(n):
            if not (S >> v) & 1:
                yield ((S, S | (1 << v)), (1.0, -1.0))
    for a, b in itertools.combinations(range(n), 2):
        ab = (1 << a) | (1 << b)
        for S in range(nvar):
            if S & ab:
                continue
            yield ((S | ab, S, S | (1 << a), S | (1 << b)),
                   (1.0, 1.0, -1.0, -1.0))


def _ones_norms(n_rows):
    """log2 ||ones(n)||_p = log2(n)/p (inf -> 0)."""
    lg = np.log2(max(n_rows, 1))
    return {p: (0.0 if p == np.inf else lg / p) for p in PSET}


def _build_lp(query, db, pairs=False, triples=False, ctx=None):
    """Build sparse (A, b, desc, c, M, nvar). desc[i] describes row i."""
    from scipy import sparse

    attrs = query.attrs
    idx = {a: i for i, a in enumerate(attrs)}
    n = len(attrs)
    nvar = 1 << n

    def M(names):
        m = 0
        for a in names:
            m |= 1 << idx[a]
        return m

    ri, ci_, vv, b, desc = [], [], [], [], []

    def add(cols, coefs, rhs, d):
        r = len(desc)
        for c_, v_ in zip(cols, coefs):
            ri.append(r); ci_.append(c_); vv.append(v_)
        b.append(rhs); desc.append(d)

    for cols, coefs in shannon_constraints(n):
        add(cols, coefs, 0.0, "shannon")
    add([0], [1.0], 0.0, "h(0)=0")
    n_stat = 0
    empty = False

    for ai_, atom in enumerate(query.atoms):
        rows = filtered_rows(db, atom, ctx)
        if len(next(iter(rows.values()))) == 0:
            empty = True
            break
        J = atom.attrs
        rid = next((a for a in J if atom.c[a] == "__rowid__"), None)
        S = sorted(J - {rid}) if rid else sorted(J)
        n_rows = len(next(iter(rows.values())))
        ones = _ones_norms(n_rows)
        seqs = []
        # U not containing rowid: real degree sequences (np.unique).
        # X=rowid reuses the full seq (each row distinct -> same counts).
        for r in range(len(S) + 1):
            for U in itertools.combinations(S, r):
                U = set(U)
                full = degseq(rows, [atom.c[u] for u in U], ctx=ctx)
                seqs.append((U, None, log2_norms(full)))
                for X in sorted(set(S) - U):
                    seqs.append((U, X, log2_norms(
                        degseq(rows, [atom.c[u] for u in U],
                               atom.c[X], ctx=ctx))))
                if rid:
                    seqs.append((U, rid, log2_norms(full)))
        # U containing rowid: every fiber is a single row -> ones seqs.
        if rid:
            for r in range(len(S) + 1):
                for U0 in itertools.combinations(S, r):
                    U = set(U0) | {rid}
                    seqs.append((U, None, ones))
                    for X in sorted(J - U):
                        seqs.append((U, X, ones))
        for U, X, norms in seqs:
            target = M(U | ({X} if X else J))
            tag = (f"{atom.table}[{ai_}].deg"
                   f"({'*' if X is None else X}|{''.join(sorted(U))})")
            for p, ln in norms.items():
                cols = [target]
                coefs = [1.0]
                if p != np.inf:
                    cols.append(M(U)); coefs.append(-(1.0 - 1.0 / p))
                else:
                    cols.append(M(U)); coefs.append(-1.0)
                add(cols, coefs, ln, f"{tag} p={p}")
                n_stat += 1

    if empty:
        return None, None, desc, None, M, nvar

    def multi_count(atoms_):
        if len(atoms_) == 2:
            ai, aj = atoms_
            sh = sorted(ai.attrs & aj.attrs)
            return pair_join_size(
                filtered_rows(db, ai, ctx), ai.c,
                filtered_rows(db, aj, ctx), aj.c,
                [ai.c[a] for a in sh], [aj.c[a] for a in sh], ctx=ctx)
        return _true_count_merge(Query(list(atoms_)), db, ctx=ctx)

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
                add([M(set().union(*[a_.attrs for a_ in atoms]))], [1.0],
                    np.log2(max(cnt, 1)),
                    f"k{k}:{'+'.join(a_.table for a_ in atoms)}")
                n_stat += 1

    c = np.zeros(nvar)
    c[M(query.group_attrs)] = -1.0
    A = sparse.coo_matrix((vv, (ri, ci_)),
                        shape=(len(desc), nvar)).tocsr()
    return A, np.asarray(b), desc, c, M, nvar, n_stat


# ------------------------------------------------- reduced (scalable) LP
# For queries with many attributes the full 2^n polymatroid LP is
# infeasible (n=23 -> 8.4M vars).  The reduced LP keeps only a subset of
# Shannon inequalities -- every kept row is still a valid inequality, so
# the bound stays sound (>= truth), just possibly looser than the full LP:
#   * all degree-statistic constraints (sets live inside one atom)
#   * a "ladder" over atoms: h(P_j) <= h(P_{j-1}) + h(A_j) - h(I_j)
#     with P_j = union of A_1..A_j, I_j = P_{j-1} cap A_j
#   * pair/triple ladder variants so marginal stats bite:
#       h(P_j) <= h(P_{j-1}) + h(U) - h(U' )
#     for U = union of the new atom with earlier connected atoms
#   * elementary submodularities/monotonicity among kept sets
NMAX_FULL_LP = 12


def _atom_order(atoms):
    """BFS order over shared-attribute connectivity."""
    adj = [set() for _ in atoms]
    for i, j in itertools.combinations(range(len(atoms)), 2):
        if atoms[i].attrs & atoms[j].attrs:
            adj[i].add(j); adj[j].add(i)
    seen, order, q_ = set(), [], []
    for s in range(len(atoms)):
        if s in seen:
            continue
        q_ = [s]; seen.add(s)
        while q_:
            i = q_.pop(0); order.append(i)
            for j in sorted(adj[i]):
                if j not in seen:
                    seen.add(j); q_.append(j)
    return order


def _build_lp_reduced(query, db, pairs=False, triples=False, ctx=None):
    """Same return contract as _build_lp but with kept-set variables."""
    from scipy import sparse

    atoms = list(query.atoms)
    k = len(atoms)
    order = _atom_order(atoms)
    A_i = [frozenset(a.attrs) for a in atoms]

    # ---- kept variable sets -------------------------------------------
    keepset = {frozenset()}
    atom_subs = []                       # all subsets inside each atom
    for a in atoms:
        subs = [frozenset(s) for r in range(len(a.attrs) + 1)
                for s in itertools.combinations(sorted(a.attrs), r)]
        atom_subs.append(subs)
        keepset.update(subs)
    P = [frozenset()]
    for i in order:
        P.append(P[-1] | A_i[i])
    keepset.update(P)
    I = [P[j] & A_i[order[j]] for j in range(k)]     # P_{j-1} cap A_j
    keepset.update(I)

    pairvar, trivar = {}, {}
    if pairs or triples:
        for pos, j in enumerate(order):
            for i in order[:pos]:
                if not (A_i[i] & A_i[j]):
                    continue
                pairvar[(i, j)] = A_i[i] | A_i[j]
                keepset.add(A_i[i] | A_i[j])
                keepset.add(A_i[i] | I[pos])
            if triples:
                for i1, i2 in itertools.combinations(order[:pos], 2):
                    U = A_i[i1] | A_i[i2] | A_i[j]
                    if (A_i[i1] & A_i[j] or A_i[i2] & A_i[j]) and \
                       (A_i[i1] & A_i[i2] or A_i[i1] & A_i[j]
                            or A_i[i2] & A_i[j]):
                        trivar[(i1, i2, j)] = U
                        keepset.add(U)
                        keepset.add(A_i[i1] | A_i[i2] | I[pos])

    keepset.add(frozenset(query.group_attrs))
    vidx = {s: t for t, s in enumerate(sorted(keepset, key=lambda x:
                                             (len(x), sorted(x))))}
    nvar = len(vidx)
    empty_m = vidx[frozenset()]

    ri, ci_, vv, b, desc = [], [], [], [], []

    def add(cols, coefs, rhs, d):
        r = len(desc)
        for c_, v_ in zip(cols, coefs):
            ri.append(r); ci_.append(c_); vv.append(v_)
        b.append(rhs); desc.append(d)

    def hv(s):
        return vidx.get(frozenset(s))

    import os, time as _t
    _dbg = os.environ.get("LPDEBUG")
    add([empty_m], [1.0], 0.0, "h(0)=0")
    n_stat = 0
    empty = False

    # ---- degree statistics (identical to full LP) ---------------------
    for ai_, atom in enumerate(atoms):
        _t0 = _t.time()
        rows = filtered_rows(db, atom, ctx)
        if len(next(iter(rows.values()))) == 0:
            empty = True
            break
        J = atom.attrs
        rid = next((a for a in J if atom.c[a] == "__rowid__"), None)
        S = sorted(J - {rid}) if rid else sorted(J)
        n_rows = len(next(iter(rows.values())))
        ones = _ones_norms(n_rows)
        seqs = []
        for r in range(len(S) + 1):
            for U in itertools.combinations(S, r):
                U = set(U)
                full = degseq(rows, [atom.c[u] for u in U], ctx=ctx)
                seqs.append((U, None, log2_norms(full)))
                for X in sorted(set(S) - U):
                    seqs.append((U, X, log2_norms(
                        degseq(rows, [atom.c[u] for u in U],
                               atom.c[X], ctx=ctx))))
                if rid:
                    seqs.append((U, rid, log2_norms(full)))
        if rid:
            for r in range(len(S) + 1):
                for U0 in itertools.combinations(S, r):
                    U = set(U0) | {rid}
                    seqs.append((U, None, ones))
                    for X in sorted(J - U):
                        seqs.append((U, X, ones))
        for U, X, norms in seqs:
            tgt = hv(U | ({X} if X else J))
            ucol = hv(U)
            if tgt is None or ucol is None:
                continue                    # shouldn't happen (inside atom)
            tag = (f"{atom.table}[{ai_}].deg"
                   f"({'*' if X is None else X}|{''.join(sorted(U))})")
            for p, ln in norms.items():
                cols = [tgt, ucol]
                coefs = [1.0, -1.0 if p == np.inf else -(1.0 - 1.0 / p)]
                add(cols, coefs, ln, f"{tag} p={p}")
                n_stat += 1
        if _dbg:
            print(f"  [dbg] stats atom {ai_} {atom.table} "
                  f"{_t.time()-_t0:.1f}s", flush=True)
    if empty:
        return None, None, desc, None, None, nvar

    # ---- pair / triple marginal stats ---------------------------------
    def multi_count(atoms_):
        if len(atoms_) == 2:
            ai, aj = atoms_
            sh = sorted(ai.attrs & aj.attrs)
            return pair_join_size(
                filtered_rows(db, ai, ctx), ai.c,
                filtered_rows(db, aj, ctx), aj.c,
                [ai.c[a] for a in sh], [aj.c[a] for a in sh], ctx=ctx)
        return _true_count_merge(Query(list(atoms_)), db, ctx=ctx)

    if pairs or triples:
        for (i, j), U in pairvar.items():
            _t0 = _t.time()
            cnt = multi_count([atoms[i], atoms[j]])
            add([hv(U)], [1.0], np.log2(max(cnt, 1)),
                f"k2:{atoms[i].table}+{atoms[j].table}")
            n_stat += 1
            if _dbg:
                print(f"  [dbg] pair {i},{j} {_t.time()-_t0:.1f}s",
                      flush=True)
        for (i1, i2, j), U in trivar.items():
            cnt = multi_count([atoms[i1], atoms[i2], atoms[j]])
            add([hv(U)], [1.0], np.log2(max(cnt, 1)),
                f"k3:{atoms[i1].table}+{atoms[i2].table}+{atoms[j].table}")
            n_stat += 1
    if _dbg:
        print(f"  [dbg] stats done, {n_stat} rows", flush=True)

    # ---- ladder decomposition -----------------------------------------
    for pos, j in enumerate(order):
        pj, pjm1 = hv(P[pos + 1]), hv(P[pos])
        add([pj, pjm1, hv(A_i[j]), hv(I[pos])],
            [1.0, -1.0, -1.0, 1.0], 0.0, f"ladder {j}")
        for (i, jj), U in pairvar.items():
            if jj == j:
                add([pj, pjm1, hv(U), hv(A_i[i] | I[pos])],
                    [1.0, -1.0, -1.0, 1.0], 0.0, f"pairlad {i},{j}")
        for (i1, i2, jj), U in trivar.items():
            if jj == j:
                add([pj, pjm1, hv(U), hv(A_i[i1] | A_i[i2] | I[pos])],
                    [1.0, -1.0, -1.0, 1.0], 0.0, f"trilad {i1},{i2},{j}")

    # ---- Shannon among kept sets --------------------------------------
    seen_row = set()
    for T in list(vidx):
        if len(T) < 2:
            continue
        for a_, b_ in itertools.combinations(sorted(T), 2):
            S = T - {a_, b_}
            Sa, Sb = T - {b_}, T - {a_}
            if S in vidx and Sa in vidx and Sb in vidx:
                key = (S, a_, b_)
                if key in seen_row:
                    continue
                seen_row.add(key)
                add([vidx[T], vidx[S], vidx[Sa], vidx[Sb]],
                    [1.0, 1.0, -1.0, -1.0], 0.0, "submod")
    anchors = [P[-1]] + P + list(pairvar.values()) + list(trivar.values())
    for B in list(vidx):
        for A0 in anchors:
            if B < A0:
                add([vidx[B], vidx[A0]], [1.0, -1.0], 0.0, "mono")
    for B in list(vidx):
        for e in B:
            if B - {e} in vidx:
                add([vidx[B - {e}], vidx[B]], [1.0, -1.0], 0.0, "mono")

    c = np.zeros(nvar)
    c[hv(query.group_attrs)] = -1.0
    if _dbg:
        print(f"  [dbg] lp rows={len(desc)} nvar={nvar}", flush=True)
    A = sparse.coo_matrix((vv, (ri, ci_)),
                        shape=(len(desc), nvar)).tocsr()
    return A, np.asarray(b), desc, c, hv, nvar, n_stat


def bound(query, db, pairs=False, triples=False, ctx=None):
    """Return (bound, lp_opt_bits, n_stat_rows)."""
    n = len(query.attrs)
    if n <= NMAX_FULL_LP:
        out = _build_lp(query, db, pairs=pairs, triples=triples, ctx=ctx)
    else:
        out = _build_lp_reduced(query, db, pairs=pairs, triples=triples,
                                ctx=ctx)
    if out[0] is None:
        return 0.0, -np.inf, 0
    A, b, desc, c, M, nvar, n_stat = out
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    return 2.0 ** (-res.fun), -res.fun, n_stat


def certify(query, db, pairs=False, triples=False, tol=1e-7, ctx=None):
    """Extract a PANDA-style proof certificate: the LP dual solution.

    Returns (bound, certificate) where certificate is a list of
    (description, dual_weight, rhs_bits).  By LP duality
        log2(bound) = sum_i dual_weight_i * rhs_i
    and every listed row is a named valid inequality, so the bound is
    reproducible as a nonnegative combination of verified facts.
    """
    if len(query.attrs) <= NMAX_FULL_LP:
        out = _build_lp(query, db, pairs=pairs, triples=triples, ctx=ctx)
    else:
        out = _build_lp_reduced(query, db, pairs=pairs, triples=triples,
                                ctx=ctx)
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
def encode_cols(rows, atom, attrs_, ctx=None):
    return _enc(rows, [atom.c[a] for a in attrs_], ctx)


def lower_bound(query, db, ctx=None):
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
            return _triangle_lb(atoms, db, ctx)

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
            return _spine_lb(atoms, si, db, ctx)
    return None


def _spine_lb(atoms, si, db, ctx=None):
    S = atoms[si]
    others = [atoms[k] for k in range(len(atoms)) if k != si]
    srows = filtered_rows(db, S, ctx)
    n = len(next(iter(srows.values())))
    sums = np.zeros(n, dtype=np.int64)
    allpos = np.ones(n, dtype=bool)
    for o in others:
        sha = sorted(S.attrs & o.attrs)
        orows = filtered_rows(db, o, ctx)
        ov, oc = np.unique(encode_cols(orows, o, sha, ctx), return_counts=True)
        deg_map = dict(zip(ov.tolist(), oc.tolist()))
        sk = encode([srows[S.c[a]] for a in sha])
        d = np.fromiter((deg_map.get(int(v), 0) for v in sk),
                        dtype=np.int64, count=n)
        sums += d
        allpos &= d > 0
    contrib = np.maximum(0, sums - (len(others) - 1))
    return int(contrib[allpos].sum())


def _triangle_lb(atoms, db, ctx=None):
    """|tri| >= sum_{(y,z) in E2} max(0, deg_1(y) + deg_3(z) - n_X)
    where deg_1(y) = |pi_X sigma_{Y=y} E1| etc. and n_X bounds the
    closing-attribute domain."""
    e1, e2, e3 = atoms
    sh12 = next(iter(e1.attrs & e2.attrs))          # Y
    sh23 = next(iter(e2.attrs & e3.attrs))          # Z
    sh13 = next(iter(e1.attrs & e3.attrs))          # X
    r1, r2, r3 = (filtered_rows(db, a, ctx) for a in atoms)
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
def _shared_key_truth(query, db, ctx=None):
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
        kcol = filtered_rows(db, atom, ctx)[atom.c[keyattr]]
        v, c = np.unique(kcol, return_counts=True)
        m = dict(zip(v.tolist(), c.tolist()))
        if acc is None:
            acc = m
        else:
            acc = {k: acc[k] * m[k] for k in acc.keys() & m.keys()}
    return int(sum(acc.values()))


def true_count(query, db, ctx=None):
    fast = _shared_key_truth(query, db, ctx)
    if fast is not None and set(query.group_attrs) == set(query.attrs):
        return fast
    return _true_count_merge(query, db, ctx)


def _true_count_merge(query, db, ctx=None):
    """Exact join size (bag semantics) via weight-propagating merge join.

    Intermediate state = dict attr -> np.ndarray plus weight '_w'; each
    step joins on shared attrs and re-contracts duplicate tuples into
    weights, so memory stays bounded by #distinct intermediate tuples.

    Row-id attributes (``__rowid__``) carry no join semantics -- they only
    multiply each tuple's weight -- so they are folded into ``_w`` at
    contract time and never materialized as join columns.  This keeps a
    star triple at O(#distinct keys) instead of O(|join result|).
    """
    atoms = [query.atoms[i] for i in _atom_order(list(query.atoms))]

    def rid_of(atom):
        return {a for a in atom.attrs if atom.c[a] == "__rowid__"}

    def keep_attrs(atom):
        return set(atom.attrs) - rid_of(atom)

    a0 = atoms[0]
    rows0 = filtered_rows(db, a0, ctx)
    cur = {a: rows0[a0.c[a]].copy() for a in keep_attrs(a0)}
    cur["_w"] = np.ones(len(next(iter(cur.values()), [])), dtype=np.int64)
    cur_attrs = keep_attrs(a0)
    cur = _contract(cur, cur_attrs)
    for atom in atoms[1:]:
        rows = filtered_rows(db, atom, ctx)
        rattrs = keep_attrs(atom)
        rcols = {a: rows[atom.c[a]] for a in rattrs}
        shared = cur_attrs & rattrs
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
        new_attrs = sorted(cur_attrs | rattrs)
        out = {a: [] for a in new_attrs}
        out["_w"] = []
        for c in range(len(common)):
            L = lorder[loff[il[c]]:loff[il[c] + 1]]
            Rr = rorder[roff[ir[c]]:roff[ir[c] + 1]]
            li_r = np.repeat(L, len(Rr))
            rj_r = np.tile(Rr, len(L))
            for a in cur_attrs:
                out[a].append(cur[a][li_r])
            for a in rattrs - cur_attrs:
                out[a].append(rcols[a][rj_r])
            out["_w"].append(cur["_w"][li_r])
        cur = {a: np.concatenate(v) if v else np.array([], dtype=np.int64)
               for a, v in out.items()}
        cur_attrs = set(new_attrs)
        if len(cur["_w"]) == 0:
            return 0
        cur = _contract(cur, cur_attrs)
    ga = set(query.group_attrs)
    nonrid = set().union(*(keep_attrs(a) for a in atoms))
    if (ga & nonrid) != nonrid:
        # projected count: distinct over the non-rid group attrs
        key = encode([cur[a] for a in sorted(ga & nonrid)])
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
