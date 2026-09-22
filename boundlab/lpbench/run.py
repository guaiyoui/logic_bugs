"""Evaluation: LpBound (faithful reimplementation) vs ours (+pair marginals).

Metric: q-error = bound / true_count  (>= 1 required; we assert it).
Arms:
  lpbound   - full per-relation lp-norm stats, p in {1..10,inf}
  ours      - same LP + exact pairwise join-count constraints
"""
from __future__ import annotations

import sys
import time
import numpy as np

from engine import Atom, Query, bound, true_count, lower_bound, certify
import data as D


def _ok(b, truth):
    return "OK" if b >= truth * (1 - 1e-9) - 1e-6 else "*** VIOLATED ***"


def row(name, q, db, use_triples=False):
    t0 = time.time()
    lb, _, _ = bound(q, db)
    t1 = time.time()
    ob, _, _ = bound(q, db, pairs=True)
    t2 = time.time()
    tt = None
    if use_triples:
        tt, _, _ = bound(q, db, pairs=True, triples=True)
    t3 = time.time()
    truth = true_count(q, db)
    t4 = time.time()
    lo = lower_bound(q, db)
    t5 = time.time()
    lo_ok = lo is None or lo <= truth
    parts = [f"{name:<22} truth={truth:>12,.0f}  "
             f"lpbound={lb:>14,.0f} (q-err {lb/max(truth,1):>8.2f}) {_ok(lb, truth)}  "
             f"+pairs={ob:>14,.0f} (q-err {ob/max(truth,1):>8.2f}) {_ok(ob, truth)}"]
    if tt is not None:
        parts.append(f"+triples={tt:>14,.0f} (q-err {tt/max(truth,1):>8.2f}) {_ok(tt, truth)}")
    parts.append(f"LB={lo if lo is not None else float('nan'):>12} "
                 f"{'OK' if lo_ok else '*** LB VIOLATED ***'}")
    parts.append(f"[b {t1-t0:.1f}s p {t2-t1:.1f}s t3 {t3-t2:.1f}s "
                 f"truth {t4-t3:.1f}s lb {t5-t4:.1f}s]")
    print("  ".join(parts), flush=True)
    return lb, ob, tt, truth, lo


def main(which):
    sets = D.synth_sets()

    if which in ("synth", "all"):
        print("=== synthetic relational: J2 = R(X,Y) |x|_X S(X,Z) ===")
        for ds in ("sym", "anti_rank", "anti_dom", "asym", "unif"):
            db = sets[ds]
            sname = "T" if "T" in db else "S"
            q = Query([Atom("R", {"X": "k", "Y": "v"}),
                       Atom(sname, {"X": "k", "Z": "v"})])
            row(f"j2/{ds}", q, db)

        # J2 with range predicate on R.p
        db = sets["sym"]
        qp = Query([Atom("R", {"X": "k", "Y": "v"},
                         pred=lambda r: (r["p"] >= 10) & (r["p"] < 40)),
                    Atom("S", {"X": "k", "Z": "v"})])
        row("j2/sym+range[10,40)", qp, db)

        # 3-star: R(X,Y) |x|_X S(X,Z) |x|_X T(X,W)
        for ds in ("sym", "anti_rank"):
            db = dict(sets[ds]); db["T"] = db["S"]
            q = Query([Atom("R", {"X": "k", "Y": "v"}),
                       Atom("S", {"X": "k", "Z": "v"}),
                       Atom("T", {"X": "k", "W": "v"})])
            row(f"star3/{ds}", q, db)

    if which in ("j3", "all"):
        print("=== synthetic 3-chain R(X,Y)-S(Y,Z)-T(Z,U) ===")
        for anti in (False, True):
            db3 = D.chain3(anti=anti)
            q3 = Query([Atom("R", {"X": "x", "Y": "y"}),
                        Atom("S", {"Y": "y", "Z": "z"}),
                        Atom("T", {"Z": "z", "U": "u"})])
            row("j3/" + ("z-anticorr" if anti else "corr"), q3, db3)

    if which in ("j4", "all"):
        print("=== synthetic 4-chain R-S-T-V (triples arm) ===")
        db4 = D.chain4()
        q4 = Query([Atom("R", {"X": "x", "Y": "y"}),
                    Atom("S", {"Y": "y", "Z": "z"}),
                    Atom("T", {"Z": "z", "U": "u"}),
                    Atom("V", {"U": "u", "W": "w"})])
        row("j4/z-anticorr", q4, db4, use_triples=True)

        # certificate demo: which inequalities prove the bound?
        qb = bound(q4, db4, pairs=True)[0]
        bd, cert, fun, lhs = certify(q4, db4, pairs=True)
        print(f"    certificate for j4+pairs: bound={bd:,.0f}, "
              f"reconstructed={2.0 ** lhs:,.0f}  "
              f"({'consistent' if abs(2 ** lhs - bd) / bd < 1e-9 else 'MISMATCH'})")
        agg = {}
        for d, lam, rhs in cert:
            key = d if not d.startswith("shannon") else "shannon"
            agg[key] = agg.get(key, 0) + 1
        print("    active constraints:",
              ", ".join(f"{k} x{v}" for k, v in sorted(agg.items())))

    if which in ("graph", "all"):
        for und in (False, True):
            tag = "undirected" if und else "directed"
            print(f"=== cit-Patents slice ({tag}, subgraph matching) ===")
            db = D.load_patents(max_edges=3_000_000, undirected=und)
            _graph_block(db, tag)


def _graph_block(db, tag):
        E = db["E"]
        s, d = E["s"], E["d"]
        nnode = int(max(s.max(), d.max())) + 1
        indeg = np.bincount(d, minlength=nnode)
        outdeg = np.bincount(s, minlength=nnode)
        # CSR adjacency for triangle counting
        order = np.argsort(s, kind="stable")
        so, do = s[order], d[order]
        out_off = np.searchsorted(so, np.arange(len(indeg) + 1))
        order_i = np.argsort(d, kind="stable")
        di, si_ = d[order_i], s[order_i]
        in_off = np.searchsorted(di, np.arange(len(indeg) + 1))

        def out_nbrs(x):
            return do[out_off[x]:out_off[x + 1]]

        def in_nbrs(x):
            return si_[in_off[x]:in_off[x + 1]]

        def tri_truth():
            cnt = 0
            for x, y in zip(s, d):
                cnt += len(np.intersect1d(out_nbrs(y), in_nbrs(x),
                                          assume_unique=True))
            return cnt

        def claw_truth():
            dd = outdeg.astype(np.int64)
            return int((dd * (dd - 1) * (dd - 2)).sum())

        def path3_truth():
            return int((indeg[s].astype(np.int64) *
                        outdeg[d].astype(np.int64)).sum())

        def path4_truth():
            # sum_{(z,w) in E} p2end(z) * outdeg(w),
            # p2end(z) = sum_{(y,z) in E} indeg(y)
            p2end = np.zeros(nnode, dtype=np.int64)
            np.add.at(p2end, d, indeg[s])
            outsum = np.zeros(nnode, dtype=np.int64)
            np.add.at(outsum, s, outdeg[d])
            return int((p2end * outsum).sum())

        for name, q, tf, tr in [
            (f"p2/{tag}", Query([Atom("E", {"X": "s", "Y": "d"}),
                                Atom("E", {"Y": "s", "Z": "d"})]),
             lambda: int((indeg.astype(np.int64) *
                          outdeg.astype(np.int64)).sum()), False),
            (f"tri/{tag}", Query([Atom("E", {"X": "s", "Y": "d"}),
                                 Atom("E", {"Y": "s", "Z": "d"}),
                                 Atom("E", {"Z": "s", "X": "d"})]), tri_truth, False),
            (f"claw3/{tag}", Query([Atom("E", {"X": "s", "Y": "d"}),
                                   Atom("E", {"X": "s", "Z": "d"}),
                                   Atom("E", {"X": "s", "W": "d"})]), claw_truth, False),
            (f"path3/{tag}", Query([Atom("E", {"X": "s", "Y": "d"}),
                                   Atom("E", {"Y": "s", "Z": "d"}),
                                   Atom("E", {"Z": "s", "W": "d"})]), path3_truth, False),
            (f"path4/{tag}", Query([Atom("E", {"X": "s", "Y": "d"}),
                                   Atom("E", {"Y": "s", "Z": "d"}),
                                   Atom("E", {"Z": "s", "W": "d"}),
                                   Atom("E", {"W": "s", "V": "d"})]), path4_truth, True),
        ]:
            t0 = time.time(); lb_, _, _ = bound(q, db)
            t1 = time.time(); ob, _, _ = bound(q, db, pairs=True)
            tt = None
            if tr:
                t1b = time.time()
                tt, _, _ = bound(q, db, pairs=True, triples=True)
                t2 = time.time()
            else:
                t2 = time.time()
            truth = tf(); t3 = time.time()
            lo = lower_bound(q, db); t4 = time.time()
            lo_ok = lo is None or lo <= truth
            parts = [f"{name:<22} truth={truth:>12,.0f}  "
                     f"lpbound={lb_:>14,.0f} (q-err {lb_/max(truth,1):>8.2f}) {_ok(lb_, truth)}  "
                     f"+pairs={ob:>14,.0f} (q-err {ob/max(truth,1):>8.2f}) {_ok(ob, truth)}"]
            if tt is not None:
                parts.append(f"+triples={tt:>14,.0f} (q-err {tt/max(truth,1):>8.2f}) {_ok(tt, truth)}")
            parts.append(f"LB={lo if lo is not None else float('nan'):>12} "
                         f"{'OK' if lo_ok else '*** LB VIOLATED ***'}")
            parts.append(f"[b {t1-t0:.1f}s p {t2-t1:.1f}s t {t3-t2:.1f}s lb {t4-t3:.1f}s]")
            print("  ".join(parts), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
