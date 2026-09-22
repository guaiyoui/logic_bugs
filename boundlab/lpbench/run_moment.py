"""Validate the moment/certificate framework on synthetic stars.

For each scenario we report
    truth        = sum_x prod d_i(x)         (exact)
    holder       = closed-form conjugate-Hoelder bound on ||deg_i||
    mcert-1      = certificate LP with univariate moments {p*e_i}
    mcert-1+2    = + pair moments {e_i+e_j}
    mcert-1+2+3  = + triple moment {e1+e2+e3}  (m=3)
    mcert-LB     = lower-side certificate LP (same moments)
    entropy-LP   = engine.bound for comparison

Prediction to verify:
  * mcert-1 <= holder  (LP only needs dominance on realized vectors)
  * mcert with the product moment alpha=(1,..,1) equals truth exactly
  * every UB >= truth, every LB <= truth
"""
import sys
import numpy as np

import data as D
import moment as MO
from engine import Atom, Query, bound


def degseq_from_rows(rows, key_col):
    k = rows[key_col]
    keys, deg = np.unique(k, return_counts=True)
    return keys, deg.astype(np.float64)


def star_case(name, dbs, key="k"):
    """dbs: list of row-dicts, each with a `key` column."""
    m = len(dbs)
    degseqs = [degseq_from_rows(r, key) for r in dbs]
    Df, mult = MO.degree_field(degseqs)
    truth = float((MO.product_values(Df) * mult).sum())
    ps = list(range(1, 11))
    uni = MO.with_support(MO.univariate_alphas(m, ps), m)
    pairs = uni + MO.pair_alphas(m)
    full = pairs + (MO.triple_alphas(m) if m >= 3 else [])

    ub1, c1, _ = MO.certificate_bound_separated(Df, mult, uni, degseqs)
    ub2, c2, _ = MO.certificate_bound_separated(Df, mult, pairs, degseqs)
    ub3 = None
    if m >= 3:
        ub3, _, _ = MO.certificate_bound_separated(Df, mult, full,
                                                   degseqs)
    lb1, _, _ = MO.certificate_bound_separated(Df, mult, pairs, degseqs,
                                               side="lower")
    ho = MO.holder_bound(degseqs, ps=ps)

    def ok(b):
        return "OK" if b is None or b >= truth - 1e-6 else "*** VIOL ***"
    def okl(b):
        return "OK" if b is None or b <= truth + 1e-6 else "*** LB VIOL ***"
    def qr(b):
        return f"{b/truth:8.2f}" if truth else ("inf" if b else "1.00*")

    print(f"{name:<16} truth={truth:>13,.0f}  K={len(Df)}")
    print(f"  {'holder(closed)':<18} {ho:>16,.0f}  q={qr(ho)} {ok(ho)}")
    print(f"  {'mcert univar':<18} {ub1:>16,.0f}  q={qr(ub1)} {ok(ub1)}")
    if ub2 is not None:
        print(f"  {'mcert +pairs':<18} {ub2:>16,.0f}  q={qr(ub2)} {ok(ub2)}")
    if ub3 is not None:
        print(f"  {'mcert +triples':<18} {ub3:>16,.0f}  q={qr(ub3)} {ok(ub3)}")
    print(f"  {'mcert LB':<18} {lb1:>16,.0f}  ratio={qr(lb1)} {okl(lb1)}")
    return truth, ub1, ub2, ho


def main():
    sets = D.synth_sets()
    print("=== m = 2 stars ===")
    for ds in ("sym", "anti_rank", "anti_dom", "asym", "unif"):
        db = sets[ds]
        sname = "T" if "T" in db else "S"
        star_case(f"j2/{ds}", [db["R"], db[sname]])
        # entropy-LP comparison
        q = Query([Atom("R", {"X": "k", "Y": "v"}),
                   Atom(sname, {"X": "k", "Z": "v"})])
        b0, _, _ = bound(q, db)
        bp, _, _ = bound(q, db, pairs=True)
        print(f"  {'entropy lpbound':<18} {b0:>16,.0f}    "
              f"{'entropy +pairs':<14} {bp:,.0f}")

    print("=== m = 3 stars ===")
    for ds in ("sym", "anti_rank"):
        db = dict(sets[ds]); db["T"] = db["S"]
        star_case(f"star3/{ds}", [db["R"], db["S"], db["T"]])
        q = Query([Atom("R", {"X": "k", "Y": "v"}),
                   Atom("S", {"X": "k", "Z": "v"}),
                   Atom("T", {"X": "k", "W": "v"})])
        b0, _, _ = bound(q, db)
        bp, _, _ = bound(q, db, pairs=True)
        bt, _, _ = bound(q, db, pairs=True, triples=True)
        print(f"  {'entropy lpbound':<18} {b0:>16,.0f}    "
              f"+pairs {bp:,.0f}   +triples {bt:,.0f}")


if __name__ == "__main__":
    main()
