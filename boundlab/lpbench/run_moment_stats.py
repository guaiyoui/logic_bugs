"""Moment-certificate bounds on real STATS queries (single-key stars).

Checks, per query:
  * mcert univar   vs  entropy lpbound   (theory: equal = Hoelder section)
  * mcert +pairs   vs  entropy +pairs    (theory: equal)
  * mcert +triples vs  entropy +triples
  * holder_gap     vs  actual lpbound->pairs improvement (predictor)
"""
import os, sys, numpy as np, pandas as pd
import sql, dbload, bench, moment as MO
from engine import filtered_rows

QFILE = "data_dl/lpbound_repo/benchmarks/workloads/stats/statsQueries.sql"
CSV = "results/bench_stats_dense.csv"
PS = list(range(1, 11))
MAXM = int(os.environ.get("MAXM", "4"))
MAXQ = int(os.environ.get("MAXQ", "0")) or None
STARTQ = int(os.environ.get("STARTQ", "0"))


def is_single_key_star(q):
    classes = {}
    for a in q.atoms:
        for at in a.attrs:
            if not at.endswith("#"):
                classes.setdefault(at, []).append(a)
    multi = {k: v for k, v in classes.items() if len(v) > 1}
    return (len(multi) == 1 and
            all(len(a.attrs) == 2 for a in q.atoms)), multi


_DB_CACHE = {}
_COLS = {}
_KEYS = {}


def _db(p):
    """Load tables once; reload a table only when the query asks for a
    column not seen before (loaded column-union)."""
    need = bench.needed_of([p])
    reload_cols, reload_keys = {}, {}
    for t, (cols, keys) in need.items():
        want = set(cols) | set(keys)
        if want - _COLS.get(t, set()):
            reload_cols[t] = want | _COLS.get(t, set())
            reload_keys[t] = set(keys) | _KEYS.get(t, set())
    if reload_cols:
        db, _ = dbload.load_stats_raw(reload_cols, reload_keys)
        for t, m_ in db.items():
            _DB_CACHE.setdefault(t, {}).update(m_)
            _COLS[t] = reload_cols[t]
            _KEYS[t] = reload_keys[t]
    return _DB_CACHE


def main():
    ref = pd.read_csv(CSV).set_index("q")
    lines = [l for l in open(QFILE).read().splitlines() if l.strip()]
    out = []
    for qi, line in enumerate(lines):
        name = f"stats.{qi + 1}"
        try:
            p = sql.parse_flat(line)
        except Exception:
            continue
        q = sql.to_query(p)
        ok, multi = is_single_key_star(q)
        m = len(q.atoms)
        if not ok or m > MAXM or m < 2:
            continue
        idx = qi + 1
        if idx <= STARTQ or (MAXQ and len(out) >= MAXQ):
            continue
        db = _db(p)
        X = next(iter(multi))
        ctx = {}
        degseqs = []
        for a in q.atoms:
            rows = filtered_rows(db, a, ctx)
            ks, dv = np.unique(rows[a.c[X]], return_counts=True)
            degseqs.append((ks, dv.astype(np.float64)))
        Df, mult = MO.degree_field(degseqs)
        truth_mc = float((MO.product_values(Df) * mult).sum())
        uni = MO.with_support(MO.univariate_alphas(m, PS), m)
        prs = uni + MO.pair_alphas(m)
        ub1, _, _ = MO.certificate_bound_separated(Df, mult, uni, degseqs)
        ub2, _, _ = MO.certificate_bound_separated(Df, mult, prs, degseqs)
        ub3 = None
        if m >= 3:
            tr = prs + MO.triple_alphas(m)
            ub3, _, _ = MO.certificate_bound_separated(Df, mult, tr,
                                                       degseqs)
        ho = MO.holder_bound(degseqs)
        r = ref.loc[name] if name in ref.index else None
        truth_db = float(r["truth"]) if r is not None and r["truth"] > 0 else np.nan
        rec = dict(q=name, m=m, K=len(Df), truth=truth_db,
                   truth_mc=truth_mc,
                   holder=ho,
                   mc_uni=ub1, mc_pairs=ub2, mc_tri=ub3,
                   lpbound=(r["lpbound"] if r is not None else np.nan),
                   epairs=(r["pairs"] if r is not None else np.nan),
                   etriples=(r["triples"] if r is not None else np.nan))
        out.append(rec)
        viol = "VIOL" if (not np.isnan(truth_db) and
                          min(x for x in (ub1, ub2, ub3) if x is not None)
                          < truth_db - 1) else "ok"
        print(f"{name}: m={m} truth={truth_db:,.0f} mc={truth_mc:,.0f} "
              f"holder={ho:,.0f} uni={ub1:,.0f} pair={ub2:,.0f} "
              f"tri={ub3 if ub3 else float('nan'):,.0f} [{viol}]",
              flush=True)
    df = pd.DataFrame(out)
    df.to_csv("results/moment_stats.csv", index=False)
    ok = df[df["truth"] > 0]
    for col in ("holder", "mc_uni", "mc_pairs", "mc_tri", "lpbound",
                "epairs", "etriples"):
        v = ok[col].dropna()
        qe = (v / ok.loc[v.index, "truth"]).to_numpy()
        print(f"== {col}: geomean={np.exp(np.log(qe).mean()):.3f} "
              f"median={np.median(qe):.3f} viol={(qe < 0.999999).sum()} "
              f"n={len(v)}")
    # does the Hoelder gap predict the pair-arm improvement?
    both = ok.dropna(subset=["holder", "epairs", "truth_mc"])
    hg = np.log(both["holder"] / both["truth_mc"])
    imp = np.log(both["lpbound"] / both["epairs"])
    print(f"== corr(log holder_gap, log entropy improvement) = "
          f"{np.corrcoef(hg, imp)[0,1]:.4f}  on n={len(both)}")
    print("wrote results/moment_stats.csv")


if __name__ == "__main__":
    main()
