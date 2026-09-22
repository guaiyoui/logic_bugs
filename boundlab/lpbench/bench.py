"""Real-benchmark driver: JOB-light / JOB-join / JOB-range / STATS / DBLP.

Arms per query: lpbound | +pairs | (+triples) | truth (duckdb) | LB.
Aggregates: geomean q-error, win/tie counts, violation counts.
"""
from __future__ import annotations

import glob
import os
import re
import sys
import threading
import time

import duckdb
import numpy as np
import pandas as pd

import dbload
from engine import bound, lower_bound
from sql import parse_flat, to_query

D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(D, "data_dl", "lpbound_repo")
WL = os.path.join(REPO, "benchmarks", "workloads")
IMDB_DIR = os.path.join(D, "data_dl", "imdb")


# ---------------------------------------------------------------- queries
def queries_of(workload, max_q=None, which=None, start=0):
    out = []
    if workload.startswith("dblp"):
        pat = os.path.join(WL, "subgraph_matching", "dblp",
                           f"dblp_query_{which}_8_*.sql")
        for f in sorted(glob.glob(pat)):
            out.append((os.path.basename(f)[:-4],
                        open(f).read().strip()))
    else:
        qf = {"joblight": "joblight/joblightQueries.sql",
              "jobjoin": "jobjoin/jobjoinQueries.sql",
              "jobrange": "jobrange/jobrangeQueries.sql",
              "stats": "stats/statsQueries.sql"}[workload]
        for i, line in enumerate(
                open(os.path.join(WL, qf)).read().splitlines(), 1):
            line = line.strip()
            if line:
                out.append((f"{workload}.{i}", line))
    out = out[start:]
    return out[:max_q] if max_q else out


def needed_of(parsed_list):
    """table -> (key cols, pred cols) over all queries."""
    need = {}
    for p in parsed_list:
        for a, colmap in p["atoms"].items():
            t = p["aliases"][a]
            k, pr = need.setdefault(t, (set(), set()))
            k.update(colmap.values())
        for a, preds in p["pred_by"].items():
            t = p["aliases"][a]
            k, pr = need.setdefault(t, (set(), set()))
            pr.update(col for col, _, _ in preds)
    return {t: (k | pr, k) for t, (k, pr) in need.items()}


# ------------------------------------------------------------------ duckdb
def make_con(workload, raw_tables):
    con = duckdb.connect()
    for t, df in raw_tables.items():
        if "." in t:
            sch, tn = t.split(".", 1)
            con.execute(f'CREATE SCHEMA IF NOT EXISTS "{sch}"')
            tmp = f"__tmp_{sch}_{tn}"
            con.register(tmp, df)
            con.execute(f'CREATE TABLE "{sch}"."{tn}" AS SELECT * FROM "{tmp}"')
        else:
            con.register(t, df)
    return con


def duck_count(con, sql, timeout=120):
    """COUNT(*) of the original query; watchdog interrupts on timeout.
    `at` is a reserved word in DuckDB but a legal alias in Postgres --
    rename that alias when it occurs."""
    sql = re.sub(r"\bAS\s+at\b", "AS at_", sql.strip().rstrip(";"))
    sql = re.sub(r"\bat\.", "at_.", sql)
    q = f"SELECT COUNT(*) FROM ({sql})"
    res = {}

    def run():
        try:
            res["n"] = con.execute(q).fetchone()[0]
        except Exception as e:                      # noqa: BLE001
            res["err"] = str(e)[:120]

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        con.interrupt()
        th.join(30)
        return None, "timeout"
    return res.get("n"), res.get("err")


def _propagate_vertex_labels(q, parsed, db):
    """DBLP: fold each vertex atom's label predicate into edge atoms as
    endpoint membership filters (LpBound-style predicate propagation).
    attr X gets allowed-id set = intersection of labels on that attr."""
    vtab = db["dblp.vertex"]
    allow = {}                                  # attr -> set of ids
    for a, preds in parsed["pred_by"].items():
        if parsed["aliases"][a] != "dblp.vertex":
            continue
        attr = next(iter(parsed["atoms"][a]))
        cur = np.ones(len(vtab["i"]), dtype=bool)
        for col, op, v in preds:
            if col == "l" and op == "=":
                cur &= vtab["l"] == v
        s = set(vtab["i"][cur].tolist())
        allow[attr] = s if attr not in allow else allow[attr] & s
    for atom in q.atoms:
        if atom.table != "dblp.edge":
            continue
        masks = {A: allow.get(A) for A in atom.attrs}
        if any(v is not None for v in masks.values()):
            cols = dict(atom.c)
            def pf(rows, masks=masks, cols=cols):
                m = np.ones(len(rows["s"]), dtype=bool)
                for A, ids in masks.items():
                    if ids is not None:
                        m &= np.isin(rows[cols[A]], list(ids))
                return m
            atom.pred = pf


# -------------------------------------------------------------------- main
def main():
    workload = sys.argv[1] if len(sys.argv) > 1 else "joblight"
    max_q = int(os.environ.get("MAXQ", "0")) or None
    which = os.environ.get("WHICH", "dense")
    timeout = int(os.environ.get("TQ", "120"))
    arms = os.environ.get("ARMS", "lpbound,pairs,lb").split(",")
    start = int(os.environ.get("STARTQ", "0"))

    qs = queries_of(workload, max_q, which, start)
    print(f"workload={workload} queries={len(qs)}", flush=True)

    parsed, skipped_q = [], []
    for name, sql in qs:
        try:
            p = parse_flat(sql)
        except Exception as e:                      # noqa: BLE001
            skipped_q.append((name, f"parse: {e}"))
            continue
        if p["skipped"] or p["like"]:
            skipped_q.append((name,
                f"unsupported conds={p['skipped']} like={p['like']}"))
            continue
        parsed.append((name, sql, p))
    print(f"parsed={len(parsed)} skipped={len(skipped_q)}", flush=True)
    for n, why in skipped_q[:5]:
        print(f"  skip {n}: {why}")

    need = needed_of([p for _, _, p in parsed])
    key_cols = {t: v[1] for t, v in need.items()}
    all_cols = {t: v[0] for t, v in need.items()}

    t0 = time.time()
    if workload == "dblp":
        db, raw = dbload.load_dblp()
    elif workload == "stats":
        db, raw = dbload.load_stats_raw(all_cols, key_cols)
    else:
        db, raw = dbload.load_imdb(all_cols, key_cols, IMDB_DIR)
    print(f"db loaded in {time.time()-t0:.0f}s "
          f"({len(db)} tables)", flush=True)

    con = make_con(workload, raw)

    rows = []
    vtab = db["dblp.vertex"] if workload == "dblp" else None
    label_ids = {}
    if vtab is not None:
        for l in np.unique(vtab["l"]):
            label_ids[int(l)] = set(vtab["i"][vtab["l"] == l].tolist())

    for qi, (name, sql, p) in enumerate(parsed):
        q = to_query(p)
        if workload == "dblp":
            _propagate_vertex_labels(q, p, db)
        rec = {"q": name, "atoms": len(q.atoms), "attrs": len(q.attrs)}
        ctx = {}
        try:
            if "lpbound" in arms:
                t1 = time.time()
                rec["lpbound"] = bound(q, db, ctx=ctx)[0]
                rec["t_lp"] = time.time() - t1
            if "pairs" in arms:
                t1 = time.time()
                rec["pairs"] = bound(q, db, pairs=True, ctx=ctx)[0]
                rec["t_p"] = time.time() - t1
            if "triples" in arms:
                t1 = time.time()
                rec["triples"] = bound(q, db, pairs=True, triples=True,
                                       ctx=ctx)[0]
                rec["t_tr"] = time.time() - t1
            if "lb" in arms:
                t1 = time.time(); rec["lb"] = lower_bound(q, db, ctx=ctx)
                rec["t_lb"] = time.time() - t1
            t1 = time.time()
            if timeout <= 0:
                rec["truth"], err = None, "skipped"
            else:
                rec["truth"], err = duck_count(con, sql, timeout)
            rec["t_t"] = time.time() - t1
            if err:
                rec["terr"] = err
        except Exception as e:                      # noqa: BLE001
            rec["err"] = str(e)[:150]
        rows.append(rec)
        if rec.get("truth"):
            msg = f"[{qi+1}/{len(parsed)}] {name}: truth={rec['truth']:,.0f}"
            for a in ("lpbound", "pairs", "triples"):
                if rec.get(a):
                    msg += f" {a}={rec[a]:,.0f}({rec[a]/rec['truth']:7.2f})"
            msg += f" lb={rec.get('lb')}"
            print(msg, flush=True)
        else:
            print(f"[{qi+1}/{len(parsed)}] {name}: "
                  f"{rec.get('err') or rec.get('terr') or 'no truth'}",
                  flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(D, "results", f"bench_{workload}_{which}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    ok = df.dropna(subset=["truth"])
    ok = ok[ok["truth"] > 0]
    if len(ok):
        for arm in ("lpbound", "pairs", "triples"):
            if arm not in ok:
                continue
            qe = (ok[arm] / ok["truth"]).to_numpy()
            print(f"== {arm}: geomean q-err={np.exp(np.log(qe).mean()):.3f} "
                  f"median={np.median(qe):.3f} p90={np.percentile(qe,90):.2f} "
                  f"max={qe.max():.1f} exact={np.mean(qe<=1.000001):.0%} "
                  f"viol={np.sum(qe < 0.999999)}")
        if {"lpbound", "pairs"} <= set(ok.columns):
            imp = (ok["pairs"] < ok["lpbound"] * 0.999).mean()
            print(f"== +pairs strictly tighter on {imp:.0%} of queries")
        if "triples" in ok.columns:
            imp3 = (ok["triples"] < ok["pairs"] * 0.999).mean()
            print(f"== +triples strictly tighter on {imp3:.0%}")
        if "lb" in ok.columns:
            lbok = ok["lb"].notna()
            print(f"== LB emitted on {lbok.mean():.0%}, "
                  f"viol={(ok.loc[lbok,'lb'] > ok.loc[lbok,'truth']).sum()}")
    elif {"lpbound", "pairs"} <= set(df.columns):
        b = df.dropna(subset=["lpbound", "pairs"])
        if len(b):
            r = (b["pairs"] / b["lpbound"]).to_numpy()
            eq = np.isclose(r, 1.0, rtol=1e-3).mean()
            print(f"== bound-only (no truth): pairs/lpbound ratio "
                  f"geomean={np.exp(np.log(r).mean()):.3f} "
                  f"ties={eq:.0%} n={len(b)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
