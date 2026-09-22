"""Oracle C: data-homomorphism invariants.

The same query evaluated on a homomorphically transformed database must
return a homomorphically transformed result.  Two instances:

Mode "map" — injective affine value map
    int -> 7*x + 11, txt -> 'q' + x, bool -> id, NULL -> NULL
    Constants inside predicates are remapped the same way, so every
    generated query is closed under the map.  Expected relation:
        Q(f(D)) == f(Q(D))     (bag equality after cell-wise remap)
    Aggregate laws: COUNT*/COUNT(col)/COUNT(DISTINCT) invariant,
    SUM/MIN/MAX transform affinely, AVG affinely with tolerance.
    Rationale: value-regime-dependent bugs (e.g. the ltree >14653-label
    comparison bug) only fire past a value boundary — this oracle walks
    queries across regime boundaries while keeping the same plan shape.

Mode "dup" — k-plex row duplication
    Every row of every table duplicated k times.  Laws:
      plain SELECT ... FROM <n dup'd tables>   -> multiplicities * k^n
      DISTINCT / GROUP BY keys / set-ops        -> invariant (set)
      UNION ALL / INTERSECT ALL / EXCEPT ALL    -> multiplicities * k
      COUNT(*) / COUNT(c) / SUM                 -> * k
      MIN / MAX / AVG / COUNT(DISTINCT)         -> invariant
      LEFT JOIN present                         -> set-compare fallback
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
COEVO_ROOT = "/home/user/work/db_safety/co_evolution_db_testing"
sys.path.insert(0, COEVO_ROOT)

from absoracle.common import (  # noqa: E402
    ColRef, DuckRunner, PGSimple, Pred, Query, QueryGen, Table, bag,
    ddl_and_inserts, gen_db,
)
from oracles.normalize import normalize_rows  # noqa: E402


# ---------------------------------------------------------------- value map
INT_SCALE = 1009  # pure scale: SUM/AVG laws stay multiplicative


def fval(v, typ: str):
    if v is None:
        return None
    if typ == "int":
        return INT_SCALE * int(v)
    if typ == "txt":
        return "q" + str(v)
    return v


def _map_side(x):
    if isinstance(x, ColRef):
        return x
    v, typ = x
    return (fval(v, typ), typ)


def map_pred(p: Pred) -> Pred:
    q = copy.copy(p)
    if p.kind == "cmp":
        q.left, q.right = _map_side(p.left), _map_side(p.right)
    elif p.kind == "isnull":
        pass
    elif p.kind == "in":
        if isinstance(p.right, Query):
            q.right = map_query(p.right)
        else:
            q.right = [fval(v, p.left.typ) for v in p.right]
    elif p.kind == "exists":
        q.right = map_query(p.right)
    else:
        q.children = [map_pred(c) for c in p.children]
    return q


def map_query(q: Query) -> Query:
    m = copy.copy(q)
    if q.setop is not None:
        m.setop = copy.copy(q.setop)
        m.setop.left = map_query(q.setop.left)
        m.setop.right = map_query(q.setop.right)
        return m
    m.where = map_pred(q.where) if q.where else None
    m.having = map_pred(q.having) if q.having else None
    return m


def map_tables(tables: list[Table], k: int | None = None) -> list[Table]:
    out = []
    for t in tables:
        if k is None:  # value map
            rows = [[fval(v, c.typ) for v, c in zip(row, t.cols)]
                    for row in t.rows]
        else:          # row duplication
            rows = [list(r) for r in t.rows for _ in range(k)]
        out.append(Table(t.name, t.cols, rows))
    return out


def map_cell(proj, v):
    """Expected transform of one result cell under the value map."""
    if v is None:
        return None
    if proj.kind == "col":
        return fval(v, proj.ref.typ)
    fn = proj.fn
    if fn in ("count", "countd"):
        return v
    rt = proj.ref.typ
    if fn in ("sum", "min", "max"):
        if rt == "int":
            return INT_SCALE * float(v) if isinstance(v, float)                 else INT_SCALE * int(v)
        if rt == "txt":
            return "q" + str(v)
        return v
    if fn == "avg":
        return round(INT_SCALE * float(v), 4)
    return v


def dup_cell(proj, v, k: int):
    """k here is already k^deg (per-group input-row scaling)."""
    if v is None:
        return None
    if proj.kind == "col":
        return v
    if proj.fn in ("count", "sum"):
        try:
            return int(v) * k
        except (TypeError, ValueError):
            return float(v) * k
    return v  # min, max, avg, countd


def expected_rows(rows, projs, cell_fn) -> list[list]:
    out = []
    for row in rows:
        out.append([cell_fn(p, v) for p, v in zip(projs, row)])
    return out


_SET_OPS = {"union", "intersect", "except"}
_ALL_OPS = {"unionall", "intersectall", "exceptall"}


def _arm_plain(a: Query) -> bool:
    return (not a.groupby and not a.distinct
            and not any(p.fn for p in a.projs))


def dup_mult(q: Query, k: int) -> int | None:
    """Per-row multiplicity factor; None => set-compare; -1 => skip case."""
    if q.setop is not None:
        # arms share the same skeleton (cloned joins) so dup degree is uniform
        if not all(_arm_plain(a) for a in (q.setop.left, q.setop.right)):
            return -1
        if q.setop.op in _SET_OPS:
            return 1
        # LEFT JOIN arms produce null-extended rows that scale by k, not
        # k^deg — multiplicity is mixed, so compare sets only.
        if any(kind == "left" for a in (q.setop.left, q.setop.right)
               for *_x, kind in a.joins):
            return None
        return k ** len(q.setop.left.aliases)
    if q.distinct or q.groupby or any(p.fn for p in q.projs):
        return 1
    if any(kind == "left" for *_x, kind in q.joins):
        return None
    return k ** len(q.aliases)


def compare_bags(label, got_rows, want_rows, diffs, set_only=False,
                 want_mult: int = 1, avg_cols: list[int] | None = None):
    got = normalize_rows(got_rows)
    want = normalize_rows(want_rows)
    if avg_cols:
        def quant(rows):
            return [[round(v, 3) if i in avg_cols and
                     isinstance(v, float) else v
                     for i, v in enumerate(r)] for r in rows]
        got, want = quant(got), quant(want)
    if set_only:
        g = set(json.dumps(r) for r in got)
        w = set(json.dumps(r) for r in want)
    else:
        g = Counter(json.dumps(r) for r in got)
        w = Counter(json.dumps(r) for r in want)
        if want_mult != 1:
            w = Counter({k2: v * want_mult for k2, v in w.items()})
    if g != w:
        diffs.append({"check": label, "kind": "result_diff",
                      "got": got[:8], "want": want[:8],
                      "got_n": len(got_rows), "want_n": len(want_rows)})


def run_case(runner, dialect, rng, mode: str, k: int = 2) -> dict | None:
    tables = gen_db(rng)
    gen = QueryGen(rng, tables)
    q = gen.gen(allow_limit=(mode == "map"), allow_avg=(mode != "map"))
    sql = q.sql()

    if mode == "map":
        t2 = map_tables(tables)
        q2 = map_query(q)
        projs = q.projs
    else:
        t2 = map_tables(tables, k=k)
        q2 = q
        projs = q.projs

    runner.setup(ddl_and_inserts(tables, dialect))
    r1, e1 = runner.run(sql)
    runner.setup(ddl_and_inserts(t2, dialect))
    r2, e2 = runner.run(q2.sql())
    if e1 or e2:
        if (e1 or "").split(":", 1)[0] != (e2 or "").split(":", 1)[0]:
            return {"sql": sql, "mode": mode,
                    "diffs": [{"kind": "error_flip",
                               "e1": e1, "e2": e2}]}
        return None

    diffs: list = []
    if mode == "map":
        exp = expected_rows(r1, projs, map_cell)
        avg_cols = [i for i, p in enumerate(projs)
                    if p.fn == "avg"]
        compare_bags("map", r2, exp, diffs, avg_cols=avg_cols)
    else:
        mult = dup_mult(q, k)
        if mult == -1:
            return None
        has_agg = q.groupby or any(p.fn for p in q.projs)
        if has_agg and any(k2 == "left" for *_x, k2 in q.joins):
            return None  # mixed multiplicity for null-extended groups
        deg = len(q.aliases) if q.setop is None \
            else len(q.setop.left.aliases)
        cell_k = k ** deg
        exp = expected_rows(r1, projs, lambda p, v: dup_cell(p, v, cell_k))
        compare_bags("dup", r2, exp, diffs,
                     set_only=(mult is None), want_mult=mult or 1)
    if diffs:
        return {"sql": sql, "mode": mode, "k": k,
                "setup": ddl_and_inserts(tables, dialect), "diffs": diffs}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["pg", "duck"], required=True)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mode", choices=["map", "dup", "both"],
                    default="both")
    ap.add_argument("--out", default="results/absoracle")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.engine == "pg":
        from targets.postgres_runner import PostgresRunner
        tag = Path(args.prefix).name if args.prefix else "embed"
        runner = PGSimple(PostgresRunner(
            f"/tmp/abshom_{tag}", pg_prefix=args.prefix))
    else:
        import duckdb
        runner = DuckRunner(duckdb)

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = out / f"homo_{args.engine}_{args.mode}_{args.seed}.jsonl"
    log = {"cases": 0, "diffs": 0, "by_mode": {}}

    if args.selftest:
        # tamper: one row short in the transformed database must fire
        runner.setup(["CREATE TABLE t(rid int, c0 int)",
                      "INSERT INTO t VALUES (1,10),(2,20)"])
        r1, _ = runner.run("SELECT c0 FROM t")
        runner.setup(["CREATE TABLE t(rid int, c0 int)",
                      "INSERT INTO t VALUES (1009,10090),(2018,20180)"])  # f() ok
        r2, _ = runner.run("SELECT c0 FROM t")
        diffs: list = []
        exp = expected_rows(r1, [type("P", (), {"kind": "col",
                            "ref": ColRef("t", "c0", "int"),
                            "fn": None})()], map_cell)
        compare_bags("map", r2, exp, diffs)
        assert not diffs, "clean map should not fire"
        runner.setup(["CREATE TABLE t(rid int, c0 int)",
                      "INSERT INTO t VALUES (1009,10090)"])  # dropped row
        r3, _ = runner.run("SELECT c0 FROM t")
        compare_bags("map", r3, exp, diffs)
        assert diffs, "tamper not detected — oracle dead"
        print("selftest: tamper detected — OK")
        return

    with rec.open("w") as fh:
        for i in range(args.cases):
            for m in (["map", "dup"] if args.mode == "both"
                      else [args.mode]):
                try:
                    hit = run_case(runner, args.engine, rng, m)
                except Exception as exc:  # noqa: BLE001
                    hit = {"mode": m, "setup_error": str(exc)}
                log["cases"] += 1
                if hit:
                    log["diffs"] += 1
                    log["by_mode"][m] = log["by_mode"].get(m, 0) + 1
                    hit["case"] = i
                    fh.write(json.dumps(hit, default=str) + "\n")
                    fh.flush()
    print(json.dumps(log))


if __name__ == "__main__":
    main()
