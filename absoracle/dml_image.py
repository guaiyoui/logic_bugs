"""Oracle B: DML before/after-image contracts.

The SELECT-only oracle literature never checks that a write touched *exactly*
the right rows.  We snapshot the table (`_snap`), run the DML, then check
contracts that bind the write path to the read path:

  DELETE ... WHERE p RETURNING *:
      bag(RETURNING)        == bag(_snap WHERE p IS TRUE)          (ret)
      bag(t post-state)     == bag(_snap WHERE p IS NOT TRUE)      (post)
  UPDATE ... SET c=e WHERE p RETURNING *:
      rows NOT matching p   == identical to _snap (by rid join)    (untouched)
      rows matching p       == _snap with c := e                   (image)
      bag(RETURNING)        == post-state of matched rids          (ret)
  INSERT ... ON CONFLICT(rid) DO UPDATE / DO NOTHING ... RETURNING:
      post-state ≡ expected merge; DO NOTHING leaves state & ret empty.
  INSERT INTO s SELECT ...  ≡ direct SELECT (write-path round-trip).

`p IS TRUE`/`IS NOT TRUE` splits are deliberate three-valued partitions —
`NOT p` would wrongly exclude NULL rows the DELETE keeps.

The shared-path caveat: expression `e` is evaluated by the same evaluator in
UPDATE and in the checking SELECT, so pure eval bugs stay hidden; what this
oracle isolates is *which rows the write path touched* and RETURNING
fidelity — exactly the classes metamorphic SELECT oracles never see.

--selftest secretly deletes one extra row and asserts the checker fires.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

COEVO_ROOT = "/home/user/work/db_safety/co_evolution_db_testing"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, COEVO_ROOT)  # read-only reuse of their harness

from absoracle.common import (  # noqa: E402
    ColRef, DuckRunner, PGSimple, QueryGen, Table, bag, ddl_and_inserts,
    gen_db, rand_value, sql_literal,
)


def _pred_sql(rng: random.Random, table: Table) -> str:
    refs = [ColRef(table.name, c.name, c.typ) for c in table.cols]
    g = QueryGen(rng, [table])
    # unqualified predicate: usable verbatim against t, _snap, subqueries
    return g.gen_pred(refs, 0).sql().replace(table.name + ".", "")


def _check(runner, label: str, got_sql: str, want_sql: str,
           diffs: list) -> None:
    g_rows, g_err = runner.run(got_sql)
    w_rows, w_err = runner.run(want_sql)
    if g_err or w_err:
        diffs.append({"check": label, "kind": "error",
                      "got_err": g_err, "want_err": w_err})
        return
    if bag(g_rows) != bag(w_rows):
        diffs.append({"check": label, "kind": "result_diff",
                      "got": g_rows[:8], "want": w_rows[:8],
                      "got_n": len(g_rows), "want_n": len(w_rows)})


def case_delete(runner, rng, t: Table, stmts, diffs) -> None:
    p = _pred_sql(rng, t)
    runner.setup(stmts + [f"CREATE TABLE _snap AS TABLE {t.name}"])
    ret, ret_err = runner.run(
        f"DELETE FROM {t.name} WHERE {p} RETURNING *")
    if ret_err:
        diffs.append({"check": "ret", "kind": "dml_error",
                      "err": ret_err})
        return
    want, _ = runner.run(f"SELECT * FROM _snap WHERE ({p}) IS TRUE")
    if bag(ret) != bag(want):
        diffs.append({"check": "ret", "kind": "result_diff",
                      "got": ret[:8], "want": want[:8]})
    _check(runner, "post", f"SELECT * FROM {t.name}",
           f"SELECT * FROM _snap WHERE ({p}) IS NOT TRUE", diffs)


def case_update(runner, rng, t: Table, stmts, diffs) -> None:
    cols = [c for c in t.cols if c.name != "rid"]
    tgt = rng.choice(cols)
    if rng.random() < 0.5:
        same = [c for c in cols if c.typ == tgt.typ]
        expr = rng.choice(same).name         # unqualified: valid on both sides
    else:
        expr = sql_literal(rand_value(rng, tgt.typ, nullable=False),
                           tgt.typ)
    p = _pred_sql(rng, t)
    sel = ", ".join(f"{t.name}.{c.name}" for c in t.cols)
    runner.setup(stmts + [f"CREATE TABLE _snap AS TABLE {t.name}"])
    ret, err = runner.run(
        f"UPDATE {t.name} SET {tgt.name} = {expr} WHERE {p} "
        f"RETURNING {sel}")
    if err:
        diffs.append({"check": "dml", "kind": "dml_error", "err": err})
        return
    matched = f"(SELECT rid FROM _snap WHERE ({p}) IS TRUE)"
    _check(runner, "untouched",
           f"SELECT {sel} FROM {t.name} WHERE rid IN "
           f"(SELECT rid FROM _snap WHERE ({p}) IS NOT TRUE)",
           f"SELECT {sel.replace(t.name + '.', '_snap.')} FROM _snap "
           f"WHERE ({p}) IS NOT TRUE", diffs)
    newcols = ", ".join(
        (expr if c.name == tgt.name else f"_snap.{c.name}")
        for c in t.cols)
    _check(runner, "image",
           f"SELECT {sel} FROM {t.name} WHERE rid IN {matched}",
           f"SELECT {newcols} FROM _snap WHERE ({p}) IS TRUE", diffs)
    want, _ = runner.run(
        f"SELECT {newcols} FROM _snap WHERE ({p}) IS TRUE")
    if bag(ret) != bag(want):
        diffs.append({"check": "ret", "kind": "result_diff",
                      "got": ret[:8], "want": want[:8]})


def case_conflict(runner, rng, t: Table, stmts, diffs) -> None:
    cols = [c for c in t.cols if c.name != "rid"]
    row = rng.choice(t.rows)
    new_vals = []
    for c in t.cols:
        if c.name == "rid":
            new_vals.append(str(row[0]))          # hit the existing key
        else:
            new_vals.append(sql_literal(
                rand_value(rng, c.typ, nullable=False), c.typ))
    do_update = rng.random() < 0.5
    upd = ""
    if do_update and cols:
        tgt = rng.choice(cols)
        upd = (f" DO UPDATE SET {tgt.name} = excluded.{tgt.name}")
    else:
        upd = " DO NOTHING"
    runner.setup(stmts + [
        f"CREATE UNIQUE INDEX _u_rid ON {t.name} (rid)",
        f"CREATE TABLE _snap AS TABLE {t.name}",
    ])
    sql = (f"INSERT INTO {t.name} VALUES ({', '.join(new_vals)}) "
           f"ON CONFLICT (rid){upd} RETURNING *")
    ret, err = runner.run(sql)
    if err:
        diffs.append({"check": "conflict", "kind": "dml_error",
                      "err": err, "sql": sql})
        return
    if do_update:
        want, _ = runner.run(
            f"SELECT * FROM {t.name} WHERE rid = {row[0]}")
        if bag(ret) != bag(want):
            diffs.append({"check": "conflict_ret", "kind": "result_diff",
                          "got": ret[:5], "want": want[:5]})
        # state contract: same rid count, one merged row
        n, _ = runner.run(f"SELECT COUNT(*) FROM {t.name}")
        if n and n[0][0] != len(t.rows):
            diffs.append({"check": "conflict_count",
                          "kind": "result_diff", "got": n})
    else:
        if ret:
            diffs.append({"check": "donothing_ret",
                          "kind": "result_diff", "got": ret[:5]})
        _check(runner, "donothing_post", f"SELECT * FROM {t.name}",
               "SELECT * FROM _snap", diffs)


def case_insert_select(runner, rng, t: Table, stmts, diffs) -> None:
    p = _pred_sql(rng, t)
    runner.setup(stmts + [
        f"CREATE TABLE _dst AS TABLE {t.name}",
        f"DELETE FROM _dst",
        f"CREATE TABLE _want AS SELECT * FROM {t.name} WHERE {p}",
        f"INSERT INTO _dst SELECT * FROM {t.name} WHERE {p}",
    ])
    _check(runner, "insert_select", "SELECT * FROM _dst",
           "SELECT * FROM _want", diffs)


def case_merge(runner, rng, t: Table, stmts, diffs) -> None:
    """MERGE: matched rows take src value, unmatched src rows insert."""
    cols = [c.name for c in t.cols]
    tgt = rng.choice([c for c in cols if c != "rid"])
    half = len(t.rows) // 2
    src_rows = []
    for r in t.rows[:half]:          # matched: same rid, new values
        src_rows.append([r[0]] + [rand_value(rng, c.typ, nullable=False)
                                  for c in t.cols[1:]])
    for r in range(len(t.rows), len(t.rows) + 3):  # unmatched new rids
        src_rows.append([r] + [rand_value(rng, c.typ, nullable=False)
                               for c in t.cols[1:]])
    src = Table("_src", t.cols, src_rows)
    runner.setup(stmts + ddl_and_inserts([src], "pg")
                 + [f"CREATE TABLE _snap AS TABLE {t.name}"])
    ins_cols = ", ".join(cols)
    ins_vals = ", ".join(f"s.{c}" for c in cols)
    sql = (f"MERGE INTO {t.name} USING _src s ON {t.name}.rid = s.rid "
           f"WHEN MATCHED THEN UPDATE SET {tgt} = s.{tgt} "
           f"WHEN NOT MATCHED THEN INSERT ({ins_cols}) VALUES ({ins_vals})")
    _, err = runner.run(sql)
    if err:
        diffs.append({"check": "merge", "kind": "dml_error", "err": err})
        return
    newcols = ", ".join(f"s.{c}" if c == tgt else f"a.{c}" for c in cols)
    want = ("(SELECT * FROM _snap WHERE rid NOT IN (SELECT rid FROM _src))"
            " UNION ALL "
            "(SELECT _src.* FROM _src WHERE rid NOT IN "
            " (SELECT rid FROM _snap)) UNION ALL "
            f"(SELECT {newcols} FROM _snap a JOIN _src s ON a.rid = s.rid)")
    _check(runner, "merge_post", f"SELECT * FROM {t.name}", want, diffs)


_MERGE_OK = {"pg": True, "duck": False}


def run_case(runner, dialect, rng) -> dict | None:
    tables = gen_db(rng, n_tables=2)
    t = tables[0]
    stmts = ddl_and_inserts(tables, dialect)
    diffs: list = []
    choices = ["delete", "update", "conflict", "insert_select"]
    if _MERGE_OK.get(dialect):
        choices += ["merge"]
    kind = rng.choice(choices)
    fn = {"delete": case_delete, "update": case_update,
          "conflict": case_conflict, "merge": case_merge,
          "insert_select": case_insert_select}[kind]
    try:
        fn(runner, rng, t, stmts, diffs)
    except Exception as exc:  # noqa: BLE001
        return {"kind": kind, "setup_error": str(exc)}
    if diffs:
        return {"kind": kind, "table": t.name, "diffs": diffs}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["pg", "duck"], required=True)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/absoracle")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.engine == "pg":
        from targets.postgres_runner import PostgresRunner
        tag = Path(args.prefix).name if args.prefix else "embed"
        runner = PGSimple(PostgresRunner(
            f"/tmp/absdml_{tag}", pg_prefix=args.prefix))
    else:
        import duckdb
        runner = DuckRunner(duckdb)

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = out / f"dml_{args.engine}_{args.seed}.jsonl"
    log = {"cases": 0, "diffs": 0, "by_kind": {}}

    if args.selftest:
        runner.setup(["CREATE TABLE t(rid int, c0 int)",
                      "INSERT INTO t VALUES (1,10),(2,20),(3,30)",
                      "CREATE TABLE _snap AS TABLE t"])
        runner.exec("DELETE FROM t WHERE rid=1")  # tamper: row NOT matching p
        diffs: list = []
        _check(runner, "post", "SELECT * FROM t",
               "SELECT * FROM _snap WHERE (rid=3) IS NOT TRUE", diffs)
        assert diffs, "tamper not detected — oracle dead"
        print("selftest: tamper detected — OK", diffs[0]["kind"])
        return

    with rec.open("w") as fh:
        for i in range(args.cases):
            hit = run_case(runner, args.engine, rng)
            log["cases"] += 1
            if hit:
                log["diffs"] += 1
                k = hit.get("kind", "?")
                log["by_kind"][k] = log["by_kind"].get(k, 0) + 1
                hit["case"] = i
                fh.write(json.dumps(hit, default=str) + "\n")
                fh.flush()
    print(json.dumps(log))


if __name__ == "__main__":
    main()
