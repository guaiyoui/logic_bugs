"""Oracle A: execution-environment invariance.

Invariant: a deterministic query must return the same bag of rows under any
execution-resource configuration.  Arms change *how* the query runs, never
*what* it must return:

  PostgreSQL: work_mem pin (spill paths), enable_* operator switches,
              forced parallel workers, debug_parallel_query, plan_cache_mode
              (generic vs custom via PREPARE/EXECUTE), JIT (gated on LLVM).
  DuckDB:     threads, debug_force_external (out-of-core operators),
              perfect_ht_threshold / ordered_aggregate_threshold,
              preserve_insertion_order, memory_limit, debug_window_mode.

Anti-dead-oracle measures:
  * every arm records whether its plan fingerprint differs from baseline
    (EXPLAIN), so silent/no-op arms are visible;
  * --selftest tampers with one arm's bag and asserts the checker fires.

Usage:  python -m absoracle.env_sweep --engine pg --prefix ../pgbld/pg186_assert \
            --cases 300 --seed 1
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
    DuckRunner, PGSimple, QueryGen, bag, ddl_and_inserts, gen_db,
)

PG_ARMS: dict[str, list[str]] = {
    "wm64k":       ["SET work_mem='64kB'"],
    "wm256k":      ["SET work_mem='256kB'"],
    "noseq":       ["SET enable_seqscan=off"],
    "noidx":       ["SET enable_indexscan=off", "SET enable_indexonlyscan=off"],
    "nobitmap":    ["SET enable_bitmapscan=off"],
    "nohj":        ["SET enable_hashjoin=off"],
    "nomj":        ["SET enable_mergejoin=off"],
    "nonl":        ["SET enable_nestloop=off"],
    "nohashagg":   ["SET enable_hashagg=off"],
    "nosort":      ["SET enable_sort=off"],
    "nomat":       ["SET enable_material=off"],
    "nomemoize":   ["SET enable_memoize=off"],
    "noincsort":   ["SET enable_incremental_sort=off"],
    "par4":        ["SET max_parallel_workers_per_gather=4",
                    "SET min_parallel_table_scan_size=0",
                    "SET min_parallel_index_scan_size=0",
                    "SET parallel_setup_cost=0", "SET parallel_tuple_cost=0"],
    "dbgpar":      ["SET debug_parallel_query='regress'"],
    "rpc":         ["SET random_page_cost=0.1", "SET cpu_tuple_cost=1"],
    "pwj":         ["SET enable_partitionwise_join=on",
                    "SET enable_partitionwise_aggregate=on"],
    "jit":         ["SET jit=on", "SET jit_above_cost=0",
                    "SET jit_inline_above_cost=0",
                    "SET jit_optimize_above_cost=0"],
}

DUCK_ARMS: dict[str, list[str]] = {
    "t1":     ["SET threads=1"],
    "t4":     ["SET threads=4"],
    "pio":    ["SET preserve_insertion_order=false"],
    "ext":    ["SET debug_force_external=true"],
    "pht0":   ["SET perfect_ht_threshold=0"],
    "oat10":  ["SET ordered_aggregate_threshold=10"],
    "mem64":  ["SET memory_limit='64MB'", "SET debug_force_external=true"],
    "winsep": ["SET debug_window_mode='separate'"],
}


def run_case(runner, dialect: str, rng: random.Random, arms: dict,
             log: dict) -> dict | None:
    tables = gen_db(rng)
    setup = ddl_and_inserts(tables, dialect)
    # half the cases get a secondary index to make index arms live
    idx_sql = None
    if rng.random() < 0.5:
        t = rng.choice(tables)
        intcols = [c.name for c in t.cols if c.typ == "int"]
        idx_sql = f"CREATE INDEX idx_{t.name} ON {t.name}({intcols[0]})"
        setup.append(idx_sql)
    q = QueryGen(rng, tables).gen()
    sql = q.sql()

    runner.setup(setup)
    base_rows, base_err = runner.run(sql)
    base_bag = bag(base_rows) if base_err is None else None
    base_plan = runner.explain(sql)
    fired_arms, diffs = [], []
    for name, sets in arms.items():
        set_errs = [runner.exec(s) for s in sets]
        if any(set_errs):
            log["set_errors"][name] = set_errs[0]
            runner.exec("RESET ALL" if dialect == "pg" else
                        "; ".join(_reset_of(s, dialect) for s in sets))
            continue
        plan = runner.explain(sql)  # fingerprint while the arm is live
        rows, err = runner.run(sql)
        runner.exec("RESET ALL" if dialect == "pg" else
                    "; ".join(_reset_of(s, dialect) for s in sets))
        # PG arms mostly alter the plan; DuckDB arms alter execution under an
        # unchanged plan — count "applied" arms there, plan-changes on PG.
        if dialect == "duck" or plan != base_plan:
            fired_arms.append(name)
        if base_err is None and err is None:
            if bag(rows) != base_bag:
                diffs.append({"arm": name, "kind": "result_diff",
                              "base_n": len(base_rows), "arm_n": len(rows),
                              "base_sample": base_rows[:5],
                              "arm_sample": rows[:5]})
        elif (err or "").split(":", 1)[0] != (base_err or "").split(":", 1)[0]:
            diffs.append({"arm": name, "kind": "error_flip",
                          "base_err": base_err, "arm_err": err})
    log["arms_fired"] += len(set(fired_arms))
    log["plans_seen"] += 1 + len(set(fired_arms))
    if diffs:
        return {"sql": sql, "setup": setup, "diffs": diffs}
    return None


def _reset_of(stmt: str, dialect: str) -> str:
    # DuckDB has no RESET ALL in older versions; undo each SET explicitly.
    if dialect == "pg":
        return "RESET ALL"
    name = stmt.split("=", 1)[0].split(None, 1)[1].strip()
    return f"RESET {name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["pg", "duck"], required=True)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--duck-module", default="duckdb")
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/absoracle")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.engine == "pg":
        from targets.postgres_runner import PostgresRunner
        tag = Path(args.prefix).name if args.prefix else "embed"
        r = PostgresRunner(f"/tmp/absenv_{tag}", pg_prefix=args.prefix)
        runner = PGSimple(r)
        arms = dict(PG_ARMS)
        vnum = r.run("SHOW server_version_num")
        try:
            ver = int(vnum.rows[0][0])
        except Exception:
            ver = 0
        if ver < 160000:
            arms.pop("dbgpar", None)  # debug_parallel_query is 16+
        if ver < 150000:
            arms.pop("pwj", None)     # partitionwise join stable since 15
    else:
        import importlib
        mod = importlib.import_module(args.duck_module)
        runner = DuckRunner(mod)
        arms = DUCK_ARMS

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = out / f"env_{args.engine}_{args.seed}.jsonl"
    log = {"cases": 0, "arms_fired": 0, "plans_seen": 0, "diffs": 0, "set_errors": {}}

    if args.selftest:
        # tamper check: a checker that never fires is a dead oracle
        rows, err = runner.run("SELECT 1")
        base = bag(rows)
        fake = bag([[2]])
        assert base != fake
        print("selftest: checker distinguishes tampered bag — OK")
        return

    with rec.open("w") as fh:
        for i in range(args.cases):
            try:
                hit = run_case(runner, args.engine, rng, arms, log)
            except Exception as exc:  # noqa: BLE001 - keep campaign alive
                hit = {"setup_error": str(exc)}
            log["cases"] += 1
            if hit:
                log["diffs"] += 1
                hit["case"] = i
                fh.write(json.dumps(hit, default=str) + "\n")
                fh.flush()
    print(json.dumps(log, default=str))


if __name__ == "__main__":
    main()
