"""Shared generator + runner helpers for the absolute-oracle probes.

Design contract (mirrors the ABSOLUTE_ORACLE_PLAN fragment discipline):
- tiny schemas: 2-3 tables, 3-5 cols each of INTEGER / VARCHAR / BOOLEAN,
  5-25 rows per table, ~15% NULLs, dense value pools so joins/group-bys hit.
- every table carries a unique ``rid`` column (row id) so probes can refer to
  individual rows and ORDER BY can be made total.
- generated queries stay inside a restricted fragment that is legal in both
  PostgreSQL and DuckDB: no arithmetic expressions, no casts/floats/LIKE/
  window functions; only col-vs-col / col-vs-const comparisons, IS [NOT]
  NULL, boolean combinators, IN/EXISTS subqueries, equijoins, GROUP BY /
  HAVING, the five core aggregates, DISTINCT, set-ops, ORDER BY + LIMIT.
- the AST keeps enough structure for the data-homomorphism oracle to remap
  constants and post-process projections deterministically.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

COEVO_ROOT = "/home/user/work/db_safety/co_evolution_db_testing"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, COEVO_ROOT)  # read-only reuse of their harness

from oracles.normalize import normalize_rows  # noqa: E402

# ----------------------------------------------------------------- values
INT_POOL = list(range(0, 12)) + [17, 23, 42]
TXT_POOL = ["a", "b", "c", "d", "ab", "xyz", "", "zz"]
BOOL_POOL = [True, False]
NULL_RATE = 0.15


def rand_value(rng: random.Random, typ: str, nullable: bool = True) -> Any:
    if nullable and rng.random() < NULL_RATE:
        return None
    if typ == "int":
        return rng.choice(INT_POOL)
    if typ == "txt":
        return rng.choice(TXT_POOL)
    if typ == "bool":
        return rng.choice(BOOL_POOL)
    raise AssertionError(typ)


def sql_literal(v: Any, typ: str) -> str:
    if v is None:
        return "NULL"
    if typ == "int":
        return str(int(v))
    if typ == "txt":
        return "'" + str(v).replace("'", "''") + "'"
    if typ == "bool":
        return "TRUE" if v else "FALSE"
    raise AssertionError(typ)


# ----------------------------------------------------------------- schema
@dataclass
class Col:
    name: str
    typ: str  # int | txt | bool
    nullable: bool = True


@dataclass
class Table:
    name: str
    cols: list[Col]
    rows: list[list[Any]]  # positional, aligned with cols

    def col(self, name: str) -> Col:
        return next(c for c in self.cols if c.name == name)


DDL_TYPE = {
    "int": {"pg": "INTEGER", "duck": "INTEGER"},
    "txt": {"pg": "VARCHAR(16)", "duck": "VARCHAR"},
    "bool": {"pg": "BOOLEAN", "duck": "BOOLEAN"},
}


def gen_db(rng: random.Random, n_tables: int | None = None) -> list[Table]:
    n_tables = n_tables or rng.choice([2, 2, 3])
    tables: list[Table] = []
    for i in range(n_tables):
        cols = [Col("rid", "int", nullable=False)]
        kinds = rng.sample(["int", "txt", "bool"],
                           k=rng.randint(2, 3))
        # guarantee at least one int besides rid so joins/aggregates exist
        if "int" not in kinds:
            kinds[0] = "int"
        for j, typ in enumerate(kinds):
            cols.append(Col(f"c{j}", typ))
        n_rows = rng.randint(5, 25)
        rows = [
            [r] + [rand_value(rng, c.typ, c.nullable) for c in cols[1:]]
            for r in range(n_rows)
        ]
        tables.append(Table(f"t{i}", cols, rows))
    return tables


def ddl_and_inserts(tables: list[Table], dialect: str) -> list[str]:
    out = []
    for t in tables:
        defs = ", ".join(
            f"{c.name} {DDL_TYPE[c.typ][dialect]}" for c in t.cols
        )
        out.append(f"CREATE TABLE {t.name} ({defs})")
        vals = ", ".join(
            "(" + ", ".join(sql_literal(v, c.typ)
                            for v, c in zip(row, t.cols)) + ")"
            for row in t.rows
        )
        out.append(f"INSERT INTO {t.name} VALUES {vals}")
    return out


# ------------------------------------------------------------ query model
@dataclass
class ColRef:
    table: str
    col: str
    typ: str

    def sql(self) -> str:
        return f"{self.table}.{self.col}"


@dataclass
class Pred:
    kind: str            # cmp | isnull | and | or | not | in | exists
    op: str | None = None
    left: Any = None     # ColRef | literal
    right: Any = None    # ColRef | literal | list | Query
    children: list["Pred"] = field(default_factory=list)

    def sql(self) -> str:
        if self.kind == "cmp":
            return f"{_side(self.left)} {self.op} {_side(self.right)}"
        if self.kind == "isnull":
            return f"{self.left.sql()} IS {'NOT ' if self.op == 'not' else ''}NULL"
        if self.kind in ("and", "or"):
            return "(" + f" {self.kind.upper()} ".join(
                c.sql() for c in self.children) + ")"
        if self.kind == "not":
            return f"(NOT {self.children[0].sql()})"
        if self.kind == "in":
            if isinstance(self.right, Query):
                inner = self.right.sql()
            else:
                inner = ", ".join(sql_literal(v, self.left.typ)
                                  for v in self.right)
            return (f"{self.left.sql()} "
                    f"{'NOT ' if self.op == 'notin' else ''}IN ({inner})")
        if self.kind == "exists":
            return f"{'NOT ' if self.op == 'notexists' else ''}EXISTS ({self.right.sql()})"
        raise AssertionError(self.kind)


def _side(x: Any) -> str:
    if isinstance(x, ColRef):
        return x.sql()
    v, typ = x
    return sql_literal(v, typ)


@dataclass
class Proj:
    """One output column; ``law`` drives the homo/dup expected transform."""
    kind: str            # col | agg
    ref: ColRef | None = None
    fn: str | None = None   # count|countd|sum|min|max|avg
    typ: str = "int"

    def sql(self) -> str:
        if self.kind == "col":
            return self.ref.sql()
        arg = "*" if self.ref is None else self.ref.sql()
        if self.fn == "count":
            return f"COUNT({arg})"
        if self.fn == "countd":
            return f"COUNT(DISTINCT {arg})"
        return f"{self.fn.upper()}({arg})"


@dataclass
class SetOp:
    op: str              # union | unionall | intersect | intersectall | except | exceptall
    left: "Query"
    right: "Query"


@dataclass
class Query:
    projs: list[Proj]
    from_tables: list[str]                 # may repeat (self-join -> alias)
    aliases: dict[str, str] = field(default_factory=dict)  # alias->table
    joins: list[tuple[str, str, ColRef, ColRef, str]] = field(default_factory=list)
    # (left_alias, right_alias, left_key, right_key, kind) kind: inner|left
    where: Pred | None = None
    groupby: list[ColRef] = field(default_factory=list)
    having: Pred | None = None
    distinct: bool = False
    orderby: list[ColRef] = field(default_factory=list)
    limit: int | None = None
    setop: SetOp | None = None

    def sql(self) -> str:
        if self.setop is not None:
            kw = {
                "union": "UNION", "unionall": "UNION ALL",
                "intersect": "INTERSECT", "intersectall": "INTERSECT ALL",
                "except": "EXCEPT", "exceptall": "EXCEPT ALL",
            }[self.setop.op]
            return f"{self.setop.left.sql()} {kw} {self.setop.right.sql()}"
        parts = ["SELECT " + ("DISTINCT " if self.distinct else "")
                 + ", ".join(p.sql() for p in self.projs)]
        first = self.from_tables[0]
        first_alias = next(a for a, t in self.aliases.items() if t == first)
        frm = f" FROM {first} {first_alias}" if first_alias != first else f" FROM {first}"
        parts.append(frm)
        for la, ra, lk, rk, kind in self.joins:
            jt = "LEFT JOIN" if kind == "left" else "JOIN"
            parts.append(f" {jt} {self.aliases[ra]} {ra} ON {lk.sql()} = {rk.sql()}")
        if self.where is not None:
            parts.append(f" WHERE {self.where.sql()}")
        if self.groupby:
            parts.append(" GROUP BY " + ", ".join(c.sql() for c in self.groupby))
        if self.having is not None:
            parts.append(f" HAVING {self.having.sql()}")
        if self.orderby:
            parts.append(" ORDER BY " + ", ".join(
                c.sql() for c in self.orderby))
        if self.limit is not None:
            parts.append(f" LIMIT {self.limit}")
        return "".join(parts)


# ------------------------------------------------------------- generators
class QueryGen:
    def __init__(self, rng: random.Random, tables: list[Table]):
        self.rng = rng
        self.tables = tables
        self.tmap = {t.name: t for t in tables}

    def colrefs(self, scope: list[str], aliases: dict | None = None
                ) -> list[ColRef]:
        amap = self.aliases if aliases is None else aliases
        refs = []
        for alias in scope:
            t = self.tmap[amap[alias]]
            for c in t.cols:
                refs.append(ColRef(alias, c.name, c.typ))
        return refs

    def _from(self) -> tuple[str, dict, list]:
        """Pick base table + join skeleton; returns (base, aliases, joins)."""
        rng = self.rng
        n_join = rng.choices([0, 1, 2], weights=[5, 4, 1])[0]
        n_join = min(n_join, len(self.tables) - 1)
        base = rng.choice(self.tables).name
        used = {base}
        aliases = {base: base}
        joins = []
        for _ in range(n_join):
            cand = [t.name for t in self.tables if t.name not in used]
            if not cand:
                break
            rt = rng.choice(cand)
            lt = rng.choice(list(used))
            li = [c for c in self.tmap[lt].cols if c.typ == "int"]
            ri = [c for c in self.tmap[rt].cols if c.typ == "int"]
            if not li or not ri:
                break
            joins.append((lt, rt, ColRef(lt, rng.choice(li).name, "int"),
                          ColRef(rt, rng.choice(ri).name, "int"),
                          "left" if rng.random() < 0.25 else "inner"))
            used.add(rt)
            aliases[rt] = rt
        if not joins and rng.random() < 0.15:
            a = base + "_s"
            joins.append((base, a, ColRef(base, "rid", "int"),
                          ColRef(a, "rid", "int"), "inner"))
            aliases[a] = base
        return base, aliases, joins

    def gen(self, allow_limit: bool = True, allow_avg: bool = True
            ) -> Query:
        self._allow_avg = allow_avg
        rng = self.rng
        base, aliases, joins = self._from()
        self.aliases = aliases
        refs = self.colrefs(list(aliases), aliases)
        q = Query(projs=[], from_tables=[base], aliases=aliases, joins=joins)
        q.where = self.gen_pred(refs, 0) if rng.random() < 0.75 else None
        mode = rng.random()
        if mode < 0.30:
            # GROUP BY: keys + aggregates only
            gb = rng.sample(refs, k=min(len(refs), rng.randint(1, 2)))
            q.groupby = gb
            q.projs = [Proj("col", ref=r, typ=r.typ) for r in gb]
            q.projs += [self.gen_agg(refs, self._allow_avg)
                        for _ in range(rng.randint(1, 2))]
            if rng.random() < 0.3:
                q.having = self.gen_pred(gb, 0, simple=True)
        elif mode < 0.45:
            # whole-table aggregate
            q.projs = [self.gen_agg(refs, self._allow_avg)
                       for _ in range(rng.randint(1, 2))]
        else:
            n_proj = rng.randint(1, min(3, len(refs)))
            q.projs = [Proj("col", ref=r, typ=r.typ)
                       for r in rng.sample(refs, n_proj)]
            if rng.random() < 0.25:
                q.distinct = True
            # LIMIT only on single-table queries: joined rid values are not
            # unique, so ORDER BY rid is not a total order and LIMIT becomes
            # legitimately nondeterministic on ties.  For DISTINCT, ORDER BY
            # columns must appear in the select list (PG enforces), so order
            # by projected cols only; LIMIT needs rid among them.
            if allow_limit and not q.joins and rng.random() < 0.3:
                if q.distinct:
                    q.orderby = [p.ref for p in q.projs if p.kind == "col"]
                    if any(r.col == "rid" for r in q.orderby):
                        q.limit = rng.randint(1, 8)
                else:
                    ob = [r for r in refs if r.col == "rid"]
                    if ob:
                        q.orderby = [rng.choice(refs), ob[0]]
                        q.limit = rng.randint(1, 8)
        if rng.random() < 0.12 and not q.orderby and q.limit is None:
            r2 = Query(projs=list(q.projs), from_tables=list(q.from_tables),
                       aliases=dict(q.aliases), joins=list(q.joins),
                       where=self.gen_pred(refs, 0),
                       groupby=list(q.groupby))
            q = Query(projs=q.projs, from_tables=q.from_tables,
                      aliases=q.aliases, joins=q.joins,
                      setop=SetOp(rng.choice(
                          ["union", "unionall", "except", "intersect"]),
                          left=q, right=r2))
        return q

    def gen_agg(self, refs: list[ColRef], allow_avg: bool = True) -> Proj:
        fns = ["count", "count", "countd", "sum", "min", "max"]
        if allow_avg:
            fns.append("avg")
        fn = self.rng.choice(fns)
        if fn == "count":
            return Proj("agg", None, "count", "int")
        pool = refs if fn in ("countd", "min", "max") else [
            r for r in refs if r.typ == "int"]
        if not pool:
            return Proj("agg", None, "count", "int")
        return Proj("agg", self.rng.choice(pool), fn, "num")

    def gen_pred(self, refs: list[ColRef], depth: int,
                 simple: bool = False) -> Pred:
        rng = self.rng
        if depth > 2:
            simple = True
        roll = rng.random()
        if roll < 0.5 or simple:
            left = rng.choice(refs)
            right_same = [r for r in refs
                          if r.typ == left.typ and r is not left]
            if right_same and rng.random() < 0.3:
                return Pred("cmp", rng.choice(
                    ["=", "<>", "<", "<=", ">", ">="]), left,
                    rng.choice(right_same))
            v = rand_value(rng, left.typ, nullable=False)
            return Pred("cmp", rng.choice(
                ["=", "=", "<>", "<", "<=", ">", ">="]), left,
                (v, left.typ))
        if roll < 0.6:
            return Pred("isnull", "not" if rng.random() < 0.5 else "is",
                        rng.choice(refs))
        if roll < 0.75:
            left = rng.choice(refs)
            vals = [rand_value(rng, left.typ, nullable=False)
                    for _ in range(rng.randint(1, 4))]
            if rng.random() < 0.3:
                vals.append(None)
            return Pred("in", "notin" if rng.random() < 0.4 else "in",
                        left, vals)
        if roll < 0.9:
            return Pred(rng.choice(["and", "or"]),
                        children=[self.gen_pred(refs, depth + 1)
                                  for _ in range(rng.randint(2, 3))])
        return Pred("not", children=[self.gen_pred(refs, depth + 1)])


# ------------------------------------------------------------- exec utils
def bag(rows: Iterable[Iterable[Any]]):
    from collections import Counter
    return Counter(
        json.dumps(list(r), sort_keys=True, default=str)
        for r in normalize_rows(rows)
    )


class DuckRunner:
    def __init__(self, module: Any):
        self.duckdb = module
        self.con = module.connect()

    def setup(self, stmts: Iterable[str]) -> None:
        cur = self.con.cursor()
        # duckdb has no droppable default schema; wipe known table names
        for name in ("t", "t0", "t1", "t2", "_snap", "_dst", "_want",
                     "_src"):
            cur.execute(f"DROP TABLE IF EXISTS {name}")
        for s in stmts:
            cur.execute(s)

    def run(self, sql: str) -> tuple[list[list[Any]], str | None]:
        try:
            cur = self.con.cursor()
            cur.execute(sql)
            rows = normalize_rows(cur.fetchall()) if cur.description else []
            return rows, None
        except Exception as exc:  # noqa: BLE001
            return [], f"{type(exc).__name__}: {exc}"

    def exec(self, sql: str) -> str | None:
        try:
            self.con.execute(sql)
            return None
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"

    def explain(self, sql: str) -> str:
        rows, err = self.run("EXPLAIN " + sql)
        return err or json.dumps(rows)


class PGSimple:
    """Thin wrapper over PostgresRunner with setup returning errors."""

    def __init__(self, runner):
        self.r = runner

    def setup(self, stmts: Iterable[str]) -> None:
        self.r.setup(stmts)

    def run(self, sql: str) -> tuple[list[list[Any]], str | None]:
        res = self.r.run(sql)
        return res.rows, res.error

    def exec(self, sql: str) -> str | None:
        return self.run(sql)[1]

    def explain(self, sql: str) -> str:
        plan = self.r.explain_plan(sql)
        return json.dumps(plan) if plan else "ERR"


def now_ms() -> int:
    return int(time.time() * 1000)
