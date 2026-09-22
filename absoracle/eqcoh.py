"""Oracle D: equivalence-coherence (EqCoh).

A relational engine implements "the same value" in half a dozen separate
places — the three-valued `=` in WHERE, dedup equality in GROUP BY /
DISTINCT / set operations, join matching, `IN`, index ordering — and all
of them must agree wherever SQL semantics requires it.  EqCoh plants
*doctrine pairs*: values that are `=`-equal but distinguishable to some
observer (numeric 1 / 1.0 / 1.00 differ in scale()/text; -0.0 / 0.0
differ under reciprocal; same integer in SMALLINT/INT/BIGINT columns),
then checks that the reimplementations cohere:

  triad    GROUP BY dedup ≡ DISTINCT ≡ UNION dedup; COUNT(DISTINCT)
  exact    grouped output self-joined on IS NOT DISTINCT FROM hits each
           row exactly once; on `=` hits exactly the non-NULL-key groups
  sat      HAVING / outer WHERE on a group key: legal outcomes keep each
           group whole (count = class size); a pushed-down qual silently
           shortens counts — the #19619-class signature
  in_ex    x IN (subq) ≡ EXISTS; NOT IN obeys its 3VL decomposition law
  setop    UNION ≡ DISTINCT∘UNION ALL; A ≡ (A EXCEPT ALL B) ⊎ (A INTERSECT
           ALL B); NULL dedup atoms; A EXCEPT A ≡ ∅
  fiber    equijoin output carries each side's own stored value (per-rid
           text check) — catches representative substitution (#16901)
  card     equijoin cardinality = Σ_class |a∩cls|·|b∩cls| (known answer)
  order    index-ordered scan ≡ seqscan+sort on the same key (PG)
  algo     probes re-run under forced plan arms (hashagg/hashjoin off)

Legality is judged against class structure computed harness-side from the
generated rows, so legal representative freedom never fires: only count
shortfalls, phantom keys, forbidden classes, and path divergence do.

--selftest corrupts one runner result and requires the checker to fire.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

COEVO_ROOT = "/home/user/work/db_safety/co_evolution_db_testing"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, COEVO_ROOT)

from absoracle.common import Col, DuckRunner, PGSimple, Table, bag  # noqa: E402

# ------------------------------------------------------------------ data

NUM_FAMILY = {"int", "int2", "int8", "num"}

NUM_CLASSES = [
    [Decimal(s) for s in grp] for grp in (
        ("1", "1.0", "1.00", "1.000"),
        ("2", "2.0", "2.00"),
        ("3", "3.0", "3.00"),
        ("0.5", "0.50"),
        ("4",), ("5",), ("7",), ("9",), ("0.25",), ("6",),
    )
]
FLT_CLASSES = [
    [0.0, -0.0], [1.0], [2.5], [-1.5], [3.25],
    [float("nan")], [float("inf")], [float("-inf")],
]
INT_POOL = list(range(0, 12)) + [17, 23, 42]
TXT_POOL = ["a", "b", "c", "d", "ab", "xyz", "", "zz", "a ", "AB"]
BOOL_POOL = [True, False]
NULL_RATE = 0.15
DOCTRINE_RATE = 0.5

EQ_TYPES = ["int", "int2", "int8", "num", "flt", "txt", "bool"]
# int8 boundary values: int64 > 2^53 lose precision in float8 rounding —
# the cross-type `=` is then non-transitive (planner EC assumption gap)
INT8_EDGE = [9007199254740991, 9007199254740992, 9007199254740993,
             9007199254740994, -9007199254740993, 2**62, 2**63 - 1,
             -2**63]
FLT8_EDGE = [9007199254740992.0, 9007199254740994.0,
             -9007199254740992.0, float(2**62)]
JSB_CLASSES = [
    ['{"a": 1}', '{"a": 1.0}', '{"a": 1.00}'],
    ['{"a": 2}', '{"a": 2.0}'],
    ['{"a": [1, 2]}', '{"a": [1.0, 2.0]}'],
    ['{"b": true}'], ['{"b": false}'], ['{}'], ['{"a": null}'],
]
EQ_DDL = {
    "pg":   {"int": "INTEGER", "int2": "SMALLINT", "int8": "BIGINT",
             "num": "NUMERIC", "flt": "DOUBLE PRECISION",
             "txt": "VARCHAR(32)", "bool": "BOOLEAN"},
    "duck": {"int": "INTEGER", "int2": "SMALLINT", "int8": "BIGINT",
             "num": "DECIMAL(18,6)", "flt": "DOUBLE",
             "txt": "VARCHAR", "bool": "BOOLEAN"},
}
EQ_DDL["pg"]["jsb"] = "JSONB"
TEXT_T = {"pg": "TEXT", "duck": "VARCHAR"}
FLT_T = {"pg": "DOUBLE PRECISION", "duck": "DOUBLE"}
INF_LIT = {"pg": "'Infinity'::float8",
           "duck": "CAST('Inf' AS DOUBLE)"}
NINF_LIT = {"pg": "'-Infinity'::float8",
            "duck": "CAST('-Inf' AS DOUBLE)"}
# `1.0/k` on PG resolves to numeric division (errors on zero); force float8
INV = {"pg": "1.0::float8/{k}", "duck": "1.0/{k}"}


def lit(v, typ, dialect):
    if v is None:
        return "NULL"
    if typ == "num":
        return str(v)                       # Decimal keeps '1.00' scale
    if typ == "flt":
        # explicit cast: bare `-0.0` in DuckDB is DECIMAL and loses its
        # sign on the way to DOUBLE; CAST('-0.0' AS DOUBLE) keeps it
        return f"CAST('{repr(float(v))}' AS {FLT_T[dialect]})"
    if typ == "jsb":
        return "'" + str(v).replace("'", "''") + "'::jsonb"
    if typ == "bool":
        return "TRUE" if v else "FALSE"
    if typ == "txt":
        return "'" + str(v).replace("'", "''") + "'"
    return str(v)


_FLT_SENT = {"float:nan": "nan", "float:+inf": "pinf",
             "float:-inf": "ninf"}


def canon(v, typ):
    """Equality-class key under the engine's `=`/grouping doctrine."""
    if v is None:
        return ("null",)
    if isinstance(v, str) and v in _FLT_SENT:     # normalize() sentinel
        return ("flt", _FLT_SENT[v])
    if typ in NUM_FAMILY:
        return ("num", Decimal(str(v)))
    if typ == "flt":
        f = float(v)
        if math.isnan(f):
            return ("flt", "nan")            # PG & DuckDB: NaN = NaN
        if f == 0.0:
            return ("flt", "zero")           # -0.0 = 0.0
        if math.isinf(f):
            return ("flt", "pinf" if f > 0 else "ninf")
        return ("flt", f)
    if typ == "jsb":
        return ("jsb", json.dumps(_jsb_norm(v), sort_keys=True,
                                  default=str))
    return (typ, v)


def _jsb_norm(v):
    """Canonical jsonb: numeric leaves collapse 1 / 1.0 / 1.00."""
    if isinstance(v, str):
        v = json.loads(v)
    if isinstance(v, dict):
        return {k: _jsb_norm(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [_jsb_norm(x) for x in v]
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float, Decimal)):
        d = Decimal(str(v)).normalize()
        if d == 0:
            d = Decimal(0)          # jsonb numeric eq merges -0.0/0.0
        return str(d)
    return v


def gen_val(rng, typ):
    if rng.random() < NULL_RATE:
        return None
    if typ == "num":
        if rng.random() < DOCTRINE_RATE:
            return rng.choice(rng.choice(NUM_CLASSES))
        return Decimal(rng.choice([1, 2, 3, 4, 5, 7, 9]))
    if typ == "flt":
        if rng.random() < 0.2:
            return rng.choice(FLT8_EDGE)
        if rng.random() < DOCTRINE_RATE:
            return rng.choice(rng.choice(FLT_CLASSES))
        return float(rng.choice([0, 1, 2, 3, 4, 5]))
    if typ in ("int", "int2", "int8"):
        if typ == "int8" and rng.random() < 0.25:
            return rng.choice(INT8_EDGE)
        return rng.choice(INT_POOL)
    if typ == "jsb":
        return rng.choice(rng.choice(JSB_CLASSES))
    if typ == "txt":
        return rng.choice(TXT_POOL)
    return rng.choice(BOOL_POOL)


def gen_db_eq(rng, dialect="pg"):
    types = EQ_TYPES + (["jsb"] if dialect == "pg" else [])
    n_t = rng.choice([2, 2, 3])
    tables = []
    for i in range(n_t):
        kinds = rng.sample(types, k=rng.randint(2, 4))
        if not any(k in NUM_FAMILY for k in kinds):
            kinds[0] = rng.choice(["int", "num"])
        cols = [Col("rid", "int")] + [Col(f"c{j}", k)
                                      for j, k in enumerate(kinds)]
        n_rows = rng.randint(6, 20)
        rows = [[r] + [gen_val(rng, c.typ) for c in cols[1:]]
                for r in range(n_rows)]
        t = Table(f"t{i}", cols, rows)
        # plant both members of one doctrine class -> guaranteed collision
        doc = [c for c in cols[1:] if c.typ in ("num", "flt", "jsb")]
        if doc and n_rows >= 2:
            c = rng.choice(doc)
            ci = cols.index(c)
            pool = {"num": NUM_CLASSES, "flt": FLT_CLASSES,
                    "jsb": JSB_CLASSES}[c.typ]
            grp = rng.choice(pool)
            if len(grp) >= 2:
                ra, rb = rng.sample(range(n_rows), 2)
                rows[ra][ci], rows[rb][ci] = grp[0], grp[1]
        tables.append(t)
    return tables


def ddl_eq(tables, dialect):
    stmts = []
    for t in tables:
        defs = ", ".join(f"{c.name} {EQ_DDL[dialect][c.typ]}"
                         for c in t.cols)
        stmts.append(f"CREATE TABLE {t.name}({defs})")
        for row in t.rows:
            vals = ", ".join(lit(v, c.typ, dialect)
                             for v, c in zip(row, t.cols))
            stmts.append(f"INSERT INTO {t.name} VALUES ({vals})")
    return stmts


# ---------------------------------------------------------------- helpers

def _diff(diffs, probe, kind, **kw):
    d = {"probe": probe, "kind": kind}
    d.update(kw)
    diffs.append(d)


def cbag(rows, types):
    """Class-level multiset: values canonicalized by the `=` doctrine."""
    return Counter(tuple(canon(v, t) for v, t in zip(row, types))
                   for row in rows)


def _cmp2(runner, diffs, probe, qa, qb, note="", types=None):
    """Compare two queries: class-level multiset (hard law) plus raw-bag
    representative difference (weak signal)."""
    ra, ea = runner.run(qa)
    rb, eb = runner.run(qb)
    if (ea is None) != (eb is None):
        _diff(diffs, probe, "error_flip", a_err=ea, b_err=eb,
              qa=qa, qb=qb, note=note)
        return False
    if ea is not None:
        return False
    if types is not None and cbag(ra, types) != cbag(rb, types):
        _diff(diffs, probe, "result_diff", a=ra[:8], b=rb[:8],
              qa=qa, qb=qb, note=note)
        return False
    if bag(ra) != bag(rb):
        kind = "rep_diff" if types is not None else "result_diff"
        _diff(diffs, probe, kind, a=ra[:8], b=rb[:8],
              qa=qa, qb=qb, note=note)
        return False
    return True


def col_info(runner, dialect, t: Table, colname: str):
    """canon-key -> [{'v': stored value, 'text': engine text, 'rid'}]."""
    ci = [c.name for c in t.cols].index(colname)
    typ = t.cols[ci].typ
    rows, err = runner.run(
        f"SELECT rid, CAST({colname} AS {TEXT_T[dialect]}) FROM {t.name}")
    if err:
        return None
    tmap = {r[0]: r[1] for r in rows}
    classes = defaultdict(list)
    for row in t.rows:
        v = row[ci]
        classes[canon(v, typ)].append(
            {"v": v, "text": tmap.get(row[0]), "rid": row[0]})
    return {"classes": dict(classes), "typ": typ}


def _scale_of_text(tx):
    if tx is None or "." not in tx:
        return 0
    return len(tx.split(".")[1])


def _fltkey(v):
    """Sign-sensitive key that also reads normalize() float sentinels."""
    if isinstance(v, str):
        return {"float:nan": "nan", "float:+inf": "inf",
                "float:-inf": "-inf"}.get(v, v)
    f = float(v)
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return repr(f)                              # '-0.0' vs '0.0'


def _same_member(emitted_raw, emitted_text, stored_v, learned_text, typ):
    """Did the engine emit this rid's own value (not a class sibling)?"""
    if stored_v is None:
        return emitted_raw is None
    if emitted_raw is None:
        return False
    if typ == "num":
        return emitted_text == learned_text     # scale-sensitive text
    if typ == "flt":
        return _fltkey(emitted_raw) == _fltkey(stored_v)
    return emitted_raw == stored_v or emitted_text == learned_text


# ------------------------------------------------------------------ probes

def probe_triad(runner, dialect, tables, rng, diffs, stats, ex=False):
    t = rng.choice(tables)
    cols = [c for c in t.cols if c.name != "rid"]
    k = rng.choice([1, 1, 2]) if len(cols) > 1 else 1
    pick = rng.sample(cols, k=min(len(cols), k))
    ks = ", ".join(c.name for c in pick)
    ty = [c.typ for c in pick]
    g = f"SELECT {ks} FROM {t.name} GROUP BY {ks}"
    d = f"SELECT DISTINCT {ks} FROM {t.name}"
    u = f"SELECT {ks} FROM {t.name} UNION SELECT {ks} FROM {t.name}"
    ok = _cmp2(runner, diffs, "triad", g, d, ks, ty)
    ok &= _cmp2(runner, diffs, "triad", g, u, ks, ty)
    if len(pick) == 1:
        cd = f"SELECT COUNT(DISTINCT {ks}) FROM {t.name}"
        # COUNT(DISTINCT) ignores NULLs; the row set keeps the NULL row
        cn = (f"SELECT COUNT(*) FROM (SELECT DISTINCT {ks} FROM {t.name} "
              f"WHERE {ks} IS NOT NULL) x")
        ok &= _cmp2(runner, diffs, "triad", cd, cn, ks, ["int"])
    stats["triad"] += ok


def probe_partition_exact(runner, dialect, tables, rng, diffs, stats,
                          ex=False):
    t = rng.choice(tables)
    cols = [c for c in t.cols if c.name != "rid"]
    k = rng.choice([1, 1, 2]) if len(cols) > 1 else 1
    pick = rng.sample(cols, k=min(len(cols), k))
    ks = ", ".join(c.name for c in pick)
    g = f"(SELECT {ks} FROM {t.name} GROUP BY {ks})"
    ns = " AND ".join(f"a.{c.name} IS NOT DISTINCT FROM b.{c.name}"
                      for c in pick)
    eq = " AND ".join(f"a.{c.name} = b.{c.name}" for c in pick)
    rn, e1 = runner.run(f"SELECT COUNT(*) FROM {g} x")
    rd, e2 = runner.run(f"SELECT COUNT(*) FROM {g} a JOIN {g} b ON {ns}")
    if e1 or e2:
        _diff(diffs, "exact", "probe_error", e1=e1, e2=e2)
        return
    if rn != rd:
        _diff(diffs, "exact", "nsdiag_count", n_g=rn, diag=rd, ks=ks)
        return
    re_, e3 = runner.run(
        f"SELECT COUNT(*) FROM {g} a JOIN {g} b ON {eq}")
    rnull, e4 = runner.run(
        f"SELECT COUNT(*) FROM {g} x WHERE "
        + " OR ".join(f"x.{c.name} IS NULL" for c in pick))
    if e3 or e4:
        _diff(diffs, "exact", "probe_error", e3=e3, e4=e4)
        return
    if re_[0][0] != rn[0][0] - rnull[0][0]:
        _diff(diffs, "exact", "eqdiag_count", n_g=rn, null_g=rnull,
              eqdiag=re_, ks=ks)
        return
    stats["exact"] += 1


def _sat_preds(dialect, col, classes):
    """(sql-on-k, eval(member_text, member_v), tag) triples."""
    k = col.name
    preds = []
    if col.typ == "num" and dialect == "pg":
        mixed = [cl for cl in classes.values()
                 if len({_scale_of_text(m["text"]) for m in cl}) > 1]
        if mixed:
            cl = random.choice(mixed)
            sv = sorted({_scale_of_text(m["text"]) for m in cl})
            s = sv[-1]
            preds.append((f"scale({k}) = {s}",
                          lambda tx, v, s=s:
                          tx is not None and _scale_of_text(tx) == s,
                          f"scale={s}"))
    discr = [cl for cl in classes.values()
             if len({m["text"] for m in cl}) > 1]
    if discr:
        cl = random.choice(discr)
        m = random.choice(cl)
        litx = (m["text"] or "").replace("'", "''")
        preds.append((f"CAST({k} AS {TEXT_T[dialect]}) = '{litx}'",
                      lambda tx, v, lx=m["text"]: tx is not None and tx == lx,
                      f"text={m['text']!r}"))
    if col.typ == "flt" and dialect == "duck":
        # PG float8 division raises on zero divisor; DuckDB returns ±Inf
        inv = INV[dialect].format(k=k)
        preds.append((f"{inv} = {INF_LIT[dialect]}",
                      lambda tx, v: isinstance(v, float) and v == 0.0
                      and math.copysign(1.0, v) > 0,
                      "inv=+inf"))
        preds.append((f"{inv} = {NINF_LIT[dialect]}",
                      lambda tx, v: isinstance(v, float) and v == 0.0
                      and math.copysign(1.0, v) < 0,
                      "inv=-inf"))
    return preds


def _check_sat(runner, q, classes, ev, diffs, probe, tag):
    """Every emitted (key-text, count) must be a legal group outcome."""
    rows, err = runner.run(q)
    if err:
        _diff(diffs, probe, "probe_error", err=err, q=q, tag=tag)
        return
    text2key = {}
    for key, cl in classes.items():
        for m in cl:
            text2key.setdefault(m["text"], key)
    emitted = set()
    for row in rows:
        kk, cnt = row[0], row[1]
        key = ("null",) if kk is None else text2key.get(kk)
        cl = classes.get(key)
        if cl is None:
            _diff(diffs, probe, "phantom_key", kk=kk, cnt=cnt, q=q,
                  tag=tag)
            continue
        emitted.add(key)
        if cnt != len(cl):
            _diff(diffs, probe, "count_shortfall", kk=kk, cnt=cnt,
                  size=len(cl), q=q, tag=tag)
        memb = next((m for m in cl if m["text"] == kk), None)
        if memb is None or not ev(memb["text"], memb["v"]):
            _diff(diffs, probe, "mixedrep", kk=kk, cnt=cnt, q=q, tag=tag)
    for key, cl in classes.items():
        npass = sum(1 for m in cl if ev(m["text"], m["v"]))
        if npass == len(cl) and key not in emitted:
            _diff(diffs, probe, "missing_class", cls=str(key), q=q,
                  tag=tag)
        if npass == 0 and key in emitted:
            _diff(diffs, probe, "forbidden_class", cls=str(key), q=q,
                  tag=tag)


def _check_win(runner, q, classes, ev, diffs, tag):
    """Window PARTITION BY keeps every input row: the legal output is a
    fully determined multiset of (member text, class size)."""
    rows, err = runner.run(q)
    if err:
        _diff(diffs, "winsat", "probe_error", err=err, q=q, tag=tag)
        return
    want = Counter()
    for key, cl in classes.items():
        for m in cl:
            if ev(m["text"], m["v"]):
                want[(m["text"], len(cl))] += 1
    got = Counter((r[0], r[1]) for r in rows)
    if got != want:
        _diff(diffs, "winsat", "legal_set_diff", q=q, tag=tag,
              got=[[k, c] for (k, c), n in got.items() for _ in range(n)],
              want=[[k, c] for (k, c), n in want.items()
                    for _ in range(n)])


def probe_saturation(runner, dialect, tables, rng, diffs, stats, ex=False):
    t = rng.choice(tables)
    cand = [c for c in t.cols if c.name != "rid"
            and c.typ in ("num", "flt")]
    if not cand:
        return
    tt = TEXT_T[dialect]
    cols = cand if ex else [rng.choice(cand)]
    for col in cols:
        info = col_info(runner, dialect, t, col.name)
        if info is None:
            return
        classes = info["classes"]
        preds = _sat_preds(dialect, col, classes)
        if not ex:
            preds = preds[:2]
        for sql_pred, ev, tag in preds:
            k = col.name
            # S1: qual on group key inside HAVING (pushdown-prone)
            _check_sat(
                runner,
                f"SELECT CAST({k} AS {tt}) kk, COUNT(*) c FROM {t.name} "
                f"GROUP BY {k} HAVING {sql_pred}",
                classes, ev, diffs, "sat", tag)
            # S2: outer WHERE on text key of grouped subquery
            if tag.startswith("text="):
                litv = sql_pred.rsplit(chr(39), 2)[1]
                _check_sat(
                    runner,
                    f"SELECT kk, c FROM (SELECT CAST({k} AS {tt}) kk, "
                    f"COUNT(*) c FROM {t.name} GROUP BY {k}) s "
                    f"WHERE kk = '{litv}'",
                    classes, ev, diffs, "sat2", tag)
            # W: window PARTITION BY — deterministic legal set (no rep
            # freedom: each row emits its own value); outer WHERE reads
            # the raw key column s.k
            _check_win(
                runner,
                f"SELECT kk, c FROM (SELECT {k}, CAST({k} AS {tt}) kk, "
                f"COUNT(*) OVER (PARTITION BY {k}) c FROM {t.name}) s "
                f"WHERE {sql_pred}",
                classes, ev, diffs, tag)
        stats["sat"] += 1


def _same_family_cols(t: Table):
    fams = defaultdict(list)
    for c in t.cols:
        if c.name == "rid":
            continue
        fam = "num" if c.typ in NUM_FAMILY else c.typ
        fams[fam].append(c.name)
    return fams


def probe_in_exists(runner, dialect, tables, rng, diffs, stats, ex=False):
    if len(tables) < 1:
        return
    t = rng.choice(tables)
    s = rng.choice(tables)
    ft = _same_family_cols(t)
    fs = _same_family_cols(s)
    fam = rng.choice(sorted(set(ft) & set(fs)))
    kt, ks = rng.choice(ft[fam]), rng.choice(fs[fam])
    q_in = (f"SELECT COUNT(*) FROM {t.name} WHERE {kt} IN "
            f"(SELECT {ks} FROM {s.name})")
    # inner alias _s: without it a same-named table self-binds and the
    # correlation never reaches the outer row
    q_ex = (f"SELECT COUNT(*) FROM {t.name} WHERE EXISTS "
            f"(SELECT 1 FROM {s.name} _s WHERE _s.{ks} = "
            f"{t.name}.{kt})")
    ok = _cmp2(runner, diffs, "in_ex", q_in, q_ex, f"{kt}~{ks}")
    q_ni = (f"SELECT COUNT(*) FROM {t.name} WHERE {kt} NOT IN "
            f"(SELECT {ks} FROM {s.name})")
    q_law = (f"SELECT COUNT(*) FROM {t.name} WHERE {kt} IS NOT NULL "
             f"AND NOT EXISTS (SELECT 1 FROM {s.name} _s "
             f"WHERE _s.{ks} IS NULL) "
             f"AND {kt} NOT IN (SELECT {ks} FROM {s.name} "
             f"WHERE {ks} IS NOT NULL)")
    ok &= _cmp2(runner, diffs, "notin_3vl", q_ni, q_law, f"{kt}~{ks}")
    stats["in_ex"] += ok


def probe_setops(runner, dialect, tables, rng, diffs, stats, ex=False):
    atoms = [("SELECT NULL UNION SELECT NULL", 1),
             ("SELECT NULL INTERSECT SELECT NULL", 1),
             ("SELECT NULL EXCEPT SELECT NULL", 0),
             ("SELECT 1 UNION SELECT NULL UNION SELECT 1", 2),
             ("SELECT 1 INTERSECT SELECT NULL", 0)]
    for q, n in atoms:
        rows, err = runner.run(q)
        if err:
            _diff(diffs, "setop", "atom_error", q=q, err=err)
        elif len(rows) != n:
            _diff(diffs, "setop", "atom_card", q=q, got=len(rows), want=n)
    t = rng.choice(tables)
    s = rng.choice(tables)
    ft = _same_family_cols(t)
    fs = _same_family_cols(s)
    fam = rng.choice(sorted(set(ft) & set(fs)))
    kt, ks = rng.choice(ft[fam]), rng.choice(fs[fam])
    A = f"SELECT {kt} FROM {t.name}"
    B = f"SELECT {ks} FROM {s.name}"
    tt = next(c.typ for c in t.cols if c.name == kt)
    ok = _cmp2(runner, diffs, "setop",
               f"{A} UNION {B}",
               f"SELECT DISTINCT * FROM ({A} UNION ALL {B}) u",
               f"{kt}~{ks}", [tt])
    rA, eA = runner.run(A)
    rE, eE = runner.run(f"{A} EXCEPT ALL {B}")
    rI, eI = runner.run(f"{A} INTERSECT ALL {B}")
    if not (eA or eE or eI):
        ty = [tt]
        if cbag(rE, ty) + cbag(rI, ty) != cbag(rA, ty):
            _diff(diffs, "setop", "mset_partition", a=rA[:8],
                  ex=rE[:8], inter=rI[:8])
            ok = False
        elif (bag(rE) + bag(rI) != bag(rA)
              and tt == next(c.typ for c in s.cols if c.name == ks)):
            # rep-level diff only meaningful when operand types match —
            # cross-type setops must coerce (int -> numeric), which is legal
            _diff(diffs, "setop", "mset_rep", a=rA[:8],
                  ex=rE[:8], inter=rI[:8])
            ok = False
    rX, eX = runner.run(f"{A} EXCEPT {A}")
    if not eX and rX:
        _diff(diffs, "setop", "self_except", rows=rX[:8])
        ok = False
    ok &= _cmp2(runner, diffs, "setop",
                f"{A} INTERSECT {A}",
                f"SELECT DISTINCT {kt} FROM {t.name}", kt, [tt])
    stats["setop"] += ok


def probe_join_fiber(runner, dialect, tables, rng, diffs, stats, ex=False):
    a = rng.choice(tables)
    b = rng.choice(tables)
    fa = _same_family_cols(a)
    fb = _same_family_cols(b)
    fams = set(fa) & set(fb)
    if not fams:
        return
    fam = rng.choice(sorted(fams))
    ka, kb = rng.choice(fa[fam]), rng.choice(fb[fam])
    ta = next(c.typ for c in a.cols if c.name == ka)
    tb = next(c.typ for c in b.cols if c.name == kb)
    tt = TEXT_T[dialect]
    ia = [c.name for c in a.cols].index(ka)
    ib = [c.name for c in b.cols].index(kb)
    va = {r[0]: r[ia] for r in a.rows}
    vb = {r[0]: r[ib] for r in b.rows}
    ra_map, e1 = runner.run(
        f"SELECT rid, CAST({ka} AS {tt}) FROM {a.name}")
    rb_map, e2 = runner.run(
        f"SELECT rid, CAST({kb} AS {tt}) FROM {b.name}")
    if e1 or e2:
        return
    tma = {r[0]: r[1] for r in ra_map}
    tmb = {r[0]: r[1] for r in rb_map}
    q = (f"SELECT a.rid, a.{ka}, CAST(a.{ka} AS {tt}), "
         f"b.rid, b.{kb}, CAST(b.{kb} AS {tt}) "
         f"FROM {a.name} a JOIN {b.name} b ON a.{ka} = b.{kb}")
    rows, err = runner.run(q)
    ok = True
    if err:
        _diff(diffs, "fiber", "probe_error", err=err, q=q)
    else:
        for r in rows:
            if not _same_member(r[1], r[2], va.get(r[0]),
                                tma.get(r[0]), ta):
                _diff(diffs, "fiber", "rep_subst_a", rid=r[0],
                      emitted=r[1], text=r[2], want=tma.get(r[0]),
                      q=q)
                ok = False
            if not _same_member(r[4], r[5], vb.get(r[3]),
                                tmb.get(r[3]), tb):
                _diff(diffs, "fiber", "rep_subst_b", rid=r[3],
                      emitted=r[4], text=r[5], want=tmb.get(r[3]),
                      q=q)
                ok = False
    # cardinality law: count = Σ_shared-class |a∩cls|·|b∩cls|
    qc = (f"SELECT COUNT(*) FROM {a.name} a JOIN {b.name} b "
          f"ON a.{ka} = b.{kb}")
    rn, e2 = runner.run(qc)
    if e2:
        _diff(diffs, "card", "probe_error", err=e2, q=qc)
    else:
        ca = Counter(canon(r[ia], ta) for r in a.rows
                     if r[ia] is not None)
        cb = Counter(canon(r[ib], tb) for r in b.rows
                     if r[ib] is not None)
        want = sum(ca[k] * cb[k] for k in ca.keys() & cb.keys())
        if rn and rn[0][0] != want:
            _diff(diffs, "card", "join_card", got=rn[0][0], want=want,
                  ka=ka, kb=kb, q=qc)
            ok = False
    stats["fiber"] += ok


def probe_order_path(runner, dialect, tables, rng, diffs, stats, ex=False):
    if dialect != "pg":
        return
    t = rng.choice(tables)
    col = rng.choice([c for c in t.cols if c.name != "rid"])
    k = col.name
    if runner.exec(f"CREATE INDEX _oix ON {t.name} ({k})"):
        return
    q = f"SELECT {k} FROM {t.name} ORDER BY {k}"
    base, e0 = runner.run(q)
    if e0:
        return
    runner.exec("SET enable_seqscan=off")
    runner.exec("SET enable_bitmapscan=off")
    r_idx, e1 = runner.run(q)
    runner.exec("RESET ALL")
    runner.exec("SET enable_indexscan=off")
    runner.exec("SET enable_indexonlyscan=off")
    runner.exec("SET enable_bitmapscan=off")
    r_seq, e2 = runner.run(q)
    runner.exec("RESET ALL")
    if e1 or e2:
        return
    if r_idx != base or r_seq != base:      # ordered compare, not bag
        _diff(diffs, "order", "path_order_diff", base=base[:10],
              idx=r_idx[:10], seq=r_seq[:10], q=q, col=k)
    else:
        stats["order"] += 1


_ALGO_ARMS = {
    "pg": {
        # legality arms: every plan the optimizer produces must be
        # semantically identical — each GUC forces a different legality
        # assumption to be exercised
        "nohashagg": "SET enable_hashagg=off",
        "nosort": "SET enable_sort=off",
        "nohj": "SET enable_hashjoin=off",
        "nomj": "SET enable_mergejoin=off",
        "nonl": "SET enable_nestloop=off",
        "noseq": "SET enable_seqscan=off",
        "noidx": ("SET enable_indexscan=off; SET enable_indexonlyscan=off;"
                  " SET enable_bitmapscan=off"),
        "noincsort": "SET enable_incremental_sort=off",
        "nopresorted": "SET enable_presorted_aggregate=off",
        "nomemoize": "SET enable_memoize=off",
        "nogathermerge": "SET enable_gathermerge=off",
        "noparhash": "SET enable_parallel_hash=off",
        "nomat": "SET enable_material=off",
        "nosjelim": "SET enable_self_join_elimination=off",
        "nodistreord": "SET enable_distinct_reordering=off",
        "nogbreord": "SET enable_group_by_reordering=off",
        "noasync": "SET enable_async_append=off",
        "noparappend": "SET enable_parallel_append=off",
        "jit": "SET jit=on; SET jit_above_cost=0",
        "lowmem": "SET work_mem='64kB'",
        "par": "SET debug_parallel_query=on",
        "genplan": "SET plan_cache_mode=force_generic_plan",
        "nopruning": "SET enable_partition_pruning=off",
        "pwagg": "SET enable_partitionwise_aggregate=on",
        "ce_on": "SET constraint_exclusion=on",
    },
    "duck": {"t1": "PRAGMA threads=1",
             "t8": "PRAGMA threads=8"},
}


def probe_algo(runner, dialect, tables, rng, diffs, stats, ex=False):
    t = rng.choice(tables)
    fams = _same_family_cols(t)
    fam = rng.choice(sorted(fams))
    k = rng.choice(fams[fam])
    q = f"SELECT {k}, COUNT(*) FROM {t.name} GROUP BY {k}"
    qtypes = [next(c.typ for c in t.cols if c.name == k), "int"]
    if len(tables) > 1 and rng.random() < 0.5:
        s = rng.choice([x for x in tables if x is not t])
        fs = _same_family_cols(s)
        if fam in fs:
            ks = rng.choice(fs[fam])
            q = (f"SELECT COUNT(*) FROM {t.name} a JOIN {s.name} b "
                 f"ON a.{k} = b.{ks}")
            qtypes = ["int"]
    base, e0 = runner.run(q)
    if e0:
        return
    plan0 = runner.explain(q)
    arms = _ALGO_ARMS[dialect]
    for name in (list(arms) if ex else rng.sample(list(arms), 3)):
        runner.exec(arms[name])
        rows, err = runner.run(q)
        plan = runner.explain(q)
        if dialect == "duck":
            runner.exec("PRAGMA threads=4")
        else:
            runner.exec("RESET ALL")
        if err:
            _diff(diffs, "algo", "arm_error", arm=name, err=err, q=q)
            continue
        fired = plan != plan0
        if cbag(rows, qtypes) != cbag(base, qtypes):
            _diff(diffs, "algo", "result_diff", arm=name, fired=fired,
                  base=base[:8], got=rows[:8], q=q)
        elif bag(rows) != bag(base):
            # class-equal but representative differs across algorithms —
            # legal nondeterminism, kept as a weak signal
            _diff(diffs, "algo", "rep_diff", arm=name, fired=fired,
                  base=base[:8], got=rows[:8], q=q)
        elif fired:
            stats[f"algo_fired_{name}"] += 1
    stats["algo"] += 1


def probe_transitive(runner, dialect, tables, rng, diffs, stats,
                     ex=False):
    """Planner ECs assume `=` is transitive. Cross-type coercion breaks it
    (int64 > 2^53 rounds into float8): a.k=b.k ∧ b.k=c.k ∧ ¬a.k=c.k is
    satisfiable. The executor-level gap is data; the violation is when the
    planner *drops* an implied qual — actual join must equal the pairwise
    count computed from the two 2-way probes."""
    a, b, c = (rng.choice(tables) for _ in range(3))
    # the gap needs mixed coercion paths: wide numeric pool = int*/num/flt;
    # prefer b=flt8 between int8/int8 sides (the >2^53 rounding seam)
    def wide(t, want):
        cs = [c.name for c in t.cols
              if c.name != "rid" and
              (c.typ in NUM_FAMILY or c.typ == "flt")]
        pref = [c.name for c in t.cols if c.name != "rid"
                and c.typ in want]
        return pref or cs
    wa, wb, wc = wide(a, ["int8"]), wide(b, ["flt"]), wide(c, ["int8"])
    if not (wa and wb and wc):
        return
    ka, kb, kc = rng.choice(wa), rng.choice(wb), rng.choice(wc)
    nv, ev_ = runner.run(
        f"SELECT COUNT(*) FROM {a.name} a, {b.name} b, {c.name} c "
        f"WHERE a.{ka} = b.{kb} AND b.{kb} = c.{kc} "
        f"AND NOT (a.{ka} = c.{kc})")
    if ev_:
        return
    gap = nv[0][0] > 0
    stats["trans_gap"] += int(gap)
    if not (gap or ex):
        stats["transitive"] += 1
        return
    rab, e1 = runner.run(
        f"SELECT a.rid, b.rid FROM {a.name} a, {b.name} b "
        f"WHERE a.{ka} = b.{kb}")
    rbc, e2 = runner.run(
        f"SELECT b.rid, c.rid FROM {b.name} b, {c.name} c "
        f"WHERE b.{kb} = c.{kc}")
    r3, e3 = runner.run(
        f"SELECT COUNT(*) FROM {a.name} a, {b.name} b, {c.name} c "
        f"WHERE a.{ka} = b.{kb} AND b.{kb} = c.{kc}")
    if e1 or e2 or e3:
        return
    by_b = defaultdict(list)
    for ra, rb in rab:
        by_b[rb].append(ra)
    c_by_b = defaultdict(list)
    for rb, rc in rbc:
        c_by_b[rb].append(rc)
    want = sum(len(by_b[x]) * len(c_by_b[x])
               for x in by_b.keys() & c_by_b.keys())
    if r3 and r3[0][0] != want:
        _diff(diffs, "transitive", "ec_implied", got=r3[0][0], want=want,
              q=f"a.{ka}=b.{kb}=c.{kc}", gap=gap)
    stats["transitive"] += 1


def probe_nsd_join(runner, dialect, tables, rng, diffs, stats, ex=False):
    """IS NOT DISTINCT FROM join ≡ = OR (both NULL), and its count is the
    full class sum including the NULL class."""
    a, b = rng.choice(tables), rng.choice(tables)
    fa, fb = _same_family_cols(a), _same_family_cols(b)
    fams = set(fa) & set(fb)
    if not fams:
        return
    fam = rng.choice(sorted(fams))
    ka, kb = rng.choice(fa[fam]), rng.choice(fb[fam])
    ta = next(c.typ for c in a.cols if c.name == ka)
    tb = next(c.typ for c in b.cols if c.name == kb)
    q1 = (f"SELECT COUNT(*) FROM {a.name} a JOIN {b.name} b "
          f"ON a.{ka} IS NOT DISTINCT FROM b.{kb}")
    q2 = (f"SELECT COUNT(*) FROM {a.name} a JOIN {b.name} b ON "
          f"(a.{ka} = b.{kb} OR (a.{ka} IS NULL AND b.{kb} IS NULL))")
    ok = _cmp2(runner, diffs, "nsd", q1, q2, f"{ka}~{kb}", ["int"])
    r1, e1 = runner.run(q1)
    if not e1:
        ia = [c.name for c in a.cols].index(ka)
        ib = [c.name for c in b.cols].index(kb)
        ca = Counter(canon(r[ia], ta) for r in a.rows)
        cb = Counter(canon(r[ib], tb) for r in b.rows)
        want = sum(ca[k] * cb[k] for k in ca.keys() & cb.keys())
        if r1 and r1[0][0] != want:
            _diff(diffs, "nsd", "nsd_card", got=r1[0][0], want=want,
                  q=q1)
            ok = False
    stats["nsd"] += ok


def probe_distinct_on(runner, dialect, tables, rng, diffs, stats,
                      ex=False):
    """DISTINCT ON (k) ... ORDER BY k, rid is deterministic: each class
    emits its min-rid row's own (k, v) — exact known answer."""
    t = rng.choice(tables)
    cols = [c for c in t.cols if c.name != "rid"]
    if len(cols) < 1:
        return
    k = rng.choice(cols)
    v = rng.choice([c for c in t.cols if c.name != k.name])
    ik = [c.name for c in t.cols].index(k.name)
    iv = [c.name for c in t.cols].index(v.name)
    q = (f"SELECT DISTINCT ON ({k.name}) {k.name}, {v.name} FROM {t.name} "
         f"ORDER BY {k.name}, rid")
    rows, err = runner.run(q)
    if err:
        _diff(diffs, "don", "probe_error", err=err, q=q)
        return
    best = {}
    for row in t.rows:
        key = canon(row[ik], k.typ)
        if key not in best or row[0] < best[key][0]:
            best[key] = row
    want = Counter((key, _nval(best[key][iv], v.typ)) for key in best)
    got = Counter((canon(r[0], k.typ), _nval(r[1], v.typ)) for r in rows)
    if got != want:
        _diff(diffs, "don", "result_diff", q=q,
              got=[[str(a), str(b)] for (a, b), n in got.items()
                   for _ in range(n)][:10],
              want=[[str(a), str(b)] for (a, b), n in want.items()
                    for _ in range(n)][:10])
    else:
        stats["don"] += 1


def _nval(v, typ):
    """Per-value key at normalize() granularity: emitted rows are already
    normalized (Decimal→float, etc.), so stored values must be mapped the
    same way before comparing — else scale/precision artifacts diff."""
    if v is None:
        return None
    if typ in ("int", "int2", "int8"):
        return int(v)
    if typ == "num":
        return float(v)                     # normalize(): Decimal → float
    if typ == "flt":
        return _fltkey(v)
    if typ == "jsb":
        return canon(v, "jsb")
    if typ == "bool":
        return bool(v)
    return str(v)


def probe_in_forms(runner, dialect, tables, rng, diffs, stats, ex=False):
    """k IN (literal list) ≡ k IN (subquery of same literals) ≡
    k = ANY(array). Three evaluators must agree, NULLs included."""
    t, s = rng.choice(tables), rng.choice(tables)
    ft, fs = _same_family_cols(t), _same_family_cols(s)
    fams = set(ft) & set(fs)
    if not fams:
        return
    # literal-list vs subquery typing asymmetry is legal (untyped literals
    # coerce to the compared type; a VALUES column keeps its type) —
    # only same-declared-type pairs share the coercion domain
    fam = rng.choice(sorted(fams))
    kt = rng.choice(ft[fam])
    ttyp = next(c.typ for c in t.cols if c.name == kt)
    same = [x for x in fs[fam]
            if next(c.typ for c in s.cols if c.name == x) == ttyp]
    if not same:
        return
    ks = rng.choice(same)
    styp = ttyp
    vals = list({r[[c.name for c in s.cols].index(ks)]
                 for r in s.rows})
    if not vals:
        return
    pick = rng.sample(vals, k=min(len(vals), rng.randint(1, 4)))
    lits = ", ".join(lit(v, styp, dialect) for v in pick)
    base = f"SELECT COUNT(*) FROM {t.name} WHERE {kt}"
    q_l = f"{base} IN ({lits})"
    q_s = (f"{base} IN (SELECT x FROM (VALUES "
           f"{', '.join('(' + lit(v, styp, dialect) + ')' for v in pick)}"
           f") v(x))")
    ok = _cmp2(runner, diffs, "inlist", q_l, q_s, kt, ["int"])
    if dialect == "pg":
        q_a = f"{base} = ANY (ARRAY[{lits}])"
    else:
        q_a = f"{base} = ANY ([{lits}])"
    ok &= _cmp2(runner, diffs, "inlist", q_l, q_a, kt, ["int"])
    stats["inlist"] += ok


def probe_parallel(runner, dialect, tables, rng, diffs, stats, ex=False):
    """Partial/parallel aggregation must merge transparently over
    doctrine classes."""
    if dialect != "pg":
        return
    t = rng.choice(tables)
    cols = [c for c in t.cols if c.name != "rid"]
    k = rng.choice(cols)
    q = f"SELECT {k.name}, COUNT(*) FROM {t.name} GROUP BY {k.name}"
    base, e0 = runner.run(q)
    if e0:
        return
    runner.exec("SET debug_parallel_query=on")
    rows, err = runner.run(q)
    runner.exec("RESET ALL")
    if err:
        return
    ty = [k.typ, "int"]
    if cbag(rows, ty) != cbag(base, ty):
        _diff(diffs, "paragg", "result_diff", base=base[:8],
              got=rows[:8], q=q)
    elif bag(rows) != bag(base):
        _diff(diffs, "paragg", "rep_diff", base=base[:8],
              got=rows[:8], q=q)
    else:
        stats["paragg"] += 1


def probe_rollup(runner, dialect, tables, rng, diffs, stats, ex=False):
    """ROLLUP(k) = per-class (k, size, 0) + total (NULL, N, 1): the NULL
    group and the super-aggregate row must not collide."""
    t = rng.choice(tables)
    k = rng.choice([c for c in t.cols if c.name != "rid"])
    ik = [c.name for c in t.cols].index(k.name)
    q = (f"SELECT {k.name}, COUNT(*), GROUPING({k.name}) FROM {t.name} "
         f"GROUP BY ROLLUP({k.name})")
    rows, err = runner.run(q)
    if err:
        return
    cls = Counter(canon(r[ik], k.typ) for r in t.rows)
    want = Counter((key, n, 0) for key, n in cls.items())
    want[(("null",), len(t.rows), 1)] += 1
    got = Counter((canon(r[0], k.typ), r[1], r[2]) for r in rows)
    if got != want:
        _diff(diffs, "rollup", "result_diff", q=q,
              got=[[str(a), b, c] for (a, b, c), n in got.items()
                   for _ in range(n)][:10],
              want=[[str(a), b, c] for (a, b, c), n in want.items()
                    for _ in range(n)][:10])
    else:
        stats["rollup"] += 1


def probe_index_range(runner, dialect, tables, rng, diffs, stats,
                      ex=False):
    """Index range scan ≡ seqscan+filter on boundary predicates —
    doctrine boundary values hit the btree boundary logic."""
    if dialect != "pg":
        return
    t = rng.choice(tables)
    k = rng.choice([c for c in t.cols if c.name != "rid"])
    ik = [c.name for c in t.cols].index(k.name)
    members = [r[ik] for r in t.rows if r[ik] is not None]
    if not members:
        return
    if runner.exec(f"CREATE INDEX _rix ON {t.name} ({k.name})"):
        return
    litv = lit(rng.choice(members), k.typ, dialect)
    for op in rng.sample(["<", "<=", ">", ">=", "="], 2):
        q = f"SELECT COUNT(*) FROM {t.name} WHERE {k.name} {op} {litv}"
        base, e0 = runner.run(q)
        if e0:
            continue
        runner.exec("SET enable_seqscan=off")
        ri, e1 = runner.run(q)
        runner.exec("RESET ALL")
        runner.exec("SET enable_indexscan=off")
        runner.exec("SET enable_indexonlyscan=off")
        runner.exec("SET enable_bitmapscan=off")
        rs, e2 = runner.run(q)
        runner.exec("RESET ALL")
        if e1 or e2:
            continue
        if ri != base or rs != base:
            _diff(diffs, "range", "boundary_diff", op=op, lit=litv,
                  base=base, idx=ri, seq=rs, q=q, col=k.name)
        else:
            stats["range"] += 1


# aggregate decomposition: agg(⊎ᵢDᵢ) = combine(agg(Dᵢ)) — the split
# combine map is exact per type (no FP assoc on num/int/text/bool)
_AGG1 = {"count": ("COUNT(*)", "SUM(a)"),
         "sum":   ("SUM({c})", "SUM(a)"),
         "min":   ("MIN({c})", "MIN(a)"),
         "max":   ("MAX({c})", "MAX(a)"),
         "bool_and": ("BOOL_AND({c})", "BOOL_AND(a)"),
         "bool_or":  ("BOOL_OR({c})", "BOOL_OR(a)")}
_AGG2 = {"avg": ("SUM({c}) s, COUNT({c}) n", "SUM(s)/SUM(n)")}
_AGG_FOR = {"int":  ["count", "sum", "min", "max", "avg", "bit_and",
                     "bit_or"],
            "int2": ["count", "sum", "min", "max", "avg", "bit_and",
                     "bit_or"],
            "int8": ["count", "sum", "min", "max", "avg", "bit_and",
                     "bit_or"],
            "num":  ["count", "sum", "min", "max", "avg"],
            "flt":  ["count", "min", "max"],
            "txt":  ["count", "min", "max"],
            "bool": ["count", "bool_and", "bool_or", "min", "max"],
            "jsb":  ["count", "min", "max"]}
_AGG1.update({"bit_and": ("BIT_AND({c})", "BIT_AND(a)"),
              "bit_or":  ("BIT_OR({c})", "BIT_OR(a)")})

_DEC_ARMS = {"par": "SET debug_parallel_query=on",
             "lowmem": "SET work_mem='64kB'",
             "nopresorted": "SET enable_presorted_aggregate=off",
             "pwagg": "SET enable_partitionwise_aggregate=on",
             "jit": "SET jit=on; SET jit_above_cost=0"}


def probe_decompose(runner, dialect, tables, rng, diffs, stats, ex=False):
    """agg over the whole table must equal combine(agg per split) — and
    partial-aggregation paths (parallel, spill) must be transparent."""
    t = rng.choice(tables)
    cols = [c for c in t.cols if c.name != "rid"]
    c = rng.choice(cols)
    aggs = _AGG_FOR.get(c.typ, ["count"])
    a = rng.choice(aggs)
    split_cols = [x.name for x in cols if x.name != c.name]
    if not split_cols:
        return
    g = rng.choice(split_cols)
    q1 = f"SELECT {_fmt_agg(a, c.name)} FROM {t.name}"
    if a in _AGG2:
        inner, outer = _AGG2[a]
        q2 = (f"SELECT {outer} FROM (SELECT "
              f"{inner.format(c=c.name)} FROM {t.name} GROUP BY {g}) s")
    else:
        inner, outer = _AGG1[a]
        q2 = (f"SELECT {outer} FROM (SELECT "
              f"{inner.format(c=c.name)} AS a FROM {t.name} "
              f"GROUP BY {g}) s")
    r1, e1 = runner.run(q1)
    r2, e2 = runner.run(q2)
    if not (e1 or e2):
        typ = "num" if a == "count" else c.typ
        if r1 and r2 and canon(r1[0][0], typ) != canon(r2[0][0], typ):
            _diff(diffs, "dec", "split_diff", agg=a, col=c.name,
                  split=g, base=r1[0], got=r2[0], q1=q1, q2=q2)
        else:
            stats["dec_split"] += 1
    # arms: parallel / spill partial-agg transparency
    q3 = f"SELECT {g}, {_fmt_agg(a, c.name)} FROM {t.name} GROUP BY {g}"
    base, e0 = runner.run(q3)
    if e0:
        return
    gt = next(x.typ for x in t.cols if x.name == g)
    for name in (list(_DEC_ARMS) if ex else rng.sample(list(_DEC_ARMS), 2)):
        if dialect == "duck":
            break
        runner.exec(_DEC_ARMS[name])
        rows, err = runner.run(q3)
        runner.exec("RESET ALL")
        if err:
            continue
        ty = [gt, "num" if a == "count" else c.typ]
        if cbag(rows, ty) != cbag(base, ty):
            _diff(diffs, "dec", "arm_diff", arm=name, agg=a, q=q3,
                  base=base[:8], got=rows[:8])
        elif bag(rows) != bag(base):
            _diff(diffs, "dec", "arm_rep", arm=name, agg=a, q=q3,
                  base=base[:8], got=rows[:8])
        else:
            stats[f"dec_arm_{name}"] += 1
    stats["dec"] += 1


def _fmt_agg(a, c):
    if a == "count":
        return "COUNT(*)"
    return f"{a.upper()}({c})" if a in _AGG1 or a in _AGG2 else f"COUNT({c})"


def probe_recursive(runner, dialect, tables, rng, diffs, stats, ex=False):
    """WITH RECURSIVE fixpoint: engine iteration must equal the
    harness-computed closure; UNION vs UNION ALL and work_mem spill arms."""
    a0 = rng.choice([0, 1, 2])
    s = rng.choice([1, 2, 3])
    k = rng.randint(8, 40)
    # linear chain: expected set is fully determined
    want = {a0 + i * s for i in range(200) if a0 + i * s <= k}
    q = (f"WITH RECURSIVE r(n) AS (SELECT {a0} UNION ALL "
         f"SELECT n + {s} FROM r WHERE n + {s} <= {k}) "
         f"SELECT n FROM r")
    rows, err = runner.run(q)
    if err:
        return
    if {r_[0] for r_ in rows} != want:
        _diff(diffs, "rec", "fixpoint_diff", q=q,
              got=len(rows), want=len(want))
    else:
        stats["rec_linear"] += 1
    # UNION-dedup variant must produce the same set
    q2 = (f"WITH RECURSIVE r(n) AS (SELECT {a0} UNION "
          f"SELECT n + {s} FROM r WHERE n + {s} <= {k}) "
          f"SELECT n FROM r")
    rows2, err2 = runner.run(q2)
    if not err2 and {r_[0] for r_ in rows2} != want:
        _diff(diffs, "rec", "union_fixpoint_diff", q=q2,
              got=len(rows2), want=len(want))
    # graph reachability on a generated edge table
    t = tables[0]
    ecols = [c.name for c in t.cols if c.typ in ("int", "int2")
             and c.name != "rid"]
    if len(ecols) >= 2 and t.rows:
        src, dst = ecols[0], ecols[1]
        i_src = [c.name for c in t.cols].index(src)
        i_dst = [c.name for c in t.cols].index(dst)
        adj = defaultdict(set)
        for row in t.rows:
            if row[i_src] is not None:
                adj[row[i_src]].add(row[i_dst])   # NULL dsts are real rows
        if not adj:
            stats["rec"] += 1
            return
        root = next(iter(adj))
        seen, frontier = {root}, [root]
        while frontier:
            nxt = frontier.pop()
            for m in adj.get(nxt, ()):
                if m not in seen:
                    seen.add(m)
                    frontier.append(m)
        q3 = (f"WITH RECURSIVE r(n) AS (SELECT {root} UNION "
              f"SELECT {dst} FROM r JOIN {t.name} e ON e.{src} = r.n) "
              f"SELECT DISTINCT n FROM r")
        rows3, err3 = runner.run(q3)
        if not err3:
            got = {r_[0] for r_ in rows3}
            if got != seen:
                _diff(diffs, "rec", "reach_diff", q=q3,
                      got=sorted(map(str, got))[:10],
                      want=sorted(map(str, seen))[:10])
            else:
                stats["rec_graph"] += 1
        # work_mem spill arm on the linear chain
        if dialect == "pg":
            runner.exec("SET work_mem='64kB'")
            rows4, err4 = runner.run(q)
            runner.exec("RESET ALL")
            if not err4 and {r_[0] for r_ in rows4} != want:
                _diff(diffs, "rec", "spill_diff", q=q)
    stats["rec"] += 1


def probe_merge(runner, dialect, tables, rng, diffs, stats, ex=False):
    """ON CONFLICT arbiter / MERGE: the unique-index equality domain must
    agree with `=` on doctrine pairs (±0, numeric scale)."""
    if dialect != "pg":
        return
    typ = rng.choice(["num", "flt", "int8"])
    ddl_t = EQ_DDL["pg"][typ]
    pool = {"num": NUM_CLASSES, "flt": FLT_CLASSES}[typ] \
        if typ != "int8" else None
    # build target with class-distinct keys
    t_rows = []
    seen = set()
    src_rows = []
    if pool:
        flat = [v for grp in pool for v in grp]
        pick = rng.sample(flat, min(len(flat), 6))
        for i, v in enumerate(pick):
            ck = canon(v, typ)
            if ck not in seen:
                seen.add(ck)
                (t_rows if i % 2 == 0 else src_rows).append(
                    [i, v, i * 10])
    else:
        vals = rng.sample(INT8_EDGE + [1, 2, 3], 6)
        for i, v in enumerate(vals):
            ck = canon(v, typ)
            if ck not in seen:
                seen.add(ck)
                (t_rows if i % 2 == 0 else src_rows).append(
                    [i, v, i * 10])
    if not src_rows:
        return
    runner.exec("DROP TABLE IF EXISTS _mt")
    runner.exec("DROP TABLE IF EXISTS _ms")
    if runner.exec(f"CREATE TABLE _mt(rid int, k {ddl_t}, v int)"):
        return
    runner.exec(f"CREATE UNIQUE INDEX _mtu ON _mt(k)")
    runner.exec(f"CREATE TABLE _ms(rid int, k {ddl_t}, v int)")
    for r_ in t_rows:
        runner.exec(f"INSERT INTO _mt VALUES "
                    f"({r_[0]},{lit(r_[1], typ, 'pg')},{r_[2]})")
    for r_ in src_rows:
        runner.exec(f"INSERT INTO _ms VALUES "
                    f"({r_[0]},{lit(r_[1], typ, 'pg')},{r_[2]})")
    # expected inserted rids: source classes absent from target
    t_keys = {canon(r_[1], typ) for r_ in t_rows}
    want_ins = {r_[0] for r_ in src_rows
                if canon(r_[1], typ) not in t_keys}
    rows, err = runner.run(
        "INSERT INTO _mt SELECT rid, k, v FROM _ms "
        "ON CONFLICT (k) DO NOTHING RETURNING rid")
    if not err:
        got_ins = {r_[0] for r_ in rows}
        if got_ins != want_ins:
            _diff(diffs, "merge", "arbiter_diff",
                  got=sorted(got_ins), want=sorted(want_ins),
                  t=[[str(x) for x in r_] for r_ in t_rows],
                  s=[[str(x) for x in r_] for r_ in src_rows])
        else:
            stats["merge_conflict"] += 1
    stats["merge"] += 1


def probe_reach(runner, dialect, tables, rng, diffs, stats, ex=False):
    """Inject fake relation stats to reach cost-gated plan shapes, then
    verify semantic equivalence — plan choice must never change results."""
    if dialect != "pg":
        return
    t = rng.choice(tables)
    cols = [c for c in t.cols if c.name != "rid"]
    k = rng.choice(cols)
    q = f"SELECT {k.name}, COUNT(*) FROM {t.name} GROUP BY {k.name}"
    base, e0 = runner.run(q)
    if e0:
        return
    runner.exec(f"ANALYZE {t.name}")
    shapes = [("huge", 5000000, 50000), ("tiny", 10, 2),
              ("mid", 100000, 500)]
    for name, n_tup, n_pg in shapes:
        plan0 = runner.explain(q)
        runner.exec(
            "SELECT pg_restore_relation_stats("
            f"'schemaname','public','relname','{t.name}',"
            f"'relpages',{n_pg}::integer,"
            f"'reltuples',{n_tup}.0::real,"
            f"'relallvisible',{n_pg}::integer)")
        plan = runner.explain(q)
        rows, err = runner.run(q)
        runner.exec(
            f"SELECT pg_clear_relation_stats('public','{t.name}')")
        runner.exec(f"ANALYZE {t.name}")
        if err:
            continue
        fired = plan != plan0
        ty = [k.typ, "int"]
        if cbag(rows, ty) != cbag(base, ty):
            _diff(diffs, "reach", "result_diff", shape=name,
                  fired=fired, base=base[:8], got=rows[:8], q=q)
        elif fired:
            stats[f"reach_fired_{name}"] += 1
    stats["reach"] += 1


PROBES = [probe_triad, probe_partition_exact, probe_saturation,
          probe_in_exists, probe_setops, probe_join_fiber,
          probe_order_path, probe_algo, probe_transitive,
          probe_nsd_join, probe_distinct_on, probe_in_forms,
          probe_parallel, probe_rollup, probe_index_range,
          probe_decompose, probe_recursive, probe_merge, probe_reach]


# ---------------------------------------------------------------- controls

def control_dbs(dialect):
    """Fixed DBs carrying known doctrine-pair bug patterns."""
    ctrls = []
    ctrls.append(("numscale", [Table(
        "t0", [Col("rid", "int"), Col("c0", "num")],
        [[0, Decimal("1")], [1, Decimal("1.0")], [2, Decimal("1.00")],
         [3, Decimal("1.000")], [4, Decimal("2")], [5, Decimal("2.0")],
         [6, Decimal("5")], [7, None]])]))
    ctrls.append(("fltzero", [
        Table("t0", [Col("rid", "int"), Col("c0", "flt")],
              [[0, 0.0], [1, -0.0], [2, 1.0], [3, float("nan")],
               [4, float("inf")], [5, 2.5], [6, None]]),
        Table("t1", [Col("rid", "int"), Col("c0", "flt")],
              [[0, -0.0], [1, 0.0], [2, 2.5], [3, float("nan")],
               [4, None]])]))
    ctrls.append(("intwidth", [
        Table("t0", [Col("rid", "int"), Col("c0", "int2"),
                     Col("c1", "int8")],
              [[0, 1, 1], [1, 2, 2], [2, 3, 3], [3, 1, 1],
               [4, None, 5], [5, 7, None]]),
        Table("t1", [Col("rid", "int"), Col("c0", "int8"),
                     Col("c1", "int")],
              [[0, 1, 1], [1, 3, 3], [2, 2, 2], [3, 5, 5],
               [4, None, 7], [5, 9, None]])]))
    # int8/float8 coercion seam: 2^53+1 = 2^53.0 (rounded), 2^53.0 = 2^53,
    # but the int8 endpoints differ — transitivity gap the planner may
    # exploit wrongly
    ctrls.append(("transedge", [
        Table("t0", [Col("rid", "int"), Col("c0", "int8")],
              [[0, 9007199254740993], [1, 1]]),
        Table("t1", [Col("rid", "int"), Col("c0", "flt")],
              [[0, 9007199254740992.0], [1, 1.0]]),
        Table("t2", [Col("rid", "int"), Col("c0", "int8")],
              [[0, 9007199254740992], [1, 1]])]))
    return ctrls


# ------------------------------------------------------------------ driver

def run_battery(runner, dialect, tables, rng, tag, ex, stats):
    runner.setup(ddl_eq(tables, dialect))
    diffs: list = []
    for probe in PROBES:
        try:
            probe(runner, dialect, tables, rng, diffs, stats, ex)
        except Exception as exc:  # noqa: BLE001
            diffs.append({"probe": probe.__name__, "kind": "harness_error",
                          "err": str(exc)[:300]})
    if diffs:
        schema = {t.name: [(c.name, c.typ) for c in t.cols]
                  for t in tables}
        return {"tag": tag, "schema": schema,
                "rows": {t.name: [[str(v) for v in r] for r in t.rows]
                         for t in tables},
                "diffs": diffs}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["pg", "duck"], required=True)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/absoracle")
    ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.engine == "pg":
        from targets.postgres_runner import PostgresRunner
        tag = Path(args.prefix).name if args.prefix else "embed"
        runner = PGSimple(PostgresRunner(
            f"/tmp/abseq_{tag}", pg_prefix=args.prefix))
        dialect = "pg"
    else:
        import duckdb
        runner = DuckRunner(duckdb)
        dialect = "duck"

    if args.selftest:
        runner.setup(["CREATE TABLE t0(rid int, c0 numeric)",
                      "INSERT INTO t0 VALUES (0,1),(1,1.0),(2,1.00)"])
        diffs: list = []
        orig = runner.run

        def corrupt(sql, *a, **k):
            rows, err = orig(sql, *a, **k)
            if "GROUP BY" in sql.upper() and not err and rows:
                rows = rows[:-1]           # drop a group: triad must fire
            return rows, err
        runner.run = corrupt                # type: ignore[assignment]
        t = Table("t0", [Col("rid", "int"), Col("c0", "num")],
                  [[0, Decimal("1")], [1, Decimal("1.0")],
                   [2, Decimal("1.00")]])
        probe_triad(runner, dialect, [t], random.Random(0), diffs,
                    Counter())
        assert diffs, "tamper not detected — oracle dead"
        print("selftest: tamper detected — OK", diffs[0]["kind"])
        return

    rng = random.Random(args.seed)
    random.seed(args.seed)                  # _sat_preds uses module random
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = out / f"eqcoh_{args.engine}_{args.seed}.jsonl"
    stats = Counter()
    n_diff = 0

    with rec.open("w") as fh:
        for name, tabs in control_dbs(dialect):
            hit = run_battery(runner, dialect, tabs, rng,
                              f"ctl:{name}", True, stats)
            if hit:
                hit["case"] = f"ctl:{name}"
                fh.write(json.dumps(hit, default=str) + "\n")
                fh.flush()
                n_diff += 1
        if not args.controls_only:
            for i in range(args.cases):
                tabs = gen_db_eq(rng, dialect)
                hit = run_battery(runner, dialect, tabs, rng,
                                  str(i), False, stats)
                stats["cases"] += 1
                if hit:
                    hit["case"] = i
                    fh.write(json.dumps(hit, default=str) + "\n")
                    fh.flush()
                    n_diff += 1
    stats["diff_cases"] = n_diff
    print(json.dumps(dict(stats)))


if __name__ == "__main__":
    main()
