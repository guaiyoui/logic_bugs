"""Flat-SQL parser for JOB/STATS-style workloads.

Handles:  SELECT * FROM t1 a, t2 b, ... WHERE cond AND cond ...;
cond := alias.col = alias.col            (join)
      | alias.col OP const               (selection)
      | alias.col BETWEEN c1 AND c2
      | alias.col IN (c1, c2, ...)
      | alias.col IS NOT NULL
Returns (atoms, join classes, table list).  Anything unrecognized is
collected into `skipped` so the caller can decide to drop the query.
"""
from __future__ import annotations

import re
import numpy as np

from engine import Atom

_COND_SPLIT = re.compile(r"\bAND\b", re.I)
_JOIN_RE = re.compile(r"^\s*(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)\s*$")
_PRED_RE = re.compile(
    r"^\s*(\w+)\.(\w+)\s*(=|>=|<=|<>|!=|>|<)\s*(.+?)\s*$")
_BETWEEN_RE = re.compile(
    r"^\s*(\w+)\.(\w+)\s+BETWEEN\s+(.+?)\s+AND\s+(.+?)\s*$", re.I)
_IN_RE = re.compile(r"^\s*(\w+)\.(\w+)\s+(NOT\s+)?IN\s*\((.*)\)\s*$", re.I)
_ISNOTNULL_RE = re.compile(r"^\s*(\w+)\.(\w+)\s+IS\s+NOT\s+NULL\s*$", re.I)
_FROM_ALIAS_RE = re.compile(r"([\w.]+)\s+(?:AS\s+)?(\w+)", re.I)


def _const(tok):
    tok = tok.strip().rstrip(";").strip()
    if (tok.startswith("'") and tok.endswith("'")) or \
       (tok.startswith('"') and tok.endswith('"')):
        return tok[1:-1]
    try:
        return int(tok)
    except ValueError:
        try:
            return float(tok)
        except ValueError:
            return tok


def parse_flat(sql):
    """-> dict(aliases, atoms, skipped, like_flag)"""
    sql = sql.strip()
    m = re.match(r"(?is)select\s+.*?\bfrom\b\s+(.*?)\s+where\s+(.*?)\s*;?\s*$",
                 sql)
    assert m, f"cannot parse: {sql[:120]}"
    froms = [x.strip() for x in m.group(1).split(",")]
    aliases = {}
    for f in froms:
        fm = _FROM_ALIAS_RE.match(f)
        aliases[fm.group(2)] = fm.group(1)
    a2t = aliases

    # gather conditions
    conds = [c.strip() for c in _COND_SPLIT.split(m.group(2)) if c.strip()]
    joins, preds, skipped = [], [], []
    like = False
    for c in conds:
        c = c.rstrip(";").strip()
        if _JOIN_RE.match(c):
            g = _JOIN_RE.match(c).groups()
            if g[0] in a2t and g[2] in a2t:
                joins.append(((g[0], g[1]), (g[2], g[3])))
                continue
            # else: numeric literal (e.g. 2012.0) masquerading as
            # alias.col -- falls through to the predicate branch
        if _ISNOTNULL_RE.match(c):
            continue                       # no-op for our purposes
        mb = _BETWEEN_RE.match(c)
        if mb:
            a, col, lo, hi = mb.groups()
            preds.append((a, col, "between", (_const(lo), _const(hi))))
            continue
        mi = _IN_RE.match(c)
        if mi:
            a, col, neg, lst = mi.groups()
            vals = [_const(t) for t in lst.split(",")]
            preds.append((a, col, "notin" if neg else "in", vals))
            continue
        mp = _PRED_RE.match(c)
        if mp and mp.group(1) in a2t:
            a, col, op, val = mp.groups()
            if "like" in op.lower() or "~~" in op:
                like = True
                continue
            v = _const(val)
            if isinstance(v, str) and op in (">", "<", ">=", "<="):
                # string range (e.g. date) -- keep as string, loader handles
                pass
            preds.append((a, col, op, v))
            continue
        skipped.append(c)

    # union-find join attributes -> logical variables
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (a1, c1), (a2, c2) in joins:
        union((a1, c1), (a2, c2))
    members = {}
    for (a1, c1), (a2, c2) in joins:
        for x in ((a1, c1), (a2, c2)):
            members.setdefault(find(x), []).append(x)
    # attribute name = canonical "table.col" of class repr
    attr_of = {}
    for root, mem in members.items():
        name = min(f"{a}.{c}" for a, c in mem)
        for mc in mem:
            attr_of[mc] = name

    atoms = {}
    for (a1, c1), (a2, c2) in joins:
        for (a, c) in ((a1, c1), (a2, c2)):
            atoms.setdefault(a, {})[attr_of[(a, c)]] = c

    pred_by = {}
    for a, col, op, v in preds:
        pred_by.setdefault(a, []).append((col, op, v))

    return dict(aliases=a2t, atoms=atoms, joins=joins,
                pred_by=pred_by, skipped=skipped, like=like)


def _adapt_const(v, dtype):
    """Match a string/loose constant to the column dtype. Fail loudly if
    a string cannot be interpreted (silent mismatch -> wrong bound)."""
    if isinstance(v, str):
        if np.issubdtype(dtype, np.integer):
            try:
                return int(v)
            except ValueError:
                pass
            if re.match(r"^\d{4}[-/]\d{1,2}", v):
                import pandas as _pd
                return _pd.Timestamp(v).value
            raise TypeError(f"cannot adapt string const {v!r} to {dtype}")
        if np.issubdtype(dtype, np.floating):
            try:
                return float(v)
            except ValueError:
                raise TypeError(f"cannot adapt string const {v!r} to {dtype}")
    if isinstance(v, (tuple, frozenset)) and not isinstance(v, str):
        return tuple(_adapt_const(x, dtype) for x in v)
    return v


def _cmp(x, op, v):
    """Compare array x to v with SQL NULL semantics: any comparison
    involving NULL is false.  Object arrays may hold float NaN sentinels
    mixed with strings, so go through pandas for them."""
    if x.dtype == object:
        import pandas as _pd
        s = _pd.Series(x)
        ok = s.notna().to_numpy()
        cur = np.zeros(len(x), bool)
        if op in ("in", "notin"):
            hit = np.isin(x, list(v))
            cur = hit & ok if op == "in" else (~hit) & ok
            return cur
        sv = s[ok]
        try:
            res = getattr(sv, _OPS[op])(v)
        except TypeError:
            res = getattr(sv.astype(str), _OPS[op])(str(v))
        cur[ok] = res.to_numpy(bool)
        return cur
    if op == "in":
        return np.isin(x, list(v))
    if op == "notin":
        return ~np.isin(x, list(v))
    return getattr(x, _OPS[op])(v)


_OPS = {"=": "__eq__", ">": "__gt__", "<": "__lt__", ">=": "__ge__",
        "<=": "__le__", "<>": "__ne__", "!=": "__ne__"}


def make_pred(pred_list):
    """pred_list: [(col, op, val)] -> callable rows(dict)->mask."""
    def f(rows):
        m = None
        for col, op, v in pred_list:
            x = rows[col]
            v = _adapt_const(v, x.dtype)
            if op == "between":
                cur = _cmp(x, ">=", v[0]) & _cmp(x, "<=", v[1])
            else:
                cur = _cmp(x, op, v)
            m = cur if m is None else (m & cur)
        return m if m is not None else np.ones(
            len(next(iter(rows.values()))), dtype=bool)
    return f


def to_query(parsed, db_unused=None):
    """parsed -> engine.Query. Each atom gets a private row-id attribute
    so that h(V0) counts row-combinations (COUNT(*) semantics), not just
    distinct join-key projections."""
    atoms = []
    for a, colmap in parsed["atoms"].items():
        table = parsed["aliases"][a]
        preds = parsed["pred_by"].get(a)
        cm = dict(colmap)
        cm[f"{a}#"] = "__rowid__"
        atoms.append(Atom(table, cm,
                          pred=make_pred(preds) if preds else None))
    # aliases that appear in FROM but have no join attrs are dropped
    # (cartesian product -- JOB queries don't have them anyway)
    from engine import Query
    return Query(atoms)
