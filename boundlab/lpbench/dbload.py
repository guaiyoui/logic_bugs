"""Dataset loaders -> db dict {table: {col: np.ndarray}}.

Join columns are int64 (strings factorized to codes, NaN -> -1 which
never matches anything since all real keys are >= 0).
Predicate columns keep SQL semantics: NaN fails every comparison.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(D, "data_dl", "lpbound_repo")


def _to_key(s, salt=0):
    """Join key -> int64 in a SHARED key space across tables:
    numerics keep their value; true NaN/None -> globally-unique negative
    per (column-block, row) so NULL never joins; non-numeric strings ->
    stable hash so equal strings in different tables still join.
    Key spaces: ids >= 0 | string hashes in (-2^32,-2] | NULLs <= -2^45.
    """
    import zlib
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().all():
        return v.to_numpy(np.int64)
    out = v.to_numpy(np.float64, copy=True)     # copy! view would mutate df
    out[~np.isfinite(out)] = 0
    out = out.astype(np.int64)
    nan_mask = s.isna().to_numpy()                  # real NULLs
    idx = np.nonzero(nan_mask)[0]
    base = (np.int64(2) ** 45 +
            np.int64(salt % 1024) * (np.int64(2) ** 27))
    out[idx] = -(base + idx.astype(np.int64))
    str_mask = ~nan_mask & ~v.notna().to_numpy()    # non-numeric strings
    for i in np.nonzero(str_mask)[0]:
        out[i] = -(zlib.crc32(str(s.iat[i]).encode()) + 2)
    return out


def _to_pred(s):
    """Predicate column: numeric->float64, datetime->int64, else raw
    object array (equality/range evaluated with Python semantics; NULL
    fails comparisons like SQL)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64").to_numpy(np.float64)
    if s.dtype == object or str(s.dtype).startswith("str"):
        v = pd.to_numeric(s, errors="coerce")
        if v.notna().mean() > 0.95:
            return v.to_numpy(np.float64)
        dt = pd.to_datetime(s, errors="coerce", format="mixed")
        if dt.notna().mean() > 0.95:
            return dt.astype("int64").to_numpy(np.int64)
        return s.to_numpy(object)
    if np.issubdtype(s.dtype, np.number):
        return s.to_numpy(np.float64)
    return s.to_numpy(object)


def _split_pred_consts(preds):
    """Return set of string constants used with range ops (unsupported
    on factorized cols)."""
    bad = set()
    for col, op, v in preds:
        if isinstance(v, str) and op in (">", "<", ">=", "<=", "between"):
            bad.add(col)
    return bad


def load_csv_tables(path_map, needed_cols, key_cols, delim=",", header=0):
    """path_map: table->csv path; needed_cols: table->set(cols);
    key_cols: table->set(cols) that are join keys."""
    db = {}
    for t, path in path_map.items():
        cols = needed_cols.get(t)
        if not cols:
            continue
        df = pd.read_csv(path, sep=delim, header=header,
                         usecols=lambda c: c in cols)
        db[t] = {}
        for c in df.columns:
            if c in key_cols.get(t, ()):
                db[t][c] = _to_key(df[c])
            else:
                db[t][c] = _to_pred(df[c])
    return db


# ------------------------------------------------------------------ dblp
def load_dblp():
    vd = os.path.join(REPO, "data", "datasets", "dblp")
    v = pd.read_csv(os.path.join(vd, "dblp_vertex.csv"),
                    sep="|", header=None, names=["i", "l", "d"])
    e = pd.read_csv(os.path.join(vd, "dblp_edge_undirected.csv"),
                    sep="|", header=None, names=["s", "t"])
    return {
        "dblp.vertex": {"i": v["i"].to_numpy(np.int64),
                        "l": v["l"].to_numpy(np.float64),
                        "d": v["d"].to_numpy(np.float64)},
        "dblp.edge": {"s": e["s"].to_numpy(np.int64),
                      "t": e["t"].to_numpy(np.int64)},
    }, {"dblp.vertex": v, "dblp.edge": e}


# ------------------------------------------------------------------ stats
def load_stats(needed_cols, key_cols):
    sd = os.path.join(REPO, "data", "datasets", "stats")
    paths = {os.path.splitext(f)[0]: os.path.join(sd, f)
             for f in os.listdir(sd) if f.endswith(".csv")}
    return load_csv_tables(paths, needed_cols, key_cols)


# ------------------------------------------------------------------- imdb
_IMDB_SCHEMA = None


def imdb_schema():
    global _IMDB_SCHEMA
    if _IMDB_SCHEMA is None:
        import json
        raw = json.load(open(os.path.join(D, "imdb_schema.json")))
        _IMDB_SCHEMA = {t: [c for c, _ in cols] for t, cols in raw.items()}
    return _IMDB_SCHEMA


def imdb_types():
    import json
    return json.load(open(os.path.join(D, "imdb_schema.json")))


def _frame_to_db(df, table, needed, keys):
    import zlib
    out = {}
    for c in df.columns:
        if c not in needed:
            continue
        if c in keys.get(table, ()):
            out[c] = _to_key(df[c], zlib.crc32(f"{table}.{c}".encode()))
        else:
            out[c] = _to_pred(df[c])
    return out


def load_imdb(needed_cols, key_cols, imdb_dir):
    """IMDB CSVs are headerless; column order comes from the JOB schema.
    Returns (db, raw_frames) -- same parsed rows for stats and truth."""
    names = imdb_schema()
    types = imdb_types()
    db, raw = {}, {}
    for t, cols in needed_cols.items():
        if t not in names:
            continue
        path = os.path.join(imdb_dir, t + ".csv")
        df = pd.read_csv(path, header=None, names=names[t],
                         dtype=str, keep_default_na=False,
                         na_values=[""], quoting=0, on_bad_lines="skip")
        for c, ty in types[t]:
            if ty in ("integer", "bigint"):
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
            else:
                df[c] = df[c].replace({"": None})
        raw[t] = df[list(cols)] if cols else df
        db[t] = _frame_to_db(df, t, cols, key_cols)
    return db, raw


def load_stats_raw(needed_cols, key_cols):
    """STATS CSVs (headered) -> (db, raw_frames)."""
    sd = os.path.join(REPO, "data", "datasets", "stats")
    db, raw = {}, {}
    for f in os.listdir(sd):
        if not f.endswith(".csv"):
            continue
        t = os.path.splitext(f)[0]
        if t not in needed_cols:
            continue
        df = pd.read_csv(os.path.join(sd, f))
        raw[t] = df
        db[t] = _frame_to_db(df, t, needed_cols[t], key_cols)
    return db, raw
