"""Datasets for lpbench.

Relations are sets of distinct tuples on their mapped columns.  To give
join keys non-trivial degrees, every table carries a second column:
for key j with degree d we emit d distinct (j, partner) rows, so the
fiber of j is exactly d.  Optional numeric payload 'p' for predicates.
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(11)


def zipf_degs(n, alpha, scale=2000):
    d = 1.0 / np.power(np.arange(1, n + 1), alpha)
    return np.maximum(1, np.round(d / d[0] * scale)).astype(np.int64)


def rel_pairs(kdegs, pdom=200_000, payload=None):
    """Rows (key, partner, [p]); for key j emit kdegs[j] DISTINCT partners."""
    keys, vals = [], []
    for j, d in enumerate(kdegs):
        d = int(d)
        if d == 0:
            continue
        v = rng.choice(pdom, size=d, replace=False)
        keys.append(np.full(d, j, dtype=np.int64))
        vals.append(v)
    rows = {"k": np.concatenate(keys), "v": np.concatenate(vals)}
    if payload == "rand":
        rows["p"] = rng.integers(0, 100, len(rows["k"]))
    elif payload == "keycorr":
        rows["p"] = (rows["k"] % 100).astype(np.int64)
    return rows


def synth_sets():
    """Binary-join scenarios R(k,v) |x| S(k,w) on shared key k."""
    n, alpha = 3000, 1.1
    a = zipf_degs(n, alpha)
    b = zipf_degs(n, alpha)
    out = {
        # same key space, heavy hitters aligned (worst case for bounds)
        "sym":  {"R": rel_pairs(a, payload="rand"),
                 "S": rel_pairs(b, payload="rand")},
        # same key space, ranks reversed: R's hot keys are S's cold keys
        "anti_rank": {"R": rel_pairs(a, payload="rand"),
                      "S": rel_pairs(b[np.argsort(a)], payload="rand")},
        # disjoint supports: S's keys live in [n, 2n)
        "anti_dom": {"R": rel_pairs(a, payload="rand"),
                     "S": rel_pairs(b, payload="rand")},
        # asymmetric support: n keys vs 40 hot keys
        "asym": {"R": rel_pairs(a, payload="rand"),
                 "T": rel_pairs(zipf_degs(40, 0.8, scale=8000))},
        # near-uniform degrees (hard to distinguish, sanity check)
        "unif": {"R": rel_pairs(rng.integers(1, 10, n), payload="rand"),
                 "S": rel_pairs(rng.integers(1, 10, n), payload="rand")},
    }
    out["anti_dom"]["S"]["k"] = out["anti_dom"]["S"]["k"] + n
    return out


def chain3(n=2000, alpha=1.1, anti=True, scale=900):
    """R(x,y) - S(y,z) - T(z,u); if anti, S.z ranks reversed vs T.z."""
    dy, dz, du = (zipf_degs(n, alpha, scale) for _ in range(3))
    Ry = np.repeat(np.arange(n), dy)
    Rx = rng.integers(0, 3 * n, len(Ry))
    # ensure (x,y) distinct
    key = Ry * (3 * n + 1) + Rx
    _, keep = np.unique(key, return_index=True)
    R = {"y": Ry[keep], "x": Rx[keep]}
    Sz_deg = np.zeros(n, dtype=np.int64)
    if anti:
        Sz_deg[np.argsort(dz)] = np.sort(dz)[::-1]      # rank-reversed
    else:
        Sz_deg = dz.copy()
    Sz = np.repeat(np.arange(n), Sz_deg)
    Sy = rng.integers(0, n, len(Sz))
    key = Sz * (3 * n + 1) + Sy
    _, keep = np.unique(key, return_index=True)
    S = {"z": Sz[keep], "y": Sy[keep]}
    Tz = np.repeat(np.arange(n), dz)
    Tu = rng.integers(0, 3 * n, len(Tz))
    key = Tz * (3 * n + 1) + Tu
    _, keep = np.unique(key, return_index=True)
    T = {"z": Tz[keep], "u": Tu[keep]}
    return {"R": R, "S": S, "T": T}


def chain4(n=1500, alpha=1.1, scale=600):
    """R(x,y)-S(y,z)-T(z,u)-V(u,w).  T.z degrees rank-reversed vs S.z."""
    dy, dz, dv = (zipf_degs(n, alpha, scale) for _ in range(3))
    Ry = np.repeat(np.arange(n), dy)
    Rx = rng.integers(0, 4 * n, len(Ry))
    Sz = np.repeat(np.arange(n), dz)
    Sy = rng.integers(0, n, len(Sz))
    Tz_deg = np.zeros(n, dtype=np.int64)
    Tz_deg[np.argsort(dz)] = np.sort(dz)[::-1]          # rank-reversed
    Tz = np.repeat(np.arange(n), Tz_deg)
    Tu = rng.integers(0, n, len(Tz))
    Vu = np.repeat(np.arange(n), dv)
    Vw = rng.integers(0, 4 * n, len(Vu))
    out = {"R": {"x": Rx, "y": Ry}, "S": {"y": Sy, "z": Sz},
           "T": {"z": Tz, "u": Tu}, "V": {"u": Vu, "w": Vw}}
    for t in out.values():
        a, b = sorted(t)
        key = encode_key(t[a], t[b])
        _, keep = np.unique(key, return_index=True)
        t[a], t[b] = t[a][keep], t[b][keep]
    return out


def encode_key(x, y):
    return x * (int(y.max()) + 1) + y


def load_patents(path="/home/user/work/join_bound/cit-Patents.txt.gz",
                 max_edges=None, undirected=False):
    import gzip
    s, d = [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            u, v = line.split()
            s.append(int(u)); d.append(int(v))
            if max_edges and len(s) >= max_edges:
                break
    s = np.array(s, dtype=np.int64); d = np.array(d, dtype=np.int64)
    if undirected:
        s, d = np.concatenate([s, d]), np.concatenate([d, s])
        # dedup so E stays a set of distinct (s,d) tuples
        key = s * (s.max() + 1) + d
        _, keep = np.unique(key, return_index=True)
        s, d = s[keep], d[keep]
    return {"E": {"s": s, "d": d}}
