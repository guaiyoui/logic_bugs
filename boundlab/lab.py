"""boundlab: mini auto-research lab for join-size bounds.

Setup: binary equi-join R(X) |x| S(X) with degree sequences a, b (sorted desc).
Ground truth: the worst-case join size consistent with the degree sequences is
    J*(a,b) = sum_i a_i * b_i          (rearrangement inequality)

Pipeline = the auto-research loop:
  1. propose   : candidate inequalities f(stats(a), stats(b)) >= J*(a,b)
                 (played by the agent here; swap in an LLM API later)
  2. verify    : random degree sequences across regimes + structured edge
                 cases + adversarial perturbation -> sound or refuted w/ cex
  3. score     : bound/actual ratio on sound candidates, vs baselines
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(7)

# --------------------------------------------------------------- statistics
def stats(d: np.ndarray) -> dict:
    n = len(d)
    return {
        "n": n,
        "l1": float(d.sum()),
        "l2": float(np.sqrt((d.astype(float) ** 2).sum())),
        "l3": float((d.astype(float) ** 3).sum() ** (1 / 3)),
        "linf": float(d[0]),
        "lmin": float(d[-1]),
    }


def prefix_l2(d: np.ndarray, k: int) -> float:
    """l2-norm of the top-k entries only -- a 'new statistic' an LLM can invent."""
    return float(np.sqrt((d[:k].astype(float) ** 2).sum()))


def prefix_l1(d: np.ndarray, k: int) -> float:
    return float(d[:k].sum())


def actual_max_join(a: np.ndarray, b: np.ndarray) -> float:
    """Worst-case join size given both degree sequences (sorted desc)."""
    m = min(len(a), len(b))
    return float((a[:m] * b[:m]).sum())


# ------------------------------------------------------------- instance gen
def sample_degseq(kind: str, n_lo=5, n_hi=400) -> np.ndarray:
    n = int(rng.integers(n_lo, n_hi))
    if kind == "uniform":
        d = rng.integers(1, 30, n)
    elif kind == "zipf":
        d = 1.0 / np.power(np.arange(1, n + 1), rng.uniform(0.6, 1.8))
        d = np.maximum(1, np.round(d / d.max() * rng.integers(5, 500))).astype(int)
    elif kind == "bimodal":            # few heavy hitters + long light tail
        k = max(1, n // 10)
        d = np.concatenate([rng.integers(100, 1000, k), rng.integers(1, 15, n - k)])
    elif kind == "flat_small":         # tiny n, big degrees -> stresses wrong pairs
        n = int(rng.integers(2, 6))
        d = rng.integers(50, 2000, n)
    else:
        raise ValueError(kind)
    return np.sort(d)[::-1].astype(float)


REGIMES = ["uniform", "zipf", "bimodal", "flat_small"]


# ---------------------------------------------------- candidates & baselines
# signature: f(sa, sb, a, b) -> claimed upper bound on actual_max_join(a,b)
def b_agm(sa, sb, a, b):      # baseline 1: cardinality x max-degree
    return min(sa["l1"] * sb["linf"], sa["linf"] * sb["l1"])


def b_cs(sa, sb, a, b):       # baseline 2: Cauchy-Schwarz (l2 x l2)
    return sa["l2"] * sb["l2"]


def b_lpmin(sa, sb, a, b):    # baseline 3: LpBound analog, min over conjugate p,q
    out = np.inf
    for p in [1.0, 1.5, 2.0, 3.0, 4.0, np.inf]:
        q = p / (p - 1) if p != 1 else np.inf
        np_ = sa["linf"] if p == np.inf else float((a.astype(float) ** p).sum() ** (1 / p))
        nq_ = sb["l1"] if q == np.inf else float((b.astype(float) ** q).sum() ** (1 / q))
        out = min(out, np_ * nq_)
        nq2 = sb["linf"] if p == 1 else None
    # symmetric direction
    for p in [1.5, 2.0, 3.0, 4.0]:
        np2 = float((b.astype(float) ** p).sum() ** (1 / p))
        q = p / (p - 1)
        nq2v = float((a.astype(float) ** q).sum() ** (1 / q))
        out = min(out, np2 * nq2v)
    return out


# --- LLM-proposed candidates (mixed: some sound, some plausible-but-false) ---
def c_amgm(sa, sb, a, b):     # valid (AM-GM) but never beats CS
    return (sa["l2"] ** 2 + sb["l2"] ** 2) / 2


def c_holder_wrong(sa, sb, a, b):   # plausible: l3 x l2 -- NOT conjugate -> false
    return sa["l3"] * sb["l2"]


def c_trunc_cs(sa, sb, a, b):  # support-truncated CS: NEW stat prefix-l2 @ min(n_a,n_b)
    k = min(sa["n"], sb["n"])
    return min(sa["l2"] * prefix_l2(b, k), sb["l2"] * prefix_l2(a, k))


def c_gap_sub(sa, sb, a, b):   # tempting 'CS minus gap' -> false
    return sa["l2"] * sb["l2"] - (a[0] - a[-1]) * (b[0] - b[-1])


def c_trunc_l1(sa, sb, a, b):  # support-truncated l_inf x l_1 variant
    k = min(sa["n"], sb["n"])
    return min(sa["linf"] * prefix_l1(b, k), sb["linf"] * prefix_l1(a, k))


BASELINES = {"agm(l1*linf)": b_agm, "cs(l2*l2)": b_cs, "lpbound-min": b_lpmin}
CANDIDATES = {
    "amgm": c_amgm,
    "holder_3x2 (non-conj)": c_holder_wrong,
    "trunc-cs (new stat)": c_trunc_cs,
    "cs-minus-gap": c_gap_sub,
    "trunc-linf*l1": c_trunc_l1,
}


# ------------------------------------------------------------------ verifier
def verify(fn, n_rand=4000, adv_rounds=6):
    """Return (sound, worst_violation, counterexample)."""
    worst_v, cex = -np.inf, None

    def check(a, b):
        nonlocal worst_v, cex
        v = actual_max_join(a, b) - fn(stats(a), stats(b), a, b)
        if v > worst_v:
            worst_v, cex = v, (a.copy(), b.copy())

    for _ in range(n_rand):
        a = sample_degseq(rng.choice(REGIMES))
        b = sample_degseq(rng.choice(REGIMES))
        check(a, b)
    # structured edge cases
    for n in range(1, 40):
        check(np.full(n, float(n + 5)), np.full(n, float(n + 5)))   # all-equal
        check(np.array([1000.] + [1.] * n), np.array([1000.] + [1.] * n))
        check(np.arange(n, 0, -1).astype(float) * 7,
              np.arange(n, 0, -1).astype(float) * 3)
    # adversarial: perturb the best counterexample found so far
    for _ in range(adv_rounds):
        if cex is None:
            break
        a, b = cex
        for _ in range(2000):
            a2 = np.maximum(1, a * rng.uniform(0.5, 2.0, len(a)))
            b2 = np.maximum(1, b * rng.uniform(0.5, 2.0, len(b)))
            check(np.sort(a2)[::-1], np.sort(b2)[::-1])
    sound = worst_v <= 1e-6 * max(1.0, abs(worst_v))
    return sound, worst_v, cex


# -------------------------------------------------------------------- eval
def main():
    print(f"{'name':<26}{'sound?':<10}{'mean':>8}{'p90':>8}{'max':>8}   ratio=bound/actual")
    rows = {}
    insts = [(sample_degseq(k), sample_degseq(k))
             for k in REGIMES * 250]
    for name, fn in {**BASELINES, **CANDIDATES}.items():
        sound, v, cex = verify(fn)
        if sound:
            r = np.array([fn(stats(a), stats(b), a, b) / actual_max_join(a, b)
                          for a, b in insts])
            rows[name] = r
            print(f"{name:<26}{'SOUND':<10}{r.mean():>8.2f}{np.percentile(r,90):>8.2f}"
                  f"{r.max():>8.2f}")
        else:
            print(f"{name:<26}{'REFUTED':<10}{'':>8}{'':>8}{'':>8}"
                  f"  cex: a={cex[0].astype(int)[:6]} b={cex[1].astype(int)[:6]}")

    print("\n-- best candidate vs best baseline, per instance --")
    base = np.minimum.reduce([rows[k] for k in BASELINES])
    for name in CANDIDATES:
        if name in rows:
            win = (rows[name] < base - 1e-9).mean()
            comb = np.minimum(base, rows[name])
            print(f"{name:<26} tighter than best-baseline on {win*100:5.1f}% of instances"
                  f"   mean improvement {(base/rows[name]).mean():.3f}x"
                  f"   combined-bound mean ratio {comb.mean():.3f} (base {base.mean():.3f})")


if __name__ == "__main__":
    main()
