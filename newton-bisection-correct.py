"""
hybrid_roots.py
Find *all real* roots of a real-coefficient polynomial with

    • Newton–Raphson  (O(log log 1/eps) iterations) – primary
    • Bisection       (O(log 1/eps)   iterations) – safe fallback

Dependencies: NumPy only.  No Horner rule; evaluation uses multiply.accumulate.
"""

import numpy as np
import csv, sys, time
from typing import List

sys.setrecursionlimit(10_000)
TOL = 1e-12                        # root precision
MAX_NR = 50                       # NR iterations before giving up


# ─────────────────────────────── I/O ─────────────────────────────────────────

def read_csv(path: str) -> np.ndarray:
    """Read one-column CSV → NumPy array (highest degree first)."""
    with open(path) as f:
        return np.array([float(r[0]) for r in csv.reader(f)], dtype=float)


# ───────────────────── polynomial helpers (no Horner) ───────────────────────

BIG = 1.0e300                 # largest finite float we ever return

def poly_val(p: np.ndarray, x: float) -> float:
    """
    Evaluate p(x) without overflow.

    • For |x| <= X_SAFE  : use the original multiply.accumulate path
    • For |x| >  X_SAFE  :   p(x) = x^n * q(1/x)
        – q() is still evaluated with multiply.accumulate (now at |1/x|<1)
        – only the **sign** and an upper-bounded magnitude are returned
          (good enough for bracketing & convergence tests)
    """
    n = len(p) - 1
    # -------- threshold below which x**n fits in 64-bit float ----------
    X_SAFE = np.exp(700.0 / max(1, n))      # because log(1e308) ≈ 709

    if abs(x) <= X_SAFE:
        # ---- plain accumulate ----
        xx = np.full(n + 1, x, dtype=float)
        xx[0] = 1.0
        xx = np.multiply.accumulate(xx)
        return float(np.dot(xx[::-1], p))

    # --------  |x| is huge: rewrite p(x) = x^n * q(1/x)  --------------
    y = 1.0 / x                              # |y| < 1   → safe powers
    q = p[::-1]                              # coeffs of q(t)
    # q(y) with the *small* argument
    yy = np.full(n + 1, y, dtype=float)
    yy[0] = 1.0
    yy = np.multiply.accumulate(yy)
    qy = float(np.dot(yy[::-1], q))

    # sign(x^n) = sign(x)^n  ( = 1 if n even, sign(x) if n odd )
    sign_xn = 1.0 if (n % 2 == 0) else np.sign(x)
    value = sign_xn * qy

    # we do **not** multiply by |x|^n (would overflow) – instead cap
    return np.clip(value, -BIG, BIG)



def derivative(p: np.ndarray) -> np.ndarray:
    """Return coefficient array of p′(x)."""
    n = p.size - 1
    if n == 0:
        return np.array([0.0])
    return p[:-1] * np.arange(n, 0, -1, dtype=float)


def normalise(p: np.ndarray) -> np.ndarray:
    """Divide by max-abs coeff ⇒ keeps values near |1|, roots unchanged."""
    m = np.max(np.abs(p))
    return p / m if m else p


# ───────────────────────── radius / bracketing ──────────────────────────────

def fujiwara_bound(p: np.ndarray) -> float:
    """
    2 * max_k |a_k / a_0|^{1/k}.  After normalisation this is very tight.
    """
    a0 = p[0]
    n = p.size - 1
    k = np.arange(1, n + 1)
    return 2.0 * np.max(np.abs(p[1:] / a0) ** (1.0 / k))


# ───────────────────── root solvers on one interval ─────────────────────────

def newton_raphson(p: np.ndarray, dp: np.ndarray,
                   x0: float, a: float, b: float,
                   tol: float = TOL, max_it: int = MAX_NR) -> float | None:
    """
    Newton iteration starting at x0.  Return root or None if it fails.
    """
    x = x0
    for _ in range(max_it):
        fx = poly_val(p, x)
        if abs(fx) < tol:
            return x
        dfx = poly_val(dp, x)
        if dfx == 0.0:
            break
        x_new = x - fx / dfx
        if not (a <= x_new <= b):
            break
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return None


def bisection(p: np.ndarray, a: float, b: float,
              tol: float = TOL) -> float:
    """
    Guaranteed root finder assuming p(a)·p(b)<0.
    """
    fa, fb = poly_val(p, a), poly_val(p, b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b

    while b - a > tol:
        m = 0.5 * (a + b)
        fm = poly_val(p, m)
        if fm == 0.0 or (b - a) < tol:
            return m
        if np.sign(fa) * np.sign(fm) < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def hybrid_interval_root(p: np.ndarray, a: float, b: float) -> float:
    """
    Try Newton first; if it fails, fall back to bisection (then refine once).
    """
    dp = derivative(p)
    mid = 0.5 * (a + b)
    root = newton_raphson(p, dp, mid, a, b)
    if root is None:
        root = bisection(p, a, b)
        # one shot of NR to polish
        root = newton_raphson(p, dp, root, a, b) or root
    return root


# ───────────────────────── all-real-roots routine ───────────────────────────

def real_roots(p: np.ndarray, tol: float = TOL) -> List[float]:
    """
    Recursively isolate via derivative critical points, then solve each bracket.
    """
    p = normalise(p)                            # keeps numbers tame
    deg = p.size - 1
    if deg == 0:
        return []
    if deg == 1:
        return [-p[1] / p[0]]

    # recurse on derivative
    crit = np.asarray(sorted(real_roots(derivative(p), tol)))
    B = fujiwara_bound(p)
    endpoints = np.concatenate(([-B], crit, [B]))

    roots: List[float] = []
    for lo, hi in zip(endpoints[:-1], endpoints[1:]):
        flo, fhi = poly_val(p, lo), poly_val(p, hi)
        # handle root exactly at a critical point
        if abs(flo) < tol:
            roots.append(lo)
        if abs(fhi) < tol:
            roots.append(hi)
        # standard sign-change bracket
        if flo * fhi < 0:
            roots.append(hybrid_interval_root(p, lo, hi))

    # unique + sorted, rounded to tolerance decimal places
    digits = int(-np.log10(tol))
    roots = sorted(set(np.round(roots, digits)))
    return roots


# ─────────────────────────────── driver ─────────────────────────────────────

def main():
    coeffs = read_csv("poly_coeff_newton.csv")      # ← your CSV
    start = time.time()
    roots = real_roots(np.array(coeffs, float))
    print(f"Hybrid Newton/Bisection real roots ({len(roots)}): {roots}")
    print("Time taken:", time.time() - start, "s\n")

    # reference result from NumPy (for verification only)
    start = time.time()
    np_roots = np.roots(coeffs)
    real_part = np.real(np_roots[np.abs(np.imag(np_roots)) < 1e-8])
    print(f"NumPy roots ({len(real_part)}): {sorted(real_part.tolist())}")
    print("Time taken", time.time() - start, "s")


if __name__ == "__main__":
    main()
