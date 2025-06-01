import csv
import sys
import time

import numpy as np

sys.setrecursionlimit(10000)
TOL = 1e-6
MAX_NR = 100


def read_csv(path):
    data = []
    with open(path, mode='r') as f:
        rows = csv.reader(f)
        for row in rows:
            data.append(float(row[0]))
    return np.array(data)


def poly_val(p, x):
    x_degrees_by_p = np.ones_like(p) * x
    x_degrees_by_p[0] = 1
    x_degrees_by_p = np.multiply.accumulate(x_degrees_by_p)
    return np.dot(p, x_degrees_by_p[::-1])


def poly_val_sign(p, x):
    if abs(x) <= 1:
        return np.sign(poly_val(p, x))
    n = len(p) - 1
    y = 1 / x
    p = p[::-1]
    if n % 2 == 0:
        return np.sign(poly_val(p, y))
    return np.sign(poly_val(p, y)) * np.sign(y)


def poly_fraction(p, q, x):
    n = len(p) - len(q)
    if np.abs(x) > 1:
        p_coefficients = p[::-1]
        q_coefficients = q[::-1]
        y = 1 / x
        lead_coeff = np.power(x, n)
        return lead_coeff * poly_val(p_coefficients, y) / poly_val(q_coefficients, y)
    return poly_val(p, x) / poly_val(q, x)


def derivative(p):
    n = p.size - 1
    deg = np.arange(n, 0, -1)
    return deg * p[:-1]


def normalize_by_max_coefficient(p):
    max_coeff = np.max(np.abs(p))
    if max_coeff != 0:
        return np.array(p / max_coeff)
    return np.array(p)


def fujiwara_bound(p):
    a_n = p[0]
    n = p.size - 1
    k = np.arange(1, n + 1)

    b = np.power(np.abs(p[1:] / a_n), 1.0 / k)
    return 2.0 * np.max(b)


def newton_raphson(p, dp, x0, a, b, tol=TOL, max_it=MAX_NR):
    x = x0
    if poly_val_sign(p, x) == 0.0:
        return x
    for _ in range(max_it):
        if poly_val_sign(p, x) == 0.0:
            break
        ratio = poly_fraction(p, dp, x)
        x_new = x - ratio
        if not (a <= x_new <= b):
            break
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return None


def bisection(p, a, b, tol=TOL):
    a_sign = poly_val_sign(p, a)
    b_sign = poly_val_sign(p, b)
    if a_sign == 0:
        return a
    if b_sign == 0:
        return b

    while b - a > tol:
        midpoint = (a + b) / 2
        mid_sign = poly_val_sign(p, midpoint)
        if mid_sign == 0 or abs(b - a) < tol:
            return midpoint
        if poly_val_sign(p, a) * mid_sign < 0:
            b = midpoint
        else:
            a = midpoint
    return (a + b) / 2


def hybrid_interval_root(p, a, b):
    dp = derivative(p)
    mid = (a + b) / 2

    newton_raphson_root = newton_raphson(p, dp, mid, a, b)
    root = newton_raphson_root
    if newton_raphson_root is None:
        bisection_root = bisection(p, a, b)
        if bisection_root is not None:
            root = newton_raphson(p, dp, bisection_root, a, b)
        else:
            root = bisection_root
    return root


def real_roots(p, a, b, tol=TOL):
    n = p.size

    if n <= 2:
        return [-p[0] / p[1]]
    else:
        crit = real_roots(normalize_by_max_coefficient(derivative(p)), a, b)

    endpoints = np.sort(np.append(crit, [a, b]))

    roots = []
    for i in range(len(endpoints) - 1):
        lo, hi = endpoints[i], endpoints[i + 1]
        flo, fhi = poly_val_sign(p, lo), poly_val_sign(p, hi)
        if flo != fhi:
            root = hybrid_interval_root(p, lo, hi)
            roots.append(root)

    digits = int(-np.log10(tol))
    roots = np.round(roots, digits)
    unique_roots = np.unique(roots)
    return unique_roots.tolist()


if __name__ == "__main__":
    coeffs = read_csv("poly_coeff_newton.csv")
    start = time.time()
    B = fujiwara_bound(coeffs)
    roots = real_roots(coeffs, -B, B)
    print(f"Hybrid Newton/Bisection real roots ({len(roots)}): {roots}")
    print("Time taken:", time.time() - start, "s\n")

    start = time.time()
    np_roots = np.roots(coeffs)
    real_part = np.real(np_roots[np.abs(np.imag(np_roots)) < TOL])
    print(f"NumPy roots ({len(real_part)}): {sorted(real_part.tolist())}")
    print("Time taken", time.time() - start, "s")
