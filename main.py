import numpy as np
import sys
import csv
from time import time

sys.setrecursionlimit(10000)
TOLERANCE = 1e-6


def read_csv(file_path):
    data = []
    with open(file_path, mode='r') as f:
        rows = csv.reader(f)
        for row in rows:
            data.append(float(row[0]))
    return data


def cauchy_bound(coefficients):
    coefficients = np.array(coefficients)
    a_n = coefficients[0]
    polys = np.abs(coefficients[1:] / a_n)
    return 1 + max(polys)


def fujiwara_bound(coefficients):
    coefficients = np.array(coefficients)
    a_n = coefficients[0]
    n = coefficients.size - 1
    k = np.arange(1, n + 1)

    b = np.power(np.abs(coefficients[1:] / a_n), 1.0 / k)
    return 2.0 * np.max(b)


def kojima_bound(coefficients):
    coefficients = np.array(coefficients)
    a_n = coefficients[0]
    n = len(coefficients) - 1
    i = np.arange(1, n + 1)
    polys = 2 * np.abs(coefficients[1:] / a_n)
    return max(polys)


def poly_val(p, x):
    n = len(p)
    xx = np.full(n, x)
    xx[0] = 1
    xx = np.multiply.accumulate(xx)
    return np.dot(xx[::-1], p)


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


def normalize_by_max_coefficient(coefficients):
    max_coefficient = np.max(np.abs(coefficients))
    if max_coefficient != 0:
        return coefficients / max_coefficient
    return coefficients


def derive_poly(coefficients, n):
    deg = np.arange(n, 0, -1)
    return deg * coefficients[:-1]


def bisection_method(coefficients, a, b, tolerance=TOLERANCE):
    a_sign = poly_val_sign(coefficients, a)
    b_sign = poly_val_sign(coefficients, b)
    if a_sign == 0:
        return a
    if b_sign == 0:
        return b

    while b - a > tolerance:
        midpoint = (a + b) / 2
        mid_sign = poly_val_sign(coefficients, midpoint)
        if mid_sign == 0 or abs(b - a) < tolerance:
            return midpoint
        if poly_val_sign(coefficients, a) * mid_sign < 0:
            b = midpoint
        else:
            a = midpoint
    return (a + b) / 2


def newton_raphson_method(coefficients, no_coefficients, initial_guess, a, b, tolerance=TOLERANCE, max_iterations=100):
    x0 = initial_guess
    if (poly_val_sign(coefficients, x0)) == 0:
        return x0
    for i in range(max_iterations):
        ratio = poly_fraction(coefficients, derive_poly(coefficients, no_coefficients - 1), x0)
        x1 = x0 - ratio
        if abs(x1 - x0) < tolerance:
            return x1
        if x1 < a or x1 > b:
            return None
        x0 = x1
    return None


def find_roots(coefficients, a, b):
    roots = []
    n = len(coefficients)
    if n <= 2:
        return [-coefficients[0] / coefficients[1]]
    else:
        d_roots = find_roots(normalize_by_max_coefficient(derive_poly(coefficients, n - 1)), a, b)

    intervals = np.sort(np.append(d_roots, [a, b]))
    for i in range(len(intervals) - 1):
        interval_start, interval_end = intervals[i], intervals[i + 1]
        bisection_root = bisection_method(coefficients, interval_start, interval_end)
        if bisection_root is not None:
            root = newton_raphson_method(coefficients, n, bisection_root, interval_start, interval_end)
            if root is not None:
                roots.append(root)

    return np.array(roots)


def main():
    file_path = 'poly_coeff_newton.csv'
    coefficients = read_csv(file_path)

    starting_time = time()
    bound = fujiwara_bound(coefficients)
    roots = find_roots(coefficients, -bound, bound)

    end_time = time() - starting_time
    print(f"Newton-Raphson and Bisection method roots: {roots.tolist()}")
    print("Time taken:", end_time)

    print()

    starting_time = time()
    roots = np.roots(coefficients)
    real_roots = np.real(roots[np.isreal(roots)])
    end_time = time() - starting_time
    print(f"numpy np.roots: {real_roots}")
    print("Time taken", end_time)


if __name__ == "__main__":
    main()
