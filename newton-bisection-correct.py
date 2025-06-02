import csv
import sys
import time

import numpy as np

sys.setrecursionlimit(10000)
TOL = 1e-6


def read_csv(path):
    data = []
    with open(path, mode='r') as f:
        rows = csv.reader(f)
        for row in rows:
            data.append(float(row[0]))
    return np.array(data)


def poly_val(p, x):
    # This is the best performing function to evaluate a polynomial, as tuaght by Dan in the lectures.
    x_degrees_by_p = np.ones_like(p) * x
    x_degrees_by_p[0] = 1
    x_degrees_by_p = np.multiply.accumulate(x_degrees_by_p)
    return np.dot(p, x_degrees_by_p[::-1])


def poly_val_sign(p, x):
    # This function evaluates the polynomial p(x) and returns the sign of the value.
    # If x is outside the interval [-1, 1], we use the transformation y = 1/x to avoid overflow, as taught by Dan in the lectures.
    if abs(x) <= 1:
        return np.sign(poly_val(p, x))
    n = len(p) - 1
    y = 1 / x
    p = p[::-1]
    if n % 2 == 0:
        return np.sign(poly_val(p, y))
    return np.sign(poly_val(p, y)) * np.sign(y)


def poly_fraction(p, q, x):
    # This function evaluates the polynomial fraction p(x) / q(x).
    # If x is outside the interval [-1, 1], we use the transformation y = 1/x to avoid overflow, as taught by Dan in the lectures.
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
    # We normalize the polynomial by its maximum coefficient to avoid overflow issues, as taught by Dan in the lectures.
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


def newton_raphson(p, dp, x0, a, b, tol=TOL, max_it=50):
    x = x0
    # max_it = 50 is the default number of iterations for convergence by scipy.optimize.newton...
    for _ in range(max_it):
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
        return a, True
    if b_sign == 0:
        return b, True

    while b - a > tol:
        midpoint = (a + b) / 2
        mid_sign = poly_val_sign(p, midpoint)
        if mid_sign == 0 or abs(b - a) < tol:
            return midpoint, False
        if poly_val_sign(p, a) * mid_sign < 0:
            b = midpoint
        else:
            a = midpoint
    return (a + b) / 2, False


def newton_raphson_and_bisection_method(p, a, b):
    dp = derivative(p)
    initial_guess = (a + b) / 2

    #  Do newton-raphson first, as it is O(loglogn) and faster for finding roots
    #  But, it is important to note that if we solve this by using bisection first, we can ensure that the initial guess is within the bounds.
    #  And that it is close to the root, which is important for convergence of newton-raphson. So overall, that is a better approach.
    #  However, this is how Dan wanted.
    newton_raphson_root = newton_raphson(p, dp, initial_guess, a, b)
    root = newton_raphson_root

    # If newton-raphson didn't find a root, we will use bisection to find a root in the interval [a, b].
    if root is None:
        point, point_is_root = bisection(p, a, b)
        # If bisection found a root, we return it.
        if point_is_root:
            return point
        # Otherwise, we will use newton-raphson to refine the point found by bisection, as a fallback.
        root = newton_raphson(p, dp, point, a, b)
    return root


def find_roots(p, a, b):
    n = p.size

    if n <= 2:
        return [-p[1] / p[0]]
    else:
        critical_points = find_roots(normalize_by_max_coefficient(derivative(p)), a, b)

    endpoints = np.sort(np.append(critical_points, [a, b]))

    roots = []
    # newton-raphson and bisection will only find a single root in an interval (which is why we loop over all possible intervals, we don't know which contain roots and which don't)
    for i in range(len(endpoints) - 1):
        low, high = endpoints[i], endpoints[i + 1]
        p_low, p_high = poly_val_sign(p, low), poly_val_sign(p, high)
        if p_low * p_high < 0:  # Because the root is where p(low) * p(high) < 0 -- so different signs.
            root = newton_raphson_and_bisection_method(p, low, high)
            if root is not None:
                roots.append(root)

    return np.array(roots)


if __name__ == "__main__":
    coeffs = read_csv("poly_coeff_newton.csv")
    start = time.time()
    B = fujiwara_bound(coeffs)
    roots = find_roots(coeffs, -B, B)
    print(f"Newton-Raphson & Bisection real roots ({len(roots)}): {roots}")
    print("Time taken:", time.time() - start, "seconds\n")

    start = time.time()
    np_roots = np.roots(coeffs)
    real_part = np.real(np_roots[np.abs(np.imag(np_roots)) < TOL])
    print(f"NumPy roots ({len(real_part)}): {sorted(real_part.tolist())}")
    print("Time taken", time.time() - start, "seconds")
