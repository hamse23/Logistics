"""Correctness checks for the Project 29 formulation and solver."""
import sys, time
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from sklearn.datasets import make_blobs, load_iris, load_wine
from sklearn.model_selection import train_test_split

from knapsack_oracle import continuous_knapsack
from bundle_svm import kernel_matrix, bundle_solve, DualOracle
from ovr_svm import OvRSVM, solve_dual_qp

rng = np.random.default_rng(0)
OK = lambda c: "PASS" if c else "**FAIL**"


def binary_data(m=80, d=4, std=2.2, seed=3):
    X, lab = make_blobs(n_samples=m, centers=2, n_features=d, cluster_std=std,
                        random_state=seed)
    X = (X - X.mean(0)) / X.std(0)
    return X, np.where(lab == 0, -1.0, 1.0)


print("=" * 78)
print("1. the oracle: O(m log m) continuous knapsack  vs  a general LP solver")
print("=" * 78)
worst = 0.0
for trial in range(40):
    m = int(rng.integers(10, 200))
    y = np.where(rng.random(m) < 0.5, -1.0, 1.0)
    if abs(y.sum()) == m:
        y[0] = -y[0]
    c = rng.normal(size=m) * rng.uniform(0.1, 10)
    C = float(rng.uniform(0.1, 5))
    a1, v1, th, tied = continuous_knapsack(c, y, C)
    r = linprog(c, A_eq=y.reshape(1, -1), b_eq=[0.0], bounds=[(0, C)] * m, method="highs")
    worst = max(worst, abs(v1 - r.fun), abs(y @ a1), max(0, a1.max() - C), max(0, -a1.min()))
print(f"   worst |value difference| / |y^T alpha| / bound violation = {worst:.3e}   {OK(worst < 1e-8)}")

print()
print("=" * 78)
print("2. strong duality:   max_lambda phi(lambda) == v(D)  (no duality gap)")
print("=" * 78)
for kern, kw in [("linear", {}), ("rbf", dict(gamma=0.3)), ("poly", dict(degree=3))]:
    X, y = binary_data()
    K = kernel_matrix(X, kind=kern, **kw)
    for C in (0.1, 1.0, 10.0):
        ref = solve_dual_qp(K, y, C)
        r = bundle_solve(K, y, C, tol=1e-10, max_iter=400)
        rel = abs(r["lb"] - ref["value"]) / max(1.0, abs(ref["value"]))
        print(f"   {kern:>6} C={C:<5} v(D)={ref['value']: .8f}  max phi={r['lb']: .8f}"
              f"  rel.err={rel:.2e}  iters={r['iters']:3d}  {OK(rel < 1e-6)}")

print()
print("=" * 78)
print("3. weak duality:  phi(lambda) <= v(D)  for arbitrary lambda")
print("=" * 78)
X, y = binary_data()
K = kernel_matrix(X, kind="rbf", gamma=0.3)
C = 1.0
vD = solve_dual_qp(K, y, C)["value"]
orc = DualOracle(K, y, C)
viol = 0
for _ in range(3000):
    beta = rng.normal(size=K.shape[0]) * rng.uniform(0, 3)
    if orc(beta)[0] > vD + 1e-7:
        viol += 1
print(f"   violations over 3000 random multipliers: {viol}   {OK(viol == 0)}")

print()
print("=" * 78)
print("4. primal recovery: the master's convex multipliers give alpha* and b")
print("=" * 78)
for kern, kw in [("linear", {}), ("rbf", dict(gamma=0.3))]:
    X, y = binary_data()
    K = kernel_matrix(X, kind=kern, **kw)
    ref = solve_dual_qp(K, y, 1.0)
    r = bundle_solve(K, y, 1.0, tol=1e-11, max_iter=600)
    da = np.linalg.norm(r["alpha"] - ref["alpha"]) / max(1.0, np.linalg.norm(ref["alpha"]))
    db = abs(r["b"] - ref["b"])
    print(f"   {kern:>6}  ||alpha_bundle - alpha_qp||/||alpha_qp|| = {da:.2e}   {OK(da < 1e-4)}")
    print(f"   {kern:>6}  |b_bundle - b_qp| = {db:.2e}   (b = -theta, the knapsack "
          f"multiplier)   {OK(db < 1e-4)}")

print()
print("=" * 78)
print("5. the identity  max_{alpha in D} sum_i a_i(1 - y_i g_i) = min_b C*sum_i hinge")
print("   (i.e. the multiplier of y^T alpha = 0 IS the SVM bias b)")
print("=" * 78)
X, y = binary_data(m=60, d=3)
C, m = 1.3, 60
bad = 0
for _ in range(200):
    g = rng.normal(size=m) * rng.uniform(0.2, 3)
    a, v, th, _ = continuous_knapsack(-(1.0 - y * g), y, C)
    lhs = -v
    bb = cp.Variable()
    rhs = cp.Problem(cp.Minimize(C * cp.sum(cp.pos(1 - cp.multiply(y, g + bb))))).solve(solver=cp.CLARABEL)
    if abs(lhs - rhs) > 1e-6 * max(1, abs(rhs)):
        bad += 1
print(f"   mismatches over 200 random score vectors: {bad}   {OK(bad == 0)}")

print()
print("=" * 78)
print("6. (M) end to end: One-vs-Rest, (A1) bundle vs (A2) general-purpose QP")
print("=" * 78)
for name, load in [("iris", load_iris), ("wine", load_wine)]:
    d = load()
    Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.3,
                                          random_state=0, stratify=d.target)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    for kern, kw in [("linear", {}), ("rbf", dict(gamma=0.2))]:
        row = []
        for meth in ("bundle", "qp"):
            t0 = time.perf_counter()
            mdl = OvRSVM(C=1.0, kernel=kern, method=meth, tol=1e-9, **kw).fit(Xtr, ytr)
            row.append((meth, mdl.score(Xte, yte), time.perf_counter() - t0, mdl.stats_))
        (_, acc_b, t_b, st), (_, acc_q, t_q, _) = row
        it = [s["iters"] for s in st]
        sd_ = [s["seeded"] for s in st]
        print(f"   {name:>5} {kern:>6}: acc bundle={acc_b:.4f}  acc qp={acc_q:.4f}  "
              f"{OK(abs(acc_b - acc_q) < 1e-9)}   bundle iters/class={it} seeded={sd_}"
              f"  [{t_b:.2f}s vs {t_q:.2f}s]")

print()
print("=" * 78)
print("7. does re-using the bundle across One-vs-Rest classes help?")
print("=" * 78)
for name, load in [("iris", load_iris), ("wine", load_wine)]:
    d = load()
    X = (d.data - d.data.mean(0)) / (d.data.std(0) + 1e-12)
    for kern, kw in [("linear", {}), ("rbf", dict(gamma=0.2))]:
        tot = {}
        for tr in ("none", "warm", "cuts", "both"):
            mdl = OvRSVM(C=1.0, kernel=kern, method="bundle", tol=1e-9,
                         transfer=tr, **kw).fit(X, d.target)
            tot[tr] = sum(s["oracle_calls"] for s in mdl.stats_)
        base = tot["none"]
        cell = "  ".join(f"{k}={tot[k]:4d}({100.0*(tot[k]-base)/base:+5.1f}%)"
                         for k in ("none", "warm", "cuts", "both"))
        print(f"   {name:>5} {kern:>6}: oracle calls  {cell}")
