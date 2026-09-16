"""
Appendix A of the report: the shared-slack-budget One-vs-Rest model and its dual.

    (M-B)  min  sum_k 1/2||w^k||^2 + C sum_{i,k} xi_i^k
           s.t. y_i^k(<w^k,phi_i>+b^k) >= 1 - xi_i^k,  xi >= 0,
                sum_k xi_i^k <= B_i            (hard budget)   or
                sum_k xi_i^k <= B_i + z_i, z >= 0, penalised by D  (soft budget)

Dualising ONLY the mq margin constraints gives, in both cases,

    phi(a) = sum_{i,k} a_i^k - 1/2 sum_k a^k' Q^k a^k
                             - sum_i B_i [ max_k (a_i^k - C) ]^+

on { a >= 0, sum_i a_i^k y_i^k = 0 for all k }, with the extra box a <= C + D in the
soft case.  This script shows:

  (1) with a HARD budget the primal is infeasible below a threshold B_min, and the
      Lagrangian dual is then unbounded -- the defect we report;
  (2) with the SOFT budget the model is always feasible and strong duality holds.
"""
import numpy as np
import cvxpy as cp
from sklearn.datasets import make_blobs


def build(m, q, seed, std, d=2):
    X, lab = make_blobs(n_samples=m, centers=q, n_features=d, cluster_std=std,
                        random_state=seed)
    X = (X - X.mean(0)) / X.std(0)
    return X, np.where(lab[:, None] == np.arange(q)[None, :], 1.0, -1.0)


def _margins(X, Y, W, b, Xi):
    q = Y.shape[1]
    return cp.multiply(Y, X @ W + cp.reshape(b, (1, q), order='C')) >= 1 - Xi


def primal(X, Y, C, B, D=None):
    m, d = X.shape; q = Y.shape[1]
    W = cp.Variable((d, q)); b = cp.Variable(q); Xi = cp.Variable((m, q), nonneg=True)
    obj = 0.5 * cp.sum_squares(W) + C * cp.sum(Xi)
    if D is None:                                   # hard budget
        cons = [_margins(X, Y, W, b, Xi), cp.sum(Xi, axis=1) <= B]
    else:                                           # soft budget
        z = cp.Variable(m, nonneg=True)
        cons = [_margins(X, Y, W, b, Xi), cp.sum(Xi, axis=1) <= B + z]
        obj = obj + D * cp.sum(z)
    p = cp.Problem(cp.Minimize(obj), cons)
    p.solve(solver=cp.CLARABEL)
    return p.status, p.value


def dual(X, Y, C, B, D=None):
    m, d = X.shape; q = Y.shape[1]
    A = cp.Variable((m, q), nonneg=True)
    quad = sum(cp.sum_squares(X.T @ cp.multiply(Y[:, k], A[:, k])) for k in range(q))
    pen = cp.sum(cp.multiply(B, cp.pos(cp.max(A - C, axis=1))))
    cons = [cp.sum(cp.multiply(Y[:, k], A[:, k])) == 0 for k in range(q)]
    if D is not None:
        cons.append(A <= C + D)                     # the box the soft budget restores
    p = cp.Problem(cp.Maximize(cp.sum(A) - 0.5 * quad - pen), cons)
    p.solve(solver=cp.CLARABEL)
    return p.status, p.value, A.value


def feasibility_threshold(X, Y):
    """Smallest uniform budget for which (M-B) with a hard budget is feasible."""
    m, d = X.shape; q = Y.shape[1]
    W = cp.Variable((d, q)); b = cp.Variable(q)
    Xi = cp.Variable((m, q), nonneg=True); t = cp.Variable()
    cp.Problem(cp.Minimize(t),
               [_margins(X, Y, W, b, Xi), cp.sum(Xi, axis=1) <= t]).solve(solver=cp.CLARABEL)
    return float(t.value)


if __name__ == "__main__":
    fmt = lambda v: f"{v: .6f}" if v is not None and np.isfinite(v) else "      inf"

    X, Y = build(40, 3, seed=5, std=2.2)
    Bmin = feasibility_threshold(X, Y)
    print("=" * 76)
    print("(1) HARD budget: the primal is infeasible below a threshold")
    print("=" * 76)
    print(f"    measured feasibility threshold  B_min = {Bmin:.6f}\n")
    print(f"    {'B':>7} {'primal':>13} {'v*':>11} {'dual':>12} {'d*':>11}")
    for Bv in [0.3, 1.0, 1.5, Bmin * 1.001, 3.0]:
        sp, vp = primal(X, Y, 1.0, np.full(40, Bv))
        sd, vd, _ = dual(X, Y, 1.0, np.full(40, Bv))
        print(f"    {Bv:7.3f} {sp:>13} {fmt(vp):>11} {sd:>12} {fmt(vd):>11}")
    print("\n    -> below B_min the model has no feasible point and the dual diverges.")
    print("       A sample where two classifiers both score ~0 needs ~1 unit of slack")
    print("       from each, so B < 2 is typically already infeasible.")

    print()
    print("=" * 76)
    print("(2) SOFT budget: always feasible, and strong duality holds")
    print("=" * 76)
    print(f"    {'m':>4}{'q':>3}{'B':>7}{'D':>6} {'primal':>10} {'v*':>12} "
          f"{'dual':>9} {'d*':>12} {'gap':>10}  non-smoothness")
    for (m_, q, seed, std, Bv, Dv) in [(40, 3, 5, 2.2, 0.3, 1.0), (40, 3, 5, 2.2, 0.7, 1.0),
                                       (40, 3, 5, 2.2, 2.0, 1.0), (40, 3, 5, 2.2, 0.7, 5.0),
                                       (60, 4, 9, 2.8, 0.5, 2.0), (30, 3, 12, 1.5, 1.0, 1.0)]:
        X, Y = build(m_, q, seed, std)
        B = np.full(m_, Bv)
        sp, vp = primal(X, Y, 1.0, B, Dv)
        sd, vd, A = dual(X, Y, 1.0, B, Dv)
        mx = A.max(1)
        kink = int(np.sum(np.abs(mx - 1.0) < 1e-4))
        tie = sum(1 for i in range(m_) if mx[i] > 1.0 + 1e-4
                  and np.sum(A[i] > mx[i] - 1e-4) >= 2)
        print(f"    {m_:>4}{q:>3}{Bv:>7.2f}{Dv:>6.1f} {sp:>10} {vp:>12.6f} {sd:>9} "
              f"{vd:>12.6f} {vp - vd:>10.1e}  at-kink={kink:2d}, contested={tie:2d}")
