"""
The Lagrangian relaxation oracle for Project 29.

The relaxed problem is

    min  c^T alpha     s.t.   y^T alpha = 0,   0 <= alpha <= C          (KP)

i.e. a linear program over a box plus ONE equality constraint: the classical
*continuous knapsack* problem.  It is solved exactly in O(m log m) by relaxing
the single equality with a scalar theta and scanning the breakpoints.

For a fixed theta the box minimiser of (c - theta*y)^T alpha is

    alpha_i(theta) = C                      if c_i - theta*y_i <  0
                     0                      if c_i - theta*y_i >  0
                     anything in [0, C]     if c_i - theta*y_i == 0   (tied)

h(theta) = y^T alpha(theta) is non-decreasing, so a theta* with h(theta*) = 0
exists; the tied components are then split to make the equality hold exactly.
Such an alpha minimises the Lagrangian of (KP) *and* is feasible, hence optimal.

The tied set is exactly where (KP) has multiple optima -- which is precisely
where the Lagrangian dual phi is non-differentiable, and (see the report) it
coincides with the support-vector condition y_i f(x_i) = 1.
"""
import numpy as np


def continuous_knapsack(c, y, C):
    """Solve (KP). Returns (alpha, optimal value, theta, n_tied)."""
    c = np.asarray(c, dtype=float)
    y = np.asarray(y, dtype=float)
    m = c.size

    # breakpoints: alpha_i switches at theta = y_i * c_i (y_i = +-1 so y_i = 1/y_i)
    brk = np.unique(y * c)
    # candidate thetas: strictly between breakpoints, outside, and at breakpoints
    mids = np.concatenate(([brk[0] - 1.0], 0.5 * (brk[:-1] + brk[1:]), [brk[-1] + 1.0]))

    def box_min(theta, tol=0.0):
        r = c - theta * y                       # reduced costs
        a = np.where(r < -tol, C, 0.0)
        return a, r

    # 1) a strictly-between-breakpoints theta that already balances the equality
    lo, hi = 0, mids.size - 1
    while lo < hi:                              # h is non-decreasing -> bisect
        mid = (lo + hi) // 2
        a, _ = box_min(mids[mid])
        if y @ a < 0.0:
            lo = mid + 1
        else:
            hi = mid
    a, _ = box_min(mids[lo])
    if abs(y @ a) <= 1e-11:
        return a, float(c @ a), float(mids[lo]), 0

    # 2) otherwise the crossing happens *at* a breakpoint: split the tied set
    j = int(np.searchsorted(brk, mids[lo]))
    for theta in brk[max(j - 1, 0): min(j + 2, brk.size)]:
        a, r = box_min(theta, tol=1e-12)
        tied = np.flatnonzero(np.abs(r) <= 1e-12)
        resid = y @ a
        for i in tied:
            if abs(resid) <= 1e-13:
                break
            step = np.clip(-resid / y[i], -a[i], C - a[i])
            a[i] += step
            resid += y[i] * step
        if abs(resid) <= 1e-9:
            return a, float(c @ a), float(theta), int(tied.size)

    raise RuntimeError("continuous_knapsack: no balancing theta found")
