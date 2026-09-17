"""
Does dualising buy anything over the deck-5 baseline?

BASELINE (what Frangioni's nonsmooth slides motivate):  bundle f(beta,b) jointly,
    f = 1/2 beta'K beta + C sum_i [1 - y_i((K beta)_i + b)]^+
    subgradient in BOTH beta and b.   b is a bundle variable.

OURS: bundle phi(beta) = -1/2||beta||_K^2 + min_{alpha in D} c'alpha.
    The knapsack minimises over b EXACTLY at every oracle call, so the model
    lives in beta alone.  b is NOT a bundle variable.

Same machinery either way: quadratic kept exact, cutting-plane model of the rest,
same proximal term, same master solver.  The only difference is how b is handled.
"""
import numpy as np, cvxpy as cp, sys
sys.path.insert(0, '/home/user/Logistics/project29/prototype')
from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from bundle_svm import kernel_matrix, DualOracle
from ovr_svm import solve_dual_qp

def baseline_bundle(K, y, C, tol=1e-8, max_iter=400, t=1.0):
    """Proximal bundle on f(beta,b); b IS a bundle variable."""
    m = K.shape[0]
    beta_bar, b_bar = np.zeros(m), 0.0
    cuts = []                                  # (val, g_beta, g_b, beta0, b0)
    def f_and_g(beta, b):
        Kb = K @ beta
        marg = y * (Kb + b)
        act = marg < 1.0
        val = 0.5 * beta @ Kb + C * np.sum(np.maximum(0.0, 1 - marg))
        g_beta = Kb - C * (K @ (y * act))
        g_b = -C * np.sum(y * act)
        return val, g_beta, g_b
    calls = 0
    fbar, gb, gbb = f_and_g(beta_bar, b_bar); calls += 1
    cuts.append((fbar, gb, gbb, beta_bar.copy(), b_bar))
    best = fbar
    for it in range(1, max_iter + 1):
        bt = cp.Variable(m); bb = cp.Variable(); v = cp.Variable()
        cons = [v >= cv + gcb @ (bt - b0) + gcbb * (bb - b0b)
                for (cv, gcb, gcbb, b0, b0b) in cuts]
        obj = v + (1.0/(2*t)) * (cp.quad_form(bt - beta_bar, cp.psd_wrap(K))
                                 + cp.square(bb - b_bar))
        p = cp.Problem(cp.Minimize(obj), cons)
        try: p.solve(solver=cp.CLARABEL)
        except Exception: p.solve(solver=cp.SCS)
        bn, bbn = np.asarray(bt.value).ravel(), float(bb.value)
        fn, gn, gnb = f_and_g(bn, bbn); calls += 1
        model = float(v.value)
        gap = best - model
        if fn < best: best = fn
        cuts.append((fn, gn, gnb, bn.copy(), bbn))
        if gap <= tol * max(1.0, abs(best)): break
        if fn <= fbar - 0.1 * (fbar - model):
            beta_bar, b_bar, fbar = bn, bbn, fn
    return best, it, calls

def ours(K, y, C, tol=1e-8, max_iter=400):
    from bundle_svm import bundle_solve
    r = bundle_solve(K, y, C, tol=tol, max_iter=max_iter)
    return -r["lb"], r["iters"], r["oracle_calls"]

print(f"{'dataset':<10}{'kern':<8}{'method':<12}{'iters':>7}{'oracle':>8}{'value':>16}  rel.err vs QP")
for nm, load, kern, kw in [("wine", load_wine, "linear", {}),
                           ("wine", load_wine, "rbf", dict(gamma=0.2)),
                           ("iris", load_iris, "rbf", dict(gamma=0.2)),
                           ("cancer", load_breast_cancer, "linear", {})]:
    d = load(); X = (d.data - d.data.mean(0))/(d.data.std(0)+1e-12)
    if len(X) > 300: X, tgt = X[:300], d.target[:300]
    else: tgt = d.target
    y = np.where(tgt == 0, 1.0, -1.0); K = kernel_matrix(X, kind=kern, **kw)
    ref = -solve_dual_qp(K, y, 1.0)["value"]          # = optimal primal value
    for label, fn in [("baseline(w,b)", baseline_bundle), ("ours(knapsack)", ours)]:
        try:
            val, its, calls = fn(K, y, 1.0)
            print(f"{nm:<10}{kern:<8}{label:<12}{its:>7}{calls:>8}{val:>16.8f}"
                  f"  {abs(val-ref)/max(1,abs(ref)):.2e}")
        except Exception as e:
            print(f"{nm:<10}{kern:<8}{label:<12}  FAILED: {type(e).__name__}: {str(e)[:50]}")
    print()
