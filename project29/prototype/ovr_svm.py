"""(M) for Project 29: multi-class SVM by One-vs-Rest, trained with (A1) or (A2)."""
import numpy as np
import cvxpy as cp

from bundle_svm import kernel_matrix, bundle_solve, transfer_cuts


def solve_dual_qp(K, y, C, solver="CLARABEL"):
    """(A2): the standard SVM dual handed to a general-purpose QP solver."""
    m = K.shape[0]
    Q = np.outer(y, y) * K
    a = cp.Variable(m, nonneg=True)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(a, cp.psd_wrap(Q)) - cp.sum(a)),
                      [a <= C, y @ a == 0])
    prob.solve(solver=getattr(cp, solver))
    alpha = np.clip(np.asarray(a.value).ravel(), 0.0, C)
    free = np.flatnonzero((alpha > 1e-6 * C) & (alpha < C - 1e-6 * C))
    f = K @ (y * alpha)
    b = float(np.mean(y[free] - f[free])) if free.size else 0.0
    return dict(alpha=alpha, b=b, value=float(prob.value), status=prob.status)


class OvRSVM:
    """One-vs-Rest multi-class SVM.

    method='bundle' -> (A1), the Lagrangian dual of the representer identity
                       solved by a proximal bundle method
    method='qp'     -> (A2), the SVM dual handed to a general-purpose QP solver

    transfer controls what is carried from one One-vs-Rest problem to the next:
      'none'  nothing            'warm'  the multiplier lambda only
      'cuts'  the bundle only    'both'  multiplier + bundle
    The kernel matrix is always computed once and shared by all K problems.
    """

    def __init__(self, C=1.0, kernel="linear", gamma=None, degree=3, coef0=1.0,
                 method="bundle", tol=1e-8, max_iter=500, t0=1.0,
                 transfer="both", n_transfer=10, verbose=False):
        self.__dict__.update(C=C, kernel=kernel, gamma=gamma, degree=degree,
                             coef0=coef0, method=method, tol=tol,
                             max_iter=max_iter, t0=t0, transfer=transfer,
                             n_transfer=n_transfer, verbose=verbose)

    def fit(self, X, labels):
        self.X_, self.classes_ = X, np.unique(labels)
        K = kernel_matrix(X, kind=self.kernel, gamma=self.gamma,
                          degree=self.degree, coef0=self.coef0)      # shared by all K
        self.K_ = K
        self.alphas_, self.bs_, self.ys_, self.stats_ = [], [], [], []
        pool, pool_y, beta_prev = [], [], None

        for cls in self.classes_:
            y = np.where(labels == cls, 1.0, -1.0)
            if self.method == "qp":
                r = solve_dual_qp(K, y, self.C)
                self.stats_.append(dict(cls=cls, iters=None, oracle_calls=None,
                                        gap=0.0, seeded=0))
            else:
                seeds = (transfer_cuts(pool, pool_y, y, self.C)
                         if (self.transfer in ("cuts", "both") and pool) else None)
                b0 = beta_prev if self.transfer in ("warm", "both") else None
                r = bundle_solve(K, y, self.C, tol=self.tol, max_iter=self.max_iter,
                                 t0=self.t0, beta0=b0, cuts0=seeds,
                                 verbose=self.verbose)
                pool = (pool + [r["alpha_oracle"]])[-self.n_transfer:]
                pool_y = (pool_y + [y])[-self.n_transfer:]
                beta_prev = r["beta"]
                self.stats_.append(dict(cls=cls, iters=r["iters"],
                                        oracle_calls=r["oracle_calls"],
                                        gap=r["gap"], seeded=len(seeds or [])))
            self.alphas_.append(r["alpha"])
            self.bs_.append(r["b"])
            self.ys_.append(y)
        return self

    def decision_function(self, Xt):
        Kt = kernel_matrix(self.X_, Xt, kind=self.kernel, gamma=self.gamma,
                           degree=self.degree, coef0=self.coef0)      # m x t
        return np.column_stack([(a * yk) @ Kt + b
                                for a, yk, b in zip(self.alphas_, self.ys_, self.bs_)])

    def predict(self, Xt):
        return self.classes_[np.argmax(self.decision_function(Xt), axis=1)]

    def score(self, Xt, yt):
        return float(np.mean(self.predict(Xt) == yt))
