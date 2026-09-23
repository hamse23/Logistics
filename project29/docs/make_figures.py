"""Figures for the Project 29 report.  Run from this directory:  python3 make_figures.py"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cutting_plane_figure(out="fig-cuttingplane"):
    """Supporting hyperplanes above a concave psi, and their pointwise minimum.

    The horizontal axis is mu, the free variable of
        psi(mu) <= psi(lambda) + <g(lambda), mu - lambda>   for all mu,
    and lambda^1, lambda^2 are the points at which the oracle was called.
    """
    psi = lambda x: -0.42 * (x - 3.2) ** 2 + 4.2
    dpsi = lambda x: -0.84 * (x - 3.2)
    xs = np.linspace(0.3, 6.1, 800)
    probes = [1.35, 5.05]                 # both on the flanks: the peak is unsampled

    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    lines = np.array([psi(p) + dpsi(p) * (xs - p) for p in probes])
    model = lines.min(axis=0)

    ax.fill_between(xs, psi(xs), model, color="0.90", zorder=0)
    for p in probes:
        ax.plot(xs, psi(p) + dpsi(p) * (xs - p), lw=1.0, color="0.6", zorder=2)
    ax.plot(xs, model, lw=1.9, color="0.2", ls=(0, (5, 2.5)), zorder=3)
    ax.plot(xs, psi(xs), lw=2.2, color="black", zorder=4)

    for k, p in enumerate(probes, 1):
        ax.plot([p], [psi(p)], "o", ms=6, color="black", zorder=5)
        ax.annotate(rf"$\lambda^{k}$", (p, psi(p)), textcoords="offset points",
                    xytext=(-4, -18), ha="center", fontsize=11)

    xstar = xs[np.argmax(psi(xs))]
    istar = int(np.argmax(model)); mstar = xs[istar]
    ax.plot([mstar, mstar], [psi(mstar), model[istar]], color="0.35", lw=1.0,
            ls=":", zorder=5)
    ax.plot([mstar], [model[istar]], "o", ms=6, mfc="white", mec="black",
            mew=1.4, zorder=6)
    ax.plot([xstar], [psi(xstar)], "*", ms=13, color="black", zorder=6)

    ax.annotate(r"$\max_\mu\,\mathrm{model}(\mu)$", (mstar, model[istar]),
                textcoords="offset points", xytext=(14, 2), ha="left", fontsize=10.5)
    ax.annotate(r"$\psi^\star$", (xstar, psi(xstar)), textcoords="offset points",
                xytext=(-17, -13), ha="center", fontsize=12)
    ax.annotate("gap", (mstar, 0.5 * (psi(mstar) + model[istar])),
                textcoords="offset points", xytext=(6, -3), fontsize=9.5, color="0.35")
    ax.annotate(r"$\psi(\mu)$", (5.9, psi(5.9)), textcoords="offset points",
                xytext=(7, -5), fontsize=12)
    ix = np.searchsorted(xs, 1.95)
    ax.annotate("cutting-plane model", (xs[ix], model[ix]),
                textcoords="offset points", xytext=(-34, 16), fontsize=9.5, color="0.2",
                arrowprops=dict(arrowstyle="-", color="0.45", lw=0.8, shrinkA=2, shrinkB=2))
    jx = np.searchsorted(xs, 0.85)
    ax.annotate("supporting hyperplane\nfrom one oracle call", (xs[jx], lines[0][jx]),
                textcoords="offset points", xytext=(18, -40), fontsize=9, color="0.4",
                ha="left",
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.8, shrinkA=2, shrinkB=2))

    ax.set_xlim(0.3, 6.6); ax.set_ylim(psi(xs).min() - 0.55, model.max() + 0.75)
    ax.set_xlabel(r"$\mu$", fontsize=13)        # the free variable, not lambda
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("0.6")
    plt.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        plt.savefig(f"{out}.{ext}", bbox_inches="tight", **({"dpi": 150} if ext == "png" else {}))
    plt.close(fig)


if __name__ == "__main__":
    cutting_plane_figure()
    print("figures written")
