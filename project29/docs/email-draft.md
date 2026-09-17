# Draft reply to Prof. Frangioni

> Draft only — read it, cut what you disagree with, send it in your own words.
> It deliberately puts the one question he has to rule on near the top, rather than
> leaving him to discover it in Section 3.6.

---

**Subject:** Re: 784AA Project 29 — revised formulation (Group 26)

Dear Prof. Frangioni,

We took the second option: (M) stays the standard C-SVM, One-vs-Rest is built on top
of it unchanged, and we looked for a different dual approach for the binary problem.

Both our earlier attempts failed for one reason we had not seen. In the SVM dual

    min ½αᵀQα − 1ᵀα   s.t.  yᵀα = 0,  0 ≤ α ≤ C

neither constraint is the difficulty: relaxing the equality leaves a box-constrained
QP with the same dense Hessian and a univariate dual, which was exactly your
objection, and relaxing the box leaves a QP too. The difficulty is Q. But Q is only
there because w was eliminated through the stationarity condition
w = Σᵢ αᵢyᵢφ(xᵢ). So we put w back as a variable, which makes that identity a
constraint, and dualise the identity instead of a constraint:

    φ(λ) = −½‖λ‖² + min_{α ∈ D} c(λ)ᵀα ,   cᵢ(λ) = −(1 + yᵢ⟨λ, φ(xᵢ)⟩),
    D = { yᵀα = 0, 0 ≤ α ≤ C }.

The relaxed problem is a linear program over a box with one equality — a continuous
knapsack — solved exactly by sorting in O(m log m). So the relaxation is genuinely
easier than the original, the dual is n- or m-dimensional rather than univariate, and
it is non-differentiable exactly where the knapsack ties, which unwinds to
yᵢf(xᵢ) = 1. D is a non-empty compact polytope, so φ is finite everywhere.

**The point we would like you to rule on before we go further.** max ψ turns out to
be the hinge-loss primal ‖w‖² + C Σ max{1 − yᵢ(⟨w,xᵢ⟩ − b), 0} — which is the SVM
exactly as it appears in your nondifferentiable-optimization slides, as the
motivation for bundle methods. So the obvious objection is that we have dualised our
way back to the problem the course already starts from, and are running a bundle
method on it.

We think the dualisation earns its place, and we measured rather than argued it. It
is not that the bias makes a direct approach impossible — f(w,b) is convex in both
and a subgradient in (w,b) is immediate, so b can simply go in the bundle. It is
that our relaxation minimises over b *exactly* at every oracle call instead of
approximating that direction with cutting planes: the knapsack multiplier θ is that
partial minimisation, and the sort is how it is computed. We are bundling
g(w) = min_b f(w,b) rather than f(w,b). Keeping everything else identical — quadratic
term exact, same cutting-plane model, same proximal term, same master solver — so
that the only difference is whether b is a bundle variable:

    dataset   kernel     b in bundle      b eliminated (ours)
    wine      linear     137 iters        29
    wine      Gaussian   400 (gap 6e-5)   133 (gap 1.1e-8)
    iris      Gaussian    96              24
    cancer    linear     220              52

A consistent factor of four to five, and one instance where the direct method does
not reach tolerance at all.

We claim exactly that much and no more: the family of cuts and the function being
optimised are the same as in the direct approach; what relaxing the representer
identity adds is the exact elimination of b at every oracle call, the SVM dual
variables via the aggregated solution from the master's dual multipliers, and a
two-sided bound on v(D) rather than a bound on our own objective only. If you
consider that too thin to count as a different dual approach, we would rather know
now than after the experiments.

On the budget model: we wrote out its dual explicitly, as you asked. Doing so turned
up something we had missed — with a hard budget the primal can simply be infeasible.
A sample lying where two classifiers both score near zero needs about one unit of
slack from each, so B < 2 is typically already infeasible, and the Lagrangian dual
then diverges. We measured the threshold at exactly B_min = 2.0 on a three-class
instance. It is repairable, by softening the budget with a penalised surplus, which
also restores a box on α, and we verified strong duality for the repaired model. We
have kept all of it in an appendix rather than as the main line, because it changes
(M) and would confound a modelling change with an algorithmic one.

Everything above is implemented and checked: strong duality to ~1e-9 for linear,
Gaussian and polynomial kernels; the knapsack matches an LP solver to 1e-12; the
recovered α and b match a general-purpose QP solver to ~1e-7; and One-vs-Rest built
on (A1) and on (A2) give identical test accuracy. We also measured what the project
asks us to discuss across the One-vs-Rest problems: warm-starting the multiplier is
neutral to harmful and transferring cuts is within noise, while sharing the kernel
matrix is worth a factor q. We will report that negative result as such.

One small thing: your list mentioned the primal and dual formulations of SVR. We
have read that as SVM, since (M) is classification; the same relaxation works for
SVR by swapping the hinge for the ε-insensitive loss, and we can add it if you
would prefer.

Sections 1–3 and the appendix are attached.

Thank you,
Tolgonai Nasipbek kyzy (626449)
Hamse Hassan Adnan (683187)
