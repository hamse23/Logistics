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

**The point we would like you to rule on before we go further.** max φ turns out to
be the hinge-loss primal ½‖w‖² + C Σ [1 − yᵢ(⟨w,φᵢ⟩ + b)]⁺. We know this is
biconjugacy and not a discovery, and we can see the objection: that we have dualised
our way back to the problem we started from, and are then running a known
cutting-plane method on it. We think the dualisation earns its place, for one
specific reason we measured rather than assumed. On wine with a Gaussian kernel, the
knapsack solution α*(λ) agrees with the plain hinge subgradient pattern
[C if yᵢfᵢ < 1] on only 41–83% of samples, but with the biased pattern
[C if yᵢ(fᵢ + b) < 1] on 100% of them, where b is the multiplier the knapsack itself
returns. So our oracle is the hinge oracle with the bias optimised out exactly at
every iteration, and the sort is that one-dimensional minimisation over b. The
cutting-plane and bundle methods for regularised risk minimisation drop the bias
precisely because with it the empirical risk is no longer a separable sum and the
closed-form subgradient is lost; what restores it is the equality yᵀα = 0, which is
in D only because b was a free variable of the primal. Relaxing the identity, rather
than writing the primal down directly, is what makes a biased oracle closed-form.

Two further things follow that the direct primal route does not give: the master
multipliers return a feasible α̂ ∈ D, so we recover the SVM dual variables and not
just the weights; and φ(λ) together with the restricted problem bracket v(D) from
both sides, so every iteration carries a certified gap.

We are claiming no more than that: the family of cuts and the function being
optimised are the same as in the published primal methods, and what the Lagrangian
formulation adds is the bias, the dual variables, and the two-sided bound. If you
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
