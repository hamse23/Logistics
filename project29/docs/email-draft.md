# Draft reply to Prof. Frangioni

> Draft only — read it, cut what you disagree with, and send it in your own words.
> Keep it short. The attachment does the work.

---

**Subject:** Re: 784AA Project 29 — revised formulation (Group 26)

Dear Prof. Frangioni,

We took the second option you offered: (M) stays the standard C-SVM with One-vs-Rest,
and we looked for a different dual approach for it.

Our two previous attempts failed for the same reason, which we had not seen clearly.
In the SVM dual

    min ½αᵀQα − 1ᵀα   s.t.  yᵀα = 0,  0 ≤ α ≤ C

neither constraint is the difficulty: relaxing the equality leaves a box-constrained
QP with the same dense Hessian (and a univariate dual, which was your first
objection), and relaxing the box leaves a QP as well. The difficulty is Q. But Q is
in the problem only because w was eliminated through the representer identity
w = Σᵢ αᵢyᵢφ(xᵢ). So we reintroduce w as a variable, which turns that identity into a
constraint, and dualise the identity rather than a constraint:

    φ(λ) = −½‖λ‖² + min_{α ∈ D} c(λ)ᵀα ,    cᵢ(λ) = −(1 + yᵢ⟨λ, φ(xᵢ)⟩),
    D = { yᵀα = 0, 0 ≤ α ≤ C }.

The relaxed problem is a linear program over a box with one equality constraint — a
continuous knapsack — solved exactly by sorting in O(m log m). So the relaxation is
genuinely easier than the original (a dense QP becomes a sort), the dual is
n-dimensional for a linear kernel and m-dimensional after the representer reduction
in general, and it is non-differentiable exactly where the knapsack has ties, which
unwinds to the support-vector condition yᵢf(xᵢ) = 1. D is a non-empty compact
polytope, so φ is finite everywhere and there are no infinite cuts.

Three things fall out that we think make this a good fit for the project rather than
just a legal one:

1. The master problem's dual is the SVM dual restricted to conv{α¹,…,α^J}, so the
   bundle method is Dantzig–Wolfe decomposition with the knapsack as pricing problem;
   the master multipliers give the primal α directly, and we get a certified gap at
   every iteration.
2. The multiplier of the knapsack equality is the SVM bias b, so b comes out of the
   oracle instead of being averaged over the free support vectors.
3. max φ is the hinge-loss primal ½‖w‖² + C Σ [1 − yᵢ(⟨w,φᵢ⟩+b)]⁺. We are aware this
   is biconjugacy rather than a surprise, but it connects (A1) to the cutting-plane
   and bundle literature for regularised risk minimisation, which gives us convergence
   rates to quote and compare against.

We implemented it before writing to you. Strong duality holds to ~1e-9 for linear,
Gaussian and polynomial kernels; the knapsack matches an LP solver to 1e-12; the
recovered α and b match a general-purpose QP solver to ~1e-7; and One-vs-Rest built
on (A1) and on (A2) give identical test accuracy. We also measured the thing the
project statement asks us to discuss: across the q One-vs-Rest problems, warm-starting
the multiplier is neutral to harmful and transferring cuts is within noise, whereas
sharing the kernel matrix (batching the q oracle calls into one BLAS-3 product) is
worth a factor q. We will report that negative result as such.

On the shared-slack-budget model: we wrote out its dual explicitly, as you asked, and
in doing so found that with a hard budget the primal can be infeasible — a sample
lying where two classifiers both score near zero needs about 1 unit of slack from
each, so B < 2 is typically already infeasible and the Lagrangian dual then diverges.
It is repairable (soften the budget with a penalised surplus, which also restores a
box on α), and we verified strong duality for the repaired model. We have kept all of
this in an appendix, but we are not proposing it as the main line: it would change
(M), which would confound a modelling change with an algorithmic one, and it leaves
the quadratic forms inside the dual function, which is what we were trying to get away
from.

Sections 1–3 and the appendix are attached. Is this an acceptable reading of (A1)?

Thank you,
Tolgonai Nasipbek kyzy (626449)
Hamse Hassan Adnan (683187)
