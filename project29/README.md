# Project 29 — Multi-class SVM, a Lagrangian dual, and a bundle method

Course 784AA (Computational Mathematics for Learning and Data Analysis), Prof. A. Frangioni.

- **(M)** kernel SVM for multi-class classification, standard One-vs-Rest.
- **(A1)** a Lagrangian dual approach whose dual is solved by a **proximal bundle method**.
- **(A2)** a general-purpose convex QP solver on the standard SVM dual, as reference.

## The decision

Prof. Frangioni offered two ways forward. **This branch takes the second one**: keep
the standard C-SVM (so (M) stays textbook One-vs-Rest) and find a *different dual
approach* for it. The shared-slack-budget model is kept, derived properly, in
Appendix A of `docs/formulation.tex` — it is no longer the main line, for reasons
given there.

## The idea in one paragraph

Both earlier proposals failed for the same reason: in the SVM dual

    min  ½ αᵀQα − 1ᵀα      s.t.   yᵀα = 0,  0 ≤ α ≤ C

*no constraint is the difficulty* — relaxing either one leaves a QP with the same
dense Hessian `Q`. The difficulty is `Q` itself. But `Q` is only there because `w`
was eliminated through the **representer identity** `w = Σᵢ αᵢyᵢφ(xᵢ)`. So put `w`
back as an explicit variable and dualise *that identity* instead of a constraint:

    φ(λ) = −½‖λ‖² + min_{α ∈ D} c(λ)ᵀα ,     cᵢ(λ) = −(1 + yᵢ⟨λ, φ(xᵢ)⟩)

The relaxed problem is a linear program over a box plus one equality — a
**continuous knapsack**, solved exactly by sorting in `O(m log m)`. This meets all
three requirements a Lagrangian approach needs:

| requirement | status |
|---|---|
| relaxation substantially easier than the original | a dense QP becomes a sort |
| dual multi-dimensional (so bundle is needed) | `λ ∈ ℝⁿ` (linear) or `ℝᵐ` (kernel) |
| dual genuinely non-differentiable | kinks sit exactly on `yᵢf(xᵢ) = 1` |

`φ` is finite everywhere (`D` is a non-empty compact polytope), so there are no
infinite cuts and every oracle call is a valid lower bound.

## Results we can already stand behind

Everything below is produced by `prototype/run_checks.py`.

- **Strong duality**: `max φ = v(D)` to ~1e-9 relative, for linear / RBF / polynomial
  kernels and `C ∈ {0.1, 1, 10}`. Weak duality: 0 violations in 3000 random multipliers.
- **The oracle is exact**: the `O(m log m)` knapsack matches a HiGHS LP solve to 1e-12
  over 40 random instances.
- **Primal recovery**: the master's convex multipliers give `α* ` to ~1e-7, and the
  knapsack's equality multiplier `θ` **is** the SVM bias `b` (verified to 1e-7, and
  separately as an identity over 200 random score vectors). No "average over the free
  support vectors" recipe needed.
- **Dantzig–Wolfe identity**: the master's dual is the SVM dual restricted to
  `conv{α¹,…,α^J}`, so the method is DW decomposition with the knapsack as pricing
  problem — and it carries a **certified gap at every iteration**, which SMO does not.
- **End to end**: (A1) and (A2) agree exactly on test accuracy (iris 0.956 linear /
  0.978 RBF; wine 1.000).

### Honest findings, to report as such

- **(A1) is slower than (A2)** on these sizes (e.g. wine+RBF 3.8s vs 0.05s). Expected:
  a dense QP of this size is exactly what an interior-point solver is best at.
- **Iteration count is governed by the number of free support vectors.** The oracle
  returns vertices of `D` (0/C vectors); every free SV must be produced by *mixing*
  vertices. Measured on wine: linear kernel → 12 free SVs → 26 iterations; RBF
  (γ=0.2) → 91 free SVs out of 178 → a bundle capped at 60 cuts **stalls** at gap
  2e-4, raising the cap to 200 converges in 153. Classic DW tailing-off, and a clean
  scaling law to report.
- **Warm-starting across One-vs-Rest classes does not help.** Transferring the
  multiplier is neutral-to-harmful (+72% oracle calls on wine+RBF); transferring cuts
  with a support-preserving repair is within noise (−4% … +2%). What *does* transfer
  is the **kernel matrix**: shared by all q problems and batchable into one BLAS-3
  product, worth a factor q in the dominant cost. This is the answer to the question
  the project statement requires us to discuss, and it is the one the measurements
  support.

## Layout

```
docs/formulation.tex     the mathematics (compiles clean, 7 pages)
docs/formulation.pdf     compiled
docs/email-draft.md      draft reply to Prof. Frangioni
prototype/
  knapsack_oracle.py     the O(m log m) continuous knapsack = the relaxation
  bundle_svm.py          kernels, the oracle, the proximal bundle, OvR cut transfer
  ovr_svm.py             (M) One-vs-Rest, trained by (A1) or (A2)
  run_checks.py          all the verifications quoted above
```

## Running it

```bash
pip install -r prototype/requirements.txt
cd prototype && python3 run_checks.py
```

## Still to do

- Larger datasets, and timing curves vs `m` for (A1) and (A2).
- A specialised master solver (the simplex QP) instead of calling cvxpy each iteration.
- The lockstep variant that batches the q oracle calls into one `K·[β¹…β^q]` product.
- Model selection (C, γ) and the OvR tie/rejection statistics.
