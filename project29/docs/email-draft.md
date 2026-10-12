# Draft reply to Prof. Frangioni

> Short on purpose: his own two emails were about 70 and 150 words. The report
> carries the argument; this only has to say what is enclosed and what we are asking.

---

**Subject:** 784AA Project 29 (Group 26) — report, parts 3.3 and 3.4

Dear Prof. Frangioni,

Following the suggestion to send the report in instalments and to settle the
theoretical part first, we enclose the parts described in Sections 3.3 and 3.4 of
the project information. The code and the experimental parts will follow.

We took the second option you offered. (M) stays the standard C-SVM with One-vs-Rest,
and we looked for a different dual approach. Our earlier attempts failed because
neither constraint of the SVM dual is the difficulty: relaxing either leaves a QP
with the same dense Hessian Q, and relaxing the equality leaves a univariate dual.
But Q is only there because w was eliminated through the stationarity condition
w = Σᵢ αᵢyᵢφ(xᵢ). So we put w back, which makes that identity a constraint, and
dualise it. What remains is a continuous knapsack solved by a sort in O(m log m), and
the dual is finite everywhere, m-dimensional, and nondifferentiable exactly at the
support-vector condition.

The point we would like you to rule on is in Section 4.7. max ψ turns out to be the
hinge-loss primal, so one can object that we have dualised our way back to where the
course starts. We think the dualisation still earns its place, because it eliminates
the bias exactly at every oracle call, returns the dual variables through the master,
and brackets v(D) from both sides. On a first small comparison this is worth a factor
of four to five in iterations. If you judge it too thin to count as a different dual
approach, we would rather know now than after the experimental work.

The budget model is in the appendix with its dual written out, as you asked. Doing
that showed us it can be infeasible: a sample where two classifiers both score near
zero needs about one unit of slack from each, so B < 2 is already infeasible and one
outlier destroys the problem. It is repairable, but we are not proposing it.

Thank you,
Tolgonai Nasipbek kyzy (626449)
Hamse Hassan Adnan (683187)
