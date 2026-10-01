# Optimization ideas — to try once the full structure is complete

Running list. Each entry says what the evidence is, what to change, what to
expect, and — importantly — whether it is a free experiment or a change to
the locked v1.3 document that needs supervisor sign-off first.

Status key:  [FREE] = try it, changes nothing the document fixes
             [DOC]  = would change a fixed default; needs sign-off
             [COST] = costs simulation time

---

## The O3 target, quantified (added after Phase 5)

Phase 5 established exactly what O3 needs, so optimization now has a number
to aim at rather than a direction.

All five methods cluster at RMS-RGE 15.5-16.4% (tandem aside at 31.4%), and
the aggregate is dominated by the Lc term (~27%). Lc is not independently
learnable: FSR fixes the round-trip length L = 2*pi*R + 2*Lc, so Lc is
slaved to R, and the error is amplified by pi. For the naive DNN,
pi * (R RMSE 0.2666 um) = 0.837 um versus an observed Lc RMSE of 0.810 um —
96.7% agreement. Across the metric this makes Lc_term ~ 9.46 x R_term.

Substituting that into Eq. 15 with w and g unchanged:

| R term | implied Lc term | RMS-RGE |
|--------|-----------------|---------|
| 2.852% (current) | 26.99% | 15.501% |
| 2.750% | 26.03% | 15.078% |
| **2.731%** | 25.84% | **14.996%  <- O3 MET** |
| 2.500% | 23.66% | 14.058% |
| 2.000% | 18.93% | 12.112% |

**O3 needs a 4.2% relative improvement in R accuracy and nothing else.**
Every item below that improves R improves Lc about 9.5x as much, for free.
Items 1, 2, 4 and 5 all plausibly deliver more than 4.2% on R.

---

## Tier 1 — biggest expected effect

### 1. Reparameterize lambda_res  [DOC]
**Evidence.** Phase 3: R2 = 0.0058 against a constant-predictor's -0.0032.
The network learned essentially nothing. Cause is established, not guessed:
d(lambda_res)/dR ~ 0.15 nm per nm of R, so lambda_res sweeps the full window
~173 times across the R range and would need R resolved to 1 part in 15,000.
**Change.** Predict detuning from 1550 nm normalized by FSR, or predict the
phase remainder (n_eff*L mod lambda) directly, and reconstruct lambda_res
from it afterwards. Both are smooth functions of geometry.
**Expect.** This is the single largest available gain. It would likely move
lambda_res from R2 ~ 0 to something usable, and unblock items 2 and 3 below.
**Caution.** The document fixes lambda_res as a linear-scale target. Raise it
with the supervisor as a reviewer-anticipation question, not a fix.

### 2. Per-target loss weighting  [FREE]
**Evidence.** Phase 3: lambda_res contributes 0.00994 of validation MSE, the
other three targets 0.00004 combined — 99.6% of the residual loss is one
unlearnable target.
**Change.** Weight the four target terms in the MSE, or use learned
uncertainty weighting (Kendall et al. homoscedastic task weighting).
**Expect.** Lets FSR, Q_L and IL actually influence training and model
selection. Should also fix item 3.
**Note.** This is a training detail, not a documented default — free to try.

### 3. Select hyperparameters on a metric that discriminates  [FREE]
**Evidence.** All 144 configurations landed between 0.0099 and 0.0102, and
the winner sat at the maximum of all four grid axes (depth 4, width 256,
lr 5e-3, wd 1e-4). That is a search ranking noise.
**Change.** Rank configurations on validation MSE over {FSR, Q_L, IL}, or on
mean per-target R2, instead of the 4-target mean.
**Expect.** A meaningful winner, probably interior to the grid. Report both
rankings in the appendix — the contrast is itself a finding.

---

### 3b. Report theta alongside g and Lc  [FREE]
**Evidence.** Phase 4 established that Q_L and IL depend on (g, Lc) only
through the coupling phase theta, so a whole curve of geometries gives the
same response. Four geometries spanning 30% of the g range and 71% of the
Lc range produce an identical Q_L.
**Change.** Report theta accuracy next to per-parameter geometry accuracy in
the Phase 6 table.
**Expect.** Separates "found a different valid design" from "got it wrong" —
which RMS-RGE alone cannot do, and which is the heart of Objective O6.

---

## Tier 2 — worth trying, moderate effect

### 3c. Treat the tandem's loss as untrustworthy across sizes  [FREE]
**Evidence.** Phase 6.3 measured correlation(tandem val loss, tandem RMS-RGE)
= **-0.674**: the tandem fits its own objective BEST where its geometry is
WORST. At n=400 it reached its lowest loss (0.00363) and its worst geometry
(RMS-RGE 103.9%). The cause is that the tandem optimises against a surrogate
that is itself retrained at each size — a weak surrogate is easy to fool, so
a low tandem loss can mean an exploitable forward model rather than a good
design.
**Change.** Never select or early-stop the tandem on its own loss alone.
Monitor RMS-RGE (or the theta test) on the validation split in parallel, and
report the forward model's accuracy next to every tandem number.
**Expect.** Stops the tandem's reported performance from being an artifact of
surrogate quality. Combined with item 4 (bounded outputs), should remove the
non-monotonic blow-up entirely.

### 4. Bound the inverse networks' outputs  [FREE]  ** NOW URGENT **
**Evidence.** Phase 4 measured it: the tandem network put 120 of 173 L_c
predictions outside the design range, including NEGATIVE coupler lengths
(down to -0.989 um), which are physically meaningless. The naive network
put 6 of 173 R values below 5 um. Every such prediction asks the frozen
forward model to extrapolate.
**Change.** Sigmoid on the inverse output layer, so predictions are confined
to [0,1] scaled = exactly the design space.
**Expect.** Every proposed geometry becomes physically realisable, and the
tandem's forward half stops being asked to extrapolate. Should improve the
closed-loop error in Phase 6 even if RMS-RGE barely moves.

### 5. Feed the network the physics-relevant combination  [FREE]
**Evidence.** All four targets depend on geometry through round-trip length
L = 2*pi*R + 2*Lc and the modal index n_eff(w), not through R and Lc
separately. The network currently has to discover that.
**Change.** Add L (and optionally n_eff) as derived input features.
**Expect.** Faster convergence, better small-sample behaviour. This is a
legitimate engineered-feature ablation in the spirit of Phase 2.5 — and
unlike kappa2, L is NOT circular: it is pure geometry, computable before any
simulation.

### 6. Average over seeds instead of picking one  [FREE]
**Evidence.** Phase 3 config differences were within seed noise, so a single
run's winner is arbitrary.
**Change.** Train the selected configuration with 5 seeds and report
mean +/- std; optionally ensemble the predictions.
**Expect.** Error bars on every number in the Phase 6 table. Reviewers ask
for this, and it costs only training time.

### 7. Transform IL  [FREE]
**Evidence.** IL spans 0.03 to 9.86 dB and is strongly right-skewed; it was
the weakest of the three learnable targets (MAPE 9.61%, just inside the 10%
bar).
**Change.** log1p transform before scaling, back-transform for reporting —
the same treatment Q_L already gets.
**Expect.** A modest gain on the target closest to failing its threshold.

---

### 7b. Make the Phase 5 selection metric match the reported metric  [DOC]
**Evidence.** Phase 5 selects hyperparameters on mean MAPE across outputs.
For the inverse models Lc reaches 0 by design, so Lc MAPE runs to several
hundred percent and dominates the search — the same unbounded-relative-error
problem that Fix 4b already solved for RMS-RGE by range-normalising Lc.
**Change.** Select on RMS-RGE (or on a range-normalised MAPE) for the
inverse direction.
**Expect.** The inverse RF/XGBoost search would optimise the quantity the
thesis actually reports, instead of one dominated by a single ill-conditioned
term.

---

## Tier 3 — structural, higher cost

### 8. More coupler simulations  [COST]
**Evidence.** The dataset carries the information of 83 simulations, not
1000. Every analytic row descends from the 48-point coupler set, the 30 mode
solves and the 5 bend runs.
**Change.** Extend the coupler LHS set. Each point is ~5-55 min.
**Expect.** Better kappa2 model in the sparse corners, especially the
small-R / small-g region where Scenario A lives.

### 9. Enumerate the solution family  [DOC]
**Evidence.** The tandem network converges to one of many valid geometries —
documented and accepted in Section 4.2.
**Change.** Train several tandem networks from different initializations and
report the spread of geometries they propose for the same target.
**Expect.** A direct empirical measurement of design-space degeneracy, which
is the physical heart of the one-to-many problem O6 is about.
**Caution.** The document explicitly scopes enumeration out. Frame as future
work unless the supervisor wants it in.

### 10. What is NOT the lever
Depth and width. 144 configurations spanning depth 2-4 and width 32-256 all
landed within 2% of each other. Do not spend time enlarging the network.
