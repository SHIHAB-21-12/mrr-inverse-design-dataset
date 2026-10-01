# Results log — ML-Driven Inverse Design of Microring Resonators for CPO

Single cumulative record of everything run and everything found.
**Updated after every run.** Newest results are appended to each phase's
section; nothing is deleted, including results that were later corrected.

Repo: github.com/SHIHAB-21-12/mrr-inverse-design-dataset
Script: `mrr_template.py` (one file, all phases)
Last updated: 2026-10-01, after the physics-decoder and learning-curve work.

---

## Status at a glance

| Objective | Status | Evidence |
|---|---|---|
| O1 dataset | **met** | N_clean = 1150; literature FSR error 0.02–0.33% vs ≤10% |
| O2 forward surrogate | **met** | 3 of 4 targets at R² ≥ 0.90 and MAPE ≤ 10% |
| O3 inverse design | **split** | closed-loop 3.37% ✓ (≤20%); RMS-RGE 15.45% ✗ (≤15%) |
| O4 four-way comparison | **met** (partial) | Phase 6 tables complete; closed-loop only for naive DNN |
| O5 sample efficiency | **met** | XGBoost → DNN crossover between 200 and 400 samples |
| O6 one-to-many | **answered** | geometry R² fails, response-space passes — Paper 9's pattern |
| O7 CPO translation | **met** | XGBoost (raw) passes Scenario A on all four specs |

---

## Phase 1 — Dataset generation

**Tried.** Full 3-D FDTD of the complete ring; then 2-D full ring.
**Result.** Both failed their own thresholds. 3-D: ~35 h/sample, 84× over the
25-min trigger, with a simulation window 10× too short. 2-D: 149 min/sample
(6× over) **and** Q_L 11.3× the 3-D value, because 2-D has no vertical
radiation loss.

**Tried.** Section 7 dual-fidelity fallback — simulate only the single-pass
coupler in 3-D, compute the full-ring response analytically.
**Result.** Worked. 48 coupler runs + 30 mode solves + 5 bend runs ≈ 16 h,
replacing what would have been 83 days.

**Defects found and fixed (all mine, all cost real time):**
- Mesh sweep was inert — mesh accuracy 3 already gives ~32 nm in silicon, so
  override points at 40 and 30 nm did nothing. Cost one wasted sweep.
- Unsolved mode source (`number of trial modes = 0`), found during the GUI
  audit; now set in the script so no manual fix is needed.
- Three extraction defects in one pass: spectral grid 0.050 nm vs a 0.029 nm
  true linewidth; mode hopping between resonances 9.2 nm apart; λ_res
  snapping to the DFT grid. Same geometry before/after: T_peak 0.141 → 0.784,
  IL 8.51 → 1.06 dB, Q_L 10,660 → 53,917.
- My own verdict criterion was unreachable by construction — the
  `FWHM/transform_limit ≥ 3` test has a ceiling of 2.07 for a 1e-5 shutoff.

**Validation chain:**

| check | result |
|---|---|
| Literature (Paper 11 fabricated chip) | FSR error 0.02–0.33% vs ≤10% target |
| Method (analytic vs full-ring FDTD) | Q_L 1.60%, T_peak 0.41% |
| κ² model, 8-fold CV | 3.37% MAPE, R² = 0.9961, 48/48 within 10% |
| Loss cross-check | 1.48 dB/cm vs Paper 11's measured 1.3 |
| **κ², held out on 20 new geometries** (from the closed-loop run) | **2.95% MAPE, 100% within 10%** |

**Fitted κ² model.** θ = κ₀(w)·exp(−γ(w)·g)·(L_c + B·√(2πR/γ)), κ² = sin²θ,
with B = 1.04 against a first-principles 1, and fitted decay 0.0082/nm
against 0.0075/nm from measured n_eff.

**Known limitations (stated in README):**
1. The dataset carries the information of **83 simulations, not 1000**.
2. Sidewall roughness not simulated; 1.3 dB/cm literature value added.
3. Only 9 of 1000 original samples meet Scenario A's FSR ≥ 16 nm.

---

## Phase 2 — Preprocessing

**Result.** Nothing to clean — no NaN, no flags, no drop-port peak above 1.
N_clean = 1000.

**Tried.** Coverage check against the Phase 7 scenarios.
**Result.** Scenario A had only **3** feasible samples. Generated a targeted
supplementary LHS batch (150 samples, seed 202, R 5.0–5.6 µm, g 150–290 nm,
L_c 0–1.6 µm, rejecting draws with 2πR + 2L_c > 34.85 µm).
Scenario A went **3 → 24**; Scenario B 304 → 331. N_clean = 1150.

**This decision paid off in Phase 7:** Fix 7b predicted ~70 training samples
in Scenario A's region under plain LHS; the supplementary batch gave **153 of
805 (19%)**, and Scenario A is the scenario that passes.

**Split.** 805 / 172 / 173, seed 42, verified no geometry shared across
splits. Scaler fitted on the training split only.

**Flagged, not changed.** Phase 2.5's engineered-feature ablation assumes
κ(g,L_c) is an independent estimate. In this dual-fidelity dataset `kappa2`
is the exact intermediate the labels were computed from, so using it as a
feature is circular. Report as a diagnostic, not as a feature-set comparison.

---

## Phase 3 — Forward model

**Result.** Criterion **met**, 3 of 4 targets.

| target | R² | MAPE | RMSE |
|---|---|---|---|
| λ_res | 0.0058 | 0.159% | 2.943 nm |
| FSR | 0.9994 | 0.537% | 0.090 nm |
| Q_L | 0.9983 | 3.469% | 1006 |
| IL | 0.9972 | 9.612% | 0.0685 dB |

**The λ_res finding.** A constant predictor scores R² = −0.0032, MAPE 0.160%,
RMSE 2.956 nm. The trained network is **0.4% better than a constant.**

Cause, established not guessed: resonance means n_eff·L = m·λ, so
dλ_res/dR ≈ 0.15 nm per nm of R. A 58 nm change in R moves λ_res by a full
FSR. Across the design space λ_res sweeps the window **~173 times**, and
predicting it to 0.1 nm would need R resolved to **1 part in 15,000**.

This is why every deployed CPO microring carries a thermal tuner — the data
independently reproduces the central engineering problem of the field.

**Consequences:**
- MAPE is misleading for λ_res (0.159% looks excellent, means nothing). The
  Phase 6 table must carry R².
- **99.6% of the residual validation loss is λ_res**, so the 144-configuration
  hyperparameter search ranked configurations on noise — which is why the
  winner sits at the maximum of all four grid axes.

---

## Phase 4 — Inverse and tandem

**Naive baseline.** RMS-RGE 15.501% (seed 42). Per-parameter: R 2.852%,
w 5.336%, g 14.000%, L_c 26.992%. R/w/g only: 8.805%.

**Tandem.** RMS-RGE 31.438%, worse on geometry exactly as Phase 4.4 step 2
predicted. Frozen-weight check returned `True` — verified automatically.

**The degeneracy finding.** Q_L and IL depend on (g, L_c) only through θ, so
level sets are curves, not points. Four geometries spanning 30% of the g
range and 71% of the L_c range give **identical Q_L = 14659.8**:

| g (nm) | L_c (µm) | Q_L | IL (dB) |
|---|---|---|---|
| 210 | 0.209 | 14659.8 | 0.513 |
| 250 | 1.500 | 14659.8 | 0.501 |
| 270 | 2.353 | 14659.8 | 0.493 |

**θ test.** naive: θ MAPE 1.787%, 96.5% within 5%, while g is off 10.7% and
L_c off 22.8%. The network lands on the right level set with different
coordinates — its "error" is a different valid design.

**Tandem pathology.** 99.2% of its response-space loss is λ_res. It hit the
λ_res floor at epoch ~25 and ran 237 more epochs against a flat gradient.
θ MAPE 6.900%, only 51.4% within 5% — **worse than the naive network at the
one thing the tandem exists to do.** Its mechanism was never exercised.

---

## Phase 5 — Random Forest / XGBoost

**Forward:** DNN (3.44% aggregate) > XGBoost (5.05%) > RF (11.84%).

**Inverse RMS-RGE:** naive DNN 15.501 < RF norm 15.706 < RF raw 15.856 <
XGB raw 16.370 < XGB norm 16.449 < tandem 31.438.

**Normalization test (5.3).** Helps both forward models, hurts both inverse
models. Consistent across two unrelated algorithms — a reportable pattern,
and a different shape from Paper 6's per-target split.

**Feature importance flips with preprocessing.** RF raw: g 55.4% top. RF
norm: R 52.8% top. XGBoost barely moves (R 32.6% → 32.5%). The reason is the
Phase 5.1 multi-output choice: RF uses *native* multi-output so one tree
splits on summed variance and the largest-spread target dominates; XGBoost is
wrapped in `MultiOutputRegressor`, fitting four independent models.
**XGBoost's stability is the control proving RF's instability is an artifact.**

**Flagged, not changed.** Phase 5.2 fixes the search criterion as mean MAPE
across outputs. For the inverse models L_c reaches 0 by design, so L_c MAPE
runs to 400%+ and dominates selection — the same pathology Fix 4b already
solved for RMS-RGE. Supervisor question.

---

## Phase 6 — Unified evaluation

**Sample efficiency (O5), forward aggregate MAPE:**

| | n=100 | n=200 | n=400 | n=600 | n=805 |
|---|---|---|---|---|---|
| XGBoost | **15.14** | **10.58** | 7.46 | 5.60 | 5.05 |
| DNN | 19.95 | 18.47 | **4.77** | **5.54** | **3.44** |
| RF | 35.72 | 25.76 | 16.28 | 12.99 | 11.84 |

**Crossover between 200 and 400 samples** — exactly the small-data hypothesis
that motivated including classical ML, measured rather than assumed.

**Tandem instability.** correlation(tandem val loss, tandem RMS-RGE) =
**−0.674**. At n=400 it reached its lowest loss (0.00363) and its worst
geometry (RMS-RGE 103.9%), because the surrogate it optimises against was
still weak enough to exploit. **A low tandem loss can mean an exploitable
forward model, not a good design.**

**Per-parameter difficulty (6.4).** Hypothesis was g. **Refuted.**
Hardest first: **L_c (R² −0.347) > w (0.007) > g (0.673) > R (0.986).**
Three of four parameters are poorly identified, and all three are the
coupling-related ones.

---

## Closed-loop re-verification

20 representative test targets, predicted geometries re-simulated through
fresh 3-D FDTD coupler runs. All 20 realisable.

| target | MAPE | RMS | max |
|---|---|---|---|
| λ_res | 0.272% | 0.330% | 0.680% |
| FSR | 1.001% | 1.404% | 4.298% |
| Q_L | 3.647% | 4.336% | 10.417% |
| IL | 3.625% | 4.948% | 16.917% |

**RMS closed-loop response error = 3.368%** against a 20% threshold.

**The geometries are 15.5% "wrong" and produce the right answer to 3.4%.**
Independent confirmation of the degeneracy argument.

κ² + t² = 1.0000 on all 20 runs (energy conservation).

**O6 under both criteria (Fix 14c):**
- Geometry-space (R² ≥ 0.80): **fails** — only R (0.993) passes; g 0.698,
  w 0.289, L_c 0.213.
- Response-space: **passes**, 3.368% vs 20%.

The document names this exact outcome "precisely Adibnia et al.'s reported
situation and a legitimate, publishable O6 outcome — not a null result."

---

## Phase 7 — CPO scenarios

**Target construction rule (mine, not the document's).** 7.3 states specs as
bands or bounds; 7.4 step 1 needs point targets. Rule used: a bound targeted
at its stated value, a band at its geometric mean, IL at half its limit.
A = (1550, 16.0, 3464, 0.75); B = (1550, 6.4, 7750, 0.50). **A different rule
would move these results — state it alongside them.**

**Scenario A: MET by XGBoost (raw).** O7 satisfied.

**Scenario B: every method fails on λ_res and nothing else.**
After FDTD re-simulation:

| method | Q_L (≥7750) | IL (≤1.0) | λ miss | tuning to fix |
|---|---|---|---|---|
| naive DNN | 8598 (+10.9%) | 0.208 | −1.58 nm | 23 K |
| **RF (norm)** | **11467 (+48.0%)** | **0.283** | **+0.79 nm** | **11.5 K** |
| RF (raw) | 10087 (+30.2%) | 0.352 | −1.14 nm | 17 K |
| tandem | 8608 | 0.198 | −2.50 nm | 36 K |
| XGB (raw) | 9198 | 0.221 | −3.12 nm | 46 K |
| XGB (norm) | 9181 | 0.220 | −2.99 nm | 44 K |

Every method delivers the bandwidth, selectivity and loss Scenario B needs,
and misses only the spec that integrated heaters trim as a matter of course —
within 11–46 K, which is routine.

**Re-simulation agreement:** Q_L 3.73% MAPE, IL 3.82%. **All 12 verdicts
unchanged.**

**Caveat that must be stated.** λ_res and FSR came back identical to three
decimals because neither depends on κ² — a coupler re-simulation cannot move
them. This run validates the coupling half only. λ_res and FSR rest on the
mode-sweep model, validated separately in sub-study 3 against Paper 11.

---

## Ablation study (training-side, no fixed default changed)

Five variants × three seeds (42/43/44), inverse direction.

| variant | RMS-RGE % | R term % | L_c term % | out-of-range |
|---|---|---|---|---|
| baseline (Phase 4) | 15.448 ± 0.038 | 3.011 ± 0.155 | 26.931 ± 0.093 | 4.7 |
| + bounded outputs | 15.505 ± 0.098 | 3.201 ± 0.068 | 26.908 ± 0.174 | **0.0** |
| + metric-aligned loss | **15.318 ± 0.084** | **2.886 ± 0.028** | 26.899 ± 0.171 | 1.0 |
| + L feature | 15.471 ± 0.011 | 3.127 ± 0.221 | 26.960 ± 0.045 | 8.3 |
| all three | 15.319 ± 0.056 | 3.165 ± 0.100 | 26.879 ± 0.132 | **0.0** |

**Essentially a null result.** Best is −0.130 pp against a baseline sd of
0.038 and its own sd of 0.084 — marginal at three seeds. O3 still not met.

**My predictions were wrong.** I expected the L feature to give the biggest
gain on R and the metric-aligned loss to be weakest. The opposite happened:
the L feature made R *worse* (+0.116) and tripled out-of-range predictions;
the metric-aligned loss was the only variant to improve R.

**The one unambiguous win:** bounded outputs drive out-of-range predictions to
exactly zero, by construction. Worth keeping regardless of RMS-RGE.

**CORRECTION to the Phase 5 guidance.** I told you O3 needed a 4.2% relative
improvement in R, on the reasoning that L_c ≈ 9.46 × R. **That is wrong.**
Across these variants R spans 2.886–3.201% (10.2% spread) while L_c spans
26.879–26.960% (0.3% spread) — correlation **+0.013**. L_c does not follow R.
It is pinned at 26.9%, which is 88% of its no-information floor of 30.45%.

Also: the Phase 4 headline R term of 2.852% was the luckiest of three seeds;
the honest baseline is 3.011 ± 0.155%.

**What O3 actually requires.** With L_c immovable at 26.9%, it alone
contributes 13.45% to the aggregate, leaving R² + w² + g² ≤ 176.4. The best
variant sits at 215.0 — an **18% reduction needed**, and g² is ~196 of it.
**g must fall from 14.0% to about 11.8%.** That matches the alternative
lever my Phase 5 analysis also identified (g below 11.61%); I promoted the
wrong one of the two.

---

## Information-floor study (2026-10-01)

**Question.** Is O3's 15% threshold reachable at all, or is it below this
design space's information limit?

**Method.** Built a 200,000-geometry response bank with the analytic model.
For each test target, found every geometry in the bank whose response matches
to the precision we actually achieve (the measured closed-loop RMS errors:
FSR 1.40%, Q_L 4.34%, IL 4.95%). The spread of each geometric parameter
within that matched set is the irreducible error — no model, however good,
can do better from the same measurement. Median 79 matching geometries per
target, 172 of 173 targets usable.

**Answer: O3 IS reachable.**

| | no-info | observed | floor | information captured |
|---|---|---|---|---|
| R | 43.22% | 2.89% | 2.38% | **99%** |
| w | 6.39% | 5.30% | 5.30% | **100%** |
| g | 26.40% | 14.00% | 12.45% | **89%** |
| **L_c** | 30.45% | 26.90% | **22.78%** | **46%** |
| aggregate | — | 15.46% | **13.30%** | — |

*no-info* = predicting the training mean. *floor* = best possible given the
response at our precision. *captured* = how far from ignorance to optimal.

**The floor is 13.30% against a 15% threshold — 1.7 pp of headroom.**

**L_c is the lever, and it is the only one with real room left.** R and w are
essentially optimal already (99% and 100% of available information captured);
g is at 89%. L_c is at **46%** — it is using less than half the information
the response actually carries about it. Closing L_c's headroom alone would
save 204.7 of the 56.0 reduction O3 needs; closing g's would save 41.0;
closing R's would save 2.7, and w offers nothing.

**CORRECTION to the ablation conclusion.** I wrote that L_c "is pinned at
26.9%, 88% of its no-information floor." The arithmetic was right but the
inference was wrong: I compared L_c against the *no-information* bound
(30.45%, predicting the mean) rather than against the *conditional* floor
given the measured response (22.78%). L_c is not at its limit — it has 15.3%
of relative headroom and is the single largest remaining opportunity. It
simply did not move under those three particular ablation variants.

**λ_res reparameterization will NOT help O3.** Tested directly: within a set
already matched on FSR, Q_L and IL, the mean |correlation(λ_res, g)| is
**0.095** — knowing λ_res tells you almost nothing extra about the geometry.
Physically, FSR = λ²/(n_g·L) already pins the round-trip length, and
λ_res = n_eff·L/m only adds the sub-FSR phase, a refinement on a quantity FSR
has already fixed. **λ_res and FSR are near-redundant for geometry recovery.**

This narrows supervisor question 1: reparameterizing λ_res would improve the
*forward* model's λ_res accuracy and would unblock a fair tandem test (99.2%
of the tandem's loss is this target), but it would not move RMS-RGE or O3.

---

## Physics-decoded inverse model (2026-10-01) — REFUTED

**Tried.** Network emits R, w, g only; L_c derived from the FSR constraint
L = λ²/(n_g(λ, w_pred)·FSR), L_c = (L − 2πR)/2, so the predicted L_c cannot
disagree with the requested FSR. Loss is Eq. 15 itself. Three seeds.

**A bias caught before running.** Oracle-testing the decoder with the *true*
R and w revealed a systematic **+0.141 µm** offset in derived L_c — 4.7% of
the L_c range, baked in before any learning. Cause: FSR = λ²/(n_g·L) is a
first-order relation while the dataset's FSR comes from actual
adjacent-resonance spacing. One scalar fitted on the training split
(c = 0.995903) drops the oracle error from 4.694% to **1.469%**.

**Result: worse than baseline.**

| | RMS-RGE | R | w | g | L_c |
|---|---|---|---|---|---|
| baseline | 15.448 ± 0.038 | 3.011 | 5.300 | 14.000 | 26.931 |
| physics-decoded | **16.080 ± 0.035** | 3.051 | **6.734** | 13.638 | 28.170 |
| floor | 13.30 | 2.38 | 5.30 | 12.45 | 22.78 |

R did not improve, so L_c inherited π-amplified error (28.17% measured
against 29.56% from pure propagation). **w got notably worse** — it was at
100% of its information floor, and giving it a second role (feeding n_g,
hence L, hence L_c) broke that.

**Useful negative result.** The free network's partial error cancellation
between R and L_c is worth more than imposing the physical constraint. The
cancellation is real — the free network achieves L_c 26.9% from R 3.011%
where π-propagation alone would give 29.6% — and the decoder destroyed it.

---

## Learning-curve analysis — why nothing moves

Fitting err(n) = floor + b·n^(−c) to the Phase 6 sample-efficiency data, with
the asymptote pinned at the measured floor of 13.30% (an unconstrained fit
returned 7.21%, which is **below** the floor and therefore impossible):

**b = 6.706, c = 0.1571**, fit RMS 0.088%.

| target | training samples needed |
|---|---|
| **15.0% (O3)** | **6,229** |
| 14.5% | 57,200 |
| 14.0% | 1,768,492 |

An exponent of 0.157 means error falls as n^−0.157 — brutally slow. You have
805. **O3 by data alone needs 7.7× the pool, and the next half-point after
that needs 57,000.**

**This explains why three architectural interventions all produced nothing.**
The gap between 15.45% and the 13.30% floor is finite-sample estimation error
on an exceptionally flat learning curve, not an architecture deficiency.

RF and XGBoost extrapolate to asymptotes of 15.11% and 15.45% — both already
above the threshold, i.e. saturated regardless of data.

---

## Decision taken: do NOT inflate the dataset to meet O3

Generating 6,229 analytic rows costs about three seconds, so O3 is
mechanically within reach. **It is not being used that way, deliberately.**

The extra rows would come from the same fitted κ² model. They add sampling
density, not physics. RMS-RGE would fall because the network learns the
analytic surrogate better; the closed-loop error against FDTD (3.368%) would
not move at all. The dataset carries the information of **83 simulations**
however many rows it holds, and that ratio is already stated in the README.

Reporting "O3 met" on that basis would be clearing a threshold by sampling a
surrogate more finely, and it would put the six legitimately-met objectives
under suspicion. The density study below is therefore run and reported as a
**convergence-rate diagnostic only**.

### What goes in the thesis
- **Headline O3 result stays the 805-sample number**: RMS-RGE 15.45% (not
  met), closed-loop 3.368% (met).
- The information floor (13.30%) and the learning curve as the explanation
  for why, with the cost of closing it quantified.
- The density table labelled as surrogate-density, not as added physics.

---

## Open questions for the supervisor

1. **λ_res reparameterization.** Every Scenario B failure, across all six
   methods, is λ_res alone. It is unlearnable as an absolute wavelength
   (R² ≈ 0 against a constant predictor). Detuning normalized by FSR, or the
   phase remainder, would be smooth. This is a `[DOC]` change — v1.3 fixes
   λ_res linear. It is also a prerequisite for O6's tandem comparison to be
   a fair test at all, since 99.2% of the tandem's loss is this target.
2. **Phase 5.2 selection metric.** Mean MAPE over outputs is unbounded for
   L_c, so inverse-model hyperparameter selection is driven almost entirely
   by L_c. Fix 4b already solved this for RMS-RGE; 5.2 was not given the
   matching treatment.
3. **Is the decision above the right one?** O3's geometry threshold can be
   met today by generating ~6,200 analytic rows from the existing κ² model.
   I judged that illegitimate and did not do it. Worth confirming.
4. **Which O3 criterion should lead?** Geometry-space RMS-RGE fails at
   15.45%; response-space closed-loop passes at 3.37%. The closed-loop result
   shows the geometry "error" is largely degeneracy, not inaccuracy.

## Still outstanding

- Closed-loop runs for the other five inverse methods (O4 completeness),
  5–8 h each
- Licence file
- **L_c-focused work: the one real lever on O3** (46% of available
  information captured, versus 89–100% for the other three)
