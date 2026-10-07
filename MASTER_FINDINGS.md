# Master Findings — ML-Driven Inverse Design of Microring Resonators for CPO

**Author:** Md Shidul Islam Shihab, Dept. of Electronics & Communication
Engineering, KUET — mdshidulislamshihab@gmail.com
**Record date:** 2026-10-07
**Status:** Draft v2 complete. All seven objectives resolved. One structural
gap remains before Q1 submission (see §9).

This file is the single authoritative record of what was done, what was
measured, and what is still open. Every number here was produced by a
documented mode of `mrr_template.py`; none is quoted from memory.

---

## 1. What the project set out to do

Resolve, for a four-parameter bus-coupled add-drop silicon microring
resonator, a disagreement in the inverse-design literature: Liu et al. [3]
showed a naive non-tandem inverse network can fail on a redundant design
space; Chen & Jiang [7] and Adibnia et al. [9] reported naive inverse
networks converging acceptably. No prior study settles it for a microring,
and none benchmarks classical ensemble ML against DNN/tandem on the same
data with the same metrics.

**Design space:** R 5–15 µm, w 400–500 nm, g 150–350 nm, L_c 0–3 µm.
220 nm SOI, Si/SiO₂, fundamental TE, 1500–1600 nm, dispersive materials.
Round-trip length L = 2πR + 2L_c. Both couplers share g and L_c.

**Targets:** λ_res, FSR, Q_L, IL.

---

## 2. How the dataset was built (the key methodological decision)

Full 3-D FDTD of the complete ring measured **~35 h/sample** — 83 days for
1,000 samples. 2-D ran in 149 min/sample but gave Q_L **11.3× too high**.

**Dual-fidelity construction (Section 7 fallback):** simulate in 3-D FDTD only
the single-pass coupler to obtain κ²; assemble the ring response analytically
from coupled-mode theory, with n_eff/n_g from a 3-D mode sweep and losses from
a bend sweep. Total ~16 h.

**Coupler model (the origin of everything that follows):**

```
κ² = sin²θ,   θ = κ₀(w)·e^(−γ(w)·g)·(L_c + B·√(2πR/γ))
```

g and L_c enter **only** through θ. Fitted B = 1.0211 (first-principles
value: 1).

**Coupler set:** 48-point LHS (seed 42) + 152-point augmented LHS (seed 2026)
= **200 3-D FDTD coupler simulations**. The extension widened κ² from
0.0021–0.3138 to 0.0019–0.4599 (+47%), improved CV R² 0.9961→0.9976, held
CV MAPE at 3.36%, and moved B from 1.0425 toward unity. Residuals flat across
2.5 decades — no sin²θ saturation.

**Dataset:** 1,150 labelled geometries (1,000 primary LHS + 150 supplementary
batch), split 805/172/173 train/val/test, seed 42. Test set used once.

---

## 3. Validation of the construction

| Test | Result |
|---|---|
| Published device [11]: FSR, 440×220 nm SOI TE, R = 20 µm | predicted 4.4008 nm vs **4.400 nm measured**; n_g 4.34429 vs 4.3451 implied |
| κ² model, 8-fold CV | MAPE 3.36%, R² 0.9976 |
| κ² model, 68 out-of-sample FDTD runs | MAPE 1.23–5.16%, **100% within 10%**, bias −0.36 to +3.33% |
| Energy conservation | κ² + t² = 1.0000 on every run |

**Caveats that must be stated whenever O1 is cited:**
1. Only FSR is validated. FSR depends on n_g and L alone — it tests the mode
   solver and length convention, **not** the coupling model, bend loss, Q_L
   or IL.
2. Agreement must be reported as "within the published figure's precision",
   not as 0.02%. Tang et al. quote 4.4 nm to 2 s.f. (±1.1%).
3. ROUGHNESS_DB_CM = 1.3 dB/cm is taken from the same paper. It does not
   enter the FSR prediction, but [11] is not an independent source for the
   loss model.
4. Tang et al.'s Q ≈ 3×10⁵ is a **stated assumption** in a scalability
   projection, not a measurement. Never use it as a validation target.
5. Nominal n_g = 4.2 in Phase 1.2 was justified by citing this device. That
   nominal enters no dataset label (labels use the fitted sweep), but the
   overlap should be disclosed.

---

## 4. Objective outcomes

| | Required | Result | Verdict |
|---|---|---|---|
| **O1** | ≥800 geometries; FSR within 10% of a published device | 1,150; FSR matches within quoted precision | **MET** |
| **O2** | R²≥0.90 & MAPE≤10% on ≥3 of 4 | 2 of 4 | **NOT MET** |
| **O3** | RMS-RGE ≤15% **and** closed-loop ≤20% | 15.506% / 3.669% | **NOT MET** |
| **O4** | 4 families, identical split, all metrics | done, no omissions | **MET** |
| **O5** | ≥5 sizes, explicit verdict | done | **MET** |
| **O6** | dual criterion, both stated | geometry fails, response passes | **RESOLVED** |
| **O7** | ≥2 CPO scenarios, explicit verdict | 0 of 12, FDTD-verified | **NOT MET** |

---

## 5. The central result (O6)

**Geometry-space criterion FAILS.** Naive DNN per-parameter R²:
R **0.9930**, w **0.2920**, g **0.6914**, L_c **0.2166** — one of four clears
0.80. Every method shows the same pattern.

**Response-space criterion PASSES.** Closed-loop re-simulated error
**3.669%** against a 20% bar, all 20 predictions realisable. Per target:
λ_res 0.203%, FSR 0.980%, Q_L 4.037%, IL 3.594%.

**Both prior literatures are correct.** They measured different things. The
disagreement was never empirical.

### Why — constraint counting

- λ_res carries almost no geometric information (see §6).
- FSR fixes L = 2πR + 2L_c.
- Q_L and IL both act through κ² = sin²θ, so they constrain **one** phase θ.

Four targets → **two effective constraints** (L and θ) → four unknowns →
**two-dimensional degenerate family**.

Confirmed: R is recoverable (R² 0.9930) because L is dominated by 2πR; w, g,
L_c are not (R² 0.13–0.69) because they enter only via θ. The naive network
recovers **θ itself to 1.824%**, with **96.5%** of predictions within 5% of
the correct level set. Four geometries with g 210–270 nm and
L_c 0.209–2.353 µm share Q_L = 14 659.8 exactly.

---

## 6. λ_res is unlearnable

| | R² | MAPE % |
|---|---|---|
| DNN | 0.0069 | 0.158 |
| RF (raw) | **−0.0950** | 0.166 |
| RF (norm) | **−0.0688** | 0.164 |
| XGB (raw) | **−0.0837** | 0.168 |
| XGB (norm) | **−0.0795** | 0.168 |

All four tree variants are **worse than predicting the mean**. Over five
seeds the DNN gives R² = 0.0078 ± 0.0011.

**Cause:** dλ_res/dR ≈ 0.15 nm/nm. Placing a resonance to 1 nm needs R to
~7 nm — one part in 1.5×10⁴ of the range.

**Consequence:** λ_res was 99.6% of forward residual loss, so the
144-configuration architecture search was ranking configurations on noise.

**Reparameterisation will not rescue it:** within an already-matched set,
mean |corr(λ_res, g)| = 0.095 — λ_res is near-redundant with FSR for geometry
recovery.

---

## 7. Three methodological results beyond the objectives

### 7.1 Geometry-space error does not predict response-space error

| model | family | RMS-RGE % | θ err (built) % | closed-loop % | ratio |
|---|---|---|---|---|---|
| naive DNN | neural | 15.506 | 1.723 | 3.669 | 2.129 |
| RF (raw) | tree | 16.039 | 2.095 | 4.589 | 2.190 |
| Tandem† | neural | 29.461 | 2.654 | 5.936 | 2.237 |
| XGB (norm) | tree | 16.546 | 7.129 | 13.329 | 1.870 |

† over the 8 of 20 buildable predictions.

- **closed-loop / θ: spread 1.20×** (ratios 1.87–2.24, mean 2.11)
- **closed-loop / RMS-RGE: spread 4.00×** (0.20–0.81)

naive DNN and XGB(norm) differ by **1.04 pp** in RMS-RGE and by **3.6×** in
closed loop. Model family is controlled: RF(raw) is a tree with low θ error
and behaves like the network. Within the tree family alone θ varies 3.4× and
closed-loop 2.9× while RMS-RGE differs by 3%.

**Ordering is not a discriminating test** — RMS-RGE happens to rank three of
four correctly. What it cannot do is say *how much* worse.

### 7.2 Physical realisability is architecture-dependent and invisible to both metrics

| model | buildable |
|---|---|
| RF (raw), RF (norm), XGB (raw), XGB (norm) | **100%** |
| naive DNN | 95.4% |
| Tandem | **33.5%** |

Tandem produced 12 of 20 unbuildable closed-loop predictions: 11 with
negative L_c (to −0.881 µm), 1 with w = 397.8 nm.

**Mechanism:** tandem is penalised only through the frozen forward surrogate,
which happily evaluates L_c = −0.88 µm and returns a plausible response.
Nothing in the objective encodes that the design space has edges.

**Random Forest's 100% is provable:** its prediction is a mean of leaf values,
hence a convex combination of training targets, hence inside their convex
hull; the design box is convex. XGBoost is additive, not convex — no
guarantee, but violated nothing here.

### 7.3 A reproducible information bound

Bank of 10⁶ geometries over the design box, labelled with the same analytic
responder and the same quality gate. For each test target, average the k
nearest responses and compute RMS-RGE against truth. This is the error of a
**specific predictor**, hence an **upper bound** on the Bayes-optimal error.

| bank N | bound % | best k |
|---|---|---|
| 12 500 | 16.063 | 25 |
| 25 000 | 15.643 | 10 |
| 50 000 | 15.612 | 25 |
| 100 000 | 15.596 | 25 |
| 200 000 | 15.479 | 50 |
| 400 000 | 15.479 | 100 |
| 800 000 | 15.005 | 5 |
| **1 000 000** | **14.626** | 7 |

**The asymmetry governs what can be concluded.** An upper bound *below* a
threshold proves the true bound is below it; *above*, it proves nothing.

Since 14.626% < 15%: **O3's threshold lies above the information content of
four scalar targets. O3 is reachable in principle; the shortfall is a
model-and-data result, not a fundamental limit.** At least **0.879 pp** of
headroom remains.

**Not converged:** best k drifts 25→7 and the descent accelerates from 0.096
to 0.530 pp per e-fold. A shifting optimum means the estimator's own
finite-sample bias is still unwinding. No rate from this sweep describes the
true bound, and a 3-parameter asymptotic fit is **not identifiable**.

**Per-parameter values are NOT individual bounds** — the estimator minimises
the aggregate, which is why the best model legitimately beats two of them
(w 5.321 vs 6.054; g 14.144 vs 14.682).

---

## 8. Other measured results

**Forward (O2).** Best = DNN at 3.568% aggregate MAPE (XGB norm 5.323%,
RF norm 11.929%). Over seeds 42–46: λ_res 0.158±0.000 (0/5 pass),
FSR 0.556±0.020 (5/5), Q_L 3.059±0.109 (5/5), **IL 11.331±1.101 (1/5)**.

IL's failure is a metric artifact: RMSE 0.072 dB, R² 0.9966, but dataset IL
has median **0.524 dB**, 48.3% of rows below 0.5 dB, minimum 0.032 dB. A
constant 0.07 dB error gives 13.7% at the median and >200% at the extreme.
Clearing 10% would need absolute accuracy below ~0.052 dB. Report O2 as not
met under the criterion as written, and note 0.072 dB is inside typical CPO
link-budget tolerance.

**Inverse (O3/O4).** RMS-RGE aggregate | R | w | g | L_c | R,w,g only:

```
naive DNN   15.506 | 2.841 | 5.321 | 14.144 | 26.930 | 8.878
Tandem      29.461 | 6.079 | 7.852 | 15.296 | 56.029 | 10.529
RF (raw)    16.039 | 5.942 | 5.741 | 14.378 | 27.460 | 9.574
RF (norm)   15.835 | 3.730 | 5.926 | 14.213 | 27.421 | 9.148
XGB (raw)   16.451 | 2.949 | 5.925 | 14.847 | 28.607 | 9.385
XGB (norm)  16.546 | 2.986 | 5.941 | 14.687 | 28.900 | 9.308
```

**Sample efficiency (O5).** Forward: XGB best ≤300 samples (13.030% mean over
100/200) vs DNN 17.465%; DNN best at full size (3.568%). Inverse: naive DNN
best at both ends (16.411% ≤300; 15.506% full).

Tandem inverse across sizes: 26.6, 36.8, **104.3**, 59.2, 29.5 — not a
learning curve; consistent with box escape.

**Difficulty ranking.** By mean R²: L_c (−0.2645) > w (0.0243) > g (0.6639) >
R (0.9866). By mean RGE: L_c (32.558) > g (14.594) > w (6.118) > R (4.088).
The two orderings **disagree on w vs g** — another instance of
metric-dependence. Under both, **the g hypothesis is REFUTED**: L_c is
hardest, not g.

**CPO translation (O7).** 0 of 12 candidates meet any scenario, FDTD-verified.
Tolerance on λ_res is half the channel spacing: ±2.0 nm (A), ±0.4 nm (B).
**11 of 12 failures involve λ_res.** The exception: RF(raw) in Scenario A
placed λ_res at 1548.058 nm (inside ±2 nm) and missed FSR by 0.13 nm.

Naive DNN closed-loop λ_res error: 3.15 nm MAPE, **4.32 nm RMS**, 10.9 nm
worst. Misses Scenario A by 2× and Scenario B by 11×. Via
dλ_res/dR ≈ 0.15 nm/nm, Scenario A needs R to ~13 nm and B to ~3 nm — below
lithographic tolerance.

**Three independent measurements agree** that absolute resonance placement is
out of reach: the Phase 1 sensitivity analysis, closed-loop re-simulation, and
scenario translation. **Inverse design delivers the shape of the response and
cannot place the resonance. CPO needs post-fabrication trimming.**

---

## 9. THE REMAINING GAP (blocks Q1)

**The ring model is unvalidated.** The κ² model has 68 out-of-sample FDTD
runs. The mode solver has the Tang FSR check. But the coupled-mode assembly
that produces **Q_L and IL** is checked against nothing — the closed-loop test
re-simulates the coupler and recomputes the ring with the same analytic model
that made the labels.

This matters because Q_L and IL carry the entire degeneracy argument. The
claim that both act only through θ is a property of the analytic model.

**Fix:** 3–5 full 3-D FDTD ring simulations comparing simulated Q_L and IL
against analytic predictions. ~35 h each → 4–7 days. This is the single
highest-value remaining experiment.

---

## 10. Decisions taken, to be defended in print

1. **O3 was purchasable and declined.** Enlarging training data with rows
   drawn from the *fitted* analytic model lowers RMS-RGE below 15%
   (measured 14.981% at ~6,229 rows). Those rows add sampling density, not
   physics; the dataset would still carry the information of 200 coupler
   simulations, and closed-loop error against FDTD would not improve. **Not
   claimed.** Recorded in paper §4.6.
2. **No extrapolated floor asymptote reported** — the sweep does not identify
   one, and a grid search for it runs to the edge of its own search range.
3. **Per-parameter floors not reported as bounds.**
4. **O1 agreement reported as "within the published figure's precision"**, not
   0.02%.

---

## 11. Still open

- [ ] **Full 3-D ring validation** (§9) — blocks Q1
- [ ] **Independent photonics expert review of the methods.** Not done. Nine
      inferences made during this project turned out wrong and were caught
      only by measurement; the dual-fidelity construction is where that
      failure mode hides unmeasured.
- [ ] LICENSE file for the repository
- [ ] Ref [12] (Torabi et al.) — unrefereed preprint; check for peer-reviewed
      version
- [ ] Ref [13] (Bogaerts et al.) — typeset Eq. 6 never read directly
      (paywalled); confirm with library access
- [ ] Replace the data-availability placeholder with a real URL/DOI
- [ ] Seed repeats for the sample-efficiency curve (currently single-seed)

---

## 12. Script modes (`mrr_template.py`)

```
--generate      build the dataset          --phase2      preprocessing + figures
--phase3        forward model + grid       --phase4      naive + tandem inverse
--phase5        RF / XGBoost               --phase6      master tables
--phase7        CPO scenarios              --phase7sim   re-simulate candidates
--closedloop    closed-loop re-verify      --ablate      3-variant ablation
--physinv       physics-decoded inverse    --density     training-density diag
--couplerset    coupler LHS                --couplerext  augmented coupler LHS
--kappafit      8-fold CV of kappa^2       --modesweep   n_eff / n_g sweep
--seedcheck     seed stability of Phase 3  --thetaall    theta vs closed-loop
--floor         information bound          --floorconv   bound convergence
--validate      O1 literature validation   --ringvalidate full 3-D ring check
```

Flags: `--n=`, `--nbank=`, `--no-movie`, method flags for `--closedloop`
(one only — the parser takes the last).
