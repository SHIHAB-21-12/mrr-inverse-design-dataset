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

## 9. FULL 3-D RING VALIDATION (the Q1 gap, now being closed)

Everything else validates the **coupler** (68 out-of-sample κ² runs) or the
**mode solver** (the Tang FSR check). The coupled-mode assembly that turns
κ², n_g and loss into **Q_L and IL** was checked against nothing: closed-loop
re-verification re-simulates the coupler and then recomputes the ring with the
same analytic model that produced the labels. Q_L and IL carry the entire
degeneracy argument, so this was the first thing a reviewer would find.

`--ringvalidate` simulates the **complete ring** in 3-D FDTD. All runs at
R = 5.20 µm, w = 450 nm, production 32 nm mesh, margins 1.0/0.9/0.7 µm,
auto-shutoff disabled so the DFT window is exactly the one the transform-limit
criterion assumed.

A 10 µm ring needs ~20 GB of solver memory at this mesh and a 14 µm ring
~36 GB, against 15.7 GB of system RAM — so the radius range **cannot** be
spanned on this machine. Validation varies g and L_c at fixed small R instead,
which exercises the coupled-mode assembly directly (Q_L and IL depend on
geometry only through κ²) and stresses the loss model hardest, since bend loss
is greatest at small R. **State the small-radius limitation in print.**

### Results — 9 runs, ~120 h, complete

Three geometries x three configurations: production mesh 32 nm, coarse mesh
40 nm, and 40 nm with 3x denser spectral sampling (a control for the
extraction method). All nine resolved (pts/FWHM 6.1–52.9,
FWHM/transform-limit 3.14–4.12).

**Simulated Q_L:**

| geometry | analytic Q | 32 nm | 40 nm | 40 nm, 6000 pts | sampling effect |
|---|---|---|---|---|---|
| G2 g=150 L_c=2.00 | 1855 | 1780 | 1749 | 1751 | +0.07% |
| G1 g=170 L_c=1.00 | 4264 | 4015 | 3892 | 3921 | +0.76% |
| G3 g=180 L_c=0.60 | 6351 | 4603 | 5229 | 5253 | +0.45% |

**Spectral sampling is NOT a factor.** Tripling the points across the
linewidth (6.1 → 17.8 for G3) moved Q by under 1% everywhere. The extraction
is sound and the numbers are robust.

### THE KEY FINDING: the validation was mesh-limited, not the model

Q depends on mesh, so **32 nm was never converged for Q**. Sub-study 2
established 32 nm for κ² — a *single-pass* quantity — and Q accumulates loss
over hundreds of round trips, so a per-pass numerical loss invisible in κ²
compounds.

Richardson extrapolation to zero mesh (2nd order assumed; two mesh sizes
cannot verify the order):

| | Richardson Q | analytic Q | error |
|---|---|---|---|
| G2 | 1833 | 1855 | **−1.2%** |
| G1 | 4182 | 4264 | **−1.9%** |
| G3 | 3447 | 6351 | −45.7% (diverges) |

**At low and mid Q the analytic model is correct to ~2%**, and the 4–6%
originally reported as model error was numerical error in the validation.

**At high Q (G3) the discrepancy is real.** Refinement moves the simulation
*away* from the model, and that is now confirmed not to be sampling or
extraction.

### Candidate explanation for G3 — HYPOTHESIS, NOT ESTABLISHED

The bend-loss model is `bend = 4·exp(b₁ + b₀·R)` — **a function of radius
only, with no wavelength dependence** (this is a fact about the code).
G3 resonates at 1555 nm; G1 and G2 at 1542–1543 nm. Bend radiation at
R = 5.2 µm rises steeply toward longer wavelengths, so a wavelength-blind
loss model would over-predict Q specifically for the reddest resonance.

It would also explain the mesh reversal: for a genuinely radiative mode a
finer mesh resolves real radiation and *lowers* Q, whereas for G1 and G2 a
finer mesh mainly removes spurious numerical loss and *raises* it.

**Testing this needs a wavelength-resolved bend-loss study — future work.**

### IL does not validate

Drop-port transmission error at 32 nm: G1 −12.5%, G2 +1.5%, G3 −17.9%. No
consistent sign, robust to sampling. G2's IL of 0.084 dB (T = 0.981) is within
2% of unity, where the absolute uncertainty of extracting IL from an FDTD
spectrum is comparable to the value itself — **G2 cannot validate IL at all**.

### FSR — the second consistent systematic

+2.2% to +2.8% across all nine runs. FSR = λ²/(n_g·L) with L = 2πR + 2L_c
exact, so the real ring's group index is ~2.5% below the model's. n_g comes
from a **straight-waveguide** mode sweep and a 5.2 µm bend is not straight.
The fix is a bent-waveguide mode solve, which would change every dataset
label — future work.

### Stated limitations for the paper

1. Validation is at small radius only (R = 5.20 µm). A 10 µm ring needs
   ~20 GB of solver memory and a 14 µm ring ~36 GB, against 15.7 GB of RAM.
2. The validation is **not mesh-converged** and cannot be on this hardware;
   a third mesh size for these geometries exceeds available memory.
3. Richardson extrapolation assumes second-order convergence, which two mesh
   sizes cannot verify.
4. Q_L accuracy is demonstrated to ~2% for Q ≲ 4300 and is **not established**
   above it.
5. IL is not validated.

### What this does and does not affect

**Unaffected** — every model was trained and scored on identical labels, so a
shared label bias cancels from every model-to-model comparison:
O6, the θ-vs-RMS-RGE result, buildability, O4 and O5.

**Must be qualified:** absolute Q_L above Q ≈ 4300; absolute IL anywhere;
O7 scenario verdicts, which were judged against Q_L specifications.

### λ_res, independently relevant to O7

The analytic model's own λ_res error against full FDTD is **0.8–1.1 nm**,
about half the Scenario A channel tolerance, before any inverse model is
involved. This is a **fourth** independent line of support for the
trimming conclusion in §8.

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

## 10b. Phase B1 — bent-mode FDE sweep (2026-10-13)

210 FDE solves: straight control + 6 radii (5-15 um) x 6 widths (400-500 nm)
x 5 wavelengths (1500-1600 nm). 10 nm mesh, PML on all four boundaries,
`bentmode.csv`. Purpose: replace straight-guide n_g and the
wavelength-independent bend-loss model with solver-measured values.

### Result 1 — the stated cause of the FSR bias is REFUTED

n_g(bent) vs n_g(straight) at 1550 nm differs by **at most 0.08%**, across
every radius and width. The ring validation (§9) measured FSR biased
**+2.2 to +2.8%** and §9 attributed it to straight-vs-bent n_g at ~2.5%.
That attribution is wrong by a factor of ~30. n_eff shows the same: R=5,
w=450, 1600 nm gives 2.29001 against 2.28881 straight (+0.05%).

**Any text in the draft attributing the FSR bias to bend effects must be
removed.** The measurement that was supposed to confirm it refutes it.

### Result 2 — bend loss is not measurable this way

Every entry of the bend-loss table is **negative** (i.e. gain, unphysical),
with magnitudes from 1e-5 down to 1e-12 dB/cm -- the solver's numerical
floor. Phase B2 as designed cannot proceed on this data.

### Result 3 — the n_g polynomial carries a systematic -0.58% offset

Phase-1 polynomial (`model_fits.json: neff`) against direct FDE at 1550 nm:

    w (nm)    poly n_g   FDE n_g    error
       400      4.4100    4.4354   -0.57%
       425      4.3609    4.3876   -0.61%
       440      4.3310    4.3560   -0.57%
       450      4.3112    4.3361   -0.57%
       475      4.2625    4.2864   -0.56%
       500      4.2163    4.2407   -0.58%

Flat across all widths -- systematic, not scatter. FSR ~ 1/n_g, so this
makes predicted FSR 0.58% high. It accounts for ~0.6 pp of the 2.2-2.8 pp
FSR bias. **~1.9 pp remains unexplained and is not bend-related.**

Round-trip length is NOT the error: `build_mrr` draws two straights of
length L_c plus two 180-deg bends, so L = 2*pi*R + 2*L_c as Eq. 1 assumes.

### Result 4 — solver reliability (disclose in limitations)

The FDE solver returns, non-deterministically, either a correct solve or a
degenerate one (all modes pinned at the bulk index, n_g undefined) from an
identical call sequence -- verified by two byte-identical state dumps giving
opposite outcomes. The script detects this and rebuilds the solver until a
guided mode appears, escalating the window (up to 3x) and trial-mode count.
Most points needed 1-3 attempts; the hardest (R=6, w=400, 1600 nm) needed 8.

Consequent caveat: **run-to-run and window-to-window scatter in n_g reaches
0.7%** (w=500: +0.23% at R=7.5, +0.70% at R=10, +0.01% at R=12.5, +0.77% at
R=15 -- non-monotonic in R, which is unphysical). That scatter is ~10x the
bent-vs-straight effect being measured, so the small differences in the n_g
table are at or below the noise floor. The -0.58% polynomial offset is well
above it and stands.

### Result 5 — the residual, measured properly (2026-10-13)

Comparing FSR percentages is a trap: FSR varies ~4% across 1500-1600 nm, so
a model FSR evaluated at lambda_res and an FDTD spacing whose pair is centred
8 nm away are different quantities. Two of my own numbers (-1.9 pp, then
-2.06%) were contaminated this way. The robust statement is in group index,
compared at matched wavelengths:

    n_g from each adjacent resonance pair, n_g = l1*l2 / (L*(l2-l1)),
    against FDE n_g interpolated at that pair's midpoint:

      geom  lam mid  ring n_g   FDE n_g     diff
        G1   1533.8    4.1863    4.3018   -2.69%
        G1   1550.2    4.1915    4.3249   -3.08%
        G1   1566.9    4.1956    4.3334   -3.18%
        G2   1520.2    4.1819    4.2847   -2.40%
        G2   1535.4    4.1868    4.3041   -2.72%
        G2   1550.9    4.1923    4.3253   -3.07%
        G2   1566.7    4.1969    4.3333   -3.15%
        G3   1529.9    4.1837    4.2962   -2.62%
        G3   1546.6    4.1894    4.3199   -3.02%
        G3   1563.6    4.1941    4.3317   -3.18%

    mean -2.91%, sd 0.26%, n = 10

**The ring's group index is 2.91% below the mode solver's.** The deficit
grows mildly with wavelength (-2.4% at 1520 nm to -3.2% at 1567 nm).

RULED OUT, each by measurement:
- Round-trip length. The straight fraction 2*Lc/L is 3.5% (G3), 5.8% (G1)
  and 10.9% (G2) -- a factor of three -- and the deficit is the same for all
  three. A length error or a straight/bend mixing error would scale with it.
- Bend effects: 0.08% (Result 1).
- FSR extraction: independent re-extraction agrees with analyse_drop to
  0.03%, 0.005% and 0.014% on the three geometries.
- The n_g polynomial: 0.26% at these geometries (Result 3 revised; the
  -0.58% figure was computed at 1550 nm against the straight guide and does
  not apply at the validated lambda_res).
- Mesh, tentatively: refining 40 -> 32 nm moves FSR AWAY from the model
  (+0.2 to +0.3%), the opposite of a convergence artefact. Rests on the
  unlabelled runs being the 32 nm production mesh; worth confirming.

REMAINING CANDIDATES: FDTD numerical dispersion at this mesh (despite the
direction above), or a genuine limitation of the coupled-mode assembly.
Distinguishing them needs a ring run at a substantially finer mesh, which is
expensive and was not attempted.

### Result 6 — ng_solver is noisy; do NOT regenerate labels with it

Finite-differencing n_eff(lambda) from bentmode.csv and comparing against the
ng_solver column over all 210 rows: mean -0.19%, sd 0.54%, range -2.45% to
+3.38%. The two are broadly consistent, so the n_eff -> n_g chain is sound,
but ng_solver is NOT smooth in wavelength. For the straight 450 nm guide the
successive steps are +0.0085, +0.0878, +0.0031, -0.0250 -- one step an order
of magnitude larger than its neighbours, a ~2% discontinuity between 1525 and
1550 nm that no physical dispersion produces.

**Phase B3 (regenerate labels with FDE n_g) should NOT proceed.** The
correction it would apply is 0.26%; the noise it would inject is ~0.5% with
2.4% excursions. It makes the labels worse.


### What this does and does not change

- Does: supplies direct FDE n_g, correcting a systematic -0.58%.
- Does: removes a wrong physical explanation from the paper.
- Does not: close the FSR gap. The ring's n_g sits 2.91% below the solver's
  and nothing tested explains it (Result 5).
- Does not: enable B2. Bend loss needs a different method entirely.

---

## 10c. Phase C — thermal trimming budget (2026-10-13)

### C1: tuning sensitivity, solved for

Each material index perturbed in turn, mode re-solved, Gamma = dn_eff/dn_i
read off directly. Perturbing both together reproduces the sum of the
separate perturbations to +0.11% to +0.17% across all six widths, so the
response is linear. dn/dT for Si and SiO2 are literature material constants,
the same class of input as the Palik index data.

    w (nm)  G_core  G_clad  G_both  add err   n_g     dlam/dT (nm/K)
       400  1.0549  0.4032  1.4557   +0.17%  4.4354   0.0700
       425  1.0649  0.3453  1.4081   +0.16%  4.3876   0.0712
       440  1.0674  0.3174  1.3828   +0.15%  4.3560   0.0718
       450  1.0687  0.3005  1.3672   +0.14%  4.3361   0.0721
       475  1.0683  0.2671  1.3338   +0.13%  4.2864   0.0728
       500  1.0665  0.2403  1.3053   +0.11%  4.2407   0.0734

dlambda/dT = 0.0719 nm/K mean. Published Si microring tuning is 0.07-0.08
nm/K and nothing here was fitted to it -- an independent cross-check, and
the only quantity in this project that agreed with expectation first time.

### C2/C3: what the trimming costs

Two physical constraints, both of which I got wrong before getting right:
- a heater shifts RED only (no cooling), so the signed error cannot simply
  be divided by dlambda/dT;
- the resonance comb repeats every FSR, so a ring is tuned onto the nearest
  resonance above it and the requirement is CAPPED at one FSR (~227 K here).

With a design bias delta, ring i needs (delta - e_i) mod FSR_i. Minimising
the worst case over delta is a covering problem on a circle, solved by
direct search.

    method      n  RMS err  spread  bias:max dT  max mW  unbias mean  unbias max
    naive      20   4.332n  18.940n      125.0K   41.66        80.1K      227.5K
    rf_raw     20   4.015n  16.433n      121.4K   40.48        70.1K      190.4K
    xgb_norm   20   3.346n  12.371n      131.9K   43.96        56.7K      204.9K
    tandem      8   3.638n   9.082n       76.0K   25.33        58.0K      104.1K

tandem is EXCLUDED from the ranking: it realises 8 of 20 predictions, so
scoring it on the subset it managed is selection after the fact.

Best fully-evaluated model: rf_raw. Heater spans 121.4 K (40.5 mW at
3 K/mW, a literature device parameter that scales power only); mean draw
18.2 mW. Scenario A (4 ch) 72.8 mW static, Scenario B (8 ch) 145.7 mW.

**Trimming is affordable.** The paper's negative result on resonance
placement prices the approach rather than invalidating it.

### The result worth emphasising

Under a biased design the figure of merit is how tightly the lambda_res
errors CLUSTER once wrapped into one FSR -- NOT their RMS. xgb_norm has the
lowest RMS of the three fully-evaluated models and the HIGHEST trimming
demand; rf_raw is worse on RMS and cheaper to trim. A systematic offset is
absorbed by the design bias for free; scatter is paid for in heater range
and static power. For any system that will be trimmed -- i.e. every deployed
microring link -- consistency beats accuracy, and inverse models should be
selected on spread.

---

## 11. Still open

- [x] **Full 3-D ring validation** (§9) — COMPLETE, 3 geometries, 41 h.
      Found a real defect: Q_L accuracy degrades with Q_L. See §9.
- [ ] **Decide how to position the Q_L finding** before submission. It does
      not touch the central contributions but it does qualify every absolute
      Q_L/IL claim and the O7 verdicts.
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
- [x] **Attribute the FSR bias** (§10b Result 5) -- DONE. It is a 2.91%
      group-index deficit, not a length, bend, extraction or fit error.
- [ ] **Decide whether to chase the 2.91% further.** Distinguishing FDTD
      numerical dispersion from a coupled-mode limitation needs one ring run
      at a substantially finer mesh (hours to days). Optional; the residual
      is already quantified and bounded.
- [ ] Confirm the unlabelled ring runs used the 32 nm production mesh, which
      the mesh-direction argument in Result 5 assumes.
- [ ] **Decide the fate of Phase B2** (wavelength-resolved bend loss). The FDE
      route gives numerical noise; a different method is required, or the
      wavelength-independent loss model stays and is disclosed as a limitation.
- [ ] Phase C assumes one heater per ring and a 3 K/mW thermal resistance.
      Crosstalk between adjacent heaters is not modelled and would raise the
      budget; a shared-heater or bias-only scheme would lower it.
- [ ] Quantify the FDE repeatability directly: solve one point N times and
      report the n_g spread (§10b Result 4 estimates 0.7% from the sweep).

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
--bentmode      bent-mode FDE sweep        --modediag    FDE config ladder
--fsrcheck      FSR attribution            --matcheck    material index models
--thermal       tuning sensitivity         --trimbudget  trimming power budget
--validate      O1 literature validation   --ringvalidate full 3-D ring check
--ringvalidate --dry   price candidates without simulating (--try=R,w,g,Lc)
```

Flags: `--n=`, `--nbank=`, `--no-movie`, method flags for `--closedloop`
(one only — the parser takes the last).
