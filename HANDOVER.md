# HANDOVER — ML-Driven Inverse Design of Microring Resonators
### Prepared 2026-10-14. Read this first; it indexes everything else.

**Author** Md Shidul Islam Shihab, Dept. of Electronics & Communication
Engineering, KUET. mdshidulislamshihab@gmail.com
**Repository** github.com/SHIHAB-21-12/mrr-inverse-design-dataset
**Working folder** `D:\Abroad\full guide\Phase1`
**Paper** draft v9, 20 pages, `paper/paper.tex`
**Script** `mrr_template.py`, 339,346 bytes, ONE file, ~7,350 lines

---

## 0. STATUS IN ONE PARAGRAPH

The framework is complete and the simulation campaign is finished. A draft
paper exists with every objective measured and reported, three of seven not
met and reported as such. The work is **not** submission-ready: it has had no
independent expert review of its physics, and it carries one quantified,
unexplained discrepancy (§3 below). My assessment is that it is **not Q1 as
it stands** — no fabricated device, one platform, one polarisation — and that
**Q2 is realistic** once the expert review is done. That assessment is mine
and should be checked against a photonics academic's.

---

## 1. STANDING CONSTRAINTS — these are locked, do not reopen

These were fixed by the author early and must be honoured by any future
session.

1. **`Thesis_Master_Reference_v1_3.pdf` is settled ground truth.** Do not
   audit it, do not suggest changes to its parameter ranges, equations,
   thresholds or objectives, do not propose restructuring it. Treat every
   fixed default as settled for the whole project. The ONLY exception: if a
   specific simulation result contradicts a stated assumption, apply the
   decision rule or fallback the document itself already specifies (Section 7
   and Phase 1.6 contain these) rather than inventing a new one.
2. **Working output, not planning documents.** Actual runnable Python, help
   setting it up, debugging real errors, interpreting results against the
   numeric thresholds already fixed in the document — never against an
   assistant's own judgement of what looks reasonable.
3. **ONE script only.** Everything lives in `mrr_template.py` as a new
   `--mode`. Never create a second script.
4. **One step at a time.** Confirm each step works before moving to the next.
5. **Step-by-step numbered instructions, always.**
6. **PowerShell, not CMD.** `&&` is invalid. `del a b` and `rmdir /s /q` are
   CMD syntax and fail. Use `Remove-Item`.
7. **Commits carry no assistant attribution.** The author's commits only.
8. **The movie monitor stays in the script. Interrupted runs must resume.**

---

## 2. WHAT THE WORK IS

An add-drop racetrack microring on 220 nm SOI, Si/SiO2 (Palik), fundamental
TE, 1500–1600 nm. Inputs R (5–15 um), w (400–500 nm), g (150–350 nm),
L_c (0–3 um). Targets lambda_res, FSR, Q_L, IL.

**The key methodological decision — the dual-fidelity construction.** Only the
*coupler* is simulated in 3-D FDTD; the ring response is assembled from it by
coupled-mode theory. This is what makes 1,150 labelled geometries affordable.
Round-trip length L = 2*pi*R + 2*L_c. The coupling angle
theta = kappa_0(w) * exp(-gamma(w) g) * (L_c + B*sqrt(2*pi*R/gamma)), so g and
L_c enter the physics ONLY through theta. That single fact drives the central
result.

Everything about the construction, its validation, and all numeric results is
in **`MASTER_FINDINGS.md`** (789 lines, sections 1–12). Do not restate it;
read it.

---

## 3. FINDINGS — the short list

Full detail and all numbers: `MASTER_FINDINGS.md`.

### The central contribution (O6)
Four targets impose only **two effective constraints** (one on L, one on
theta) on four unknowns, leaving a 2-D degenerate family of geometries with
indistinguishable responses. Geometry recovery therefore fails (only R
reaches R^2 >= 0.80; w, g, L_c sit at 0.13–0.69) while response recovery
succeeds (closed-loop 3.669%). Both prior literatures are correct; they were
measuring different things. Direct confirmation: four geometries with gaps
spanning 210–270 nm and coupling lengths spanning 0.209–2.353 um share a
loaded Q of 14,659.8 to the last digit.

### Objective outcomes
O1 met · **O2 not met** · **O3 not met** · O4 met · O5 met · **O6 resolved**
· **O7 not met** (0 of 12, FDTD-verified).

### Validation nobody in this literature usually does
Nine full 3-D FDTD ring simulations, ~120 h, validating the CMT assembly.
lambda_res agrees to 0.10%. This is what exposed the next item.

### The open discrepancy — THE thing to bring to an expert
**The ring's group index is 2.91% +/- 0.26% below the mode solver's**,
measured at ten matched wavelength points across three geometries. Excluded by
measurement: round-trip length (the deficit is identical across straight
fractions of 3.5%, 5.8% and 10.9% — a length error would scale with it), bend
effects (0.08%), FSR extraction (agrees to 0.03%), the n_g polynomial (0.26%).
Mesh is tentatively excluded: refining 40 -> 32 nm moves FSR *away* from the
model. Remaining candidates: FDTD numerical dispersion, or a limitation of the
coupled-mode assembly itself.

### Phase C — the application argument
Thermo-optic tuning solved for, not assumed: dlambda/dT = 0.0719 nm/K
(0.0700–0.0734 across widths), independently consistent with the published
0.07–0.08 nm/K. Heat-only physics with an optimal design bias, solved as a
circular covering problem: the best fully-evaluated model needs a heater
spanning **121 K**, drawing **18.2 mW** typical, i.e. **73 mW** static for a
4-channel link and **146 mW** for 8.

### The claim most likely to be NEW
Under a biased design the figure of merit is how tightly the lambda_res errors
**cluster once wrapped into one FSR — not their RMS**. XGBoost has the lowest
RMS error of the three fully-evaluated models and the HIGHEST trimming cost;
the random forest is worse on RMS and cheaper to trim. Every deployed
microring link is trimmed, so inverse models should be selected on spread.
I am not aware of this being stated in the inverse-design literature.

### Robustness work completed
- **Label fidelity**: O6's verdict is unchanged under a +/-3% FSR label bias
  (trained on shifted labels, tested on true targets). Caveat: a uniform
  scaling is the simplest form the error could take; the real discrepancy is
  geometry-dependent.
- **Five-seed sample efficiency**: ONE published claim was retracted —
  "XGBoost forward wins below 300 samples" is within seed noise (gap 1.49,
  pooled sd 3.09). Three other claims survive. The tandem inverse is
  **unusable**, not merely worse: 29–118% RMS-RGE, non-monotonic, sd near a
  third of the mean. O3 now has an error bar: 15.500 +/- 0.069% against the
  15% threshold — missed by seven sd, reproducibly.

---

## 4. KNOWN DEFECTS AND CAVEATS — carry these forward

1. **`_fit_sized(..., seed=NN_SEED)` ignores its seed argument.** It never
   reaches the estimator. Any other caller relying on it to vary has been
   comparing identical fits. NOT yet audited.
2. **The FDE solver is non-deterministic.** Identical call sequences produce
   either a correct solve or a degenerate one (all modes at the bulk index,
   n_g undefined). Two byte-identical state dumps gave opposite outcomes. The
   script detects this and rebuilds the solver until a guided mode appears,
   escalating window size and trial-mode count. Root cause NOT understood.
3. **Never set `use max index`/`n` or `calculate group index` on the FDE.**
   Each, alone, pins every mode at a seed index with n_g undefined. Both are
   already the solver's defaults. Measured 2026-10-13.
4. **`ng_solver` in `bentmode.csv` is noisy** (sd 0.54%, excursions to 2.4%)
   and NOT smooth in wavelength. Do not regenerate dataset labels from it:
   the correction is 0.26%, the injected noise larger.
5. **Phase C assumes one heater per ring, no thermal crosstalk.** In a real
   8-channel array this is optimistic. First thing a referee will ask.
6. **Bend loss cannot be obtained from the FDE eigensolve** at these radii —
   the imaginary part is negative at every grid point and at the numerical
   floor. Phase B2 is dead by this route.
7. **Five seeds is a small sample for a standard deviation.** Only clearly
   separated and clearly unseparated comparisons are treated as results.
8. **Q_L validated only below ~4300. IL not validated at all.**

---

## 5. THE ASSISTANT'S ERROR LOG — read this before trusting any analysis

Over the sessions that produced this work the assistant was wrong more than a
dozen times. Every correction came from the author's data or from an internal
inconsistency in the output, never from the assistant's own reasoning. The
recurring failure modes, so a future session can avoid them:

- **Building tests that measure nothing, then reading the flat result as
  robustness.** Happened twice: a theta-ordering test that was vacuous at
  n=2, and a label-sensitivity test that scaled train and test together when
  tree splits and standardisation are both invariant to that. Signature: a
  suspiciously clean result accepted before checking whether the quantity
  could vary at all.
- **Guessing at solver API behaviour instead of measuring it.** Roughly seven
  consecutive wrong inferences about the Lumerical FDE. Resolved only by
  dumping solver state and comparing.
- **Quoting a number computed in the wrong place.** The FSR residual was
  stated as 1.9 pp, then 2.06%, before being measured properly at 2.91% —
  the earlier figures compared quantities at mismatched wavelengths.
- **Getting physics wrong in the direction that flatters the result.** The
  trimming budget initially assumed cooling was available, then ignored the
  FSR wrap. Both were caught by impossible numbers in the output.

**The measurements in this repository are sound and were checked repeatedly
against the solver. The interpretations are the assistant's and deserve a
second opinion.**

---

## 6. SCRIPT MODES

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
--bentmode      bent-mode FDE sweep        --modediag    FDE config ladder
--fsrcheck      FSR attribution            --matcheck    material index models
--thermal       tuning sensitivity         --trimbudget  trimming power budget
--labelsens     O6 label-fidelity test     --seedcurve   5-seed sample efficiency
--seedfig       error-bar figure
```
Flags: `--n=`, `--nbank=`, `--no-movie`, `--dry`, `--try=`, `--margins=`,
`--mesh=`, `--points=`. `--closedloop` takes ONE method flag (the parser uses
the last one given).

**`--matcheck` is BROKEN** — its accessor returns a constant index with zero
dispersion, which is not physical. Its output is invalid. Either fix or delete.

---

## 7. WHAT REMAINS

**Priority 1 — free, and the largest remaining risk**
Independent expert review of the physical methods. Two specific questions:
(a) why would a 3-D FDTD ring show a group index 2.91% below the mode
solver's, when length, bend, extraction and fit are all excluded?
(b) does the trimming argument hold — heat-only tuning, bias by circular
covering, and selection on spread rather than RMS?

**Priority 2 — minutes to an hour**
- Audit other callers of `_fit_sized` for the ignored `seed` argument
- LICENSE file
- Check ref [12] (Torabi et al.) for a peer-reviewed version — currently an
  unrefereed preprint
- Confirm ref [13] (Bogaerts et al.) Eq. 6 — typeset version never read
- Replace the data-availability placeholder with a real URL/DOI

**Priority 3 — optional, expensive**
- One ring run at a substantially finer mesh, to separate FDTD numerical
  dispersion from a CMT limitation (the 2.91%)
- Seed repeats for the ablation study
- Label sensitivity against a geometry-dependent error (needs the dataset
  regenerated from full 3-D simulation — the expensive path the dual-fidelity
  construction exists to avoid)
- Fix or remove `--matcheck`

---

## 8. FILE INVENTORY

### Upload these to a new chat (in this order)
| File | Why |
|---|---|
| `HANDOVER.md` | this file — the index |
| `MASTER_FINDINGS.md` | every finding and number, 789 lines |
| `Thesis_Master_Reference_v1_3.pdf` | the locked methodology |
| `paper/paper.tex` | current draft v9 |
| `mrr_template.py` | the single script, 339,346 bytes |

That is enough to resume. Upload the rest only when a task needs it.

### On demand
`results/*.txt` — every report in human-readable form:
`bentmode_report.txt`, `fsr_attribution.txt`, `thermal_c1.txt`,
`trim_budget.txt`, `label_sensitivity.txt`, `seedcurve.txt`,
`phase3..phase7_report.txt`, `floor.txt`, `o1_validation.txt`,
`kappa2_cv.txt`, `ablation_report.txt`, `density_report.txt`,
`physinv_report.txt`, `theta_vs_closedloop.txt`
`validation/ringvalidate.json` + `.txt` — the nine 3-D ring runs
`results/*.json` — machine-readable results
`results/bentmode.csv` — 210 FDE solves
`data/` — train/val/test splits + `scaler.json`
`mrr_dataset.csv` — the 1,150-geometry dataset

### Keep but never upload (large, regenerable)
`results/*.pt` (model checkpoints), `results/*.png|pdf` (figures),
`results/ringval_*_spectrum.npz` (raw spectra — irreplaceable, back up
separately), `archive/`, `records/`

---

## 9. CLEANUP — safe deletions

Run in PowerShell from `D:\Abroad\full guide\Phase1`:

```
Remove-Item results\modeprobe.lms -Force -ErrorAction SilentlyContinue
Remove-Item closedloop_scratch.fsp -Force -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Remove-Item *_done.fsp -Force -ErrorAction SilentlyContinue
```

`modeprobe.lms` is a debug artefact from the FDE investigation and is
regenerated on demand. `closedloop_scratch.fsp` and `*_done.fsp` are solver
scratch files.

**Check before deleting — do NOT delete blind:**
- `template_R10.000um_w450.0nm_g200.0nm_*.fsp` (837 KB, repo root) — a
  leftover Lumerical layout. Regenerable by `build_mrr`. Confirm it is not
  referenced anywhere, then delete.
- `archive/` and `records/` — contents unknown to this session. Inspect
  before removing anything.
- `OPTIMIZATION_IDEAS.md`, `RESULTS.md` (root) — superseded by
  `MASTER_FINDINGS.md` and the paper, but check for anything unique first.

**NEVER delete:** `results/ringval_*_spectrum.npz` (120 h of simulation),
`results/bentmode.csv`, `data/`, `mrr_dataset.csv`, `results/model_fits.json`,
`validation/`.

Everything is committed and pushed, so anything deleted in error is
recoverable with `git checkout`.
