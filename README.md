# Microring Resonator Dataset - Phase 1

Dataset and simulation records for the BSc thesis *ML-Driven Inverse Design of
Microring Resonators for Co-Packaged Optics / High-Bandwidth Optical
Interconnects*. This repository is the Phase 1 deliverable (Phase 1.5 step 8).

**Device:** add-drop racetrack microring, 220 nm SOI, silicon core, SiO2
cladding, fundamental TE mode, 1500-1600 nm.
**Solver:** Ansys Lumerical FDTD 2024 R1, driven from Python via `lumapi`.

## The dataset

`data/mrr_dataset.csv` - 1,000 samples, Latin Hypercube over the four design
parameters (seed 42).

| Input | Range | Output | Unit |
|---|---|---|---|
| `R_um` ring radius | 5-15 um | `lambda_res_nm` resonance nearest 1550 nm | nm |
| `w_nm` waveguide width | 400-500 nm | `FSR_nm` free spectral range | nm |
| `g_nm` coupling gap | 150-350 nm | `FWHM_nm` linewidth | nm |
| `Lc_um` coupling length | 0-3 um | `Q_L` loaded quality factor | - |
| | | `IL_dB` drop-port insertion loss | dB |
| | | `n_eff`, `n_g` at the resonance | - |

Diagnostic columns: `kappa2`, `t`, `a`, `loss_bend_dB`, `loss_rough_dB`,
`m_order`, `n_res_in_window`, `origin`, `flag`.

## How it was generated - read this first

Full-ring 3-D FDTD was **not affordable**: one sample projected to ~35 h
against a 25 min/sample limit (see `records/substudy1_timing_verdict.txt`).
The dataset therefore uses the **dual-fidelity** strategy:

1. **3-D FDTD of the coupling region only** (48 geometries). The coupler is
   single-pass, so its cost does not scale with Q.
2. **Eigenmode sweep** for n_eff and n_g versus width and wavelength (30 solves).
3. **3-D FDTD of a 90-degree bend** for bend-radiation loss versus radius (5 radii).
4. **Coupled-mode theory** (thesis Eqs. 2-7) turns those inputs into the full
   ring response, evaluated analytically for each sample.

Every row carries an `origin` label saying so.

### Limitations

- **The dataset contains the information of 83 simulations, not 1,000.** Each
  row is an analytic evaluation of models fitted to those runs.
- **Sidewall roughness is not simulated.** A literature value of 1.3 dB/cm
  (measured, Paper 11) is added to the simulated bend radiation.
- **Bend loss was measured at w = 450 nm only**, and fits with +/-20-40%
  residuals. This matters mainly for the ~27 samples with Q_L > 10^5.
- **`lambda_res_nm` is a sawtooth by definition**: "nearest 1550 nm" wraps by
  one FSR as geometry changes. Use `m_order` to unwrap it if needed.
- **`FSR_nm`** is the spacing to the nearest adjacent resonance; with dispersion
  it can differ slightly from the spacing across 1550 nm.

## Validation

| Check | Result | Record |
|---|---|---|
| Literature, FSR vs a fabricated device | 0.02-0.33% (target <= 10%) | `substudy3_literature_validation.txt` |
| Method, analytic vs full-ring FDTD | Q_L 1.60%, T_peak 0.41% | `method_validation.txt` |
| Coupler model, 8-fold cross-validation | 3.37% MAPE, R^2 0.996 | `results/kappa2_parity.png` |
| Coupling range | Q_L 10^2.88-10^5.27, 96.8% in target band | `substudy4_coupling_boundary.txt` |

## Reproducing the dataset

Requires only the raw result files in `results/` - no Lumerical licence:

    python mrr_template.py --generate

This refits every model from the raw FDTD CSVs and rewrites `mrr_dataset.csv`.

To re-run the simulations themselves (requires Lumerical FDTD):

    python mrr_template.py --couplerset   # 48 coupler runs, ~12.5 h
    python mrr_template.py --modesweep    # n_eff, n_g + literature check
    python mrr_template.py --bendloss     # bend loss vs R

## Repository layout

    mrr_template.py              all simulation and generation code
    data/                        the dataset and its quality log
    results/                     raw FDTD outputs and fitted models
    records/                     sub-study reports and solver evidence logs
