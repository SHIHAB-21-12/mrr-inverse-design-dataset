"""
mrr_template.py  --  Phase 1.5, step 1

Parameterized Lumerical FDTD (3D) template for the add-drop racetrack microring
resonator defined in Thesis_Master_Reference_v1.3, Sections 4.1.1-4.1.4.

R, w, g, Lc are exposed as script variables (Phase 1.5 step 1 requirement).
Mesh settings are exposed separately because Phase 1.6 sub-study 2 fixes them.

All internal units are SI (metres), as required by lumapi.

Usage:
    py -3 mrr_template.py              # 15 objects, movie monitor included
    py -3 mrr_template.py --no-movie   # 14 objects, no movie monitor
    py -3 mrr_template.py --quick      # movie + short sim time, fast preview
    py -3 mrr_template.py --substudy2       # sub-study 2, 3-D (Section 4.1.4)
    py -3 mrr_template.py --substudy2 --2d  # sub-study 2, 2-D (Section 7)
    py -3 mrr_template.py --calibrate --2d  # measure the true linewidth first
    py -3 mrr_template.py --substudy2c      # sub-study 2 on the COUPLER (2-D)
    py -3 mrr_template.py --substudy2c --3d # sub-study 2 on the COUPLER (3-D)
    py -3 mrr_template.py --bendloss        # bend-radiation loss vs R (3-D)
    py -3 mrr_template.py --couplerset      # LHS coupler training set (3-D)
    py -3 mrr_template.py --modesweep       # n_eff, n_g vs w  +  sub-study 3
    py -3 mrr_template.py --generate        # build the dataset from the raw CSVs
    py -3 mrr_template.py --phase2          # Phase 2 preprocessing and split
    py -3 mrr_template.py --phase3          # Phase 3 forward model (resumable)
    py -3 mrr_template.py --phase4          # Phase 4 naive inverse + tandem
    py -3 mrr_template.py --phase5          # Phase 5 RF / XGBoost (resumable)
    py -3 mrr_template.py --phase6          # Phase 6 master tables + curves
    py -3 mrr_template.py --phase7          # Phase 7 CPO scenario validation
    py -3 mrr_template.py --phase7sim       # re-simulate the candidates (Lumerical)
    py -3 mrr_template.py --ablate          # ablation of the free optimizations
    py -3 mrr_template.py --physinv         # physics-decoded inverse model
    py -3 mrr_template.py --density         # training-density diagnostic
    py -3 mrr_template.py --closedloop      # closed-loop re-verification (Lumerical)
         optional:  --tandem  --rf_norm  --xgb_norm   --n=20

The movie monitor (movie_xy) is ON by default. It records field data at every
time step over a large plane, which inflates runtime and memory, so it must
be absent for Phase 1.6 sub-study 2 (mesh convergence), sub-study 1 (timing
calibration), and Phase 1.5 step 5 (mass production) -- if one is present
during timing calibration you measure the wrong seconds per sample and
mis-scope the whole dataset against the 15 min/sample rule.

The sub-study and batch scripts call build_mrr() with movie=False explicitly,
so that default only affects manual runs of this file. For a manual run whose
timing you intend to trust, pass --no-movie.
"""

# ---------------------------------------------------------------------------
# Scope: Phase 1.5 step 1 only (the parameterized template). The Phase 1.6
# sub-studies, LHS sampling, batch execution, post-processing and dataset
# release are separate scripts.
#
# Verified state (R=10um, w=450nm, g=200nm, Lc=1.5um), all 14 production
# objects checked field-by-field in the Lumerical GUI:
#   L = 65.8319 um      FSR = 8.689 nm       ~11.5 resonances in window
#   both coupling gaps  = 0.200000 um exactly
#   dt = 0.0436217 fs   (matches the Courant limit for the 20 nm gap mesh)
#   source mode         = fundamental TE, n_eff = 2.351764, 98% TE fraction
# ---------------------------------------------------------------------------

from __future__ import annotations

import csv
import glob
import importlib.util
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# FIXED DEVICE CONSTANTS  --  Section 4.1.1. Do not edit.
# ---------------------------------------------------------------------------

H_CORE = 220e-9          # SOI device-layer thickness, fixed (not swept)
LAMBDA_MIN = 1500e-9     # scan window lower bound
LAMBDA_MAX = 1600e-9     # scan window upper bound
LAMBDA_TARGET = 1550e-9  # nominal target wavelength
# Section 4.1.4 item 3 sets a MINIMUM of 2,000 and requires verification
# against the narrowest expected linewidth in Phase 1.6. That check failed at
# 2,001 points (0.050 nm grid vs a ~0.034 nm expected FWHM), so the grid is
# raised to 0.005 nm. Costs DFT memory, not simulation time.
N_FREQ_POINTS = 20001

# --- Phase 1.6 sub-study 2: mesh convergence -------------------------------
# Sweep variable is the scalar mesh override d (dx=dy=dz=d), with global mesh
# accuracy held at 3 -- the direct analogue of Paper 10 Section 2.3. Coarse to
# fine, so useful data arrives early if the sweep is interrupted.
MESH_SWEEP_NM = [32.0, 24.0, 20.0, 16.0, 12.0, 8.0]
CONVERGENCE_GRADIENT = 1e-4          # Phase 1.6 sub-study 2 pass criterion
CALIBRATE_SIM_TIME = 600e-12         # --calibrate: long window to measure
                                     # the true linewidth (auto-shutoff ends it)

# --- coupler-only sub-study (dual-fidelity path, Section 7) ----------------
# The coupler is NOT a resonator: light transits once, so there is no ringdown
# to wait out. A few ps is ample and a coarse spectral grid is fine because
# kappa^2(lambda) is smooth.
COUPLER_SIM_TIME = 5e-12
COUPLER_FREQ_POINTS = 201
SUBSTUDY2C_CSV = "substudy2_coupler_convergence_{dim}.csv"

# --- bend-radiation loss vs radius -----------------------------------------
# A 90-degree bend is single-pass, so this is cheap. R must be swept because
# bend radiation varies strongly across the Section 4.1.2 range 5-15 um.
# NOTE: Palik Si has ~zero absorption at 1550 nm and FDTD does not model
# sidewall roughness, so this measures BEND RADIATION ONLY. Add a literature
# roughness term (Paper 11: 1.3 dB/cm measured) for a realistic total.
BEND_R_SWEEP_UM = [5.0, 7.5, 10.0, 12.5, 15.0]
BEND_MESH_NM = 32.0
BEND_STUB = 3.0e-6
BENDLOSS_CSV = "bendloss_{dim}.csv"

# --- coupler training set (the ONLY bulk FDTD cost of the dataset) ---------
# LHS over all four Section 4.1.2 parameters, seed 42 as fixed in Phase 1.5
# step 4. kappa^2 depends on all four: g and Lc directly, w through mode
# confinement, R through the curved approach either side of the straight.
# Once this is done every dataset sample is an analytic evaluation of
# Eqs. 2-6 and costs milliseconds.
COUPLERSET_N = 48
COUPLERSET_SEED = 42
COUPLERSET_CSV = "couplerset_{dim}.csv"
# Augmentation of the coupler set (publication strengthening). The original
# 48-point LHS is KEPT: a Latin hypercube of n points is not a subset of one
# with more points, so re-drawing at n=200 would orphan the 16.7 h already
# spent. A second, independent LHS is appended instead -- a sequential
# augmented design. Report it as: "48-point LHS (seed 42) augmented with a
# 152-point LHS (seed 2026); 200 coupler simulations in total."
COUPLEREXT_N = 152
COUPLEREXT_SEED = 2026
COUPLEREXT_IDX0 = 1000

# --- mode sweep: n_eff(lambda, w) and n_g(lambda, w) -----------------------
# FDTD's source eigensolver reports n_eff but NOT group index (the GUI shows
# N/A), so n_g is derived from dispersion: n_g = n_eff - lambda*dn_eff/dlambda.
# 440 nm is included because Paper 11's fabricated ring is 440 x 220 nm --
# that point drives the sub-study 3 literature validation.
MODE_W_NM = [400.0, 425.0, 440.0, 450.0, 475.0, 500.0]
MODE_LAM_NM = [1500.0, 1525.0, 1550.0, 1575.0, 1600.0]
MODE_MESH_NM = 10.0                 # cross-section only, so a fine mesh is cheap
MODESWEEP_CSV = "modesweep.csv"

# --- Bent-waveguide FDE sweep (Phase B) ------------------------------------
# The straight-waveguide mode sweep is the source of two measured defects in
# the ring validation: FSR biased +2.5 % (n_g of a 5.2 um bend is not n_g of a
# straight guide) and a bend-loss model with no wavelength dependence. An FDE
# solve WITH a bend radius returns a COMPLEX effective index: the real part
# fixes the first, the imaginary part fixes the second. Same solver, one sweep.
BENTMODE_CSV = "bentmode.csv"
BENTMODE_R_UM = [5.0, 6.0, 7.5, 10.0, 12.5, 15.0]
BENTMODE_MESH_NM = 10.0      # FDE is cheap; finer than the FDTD production mesh
BENTMODE_TE_MIN = 0.50       # quasi-TE; the fundamental reads ~0.97 here
# A GUIDED mode must lie strictly between the cladding and core indices. The
# FDE solver also returns non-guided solutions sitting at the core index
# (n_eff = 3.471 = bulk Si at 1600 nm, with n_g undefined and zero loss);
# selecting on "highest n_eff" alone picks those. These bounds exclude them.
BENTMODE_NEFF_MIN = 1.45     # just above the SiO2 cladding index (1.444)
BENTMODE_NEFF_MAX = 3.40
# SUPERSEDED 2026-10-13. Searching near an expected guided index was tried
# and measured HARMFUL: setting "use max index"/"n" pins every returned mode
# at the seed index with n_g undefined (--modediag row 5). The solver's own
# default search plus PML mode removal finds the guided mode correctly. This
# constant is retained only because --modediag still exercises the failing
# configuration; nothing in the production path reads it.
BENTMODE_SEARCH_N = 2.30

# ---- Phase C: thermo-optic trimming -----------------------------------
# dn/dT are MATERIAL constants from literature, the same class of input as
# the Palik index data already used throughout. Everything geometry
# dependent -- confinement, n_g, the resulting dlambda/dT -- is solved for.
DNDT_SI = 1.86e-4        # /K, crystalline Si near 1550 nm
DNDT_SIO2 = 1.0e-5       # /K, thermal oxide
THERMAL_LAM_NM = 1550.0
THERMAL_DN = 5.0e-3      # index perturbation for the sensitivity solve
THERMAL_JSON = "thermal_c1.json"
# Heater thermal resistance, K per mW of electrical drive. This is a DEVICE
# parameter from the literature, not a result of this work, and it scales the
# power column linearly -- the kelvin column is independent of it. Reported
# values for silicon ring heaters span roughly 1-10 K/mW depending on whether
# the ring is thermally isolated (undercut) or sits on bulk oxide.
HEATER_K_PER_MW = 3.0
RINGVAL_FDE_ATTEMPTS = 10
# (window scale, trial modes) per retry attempt. The first three repeat the
# default configuration, which is what recovers the intermittent failures;
# later attempts enlarge the solver window for weakly confined modes.
BENTMODE_ESCALATION = [
    (1.00, 10), (1.00, 10), (1.00, 10),
    (1.67, 10), (1.67, 10), (1.67, 20),
    (2.33, 20), (2.33, 20), (3.00, 30), (3.00, 30),
]
# Paper 11 (Tang et al. 2024): fabricated SOI TE ring, measured values
P11_W_NM, P11_R_UM, P11_FSR_NM = 440.0, 20.0, 4.4

# --- dataset generation (Section 7 dual-fidelity path) ----------------------
DATASET_N = 1000                    # Phase 1.5 step 4 target
DATASET_SEED = 42                   # Phase 1.5 step 4
ROUGHNESS_DB_CM = 1.3               # Paper 11, measured; FDTD models none
DATASET_CSV = "mrr_dataset.csv"

# --- Phase 2: preprocessing ------------------------------------------------
SPLIT_SEED = 42                 # Phase 2.4
SPLIT_FRAC = (0.70, 0.15, 0.15) # train / val / test, Phase 2.4
TARGETS = ["lambda_res_nm", "FSR_nm", "Q_L", "IL_dB"]
INPUTS = ["R_um", "w_nm", "g_nm", "Lc_um"]
# Phase 2.2 step 3: supplementary LHS batch aimed at an under-sampled region.
# Scenario A needs FSR >= 16 nm, i.e. L <= lambda^2/(n_g*FSR) ~ 34.85 um, so
# 2*pi*R + 2*Lc <= 34.85 -- a thin corner that plain LHS barely reaches.
SUPP_N = 150
SUPP_SEED = 202                 # distinct from the main LHS seed
SUPP_BOX = dict(R=(5.0, 5.6), w=(400.0, 500.0), g=(150.0, 290.0), Lc=(0.0, 1.6))
SUPP_L_MAX_UM = 34.85

# --- Phase 3: forward model ------------------------------------------------
NN_SEED = 42
NN_HIDDEN = (64, 128, 64)       # Phase 3.2 fixed default
NN_LEAKY = 0.2                  # Phase 3.2 Leaky ReLU alpha
NN_BATCH = 16                   # Phase 3.3
NN_LR = 1e-3                    # Phase 3.3 initial
NN_LR_DECAY = 0.97              # Phase 3.3 exponential decay, per epoch
NN_PATIENCE = 40                # Phase 3.3 early stopping
NN_MAX_EPOCHS = 600
# Phase 3.4 search grid -- report this verbatim in the thesis appendix
GRID_DEPTH = [2, 3, 4]
GRID_WIDTH = [32, 64, 128, 256]
GRID_LR = [1e-4, 5e-4, 1e-3, 5e-3]
GRID_WD = [1e-6, 1e-5, 1e-4]
GRID_SEARCH_EPOCHS = 300        # shorter budget during the search itself

# --- Phase 4: inverse + tandem ---------------------------------------------
# Phase 4.3 fixes the topology for BOTH inverse networks at the Phase 3.2
# default 64-128-64 -- deliberately NOT the Phase 3.4 search winner, so that
# the naive-vs-tandem comparison isolates the loss function, not the size.
INV_HIDDEN = (64, 128, 64)
INV_LR = NN_LR                  # Phase 4.4: "same optimizer ... as Phase 3.3"
INV_WD = 1e-5                   # Phase 3.3 states lambda in [1e-6, 1e-5]
INV_SEED_NAIVE = NN_SEED        # Phase 4.4 step 2 requires the tandem network
INV_SEED_TANDEM = NN_SEED + 1   # to use a DIFFERENT random initialisation
LC_RANGE_UM = 3.0               # Phase 6.2 Fix 4b: s_p for Lc is the range width

# --- Phase 5: Random Forest / XGBoost --------------------------------------
CV_FOLDS = 5                    # Phase 2.4 / 5.2: 5-fold CV on the 70% pool
RF_GRID = dict(n_estimators=[100, 200, 300, 500],
               max_depth=[10, 20, 40, 80, None],
               min_samples_leaf=[1, 2, 5, 7],
               max_features=["sqrt", "log2", None])
XGB_GRID = dict(n_estimators=[100, 300, 500],
                max_depth=[3, 6, 9],
                learning_rate=[0.01, 0.05, 0.1, 0.3],
                subsample=[0.7, 0.85, 1.0],
                colsample_bytree=[0.7, 0.85, 1.0],
                reg_lambda=[0, 1, 5])
XGB_N_ITER = 60                 # Phase 5.2 permits RandomizedSearchCV >= 60

# --- Phase 6: evaluation ---------------------------------------------------
SAMPLE_SIZES = [100, 200, 400, 600, None]   # Phase 6.3; None = full pool
CLOSEDLOOP_N = 20                           # Phase 6.2 requires n >= 20

# --- Phase 7: CPO scenarios ------------------------------------------------
C_LIGHT = 2.99792458e8
# Phase 7.3 fixes each scenario as a BAND or a BOUND, not a point target,
# while Phase 7.4 step 1 asks for a target TUPLE. The construction rule used
# here is stated in the report: a bound is targeted at its stated value, a
# band at its geometric mean, and IL at half its limit so the query asks for
# a design with margin rather than one at the edge of failure.
SCENARIOS = {
    "A": dict(name="Moderate-density CPO link (4 channels, 4 nm grid)",
              lam=1550.0, dlam_ch=4.0, n_ch_req=4,
              fsr_min=16.0, q_lo=1500.0, q_hi=8000.0, il_max=1.5,
              target=(1550.0, 16.0, float(np.sqrt(1500.0*8000.0)), 0.75)),
    "B": dict(name="Dense CPO link (8 channels, 100 GHz / 0.8 nm ITU grid)",
              lam=1550.0, dlam_ch=0.8, n_ch_req=8,
              fsr_min=6.4, q_lo=7750.0, q_hi=None, il_max=1.0,
              target=(1550.0, 6.4, 7750.0, 0.50)),
}
IL_THROUGH_DB = 0.1      # Fix 6's assumed off-resonance through-port loss

# --- Ablation study (optimization items 2, 4, 5) ---------------------------
# None of these change a fixed default in v1.3; they are training-side
# choices the document does not specify. Each is measured separately so the
# effect of each can be attributed, and each runs over several seeds so a
# difference can be told apart from seed noise.
ABLATE_SEEDS = [42, 43, 44]
NG_NOMINAL = 4.2         # for the derived round-trip-length feature

# --- Training-density study ------------------------------------------------
# DIAGNOSTIC ONLY. Extra rows are drawn from the SAME fitted kappa^2 model, so
# they add sampling density, not physics. The dataset still carries the
# information of the fitted coupler set. These results measure how fast the
# problem converges toward its information floor; they are NOT a route to
# claiming O3, and the report says so.
DENSITY_SIZES = [805, 2000, 4000, 6000]
DENSITY_SEED = 909
SUBSTUDY2_CSV = "substudy2_mesh_convergence_{dim}.csv"

# --- movie monitor settings ------------------------------------------------
MOVIE_SIM_TIME = 3e-12   # --quick only: enough to watch light enter the ring
MOVIE_H_RES = 800        # movie plane pixels; higher = sharper but slower

MAT_CORE = "Si (Silicon) - Palik"
MAT_CLAD = "SiO2 (Glass) - Palik"

# Eigenmode search depth for the mode source. Must be set explicitly: Lumerical
# leaves this at 0 ("not yet solved") otherwise, which is NOT the configuration
# validated during template verification.
N_TRIAL_MODES = 10

# Parameter ranges, Section 4.1.2. Enforced, not advisory.
RANGES = {
    "R":  (5.0e-6,   15.0e-6),
    "w":  (400e-9,   500e-9),
    "g":  (150e-9,   350e-9),
    "Lc": (0.0,      3.0e-6),
}


# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------

@dataclass
class MRRParams:
    """The four free geometric parameters of Section 4.1.2."""
    R: float             # ring radius [m]
    w: float             # waveguide width [m]
    g: float             # coupling gap, BOTH couplers [m]
    Lc: float            # coupling length, BOTH couplers [m]

    def validate(self) -> None:
        for name, (lo, hi) in RANGES.items():
            v = getattr(self, name)
            if not (lo - 1e-18 <= v <= hi + 1e-18):
                raise ValueError(
                    f"{name} = {v:.4e} m is outside the Section 4.1.2 range "
                    f"[{lo:.4e}, {hi:.4e}] m"
                )

    @property
    def L(self) -> float:
        """Round-trip length, Eq. 1:  L = 2*pi*R + 2*Lc."""
        import math
        return 2.0 * math.pi * self.R + 2.0 * self.Lc

    def tag(self) -> str:
        return (f"R{self.R*1e6:.3f}um_w{self.w*1e9:.1f}nm_"
                f"g{self.g*1e9:.1f}nm_Lc{self.Lc*1e6:.3f}um")


@dataclass
class MeshConfig:
    """
    Mesh settings. The override resolution is the sweep variable for
    Phase 1.6 sub-study 2; do not treat the defaults below as converged.
    """
    mesh_accuracy: int = 3          # global non-uniform auto mesh
    override_dx: float = 25e-9      # along propagation
    override_dy: float = 20e-9      # ACROSS THE COUPLING GAP - critical axis
    override_dz: float = 25e-9      # vertical

    @classmethod
    def uniform(cls, d: float, mesh_accuracy: int = 3) -> "MeshConfig":
        """Single-scalar mesh for a clean 1-D convergence sweep."""
        return cls(mesh_accuracy=mesh_accuracy,
                   override_dx=d, override_dy=d, override_dz=d)

    def label(self) -> str:
        return (f"ma{self.mesh_accuracy}_dx{self.override_dx*1e9:.1f}"
                f"_dy{self.override_dy*1e9:.1f}_dz{self.override_dz*1e9:.1f}")


@dataclass
class DomainConfig:
    """Simulation-domain margins and run controls."""
    x_margin: float = 2.0e-6        # past ring extent, along bus
    y_margin: float = 1.5e-6        # past outer bus edge
    z_margin: float = 1.0e-6        # above/below core
    coupler_pad_x: float = 0.6e-6   # mesh-override overhang into the bends
    coupler_pad_y: float = 0.15e-6  # mesh-override overhang past outer edges
    sim_time: float = 50e-12        # 50 ps; auto-shutoff usually ends earlier
    auto_shutoff_min: float = 1e-5
    port_inset: float = 1.0e-6      # source/monitor inset from the PML face
    monitor_y_span: float = 2.5e-6
    monitor_z_span: float = 2.0e-6
    # "3D" is the Section 4.1.4 default. "2D" is the Section 7 downgrade,
    # triggered by the timing gate; the 3-D subset quantifies the offset.
    dimension: str = "3D"
    freq_points: int = N_FREQ_POINTS   # coupler runs override this
    movie: bool = True              # ON by default; --no-movie to disable
    # Mid-run checkpointing. None = off. A number = checkpoint every N minutes,
    # letting an interrupted single run resume instead of restarting.
    # DECIDE THIS BEFORE Phase 1.6 sub-study 1: whatever is used in production
    # must also be active during timing calibration, or the measured
    # seconds/sample will not represent production.
    checkpoint_minutes: float | None = 30.0
    extra_layout: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DERIVED GEOMETRY  --  all coordinates in one place, so the layout is auditable
# ---------------------------------------------------------------------------

@dataclass
class Layout:
    ring_half_x: float
    ring_half_y: float
    bus_y: float          # |centreline y| of each bus
    bus_outer_y: float
    sim_x_span: float
    sim_y_span: float
    sim_z_span: float
    x_src: float
    x_through: float
    x_drop: float


def compute_layout(p: MRRParams, d: DomainConfig) -> Layout:
    """
    Racetrack centred on the origin.

      top straight     y = +R,  x in [-Lc/2, +Lc/2]
      bottom straight  y = -R,  x in [-Lc/2, +Lc/2]
      left bend        180 deg arc, centre (-Lc/2, 0), centreline radius R
      right bend       180 deg arc, centre (+Lc/2, 0), centreline radius R

    Centreline length = 2*pi*R + 2*Lc, i.e. Eq. 1 exactly.

    Buses run parallel to the straights, gap g edge-to-edge:
      bottom bus (input -> through)  y = -(R + w + g)
      top bus    (drop  <- add)      y = +(R + w + g)
    """
    ring_half_x = p.Lc / 2.0 + p.R + p.w / 2.0
    ring_half_y = p.R + p.w / 2.0
    bus_y = p.R + p.w + p.g
    bus_outer_y = bus_y + p.w / 2.0

    sim_x_span = 2.0 * (ring_half_x + d.x_margin)
    sim_y_span = 2.0 * (bus_outer_y + d.y_margin)
    sim_z_span = 2.0 * (H_CORE / 2.0 + d.z_margin)

    x_edge = sim_x_span / 2.0
    x_src = -(x_edge - d.port_inset)
    x_through = +(x_edge - d.port_inset)
    x_drop = -(x_edge - d.port_inset) + 0.35e-6   # inboard of the source plane

    return Layout(ring_half_x, ring_half_y, bus_y, bus_outer_y,
                  sim_x_span, sim_y_span, sim_z_span,
                  x_src, x_through, x_drop)


# ---------------------------------------------------------------------------
# LUMAPI LOADER  (Windows)
# ---------------------------------------------------------------------------

def _candidate_lumapi_paths():
    pats = [
        r"C:\Program Files\Lumerical\v*\api\python\lumapi.py",
        r"C:\Program Files\Lumerical\FDTD\api\python\lumapi.py",
        r"C:\Program Files (x86)\Lumerical\v*\api\python\lumapi.py",
    ]
    env = os.environ.get("LUMAPI_PATH")
    found = []
    if env:
        found.append(env)
    for pat in pats:
        found.extend(sorted(glob.glob(pat), reverse=True))  # newest version first
    return found


def load_lumapi():
    """Import lumapi from a Lumerical install without touching PYTHONPATH."""
    for path in _candidate_lumapi_paths():
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("lumapi", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["lumapi"] = mod
            spec.loader.exec_module(mod)
            print(f"[lumapi] loaded from {path}")
            return mod
    raise ImportError(
        "Could not locate lumapi.py. Searched:\n  "
        + "\n  ".join(_candidate_lumapi_paths())
        + "\nSet the LUMAPI_PATH environment variable to the full path of "
          "lumapi.py if your install is elsewhere."
    )


# ---------------------------------------------------------------------------
# TEMPLATE BUILDER
# ---------------------------------------------------------------------------

def _check_materials(fdtd) -> None:
    available = fdtd.getmaterial().split("\n")
    available = [a.strip() for a in available]
    for m in (MAT_CORE, MAT_CLAD):
        if m not in available:
            raise RuntimeError(
                f"Material '{m}' not found in this Lumerical install's database. "
                f"Section 4.1.1 requires dispersive Si and SiO2 models. "
                f"Available materials:\n" + "\n".join(available)
            )


def build_mrr(fdtd,
              p: MRRParams,
              mesh: MeshConfig | None = None,
              dom: DomainConfig | None = None,
              check_materials: bool = True) -> Layout:
    """
    Clear the session and build the full parameterized device.
    Returns the computed Layout for inspection/logging.
    """
    p.validate()
    mesh = mesh or MeshConfig()
    dom = dom or DomainConfig()
    lay = compute_layout(p, dom)

    fdtd.switchtolayout()
    fdtd.deleteall()
    if check_materials:
        _check_materials(fdtd)

    z_core = 0.0
    bus_x_min = -lay.sim_x_span / 2.0 - 1.0e-6   # run buses through the PML
    bus_x_max = +lay.sim_x_span / 2.0 + 1.0e-6

    # ---- cladding (lower mesh priority than the core) --------------------
    fdtd.addrect()
    fdtd.set("name", "cladding")
    fdtd.set("x", 0.0); fdtd.set("x span", lay.sim_x_span + 4e-6)
    fdtd.set("y", 0.0); fdtd.set("y span", lay.sim_y_span + 4e-6)
    fdtd.set("z", z_core); fdtd.set("z span", lay.sim_z_span + 4e-6)
    fdtd.set("material", MAT_CLAD)
    fdtd.set("override mesh order from material database", True)
    fdtd.set("mesh order", 3)

    def _core_rect(name, x, xspan, y, yspan):
        fdtd.addrect()
        fdtd.set("name", name)
        fdtd.set("x", x); fdtd.set("x span", xspan)
        fdtd.set("y", y); fdtd.set("y span", yspan)
        fdtd.set("z", z_core); fdtd.set("z span", H_CORE)
        fdtd.set("material", MAT_CORE)
        fdtd.set("override mesh order from material database", True)
        fdtd.set("mesh order", 2)

    def _core_bend(name, xc, theta_start, theta_stop):
        fdtd.addring()
        fdtd.set("name", name)
        fdtd.set("x", xc); fdtd.set("y", 0.0)
        fdtd.set("z", z_core); fdtd.set("z span", H_CORE)
        fdtd.set("inner radius", p.R - p.w / 2.0)
        fdtd.set("outer radius", p.R + p.w / 2.0)
        fdtd.set("theta start", theta_start)
        fdtd.set("theta stop", theta_stop)
        fdtd.set("material", MAT_CORE)
        fdtd.set("override mesh order from material database", True)
        fdtd.set("mesh order", 2)

    # ---- ring: two straights + two 180 deg bends -------------------------
    if p.Lc > 0.0:
        _core_rect("ring_straight_bottom", 0.0, p.Lc, -p.R, p.w)
        _core_rect("ring_straight_top",    0.0, p.Lc, +p.R, p.w)
    _core_bend("ring_bend_left",  -p.Lc / 2.0,  90.0, 270.0)
    _core_bend("ring_bend_right", +p.Lc / 2.0, -90.0,  90.0)

    # ---- bus waveguides --------------------------------------------------
    for name, ysign in (("bus_input_through", -1.0), ("bus_drop_add", +1.0)):
        fdtd.addrect()
        fdtd.set("name", name)
        fdtd.set("x min", bus_x_min); fdtd.set("x max", bus_x_max)
        fdtd.set("y", ysign * lay.bus_y); fdtd.set("y span", p.w)
        fdtd.set("z", z_core); fdtd.set("z span", H_CORE)
        fdtd.set("material", MAT_CORE)
        fdtd.set("override mesh order from material database", True)
        fdtd.set("mesh order", 2)

    # ---- FDTD region -----------------------------------------------------
    two_d = dom.dimension.upper().startswith("2")
    fdtd.addfdtd()
    fdtd.set("dimension", "2D" if two_d else "3D")
    fdtd.set("x", 0.0); fdtd.set("x span", lay.sim_x_span)
    fdtd.set("y", 0.0); fdtd.set("y span", lay.sim_y_span)
    fdtd.set("z", z_core); fdtd.set("z span", lay.sim_z_span)
    fdtd.set("mesh accuracy", mesh.mesh_accuracy)
    fdtd.set("simulation time", dom.sim_time)
    fdtd.set("auto shutoff min", dom.auto_shutoff_min)
    bcs = ["x min bc", "x max bc", "y min bc", "y max bc"]
    if not two_d:
        bcs += ["z min bc", "z max bc"]
    for bc in bcs:
        fdtd.set(bc, "PML")

    if dom.checkpoint_minutes is not None:
        # Property names differ between Lumerical versions. Warn rather than
        # crash a long batch run if this install spells them differently.
        try:
            fdtd.set("checkpoint during simulation", True)
            fdtd.set("checkpoint period", dom.checkpoint_minutes)
            fdtd.set("checkpoint at shutoff", True)
        except Exception as exc:                      # noqa: BLE001
            print(f"[warn] could not enable checkpointing: {exc}")
            print("[warn] simulations will NOT resume after an interruption; "
                  "check the property names on the FDTD Advanced options tab")

    # ---- mesh overrides at BOTH coupling regions -------------------------
    # y is the gap-normal axis here. (Paper 10 reports the gap-normal
    # direction as the sensitive one; in their layout that axis was x.)
    gap_span_y = 2.0 * p.w + p.g + 2.0 * dom.coupler_pad_y
    for name, ysign in (("mesh_coupler_bottom", -1.0),
                        ("mesh_coupler_top", +1.0)):
        y_ring_edge = ysign * (p.R - p.w / 2.0)
        y_bus_edge = ysign * (lay.bus_y + p.w / 2.0)
        y_centre = 0.5 * (y_ring_edge + y_bus_edge)
        fdtd.addmesh()
        fdtd.set("name", name)
        fdtd.set("x", 0.0)
        fdtd.set("x span", p.Lc + 2.0 * dom.coupler_pad_x)
        fdtd.set("y", y_centre)
        fdtd.set("y span", gap_span_y)
        fdtd.set("z", z_core)
        fdtd.set("z span", H_CORE + 2.0 * dom.coupler_pad_y)
        fdtd.set("override x mesh", True); fdtd.set("dx", mesh.override_dx)
        fdtd.set("override y mesh", True); fdtd.set("dy", mesh.override_dy)
        fdtd.set("override z mesh", True); fdtd.set("dz", mesh.override_dz)

    # ---- source: fundamental TE, broadband 1500-1600 nm ------------------
    fdtd.addmode()
    fdtd.set("name", "source_input")
    fdtd.set("injection axis", "x-axis")
    fdtd.set("direction", "Forward")
    fdtd.set("x", lay.x_src)
    fdtd.set("y", -lay.bus_y); fdtd.set("y span", dom.monitor_y_span)
    fdtd.set("z", z_core)
    if not two_d:
        fdtd.set("z span", dom.monitor_z_span)
    fdtd.set("mode selection", "fundamental TE mode")
    fdtd.set("number of trial modes", N_TRIAL_MODES)
    fdtd.set("override global source settings", False)

    fdtd.setglobalsource("set wavelength", True)
    fdtd.setglobalsource("wavelength start", LAMBDA_MIN)
    fdtd.setglobalsource("wavelength stop", LAMBDA_MAX)

    # ---- monitors: through port and drop port ----------------------------
    fdtd.setglobalmonitor("use source limits", True)
    fdtd.setglobalmonitor("use wavelength spacing", True)
    fdtd.setglobalmonitor("frequency points", dom.freq_points)

    def _power_monitor(name, x, y):
        fdtd.addpower()
        fdtd.set("name", name)
        fdtd.set("monitor type", "Linear Y" if two_d else "2D X-normal")
        fdtd.set("x", x)
        fdtd.set("y", y); fdtd.set("y span", dom.monitor_y_span)
        fdtd.set("z", z_core)
        if not two_d:
            fdtd.set("z span", dom.monitor_z_span)
        fdtd.set("override global monitor settings", False)

    # Light enters +x on the bottom bus. In the ring it travels +x along the
    # bottom straight and -x along the top straight, so the drop port is the
    # LEFT end of the top bus and the add port is its right end.
    _power_monitor("monitor_through", lay.x_through, -lay.bus_y)
    _power_monitor("monitor_drop",    lay.x_drop,    +lay.bus_y)

    # index monitor, for the geometry check in inspect_build()
    fdtd.addindex()
    fdtd.set("name", "index_xy")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x", 0.0); fdtd.set("x span", lay.sim_x_span)
    fdtd.set("y", 0.0); fdtd.set("y span", lay.sim_y_span)
    fdtd.set("z", z_core)

    # ---- field-animation monitor (on unless movie=False) -----------------
    if dom.movie:
        fdtd.addmovie()
        fdtd.set("name", "movie_xy")
        fdtd.set("monitor type", "2D Z-normal")
        fdtd.set("x", 0.0); fdtd.set("x span", lay.sim_x_span)
        fdtd.set("y", 0.0); fdtd.set("y span", lay.sim_y_span)
        fdtd.set("z", z_core)
        fdtd.set("horizontal resolution", MOVIE_H_RES)
        fdtd.set("lock aspect ratio", True)
        # autoscale each frame, else the ring fields are invisible next to
        # the much brighter bus
        fdtd.set("scale", 1)

    return lay


# ---------------------------------------------------------------------------
# INSPECTION
# ---------------------------------------------------------------------------

def inspect_build(p: MRRParams, lay: Layout, mesh: MeshConfig,
                  dom: DomainConfig) -> str:
    import math
    ng_nominal = 4.2                       # Section 4.1.2, fixed nominal value
    fsr_pred = LAMBDA_TARGET ** 2 / (ng_nominal * p.L)   # Eq. 4
    n_peaks = (LAMBDA_MAX - LAMBDA_MIN) / fsr_pred
    lines = [
        "=" * 66,
        "GEOMETRY CHECK",
        "=" * 66,
        f"  R  = {p.R*1e6:8.3f} um      w  = {p.w*1e9:7.1f} nm",
        f"  g  = {p.g*1e9:8.1f} nm      Lc = {p.Lc*1e6:7.3f} um",
        f"  h  = {H_CORE*1e9:8.1f} nm  (fixed)",
        "",
        f"  Round-trip length L (Eq. 1) : {p.L*1e6:.4f} um",
        f"  Predicted FSR (Eq. 4, ng={ng_nominal}) : {fsr_pred*1e9:.3f} nm",
        f"  Resonances in 1500-1600 nm window : ~{n_peaks:.1f}"
        + ("   OK (>=2 needed for FSR)" if n_peaks >= 2 else "   *** TOO FEW ***"),
        "",
        f"  Bus centreline |y|  : {lay.bus_y*1e6:.4f} um",
        f"  Ring half-extent x  : {lay.ring_half_x*1e6:.4f} um",
        f"  Sim domain (x,y,z)  : {lay.sim_x_span*1e6:.2f} x "
        f"{lay.sim_y_span*1e6:.2f} x {lay.sim_z_span*1e6:.2f} um",
        f"  Source x / through x / drop x : {lay.x_src*1e6:.3f} / "
        f"{lay.x_through*1e6:.3f} / {lay.x_drop*1e6:.3f} um",
        "",
        f"  Mesh : accuracy {mesh.mesh_accuracy}, override "
        f"dx/dy/dz = {mesh.override_dx*1e9:.1f}/{mesh.override_dy*1e9:.1f}/"
        f"{mesh.override_dz*1e9:.1f} nm",
        f"  Cells across the {p.g*1e9:.0f} nm gap : "
        f"{p.g/mesh.override_dy:.1f}",
        f"  Sim time {dom.sim_time*1e12:.0f} ps, auto-shutoff "
        f"{dom.auto_shutoff_min:g}",
        f"  Monitor frequency points : {N_FREQ_POINTS}",
        f"  Solver dimension : {dom.dimension.upper()}"
        + ("   <-- Section 7 downgrade, log it in the Risk Log"
           if dom.dimension.upper().startswith("2") else ""),
        f"  Source: fundamental TE mode, {N_TRIAL_MODES} trial modes",
        "  Checkpointing : " + ("OFF (interrupted run restarts from zero)"
                                if dom.checkpoint_minutes is None
                                else f"every {dom.checkpoint_minutes:g} min"),
        "=" * 66,
    ]
    lines += [
        f"  Objects built : {15 if dom.movie else 14}"
        + ("  (includes movie_xy)" if dom.movie else "  (no movie monitor)"),
        "=" * 66,
    ]
    if dom.movie:
        lines += [
            "*** MOVIE MONITOR PRESENT (movie_xy) ***",
            "    Runtime and memory are inflated. Do NOT use this build for",
            "    sub-study 2, sub-study 1, or mass production -- rerun with",
            "    --no-movie for any run whose timing you intend to trust.",
            "=" * 66,
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# COUPLER-ONLY GEOMETRY  (Section 7 dual-fidelity path)
# ---------------------------------------------------------------------------

def build_coupler(fdtd, p, mesh=None, dom=None):
    """
    Build the bottom coupling region ONLY: the bus, the ring's bottom straight,
    and the two quarter-bends that carry the ring waveguide up to y = 0, where
    vertical stubs take it out through the PML.

    Because the loop is open, light transits once. There is no resonant
    build-up, so the run is short and its cost does not scale with Q -- which
    is what makes the dual-fidelity path affordable.

    Launch the fundamental TE mode into the bus and measure:
        t^2     = power remaining in the bus   (monitor_through)
        kappa^2 = power crossed into the ring  (monitor_cross)
    These feed Eqs. 2, 3, 5 and 6 directly.
    """
    p.validate()
    mesh = mesh or MeshConfig()
    dom = dom or DomainConfig()
    two_d = dom.dimension.upper().startswith("2")
    z_core = 0.0

    x_cut = p.Lc / 2.0 + p.R              # where the ring reaches y = 0
    bus_y = p.R + p.w + p.g
    y_min = -(bus_y + p.w / 2.0 + dom.y_margin)
    y_max = dom.y_margin + 1.0e-6
    sim_x_span = 2.0 * (x_cut + p.w / 2.0 + dom.x_margin)
    sim_y_span = y_max - y_min
    y_ctr = 0.5 * (y_max + y_min)

    fdtd.switchtolayout()
    fdtd.deleteall()

    def _rect(name, xmin, xmax, y, yspan, mat=MAT_CORE, order=2):
        fdtd.addrect(); fdtd.set("name", name)
        fdtd.set("x min", xmin); fdtd.set("x max", xmax)
        fdtd.set("y", y); fdtd.set("y span", yspan)
        fdtd.set("z", z_core); fdtd.set("z span", H_CORE)
        fdtd.set("material", mat)
        fdtd.set("override mesh order from material database", True)
        fdtd.set("mesh order", order)

    fdtd.addrect(); fdtd.set("name", "cladding")
    fdtd.set("x", 0.0); fdtd.set("x span", sim_x_span + 4e-6)
    fdtd.set("y", y_ctr); fdtd.set("y span", sim_y_span + 4e-6)
    fdtd.set("z", z_core); fdtd.set("z span", 4e-6)
    fdtd.set("material", MAT_CLAD)
    fdtd.set("override mesh order from material database", True)
    fdtd.set("mesh order", 3)

    if p.Lc > 0.0:
        _rect("ring_straight_bottom", -p.Lc / 2.0, p.Lc / 2.0, -p.R, p.w)
    for name, xc, th0, th1 in (("ring_bend_left", -p.Lc / 2.0, 180.0, 270.0),
                               ("ring_bend_right", p.Lc / 2.0, 270.0, 360.0)):
        fdtd.addring(); fdtd.set("name", name)
        fdtd.set("x", xc); fdtd.set("y", 0.0)
        fdtd.set("z", z_core); fdtd.set("z span", H_CORE)
        fdtd.set("inner radius", p.R - p.w / 2.0)
        fdtd.set("outer radius", p.R + p.w / 2.0)
        fdtd.set("theta start", th0); fdtd.set("theta stop", th1)
        fdtd.set("material", MAT_CORE)
        fdtd.set("override mesh order from material database", True)
        fdtd.set("mesh order", 2)

    # vertical stubs carrying the open ring ends out through the PML
    for name, xc in (("ring_exit_left", -x_cut), ("ring_exit_right", x_cut)):
        fdtd.addrect(); fdtd.set("name", name)
        fdtd.set("x", xc); fdtd.set("x span", p.w)
        fdtd.set("y min", -1e-9); fdtd.set("y max", y_max + 1.0e-6)
        fdtd.set("z", z_core); fdtd.set("z span", H_CORE)
        fdtd.set("material", MAT_CORE)
        fdtd.set("override mesh order from material database", True)
        fdtd.set("mesh order", 2)

    _rect("bus_input_through", -sim_x_span / 2.0 - 1e-6,
          sim_x_span / 2.0 + 1e-6, -bus_y, p.w)

    fdtd.addfdtd()
    fdtd.set("dimension", "2D" if two_d else "3D")
    fdtd.set("x", 0.0); fdtd.set("x span", sim_x_span)
    fdtd.set("y", y_ctr); fdtd.set("y span", sim_y_span)
    fdtd.set("z", z_core)
    if not two_d:
        fdtd.set("z span", 2.0 * (H_CORE / 2.0 + dom.z_margin))
    fdtd.set("mesh accuracy", mesh.mesh_accuracy)
    fdtd.set("simulation time", dom.sim_time)
    fdtd.set("auto shutoff min", dom.auto_shutoff_min)
    bcs = ["x min bc", "x max bc", "y min bc", "y max bc"]
    if not two_d:
        bcs += ["z min bc", "z max bc"]
    for bc in bcs:
        fdtd.set(bc, "PML")

    y_ring_edge = -(p.R - p.w / 2.0)
    y_bus_edge = -(bus_y + p.w / 2.0)
    fdtd.addmesh(); fdtd.set("name", "mesh_coupler")
    fdtd.set("x", 0.0); fdtd.set("x span", p.Lc + 2.0 * dom.coupler_pad_x)
    fdtd.set("y", 0.5 * (y_ring_edge + y_bus_edge))
    fdtd.set("y span", 2.0 * p.w + p.g + 2.0 * dom.coupler_pad_y)
    fdtd.set("z", z_core); fdtd.set("z span", H_CORE + 2.0 * dom.coupler_pad_y)
    fdtd.set("override x mesh", True); fdtd.set("dx", mesh.override_dx)
    fdtd.set("override y mesh", True); fdtd.set("dy", mesh.override_dy)
    fdtd.set("override z mesh", True); fdtd.set("dz", mesh.override_dz)

    x_in = -(sim_x_span / 2.0 - dom.port_inset)
    fdtd.addmode(); fdtd.set("name", "source_input")
    fdtd.set("injection axis", "x-axis"); fdtd.set("direction", "Forward")
    fdtd.set("x", x_in)
    fdtd.set("y", -bus_y); fdtd.set("y span", dom.monitor_y_span)
    fdtd.set("z", z_core)
    if not two_d:
        fdtd.set("z span", dom.monitor_z_span)
    fdtd.set("mode selection", "fundamental TE mode")
    fdtd.set("number of trial modes", N_TRIAL_MODES)
    fdtd.set("override global source settings", False)

    fdtd.setglobalsource("set wavelength", True)
    fdtd.setglobalsource("wavelength start", LAMBDA_MIN)
    fdtd.setglobalsource("wavelength stop", LAMBDA_MAX)
    fdtd.setglobalmonitor("use source limits", True)
    fdtd.setglobalmonitor("use wavelength spacing", True)
    fdtd.setglobalmonitor("frequency points", dom.freq_points)

    fdtd.addpower(); fdtd.set("name", "monitor_through")
    fdtd.set("monitor type", "Linear Y" if two_d else "2D X-normal")
    fdtd.set("x", sim_x_span / 2.0 - dom.port_inset)
    fdtd.set("y", -bus_y); fdtd.set("y span", dom.monitor_y_span)
    fdtd.set("z", z_core)
    if not two_d:
        fdtd.set("z span", dom.monitor_z_span)

    fdtd.addpower(); fdtd.set("name", "monitor_cross")
    fdtd.set("monitor type", "Linear X" if two_d else "2D Y-normal")
    fdtd.set("y", y_max - dom.port_inset)
    fdtd.set("x", x_cut); fdtd.set("x span", dom.monitor_y_span)
    fdtd.set("z", z_core)
    if not two_d:
        fdtd.set("z span", dom.monitor_z_span)

    return dict(x_cut=x_cut, bus_y=bus_y,
                sim_x_span=sim_x_span, sim_y_span=sim_y_span)


def _at_target(lam, T, target=LAMBDA_TARGET):
    """Linear interpolation of a smooth spectrum at one wavelength."""
    if lam.size == 0:
        return float("nan")
    return float(np.interp(target, lam, T))


_S2C_FIELDS = ["d_nm", "cells_across_gap", "kappa2", "t2", "sum_k2_t2",
               "t_coeff", "runtime_s"]


def run_substudy2_coupler(dimension="2D"):
    """
    Phase 1.6 sub-study 2, re-aimed at the coupler.

    Same method and same 1e-4 gradient criterion as before, applied to
    kappa^2 -- the quantity the dual-fidelity path actually depends on --
    instead of a resonant transmittance the full-ring route cannot afford.
    """
    p = MRRParams(R=10.0e-6, w=450e-9, g=200e-9, Lc=1.5e-6)
    dom = DomainConfig(movie=False, dimension=dimension,
                       sim_time=COUPLER_SIM_TIME,
                       freq_points=COUPLER_FREQ_POINTS)

    csv_path = os.path.abspath(SUBSTUDY2C_CSV.format(dim=dimension.lower()))
    done = _s2_load(csv_path)
    todo = [d for d in MESH_SWEEP_NM if d not in done]
    print(f"[coupler] {len(done)} done, {len(todo)} to run  ({dimension}, "
          f"{COUPLER_SIM_TIME*1e12:.0f} ps window)")

    lumapi = load_lumapi()
    for d_nm in todo:
        mesh = MeshConfig.uniform(d_nm * 1e-9)
        print(f"\n[coupler] d = {d_nm:g} nm "
              f"({p.g/(d_nm*1e-9):.2f} cells across the gap) ...")
        t0 = time.time()
        with lumapi.FDTD(hide=True) as fdtd:
            build_coupler(fdtd, p, mesh, dom)
            fdtd.save(os.path.abspath(
                f"coupler_{dimension.lower()}_d{d_nm:g}nm.fsp"))
            fdtd.run()
            lam_t, Tt = get_spectrum(fdtd, "monitor_through")
            lam_c, Tc = get_spectrum(fdtd, "monitor_cross")
        dt = time.time() - t0
        t2 = _at_target(lam_t, Tt)
        k2 = _at_target(lam_c, Tc)
        row = dict(d_nm=d_nm, cells_across_gap=round(p.g / (d_nm * 1e-9), 3),
                   kappa2=k2, t2=t2, sum_k2_t2=k2 + t2,
                   t_coeff=float(np.sqrt(t2)) if t2 == t2 and t2 >= 0 else float("nan"),
                   runtime_s=round(dt, 1))
        with open(csv_path, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=_S2C_FIELDS)
            if fh.tell() == 0:
                wr.writeheader()
            wr.writerow(row)
        done[d_nm] = {k: str(v) for k, v in row.items()}
        print(f"[coupler] kappa^2 {k2:.6f}  t^2 {t2:.6f}  "
              f"sum {k2+t2:.4f}  {dt:.1f}s")

    # report
    ds = sorted(done.keys(), reverse=True)
    print("\n" + "=" * 78)
    print("PHASE 1.6 SUB-STUDY 2 (COUPLER) -- MESH CONVERGENCE OF kappa^2")
    print("=" * 78)
    print(f"{'d(nm)':>7}{'cells':>7}{'kappa^2':>11}{'t^2':>10}"
          f"{'k2+t2':>9}{'|dk2/dN|':>12}{'runtime':>10}")
    print("-" * 78)
    prev = None
    verdict = None
    for d in ds:
        r = done[d]
        k2 = float(r["kappa2"]); N = float(r["cells_across_gap"])
        g = ""
        if prev is not None:
            dN = abs(N - prev[1])
            gv = abs(k2 - prev[0]) / dN if dN else float("nan")
            g = f"{gv:.3e}"
            if gv < CONVERGENCE_GRADIENT and verdict is None:
                verdict = d
        print(f"{d:7.1f}{N:7.2f}{k2:11.6f}{float(r['t2']):10.6f}"
              f"{float(r['sum_k2_t2']):9.4f}{g:>12}{float(r['runtime_s']):9.1f}s")
        prev = (k2, N)
    print("-" * 78)
    print(f"Criterion: |d(kappa^2)/d(cells across gap)| < {CONVERGENCE_GRADIENT:g}")
    if verdict is None:
        print("VERDICT: not converged across the swept range -- extend "
              "MESH_SWEEP_NM finer.")
    else:
        print(f"VERDICT: converged at d = {verdict:g} nm.")
    print("Sanity: kappa^2 + t^2 should be slightly below 1; the shortfall is "
          "bend radiation\n         and is itself a useful loss measurement.")
    print("=" * 78)
    print(f"\nResults: {csv_path}")


# ---------------------------------------------------------------------------
# BEND-RADIATION LOSS vs RADIUS
# ---------------------------------------------------------------------------

def build_bend(fdtd, R, w, mesh, dom):
    """
    One 90-degree bend of radius R with straight stubs at each end.
    Light enters the vertical stub travelling -y and leaves the horizontal
    stub travelling +x, so transmission gives the loss of a quarter bend.
    """
    two_d = dom.dimension.upper().startswith("2")
    z = 0.0
    outer = R + w / 2.0
    x_lo = -(outer + dom.x_margin)
    x_hi = BEND_STUB + dom.x_margin
    y_lo = -(outer + dom.y_margin)
    y_hi = BEND_STUB + dom.y_margin

    fdtd.switchtolayout(); fdtd.deleteall()

    fdtd.addrect(); fdtd.set("name", "cladding")
    fdtd.set("x", 0.5 * (x_lo + x_hi)); fdtd.set("x span", (x_hi - x_lo) + 4e-6)
    fdtd.set("y", 0.5 * (y_lo + y_hi)); fdtd.set("y span", (y_hi - y_lo) + 4e-6)
    fdtd.set("z", z); fdtd.set("z span", 4e-6)
    fdtd.set("material", MAT_CLAD)
    fdtd.set("override mesh order from material database", True)
    fdtd.set("mesh order", 3)

    fdtd.addring(); fdtd.set("name", "bend")
    fdtd.set("x", 0.0); fdtd.set("y", 0.0)
    fdtd.set("z", z); fdtd.set("z span", H_CORE)
    fdtd.set("inner radius", R - w / 2.0)
    fdtd.set("outer radius", R + w / 2.0)
    fdtd.set("theta start", 180.0); fdtd.set("theta stop", 270.0)
    fdtd.set("material", MAT_CORE)
    fdtd.set("override mesh order from material database", True)
    fdtd.set("mesh order", 2)

    fdtd.addrect(); fdtd.set("name", "stub_in")      # vertical, at x = -R
    fdtd.set("x", -R); fdtd.set("x span", w)
    fdtd.set("y min", -1e-9); fdtd.set("y max", y_hi + 1e-6)
    fdtd.set("z", z); fdtd.set("z span", H_CORE)
    fdtd.set("material", MAT_CORE)
    fdtd.set("override mesh order from material database", True)
    fdtd.set("mesh order", 2)

    fdtd.addrect(); fdtd.set("name", "stub_out")     # horizontal, at y = -R
    fdtd.set("y", -R); fdtd.set("y span", w)
    fdtd.set("x min", -1e-9); fdtd.set("x max", x_hi + 1e-6)
    fdtd.set("z", z); fdtd.set("z span", H_CORE)
    fdtd.set("material", MAT_CORE)
    fdtd.set("override mesh order from material database", True)
    fdtd.set("mesh order", 2)

    fdtd.addfdtd()
    fdtd.set("dimension", "2D" if two_d else "3D")
    fdtd.set("x", 0.5 * (x_lo + x_hi)); fdtd.set("x span", x_hi - x_lo)
    fdtd.set("y", 0.5 * (y_lo + y_hi)); fdtd.set("y span", y_hi - y_lo)
    fdtd.set("z", z)
    if not two_d:
        fdtd.set("z span", 2.0 * (H_CORE / 2.0 + dom.z_margin))
    fdtd.set("mesh accuracy", mesh.mesh_accuracy)
    fdtd.set("simulation time", dom.sim_time)
    fdtd.set("auto shutoff min", dom.auto_shutoff_min)
    bcs = ["x min bc", "x max bc", "y min bc", "y max bc"]
    if not two_d:
        bcs += ["z min bc", "z max bc"]
    for bc in bcs:
        fdtd.set(bc, "PML")

    # fine mesh over the whole bend, where radiation happens
    fdtd.addmesh(); fdtd.set("name", "mesh_bend")
    fdtd.set("x", -0.5 * outer); fdtd.set("x span", outer + 2 * w)
    fdtd.set("y", -0.5 * outer); fdtd.set("y span", outer + 2 * w)
    fdtd.set("z", z); fdtd.set("z span", H_CORE + 2 * dom.coupler_pad_y)
    fdtd.set("override x mesh", True); fdtd.set("dx", mesh.override_dx)
    fdtd.set("override y mesh", True); fdtd.set("dy", mesh.override_dy)
    fdtd.set("override z mesh", True); fdtd.set("dz", mesh.override_dz)

    fdtd.setglobalsource("set wavelength", True)
    fdtd.setglobalsource("wavelength start", LAMBDA_MIN)
    fdtd.setglobalsource("wavelength stop", LAMBDA_MAX)
    fdtd.setglobalmonitor("use source limits", True)
    fdtd.setglobalmonitor("use wavelength spacing", True)
    fdtd.setglobalmonitor("frequency points", dom.freq_points)

    fdtd.addmode(); fdtd.set("name", "source_input")
    fdtd.set("injection axis", "y-axis"); fdtd.set("direction", "Backward")
    fdtd.set("y", BEND_STUB - 0.5e-6)
    fdtd.set("x", -R); fdtd.set("x span", dom.monitor_y_span)
    fdtd.set("z", z)
    if not two_d:
        fdtd.set("z span", dom.monitor_z_span)
    fdtd.set("mode selection", "fundamental TE mode")
    fdtd.set("number of trial modes", N_TRIAL_MODES)
    fdtd.set("override global source settings", False)

    fdtd.addpower(); fdtd.set("name", "monitor_out")
    fdtd.set("monitor type", "Linear Y" if two_d else "2D X-normal")
    fdtd.set("x", BEND_STUB - 0.5e-6)
    fdtd.set("y", -R); fdtd.set("y span", dom.monitor_y_span)
    fdtd.set("z", z)
    if not two_d:
        fdtd.set("z span", dom.monitor_z_span)



def _dump_fde_state(m, tag):
    """
    Print the solver state as the session actually holds it, right before
    findmodes(). Used to compare two code paths that send what appears to be
    the same sequence of calls but solve differently.
    """
    print(f"[state] ---- {tag} ----")
    for k in ("solver type", "y span", "z span",
              "define y mesh by", "dy", "define z mesh by", "dz",
              "y min bc", "y max bc", "z min bc", "z max bc",
              "mesh cells y", "mesh cells z"):
        try:
            print(f"[state]   set {k:22s} = {m.get(k)!r}")
        except Exception as exc:                        # noqa: BLE001
            print(f"[state]   set {k:22s} : {type(exc).__name__}")
    for k in ("wavelength", "number of trial modes", "use max index", "n",
              "bent waveguide", "bend radius", "calculate group index"):
        try:
            print(f"[state]   ana {k:22s} = {m.getanalysis(k)!r}")
        except Exception as exc:                        # noqa: BLE001
            print(f"[state]   ana {k:22s} : {type(exc).__name__}")


def _fde_mode(m, R_um, w_nm, lam_nm, probe=False):
    """
    One FDE solve of the bent waveguide cross-section. Returns
    (neff_real, neff_imag, te_fraction). R_um = None solves the straight
    guide, which is the control: it must reproduce the existing straight
    mode sweep, and if it does not, nothing else here can be trusted.
    """
    m.switchtolayout(); m.deleteall()
    m.addrect(); m.set("name", "cladding")
    m.set("x", 0.0); m.set("x span", 6e-6)
    m.set("y", 0.0); m.set("y span", 6e-6)
    m.set("z", 0.0); m.set("z span", 6e-6)
    m.set("material", MAT_CLAD)
    m.set("override mesh order from material database", True)
    m.set("mesh order", 3)
    m.addrect(); m.set("name", "wg")
    m.set("x", 0.0); m.set("x span", 6e-6)
    m.set("y", 0.0); m.set("y span", w_nm * 1e-9)
    m.set("z", 0.0); m.set("z span", H_CORE)
    m.set("material", MAT_CORE)
    m.set("override mesh order from material database", True)
    m.set("mesh order", 2)

    d = BENTMODE_MESH_NM * 1e-9

    def _build_solver(scale=1.0, ntrial=None):
        """
        Add the FDE and apply every solver + analysis setting.

        scale enlarges the solver window. A weakly confined mode (narrow
        guide, long wavelength, tight bend) extends further into the cladding
        than a well confined one, so the default window can put PML where the
        field still has amplitude. Enlarging it is the escalation used when
        repeated rebuilds at the default size keep returning a degenerate
        solve.
        """
        if ntrial is None:
            ntrial = N_TRIAL_MODES
        m.addfde()
        m.set("solver type", "2D X normal")
        m.set("x", 0.0)
        m.set("y", 0.0); m.set("y span", 3.0e-6 * scale)
        m.set("z", 0.0); m.set("z span", 2.5e-6 * scale)
        m.set("define y mesh by", "maximum mesh step"); m.set("dy", d)
        m.set("define z mesh by", "maximum mesh step"); m.set("dz", d)
        m.set("y min bc", "PML"); m.set("y max bc", "PML")
        m.set("z min bc", "PML"); m.set("z max bc", "PML")
        m.setanalysis("wavelength", lam_nm * 1e-9)
        m.setanalysis("number of trial modes", ntrial)
        # No near-n search and no "calculate group index": each of those, on
        # its own, has been measured to pin every mode at a seed index with
        # n_g undefined (2026-10-12). PML already removes the radiation modes.
        if R_um is not None:
            m.setanalysis("bent waveguide", 1)
            m.setanalysis("bend radius", R_um * 1e-6)
            m.setanalysis("bend orientation", 0.0)
        return m.findmodes()

    # RETRY. The identical call sequence has been observed to produce either a
    # correct solve (PML active, 2 modes, n_g ~ 4.35) or a degenerate one
    # (PML inactive, N modes all at the bulk index, n_g undefined) in
    # different sessions -- see --modediag row 4 against --bentmode on
    # 2026-10-12. The root cause is not understood, so the solve is verified
    # and the solver rebuilt from scratch until it produces a real result.
    # A degenerate solve is cheap to detect: every mode shares one index.
    searched = None
    nfound = 0
    used = None
    for attempt in range(1, RINGVAL_FDE_ATTEMPTS + 1):
        scale, ntrial = BENTMODE_ESCALATION[
            min(attempt - 1, len(BENTMODE_ESCALATION) - 1)]
        used = (scale, ntrial)
        nfound = _build_solver(scale, ntrial)
        # ACCEPTANCE. Test for the thing actually needed -- at least one
        # guided mode -- not for a proxy. An earlier version accepted any
        # solve whose indices were not all identical, which passed a result
        # holding two distinct bulk-silicon values and no guided mode at all.
        idx, guided_here = [], 0
        for i in range(1, int(ntrial) + 1):
            nm0 = f"FDE::data::mode{i}"
            try:
                ne0 = complex(np.array(
                    m.getdata(nm0, "neff")).flatten()[0])
            except Exception:                           # noqa: BLE001
                break
            idx.append(round(ne0.real, 6))
            try:
                ng0 = complex(np.array(
                    m.getdata(nm0, "ng")).flatten()[0]).real
            except Exception:                           # noqa: BLE001
                ng0 = float("nan")
            try:
                te0 = complex(np.array(
                    m.getdata(nm0, "TE polarization fraction")
                ).flatten()[0]).real
            except Exception:                           # noqa: BLE001
                te0 = float("nan")
            if (BENTMODE_NEFF_MIN < ne0.real < BENTMODE_NEFF_MAX
                    and np.isfinite(ng0)
                    and (te0 != te0 or te0 >= BENTMODE_TE_MIN)):
                guided_here += 1
        if guided_here:
            if attempt > 1:
                print(f"[bentmode] solve recovered on attempt {attempt} "
                      f"(R={R_um}, w={w_nm}, {lam_nm} nm) with window "
                      f"x{used[0]:.2f}, {used[1]:.0f} trial modes "
                      f"-- {guided_here} guided mode(s)")
            break
        if attempt < RINGVAL_FDE_ATTEMPTS:
            nxt = BENTMODE_ESCALATION[
                min(attempt, len(BENTMODE_ESCALATION) - 1)]
            print(f"[bentmode] no guided mode ({len(idx)} modes, "
                  f"{len(set(idx))} distinct index/indices) "
                  f"at R={R_um}, w={w_nm}, {lam_nm} nm -- rebuilding "
                  f"the solver, attempt {attempt + 1} "
                  f"(window x{nxt[0]:.2f}, {nxt[1]:.0f} trial modes)")
            m.switchtolayout()
            try:
                m.select("FDE"); m.delete()
            except Exception:                           # noqa: BLE001
                pass
    if probe:
        print(f"[bentmode] probe: findmodes() returned {nfound!r} "
              f"(trial modes requested: {N_TRIAL_MODES})")

    # SELECTION. Lumerical's "TE polarization fraction" uses a convention
    # that does not match the waveguide sense here (no mode exceeded 0.8 even
    # for a strip whose fundamental mode is quasi-TE at 98 % by the FDTD
    # source's own reckoning). Rather than guess which convention means what,
    # select on something unambiguous: for a 220 nm SOI strip 400-500 nm wide
    # the fundamental quasi-TE mode has the HIGHEST real effective index. Both
    # polarisation fractions are recorded so the choice can be checked after
    # the fact instead of trusted.
    best, seen, recs = None, 0, []
    for i in range(1, N_TRIAL_MODES + 1):
        nm_ = f"FDE::data::mode{i}"
        try:
            ne = complex(np.array(m.getdata(nm_, "neff")).flatten()[0])
        except Exception:                               # noqa: BLE001
            break
        seen = i
        rec = dict(neff=ne)
        for key, out in (("ng", "ng"), ("loss", "loss"),
                         ("TE polarization fraction", "te"),
                         ("waveguide TE/TM fraction", "wg_te")):
            try:
                rec[out] = complex(
                    np.array(m.getdata(nm_, key)).flatten()[0]).real
            except Exception:                           # noqa: BLE001
                rec[out] = float("nan")
        # Guided-mode test: index between cladding and core, a finite group
        # index, and predominantly TE. All three come from the solver.
        guided = (BENTMODE_NEFF_MIN < ne.real < BENTMODE_NEFF_MAX
                  and np.isfinite(rec.get("ng", float("nan")))
                  and (rec.get("te") != rec.get("te")      # NaN -> unavailable
                       or rec["te"] >= BENTMODE_TE_MIN))
        rec["guided"] = guided
        rec["i"] = i
        recs.append(rec)
        if probe:
            print(f"[bentmode]   mode{i:<2d} neff={ne.real:8.5f}"
                  f"{ne.imag:+10.3e}j  ng={rec.get('ng'):9.5f}  "
                  f"TE={rec.get('te'):6.3f}  wgTE={rec.get('wg_te'):6.3f}  "
                  f"loss={rec.get('loss'):10.4g}  "
                  f"{'GUIDED' if guided else 'rejected'}")
        if guided and (best is None or ne.real > best["neff"].real):
            best = rec
    if probe:
        print(f"[bentmode] probe: "
              f"{seen} mode(s) readable; selected "
              f"neff = {best['neff'].real:.5f}, ng = {best.get('ng'):.5f}, "
              f"TE frac = {best.get('te'):.3f}, "
              f"wg TE/TM = {best.get('wg_te'):.3f}" if best else
              f"[bentmode] probe: {seen} mode(s) readable, none guided")
    if best is not None:
        _ng = best.get("ng", float("nan"))
        if np.isfinite(_ng) and _ng <= best["neff"].real * 1.05:
            raise RuntimeError(
                f"group index has collapsed onto the phase index at "
                f"R={R_um} um, w={w_nm} nm, {lam_nm} nm: n_g = {_ng:.5f} "
                f"against n_eff = {best['neff'].real:.5f}. A guided silicon "
                f"mode has n_g well above n_eff (about 4.35 here), so this "
                f"is the dispersion calculation not running, not a physical "
                f"result. Refusing to record it.")
    if best is None:
        # Print what the solver actually returned. A rejection is only
        # informative if the rejected candidates are visible.
        print(f"[bentmode] no guided mode at R={R_um} um, w={w_nm} nm, "
              f"{lam_nm} nm -- the {seen} returned mode(s) were:")
        for r in recs:
            print(f"[bentmode]   mode{r['i']:<2d} "
                  f"neff={r['neff'].real:8.5f}{r['neff'].imag:+10.3e}j  "
                  f"ng={r.get('ng'):9.5f}  TE={r.get('te'):6.3f}  "
                  f"wgTE={r.get('wg_te'):6.3f}  loss={r.get('loss'):10.4g}")
        raise RuntimeError(
            f"no guided mode at R={R_um} um, w={w_nm} nm, {lam_nm} nm "
            f"({seen} mode(s) readable; required "
            f"{BENTMODE_NEFF_MIN} < neff < {BENTMODE_NEFF_MAX}, finite ng, "
            f"TE fraction >= {BENTMODE_TE_MIN})")
    return best


def _bent_loss_db_per_cm(neff_imag, lam_nm):
    """
    Propagation loss from the imaginary part of the effective index.
    Power decays as exp(-2 k0 Im(neff) z), so
      loss[dB/m] = 10 log10(e) * 2 * (2 pi / lambda) * Im(neff).
    """
    k0 = 2.0 * np.pi / (lam_nm * 1e-9)
    return float(10.0 * np.log10(np.e) * 2.0 * k0 * abs(neff_imag) / 100.0)



def _diag_build(m, cfg):
    """Build structures + FDE for one diagnostic configuration."""
    m.switchtolayout(); m.deleteall()
    m.addrect(); m.set("name", "cladding")
    m.set("x", 0.0); m.set("x span", 6e-6)
    m.set("y", 0.0); m.set("y span", 6e-6)
    m.set("z", 0.0); m.set("z span", 6e-6)
    m.set("material", MAT_CLAD)
    m.set("override mesh order from material database", True)
    m.set("mesh order", 3)
    m.addrect(); m.set("name", "wg")
    m.set("x", 0.0); m.set("x span", 6e-6)
    m.set("y", 0.0); m.set("y span", 400e-9)
    m.set("z", 0.0); m.set("z span", H_CORE)
    m.set("material", MAT_CORE)
    m.set("override mesh order from material database", True)
    m.set("mesh order", 2)

    def _fde():
        m.addfde()
        m.set("solver type", "2D X normal")
        m.set("x", 0.0)
        m.set("y", 0.0); m.set("y span", 3.0e-6)
        m.set("z", 0.0); m.set("z span", 2.5e-6)
        if cfg["mesh"]:
            d = BENTMODE_MESH_NM * 1e-9
            m.set("define y mesh by", "maximum mesh step"); m.set("dy", d)
            m.set("define z mesh by", "maximum mesh step"); m.set("dz", d)
        if cfg["pml"]:
            m.set("y min bc", "PML"); m.set("y max bc", "PML")
            m.set("z min bc", "PML"); m.set("z max bc", "PML")

    _fde()
    if cfg["recreate"]:
        # Mimic the GUI sequence that worked: delete the solver and add it
        # again before solving.
        m.switchtolayout(); m.select("FDE"); m.delete()
        _fde()

    m.setanalysis("wavelength", 1500.0e-9)
    m.setanalysis("number of trial modes", N_TRIAL_MODES)
    if cfg["search"]:
        m.setanalysis("use max index", 0)
        m.setanalysis("n", BENTMODE_SEARCH_N)
    if cfg.get("calcng"):
        m.setanalysis("calculate group index", 1)
    if cfg["bent"] == 1:
        m.setanalysis("bent waveguide", 0)
    elif cfg["bent"] == 2:
        m.setanalysis("bent waveguide", 1)
        m.setanalysis("bend radius", 5.0e-6)
        m.setanalysis("bend orientation", 0.0)
    if cfg.get("dump"):
        _dump_fde_state(m, "_diag_build row 4")
    return m.findmodes()


def _diag_read(m, nfound):
    """Return (n_modes, best_neff, best_ng, best_te) for a solved FDE."""
    out = []
    for i in range(1, N_TRIAL_MODES + 1):
        nm_ = f"FDE::data::mode{i}"
        try:
            ne = complex(np.array(m.getdata(nm_, "neff")).flatten()[0])
        except Exception:                               # noqa: BLE001
            break
        vals = {}
        for key, o in (("ng", "ng"), ("TE polarization fraction", "te")):
            try:
                vals[o] = complex(
                    np.array(m.getdata(nm_, key)).flatten()[0]).real
            except Exception:                           # noqa: BLE001
                vals[o] = float("nan")
        out.append((ne, vals["ng"], vals["te"]))
    if not out:
        return 0, float("nan"), float("nan"), float("nan")
    # Distinct real parts -- the signature that separated a real solve from a
    # seed-pinned one in every GUI test.
    distinct = len({round(o[0].real, 6) for o in out})
    best = max(out, key=lambda o: o[0].real)
    return distinct, best[0].real, best[1], best[2]


def run_modediag():
    """
    Run the FDE configuration ladder from Python and report which rows
    produce a real solve. Every row below was verified to work in the MODE
    GUI on 2026-10-12; this measures whether lumapi reproduces that, instead
    of assuming it does.

    A real solve shows several DISTINCT effective indices and a finite n_g.
    A failed solve pins every mode at the search seed with n_g undefined.
    """
    lumapi = load_lumapi()
    rows = [
        ("4 mesh+pml (good)", dict(mesh=1, pml=1, search=0, bent=0,
                                   recreate=0, dump=1)),
    ]
    print("[modediag] straight guide, w = 400 nm, lambda = 1500 nm")
    print("[modediag] expect the real mode near neff 2.2966, ng 4.349")
    for hide in (True,):
        print(f"\n[modediag] ---- lumapi.MODE(hide={hide}) ----")
        print(f"[modediag] {'config':<24} {'nfound':>7} {'distinct':>9} "
              f"{'neff':>9} {'ng':>9} {'TE':>7}  verdict")
        try:
            with lumapi.MODE(hide=hide) as m:
                for name, cfg in rows:
                    try:
                        nf = _diag_build(m, cfg)
                        dist, ne, ng, te = _diag_read(m, nf)
                        ok = dist > 1 and np.isfinite(ng)
                        print(f"[modediag] {name:<24} {nf:>7} {dist:>9} "
                              f"{ne:>9.5f} {ng:>9.5f} {te:>7.3f}  "
                              f"{'OK' if ok else 'FAILED'}")
                    except Exception as exc:            # noqa: BLE001
                        print(f"[modediag] {name:<24} raised "
                              f"{type(exc).__name__}: {exc}")
        except Exception as exc:                        # noqa: BLE001
            print(f"[modediag] session with hide={hide} failed: {exc}")
    # Same session, same structure: call the production function itself.
    # If this fails where row 4 above succeeded, the difference is inside
    # _fde_mode and the two state dumps will show where.
    print("\n[modediag] ---- calling _fde_mode() in the same session ----")
    try:
        with lumapi.MODE(hide=True) as m:
            md = _fde_mode(m, None, 400.0, 1500.0, probe=True)
            print(f"[modediag] _fde_mode OK: neff = {md['neff'].real:.5f}, "
                  f"ng = {md.get('ng'):.5f}")
    except Exception as exc:                            # noqa: BLE001
        print(f"[modediag] _fde_mode FAILED: {exc}")

    print("\n[modediag] done -- send the whole output")



def run_fsrcheck():
    """
    Attribute the ring-validation FSR bias. Computes FSR four ways for each
    validated geometry and reports where each gap sits:

      1. analytic, as the dataset labels were generated   (ringvalidate.json)
      2. lambda^2/(n_g L) with the Phase-1 POLYNOMIAL n_g
      3. lambda^2/(n_g L) with the DIRECT FDE n_g         (bentmode.csv)
      4. FDTD, re-extracted here from the saved spectrum

    1 vs 2 isolates the first-order FSR relation from the label pipeline.
    2 vs 3 isolates the polynomial fit error.
    3 vs 4 is the residual: not geometry, not the fit, not bend n_g.
    """
    import json
    import glob as _glob
    vdir = "validation" if os.path.isdir("validation") else "."
    jpath = os.path.join(vdir, "ringvalidate.json")
    if not os.path.isfile(jpath):
        raise SystemExit(f"[fsrcheck] {jpath} not found")
    done = json.load(open(jpath))
    nc = json.load(open(_find("model_fits.json")))["neff"]

    def ng_poly(lam, w):
        x = (lam - 1550.0) / 50.0; y = (w - 450.0) / 50.0
        B = [1.0, x, y, x*x, x*y, y*y, x*x*y, x*y*y, y**3]
        dB = [0.0, 1.0, 0.0, 2*x, y, 0.0, 2*x*y, y*y, 0.0]
        neff = sum(c*b for c, b in zip(nc, B))
        dneff = sum(c*d for c, d in zip(nc, dB))
        return neff - lam * dneff / 50.0

    # --- FDE n_g, bilinear in (R, lambda) at the geometry's width ----------
    rows = []
    with open(_find(BENTMODE_CSV), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["R_um"] == "straight":
                continue
            try:
                rows.append((float(r["R_um"]), float(r["w_nm"]),
                             float(r["lam_nm"]), float(r["ng_solver"])))
            except (ValueError, KeyError):
                continue
    rows = np.array(rows) if rows else None

    def ng_fde(R, w, lam):
        if rows is None:
            return float("nan")
        m = np.isclose(rows[:, 1], w)
        if not m.any():                      # nearest available width
            w = rows[np.argmin(np.abs(rows[:, 1] - w)), 1]
            m = np.isclose(rows[:, 1], w)
        sub = rows[m]
        Rs = np.unique(sub[:, 0])
        lo = Rs[Rs <= R].max() if (Rs <= R).any() else Rs.min()
        hi = Rs[Rs >= R].min() if (Rs >= R).any() else Rs.max()

        def at(Rv):
            t = sub[np.isclose(sub[:, 0], Rv)]
            t = t[np.argsort(t[:, 2])]
            return float(np.interp(lam, t[:, 2], t[:, 3]))
        if hi == lo:
            return at(lo)
        f = (R - lo) / (hi - lo)
        return (1.0 - f) * at(lo) + f * at(hi)

    # --- independent FSR from the saved spectrum --------------------------
    def fsr_from_spectrum(g, lam_ref):
        pat = (f"ringval_{g['R']:.3f}_{g['w']:.1f}_{g['g']:.1f}_"
               f"{g['Lc']:.3f}_*_spectrum.npz")
        hits = sorted(_glob.glob(os.path.join("results", pat)) +
                      _glob.glob(pat))
        if not hits:
            return (float("nan"), float("nan"), 0, "", [], np.array([]),
                    float("nan"), float("nan"), float("nan"), float("nan"))
        d = np.load(hits[-1])
        lam = np.asarray(d["lam"]).ravel() * 1e9          # nm
        T = np.asarray(d["T"]).ravel()
        o = np.argsort(lam); lam, T = lam[o], T[o]
        thr = T.min() + 0.25 * (T.max() - T.min())
        pk = [i for i in range(1, len(T) - 1)
              if T[i] > T[i-1] and T[i] >= T[i+1] and T[i] > thr]
        if len(pk) < 2:
            return (float("nan"), float("nan"), len(pk),
                    os.path.basename(hits[-1]), [], np.array([]),
                    float(np.median(np.diff(lam))),
                    float(T.min()), float(T.max()), thr)
        # parabolic refinement of each peak position
        pos = []
        for i in pk:
            y0, y1, y2 = T[i-1], T[i], T[i+1]
            den = (y0 - 2*y1 + y2)
            dx = 0.5 * (y0 - y2) / den if den else 0.0
            pos.append(lam[i] + dx * (lam[i+1] - lam[i-1]) / 2.0)
        pos = sorted(pos)
        sp = np.diff(pos)
        # LOCAL FSR: the spacing of the pair bracketing lambda_res. FSR varies
        # ~4% across this band, so a band-average is a different quantity from
        # what analyse_drop records and the two must not be compared.
        loc = float("nan")
        if len(sp):
            k = int(np.argmin([abs(0.5*(pos[i]+pos[i+1]) - lam_ref)
                               for i in range(len(sp))]))
            loc = float(sp[k])
        return (loc, float(np.mean(sp)), len(pos),
                os.path.basename(hits[-1]), pos, sp,
                float(np.median(np.diff(lam))),
                float(T.min()), float(T.max()), thr)

    L = ["FSR ATTRIBUTION", "=" * 78,
         "Where the ring-validation FSR bias comes from. Each column is the",
         "same FSR computed a different way; the gaps attribute the bias.", "",
         f"  {'R':>5s} {'Lc':>5s} {'lam_res':>9s} {'1 label':>9s} "
         f"{'2 poly':>9s} {'3 FDE':>9s} {'4 FDTD':>9s} {'4 re-ex':>9s}",
         "  " + "-" * 74]
    acc = {"1v4": [], "2v3": [], "3v4": [], "1v2": []}
    diag = []
    for v in done.values():
        g = v["geom"]
        lam0 = float(v["sim"]["lambda_res_nm"])
        Lnm = (2.0 * np.pi * g["R"] + 2.0 * g["Lc"]) * 1e3
        f1 = float(v["ana"]["FSR_nm"])
        f2 = lam0 * lam0 / (ng_poly(lam0, g["w"]) * Lnm)
        f3 = lam0 * lam0 / (ng_fde(g["R"], g["w"], lam0) * Lnm)
        f4 = float(v["sim"]["FSR_nm"])
        f4b, f4m, npk, fn, pos, sp, grid, tmin, tmax, thr = \
            fsr_from_spectrum(g, lam0)
        diag.append((g, lam0, npk, v["sim"].get("n_peaks"), grid,
                     tmin, tmax, thr, pos, sp, f4, f4b, f4m, fn,
                     v.get("mesh_nm"), v.get("freq_points"),
                     v.get("grid_nm"), v["sim"].get("Q_L")))
        L.append(f"  {g['R']:5.2f} {g['Lc']:5.2f} {lam0:9.3f} "
                 f"{f1:9.4f} {f2:9.4f} {f3:9.4f} {f4:9.4f} {f4b:9.4f}")
        acc["1v2"].append(100.0 * (f1 - f2) / f2)
        acc["2v3"].append(100.0 * (f2 - f3) / f3)
        acc["3v4"].append(100.0 * (f3 - f4) / f4)
        acc["1v4"].append(100.0 * (f1 - f4) / f4)
    L += ["", "  ATTRIBUTION (% of the FDTD value unless stated)", "",
          f"  {'gap':<34s} {'mean':>8s} {'min':>8s} {'max':>8s}",
          "  " + "-" * 60]
    for k, lab in (("1v4", "total bias: label vs FDTD"),
                   ("1v2", "label pipeline vs lam^2/(n_g L)"),
                   ("2v3", "polynomial n_g vs direct FDE n_g"),
                   ("3v4", "RESIDUAL: FDE n_g model vs FDTD")):
        a_ = np.array(acc[k])
        L.append(f"  {lab:<34s} {a_.mean():+7.2f}% {a_.min():+7.2f}% "
                 f"{a_.max():+7.2f}%")
    # ---- extraction diagnostics -----------------------------------------
    L += ["", "  EXTRACTION DIAGNOSTIC -- is the re-extraction trustworthy?",
          "", "  A missed resonance inflates the mean spacing; a spurious one",
          "  deflates it. Both show up in the per-gap list below.", ""]
    # Use the run the saved spectrum belongs to (the finest frequency
    # sampling), not whichever row happens to come first: a different run has
    # a different lambda_res and therefore a different local pair.
    best_row = {}
    for d in diag:
        kk = (d[0]["R"], d[0]["w"], d[0]["g"], d[0]["Lc"])
        if kk not in best_row or (d[15] or 0) > (best_row[kk][15] or 0):
            best_row[kk] = d
    seen_g = set()
    for (g, lam0, npk, nstored, grid, tmin, tmax, thr,
         pos, sp, f4, f4b, f4m, fn, mnm, nfp, gnm, qq) in best_row.values():
        kk = (g["R"], g["w"], g["g"], g["Lc"])
        if kk in seen_g:
            continue
        seen_g.add(kk)
        L.append(f"  R={g['R']:.2f} Lc={g['Lc']:.2f}  file {fn}")
        L.append(f"    grid {grid:.4f} nm   T in [{tmin:.4g}, {tmax:.4g}]"
                 f"   threshold {thr:.4g}")
        L.append(f"    peaks found here: {npk}    peaks recorded by "
                 f"analyse_drop: {nstored}")
        if len(pos):
            L.append("    peak positions (nm): " +
                     " ".join(f"{x:.3f}" for x in pos))
        if len(sp):
            L.append("    adjacent spacings (nm): " +
                     " ".join(f"{x:.4f}" for x in sp))
            L.append(f"    spacing mean {sp.mean():.4f}  median "
                     f"{np.median(sp):.4f}  min {sp.min():.4f}  max "
                     f"{sp.max():.4f}")
            L.append(f"    stored FSR {f4:.4f}   LOCAL at lam_res "
                     f"{f4b:.4f}   band mean {f4m:.4f}")
            L.append(f"    -> local vs stored: "
                     f"{100.0*(f4b-f4)/f4:+.3f}%   (band mean vs stored: "
                     f"{100.0*(f4m-f4)/f4:+.3f}%, a different quantity)")
        L.append("")
    L += ["", "  FSR vs MESH -- the remaining candidate for the residual",
          "", "  The FDTD runs used a 40 nm mesh on a 220 nm core. Mesh",
          "  staircasing shifts the effective index, and FSR is directly",
          "  sensitive to it. If FSR moves with mesh, the residual is",
          "  numerical, as the Q_L discrepancy already proved to be.", "",
          f"  {'R':>5s} {'Lc':>5s} {'mesh nm':>8s} {'nfreq':>7s} "
          f"{'grid nm':>8s} {'lam_res':>9s} {'FSR':>9s} {'Q_L':>9s}",
          "  " + "-" * 68]
    for v in sorted(done.values(),
                    key=lambda d: (d["geom"]["Lc"], d.get("mesh_nm") or 0)):
        g = v["geom"]
        L.append(f"  {g['R']:5.2f} {g['Lc']:5.2f} "
                 f"{(v.get('mesh_nm') or float('nan')):8.1f} "
                 f"{(v.get('freq_points') or 0):7d} "
                 f"{(v.get('grid_nm') or float('nan')):8.4f} "
                 f"{v['sim']['lambda_res_nm']:9.3f} "
                 f"{v['sim']['FSR_nm']:9.4f} {v['sim']['Q_L']:9.0f}")
    L += ["",
          "  Column 4 re-ex is now the LOCAL spacing at lam_res, the same",
          "  by peak finding. If it disagrees with column 4, the extraction",
          "  itself is in question; if it agrees, the residual is physical or",
          "  lies in the coupled-mode assembly."]
    txt = "\n".join(L)
    print("\n" + txt)
    resdir = "results" if os.path.isdir("results") else "."
    open(os.path.join(resdir, "fsr_attribution.txt"), "w").write(txt + "\n")
    print(f"\n[fsrcheck] wrote {resdir}/fsr_attribution.txt")



def run_matcheck():
    """
    Compare the silicon and oxide index models the two solvers actually use.

    The ring validation shows lambda_res correct to 0.10% but FSR off by ~2%:
    n_eff is right and dn_eff/dlambda is not. FDTD fits the Palik data with a
    multi-coefficient model over the whole band; FDE samples the tabulated
    index per wavelength. Two fits can agree on n and disagree on dn/dlambda,
    which is this signature exactly.

    Accessor names differ between installs, so each is tried and the one that
    worked is reported. Nothing is assumed.
    """
    lumapi = load_lumapi()
    lam = np.linspace(1500.0, 1600.0, 11)
    f = C_LIGHT / (lam * 1e-9)

    def sample(sess, tag):
        out = {}
        for mat in (MAT_CORE, MAT_CLAD):
            got, how = None, None
            for name, call in (
                ("getfdtdindex",
                 lambda: sess.getfdtdindex(mat, f, float(f.min()),
                                           float(f.max()))),
                ("getindex", lambda: sess.getindex(mat, f)),
                ("getmaterial(nk)",
                 lambda: sess.getmaterial(mat, "refractive index")),
            ):
                try:
                    v = np.array(call()).flatten()
                    if v.size >= lam.size:
                        got, how = v[:lam.size].real, name
                        break
                except Exception:                       # noqa: BLE001
                    continue
            out[mat] = (got, how)
            print(f"[matcheck] {tag:5s} {mat:24s} via "
                  f"{how if how else 'NO ACCESSOR WORKED'}")
        return out

    res = {}
    for tag, opener in (("FDTD", lambda: lumapi.FDTD(hide=True)),
                        ("MODE", lambda: lumapi.MODE(hide=True))):
        try:
            with opener() as sess:
                res[tag] = sample(sess, tag)
        except Exception as exc:                        # noqa: BLE001
            print(f"[matcheck] {tag} session failed: {exc}")
            res[tag] = {}

    L = ["MATERIAL INDEX MODELS -- FDTD vs FDE", "=" * 78,
         "lambda_res matches to 0.10% while FSR is off ~2%, i.e. n_eff is",
         "right and dn_eff/dlambda is not. If the two solvers disagree on",
         "dn/dlambda for silicon, that is the cause.", ""]
    for mat in (MAT_CORE, MAT_CLAD):
        a = res.get("FDTD", {}).get(mat, (None, None))[0]
        b = res.get("MODE", {}).get(mat, (None, None))[0]
        L += [f"  {mat}", f"  {'lam nm':>8s} {'FDTD n':>10s} {'FDE n':>10s} "
              f"{'diff':>10s} {'diff %':>9s}", "  " + "-" * 52]
        if a is None or b is None:
            L.append("    not available from one or both solvers")
            L.append("")
            continue
        for i, l in enumerate(lam):
            L.append(f"  {l:8.1f} {a[i]:10.5f} {b[i]:10.5f} "
                     f"{a[i]-b[i]:10.5f} {100*(a[i]-b[i])/b[i]:8.3f}%")
        sa = np.polyfit(lam, a, 1)[0]
        sb = np.polyfit(lam, b, 1)[0]
        L += ["", f"    dn/dlambda   FDTD {sa:.3e}   FDE {sb:.3e}   "
              f"difference {100*(sa-sb)/sb:+.2f}%",
              "    (n_g = n_eff - lambda dn_eff/dlambda, so a slope",
              "     difference here propagates straight into FSR)", ""]
    txt = "\n".join(L)
    print("\n" + txt)
    resdir = "results" if os.path.isdir("results") else "."
    open(os.path.join(resdir, "material_check.txt"), "w").write(txt + "\n")
    print(f"\n[matcheck] wrote {resdir}/material_check.txt")



def _fde_fixed_index(m, w_nm, lam_nm, n_core, n_clad):
    """
    One FDE solve with BOTH materials replaced by fixed indices. Used only to
    measure how n_eff responds to an index change; the absolute n_eff from a
    dispersionless solve is not used for anything else.
    """
    d = BENTMODE_MESH_NM * 1e-9

    def build():
        m.switchtolayout(); m.deleteall()
        for nm_, ys, zs, idx, order in (
                ("cladding", 6e-6, 6e-6, n_clad, 3),
                ("wg", w_nm * 1e-9, H_CORE, n_core, 2)):
            m.addrect(); m.set("name", nm_)
            m.set("x", 0.0); m.set("x span", 6e-6)
            m.set("y", 0.0); m.set("y span", ys)
            m.set("z", 0.0); m.set("z span", zs)
            m.set("material", "<Object defined dielectric>")
            m.set("index", idx)
            m.set("override mesh order from material database", True)
            m.set("mesh order", order)
        m.addfde()
        m.set("solver type", "2D X normal")
        m.set("x", 0.0)
        m.set("y", 0.0); m.set("y span", 3.0e-6)
        m.set("z", 0.0); m.set("z span", 2.5e-6)
        m.set("define y mesh by", "maximum mesh step"); m.set("dy", d)
        m.set("define z mesh by", "maximum mesh step"); m.set("dz", d)
        m.set("y min bc", "PML"); m.set("y max bc", "PML")
        m.set("z min bc", "PML"); m.set("z max bc", "PML")
        m.setanalysis("wavelength", lam_nm * 1e-9)
        m.setanalysis("number of trial modes", N_TRIAL_MODES)
        return m.findmodes()

    # same verify-and-rebuild loop the sweep uses
    for attempt in range(1, RINGVAL_FDE_ATTEMPTS + 1):
        build()
        best = None
        for i in range(1, N_TRIAL_MODES + 1):
            nm0 = f"FDE::data::mode{i}"
            try:
                ne = complex(np.array(m.getdata(nm0, "neff")).flatten()[0])
            except Exception:                           # noqa: BLE001
                break
            try:
                te = complex(np.array(m.getdata(
                    nm0, "TE polarization fraction")).flatten()[0]).real
            except Exception:                           # noqa: BLE001
                te = float("nan")
            # Use the sweep's own guided-mode bounds, not the local core
            # index: with the core perturbed upward a bulk-silicon mode at
            # 3.47656 is strictly below n_core + dn and passes, which is what
            # made the w = 400 and 425 rows disagree by 1.25 in n_eff.
            if (BENTMODE_NEFF_MIN < ne.real < BENTMODE_NEFF_MAX
                    and (te != te or te >= BENTMODE_TE_MIN)
                    and (best is None or ne.real > best.real)):
                best = ne
        if best is not None:
            return best.real
        if attempt < RINGVAL_FDE_ATTEMPTS:
            m.switchtolayout()
            try:
                m.select("FDE"); m.delete()
            except Exception:                           # noqa: BLE001
                pass
    raise RuntimeError(f"no guided mode at w={w_nm} nm with fixed indices "
                       f"n_core={n_core}, n_clad={n_clad}")


def run_thermal():
    """
    Phase C1. Resonance tuning sensitivity dlambda_res/dT per geometry.

    Measures, from the solver, how n_eff responds to an index change in the
    core and in the cladding separately:

        Gamma_core = dn_eff/dn_core,  Gamma_clad = dn_eff/dn_clad

    then combines them with the literature dn/dT of each material:

        dn_eff/dT = Gamma_core * (dn/dT)_Si + Gamma_clad * (dn/dT)_SiO2
        dlambda_res/dT = lambda * (dn_eff/dT) / n_g

    n_g comes from bentmode.csv, i.e. from the solver, not from a fit.
    """
    lumapi = load_lumapi()
    lam = THERMAL_LAM_NM

    ng_by_w = {}
    try:
        with open(_find(BENTMODE_CSV), newline="") as fh:
            for r in csv.DictReader(fh):
                if r["R_um"] != "straight":
                    continue
                if abs(float(r["lam_nm"]) - lam) < 1e-6:
                    ng_by_w[float(r["w_nm"])] = float(r["ng_solver"])
    except Exception as exc:                            # noqa: BLE001
        print(f"[thermal] could not read {BENTMODE_CSV}: {exc}")

    n_si = 3.47656        # index at 1550 nm as both solvers report it
    n_ox = 1.44402
    rows = []
    with lumapi.MODE(hide=True) as m:
        for w in MODE_W_NM:
            # Four solves per width. The first three give the two
            # sensitivities; the fourth perturbs BOTH materials and tests
            # additivity, which is a measured check rather than an assumed
            # one. A mode-selection slip in any single solve shows up as a
            # wild Gamma, so the four n_eff values are also required to agree
            # to within the perturbation itself before the row is accepted.
            ok, n0 = False, float("nan")
            for attempt in range(1, 4):
                n0 = _fde_fixed_index(m, w, lam, n_si, n_ox)
                nc = _fde_fixed_index(m, w, lam, n_si + THERMAL_DN, n_ox)
                nl = _fde_fixed_index(m, w, lam, n_si, n_ox + THERMAL_DN)
                nb = _fde_fixed_index(m, w, lam,
                                      n_si + THERMAL_DN, n_ox + THERMAL_DN)
                sp = max(nc, nl, nb, n0) - min(nc, nl, nb, n0)
                if sp < 10.0 * THERMAL_DN:
                    ok = True
                    break
                print(f"[thermal] w={w:5.0f} nm: the four solves disagree by "
                      f"{sp:.4f} in n_eff, far more than the {THERMAL_DN:.1e} "
                      f"perturbation -- a mode-selection slip. Retrying "
                      f"({attempt + 1}).")
            g_core = (nc - n0) / THERMAL_DN
            g_clad = (nl - n0) / THERMAL_DN
            g_both = (nb - n0) / THERMAL_DN
            add_err = (100.0 * (g_core + g_clad - g_both) / g_both
                       if g_both else float("nan"))
            dneff_dT = g_core * DNDT_SI + g_clad * DNDT_SIO2
            ng = ng_by_w.get(w, float("nan"))
            dlam_dT = lam * dneff_dT / ng if ng == ng else float("nan")
            rows.append((w, n0, g_core, g_clad, g_both, add_err,
                         dneff_dT, ng, dlam_dT, ok))
            print(f"[thermal] w={w:5.0f} nm  G_core {g_core:.4f}  "
                  f"G_clad {g_clad:.4f}  G_both {g_both:.4f}  "
                  f"additivity {add_err:+.2f}%  dlam/dT {dlam_dT:.4f} nm/K"
                  f"{'' if ok else '   <-- UNRELIABLE'}")

    L = ["THERMO-OPTIC TUNING SENSITIVITY (Phase C1)", "=" * 78,
         "Gamma = dn_eff/dn_material, solved for directly by perturbing each",
         "material index and re-solving. dn/dT are literature material",
         "constants, the same class of input as the Palik data used",
         "throughout; everything geometry dependent is solved for.", "",
         f"  lambda {lam:.0f} nm,  dn/dT: Si {DNDT_SI:.3e}/K, "
         f"SiO2 {DNDT_SIO2:.3e}/K,  perturbation {THERMAL_DN:.1e}", "",
         f"  {'w nm':>6} {'n_eff':>8} {'G_core':>8} {'G_clad':>8} "
         f"{'G_both':>8} {'add err':>8} {'n_g':>8} {'dlam/dT':>9} {'ok':>4}",
         "  " + "-" * 80]
    for w, n0, gc, gl, gb, ae, dn, ng, dl, ok in rows:
        L.append(f"  {w:6.0f} {n0:8.5f} {gc:8.4f} {gl:8.4f} {gb:8.4f} "
                 f"{ae:+7.2f}% {ng:8.4f} {dl:9.4f} {'yes' if ok else 'NO':>4}")
    good = [r for r in rows if r[9]]
    L += ["", "  CHECKS, each measured rather than assumed:",
          "  - additivity: G_core + G_clad against G_both, the two materials",
          "    perturbed together. Agreement means the response is linear and",
          "    the two sensitivities can be combined.",
          "  - a row marked NO had its four solves land on different modes and",
          "    must not be used."]
    if good:
        gb_ = np.array([r[4] for r in good])
        ne_ = np.array([r[1] for r in good])
        ng_ = np.array([r[7] for r in good])
        dl_ = np.array([r[8] for r in good])
        L += ["", f"  G_both spans {gb_.min():.3f} to {gb_.max():.3f}; "
              f"n_g/n_eff spans {(ng_/ne_).min():.3f} to "
              f"{(ng_/ne_).max():.3f}.",
              "  Perturbation theory gives dn_eff/dn_i = Gamma_i^power *",
              "  n_g/n_eff, so G_both below n_g/n_eff indicates mode energy",
              "  outside the two perturbed rectangles (the PML region). The",
              "  cladding term contributes ~1.5% of dn_eff/dT, so this",
              "  affects dlambda/dT only marginally.", "",
              f"  dlambda/dT = {dl_.mean():.4f} nm/K "
              f"(spread {dl_.min():.4f} to {dl_.max():.4f}) over "
              f"{len(good)} usable widths.",
              "  Published silicon microring tuning is ~0.07-0.08 nm/K, so",
              "  this is consistent with the literature it was not fitted to.",
              "", "  NEXT (C2): required dT per ring = lambda_res error /",
              "  (dlambda/dT), using the closed-loop errors already measured."]
    txt = "\n".join(L)
    print("\n" + txt)
    resdir = "results" if os.path.isdir("results") else "."
    open(os.path.join(resdir, "thermal_c1.txt"), "w").write(txt + "\n")
    import json as _j
    _j.dump({"lambda_nm": lam, "dndt_si": DNDT_SI, "dndt_sio2": DNDT_SIO2,
             "rows": [{"w_nm": r[0], "n_eff": r[1], "g_core": r[2],
                       "g_clad": r[3], "g_both": r[4], "add_err_pct": r[5],
                       "dneff_dT": r[6], "ng": r[7], "dlam_dT_nm_per_K": r[8],
                       "ok": bool(r[9])} for r in rows]},
            open(os.path.join(resdir, THERMAL_JSON), "w"), indent=1)
    print(f"\n[thermal] wrote {resdir}/thermal_c1.txt and {THERMAL_JSON}")



def run_trimbudget():
    """
    Phase C2/C3. How much thermal trimming each inverse-designed ring needs
    to land inside its channel window, and what that costs in power.

    Inputs, all already measured:
      - dlambda/dT per width, from --thermal (thermal_c1.json)
      - the closed-loop re-simulated lambda_res for each model, from the
        3-D FDTD coupler runs (closedloop_*.json)
      - the requested target lambda_res for those same test points

    The kelvin column depends only on measured quantities. The power column
    additionally uses HEATER_K_PER_MW, a device parameter from the
    literature, and scales linearly with it.
    """
    import json
    resdir = "results" if os.path.isdir("results") else "."
    tj = os.path.join(resdir, THERMAL_JSON)
    if not os.path.isfile(tj):
        raise SystemExit(f"[trim] {tj} not found -- run --thermal first")
    tc = json.load(open(tj))
    good = [r for r in tc["rows"] if r["ok"]]
    if not good:
        raise SystemExit("[trim] no usable widths in thermal_c1.json")
    tw = np.array([r["w_nm"] for r in good])
    td = np.array([r["dlam_dT_nm_per_K"] for r in good])
    order = np.argsort(tw); tw, td = tw[order], td[order]

    def dlam_dT(w):
        return float(np.interp(w, tw, td))

    _, _, _ = None, None, None
    Gr_te, Rr_te = _load_split_raw("test")
    i_lam = TARGETS.index("lambda_res_nm")

    methods = []
    for fn in sorted(os.listdir(resdir)):
        if fn.startswith("closedloop_") and fn.endswith(".json"):
            methods.append(fn[len("closedloop_"):-len(".json")])
    if not methods:
        raise SystemExit("[trim] no closedloop_*.json in results/")

    L = ["THERMAL TRIMMING BUDGET (Phase C2/C3)", "=" * 78,
         "Resonance placement is the binding constraint for CPO: the O7",
         "scenarios fail almost entirely on lambda_res. This asks what it",
         "costs to fix that with a heater instead of with a better model.",
         "",
         f"  dlambda/dT = {td.min():.4f} to {td.max():.4f} nm/K over "
         f"w = {tw.min():.0f}-{tw.max():.0f} nm (measured, --thermal)",
         f"  heater thermal resistance {HEATER_K_PER_MW:g} K/mW "
         "(LITERATURE device parameter; scales the power column only)",
         "",
         "  Scenario A window +/-2.0 nm (4 nm channel grid),",
         "  Scenario B window +/-0.4 nm (0.8 nm ITU grid).", ""]
    L += ["  A heater shifts a resonance RED only. 'biased' assumes each ring",
          "  is designed deliberately blue of its channel and heated up, so",
          "  the error SPREAD sets the requirement; 'unbiased' assumes no such",
          "  bias, so a ring sitting red of its channel must be heated nearly",
          "  a full FSR to bring the next resonance onto the channel.", "",
          f"  {'method':>10} {'n':>4} {'RMS err':>9} {'spread':>9} "
          f"{'mean dT':>8} {'max dT':>8} {'max mW':>8} "
          f"{'unb mean':>9} {'unb max':>8} {'in A':>6} {'in B':>6}",
          "  " + (" " * 34) + "<------ biased ------>  <-- unbiased -->",
          "  " + "-" * 100]
    store = {}
    for meth in methods:
        rec = json.load(open(os.path.join(resdir, f"closedloop_{meth}.json")))
        errs, sl, fs = [], [], []
        for k, v in rec.items():
            if v.get("status") != "ok":
                continue
            try:
                tgt = float(Rr_te[int(k)][i_lam])
                sim = float(v["resim"]["lambda_res_nm"])
                w = float(v["geom"][1])
            except Exception:                           # noqa: BLE001
                continue
            e = sim - tgt
            errs.append(e)
            sl.append(dlam_dT(w))
            # the ring's own FSR sets how far "one resonance along" is
            try:
                fs.append(float(v["resim"]["FSR_nm"]))
            except Exception:                           # noqa: BLE001
                fs.append(float("nan"))
        if not errs:
            L.append(f"  {meth:>10}    --  no usable closed-loop points")
            continue
        e = np.array(errs); sens = np.array(sl)
        rms_e = float(np.sqrt(np.mean(e ** 2)))
        # HEAT-ONLY PHYSICS, done properly. A heater moves a resonance red
        # only, AND the resonance comb repeats every FSR, so ring i needs a
        # shift of (delta - e_i) mod FSR_i for a design bias delta. The
        # requirement is therefore bounded by one FSR however bad the model
        # is -- you use the next resonance along. Choosing delta to minimise
        # the worst case is a covering problem on a circle; it is solved here
        # by scanning delta, which is exact to the grid and needs no
        # assumption about the error distribution.
        #
        # An earlier version used (e_i - e_min)/sens, which ignores the wrap
        # and produced a "biased" requirement LARGER than the unbiased one --
        # impossible, and the reason this is now computed rather than argued.
        fsr_nm = np.array(fs)
        fsr_nm = np.where(np.isfinite(fsr_nm), fsr_nm, np.nanmedian(fsr_nm))
        grid = np.linspace(0.0, float(np.nanmax(fsr_nm)), 4001)
        shifts = np.mod(grid[:, None] - e[None, :], fsr_nm[None, :])
        dts = shifts / sens[None, :]
        worst = dts.max(axis=1)
        j = int(np.argmin(worst))
        delta_opt = float(grid[j])
        dt_b = dts[j]
        dt_u = np.mod(-e, fsr_nm) / sens
        mw_b = dt_b / HEATER_K_PER_MW
        inA = 100.0 * np.mean(np.abs(e) <= 2.0)
        inB = 100.0 * np.mean(np.abs(e) <= 0.4)
        store[meth] = dict(n=len(e), rms_err=rms_e,
                           spread_nm=float(e.max() - e.min()),
                           delta_opt_nm=float(np.mod(
                               delta_opt, float(np.nanmean(fsr_nm)))),
                           fsr_mean_nm=float(np.nanmean(fsr_nm)),
                           max_dt_biased=float(dt_b.max()),
                           mean_dt_biased=float(dt_b.mean()),
                           max_mw_biased=float(mw_b.max()),
                           mean_mw_biased=float(mw_b.mean()),
                           mean_dt_unbiased=float(dt_u.mean()),
                           max_dt_unbiased=float(dt_u.max()),
                           inA=inA, inB=inB)
        L.append(f"  {meth:>10} {len(e):4d} {rms_e:8.3f}n "
                 f"{e.max()-e.min():8.3f}n {dt_b.mean():7.1f}K "
                 f"{dt_b.max():7.1f}K {mw_b.max():7.2f} "
                 f"{dt_u.mean():7.1f}K {dt_u.max():7.1f}K "
                 f"{inA:5.0f}% {inB:5.0f}%")
    L += ["", "  'in A' / 'in B' are the fraction of rings already inside the",
          "  window WITHOUT trimming. The dT columns are what the rest need.",
          "",
          "  INTERPRETATION", "",
          "  A heater must hold its ring at the trimmed temperature for as",
          "  long as the link is powered, so this is static power per ring,",
          "  paid continuously, multiplied by the channel count.", ""]
    if store:
        # Compare only models evaluated on the SAME number of targets. A
        # model whose unrealisable predictions were dropped is scored on the
        # subset it happened to succeed on, which is selection after the
        # fact, not a fairer test. tandem realises 33.5% of its predictions
        # and would otherwise win on 8 self-selected points.
        nmax = max(v["n"] for v in store.values())
        full = {m: v for m, v in store.items() if v["n"] == nmax}
        partial = sorted(m for m in store if m not in full)
        best = min(full, key=lambda m: full[m]["max_dt_biased"])
        b = store[best]
        if partial:
            plist = ", ".join(
                "%s (n=%d)" % (m, store[m]["n"]) for m in partial)
            L += [f"  Ranked over the {len(full)} model(s) evaluated on all "
                  f"{nmax} targets. Excluded from the ranking: "
                  f"{plist} -- scored on fewer points because",
                  "  unrealisable predictions were dropped -- that is a",
                  "  self-selected subset, not an easier-won result.", ""]
        L += [f"  Best model by trimming demand: {best} "
              f"(optimal design bias {b['delta_opt_nm']:+.3f} nm, "
              f"FSR {b['fsr_mean_nm']:.2f} nm).",
              f"  Biased design: heater must span {b['max_dt_biased']:.1f} K "
              f"({b['max_mw_biased']:.1f} mW at "
              f"{HEATER_K_PER_MW:g} K/mW); mean draw across rings "
              f"{b['mean_mw_biased']:.1f} mW.",
              f"  Unbiased: {b['mean_dt_unbiased']:.0f} K on average, "
              f"{b['max_dt_unbiased']:.0f} K worst -- which is why the bias",
              "  is not optional.", ""]
        for nch, scen in ((4, "A"), (8, "B")):
            L.append(f"  Scenario {scen}: {nch} rings x "
                     f"{b['mean_mw_biased']:.2f} mW mean = "
                     f"{nch * b['mean_mw_biased']:.1f} mW static "
                     f"(worst case {nch * b['max_mw_biased']:.1f} mW).")
        L += ["",
              "  The heater must be SIZED for the max and will typically",
              "  DRAW the mean. Both are quoted because they answer different",
              "  questions: what to build, and what it costs to run.",
              "",
              "  Note what this rewards: the figure of merit is how tightly the",
              "  lambda_res errors CLUSTER once wrapped into one FSR, not the",
              "  RMS error. A large but consistent offset is absorbed by the",
              "  design bias and costs nothing; scatter is what costs power.",
              "",
              "  The requirement is capped at one FSR by construction: a ring",
              "  that is further off than that is tuned onto the next",
              "  resonance instead. So no model, however poor at placing",
              f"  lambda_res, can demand more than ~{float(np.nanmax(fsr_nm))/np.mean(sens):.0f} K here."]
    txt = "\n".join(L)
    print("\n" + txt)
    open(os.path.join(resdir, "trim_budget.txt"), "w").write(txt + "\n")
    json.dump(store, open(os.path.join(resdir, "trim_budget.json"), "w"),
              indent=1)
    print(f"\n[trim] wrote {resdir}/trim_budget.txt")



LABELSENS_SCALES = [0.97, 1.00, 1.03]


def run_labelsens():
    """
    Label-fidelity sensitivity for O6.

    The dataset's FSR labels come from the coupled-mode assembly, which the
    full 3-D ring validation measured to disagree with FDTD by 2.91% in group
    index (section 10b). FSR = lambda^2/(n_g L), so the labels inherit that.
    A referee is entitled to ask what the O6 verdict means if the labels are
    systematically off by ~3%.

    This answers it directly: rescale the FSR labels by +/-3%, retrain, and
    report whether the verdict changes. O6's geometry criterion counts
    parameters reaching R^2 >= 0.80; its response criterion asks whether the
    requested response is reproduced. If the degeneracy is structural -- four
    targets collapsing to two effective constraints -- a uniform shift in one
    label cannot create or remove it, and the counts will not move.

    Tree models only. Retraining the networks per scale would add hours
    without changing what is being tested, and the response column here is a
    SURROGATE proxy, not the 3-D FDTD closed loop -- it compares predicted
    geometry pushed back through a forward model, so it is a consistency
    check, not an independent verification.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
    except Exception as exc:                            # noqa: BLE001
        raise SystemExit(f"[labelsens] scikit-learn required: {exc}")

    Xtr, Ytr = _load_split_raw("train")
    Xte, Yte = _load_split_raw("test")
    j_fsr = TARGETS.index("FSR_nm")

    L = ["LABEL-FIDELITY SENSITIVITY OF O6", "=" * 78,
         "The FSR labels come from the coupled-mode assembly, which the 3-D",
         "ring validation found to disagree with FDTD by 2.91% in n_g. This",
         "asks whether the O6 verdict survives a systematic shift of that",
         "size in the labels.", "",
         "Tree models only; the response column is a forward-surrogate proxy,",
         "not the 3-D FDTD closed loop.", "",
         f"  {'FSR x':>7} {'RMS-RGE':>9} " +
         " ".join(f"{'R2 ' + k.split('_')[0]:>9}" for k in INPUTS) +
         f" {'n>=0.80':>8} {'resp MAPE':>10}",
         "  " + "-" * 78]
    out = {}
    for sc in LABELSENS_SCALES:
        # TRAIN on biased labels, TEST against true targets. Scaling both
        # sides is invariant -- trees split on thresholds that rescale with
        # the feature, and standardisation absorbs a uniform factor, so no
        # model in this pipeline can respond to it. An earlier version did
        # exactly that and reported identical numbers to four decimals as
        # though they were evidence of robustness. The mismatch between the
        # labels a model was trained on and the targets a user requests is
        # what a biased label actually costs.
        ytr = Ytr.copy()
        ytr[:, j_fsr] *= sc
        yte = Yte            # true targets, unscaled
        inv = RandomForestRegressor(n_estimators=400, random_state=42,
                                    n_jobs=-1).fit(ytr, Xtr)
        P = inv.predict(yte)
        agg, per, p12 = _rms_rge(P, Xte)
        r2 = [float(r2_score(Xte[:, j], P[:, j])) for j in range(len(INPUTS))]
        n80 = sum(1 for v in r2 if v >= 0.80)
        # forward model on TRUE labels: it stands in for the real device
        fwd = RandomForestRegressor(n_estimators=400, random_state=42,
                                    n_jobs=-1).fit(Xtr, Ytr)
        Rp = fwd.predict(P)
        den = np.where(np.abs(yte) > 1e-12, np.abs(yte), np.nan)
        mape = float(np.nanmean(np.abs(Rp - yte) / den) * 100.0)
        out[f"{sc:.2f}"] = dict(rms_rge=agg, r2=r2, n80=n80, resp_mape=mape)
        L.append(f"  {sc:7.2f} {agg:8.3f}% " +
                 " ".join(f"{v:9.4f}" for v in r2) +
                 f" {n80:8d} {mape:9.3f}%")

    base = out["1.00"]
    L += ["", "  VERDICT", ""]
    same = all(v["n80"] == base["n80"] for v in out.values())
    L.append("  O6 geometry criterion (parameters reaching R^2 >= 0.80): "
             + ("UNCHANGED at %d of %d under +/-3%% label shift."
                % (base["n80"], len(INPUTS)) if same
                else "CHANGES under a +/-3% label shift -- the verdict is "
                     "NOT robust and this must be reported."))
    sp = max(v["rms_rge"] for v in out.values()) - \
        min(v["rms_rge"] for v in out.values())
    L.append(f"  RMS-RGE moves by {sp:.3f} percentage points across the "
             f"+/-3% range.")
    rp = max(v["resp_mape"] for v in out.values()) - \
        min(v["resp_mape"] for v in out.values())
    L.append(f"  Surrogate response MAPE moves by {rp:.3f} points.")
    L += ["",
          "  Read the 1.00 row as the baseline: it is the pipeline as",
          "  published. The 0.97 and 1.03 rows show what a 3% label bias",
          "  costs when the model is then asked for true targets.", "",
          "  A uniform scaling of one label is the SIMPLEST form the label",
          "  error could take. It is not a proof of robustness against a",
          "  geometry-dependent error, which is what a 2.91% n_g deficit",
          "  with a mild wavelength trend actually is. Stated as a bound on",
          "  what was tested, not as a general claim."]
    txt = "\n".join(L)
    print("\n" + txt)
    resdir = "results" if os.path.isdir("results") else "."
    open(os.path.join(resdir, "label_sensitivity.txt"), "w").write(txt + "\n")
    import json as _j
    _j.dump(out, open(os.path.join(resdir, "label_sensitivity.json"), "w"),
            indent=1)
    print(f"\n[labelsens] wrote {resdir}/label_sensitivity.txt")



SEEDCURVE_SEEDS = [42, 43, 44, 45, 46]


def run_seedcurve():
    """
    Sample-efficiency curve repeated over seeds.

    Phase 6.3 reports the curve at a single seed, which cannot distinguish a
    real ordering between methods from initialisation noise. This repeats the
    whole curve over SEEDCURVE_SEEDS and reports mean +/- sd at every size,
    so the Phase 6 conclusions can be checked against their own error bars.

    Resumable: results are checkpointed per (seed, size).
    """
    import json
    torch = _torch()
    outdir = "data" if os.path.isdir("data") else "."
    resdir = "results" if os.path.isdir("results") else "."
    sc = json.load(open(os.path.join(outdir, "scaler.json")))["scaler"]
    p5 = json.load(open(os.path.join(resdir, "phase5_results.json")))
    fck = torch.load(os.path.join(resdir, "phase3_forward_model.pt"),
                     weights_only=False)

    Gn_tr, Rn_tr, _ = _load_split("train")
    Gn_va, Rn_va, _ = _load_split("val")
    Gn_te, Rn_te, _ = _load_split("test")
    Gr_te, Rr_te = _load_split_raw("test")
    sizes = [n if n else len(Gn_tr) for n in SAMPLE_SIZES]

    ck_path = os.path.join(resdir, "seedcurve.json")
    done = json.load(open(ck_path)) if os.path.isfile(ck_path) else {}
    todo = [(sd, n) for sd in SEEDCURVE_SEEDS for n in sizes
            if f"{sd}|{n}" not in done]
    print(f"[seedcurve] {len(SEEDCURVE_SEEDS)} seeds x {len(sizes)} sizes, "
          f"{len(todo)} to run")

    t0 = time.time()
    for c, (sd, n) in enumerate(todo, 1):
        print(f"[seedcurve] seed {sd}, n = {n}  ({c}/{len(todo)})")
        rec = {}
        torch.manual_seed(sd)
        fnet, _ = _build_mlp(torch, fck["depth"], fck["width"])
        _train_mlp(torch, fnet, Gn_tr[:n], Rn_tr[:n], Gn_va, Rn_va,
                   fck["lr"], fck["wd"])
        fnet.eval()
        with torch.no_grad():
            P = _res_physical(fnet(torch.from_numpy(Gn_te)).numpy(), sc)
        rec["DNN forward"] = float(np.mean(
            [_metrics(Rr_te[:, j], P[:, j])[1] for j in range(4)]))
        for prm in fnet.parameters():
            prm.requires_grad_(False)

        gn = _build_inverse(torch, sd)
        _train_inverse(torch, gn, Gn_tr[:n], Rn_tr[:n], Gn_va, Rn_va,
                       forward=None, tag=f"s{sd}-n{n}-naive")
        gn.eval()
        with torch.no_grad():
            rec["naive DNN inverse"] = _rms_rge(
                _geo_physical(gn(torch.from_numpy(Rn_te)).numpy(), sc),
                Gr_te)[0]

        gt = _build_inverse(torch, sd + 1)
        _train_inverse(torch, gt, Gn_tr[:n], Rn_tr[:n], Gn_va, Rn_va,
                       forward=fnet, tag=f"s{sd}-n{n}-tandem")
        gt.eval()
        with torch.no_grad():
            rec["Tandem inverse"] = _rms_rge(
                _geo_physical(gt(torch.from_numpy(Rn_te)).numpy(), sc),
                Gr_te)[0]

        for kind in ("rf", "xgb"):
            pref = "estimator__" if kind == "xgb" else ""
            for direction, Xa, Ya in (("forward", Gn_tr, Rn_tr),
                                      ("inverse", Rn_tr, Gn_tr)):
                bp = {k.replace("estimator__", ""): v for k, v
                      in p5[f"{direction}_{kind}_norm"]["best"].items()}
                bp = {f"{pref}{k}": v for k, v in bp.items()}
                # the seed must actually reach the estimator; _fit_sized
                # accepts a seed argument but does not apply it
                bp[f"{pref}random_state"] = int(sd)
                e = _fit_sized(kind, direction, n, Xa, Ya, bp)
                if direction == "forward":
                    Pf = _res_physical(e.predict(Gn_te), sc)
                    rec[f"{kind.upper()} forward"] = float(np.mean(
                        [_metrics(Rr_te[:, j], Pf[:, j])[1]
                         for j in range(4)]))
                else:
                    rec[f"{kind.upper()} inverse"] = _rms_rge(
                        _geo_physical(e.predict(Rn_te), sc), Gr_te)[0]

        done[f"{sd}|{n}"] = rec
        json.dump(done, open(ck_path, "w"), indent=1)
        el = (time.time() - t0) / 60.0
        print(f"[seedcurve]   done ({el:.1f} min elapsed, "
              f"{el / c * (len(todo) - c):.1f} min left)")

    fwd_keys = ["DNN forward", "RF forward", "XGB forward"]
    inv_keys = ["naive DNN inverse", "Tandem inverse", "RF inverse",
                "XGB inverse"]

    def stat(k, n):
        v = [done[f"{sd}|{n}"][k] for sd in SEEDCURVE_SEEDS
             if f"{sd}|{n}" in done and k in done[f"{sd}|{n}"]]
        return (float(np.mean(v)), float(np.std(v)), len(v)) if v else \
            (float("nan"), float("nan"), 0)

    L = ["SAMPLE-EFFICIENCY CURVE OVER SEEDS", "=" * 92,
         f"Seeds {SEEDCURVE_SEEDS}. Phase 6.3 reports this curve at one seed,",
         "which cannot separate a real ordering from initialisation noise.",
         "Each cell is mean +/- sd over the seeds.", ""]
    for title, keys, unit in (("FORWARD -- aggregate test MAPE (%)",
                               fwd_keys, "%"),
                              ("INVERSE -- RMS-RGE (%)", inv_keys, "%")):
        L += ["", f"  {title}",
              "  " + f"{'method':<20}" +
              "".join(f"{('n=' + str(n)):>18}" for n in sizes)]
        for k in keys:
            cells = []
            for n in sizes:
                m, sd_, cnt = stat(k, n)
                cells.append(f"{m:9.3f}+-{sd_:<6.3f}")
            L.append("  " + f"{k:<20}" + "".join(f"{c:>18}" for c in cells))

    L += ["", "  DOES THE SINGLE-SEED CONCLUSION SURVIVE?", ""]
    small = [n for n in sizes if n <= 300]
    for label, keys, where in (("forward, n <= 300", fwd_keys, small),
                               ("forward, full size", fwd_keys, [sizes[-1]]),
                               ("inverse, n <= 300", inv_keys, small),
                               ("inverse, full size", inv_keys, [sizes[-1]])):
        if not where:
            continue
        sc_ = {k: np.mean([stat(k, n)[0] for n in where]) for k in keys}
        sd_ = {k: np.mean([stat(k, n)[1] for n in where]) for k in keys}
        rank = sorted(sc_, key=lambda k: sc_[k])
        a, b_ = rank[0], rank[1]
        gap = sc_[b_] - sc_[a]
        pooled = float(np.sqrt(sd_[a] ** 2 + sd_[b_] ** 2))
        verdict = ("SEPARATED" if gap > pooled else
                   "NOT separated -- within seed noise")
        L.append(f"  {label:<20} best {a} ({sc_[a]:.3f}), "
                 f"runner-up {b_} ({sc_[b_]:.3f});")
        L.append(f"  {'':<20} gap {gap:.3f} vs pooled sd {pooled:.3f} "
                 f"-> {verdict}")
    L += ["", "  A gap smaller than the pooled seed spread is not evidence of",
          "  an ordering. Where that is the case the single-seed Phase 6",
          "  statement must be softened to 'comparable'."]
    txt = "\n".join(L)
    print("\n" + txt)
    open(os.path.join(resdir, "seedcurve.txt"), "w").write(txt + "\n")
    print(f"\n[seedcurve] wrote {resdir}/seedcurve.txt")


def run_bentmode():
    """
    Bent-waveguide FDE sweep over (R, w, lambda).

    Fixes, at the source, the two defects the full 3-D ring validation
    measured: n_g taken from a straight guide, and a bend-loss model with no
    wavelength dependence. Everything here is solver output -- no fitted
    analytic form is introduced.

    Resumable. Writes bentmode.csv.
    """
    lumapi = load_lumapi()
    csv_path = _out("results", BENTMODE_CSV)
    fields = ["R_um", "w_nm", "lam_nm", "neff_real", "neff_imag",
              "ng_solver", "te_fraction", "wg_te_fraction",
              "loss_dB_per_cm_solver", "loss_dB_per_cm"]
    done = set()
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r["R_um"], r["w_nm"], r["lam_nm"]))

    # straight (R = None) first: it is the control against modesweep.csv
    jobs = [(None, w, l) for w in MODE_W_NM for l in MODE_LAM_NM]
    jobs += [(R, w, l) for R in BENTMODE_R_UM
             for w in MODE_W_NM for l in MODE_LAM_NM]
    todo = [j for j in jobs
            if (("straight" if j[0] is None else f"{j[0]:g}"),
                f"{j[1]:g}", f"{j[2]:g}") not in done]
    print(f"[bentmode] {len(jobs)} solves, {len(jobs)-len(todo)} cached, "
          f"{len(todo)} to run")
    print(f"[bentmode] FDE mesh {BENTMODE_MESH_NM:g} nm, "
          f"TE fraction >= {BENTMODE_TE_MIN:g}")

    t0 = time.time()
    with lumapi.MODE(hide=True) as m:
        for k, (R, w, l) in enumerate(todo, 1):
            try:
                md = _fde_mode(m, R, w, l, probe=(k == 1))
            except Exception as exc:                    # noqa: BLE001
                print(f"[bentmode] FAILED at R={R} w={w} lam={l}: {exc}")
                print("[bentmode] send this message -- an API name may differ "
                      "on this install")
                return
            ner, nei = md["neff"].real, md["neff"].imag
            ngv = md.get("ng", float("nan"))
            if not (BENTMODE_NEFF_MIN < ner < BENTMODE_NEFF_MAX
                    and np.isfinite(ngv)):
                print(f"[bentmode] REFUSED to record R={R} w={w} lam={l}: "
                      f"neff={ner:.5f}, ng={ngv} is not a guided mode")
                return
            # The solver reports loss in dB/m. Im(neff) gives the same thing
            # independently; both are stored so a disagreement is visible
            # rather than averaged away.
            row = dict(R_um=("straight" if R is None else f"{R:g}"),
                       w_nm=f"{w:g}", lam_nm=f"{l:g}",
                       neff_real=ner, neff_imag=nei,
                       ng_solver=md.get("ng", float("nan")),
                       te_fraction=md.get("te", float("nan")),
                       wg_te_fraction=md.get("wg_te", float("nan")),
                       loss_dB_per_cm_solver=md.get("loss", float("nan"))/100.0,
                       loss_dB_per_cm=_bent_loss_db_per_cm(nei, l))
            with open(csv_path, "a", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=fields)
                if fh.tell() == 0:
                    wr.writeheader()
                wr.writerow(row)
            if k % 10 == 0 or k == len(todo):
                el = (time.time() - t0) / 60.0
                print(f"[bentmode] {k}/{len(todo)}  R={row['R_um']:>8s} "
                      f"w={w:g} lam={l:g}  neff {ner:.5f}  "
                      f"ng {row['ng_solver']:.5f}  "
                      f"loss {row['loss_dB_per_cm_solver']:.4g} dB/cm  "
                      f"({el:.1f} min, {el/k*len(todo):.0f} min total)")
    print(f"\n[bentmode] wrote {csv_path}")
    _bentmode_report(csv_path)


def _bentmode_report(csv_path):
    """Bent vs straight n_g, and the wavelength dependence of bend loss."""
    rows = list(csv.DictReader(open(csv_path, newline="")))
    if not rows:
        print("[bentmode] no rows"); return
    byk = {}
    for r in rows:
        byk.setdefault((r["R_um"], float(r["w_nm"])), []).append(
            (float(r["lam_nm"]), float(r["neff_real"]),
             float(r.get("loss_dB_per_cm_solver") or "nan"),
             float(r.get("ng_solver") or "nan")))

    def ng_at(key, lam0=1550.0):
        """Solver-reported n_g at lam0; no finite differencing needed."""
        v = sorted(byk.get(key, []))
        if not v:
            return float("nan")
        for lam, _ne, _ls, ng in v:
            if abs(lam - lam0) < 1e-6:
                return ng
        return float("nan")

    L = ["BENT-WAVEGUIDE FDE SWEEP", "=" * 72,
         "Every number below is FDE solver output. n_g is obtained from the",
         "wavelength dependence of n_eff, the same way the straight sweep",
         "does it, so the two are directly comparable.", "",
         "GROUP INDEX AT 1550 nm", "",
         f"  {'w (nm)':>8s} {'straight':>10s}" +
         "".join(f"{('R=' + f'{R:g}'):>10s}" for R in BENTMODE_R_UM),
         "  " + "-" * (18 + 10 * len(BENTMODE_R_UM))]
    for w in MODE_W_NM:
        st = ng_at(("straight", w))
        cells = "".join(f"{ng_at((f'{R:g}', w)):10.4f}" for R in BENTMODE_R_UM)
        L.append(f"  {w:8.0f} {st:10.4f}" + cells)
    L += ["", "  DIFFERENCE FROM STRAIGHT (%), the FSR bias this explains", "",
          f"  {'w (nm)':>8s}" +
          "".join(f"{('R=' + f'{R:g}'):>10s}" for R in BENTMODE_R_UM),
          "  " + "-" * (8 + 10 * len(BENTMODE_R_UM))]
    for w in MODE_W_NM:
        st = ng_at(("straight", w))
        cells = ""
        for R in BENTMODE_R_UM:
            b = ng_at((f"{R:g}", w))
            cells += f"{100.0*(b-st)/st:9.2f}%" if st == st and b == b else "       ---"
        L.append(f"  {w:8.0f}" + cells)
    L += ["", "  MEASURED 2026-10-13: every entry is below 0.1 %, against the",
          "  +2.2 to +2.8 % FSR bias this phase was built to explain. The",
          "  straight-versus-bent n_g hypothesis is REFUTED -- the effect is",
          "  ~30x too small. Note also that the scatter across R at a fixed w",
          "  (up to 0.77 % at w = 500 nm, non-monotonic in R) exceeds the",
          "  effect itself, so these differences are not resolvable here.", "",
          "BEND LOSS vs WAVELENGTH (dB/cm), w = 450 nm", "",
          f"  {'R (um)':>8s}" +
          "".join(f"{l:>12.0f}" for l in MODE_LAM_NM),
          "  " + "-" * (8 + 12 * len(MODE_LAM_NM))]
    for R in BENTMODE_R_UM:
        v = dict((x[0], x[2]) for x in byk.get((f"{R:g}", 450.0), []))  # loss
        L.append(f"  {R:8.1f}" +
                 "".join(f"{v.get(l, float('nan')):12.4g}" for l in MODE_LAM_NM))
    L += ["", "  MEASURED 2026-10-13: every entry is NEGATIVE, i.e. gain, which",
          "  is not physical, and the magnitudes (1e-5 down to 1e-12 dB/cm)",
          "  sit at the solver's numerical floor. These are noise, not bend",
          "  loss, and cannot support a wavelength-dependent loss model."]
    resdir = "results" if os.path.isdir("results") else "."
    txt = "\n".join(L)
    open(os.path.join(resdir, "bentmode_report.txt"), "w").write(txt + "\n")
    print("\n" + txt)


def run_bendloss(dimension="3D"):
    """Measure bend-radiation loss vs R, at the production mesh."""
    w = 450e-9
    mesh = MeshConfig.uniform(BEND_MESH_NM * 1e-9)
    dom = DomainConfig(movie=False, dimension=dimension,
                       sim_time=COUPLER_SIM_TIME,
                       freq_points=COUPLER_FREQ_POINTS)
    csv_path = os.path.abspath(BENDLOSS_CSV.format(dim=dimension.lower()))
    done = {}
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as fh:
            done = {float(r["R_um"]): r for r in csv.DictReader(fh)}
    todo = [R for R in BEND_R_SWEEP_UM if R not in done]
    fields = ["R_um", "T_quarter", "loss_dB_quarter", "arc_um",
              "loss_dB_per_cm", "a_roundtrip", "runtime_s"]
    print(f"[bend] {len(done)} done, {len(todo)} to run  ({dimension}, "
          f"mesh {BEND_MESH_NM:g} nm)")

    lumapi = load_lumapi()
    for R_um in todo:
        R = R_um * 1e-6
        print(f"\n[bend] R = {R_um:g} um ...")
        t0 = time.time()
        with lumapi.FDTD(hide=True) as fdtd:
            build_bend(fdtd, R, w, mesh, dom)
            fdtd.save(os.path.abspath(
                f"bend_{dimension.lower()}_R{R_um:g}um.fsp"))
            fdtd.run()
            lam, T = get_spectrum(fdtd, "monitor_out")
        dt = time.time() - t0
        Tq = _at_target(lam, T)
        loss_q = -10.0 * np.log10(max(Tq, 1e-12))
        arc = np.pi * R / 2.0
        per_cm = loss_q / (arc * 100.0)
        # racetrack round trip = 4 quarter bends (straights are lossless here)
        a_rt = float(np.sqrt(max(Tq, 0.0) ** 4))
        row = dict(R_um=R_um, T_quarter=Tq, loss_dB_quarter=loss_q,
                   arc_um=arc * 1e6, loss_dB_per_cm=per_cm,
                   a_roundtrip=a_rt, runtime_s=round(dt, 1))
        with open(csv_path, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            if fh.tell() == 0:
                wr.writeheader()
            wr.writerow(row)
        done[R_um] = {k: str(v) for k, v in row.items()}
        print(f"[bend] T {Tq:.6f}  loss {loss_q:.5f} dB/quarter  "
              f"{per_cm:.3f} dB/cm  a {a_rt:.6f}  {dt:.1f}s")

    print("\n" + "=" * 76)
    print("BEND-RADIATION LOSS vs RADIUS  (bend radiation only - see note)")
    print("=" * 76)
    print(f"{'R(um)':>7}{'T_quarter':>12}{'dB/quarter':>13}{'dB/cm':>10}"
          f"{'a_roundtrip':>14}{'runtime':>10}")
    print("-" * 76)
    for R_um in sorted(done.keys()):
        r = done[R_um]
        print(f"{R_um:7.1f}{float(r['T_quarter']):12.6f}"
              f"{float(r['loss_dB_quarter']):13.5f}"
              f"{float(r['loss_dB_per_cm']):10.3f}"
              f"{float(r['a_roundtrip']):14.6f}"
              f"{float(r['runtime_s']):9.1f}s")
    print("-" * 76)
    print("a_roundtrip is the AMPLITUDE survival over 4 quarter bends, i.e. the")
    print("'a' in Eqs. 2, 3 and 6. It excludes sidewall-roughness scattering,")
    print("which FDTD does not model -- add a literature term (Paper 11 measured")
    print("1.3 dB/cm) for a realistic total before generating the dataset.")
    print("=" * 76)
    print(f"\nResults: {csv_path}")


# ---------------------------------------------------------------------------
# COUPLER TRAINING SET  (LHS over R, w, g, Lc)
# ---------------------------------------------------------------------------

def lhs_samples(n, seed=COUPLERSET_SEED):
    """Latin Hypercube over the four Section 4.1.2 ranges."""
    from scipy.stats import qmc
    keys = ["R", "w", "g", "Lc"]
    lo = np.array([RANGES[k][0] for k in keys])
    hi = np.array([RANGES[k][1] for k in keys])
    u = qmc.LatinHypercube(d=4, seed=seed).random(n)
    pts = lo + u * (hi - lo)
    return [MRRParams(R=float(r[0]), w=float(r[1]),
                      g=float(r[2]), Lc=float(r[3])) for r in pts]


def run_couplerset(dimension="3D", n=COUPLERSET_N):
    """
    Measure kappa^2 and t^2 across the design space at the production mesh.
    Resumable: each point is appended as it completes, and re-running skips
    whatever is already in the CSV.
    """
    mesh = MeshConfig.uniform(BEND_MESH_NM * 1e-9)     # 32 nm, per sub-study 2
    dom = DomainConfig(movie=False, dimension=dimension,
                       sim_time=COUPLER_SIM_TIME,
                       freq_points=COUPLER_FREQ_POINTS)
    csv_path = os.path.abspath(COUPLERSET_CSV.format(dim=dimension.lower()))
    fields = ["idx", "R_um", "w_nm", "g_nm", "Lc_um",
              "kappa2", "t2", "sum_k2_t2", "runtime_s"]

    done = set()
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as fh:
            done = {int(r["idx"]) for r in csv.DictReader(fh)}

    pts = lhs_samples(n)
    todo = [(i, p) for i, p in enumerate(pts) if i not in done]
    est = len(todo) * 15.0 / 60.0
    print(f"[couplerset] {len(done)} done, {len(todo)} to run "
          f"({dimension}, mesh {BEND_MESH_NM:g} nm, LHS seed "
          f"{COUPLERSET_SEED})")
    print(f"[couplerset] rough estimate {est:.1f} h at ~15 min/point")

    lumapi = load_lumapi()
    for i, p in todo:
        print(f"\n[couplerset] {i+1}/{n}  R={p.R*1e6:.3f}um w={p.w*1e9:.1f}nm "
              f"g={p.g*1e9:.1f}nm Lc={p.Lc*1e6:.3f}um ...")
        t0 = time.time()
        try:
            with lumapi.FDTD(hide=True) as fdtd:
                build_coupler(fdtd, p, mesh, dom)
                # Lumerical needs a saved file path before run(); a single
                # scratch file is reused so 48 points do not leave 48 files.
                fdtd.save(os.path.abspath("couplerset_scratch.fsp"))
                fdtd.run()
                lam_t, Tt = get_spectrum(fdtd, "monitor_through")
                lam_c, Tc = get_spectrum(fdtd, "monitor_cross")
        except Exception as exc:                        # noqa: BLE001
            print(f"[couplerset] point {i+1} FAILED: {exc}")
            print("[couplerset] re-run the same command to retry it")
            time.sleep(20)                              # let engines release
            continue
        dt = time.time() - t0
        t2 = _at_target(lam_t, Tt)
        k2 = _at_target(lam_c, Tc)
        row = dict(idx=i, R_um=p.R * 1e6, w_nm=p.w * 1e9, g_nm=p.g * 1e9,
                   Lc_um=p.Lc * 1e6, kappa2=k2, t2=t2, sum_k2_t2=k2 + t2,
                   runtime_s=round(dt, 1))
        with open(csv_path, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            if fh.tell() == 0:
                wr.writeheader()
            wr.writerow(row)
        print(f"[couplerset] kappa^2 {k2:.6f}  t^2 {t2:.6f}  "
              f"sum {k2+t2:.4f}  {dt/60:.1f} min")

    print(f"\n[couplerset] results: {csv_path}")


def run_couplerext(n=COUPLEREXT_N, dimension="3D"):
    """
    Append an independent second Latin hypercube to the coupler set, keeping
    every existing row. Indices start at COUPLEREXT_IDX0 so they can never
    collide with the original run's 0..47, and so --couplerset still resumes
    correctly if it is ever re-run.
    Fully resumable: each point is written as it finishes.
    """
    from scipy.stats import qmc
    mesh = MeshConfig.uniform(BEND_MESH_NM * 1e-9)
    dom = DomainConfig(movie=False, dimension=dimension,
                       sim_time=COUPLER_SIM_TIME,
                       freq_points=COUPLER_FREQ_POINTS)
    name = COUPLERSET_CSV.format(dim=dimension.lower())
    try:
        csv_path = os.path.abspath(_find(name))      # . then results/
    except FileNotFoundError:
        raise SystemExit(f"[couplerext] {name} not found in . or results/ "
                         "-- run --couplerset first")
    print(f"[couplerext] appending to {csv_path}")
    fields = ["idx", "R_um", "w_nm", "g_nm", "Lc_um",
              "kappa2", "t2", "sum_k2_t2", "runtime_s"]
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    done = {int(r["idx"]) for r in rows}
    n_orig = sum(1 for i in done if i < COUPLEREXT_IDX0)
    n_ext = sum(1 for i in done if i >= COUPLEREXT_IDX0)
    print(f"[couplerext] existing: {n_orig} original + {n_ext} extension "
          f"= {len(done)} coupler runs")

    keys = ["R", "w", "g", "Lc"]
    lo = np.array([RANGES[k][0] for k in keys])
    hi = np.array([RANGES[k][1] for k in keys])
    u = qmc.LatinHypercube(d=4, seed=COUPLEREXT_SEED).random(n)
    pts = lo + u * (hi - lo)
    todo = [(COUPLEREXT_IDX0 + i, MRRParams(*map(float, r)))
            for i, r in enumerate(pts)
            if COUPLEREXT_IDX0 + i not in done]
    print(f"[couplerext] {len(todo)} to run  -> target "
          f"{n_orig + n} coupler simulations total")
    print(f"[couplerext] estimate {len(todo)*20.9/60:.0f} h at the "
          f"20.9 min/point average of your first 48 runs")
    print("[couplerext] resumable: stop any time, re-run to continue\n")

    lumapi = load_lumapi()
    t_start = time.time()
    for c, (idx, p) in enumerate(todo, 1):
        print(f"[couplerext] {c}/{len(todo)} (idx {idx})  "
              f"R={p.R*1e6:.3f}um w={p.w*1e9:.1f}nm "
              f"g={p.g*1e9:.1f}nm Lc={p.Lc*1e6:.3f}um ...")
        t0 = time.time()
        try:
            with lumapi.FDTD(hide=True) as fdtd:
                build_coupler(fdtd, p, mesh, dom)
                fdtd.save(os.path.abspath("couplerext_scratch.fsp"))
                fdtd.run()
                lam_t, Tt = get_spectrum(fdtd, "monitor_through")
                lam_c, Tc = get_spectrum(fdtd, "monitor_cross")
        except Exception as exc:                        # noqa: BLE001
            print(f"[couplerext] point {idx} FAILED: {exc}")
            print("[couplerext] re-run the same command to retry it")
            time.sleep(20)
            continue
        dt = time.time() - t0
        t2 = _at_target(lam_t, Tt)
        k2 = _at_target(lam_c, Tc)
        with open(csv_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(
                dict(idx=idx, R_um=p.R*1e6, w_nm=p.w*1e9, g_nm=p.g*1e9,
                     Lc_um=p.Lc*1e6, kappa2=k2, t2=t2, sum_k2_t2=k2+t2,
                     runtime_s=round(dt, 1)))
        el = (time.time() - t_start) / 60
        print(f"[couplerext]   kappa^2 {k2:.6f}  t^2 {t2:.6f}  "
              f"sum {k2+t2:.4f}  {dt/60:.1f} min   "
              f"[{el/60:.1f} h elapsed, ~{(len(todo)-c)*el/c/60:.1f} h left]")

    with open(csv_path, newline="") as fh:
        tot = len(list(csv.DictReader(fh)))
    print(f"\n[couplerext] coupler set now holds {tot} simulations")
    print("[couplerext] NEXT, in order:")
    print("    py -3 mrr_template.py --generate   (rebuild the dataset)")
    print("    py -3 mrr_template.py --phase2     (re-split)")
    print("    py -3 mrr_template.py --phase3     "
          "(delete results/phase3_gridsearch.csv first to re-search)")
    print("    ... then phases 4-7 as before")


# ---------------------------------------------------------------------------
# MODE SWEEP  +  SUB-STUDY 3 (literature validation)
# ---------------------------------------------------------------------------

def _neff_at(fdtd, w, lam):
    """Build a straight-waveguide cross-section and solve its fundamental TE mode."""
    fdtd.switchtolayout(); fdtd.deleteall()
    fdtd.addrect(); fdtd.set("name", "cladding")
    fdtd.set("x", 0.0); fdtd.set("x span", 6e-6)
    fdtd.set("y", 0.0); fdtd.set("y span", 6e-6)
    fdtd.set("z", 0.0); fdtd.set("z span", 6e-6)
    fdtd.set("material", MAT_CLAD)
    fdtd.set("override mesh order from material database", True)
    fdtd.set("mesh order", 3)
    fdtd.addrect(); fdtd.set("name", "wg")
    fdtd.set("x", 0.0); fdtd.set("x span", 6e-6)
    fdtd.set("y", 0.0); fdtd.set("y span", w)
    fdtd.set("z", 0.0); fdtd.set("z span", H_CORE)
    fdtd.set("material", MAT_CORE)
    fdtd.set("override mesh order from material database", True)
    fdtd.set("mesh order", 2)
    fdtd.addfdtd(); fdtd.set("dimension", "3D")
    fdtd.set("x", 0.0); fdtd.set("x span", 1e-6)
    fdtd.set("y", 0.0); fdtd.set("y span", 3e-6)
    fdtd.set("z", 0.0); fdtd.set("z span", 2.5e-6)
    fdtd.set("mesh accuracy", 3)
    fdtd.addmesh(); fdtd.set("name", "mesh_xsec")
    fdtd.set("x", 0.0); fdtd.set("x span", 1e-6)
    fdtd.set("y", 0.0); fdtd.set("y span", w + 0.6e-6)
    fdtd.set("z", 0.0); fdtd.set("z span", H_CORE + 0.6e-6)
    d = MODE_MESH_NM * 1e-9
    fdtd.set("override x mesh", True); fdtd.set("dx", d)
    fdtd.set("override y mesh", True); fdtd.set("dy", d)
    fdtd.set("override z mesh", True); fdtd.set("dz", d)
    fdtd.addmode(); fdtd.set("name", "src")
    fdtd.set("injection axis", "x-axis"); fdtd.set("direction", "Forward")
    fdtd.set("x", 0.0)
    fdtd.set("y", 0.0); fdtd.set("y span", 2.5e-6)
    fdtd.set("z", 0.0); fdtd.set("z span", 2.2e-6)
    fdtd.set("override global source settings", True)
    fdtd.set("center wavelength", lam)
    fdtd.set("wavelength span", 1e-9)
    fdtd.set("mode selection", "fundamental TE mode")
    fdtd.set("number of trial modes", N_TRIAL_MODES)
    fdtd.select("src")
    fdtd.updatesourcemode()
    r = fdtd.getresult("src", "neff")
    if isinstance(r, dict):
        r = r.get("neff", next(iter(r.values())))
    return float(np.real(np.array(r).flatten()[0]))


def run_modesweep():
    """n_eff and n_g across the w range, then the Paper 11 FSR validation."""
    csv_path = os.path.abspath(MODESWEEP_CSV)
    rows = []
    lumapi = load_lumapi()
    t0 = time.time()
    with lumapi.FDTD(hide=True) as fdtd:
        # probe the first solve so an API mismatch fails loudly and early
        try:
            test = _neff_at(fdtd, 450e-9, 1550e-9)
        except Exception as exc:                        # noqa: BLE001
            print(f"[modesweep] mode solve FAILED: {exc}")
            try:
                print("[modesweep] results available on the source:",
                      fdtd.getresult("src"))
            except Exception:                           # noqa: BLE001
                pass
            print("[modesweep] send this output -- the result name differs "
                  "on this install")
            return
        print(f"[modesweep] probe: w=450 nm, 1550 nm -> n_eff {test:.6f}  "
              f"(GUI earlier gave 2.351764)")
        for w_nm in MODE_W_NM:
            ne = []
            for lam_nm in MODE_LAM_NM:
                ne.append(_neff_at(fdtd, w_nm * 1e-9, lam_nm * 1e-9))
            ne = np.array(ne); lam = np.array(MODE_LAM_NM)
            ng = ne - lam * np.gradient(ne, lam)        # n_g = n - l*dn/dl
            for l, a, b in zip(lam, ne, ng):
                rows.append(dict(w_nm=w_nm, lam_nm=l, n_eff=a, n_g=b))
            print(f"[modesweep] w={w_nm:5.1f} nm  n_eff(1550)="
                  f"{ne[2]:.5f}  n_g(1550)={ng[2]:.5f}")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["w_nm", "lam_nm", "n_eff", "n_g"])
        wr.writeheader(); wr.writerows(rows)

    # ---- SUB-STUDY 3: literature validation against Paper 11 ------------
    ng440 = next(r["n_g"] for r in rows
                 if r["w_nm"] == P11_W_NM and r["lam_nm"] == 1550.0)
    L = 2 * np.pi * P11_R_UM * 1e-6
    fsr = (1550e-9) ** 2 / (ng440 * L) * 1e9
    err = abs(fsr - P11_FSR_NM) / P11_FSR_NM * 100
    print("\n" + "=" * 70)
    print("PHASE 1.6 SUB-STUDY 3 -- LITERATURE VALIDATION (Paper 11)")
    print("=" * 70)
    print(f"  Device (measured)  : {P11_W_NM:g} x 220 nm SOI TE ring, "
          f"R = {P11_R_UM:g} um")
    print(f"  Measured FSR       : {P11_FSR_NM:.3f} nm")
    print(f"  Simulated n_g      : {ng440:.5f}  (w = {P11_W_NM:g} nm, 1550 nm)")
    print(f"  Predicted FSR      : {fsr:.4f} nm   (Eq. 4)")
    print(f"  Relative error     : {err:.2f} %")
    print(f"  Objective O1 target: <= 10 %   ->  "
          + ("PASS" if err <= 10 else "*** FAIL ***"))
    print("=" * 70)
    print(f"\nmode sweep: {csv_path}   ({(time.time()-t0)/60:.1f} min)")


# ---------------------------------------------------------------------------
# DATASET GENERATOR  (every model refitted from the raw FDTD CSVs)
# ---------------------------------------------------------------------------

def _out(folder, name):
    """Write into the repo folder if it exists, else the current folder."""
    return os.path.join(folder, name) if os.path.isdir(folder) else name


def _find(name):
    """Look in the current folder first, then results/ (the repo layout)."""
    for cand in (name, os.path.join("results", name)):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"{name} not found in . or results/")


def _fit_models():
    """Refit kappa^2, n_eff and bend loss from the raw result files."""
    import json
    from scipy.optimize import least_squares
    # kappa^2 : 5-parameter coupled-mode model (sub-study 2 on the coupler)
    C = np.array([[float(r[k]) for k in ("R_um", "w_nm", "g_nm", "Lc_um", "kappa2")]
                  for r in csv.DictReader(open(_find("couplerset_3d.csv")))])
    R, w, g, Lc, k2 = C.T
    th = np.arcsin(np.sqrt(k2))
    def km(q, R, w, g, Lc):
        lk0, a1, g0, g1, B = q; dw = w - 450.0; gam = g0 + g1 * dw
        return np.exp(lk0 + a1 * dw) * np.exp(-gam * g) * (
            Lc + B * np.sqrt(2 * np.pi * R * 1e3 / gam) * 1e-3)
    kq = least_squares(lambda q: np.log(km(q, R, w, g, Lc)) - np.log(th),
                       [0, 0, .008, 3e-5, 1], max_nfev=40000).x
    # n_eff(lambda, w) : smooth 9-term polynomial (suppresses the lambda artefact)
    M = np.array([[float(r[k]) for k in ("lam_nm", "w_nm", "n_eff")]
                  for r in csv.DictReader(open(_find("modesweep.csv")))])
    x = (M[:, 0] - 1550) / 50; y = (M[:, 1] - 450) / 50
    Bm = np.stack([np.ones_like(x), x, y, x*x, x*y, y*y, x*x*y, x*y*y, y**3], 1)
    nc = np.linalg.lstsq(Bm, M[:, 2], rcond=None)[0]
    # bend radiation(R) : exponential, the physically correct form
    Bd = np.array([[float(r["R_um"]), float(r["loss_dB_quarter"])]
                   for r in csv.DictReader(open(_find("bendloss_3d.csv")))])
    bp = np.polyfit(Bd[:, 0], np.log(Bd[:, 1]), 1)
    fits = dict(kappa2=kq.tolist(), neff=nc.tolist(), bend=[bp[1], bp[0]],
                roughness_dB_cm=ROUGHNESS_DB_CM, n_coupler_points=len(k2))
    json.dump(fits, open(_out("results", "model_fits.json"), "w"), indent=1)
    return kq, nc, bp


def _make_responder():
    """Build the analytic response function from the fitted models."""
    from scipy.optimize import brentq
    kq, nc, bp = _fit_models()
    b = lambda x, y: np.array([1, x, y, x*x, x*y, y*y, x*x*y, x*y*y, y**3])
    db = lambda x, y: np.array([0, 1, 0, 2*x, y, 0, 2*x*y, y*y, 0])
    neff = lambda l, w: float(b((l-1550)/50, (w-450)/50) @ nc)
    ng = lambda l, w: neff(l, w) - l * float(db((l-1550)/50, (w-450)/50) @ nc) / 50

    def respond(R, w, g, Lc):
        L = (2*np.pi*R + 2*Lc) * 1e3                                   # Eq. 1, nm
        f = lambda l, m: neff(l, w) * L / l - m                        # m*lam = n_eff*L
        lo = int(np.ceil(neff(1600, w)*L/1600)); hi = int(np.floor(neff(1500, w)*L/1500))
        res = np.sort([brentq(f, 1495, 1605, args=(m,)) for m in range(lo, hi+1)])
        i = int(np.argmin(abs(res - 1550))); lr = res[i]              # nearest 1550
        fsr = float(np.min(np.abs(np.delete(res, i) - lr))) if res.size > 1 else np.nan
        dw = w - 450.0; gam = kq[2] + kq[3]*dw
        th = np.exp(kq[0] + kq[1]*dw)*np.exp(-gam*g)*(Lc + kq[4]*np.sqrt(2*np.pi*R*1e3/gam)*1e-3)
        k2 = float(np.sin(min(th, np.pi/2))**2); t2 = 1 - k2
        bend = 4*np.exp(bp[1] + bp[0]*R); rough = ROUGHNESS_DB_CM*(2*np.pi*R + 2*Lc)*1e-4
        a = 10**(-(bend + rough)/20); x = t2*a
        Tpk = (1 - t2)**2 * a / (1 - x)**2                              # Eq. 3, t1 = t2
        ch = (1 + x*x - 2*(1 - x)**2) / (2*x)                           # exact half max
        flag = ""
        if -1 < ch < 1:
            fwhm = 2*np.arccos(ch)*lr**2 / (2*np.pi*ng(lr, w)*L)
        else:
            fwhm = np.nan; flag = "no_half_max"
        if Tpk > 1:
            flag = (flag + ";T_drop_gt_1").strip(";")
        return dict(R_um=R, w_nm=w, g_nm=g, Lc_um=Lc, n_eff=neff(lr, w), n_g=ng(lr, w),
                    lambda_res_nm=lr, FSR_nm=fsr, FWHM_nm=fwhm,
                    Q_L=lr/fwhm if fwhm == fwhm else np.nan,               # Eq. 5
                    IL_dB=-10*np.log10(Tpk),                                # Eq. 7
                    kappa2=k2, t=np.sqrt(t2), a=a, loss_bend_dB=bend, loss_rough_dB=rough,
                    m_order=int(round(neff(lr, w)*L/lr)), n_res_in_window=int(res.size),
                    origin="dual-fidelity: 3D FDTD coupler/mode/bend + CMT Eqs 2-7",
                    flag=flag)

    return respond


def run_generate(n=DATASET_N):
    from scipy.stats import qmc
    respond = _make_responder()
    keys = ("R", "w", "g", "Lc"); sc = (1e6, 1e9, 1e9, 1e6)
    lo = np.array([RANGES[k][0]*s for k, s in zip(keys, sc)])
    hi = np.array([RANGES[k][1]*s for k, s in zip(keys, sc)])
    P = lo + qmc.LatinHypercube(d=4, seed=DATASET_SEED).random(n) * (hi - lo)
    t0 = time.time(); rows = [respond(*p) for p in P]; dt = time.time() - t0
    out_csv = _out("data", DATASET_CSV)
    with open(out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

    flagged = [(i, r["flag"]) for i, r in enumerate(rows) if r["flag"]]
    Q = np.array([r["Q_L"] for r in rows])
    with open(_out("data", "data_quality_log.txt"), "w") as fh:
        fh.write(f"DATA-QUALITY LOG  (Phase 1.5 step 7)\nN={n}, LHS seed {DATASET_SEED}\n")
        fh.write(f"generated in {dt:.2f} s\nflagged samples: {len(flagged)}\n")
        for i, fl in flagged:
            fh.write(f"  row {i}: {fl}\n")
        fh.write(f"Q_L range: {Q.min():.1f} to {Q.max():.1f} "
                 f"(log10 {np.log10(Q.min()):.2f} to {np.log10(Q.max()):.2f})\n")
        fh.write("NOTE: FSR is the spacing to the NEAREST adjacent resonance; with\n"
                 "dispersion it can differ slightly from the spacing across 1550 nm.\n")
    print(f"[generate] {n} samples in {dt:.2f} s -> {out_csv}")
    print(f"[generate] flagged: {len(flagged)}   Q_L log10 "
          f"{np.log10(Q.min()):.2f} to {np.log10(Q.max()):.2f}")
    print("[generate] also wrote model_fits.json and data_quality_log.txt")


# ---------------------------------------------------------------------------
# PHASE 2 -- DATA PREPROCESSING
# ---------------------------------------------------------------------------

def _load_dataset():
    path = _find(DATASET_CSV) if os.path.isfile(DATASET_CSV) else os.path.join(
        "data", DATASET_CSV)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{DATASET_CSV} not found; run --generate first")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh)), path


def _clean(rows, log):
    """Phase 2.1. Returns the surviving rows and N_clean."""
    keep, dropped = [], []
    for i, r in enumerate(rows):
        why = []
        if r.get("flag"):
            why.append(f"flag={r['flag']}")
        for k in TARGETS + INPUTS:
            v = float(r[k])
            if not np.isfinite(v):
                why.append(f"{k} not finite")
        if float(r["Q_L"]) <= 0:
            why.append("Q_L <= 0")
        if float(r["FSR_nm"]) <= 0:
            why.append("FSR <= 0")
        # Phase 2.1 step 2: clip a drop-port peak fractionally above 1
        if float(r["IL_dB"]) < 0:
            r["IL_dB"] = "0.0"
            log.append(f"  row {i}: IL < 0 (T_drop > 1) clipped to 0 dB")
        if why:
            dropped.append((i, "; ".join(why)))
        else:
            keep.append(r)
    for i, why in dropped:
        log.append(f"  row {i}: REMOVED -- {why}")
    return keep, len(keep)


def _scenario_counts(A):
    """Phase 7 scenario feasibility, used as the coverage criterion."""
    a = ((A["FSR_nm"] >= 16.0) & (A["Q_L"] >= 1500) & (A["Q_L"] <= 8000)
         & (A["IL_dB"] <= 1.5))
    b = (A["FSR_nm"] >= 6.4) & (A["Q_L"] >= 7750) & (A["IL_dB"] <= 1.0)
    return int(a.sum()), int(b.sum())


def _supplementary(n_have_A):
    """
    Phase 2.2 step 3: one targeted LHS batch in the under-sampled corner.
    Rejects draws whose round-trip length cannot reach FSR >= 16 nm.
    """
    from scipy.stats import qmc
    respond = _make_responder()
    lo = np.array([SUPP_BOX[k][0] for k in ("R", "w", "g", "Lc")])
    hi = np.array([SUPP_BOX[k][1] for k in ("R", "w", "g", "Lc")])
    u = qmc.LatinHypercube(d=4, seed=SUPP_SEED).random(SUPP_N * 3)
    P = lo + u * (hi - lo)
    out = []
    for R, w, g, Lc in P:
        if 2 * np.pi * R + 2 * Lc > SUPP_L_MAX_UM:
            continue                       # cannot reach FSR >= 16 nm
        r = respond(R, w, g, Lc)
        r["origin"] = ("supplementary targeted batch (Phase 2.2 step 3); "
                       + r["origin"])
        out.append(r)
        if len(out) >= SUPP_N:
            break
    return out


def _fit_scaler(rows):
    """Min-max on inputs and targets, fitted on the TRAINING split only."""
    sc = {}
    for k in INPUTS:
        v = np.array([float(r[k]) for r in rows])
        sc[k] = dict(min=float(v.min()), max=float(v.max()), transform="minmax")
    for k in TARGETS:
        v = np.array([float(r[k]) for r in rows])
        if k == "Q_L":                     # Phase 2.3: log10 before scaling
            v = np.log10(v)
            sc[k] = dict(min=float(v.min()), max=float(v.max()),
                         transform="log10+minmax")
        else:
            sc[k] = dict(min=float(v.min()), max=float(v.max()),
                         transform="minmax")
    return sc


def _apply_scaler(rows, sc):
    out = []
    for r in rows:
        d = dict(r)
        for k, p in sc.items():
            v = float(r[k])
            if p["transform"].startswith("log10"):
                v = np.log10(v)
            rng = p["max"] - p["min"]
            d[k + "_scaled"] = (v - p["min"]) / rng if rng else 0.0
        out.append(d)
    return out


def _coverage_figures(rows, parts, outdir):
    """Phase 2.2 figure set: marginals, pairwise scatter, split check."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[phase2] matplotlib not installed -- skipping figures")
        print("[phase2] install with:  py -3 -m pip install matplotlib")
        return []
    written = []
    sup = np.array([r["origin"].startswith("supplementary") for r in rows])
    V = {k: np.array([float(r[k]) for r in rows]) for k in INPUTS + TARGETS}
    lbl = {"R_um": "R (um)", "w_nm": "w (nm)", "g_nm": "g (nm)",
           "Lc_um": "Lc (um)", "lambda_res_nm": "lambda_res (nm)",
           "FSR_nm": "FSR (nm)", "Q_L": "Q_L", "IL_dB": "IL (dB)"}

    # --- figure 1: marginal distributions, 4 inputs + 4 outputs -----------
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.5))
    for i, k in enumerate(INPUTS + TARGETS):
        a = ax[i // 4][i % 4]
        v = np.log10(V[k]) if k == "Q_L" else V[k]
        a.hist(v[~sup], bins=40, color="#4C72B0", label="LHS main")
        if sup.any():
            a.hist(v[sup], bins=40, color="#DD8452", alpha=0.85,
                   label="supplementary")
        a.set_xlabel("log10(Q_L)" if k == "Q_L" else lbl[k], fontsize=9)
        a.tick_params(labelsize=8)
        if i == 0:
            a.set_ylabel("count", fontsize=9); a.legend(fontsize=7)
    ax[0][0].figure.suptitle(
        f"Phase 2.2 marginal distributions  (N = {len(rows)})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    f = os.path.join(outdir, "phase2_marginals.png")
    fig.savefig(f, dpi=150); plt.close(fig); written.append(f)

    # --- figure 2: pairwise input scatter, coloured by log10(Q_L) --------
    q = np.log10(V["Q_L"])
    pairs = [(a, b) for i, a in enumerate(INPUTS) for b in INPUTS[i + 1:]]
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    for i, (a_, b_) in enumerate(pairs):
        a = ax[i // 3][i % 3]
        sc = a.scatter(V[a_], V[b_], c=q, s=7, cmap="viridis")
        a.set_xlabel(lbl[a_], fontsize=9); a.set_ylabel(lbl[b_], fontsize=9)
        a.tick_params(labelsize=8)
    fig.colorbar(sc, ax=ax, label="log10(Q_L)", shrink=0.8)
    fig.suptitle("Phase 2.2 pairwise input coverage", fontsize=11)
    f = os.path.join(outdir, "phase2_pairwise_inputs.png")
    fig.savefig(f, dpi=150); plt.close(fig); written.append(f)

    # --- figure 3: output space with the two scenario targets ------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    a = ax[0]
    a.scatter(V["FSR_nm"][~sup], V["Q_L"][~sup], s=8, c="#4C72B0",
              label="LHS main")
    if sup.any():
        a.scatter(V["FSR_nm"][sup], V["Q_L"][sup], s=10, c="#DD8452",
                  label="supplementary")
    a.axvline(16.0, ls="--", c="crimson", lw=1)
    a.axhspan(1500, 8000, color="crimson", alpha=0.10)
    a.axvline(6.4, ls="--", c="seagreen", lw=1)
    a.axhline(7750, ls="--", c="seagreen", lw=1)
    a.set_yscale("log"); a.set_xlabel("FSR (nm)", fontsize=9)
    a.set_ylabel("Q_L", fontsize=9)
    a.set_title("output space (red = Scenario A, green = Scenario B)",
                fontsize=10)
    a.legend(fontsize=8); a.tick_params(labelsize=8)

    a = ax[1]
    a.scatter(V["Q_L"], V["IL_dB"], s=8, c="#4C72B0")
    a.axhline(1.5, ls="--", c="crimson", lw=1)
    a.axhline(1.0, ls="--", c="seagreen", lw=1)
    a.set_xscale("log"); a.set_xlabel("Q_L", fontsize=9)
    a.set_ylabel("IL (dB)", fontsize=9)
    a.set_title("insertion loss vs Q_L", fontsize=10)
    a.tick_params(labelsize=8)
    fig.tight_layout()
    f = os.path.join(outdir, "phase2_output_coverage.png")
    fig.savefig(f, dpi=150); plt.close(fig); written.append(f)

    # --- figure 4: split check -- the three splits must overlap ----------
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    cols = dict(train="#4C72B0", val="#DD8452", test="#55A868")
    for i, k in enumerate(TARGETS):
        a = ax[i]
        for name, part in parts.items():
            v = np.array([float(r[k]) for r in part])
            if k == "Q_L":
                v = np.log10(v)
            a.hist(v, bins=30, histtype="step", density=True, lw=1.4,
                   color=cols[name], label=name)
        a.set_xlabel("log10(Q_L)" if k == "Q_L" else lbl[k], fontsize=9)
        a.tick_params(labelsize=8)
        if i == 0:
            a.set_ylabel("density", fontsize=9); a.legend(fontsize=8)
    fig.suptitle("Phase 2.4 split check -- target distributions "
                 "(train / val / test should overlap)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    f = os.path.join(outdir, "phase2_split_check.png")
    fig.savefig(f, dpi=150); plt.close(fig); written.append(f)
    return written


def run_phase2():
    import json
    rows, src = _load_dataset()
    log = ["PHASE 2 -- DATA PREPROCESSING", "=" * 68,
           f"source: {src}", f"rows in: {len(rows)}", "",
           "2.1 CLEANING", "-" * 68]

    n_in = len(rows)
    rows, n_clean = _clean(rows, log)
    log += [f"  removed: {n_in - n_clean}", f"  N_clean = {n_clean}", ""]

    A = {k: np.array([float(r[k]) for r in rows]) for k in INPUTS + TARGETS}
    nA, nB = _scenario_counts(A)
    log += ["2.2 COVERAGE CHECK", "-" * 68,
            f"  Scenario A feasible samples: {nA}",
            f"  Scenario B feasible samples: {nB}"]

    # --- Phase 2.2 step 3: supplementary batch if a clear gap exists --------
    added = []
    if nA < 30:
        log.append(f"  GAP: only {nA} samples meet all Scenario A specs -> "
                   "generating a targeted supplementary batch")
        added = _supplementary(nA)
        rows += added
        A = {k: np.array([float(r[k]) for r in rows]) for k in INPUTS + TARGETS}
        nA2, nB2 = _scenario_counts(A)
        log += [f"  added {len(added)} samples in R {SUPP_BOX['R']} um, "
                f"g {SUPP_BOX['g']} nm, Lc {SUPP_BOX['Lc']} um",
                f"  Scenario A feasible after: {nA2}  (was {nA})",
                f"  Scenario B feasible after: {nB2}  (was {nB})",
                "  these rows are marked in the origin column"]
        n_clean = len(rows)
    else:
        log.append("  no supplementary batch needed")
    log += ["", f"  N_clean (final) = {n_clean}", ""]

    log += ["  marginal ranges:", "-" * 68]
    for k in INPUTS + TARGETS:
        v = A[k]
        log.append(f"    {k:15s} {v.min():12.4f}  {np.median(v):12.4f}  "
                   f"{v.max():12.4f}")

    # --- Phase 2.4 split ----------------------------------------------------
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(len(rows))
    n_tr = int(round(SPLIT_FRAC[0] * len(rows)))
    n_va = int(round(SPLIT_FRAC[1] * len(rows)))
    parts = dict(train=[rows[i] for i in idx[:n_tr]],
                 val=[rows[i] for i in idx[n_tr:n_tr + n_va]],
                 test=[rows[i] for i in idx[n_tr + n_va:]])
    log += ["", "2.4 SPLIT (whole geometry samples, seed "
            f"{SPLIT_SEED})", "-" * 68]
    for name, part in parts.items():
        sup = sum(1 for r in part if r["origin"].startswith("supplementary"))
        log.append(f"  {name:5s} {len(part):5d}  "
                   f"({100*len(part)/len(rows):4.1f}%)  "
                   f"supplementary rows: {sup}")

    # --- Phase 2.3 transforms, fitted on TRAIN ONLY -------------------------
    sc = _fit_scaler(parts["train"])
    log += ["", "2.3 TRANSFORMS", "-" * 68,
            "  scaler fitted on the TRAINING split only, then applied to val",
            "  and test, so no test information reaches the fit",
            "  Q_L: log10 before min-max (Phase 2.3); the others stay linear"]
    for k, p in sc.items():
        log.append(f"    {k:15s} {p['transform']:14s} "
                   f"min {p['min']:12.5f}  max {p['max']:12.5f}")

    outdir = "data" if os.path.isdir("data") else "."
    for name, part in parts.items():
        scaled = _apply_scaler(part, sc)
        with open(os.path.join(outdir, f"{name}.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(scaled[0]))
            wr.writeheader(); wr.writerows(scaled)
    json.dump(dict(seed=SPLIT_SEED, fractions=SPLIT_FRAC, n_clean=n_clean,
                   fitted_on="train split only", inputs=INPUTS,
                   targets=TARGETS, scaler=sc),
              open(os.path.join(outdir, "scaler.json"), "w"), indent=1)

    log += ["", "2.5 ENGINEERED FEATURE (ablation, optional)", "-" * 68,
            "  The dataset already carries kappa2 as a column. Note before",
            "  using it as an input feature: in this dual-fidelity dataset",
            "  kappa2 is the exact intermediate the targets were computed",
            "  from, not an independent estimate. Feeding it to a model",
            "  hands over most of the forward map, so any gain is not",
            "  comparable to the Paper 6-style engineered-feature result.",
            "  Report it as a diagnostic, not as a fair feature-set comparison."]

    figs = _coverage_figures(rows, parts, outdir)
    if figs:
        log += ["", "2.2 FIGURE SET", "-" * 68]
        log += [f"  {os.path.basename(f)}" for f in figs]

    with open(os.path.join(outdir, "phase2_log.txt"), "w") as fh:
        fh.write("\n".join(x for x in log if x is not None) + "\n")
    print("\n".join(x for x in log if x is not None))
    print(f"\n[phase2] wrote train/val/test.csv, scaler.json and "
          f"phase2_log.txt to {outdir}/")


# ---------------------------------------------------------------------------
# PHASE 3 -- FORWARD MODEL  (geometry -> response)
# ---------------------------------------------------------------------------

def _torch():
    try:
        import torch
        return torch
    except ImportError:
        raise SystemExit(
            "\n[phase3] PyTorch is not installed.\n"
            "  install it with:\n"
            "    py -3 -m pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu\n"
            "  (CPU build is enough -- the networks here are tiny)\n")


def _load_split(name):
    d = "data" if os.path.isdir("data") else "."
    path = os.path.join(d, f"{name}.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"[phase3] {path} not found -- run --phase2 first")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    X = np.array([[float(r[k + "_scaled"]) for k in INPUTS] for r in rows],
                 dtype=np.float32)
    Y = np.array([[float(r[k + "_scaled"]) for k in TARGETS] for r in rows],
                 dtype=np.float32)
    return X, Y, rows


def _build_mlp(torch, depth, width, n_in=4, n_out=4):
    """
    Phase 3.2 topology. At the default depth 3 the widths are the fixed
    64-128-64; other depths use a flat `width`, which is what the Phase 3.4
    grid varies.
    """
    import torch.nn as nn
    widths = list(NN_HIDDEN) if (depth == 3 and width == 128) else [width] * depth
    layers, prev = [], n_in
    for h in widths:
        layers += [nn.Linear(prev, h), nn.LeakyReLU(NN_LEAKY)]
        prev = h
    layers += [nn.Linear(prev, n_out)]
    net = nn.Sequential(*layers)
    for m in net:                       # Phase 3.3 Glorot/Xavier uniform
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net, widths


def _train_mlp(torch, net, Xtr, Ytr, Xva, Yva, lr, wd,
               max_epochs=NN_MAX_EPOCHS, patience=NN_PATIENCE, quiet=True):
    """Phase 3.3 training loop. Returns best val MSE and the loss curves."""
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                    batch_size=NN_BATCH, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=NN_LR_DECAY)
    lossf = nn.MSELoss()
    xva, yva = torch.from_numpy(Xva), torch.from_numpy(Yva)
    best, best_state, bad = float("inf"), None, 0
    tr_curve, va_curve = [], []
    for ep in range(max_epochs):
        net.train(); tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            l = lossf(net(xb), yb)
            l.backward(); opt.step()
            tot += l.item() * len(xb)
        sch.step()
        net.eval()
        with torch.no_grad():
            v = lossf(net(xva), yva).item()
        tr_curve.append(tot / len(Xtr)); va_curve.append(v)
        if v < best - 1e-9:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if not quiet and ep % 25 == 0:
            print(f"    epoch {ep:4d}  train {tot/len(Xtr):.5f}  val {v:.5f}")
    if best_state is not None:
        net.load_state_dict(best_state)
    return best, tr_curve, va_curve, ep + 1


def _unscale(v, p):
    v = v * (p["max"] - p["min"]) + p["min"]
    return 10.0 ** v if p["transform"].startswith("log10") else v


def _metrics(y_true, y_pred):
    """R2, MAPE (%), RMSE -- computed in physical units."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return r2, mape, rmse


def run_phase3():
    import json
    torch = _torch()
    torch.manual_seed(NN_SEED); np.random.seed(NN_SEED)
    outdir = "data" if os.path.isdir("data") else "."
    resdir = "results" if os.path.isdir("results") else "."

    Xtr, Ytr, _ = _load_split("train")
    Xva, Yva, _ = _load_split("val")
    Xte, Yte, _ = _load_split("test")
    sc = json.load(open(os.path.join(outdir, "scaler.json")))["scaler"]
    print(f"[phase3] train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}")

    # --- Phase 3.4 hyperparameter search, resumable -----------------------
    ckpt = os.path.join(resdir, "phase3_gridsearch.csv")
    done = {}
    if os.path.isfile(ckpt):
        with open(ckpt, newline="") as fh:
            for r in csv.DictReader(fh):
                done[(int(r["depth"]), int(r["width"]), float(r["lr"]),
                      float(r["wd"]))] = float(r["val_mse"])
        print(f"[phase3] resuming -- {len(done)} configurations already done")
    grid = [(d, wd_, lr, l2) for d in GRID_DEPTH for wd_ in GRID_WIDTH
            for lr in GRID_LR for l2 in GRID_WD]
    todo = [c for c in grid if c not in done]
    print(f"[phase3] grid: {len(grid)} configurations, {len(todo)} to run")

    new_file = not os.path.isfile(ckpt)
    fh = open(ckpt, "a", newline="")
    wr = csv.writer(fh)
    if new_file:
        wr.writerow(["depth", "width", "lr", "wd", "val_mse", "epochs"])
        fh.flush()
    t0 = time.time()
    for i, (d, w, lr, l2) in enumerate(todo, 1):
        torch.manual_seed(NN_SEED)
        net, _ = _build_mlp(torch, d, w)
        v, _, _, eps = _train_mlp(torch, net, Xtr, Ytr, Xva, Yva, lr, l2,
                                  max_epochs=GRID_SEARCH_EPOCHS)
        done[(d, w, lr, l2)] = v
        wr.writerow([d, w, lr, l2, f"{v:.8f}", eps]); fh.flush()
        el = time.time() - t0
        print(f"[phase3] {i}/{len(todo)}  depth {d} width {w:3d} "
              f"lr {lr:g} wd {l2:g}  val MSE {v:.6f}  "
              f"({el/i:.0f} s/config, ~{(len(todo)-i)*el/i/60:.0f} min left)")
    fh.close()

    best_cfg = min(done, key=done.get)
    d, w, lr, l2 = best_cfg
    print(f"\n[phase3] best: depth {d} width {w} lr {lr:g} wd {l2:g}  "
          f"val MSE {done[best_cfg]:.6f}")

    # --- Phase 3.5 step 4: retrain the winner, full budget ----------------
    torch.manual_seed(NN_SEED)
    net, widths = _build_mlp(torch, d, w)
    vbest, tr_c, va_c, eps = _train_mlp(torch, net, Xtr, Ytr, Xva, Yva,
                                        lr, l2, quiet=False)
    print(f"[phase3] retrained: {eps} epochs, best val MSE {vbest:.6f}")

    # --- Phase 3.5 step 5: freeze and save --------------------------------
    wpath = os.path.join(resdir, "phase3_forward_model.pt")
    torch.save(dict(state_dict=net.state_dict(), depth=d, width=w,
                    widths=widths, lr=lr, wd=l2, leaky=NN_LEAKY,
                    inputs=INPUTS, targets=TARGETS, seed=NN_SEED), wpath)

    # --- Phase 3.5 step 6: test-set metrics, in physical units ------------
    net.eval()
    with torch.no_grad():
        P = net(torch.from_numpy(Xte)).numpy()
    lines = ["PHASE 3 -- FORWARD MODEL", "=" * 68,
             f"grid searched: {len(done)} configurations "
             f"({len(GRID_DEPTH)}x{len(GRID_WIDTH)}x{len(GRID_LR)}x"
             f"{len(GRID_WD)})",
             f"selected: depth {d}, widths {widths}, lr {lr:g}, "
             f"weight decay {l2:g}",
             f"selection criterion: minimum validation MSE "
             f"({done[best_cfg]:.6f})",
             f"retrained {eps} epochs, early stopping patience {NN_PATIENCE}",
             "", "test-set accuracy (physical units, Q_L back-transformed "
             "to linear)", "-" * 68,
             f"  {'target':16s} {'R2':>9s} {'MAPE %':>9s} {'RMSE':>13s}"]
    ok = 0
    for j, k in enumerate(TARGETS):
        yt = _unscale(Yte[:, j].astype(np.float64), sc[k])
        yp = _unscale(P[:, j].astype(np.float64), sc[k])
        r2, mape, rmse = _metrics(yt, yp)
        flag = ""
        if r2 >= 0.90 and mape <= 10.0:
            ok += 1
        else:
            flag = "   <-- misses the Phase 3 criterion"
        lines.append(f"  {k:16s} {r2:9.4f} {mape:9.3f} {rmse:13.5g}{flag}")
    lines += ["", f"targets meeting R2 >= 0.90 AND MAPE <= 10%: {ok} of 4",
              "Phase 3 success criterion is >= 3 of 4: "
              + ("MET" if ok >= 3 else "NOT MET")]
    if ok < 3:
        lines += ["", "revision trigger (Section 5): if validation loss "
                  "plateaus far above", "training loss, increase "
                  "regularization before assuming insufficient data.",
                  "Compare the two curves in phase3_loss_curves.png."]
    lines += ["", f"frozen weights: {wpath}",
              "This exact model is reused unmodified as f_theta in Phase 4."]

    # --- loss curves ------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
        ax[0].plot(tr_c, label="train", lw=1.3)
        ax[0].plot(va_c, label="validation", lw=1.3)
        ax[0].set_yscale("log"); ax[0].set_xlabel("epoch")
        ax[0].set_ylabel("MSE (scaled units)"); ax[0].legend()
        ax[0].set_title(f"Phase 3.5 training curves "
                        f"(depth {d}, width {w})", fontsize=10)
        vm = np.array(sorted(done.values()))
        ax[1].plot(vm, ".", ms=4)
        ax[1].set_yscale("log"); ax[1].set_xlabel("configuration (sorted)")
        ax[1].set_ylabel("best validation MSE")
        ax[1].set_title(f"Phase 3.4 search over {len(done)} configurations",
                        fontsize=10)
        fig.tight_layout()
        f = os.path.join(resdir, "phase3_loss_curves.png")
        fig.savefig(f, dpi=150); plt.close(fig)
        lines.append(f"loss curves: {f}")
    except ImportError:
        pass

    txt = "\n".join(lines)
    open(os.path.join(resdir, "phase3_report.txt"), "w").write(txt + "\n")
    print("\n" + txt)
    print(f"\n[phase3] wrote phase3_report.txt, phase3_gridsearch.csv, "
          f"phase3_forward_model.pt to {resdir}/")


# ---------------------------------------------------------------------------
# PHASE 4 -- INVERSE MODEL AND TANDEM NETWORK
# ---------------------------------------------------------------------------

def _load_forward(torch):
    """Load and FREEZE the Phase 3 forward model."""
    resdir = "results" if os.path.isdir("results") else "."
    path = os.path.join(resdir, "phase3_forward_model.pt")
    if not os.path.isfile(path):
        raise SystemExit(f"[phase4] {path} not found -- run --phase3 first")
    ck = torch.load(path, weights_only=False)
    net, _ = _build_mlp(torch, ck["depth"], ck["width"])
    net.load_state_dict(ck["state_dict"])
    for prm in net.parameters():
        prm.requires_grad_(False)
    net.eval()
    return net, ck


def _build_inverse(torch, seed):
    """Phase 4.3 topology, fixed at 64-128-64 for both inverse networks."""
    import torch.nn as nn
    torch.manual_seed(seed)
    layers, prev = [], 4
    for h in INV_HIDDEN:
        layers += [nn.Linear(prev, h), nn.LeakyReLU(NN_LEAKY)]
        prev = h
    layers += [nn.Linear(prev, 4)]
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net


def _train_inverse(torch, g, Xgeo_tr, Yres_tr, Xgeo_va, Yres_va,
                   forward=None, tag=""):
    """
    Phase 4.4. forward=None  -> naive baseline, Eq. 8 (parameter-space loss)
               forward=f     -> tandem,         Eq. 9 (response-space loss)
    In both cases the network input is the RESPONSE and its output is GEOMETRY.
    """
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    dl = DataLoader(TensorDataset(torch.from_numpy(Yres_tr),
                                  torch.from_numpy(Xgeo_tr)),
                    batch_size=NN_BATCH, shuffle=True)
    opt = torch.optim.Adam(g.parameters(), lr=INV_LR, weight_decay=INV_WD)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=NN_LR_DECAY)
    lossf = nn.MSELoss()
    rva, gva = torch.from_numpy(Yres_va), torch.from_numpy(Xgeo_va)
    best, best_state, bad = float("inf"), None, 0
    tr_c, va_c = [], []
    for ep in range(NN_MAX_EPOCHS):
        g.train(); tot = 0.0
        for rb, gb in dl:                      # rb = response, gb = geometry
            opt.zero_grad()
            pred_geo = g(rb)
            if forward is None:
                l = lossf(pred_geo, gb)                    # Eq. 8
            else:
                l = lossf(forward(pred_geo), rb)           # Eq. 9
            l.backward(); opt.step()
            tot += l.item() * len(rb)
        sch.step()
        g.eval()
        with torch.no_grad():
            pv = g(rva)
            v = (lossf(pv, gva) if forward is None
                 else lossf(forward(pv), rva)).item()
        tr_c.append(tot / len(Yres_tr)); va_c.append(v)
        if v < best - 1e-9:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in g.state_dict().items()}
        else:
            bad += 1
            if bad >= NN_PATIENCE:
                break
        if ep % 25 == 0:
            print(f"    [{tag}] epoch {ep:4d}  train {tot/len(Yres_tr):.5f}  "
                  f"val {v:.5f}")
    if best_state is not None:
        g.load_state_dict(best_state)
    return best, tr_c, va_c, ep + 1


def _rms_rge(geo_pred, geo_true):
    """
    Phase 6.2 Eq. 15 with the v1.3 Fix 4 convention.
      R, w, g : s_p = p_target          (Paper 12 convention)
      Lc      : s_p = 3 um range width  (well defined at Lc = 0)
    geo arrays are in PHYSICAL units, columns ordered as INPUTS.
    Returns the aggregate %, the four per-parameter RMS terms, and the
    {R,w,g}-only Paper-12-convention value.
    """
    terms = []
    for j, k in enumerate(INPUTS):
        d = geo_pred[:, j] - geo_true[:, j]
        sp = LC_RANGE_UM if k == "Lc_um" else geo_true[:, j]
        terms.append((d / sp) ** 2)
    per = [100.0 * float(np.sqrt(np.mean(t))) for t in terms]
    agg = 100.0 * float(np.sqrt(np.mean(np.mean(np.stack(terms), axis=0))))
    p12 = 100.0 * float(np.sqrt(np.mean(np.mean(np.stack(terms[:3]), axis=0))))
    return agg, per, p12


def _geo_physical(arr, sc):
    out = np.empty_like(arr, dtype=np.float64)
    for j, k in enumerate(INPUTS):
        out[:, j] = _unscale(arr[:, j].astype(np.float64), sc[k])
    return out


def _res_physical(arr, sc):
    out = np.empty_like(arr, dtype=np.float64)
    for j, k in enumerate(TARGETS):
        out[:, j] = _unscale(arr[:, j].astype(np.float64), sc[k])
    return out


def _theta(R, w, g, Lc):
    """
    The coupling phase from the Phase 1.6 kappa^2 model:
        theta = exp(k0 + k1*dw) * exp(-gamma(w)*g) * (Lc + B*sqrt(2*pi*R/gamma))
    kappa^2 = sin^2(theta), so every geometry on a level set of theta has the
    SAME coupling and therefore the same Q_L and IL. This is the exact
    degeneracy underlying Objective O6's one-to-many problem.
    """
    kq, _, _ = _fit_models()
    dw = w - 450.0
    gam = kq[2] + kq[3] * dw
    return (np.exp(kq[0] + kq[1] * dw) * np.exp(-gam * g)
            * (Lc + kq[4] * np.sqrt(2 * np.pi * R * 1e3 / gam) * 1e-3))


def _degeneracy_check(Pred, True_, label):
    """
    Does the network land on the CORRECT level set of theta, even when its
    (g, Lc) differ from the ground truth? If yes, its geometry 'error' is a
    different valid solution, not a wrong answer.
    """
    th_p = np.array([_theta(*r) for r in Pred])
    th_t = np.array([_theta(*r) for r in True_])
    rel = 100.0 * np.abs(th_p - th_t) / np.abs(th_t)
    # how far apart are the geometries themselves, in the degenerate plane?
    dg = 100.0 * np.abs(Pred[:, 2] - True_[:, 2]) / True_[:, 2]
    dl = 100.0 * np.abs(Pred[:, 3] - True_[:, 3]) / LC_RANGE_UM
    return dict(label=label, theta_mape=float(np.mean(rel)),
                theta_med=float(np.median(rel)),
                g_mape=float(np.mean(dg)), lc_rge=float(np.mean(dl)),
                within5=float(100.0 * np.mean(rel <= 5.0)))


def run_phase4():
    import json
    torch = _torch()
    outdir = "data" if os.path.isdir("data") else "."
    resdir = "results" if os.path.isdir("results") else "."
    sc = json.load(open(os.path.join(outdir, "scaler.json")))["scaler"]

    Gtr, Rtr, _ = _load_split("train")      # G = geometry, R = response
    Gva, Rva, _ = _load_split("val")
    Gte, Rte, _ = _load_split("test")
    f, ck = _load_forward(torch)
    print(f"[phase4] frozen forward model: depth {ck['depth']} "
          f"width {ck['width']}  ({sum(p.numel() for p in f.parameters())} "
          f"parameters, all frozen)")

    L = ["PHASE 4 -- INVERSE MODEL AND TANDEM NETWORK", "=" * 70,
         f"inverse topology (Phase 4.3): 4 -> {' -> '.join(map(str, INV_HIDDEN))}"
         f" -> 4, Leaky ReLU alpha {NN_LEAKY}",
         f"optimizer: Adam lr {INV_LR:g}, weight decay {INV_WD:g}, "
         f"batch {NN_BATCH}, patience {NN_PATIENCE}",
         f"naive init seed {INV_SEED_NAIVE}, tandem init seed "
         f"{INV_SEED_TANDEM} (Phase 4.4 requires these to differ)", ""]

    # ===================== step 1: naive inverse baseline =================
    print("\n[phase4] step 1 -- naive inverse baseline (Eq. 8)")
    gn = _build_inverse(torch, INV_SEED_NAIVE)
    vn, trn, van, epn = _train_inverse(torch, gn, Gtr, Rtr, Gva, Rva,
                                       forward=None, tag="naive")
    gn.eval()
    with torch.no_grad():
        pn = gn(torch.from_numpy(Rte)).numpy()
    Pn = _geo_physical(pn, sc); Gt = _geo_physical(Gte, sc)
    agg_n, per_n, p12_n = _rms_rge(Pn, Gt)
    # forward-consistency of the naive network: what response does its
    # geometry actually produce, according to the frozen forward model?
    with torch.no_grad():
        rn = f(torch.from_numpy(pn)).numpy()
    Rn = _res_physical(rn, sc); Rt = _res_physical(Rte, sc)

    L += ["STEP 1 -- NAIVE INVERSE BASELINE (Eq. 8, parameter-space loss)",
          "-" * 70, f"  trained {epn} epochs, best val loss {vn:.6f}", "",
          "  geometry accuracy vs ground truth (its own training objective)",
          f"  {'parameter':12s} {'R2':>9s} {'MAPE %':>9s} {'RMSE':>12s}"]
    for j, k in enumerate(INPUTS):
        r2, mp, rm = _metrics(Gt[:, j], Pn[:, j])
        L.append(f"  {k:12s} {r2:9.4f} {mp:9.3f} {rm:12.5g}")
    L += ["", f"  RMS-RGE (Eq. 15, v1.3 convention) = {agg_n:.3f} %",
          f"    per-parameter terms: " + ", ".join(
              f"{k} {v:.3f}%" for k, v in zip(INPUTS, per_n)),
          f"    Paper-12 convention over R,w,g only = {p12_n:.3f} %  "
          f"(Paper 12 reports 3.46 / 5.14 %)",
          f"    O3 threshold is RMS-RGE <= 15%: "
          + ("MET" if agg_n <= 15.0 else "NOT MET"), "",
          "  forward consistency -- response of the naive network's geometry",
          f"  {'target':16s} {'R2':>9s} {'MAPE %':>9s}"]
    for j, k in enumerate(TARGETS):
        r2, mp, _ = _metrics(Rt[:, j], Rn[:, j])
        L.append(f"  {k:16s} {r2:9.4f} {mp:9.3f}")

    # ===================== step 2: tandem network =========================
    print("\n[phase4] step 2 -- tandem network (Eq. 9, frozen forward)")
    gt = _build_inverse(torch, INV_SEED_TANDEM)
    before = [p.clone() for p in f.parameters()]
    vt, trt, vat, ept = _train_inverse(torch, gt, Gtr, Rtr, Gva, Rva,
                                       forward=f, tag="tandem")
    frozen_ok = all(bool(torch.equal(a, b))
                    for a, b in zip(before, f.parameters()))
    gt.eval()
    with torch.no_grad():
        pt = gt(torch.from_numpy(Rte)).numpy()
        rt_ = f(torch.from_numpy(pt)).numpy()
    Pt = _geo_physical(pt, sc); Rrec = _res_physical(rt_, sc)
    agg_t, per_t, p12_t = _rms_rge(Pt, Gt)

    L += ["", "STEP 2 -- TANDEM NETWORK (Eq. 9, response-space loss)",
          "-" * 70, f"  trained {ept} epochs, best val loss {vt:.6f}",
          f"  frozen-weight check: forward weights unchanged after training "
          f"= {frozen_ok}", "",
          "  PRIMARY METRIC -- reconstructed response f(g(R)) vs requested R",
          f"  {'target':16s} {'R2':>9s} {'MAPE %':>9s} {'RMSE':>12s}"]
    tand_mape = {}
    for j, k in enumerate(TARGETS):
        r2, mp, rm = _metrics(Rt[:, j], Rrec[:, j])
        tand_mape[k] = mp
        L.append(f"  {k:16s} {r2:9.4f} {mp:9.3f} {rm:12.5g}")
    L += ["", "  DIAGNOSTIC -- predicted geometry vs ground-truth geometry",
          "  (Phase 4.4 step 2 expects this to be WORSE than the naive "
          "baseline's,", "   by construction -- the tandem network is never "
          "asked to match it)",
          f"  {'parameter':12s} {'R2':>9s} {'MAPE %':>9s}"]
    for j, k in enumerate(INPUTS):
        r2, mp, _ = _metrics(Gt[:, j], Pt[:, j])
        L.append(f"  {k:12s} {r2:9.4f} {mp:9.3f}")
    # Phase 4 risk check: the inverse networks' outputs are unconstrained,
    # so they can propose geometries outside the Phase 1.2 design ranges,
    # where the frozen forward model is extrapolating rather than predicting.
    L += ["", "  DESIGN-RANGE CHECK (both networks output unconstrained "
          "values)", "-" * 70]
    for tagn, arr in (("naive", Pn), ("tandem", Pt)):
        bad = []
        for j, k in enumerate(INPUTS):
            lo, hi = RANGES[{"R_um": "R", "w_nm": "w", "g_nm": "g",
                             "Lc_um": "Lc"}[k]]
            scl = {"R_um": 1e6, "w_nm": 1e9, "g_nm": 1e9, "Lc_um": 1e6}[k]
            lo, hi = lo * scl, hi * scl
            n_out = int(((arr[:, j] < lo) | (arr[:, j] > hi)).sum())
            if n_out:
                bad.append(f"{k} {n_out}/{len(arr)} "
                           f"(range {arr[:, j].min():.3f}..{arr[:, j].max():.3f}"
                           f", allowed {lo:g}..{hi:g})")
        L.append(f"    {tagn:7s} outside the design space: "
                 + ("; ".join(bad) if bad else "none -- all predictions "
                    "are physically realisable"))

    L += ["", f"  RMS-RGE (diagnostic) = {agg_t:.3f} %   "
          f"[naive baseline: {agg_n:.3f} %]",
          f"    per-parameter terms: " + ", ".join(
              f"{k} {v:.3f}%" for k, v in zip(INPUTS, per_t)),
          f"    Paper-12 convention over R,w,g only = {p12_t:.3f} %"]

    # ---- O6 degeneracy test ---------------------------------------------
    # The kappa^2 model makes Q_L and IL depend on (g, Lc) only through the
    # single combination theta. A network that recovers theta correctly while
    # missing g and Lc individually has found a DIFFERENT VALID geometry --
    # which is the tandem network's entire purpose.
    L += ["", "O6 DEGENERACY TEST -- did the network find a valid alternative?",
          "-" * 70,
          "  Q_L and IL depend on (g, Lc) only through theta, so any geometry",
          "  on a level set of theta is an equally correct answer. If theta is",
          "  recovered while g and Lc individually are not, the geometry error",
          "  is degeneracy, not inaccuracy.",
          f"  {'network':8s} {'theta MAPE':>11s} {'theta median':>13s} "
          f"{'within 5%':>10s} {'g MAPE':>9s} {'Lc RGE':>9s}"]
    for dd in (_degeneracy_check(Pn, Gt, "naive"),
               _degeneracy_check(Pt, Gt, "tandem")):
        L.append(f"  {dd['label']:8s} {dd['theta_mape']:11.3f} "
                 f"{dd['theta_med']:13.3f} {dd['within5']:9.1f}% "
                 f"{dd['g_mape']:9.3f} {dd['lc_rge']:9.3f}")
    L += ["  (theta MAPE much smaller than g/Lc errors => the network is",
          "   landing on the right level set with the wrong coordinates,",
          "   i.e. proposing a legitimately different geometry)"]

    # ---- per-target loss share, the Phase 3 lambda_res concern -----------
    L += ["", "TANDEM LOSS BREAKDOWN BY TARGET (scaled units)", "-" * 70,
          "  which target is actually driving the response-space gradient:"]
    with torch.no_grad():
        err = (torch.from_numpy(rt_) - torch.from_numpy(Rte)) ** 2
    share = err.mean(dim=0).numpy()
    for j, k in enumerate(TARGETS):
        L.append(f"    {k:16s} {share[j]:.6f}   "
                 f"{100*share[j]/share.sum():5.1f}% of the loss")

    # ===================== O6 verdict =====================================
    fwd_mape = {}
    for j, k in enumerate(TARGETS):
        _, mp, _ = _metrics(Rt[:, j], Rn[:, j])
        fwd_mape[k] = mp
    wins = sum(1 for k in TARGETS if tand_mape[k] <= fwd_mape[k])
    L += ["", "=" * 70, "OBJECTIVE O6 -- one-to-many ambiguity test", "=" * 70,
          "  O6 asks whether the naive direct-inverse DNN converges "
          "acceptably, or",
          "  shows the non-convergence/high-error pattern of Paper 3.", "",
          f"  naive  RMS-RGE {agg_n:7.3f} %   "
          + ("(converged acceptably, O3 threshold met)" if agg_n <= 15.0
             else "(exceeds the O3 15% threshold)"),
          f"  tandem RMS-RGE {agg_t:7.3f} %   (diagnostic only)", "",
          "  response-space comparison (the criterion that matters for O6):",
          f"  {'target':16s} {'naive MAPE':>12s} {'tandem MAPE':>13s}"]
    for k in TARGETS:
        L.append(f"  {k:16s} {fwd_mape[k]:12.3f} {tand_mape[k]:13.3f}")
    L += ["", f"  tandem is better or equal on {wins} of 4 targets",
          "  Section 5 criterion: tandem's reconstructed-response MAPE should "
          "be",
          "  competitive with the forward model's own accuracy.",
          "", "  NOTE: this run covers Phase 4.4 steps 1-2 only. Step 3 (the",
          "  physics-informed loss, Eq. 10-11) is a stretch goal the document",
          "  says to run only after steps 1-2 are complete and documented."]

    for name, net in (("phase4_naive_inverse.pt", gn),
                      ("phase4_tandem_inverse.pt", gt)):
        torch.save(dict(state_dict=net.state_dict(), hidden=INV_HIDDEN,
                        leaky=NN_LEAKY), os.path.join(resdir, name))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        ax[0].plot(trn, label="naive train", lw=1.2)
        ax[0].plot(van, label="naive val", lw=1.2)
        ax[0].set_yscale("log"); ax[0].legend(fontsize=8)
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE (geometry space)")
        ax[0].set_title("naive inverse, Eq. 8", fontsize=10)
        ax[1].plot(trt, label="tandem train", lw=1.2, color="#DD8452")
        ax[1].plot(vat, label="tandem val", lw=1.2, color="#55A868")
        ax[1].set_yscale("log"); ax[1].legend(fontsize=8)
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("MSE (response space)")
        ax[1].set_title("tandem, Eq. 9", fontsize=10)
        x = np.arange(4); wdt = 0.35
        ax[2].bar(x - wdt/2, [fwd_mape[k] for k in TARGETS], wdt, label="naive")
        ax[2].bar(x + wdt/2, [tand_mape[k] for k in TARGETS], wdt,
                  label="tandem")
        ax[2].set_xticks(x); ax[2].set_xticklabels(
            ["lam_res", "FSR", "Q_L", "IL"], fontsize=8)
        ax[2].set_ylabel("response MAPE %"); ax[2].legend(fontsize=8)
        ax[2].set_title("response-space accuracy (O6)", fontsize=10)
        fig.tight_layout()
        fp = os.path.join(resdir, "phase4_curves.png")
        fig.savefig(fp, dpi=150); plt.close(fig)
        L.append(f"\nfigures: {fp}")
    except ImportError:
        pass

    txt = "\n".join(L)
    open(os.path.join(resdir, "phase4_report.txt"), "w").write(txt + "\n")
    print("\n" + txt)
    print(f"\n[phase4] wrote phase4_report.txt and both model files "
          f"to {resdir}/")


# ---------------------------------------------------------------------------
# PHASE 5 -- RANDOM FOREST / XGBOOST BASELINES
# ---------------------------------------------------------------------------

def _load_split_raw(name):
    """Physical-unit columns, for the Phase 5.3 unnormalized condition."""
    d = "data" if os.path.isdir("data") else "."
    with open(os.path.join(d, f"{name}.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    X = np.array([[float(r[k]) for k in INPUTS] for r in rows])
    Y = np.array([[float(r[k]) for k in TARGETS] for r in rows])
    return X, Y


def _to_physical(arr, cols, sc, scaled):
    """Back-transform to physical units if the array is in scaled space."""
    if not scaled:
        return np.asarray(arr, dtype=np.float64)
    out = np.empty(np.shape(arr), dtype=np.float64)
    for j, k in enumerate(cols):
        out[:, j] = _unscale(np.asarray(arr)[:, j].astype(np.float64), sc[k])
    return out


def _mape_scorer(cols, sc, scaled):
    """
    Phase 5.2: minimise the MEAN MAPE across all four outputs.
    Always evaluated in PHYSICAL units, so the normalized and unnormalized
    conditions of Phase 5.3 are directly comparable.
    """
    from sklearn.metrics import make_scorer

    def neg_mean_mape(y_true, y_pred):
        t = _to_physical(y_true, cols, sc, scaled)
        p = _to_physical(y_pred, cols, sc, scaled)
        m = [np.mean(np.abs((t[:, j] - p[:, j]) / t[:, j])) * 100.0
             for j in range(t.shape[1])]
        return -float(np.mean(m))

    return make_scorer(neg_mean_mape, greater_is_better=True)


def _make_estimator(kind):
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        # Phase 5.1: RF uses its NATIVE multi-output support
        return RandomForestRegressor(random_state=NN_SEED, n_jobs=-1), RF_GRID
    from sklearn.multioutput import MultiOutputRegressor
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise SystemExit(
            "\n[phase5] xgboost is not installed.\n"
            "  install it with:  py -3 -m pip install xgboost\n")
    # Phase 5.1: XGBoost is wrapped in MultiOutputRegressor
    base = MultiOutputRegressor(
        XGBRegressor(random_state=NN_SEED, n_jobs=1, verbosity=0))
    grid = {f"estimator__{k}": v for k, v in XGB_GRID.items()}
    return base, grid


def _search(kind, Xtr, Ytr, cols, sc, scaled, tag):
    """
    Phase 5.4 steps 2-3. RF gets the full grid; XGBoost uses the
    RandomizedSearchCV fallback the document explicitly permits, and the
    choice is logged either way.
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
    est, grid = _make_estimator(kind)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    sco = _mape_scorer(cols, sc, scaled)
    n_full = int(np.prod([len(v) for v in grid.values()]))
    if kind == "rf":
        se = GridSearchCV(est, grid, scoring=sco, cv=cv, n_jobs=-1,
                          refit=True)
        how = f"GridSearchCV, full grid, {n_full} configurations"
    else:
        se = RandomizedSearchCV(est, grid, n_iter=XGB_N_ITER, scoring=sco,
                                cv=cv, n_jobs=-1, random_state=NN_SEED,
                                refit=True)
        how = (f"RandomizedSearchCV, {XGB_N_ITER} of {n_full} configurations "
               f"(Phase 5.2 fallback, full grid too large)")
    t0 = time.time()
    print(f"[phase5] {tag}: {how} x {CV_FOLDS}-fold CV ...")
    se.fit(Xtr, Ytr)
    print(f"[phase5] {tag}: best CV MAPE {-se.best_score_:.4f} %  "
          f"({time.time()-t0:.0f} s)")
    return se, how


def run_phase5():
    import json
    outdir = "data" if os.path.isdir("data") else "."
    resdir = "results" if os.path.isdir("results") else "."
    sc = json.load(open(os.path.join(outdir, "scaler.json")))["scaler"]

    Gn_tr, Rn_tr, _ = _load_split("train")      # normalized
    Gn_te, Rn_te, _ = _load_split("test")
    Gr_tr, Rr_tr = _load_split_raw("train")     # raw physical units
    Gr_te, Rr_te = _load_split_raw("test")

    cache = os.path.join(resdir, "phase5_results.json")
    done = json.load(open(cache)) if os.path.isfile(cache) else {}
    if done:
        print(f"[phase5] resuming -- {len(done)} of 8 searches already done")

    L = ["PHASE 5 -- RANDOM FOREST / XGBOOST BASELINES", "=" * 74,
         f"5-fold CV on the {len(Gn_tr)}-sample training pool, "
         "minimising mean MAPE",
         "MAPE is always computed in PHYSICAL units, so the normalized and",
         "unnormalized conditions of Phase 5.3 are directly comparable.", ""]
    imp_lines, rge_lines = [], []

    # four models x two preprocessing conditions = eight searches
    jobs = []
    for kind in ("rf", "xgb"):
        for direction in ("forward", "inverse"):
            for cond in ("raw", "norm"):
                jobs.append((kind, direction, cond))

    for kind, direction, cond in jobs:
        tag = f"{direction}_{kind}_{cond}"
        scaled = (cond == "norm")
        if direction == "forward":
            Xtr = Gn_tr if scaled else Gr_tr
            Ytr = Rn_tr if scaled else Rr_tr
            Xte = Gn_te if scaled else Gr_te
            Yte_phys = Rr_te
            cols_out, cols_in = TARGETS, INPUTS
        else:
            Xtr = Rn_tr if scaled else Rr_tr
            Ytr = Gn_tr if scaled else Gr_tr
            Xte = Rn_te if scaled else Rr_te
            Yte_phys = Gr_te
            cols_out, cols_in = INPUTS, TARGETS

        if tag in done:
            rec = done[tag]
            print(f"[phase5] {tag}: cached")
        else:
            se, how = _search(kind, Xtr, Ytr, cols_out, sc, scaled, tag)
            P = _to_physical(se.best_estimator_.predict(Xte), cols_out, sc,
                             scaled)
            rec = dict(how=how, cv_mape=float(-se.best_score_),
                       best=({k: (v if not isinstance(v, (np.integer,
                                                          np.floating))
                                  else v.item())
                              for k, v in se.best_params_.items()}),
                       per_target={}, pred=P.tolist())
            for j, k in enumerate(cols_out):
                r2, mp, rm = _metrics(Yte_phys[:, j], P[:, j])
                rec["per_target"][k] = dict(R2=r2, MAPE=mp, RMSE=rm)
            if direction == "forward":
                est = se.best_estimator_
                fi = (est.feature_importances_ if kind == "rf" else
                      np.mean([e.feature_importances_
                               for e in est.estimators_], axis=0))
                rec["importance"] = {k: float(v)
                                     for k, v in zip(cols_in, fi)}
            else:
                agg, per, p12 = _rms_rge(P, Gr_te)
                rec["rms_rge"] = agg
                rec["rms_rge_per"] = per
                rec["rms_rge_p12"] = p12
            done[tag] = rec
            json.dump(done, open(cache, "w"), indent=1)

        L += [f"{tag}", "-" * 74, f"  search: {rec['how']}",
              f"  best CV MAPE: {rec['cv_mape']:.4f} %",
              f"  selected: " + ", ".join(
                  f"{k.replace('estimator__','')}={v}"
                  for k, v in sorted(rec["best"].items())),
              f"  {'output':16s} {'R2':>9s} {'MAPE %':>9s} {'RMSE':>12s}"]
        for k in cols_out:
            d = rec["per_target"][k]
            L.append(f"  {k:16s} {d['R2']:9.4f} {d['MAPE']:9.3f} "
                     f"{d['RMSE']:12.5g}")
        if "rms_rge" in rec:
            L.append(f"  RMS-RGE = {rec['rms_rge']:.3f} %   "
                     f"(R,w,g only: {rec['rms_rge_p12']:.3f} %)   "
                     f"O3 <= 15%: "
                     + ("MET" if rec["rms_rge"] <= 15.0 else "NOT MET"))
            rge_lines.append((tag, rec["rms_rge"], rec["rms_rge_p12"]))
        if "importance" in rec:
            imp_lines.append((tag, rec["importance"]))
        L.append("")

    # ---- Phase 5.3 verdict: did normalization help? ---------------------
    L += ["=" * 74, "5.3 NORMALIZATION TEST (mandatory)", "=" * 74,
          "  Paper 6 found normalization helped two targets and hurt one.",
          "  This design space, same comparison:", "",
          f"  {'model':20s} {'raw MAPE':>10s} {'norm MAPE':>11s} "
          f"{'verdict':>12s}"]
    for kind in ("rf", "xgb"):
        for direction in ("forward", "inverse"):
            a = done.get(f"{direction}_{kind}_raw")
            b = done.get(f"{direction}_{kind}_norm")
            if not (a and b):
                continue
            ma = np.mean([v["MAPE"] for v in a["per_target"].values()])
            mb = np.mean([v["MAPE"] for v in b["per_target"].values()])
            L.append(f"  {direction+'_'+kind:20s} {ma:10.3f} {mb:11.3f} "
                     f"{('normalization helps' if mb < ma else 'raw is better'):>12s}")
    L += ["", "  Note: tree splits are order-based, so a monotone rescaling of",
          "  the INPUTS cannot change a tree's structure. Any difference here",
          "  comes from rescaling the TARGETS, which reweights the multi-output",
          "  variance criterion, plus the log10 on Q_L in the normalized set."]

    # ---- feature importances (Phase 5.4 step 6) -------------------------
    if imp_lines:
        L += ["", "=" * 74, "FEATURE IMPORTANCES (forward models)", "=" * 74]
        for tag, imp in imp_lines:
            tot = sum(imp.values()) or 1.0
            L.append(f"  {tag}: " + ", ".join(
                f"{k} {100*v/tot:.1f}%" for k, v in sorted(
                    imp.items(), key=lambda kv: -kv[1])))
        L += ["", "  CAUTION on reading these. A multi-output tree splits to",
              "  reduce SUMMED variance across the four targets, so whichever",
              "  target has the largest numerical spread dominates. In raw",
              "  units Q_L spans ~10^3-10^5 and swamps lambda_res, FSR and IL,",
              "  so the ranking mostly reflects what drives Q_L. Normalizing",
              "  equalises the targets and the ranking can change completely.",
              "  Report BOTH rankings and say which preprocessing produced",
              "  each; a single ranking presented alone is not meaningful."]

    # ---- honest note on the Phase 5.2 selection metric -------------------
    L += ["", "=" * 74, "NOTE ON THE SELECTION METRIC FOR THE INVERSE MODELS",
          "=" * 74,
          "  Phase 5.2 fixes the search criterion as mean MAPE across all four",
          "  outputs, and that is what was used here. For the INVERSE models",
          "  the outputs are geometries, and Lc reaches 0 by design (Phase",
          "  1.2), so a relative error on Lc is unbounded -- Lc MAPE runs to",
          "  several hundred percent and dominates the mean, meaning the",
          "  hyperparameter selection is driven almost entirely by Lc.",
          "  This is the same pathology that Phase 6.2 Fix 4b already fixed",
          "  for RMS-RGE by range-normalising the Lc term; the Phase 5.2",
          "  selection metric was not given the matching treatment.",
          "  Reported, not changed: Phase 5.2's criterion is a fixed default.",
          "  Worth raising with the supervisor -- an RMS-RGE-based selection",
          "  criterion would make the inverse search consistent with the",
          "  metric the thesis actually reports."]

    if rge_lines:
        L += ["", "=" * 74, "O3 INVERSE-DESIGN THRESHOLD", "=" * 74,
              "  Phase 4:  naive DNN 15.501 %, tandem 31.438 %"]
        for tag, agg, p12 in rge_lines:
            L.append(f"  {tag:24s} {agg:8.3f} %   (R,w,g only {p12:.3f} %)")
        best = min(rge_lines, key=lambda t: t[1])
        L += ["", f"  best Phase 5 method: {best[0]} at {best[1]:.3f} %",
              "  O3 requires RMS-RGE <= 15% for AT LEAST ONE method: "
              + ("MET" if best[1] <= 15.0 else "NOT MET by Phase 5 either")]

    txt = "\n".join(L)
    open(os.path.join(resdir, "phase5_report.txt"), "w").write(txt + "\n")
    print("\n" + txt)
    print(f"\n[phase5] wrote phase5_report.txt and phase5_results.json "
          f"to {resdir}/")


# ---------------------------------------------------------------------------
# PHASE 6 -- UNIFIED EVALUATION
# ---------------------------------------------------------------------------

def _row(name, vals, w=11):
    return f"  {name:<22s}" + "".join(f"{v:>{w}}" for v in vals)


def _dnn_predict(torch, path, X):
    ck = torch.load(path, weights_only=False)
    if "depth" in ck:
        net, _ = _build_mlp(torch, ck["depth"], ck["width"])
    else:
        net = _build_inverse(torch, NN_SEED)
    net.load_state_dict(ck["state_dict"]); net.eval()
    with torch.no_grad():
        return net(torch.from_numpy(X)).numpy(), net


def _fit_sized(kind, direction, n, Xtr, Ytr, best, seed=NN_SEED):
    """Refit one tree model on the first n training rows (Phase 6.3)."""
    est, _ = _make_estimator(kind)
    est.set_params(**{k: v for k, v in best.items()})
    est.fit(Xtr[:n], Ytr[:n])
    return est


def run_phase6():
    import json
    torch = _torch()
    outdir = "data" if os.path.isdir("data") else "."
    resdir = "results" if os.path.isdir("results") else "."
    sc = json.load(open(os.path.join(outdir, "scaler.json")))["scaler"]
    p5 = json.load(open(os.path.join(resdir, "phase5_results.json")))

    Gn_tr, Rn_tr, _ = _load_split("train")
    Gn_va, Rn_va, _ = _load_split("val")
    Gn_te, Rn_te, _ = _load_split("test")
    Gr_tr, Rr_tr = _load_split_raw("train")
    Gr_te, Rr_te = _load_split_raw("test")

    L = ["PHASE 6 -- UNIFIED EVALUATION", "=" * 92,
         f"identical held-out test set, n = {len(Gr_te)}, used once.",
         "Every method appears in every table (Phase 6.4), including where "
         "it performs worst.", ""]

    # ===== 6.1 forward master table ======================================
    Pdnn, _ = _dnn_predict(torch, os.path.join(resdir,
                                               "phase3_forward_model.pt"),
                           Gn_te)
    Pdnn = _res_physical(Pdnn, sc)
    fwd = {"DNN (Phase 3)": Pdnn}
    for kind in ("rf", "xgb"):
        for cond in ("raw", "norm"):
            tag = f"forward_{kind}_{cond}"
            if tag in p5:
                fwd[f"{kind.upper()} ({cond})"] = np.array(p5[tag]["pred"])

    L += ["=" * 92, "6.1  MASTER TABLE -- FORWARD DIRECTION "
          "(geometry -> response)", "=" * 92,
          "  all three metrics reported for every target, never only one", ""]
    for metric, idx in (("R2", 0), ("MAPE %", 1), ("RMSE", 2)):
        L.append(f"  --- {metric} ---")
        L.append(_row("method", [t.replace("_nm", "").replace("_dB", "")
                                 for t in TARGETS], 13))
        for name, P in fwd.items():
            vals = []
            for j in range(4):
                m = _metrics(Rr_te[:, j], P[:, j])[idx]
                vals.append(f"{m:.4f}" if idx == 0 else
                            (f"{m:.3f}" if idx == 1 else f"{m:.5g}"))
            L.append(_row(name, vals, 13))
        L.append("")
    L.append("  aggregate (mean MAPE across the four targets):")
    agg_fwd = {}
    for name, P in fwd.items():
        a = float(np.mean([_metrics(Rr_te[:, j], P[:, j])[1] for j in range(4)]))
        agg_fwd[name] = a
        L.append(f"    {name:24s} {a:8.3f} %")
    bf = min(agg_fwd, key=agg_fwd.get)
    L += [f"  best forward method: {bf} ({agg_fwd[bf]:.3f} %)", ""]

    # ===== 6.2 inverse master table ======================================
    Pnai, _ = _dnn_predict(torch, os.path.join(resdir,
                                               "phase4_naive_inverse.pt"),
                           Rn_te)
    Ptan, _ = _dnn_predict(torch, os.path.join(resdir,
                                               "phase4_tandem_inverse.pt"),
                           Rn_te)
    inv = {"naive DNN": _geo_physical(Pnai, sc),
           "Tandem": _geo_physical(Ptan, sc)}
    for kind in ("rf", "xgb"):
        for cond in ("raw", "norm"):
            tag = f"inverse_{kind}_{cond}"
            if tag in p5:
                inv[f"{kind.upper()} ({cond})"] = np.array(p5[tag]["pred"])

    L += ["=" * 92, "6.2  MASTER TABLE -- INVERSE DIRECTION "
          "(response -> geometry)", "=" * 92]
    for metric, idx in (("R2", 0), ("MAPE %", 1), ("RMSE", 2)):
        L.append(f"  --- {metric} ---")
        L.append(_row("method", [k.replace("_um", "").replace("_nm", "")
                                 for k in INPUTS], 13))
        for name, P in inv.items():
            vals = []
            for j in range(4):
                m = _metrics(Gr_te[:, j], P[:, j])[idx]
                vals.append(f"{m:.4f}" if idx == 0 else
                            (f"{m:.3f}" if idx == 1 else f"{m:.5g}"))
            L.append(_row(name, vals, 13))
        L.append("")

    L += ["  RMS-RGE (Eq. 15), with the per-parameter breakdown Phase 6.2 "
          "requires:",
          _row("method", ["aggregate", "R", "w", "g", "Lc", "R,w,g only"], 12)]
    rge = {}
    for name, P in inv.items():
        agg, per, p12 = _rms_rge(P, Gr_te)
        rge[name] = agg
        L.append(_row(name, [f"{agg:.3f}"] + [f"{v:.3f}" for v in per]
                      + [f"{p12:.3f}"], 12))
    bi = min(rge, key=rge.get)
    L += ["", f"  best inverse method: {bi} ({rge[bi]:.3f} %)",
          f"  O3 requires RMS-RGE <= 15% for at least one method: "
          + ("MET" if rge[bi] <= 15.0 else "NOT MET"),
          "  (Paper 12 reports 3.46% FDTD / 5.14% analytical under the "
          "R,w,g convention)", ""]

    # ===== 6.4 per-parameter difficulty ranking ==========================
    L += ["=" * 92, "6.4  PER-PARAMETER DIFFICULTY RANKING (mandatory)",
          "=" * 92,
          "  The document hypothesises that the coupling gap g is the hardest",
          "  parameter, by analogy with Paper 6 finding radius hardest. "
          "Confirm or refute:", ""]
    diff = {}
    for j, k in enumerate(INPUTS):
        r2s = [_metrics(Gr_te[:, j], P[:, j])[0] for P in inv.values()]
        rge_terms = [_rms_rge(P, Gr_te)[1][j] for P in inv.values()]
        diff[k] = (float(np.mean(r2s)), float(np.mean(rge_terms)))
    L.append(_row("parameter", ["mean R2", "mean RGE %"], 14))
    for k, (r2, rg) in diff.items():
        L.append(_row(k, [f"{r2:.4f}", f"{rg:.3f}"], 14))
    order = sorted(diff, key=lambda k: diff[k][0])
    L += ["", f"  hardest by mean R2:      {order[0]} "
          f"(R2 = {diff[order[0]][0]:.4f})",
          f"  easiest by mean R2:      {order[-1]} "
          f"(R2 = {diff[order[-1]][0]:.4f})",
          f"  full ranking, hardest first: " + " > ".join(order), "",
          ("  VERDICT: the g hypothesis is CONFIRMED." if order[0] == "g_nm"
           else f"  VERDICT: the g hypothesis is REFUTED -- {order[0]} is "
                f"harder than g_nm."),
          f"  (g_nm mean R2 = {diff['g_nm'][0]:.4f}, "
          f"mean RGE term = {diff['g_nm'][1]:.3f} %)", ""]

    # ===== 6.3 sample-efficiency curve ===================================
    L += ["=" * 92, "6.3  SAMPLE-EFFICIENCY CURVE", "=" * 92]
    sizes = [n if n else len(Gn_tr) for n in SAMPLE_SIZES]
    ck_path = os.path.join(resdir, "phase6_sample_efficiency.json")
    curves = json.load(open(ck_path)) if os.path.isfile(ck_path) else {}

    for n in sizes:
        key = str(n)
        if key in curves:
            print(f"[phase6] n={n}: cached")
            continue
        print(f"[phase6] sample-efficiency at n={n} ...")
        rec = {}
        # --- DNN forward, then the tandem that depends on it -------------
        torch.manual_seed(NN_SEED)
        fck = torch.load(os.path.join(resdir, "phase3_forward_model.pt"),
                         weights_only=False)
        fnet, _ = _build_mlp(torch, fck["depth"], fck["width"])
        _train_mlp(torch, fnet, Gn_tr[:n], Rn_tr[:n], Gn_va, Rn_va,
                   fck["lr"], fck["wd"])
        fnet.eval()
        with torch.no_grad():
            P = _res_physical(fnet(torch.from_numpy(Gn_te)).numpy(), sc)
        rec["DNN forward"] = float(np.mean(
            [_metrics(Rr_te[:, j], P[:, j])[1] for j in range(4)]))
        for prm in fnet.parameters():
            prm.requires_grad_(False)
        # --- naive inverse ----------------------------------------------
        gn = _build_inverse(torch, INV_SEED_NAIVE)
        _train_inverse(torch, gn, Gn_tr[:n], Rn_tr[:n], Gn_va, Rn_va,
                       forward=None, tag=f"n{n}-naive")
        gn.eval()
        with torch.no_grad():
            Pg = _geo_physical(gn(torch.from_numpy(Rn_te)).numpy(), sc)
        rec["naive DNN inverse"] = _rms_rge(Pg, Gr_te)[0]
        # --- tandem, using the same-size forward model -------------------
        gt = _build_inverse(torch, INV_SEED_TANDEM)
        _train_inverse(torch, gt, Gn_tr[:n], Rn_tr[:n], Gn_va, Rn_va,
                       forward=fnet, tag=f"n{n}-tandem")
        gt.eval()
        with torch.no_grad():
            Pg = _geo_physical(gt(torch.from_numpy(Rn_te)).numpy(), sc)
        rec["Tandem inverse"] = _rms_rge(Pg, Gr_te)[0]
        # --- trees -------------------------------------------------------
        for kind in ("rf", "xgb"):
            bp = {k.replace("estimator__", ""): v
                  for k, v in p5[f"forward_{kind}_norm"]["best"].items()}
            bp = {(f"estimator__{k}" if kind == "xgb" else k): v
                  for k, v in bp.items()}
            e = _fit_sized(kind, "forward", n, Gn_tr, Rn_tr, bp)
            P = _res_physical(e.predict(Gn_te), sc)
            rec[f"{kind.upper()} forward"] = float(np.mean(
                [_metrics(Rr_te[:, j], P[:, j])[1] for j in range(4)]))
            bp = {k.replace("estimator__", ""): v
                  for k, v in p5[f"inverse_{kind}_norm"]["best"].items()}
            bp = {(f"estimator__{k}" if kind == "xgb" else k): v
                  for k, v in bp.items()}
            e = _fit_sized(kind, "inverse", n, Rn_tr, Gn_tr, bp)
            Pg = _geo_physical(e.predict(Rn_te), sc)
            rec[f"{kind.upper()} inverse"] = _rms_rge(Pg, Gr_te)[0]
        curves[key] = rec
        json.dump(curves, open(ck_path, "w"), indent=1)

    fwd_keys = ["DNN forward", "RF forward", "XGB forward"]
    inv_keys = ["naive DNN inverse", "Tandem inverse", "RF inverse",
                "XGB inverse"]
    L += ["", "  FORWARD -- aggregate test MAPE (%) vs training-set size",
          _row("method", [str(n) for n in sizes], 11)]
    for k in fwd_keys:
        L.append(_row(k, [f"{curves[str(n)][k]:.3f}" for n in sizes], 11))
    L += ["", "  INVERSE -- RMS-RGE (%) vs training-set size",
          _row("method", [str(n) for n in sizes], 11)]
    for k in inv_keys:
        L.append(_row(k, [f"{curves[str(n)][k]:.3f}" for n in sizes], 11))

    # --- the mandatory written verdict (Fix 1) ---------------------------
    small = [n for n in sizes if n <= 300]
    def _best(keys, ns):
        sc_ = {k: np.mean([curves[str(n)][k] for n in ns]) for k in keys}
        return min(sc_, key=sc_.get), sc_
    bs_f, sf = _best(fwd_keys, small)
    bl_f, lf = _best(fwd_keys, [sizes[-1]])
    bs_i, si = _best(inv_keys, small)
    bl_i, li = _best(inv_keys, [sizes[-1]])
    L += ["", "  MANDATORY VERDICT (Objective O5, Phase 6.3 Fix 1)",
          "  " + "-" * 88,
          f"  (a) In the <= 300-sample regime, the lowest aggregate test MAPE",
          f"      in the FORWARD direction is achieved by {bs_f} "
          f"({sf[bs_f]:.3f} % mean over n = {small}).",
          f"      In the INVERSE direction the lowest RMS-RGE is achieved by",
          f"      {bs_i} ({si[bs_i]:.3f} % mean over the same sizes).",
          f"  (b) At the full dataset size (n = {sizes[-1]}), the lowest "
          f"aggregate test MAPE",
          f"      in the FORWARD direction is achieved by {bl_f} "
          f"({lf[bl_f]:.3f} %),",
          f"      and the lowest RMS-RGE by {bl_i} ({li[bl_i]:.3f} %).", "",
          "  Anchors for the discussion: Paper 6 drew its headline result from",
          "  462 FEM samples and Paper 12 from 2,500 geometries, so this "
          "curve's",
          "  100-805 range sits inside the operating range of the closest "
          "published work."]

    # --- figure -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        for k in fwd_keys:
            ax[0].plot(sizes, [curves[str(n)][k] for n in sizes], "o-",
                       label=k, lw=1.6)
        ax[0].set_xlabel("training-set size"); ax[0].set_yscale("log")
        ax[0].set_ylabel("aggregate test MAPE (%)")
        ax[0].set_title("Phase 6.3 forward direction", fontsize=10)
        ax[0].axvspan(0, 300, color="grey", alpha=0.12)
        ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        for k in inv_keys:
            ax[1].plot(sizes, [curves[str(n)][k] for n in sizes], "o-",
                       label=k, lw=1.6)
        ax[1].axhline(15.0, ls="--", c="crimson", lw=1.2, label="O3 threshold")
        ax[1].axvspan(0, 300, color="grey", alpha=0.12)
        ax[1].set_xlabel("training-set size"); ax[1].set_ylabel("RMS-RGE (%)")
        ax[1].set_title("Phase 6.3 inverse direction "
                        "(shaded = the <=300-sample regime)", fontsize=10)
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fp = os.path.join(resdir, "phase6_sample_efficiency.png")
        fig.savefig(fp, dpi=150); plt.close(fig)
        L.append(f"\n  figure: {fp}")
    except ImportError:
        pass

    L += ["", "=" * 92, "STILL OUTSTANDING", "=" * 92,
          "  Closed-loop response error (Phase 6.2 / Phase 4.4 step 4) is "
          "MANDATORY and",
          "  is not covered by this run -- it needs Lumerical. Run:",
          "      py -3 mrr_template.py --closedloop",
          f"  That re-simulates the predicted geometries for "
          f"{CLOSEDLOOP_N} representative test",
          "  targets through the same dual-fidelity pipeline that built the "
          "dataset",
          "  (3-D coupler FDTD -> kappa^2 -> analytic ring), so it is an",
          "  independent check and not a replay of the analytic model."]

    txt = "\n".join(L)
    open(os.path.join(resdir, "phase6_report.txt"), "w").write(txt + "\n")
    print("\n" + txt)
    print(f"\n[phase6] wrote phase6_report.txt to {resdir}/")


# ---------------------------------------------------------------------------
# CLOSED-LOOP RE-VERIFICATION  (Phase 4.4 step 4 / Phase 6.2)
# ---------------------------------------------------------------------------

def _ring_given_kappa2(R, w, g, Lc, k2):
    """
    The same ring model as _make_responder, but with kappa^2 supplied from a
    fresh 3-D FDTD coupler run instead of from the fitted kappa^2 model.
    n_eff / n_g still come from the mode sweep, which is an independent
    measurement, and the loss terms from the bend sweep.
    """
    from scipy.optimize import brentq
    kq, nc, bp = _fit_models()
    b = lambda x, y: np.array([1, x, y, x*x, x*y, y*y, x*x*y, x*y*y, y**3])
    db = lambda x, y: np.array([0, 1, 0, 2*x, y, 0, 2*x*y, y*y, 0])
    neff = lambda l, ww: float(b((l-1550)/50, (ww-450)/50) @ nc)
    ng = lambda l, ww: neff(l, ww) - l * float(
        db((l-1550)/50, (ww-450)/50) @ nc) / 50

    L = (2*np.pi*R + 2*Lc) * 1e3
    f = lambda l, m: neff(l, w) * L / l - m
    lo = int(np.ceil(neff(1600, w)*L/1600))
    hi = int(np.floor(neff(1500, w)*L/1500))
    res = np.sort([brentq(f, 1495, 1605, args=(m,)) for m in range(lo, hi+1)])
    i = int(np.argmin(abs(res - 1550))); lr = res[i]
    fsr = (float(np.min(np.abs(np.delete(res, i) - lr)))
           if res.size > 1 else np.nan)
    t2 = 1.0 - k2
    bend = 4*np.exp(bp[1] + bp[0]*R)
    rough = ROUGHNESS_DB_CM*(2*np.pi*R + 2*Lc)*1e-4
    a = 10**(-(bend + rough)/20); x = t2*a
    Tpk = (1 - t2)**2 * a / (1 - x)**2
    ch = (1 + x*x - 2*(1 - x)**2) / (2*x)
    fwhm = (2*np.arccos(ch)*lr**2 / (2*np.pi*ng(lr, w)*L)
            if -1 < ch < 1 else np.nan)
    return {"lambda_res_nm": lr, "FSR_nm": fsr,
            "Q_L": lr/fwhm if fwhm == fwhm else np.nan,
            "IL_dB": -10*np.log10(Tpk)}


def _coupler_kappa2_fdtd(R, w, g, Lc):
    """One 3-D FDTD coupler run at the production mesh -> measured kappa^2."""
    lumapi = load_lumapi()
    mesh = MeshConfig.uniform(BEND_MESH_NM * 1e-9)
    dom = DomainConfig(movie=False, dimension="3D",
                       sim_time=COUPLER_SIM_TIME,
                       freq_points=COUPLER_FREQ_POINTS)
    p = MRRParams(R * 1e-6, w * 1e-9, g * 1e-9, Lc * 1e-6)
    with lumapi.FDTD(hide=True) as fdtd:
        build_coupler(fdtd, p, mesh, dom)
        fdtd.save(os.path.abspath("closedloop_scratch.fsp"))
        fdtd.run()
        lam_c, Tc = get_spectrum(fdtd, "monitor_cross")
        lam_t, Tt = get_spectrum(fdtd, "monitor_through")
    return _at_target(lam_c, Tc), _at_target(lam_t, Tt)


def _n_couplerset():
    """Number of points in the coupler set the kappa^2 model was fitted on."""
    try:
        import csv as _csv
        n = 0
        for dim in ("3d", "2d"):
            try:
                f = _find(COUPLERSET_CSV.format(dim=dim))
            except FileNotFoundError:
                continue
            with open(f, newline="") as fh:
                n = max(n, sum(1 for _ in _csv.DictReader(fh)))
        return n if n else "?"
    except Exception:
        return "?"


def _cl_ref(method="naive"):
    """Measured closed-loop RMS error for a method, read from its own report."""
    try:
        f = _find(f"closedloop_{method}.txt")
        for ln in open(f):
            if "RMS closed-loop response error" in ln:
                return ln.split("=")[1].split("%")[0].strip() + " %"
    except Exception:
        pass
    return "see closedloop_naive.txt"


def _cv_mape_ref():
    """CV MAPE of the fitted kappa^2 model, read from its own output."""
    try:
        import json as _j
        f = _find("kappa2_model.json")
        if f and os.path.isfile(f):
            d = _j.load(open(f))
            for k in ("cv_mape", "mape_cv", "cv_mape_pct", "mape"):
                if k in d:
                    return f"{float(d[k]):.2f} %"
    except Exception:
        pass
    return "the value in kappa2_cv.txt"


def run_closedloop(method="naive", n=CLOSEDLOOP_N):
    """
    Take a model's predicted geometries for n representative test targets,
    re-simulate each through the SAME dual-fidelity pipeline that built the
    dataset (3-D coupler FDTD -> kappa^2 -> analytic ring), and compare the
    re-simulated response against the response that was originally asked for.

    The 3-D FDTD coupler run is what makes this independent: feeding the
    predictions back through the analytic kappa^2 model alone would simply
    replay the model that produced the labels.
    """
    import json
    torch = _torch()
    outdir = "data" if os.path.isdir("data") else "."
    resdir = "results" if os.path.isdir("results") else "."
    sc = json.load(open(os.path.join(outdir, "scaler.json")))["scaler"]
    Gn_te, Rn_te, rows = _load_split("test")
    Gr_te, Rr_te = _load_split_raw("test")

    if method in ("naive", "tandem"):
        f = os.path.join(resdir, f"phase4_{method}_inverse.pt")
        P, _ = _dnn_predict(torch, f, Rn_te)
        P = _geo_physical(P, sc)
    else:
        p5 = json.load(open(os.path.join(resdir, "phase5_results.json")))
        P = np.array(p5[f"inverse_{method}"]["pred"])

    # evenly spaced through the test set = representative, not cherry-picked
    idx = np.linspace(0, len(P) - 1, n).astype(int)

    ck = os.path.join(resdir, f"closedloop_{method}.json")
    done = json.load(open(ck)) if os.path.isfile(ck) else {}
    print(f"[closedloop] method={method}, {n} targets, "
          f"{len(done)} already done")

    skipped = []
    for c, i in enumerate(idx, 1):
        key = str(int(i))
        if key in done:
            continue
        R, w, g, Lc = P[i]
        bad = []
        for nm, v, (lo, hi) in (("R", R, (5.0, 15.0)), ("w", w, (400., 500.)),
                                ("g", g, (150., 350.)), ("Lc", Lc, (0., 3.))):
            if not (lo <= v <= hi):
                bad.append(f"{nm}={v:.3f} outside [{lo:g},{hi:g}]")
        if bad:
            # a geometry outside the design space cannot be built or meshed;
            # it counts as a failure of the method, not as a missing data point
            done[key] = dict(status="unrealisable", why="; ".join(bad),
                             geom=[float(x) for x in P[i]])
            skipped.append((int(i), "; ".join(bad)))
            json.dump(done, open(ck, "w"), indent=1)
            print(f"[closedloop] {c}/{n}: UNREALISABLE -- {'; '.join(bad)}")
            continue
        print(f"[closedloop] {c}/{n}: R={R:.3f}um w={w:.1f}nm "
              f"g={g:.1f}nm Lc={Lc:.3f}um ...")
        t0 = time.time()
        try:
            k2, t2 = _coupler_kappa2_fdtd(R, w, g, Lc)
        except Exception as exc:                        # noqa: BLE001
            print(f"[closedloop]   FAILED: {exc}  -- re-run to retry")
            time.sleep(20)
            continue
        resp = _ring_given_kappa2(R, w, g, Lc, k2)
        done[key] = dict(status="ok", kappa2=float(k2), t2=float(t2),
                         geom=[float(x) for x in P[i]],
                         resim={k: float(resp[k]) for k in TARGETS},
                         minutes=(time.time() - t0) / 60.0)
        json.dump(done, open(ck, "w"), indent=1)
        print(f"[closedloop]   kappa^2 {k2:.6f}  t^2 {t2:.6f}  "
              f"sum {k2+t2:.4f}  ({(time.time()-t0)/60:.1f} min)")

    # ---- report ---------------------------------------------------------
    L = [f"CLOSED-LOOP RE-VERIFICATION -- {method}", "=" * 78,
         f"{n} representative test targets, evenly spaced through the test "
         "set.",
         "Each predicted geometry was re-simulated with a 3-D FDTD coupler "
         "run,",
         "then the ring response was computed analytically -- the same "
         "pipeline",
         "that produced the dataset, so the FDTD step is an independent "
         "check.", ""]
    ok = [k for k, v in done.items() if v["status"] == "ok"]
    un = [k for k, v in done.items() if v["status"] == "unrealisable"]
    L += [f"  simulated successfully : {len(ok)}",
          f"  physically unrealisable: {len(un)}"]
    for k in un:
        L.append(f"      sample {k}: {done[k]['why']}")
    if un:
        L += ["  NOTE: an unrealisable geometry is a failure of the method, "
              "not a",
              "  missing measurement. It is counted in the denominator below."]
    L.append("")

    # ---- free held-out validation of the Phase 1 kappa^2 model ----------
    # These geometries were never in the fitted coupler set, so their
    # freshly measured kappa^2 is an out-of-sample test of the fitted model.
    if ok:
        respond = _make_responder()
        er = []
        for k in ok:
            R, w, g, Lc = done[k]["geom"]
            km = done[k]["kappa2"]
            kp = respond(R, w, g, Lc)["kappa2"]
            er.append(100.0 * (kp - km) / km)
        er = np.array(er)
        L += ["  HELD-OUT VALIDATION OF THE PHASE 1 kappa^2 MODEL",
              "  " + "-" * 74,
              f"  These geometries were not in the {_n_couplerset()}-point "
              "coupler set the "
              "model was",
              "  fitted on, so this is an out-of-sample test obtained free "
              "from this run.",
              f"    MAPE {np.mean(np.abs(er)):.2f} %   "
              f"RMS {np.sqrt(np.mean(er**2)):.2f} %   "
              f"bias {np.mean(er):+.2f} %   max {np.max(np.abs(er)):.2f} %",
              f"    within 10%: {100*np.mean(np.abs(er)<=10):.0f} %   "
              f"within 5%: {100*np.mean(np.abs(er)<=5):.0f} %",
              f"    (Phase 1 reported {_cv_mape_ref()} MAPE by cross-"
              "validation on the "
              "training points)",
              "  energy conservation on every run: kappa^2 + t^2 = "
              + f"{np.mean([done[k]['kappa2']+done[k].get('t2',np.nan) for k in ok]):.4f}"
              " (mean)", ""]
        L += ["  per-target closed-loop error (re-simulated vs requested)",
              f"  {'target':16s} {'MAPE %':>10s} {'RMS %':>10s} {'max %':>10s}"]
        rms_all = []
        for j, t in enumerate(TARGETS):
            e = []
            for k in ok:
                want = Rr_te[int(k), j]
                got = done[k]["resim"][t]
                e.append(100.0 * (got - want) / want)
            e = np.array(e)
            rms_all.append(np.sqrt(np.mean(e ** 2)))
            L.append(f"  {t:16s} {np.mean(np.abs(e)):10.3f} "
                     f"{rms_all[-1]:10.3f} {np.max(np.abs(e)):10.3f}")
        overall = float(np.sqrt(np.mean(np.array(rms_all) ** 2)))
        # penalise unrealisable predictions rather than quietly dropping them
        frac = len(ok) / max(1, len(ok) + len(un))
        # O3 HAS TWO HALVES and both must hold for the SAME method:
        #   aggregate RMS-RGE <= 15%   AND   closed-loop error <= 20%
        # Reporting only the closed-loop half overstates the result.
        rge_agg = _rms_rge(P, Gr_te)[0]
        cl_ok = overall <= 20.0
        rge_ok = rge_agg <= 15.0
        L += ["", f"  RMS closed-loop response error = {overall:.3f} %  "
              f"(over the {len(ok)} realisable predictions)",
              f"  realisable fraction = {100*frac:.1f} %",
              "",
              "  OBJECTIVE O3 -- BOTH HALVES REQUIRED FOR THE SAME METHOD",
              "  --------------------------------------------------------",
              f"    aggregate RMS-RGE      {rge_agg:7.3f} %  (<= 15 %)   "
              + ("PASS" if rge_ok else "FAIL"),
              f"    closed-loop RMS error  {overall:7.3f} %  (<= 20 %)   "
              + ("PASS" if cl_ok else "FAIL"),
              f"    this method meets O3: "
              + ("YES" if (cl_ok and rge_ok and not un) else "NO")]
        if cl_ok and rge_ok and un:
            L.append(f"      (both bars cleared, but {len(un)} of "
                     f"{len(ok)+len(un)} predictions were unbuildable)")
        if un:
            L += ["", f"  NOTE: {len(un)} unrealisable predictions. An "
                  "unbuildable geometry is not a",
                  "  design; the realisable fraction above is a primary "
                  "result, not a caveat."]
    txt = "\n".join(L)
    open(os.path.join(resdir, f"closedloop_{method}.txt"), "w").write(
        txt + "\n")
    print("\n" + txt)


# ---------------------------------------------------------------------------
# PHASE 7 -- CPO / INTERCONNECT MAPPING
# ---------------------------------------------------------------------------

def _df3db_ghz(fwhm_nm, lam_nm):
    """Eq. 16: 3-dB optical bandwidth in GHz."""
    return C_LIGHT * (fwhm_nm * 1e-9) / (lam_nm * 1e-9) ** 2 / 1e9


def _scale_target(tup, sc):
    """Physical target tuple -> the scaled vector the inverse models expect."""
    out = []
    for v, k in zip(tup, TARGETS):
        p = sc[k]
        x = np.log10(v) if p["transform"].startswith("log10") else v
        out.append((x - p["min"]) / (p["max"] - p["min"]))
    return np.array([out], dtype=np.float32)


def run_phase7():
    import json
    torch = _torch()
    outdir = "data" if os.path.isdir("data") else "."
    resdir = "results" if os.path.isdir("results") else "."
    sc = json.load(open(os.path.join(outdir, "scaler.json")))["scaler"]
    p5 = json.load(open(os.path.join(resdir, "phase5_results.json")))
    respond = _make_responder()
    Gr_tr, _ = _load_split_raw("train")

    L = ["PHASE 7 -- APPLICATION-SPECIFIC VALIDATION (CPO MAPPING)", "=" * 86,
         "Target tuples are CONSTRUCTED here, because Phase 7.3 states each",
         "spec as a band or a bound while Phase 7.4 step 1 needs a point "
         "target.",
         "Rule used: a bound is targeted at its stated value, a band at its",
         "geometric mean, and IL at half its limit so the query asks for "
         "margin",
         "rather than for a design sitting exactly at the failure edge.",
         "A different construction would move these numbers; state the rule "
         "in the",
         "thesis alongside the results.", ""]

    # --- the inverse models ----------------------------------------------
    models = {}
    for mth in ("naive", "tandem"):
        f = os.path.join(resdir, f"phase4_{mth}_inverse.pt")
        if os.path.isfile(f):
            models[f"{mth} DNN"] = ("dnn", f)
    for kind in ("rf", "xgb"):
        for cond in ("raw", "norm"):
            if f"inverse_{kind}_{cond}" in p5:
                models[f"{kind.upper()} ({cond})"] = ("tree",
                                                      f"inverse_{kind}_{cond}")

    cands = {}
    for sk, S in SCENARIOS.items():
        tgt = S["target"]
        L += ["=" * 86, f"SCENARIO {sk} -- {S['name']}", "=" * 86,
              f"  spec: lambda_res = {S['lam']:g} nm, FSR >= {S['fsr_min']:g} "
              f"nm, " + (f"Q_L {S['q_lo']:.0f}-{S['q_hi']:.0f}"
                         if S["q_hi"] else f"Q_L >= {S['q_lo']:.0f}")
              + f", IL <= {S['il_max']:g} dB",
              f"  constructed target tuple: lambda {tgt[0]:.1f} nm, "
              f"FSR {tgt[1]:.2f} nm, Q_L {tgt[2]:.0f}, IL {tgt[3]:.2f} dB", ""]

        # --- Fix 7b: how close is this scenario to the training data? ----
        if sk == "A":
            ng = 4.2
            Lmax = (S["lam"] ** 2) / (ng * S["fsr_min"]) / 1e3
            Rmax = Lmax / (2 * np.pi)
            inreg = int(np.sum((Gr_tr[:, 0] <= Rmax)))
            L += ["  BOUNDARY-PROXIMITY CHECK (v1.3 Fix 7b -- report this "
                  "explicitly)",
                  f"    FSR >= {S['fsr_min']:g} nm at n_g = {ng} needs "
                  f"L <= {Lmax:.2f} um, i.e. R <= {Rmax:.2f} um at Lc = 0.",
                  f"    That is the lowest {100*(Rmax-5.0)/10.0:.1f} % of the "
                  f"R range.",
                  f"    Training samples with R <= {Rmax:.2f} um: {inreg} of "
                  f"{len(Gr_tr)} ({100*inreg/len(Gr_tr):.1f} %).",
                  "    Scenario A queries therefore sit near the edge of the "
                  "training",
                  "    distribution -- Paper 9 probed exactly this with its "
                  "'furthest data' test.", ""]

        Xs = _scale_target(tgt, sc)
        Xr = np.array([tgt])
        L += ["  CANDIDATE GEOMETRIES AND THEIR PREDICTED RESPONSE",
              f"  {'method':16s} {'R um':>8s} {'w nm':>8s} {'g nm':>8s} "
              f"{'Lc um':>8s} | {'lam':>9s} {'FSR':>7s} {'Q_L':>9s} "
              f"{'IL':>7s}  verdict"]
        cands[sk] = {}
        for name, (kind, ref) in models.items():
            if kind == "dnn":
                P, _ = _dnn_predict(torch, ref, Xs)
                geo = _geo_physical(P, sc)[0]
            else:
                est_cond = ref.endswith("norm")
                from sklearn.ensemble import RandomForestRegressor  # noqa
                # the Phase 5 models were not pickled; refit is cheap and
                # deterministic from the recorded best configuration
                kk = "rf" if "_rf_" in ref else "xgb"
                Gn_tr, Rn_tr, _ = _load_split("train")
                bp = p5[ref]["best"]
                e = _fit_sized(kk, "inverse", len(Gn_tr),
                               Rn_tr if est_cond else _load_split_raw(
                                   "train")[1],
                               Gn_tr if est_cond else Gr_tr, bp)
                X = Xs if est_cond else Xr
                geo = (_geo_physical(e.predict(X), sc)[0] if est_cond
                       else e.predict(X)[0])
            R, w, g, Lc = [float(v) for v in geo]
            bad = []
            for nm, v, (lo, hi) in (("R", R, (5., 15.)), ("w", w, (400., 500.)),
                                    ("g", g, (150., 350.)), ("Lc", Lc, (0., 3.))):
                if not (lo <= v <= hi):
                    bad.append(nm)
            if bad:
                L.append(f"  {name:16s} {R:8.3f} {w:8.1f} {g:8.1f} {Lc:8.3f} "
                         f"| UNREALISABLE -- outside range: {', '.join(bad)}")
                cands[sk][name] = dict(geom=[R, w, g, Lc],
                                       status="unrealisable")
                continue
            r = respond(R, w, g, Lc)
            ok = (abs(r["lambda_res_nm"] - S["lam"]) <= S["dlam_ch"] / 2
                  and r["FSR_nm"] >= S["fsr_min"]
                  and r["Q_L"] >= S["q_lo"]
                  and (S["q_hi"] is None or r["Q_L"] <= S["q_hi"])
                  and r["IL_dB"] <= S["il_max"])
            miss = []
            if abs(r["lambda_res_nm"] - S["lam"]) > S["dlam_ch"] / 2:
                miss.append(f"lam off by "
                            f"{r['lambda_res_nm']-S['lam']:+.2f} nm")
            if r["FSR_nm"] < S["fsr_min"]:
                miss.append(f"FSR short by "
                            f"{S['fsr_min']-r['FSR_nm']:.2f} nm")
            if r["Q_L"] < S["q_lo"]:
                miss.append(f"Q_L low by {S['q_lo']-r['Q_L']:.0f}")
            if S["q_hi"] and r["Q_L"] > S["q_hi"]:
                miss.append(f"Q_L high by {r['Q_L']-S['q_hi']:.0f}")
            if r["IL_dB"] > S["il_max"]:
                miss.append(f"IL over by {r['IL_dB']-S['il_max']:.2f} dB")
            L.append(f"  {name:16s} {R:8.3f} {w:8.1f} {g:8.1f} {Lc:8.3f} | "
                     f"{r['lambda_res_nm']:9.3f} {r['FSR_nm']:7.3f} "
                     f"{r['Q_L']:9.0f} {r['IL_dB']:7.3f}  "
                     + ("PASS" if ok else "FAIL: " + "; ".join(miss)))
            cands[sk][name] = dict(geom=[R, w, g, Lc], status="ok",
                                   pred={k: float(r[k]) for k in TARGETS},
                                   fwhm=float(r["FWHM_nm"]), passes=bool(ok))

        # --- 7.1 / 7.2 translation for whichever candidates are realisable
        L += ["", "  CPO TRANSLATION (Eqs. 16-18) for each realisable "
              "candidate", "  " + "-" * 82,
              f"  {'method':16s} {'FWHM nm':>9s} {'df_3dB GHz':>11s} "
              f"{'N_ch':>6s} {'IL worst-case':>14s} {'IL per-channel':>15s}"]
        for name, d in cands[sk].items():
            if d["status"] != "ok":
                L.append(f"  {name:16s}   (unrealisable geometry -- no CPO "
                         "translation)")
                continue
            fw = d["fwhm"]
            df = _df3db_ghz(fw, d["pred"]["lambda_res_nm"])
            nch = d["pred"]["FSR_nm"] / S["dlam_ch"]
            il = d["pred"]["IL_dB"]
            N = S["n_ch_req"]
            worst = N * il                                   # Eq. 18
            perch = il + (N - 1) * IL_THROUGH_DB             # Fix 6 refinement
            L.append(f"  {name:16s} {fw:9.4f} {df:11.1f} {nch:6.1f} "
                     f"{worst:11.2f} dB {perch:12.2f} dB")
        L += ["",
              f"  Eq. 18 bound assumes every one of the {S['n_ch_req']} rings "
              "contributes its full",
              "  drop-port loss. Fix 6 states this deliberately overstates "
              "the real",
              "  per-channel loss: a channel dropped at ring k sees one drop "
              "loss plus",
              f"  (k-1) off-resonance transits, so the refined column uses "
              f"IL + {S['n_ch_req']-1} x {IL_THROUGH_DB:g} dB.",
              "  The threshold itself is unchanged -- only its justification.",
              f"  Data-rate note: with {S['n_ch_req']} channels on a "
              f"{S['dlam_ch']:g} nm grid, a 3-dB bandwidth of",
              "  df_3dB supports roughly df_3dB/0.75 Gb/s NRZ under the "
              "common assumption",
              "  that the optical filter bandwidth must exceed ~0.75x the "
              "symbol rate.", ""]

    # --- scenario verdict summary (Phase 7.4 step 4, mandatory) ----------
    L += ["=" * 86, "SCENARIO VERDICT SUMMARY (Phase 7.4 step 4)", "=" * 86]
    for sk in SCENARIOS:
        p = [n for n, d in cands[sk].items()
             if d["status"] == "ok" and d["passes"]]
        L.append(f"  Scenario {sk}: "
                 + (f"MET by {len(p)} method(s) -- {', '.join(p)}" if p
                    else "NOT met by any method on every spec simultaneously"))
    allp = any(d["status"] == "ok" and d["passes"]
               for sk in SCENARIOS for d in cands[sk].values())
    L += ["", "  O7 requires >= 1 method to meet all four specs in >= 1 "
          "scenario: "
          + ("MET" if allp else "NOT MET"),
          "  Section 5 revision trigger for Phase 7: if no method meets IL or "
          "Q",
          "  tolerance, revisit the Phase 1.6 coupling-boundary check."]

    # --- 7.5 thermal tunability -------------------------------------------
    L += ["", "=" * 86, "7.5  THERMAL TUNABILITY (qualitative, out of core "
          "scope)", "=" * 86]
    dndT = 1.86e-4
    for ng_ in (4.2, 4.31):
        L.append(f"  at n_g = {ng_}: dlambda/dT ~ 1550/{ng_} x {dndT:.2e} "
                 f"= {1550/ng_*dndT*1e3:.1f} pm/K")
    L += ["  v1.3 note: Eq. 19 needs the MODAL dn_eff/dT, while 1.86e-4 is "
          "the MATERIAL",
          "  dn/dT for silicon; the two coincide only at unity confinement, "
          "so the",
          "  quoted 70-90 pm/K band is right at its lower end and optimistic "
          "at its upper.",
          "  A fabrication offset of one channel spacing would need:"]
    for sk, S in SCENARIOS.items():
        L.append(f"    Scenario {sk}: {S['dlam_ch']:g} nm / "
                 f"{1550/4.2*dndT*1e3:.1f} pm/K = "
                 f"{S['dlam_ch']*1e3/(1550/4.2*dndT*1e3):.0f} K of tuning")
    L += ["  Use the mode solver's own dn_eff/dT if this extension is "
          "pursued.", "",
          "=" * 86, "STILL OUTSTANDING", "=" * 86,
          "  Phase 7.4 step 3 asks for the candidate geometries to be "
          "RE-SIMULATED,",
          "  not just evaluated through the analytic model. The verdicts "
          "above use",
          "  the analytic response, which is the same model that produced the "
          "labels.",
          "  For the independent check run:",
          "      py -3 mrr_template.py --phase7sim",
          "  That puts each realisable candidate through a 3-D FDTD coupler "
          "run,",
          "  as the Phase 6.2 closed-loop procedure does."]

    json.dump(cands, open(os.path.join(resdir, "phase7_candidates.json"), "w"),
              indent=1)
    txt = "\n".join(L)
    open(os.path.join(resdir, "phase7_report.txt"), "w").write(txt + "\n")
    print("\n" + txt)
    print(f"\n[phase7] wrote phase7_report.txt and phase7_candidates.json "
          f"to {resdir}/")


def run_phase7sim():
    """
    Phase 7.4 step 3: re-simulate every realisable Scenario candidate through
    a fresh 3-D FDTD coupler run, so the scenario verdicts rest on an
    independent measurement rather than on the analytic model that produced
    the training labels. Resumable.
    """
    import json
    resdir = "results" if os.path.isdir("results") else "."
    cpath = os.path.join(resdir, "phase7_candidates.json")
    if not os.path.isfile(cpath):
        raise SystemExit("[phase7sim] run --phase7 first")
    cands = json.load(open(cpath))
    ck = os.path.join(resdir, "phase7_resim.json")
    done = json.load(open(ck)) if os.path.isfile(ck) else {}

    jobs = [(sk, nm, d) for sk, mm in cands.items() for nm, d in mm.items()
            if d["status"] == "ok"]
    print(f"[phase7sim] {len(jobs)} realisable candidates, "
          f"{len(done)} already simulated")
    for i, (sk, nm, d) in enumerate(jobs, 1):
        key = f"{sk}|{nm}"
        if key in done:
            continue
        R, w, g, Lc = d["geom"]
        print(f"[phase7sim] {i}/{len(jobs)}  {sk} {nm}: R={R:.3f}um "
              f"w={w:.1f}nm g={g:.1f}nm Lc={Lc:.3f}um ...")
        t0 = time.time()
        try:
            k2, t2 = _coupler_kappa2_fdtd(R, w, g, Lc)
        except Exception as exc:                        # noqa: BLE001
            print(f"[phase7sim]   FAILED: {exc} -- re-run to retry")
            time.sleep(20)
            continue
        r = _ring_given_kappa2(R, w, g, Lc, k2)
        done[key] = dict(kappa2=float(k2), t2=float(t2),
                         resim={k: float(r[k]) for k in TARGETS},
                         minutes=(time.time() - t0) / 60.0)
        json.dump(done, open(ck, "w"), indent=1)
        print(f"[phase7sim]   kappa^2 {k2:.6f}  "
              f"({(time.time()-t0)/60:.1f} min)")

    L = ["PHASE 7.4 STEP 3 -- RE-SIMULATED SCENARIO VERDICTS", "=" * 86,
         "Each candidate geometry was re-simulated with a fresh 3-D FDTD "
         "coupler run.",
         "These verdicts replace the analytic ones in phase7_report.txt.", ""]
    for sk, S in SCENARIOS.items():
        L += [f"SCENARIO {sk} -- {S['name']}", "-" * 86,
              f"  {'method':16s} {'lam':>9s} {'FSR':>7s} {'Q_L':>9s} "
              f"{'IL':>7s} {'df_3dB':>9s} {'N_ch':>6s}  verdict"]
        for nm, d in cands.get(sk, {}).items():
            key = f"{sk}|{nm}"
            if d["status"] != "ok":
                L.append(f"  {nm:16s}  unrealisable geometry -- counts as a "
                         "FAIL for this scenario")
                continue
            if key not in done:
                L.append(f"  {nm:16s}  not simulated yet")
                continue
            r = done[key]["resim"]
            miss = []
            if abs(r["lambda_res_nm"] - S["lam"]) > S["dlam_ch"] / 2:
                miss.append(f"lam {r['lambda_res_nm']-S['lam']:+.2f} nm")
            if r["FSR_nm"] < S["fsr_min"]:
                miss.append(f"FSR -{S['fsr_min']-r['FSR_nm']:.2f} nm")
            if r["Q_L"] < S["q_lo"]:
                miss.append(f"Q_L -{S['q_lo']-r['Q_L']:.0f}")
            if S["q_hi"] and r["Q_L"] > S["q_hi"]:
                miss.append(f"Q_L +{r['Q_L']-S['q_hi']:.0f}")
            if r["IL_dB"] > S["il_max"]:
                miss.append(f"IL +{r['IL_dB']-S['il_max']:.2f} dB")
            fw = r["lambda_res_nm"] / r["Q_L"]
            L.append(f"  {nm:16s} {r['lambda_res_nm']:9.3f} {r['FSR_nm']:7.3f}"
                     f" {r['Q_L']:9.0f} {r['IL_dB']:7.3f} "
                     f"{_df3db_ghz(fw, r['lambda_res_nm']):9.1f} "
                     f"{r['FSR_nm']/S['dlam_ch']:6.1f}  "
                     + ("PASS" if not miss else "FAIL: " + "; ".join(miss)))
        L.append("")
    txt = "\n".join(L)
    open(os.path.join(resdir, "phase7_resim.txt"), "w").write(txt + "\n")
    print("\n" + txt)


# ---------------------------------------------------------------------------
# ABLATION STUDY
# ---------------------------------------------------------------------------

def _build_inverse_v(torch, seed, n_in=4, bounded=False):
    """Phase 4.3 topology, optionally with a bounded (sigmoid) output and a
    wider input layer for the derived feature."""
    import torch.nn as nn
    torch.manual_seed(seed)
    layers, prev = [], n_in
    for h in INV_HIDDEN:
        layers += [nn.Linear(prev, h), nn.LeakyReLU(NN_LEAKY)]
        prev = h
    layers += [nn.Linear(prev, 4)]
    if bounded:
        # the scaled design space is exactly [0,1]^4, so a sigmoid makes
        # every prediction physically realisable by construction
        layers += [nn.Sigmoid()]
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net


def _add_L_feature(Rres, sc):
    """
    Derived input for the inverse models: FSR fixes the round-trip length
    through L = lambda^2 / (n_g * FSR), and L = 2*pi*R + 2*Lc is exactly the
    combination the inverse problem needs. Handing it over saves the network
    from having to learn a reciprocal. Unlike kappa2 this is not circular --
    it is computed from the REQUESTED response, not from the label pipeline.
    """
    lam = _unscale(Rres[:, 0].astype(np.float64), sc["lambda_res_nm"])
    fsr = _unscale(Rres[:, 1].astype(np.float64), sc["FSR_nm"])
    Lum = lam ** 2 / (NG_NOMINAL * fsr) / 1e3            # um
    lo, hi = 2*np.pi*5.0, 2*np.pi*15.0 + 6.0             # the physical span
    col = np.clip((Lum - lo) / (hi - lo), -0.5, 1.5)
    return np.hstack([Rres, col.reshape(-1, 1).astype(np.float32)])


def _rge_loss(torch, pred_scaled, true_phys, sc):
    """
    Loss aligned with the metric the thesis actually reports (Eq. 15):
    relative error on R, w, g and range-normalised error on Lc. The plain
    MSE on scaled coordinates weights these four quite differently.
    """
    terms = []
    for j, k in enumerate(INPUTS):
        p = sc[k]
        phys = pred_scaled[:, j] * (p["max"] - p["min"]) + p["min"]
        tj = true_phys[:, j]
        den = (torch.full_like(tj, LC_RANGE_UM) if k == "Lc_um"
               else torch.clamp(tj, min=1e-6))
        terms.append(((phys - tj) / den) ** 2)
    return torch.stack(terms, dim=1).mean()


def _train_variant(torch, net, Xtr, Ytr_s, Ytr_p, Xva, Yva_s, Yva_p, sc,
                   metric_loss):
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                  torch.from_numpy(Ytr_s),
                                  torch.from_numpy(Ytr_p)),
                    batch_size=NN_BATCH, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=INV_LR, weight_decay=INV_WD)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=NN_LR_DECAY)
    mse = nn.MSELoss()
    xva = torch.from_numpy(Xva); yva_s = torch.from_numpy(Yva_s)
    yva_p = torch.from_numpy(Yva_p)
    best, state, bad = float("inf"), None, 0
    for ep in range(NN_MAX_EPOCHS):
        net.train()
        for xb, ys, yp in dl:
            opt.zero_grad()
            out = net(xb)
            l = (_rge_loss(torch, out, yp, sc) if metric_loss
                 else mse(out, ys))
            l.backward(); opt.step()
        sch.step()
        net.eval()
        with torch.no_grad():
            o = net(xva)
            v = (_rge_loss(torch, o, yva_p, sc) if metric_loss
                 else mse(o, yva_s)).item()
        if v < best - 1e-9:
            best, bad = v, 0
            state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= NN_PATIENCE:
                break
    if state is not None:
        net.load_state_dict(state)
    return best, ep + 1


def run_ablate():
    import json
    torch = _torch()
    outdir = "data" if os.path.isdir("data") else "."
    resdir = "results" if os.path.isdir("results") else "."
    sc = json.load(open(os.path.join(outdir, "scaler.json")))["scaler"]
    Gn_tr, Rn_tr, _ = _load_split("train")
    Gn_va, Rn_va, _ = _load_split("val")
    Gn_te, Rn_te, _ = _load_split("test")
    Gr_tr, _ = _load_split_raw("train")
    Gr_va, _ = _load_split_raw("val")
    Gr_te, _ = _load_split_raw("test")

    variants = [
        ("baseline (Phase 4)",        False, False, False),
        ("+ bounded outputs",         True,  False, False),
        ("+ metric-aligned loss",     False, True,  False),
        ("+ L feature",               False, False, True),
        ("all three",                 True,  True,  True),
    ]
    ck = os.path.join(resdir, "ablation.json")
    done = json.load(open(ck)) if os.path.isfile(ck) else {}

    for name, bnd, met, lft in variants:
        for seed in ABLATE_SEEDS:
            key = f"{name}|{seed}"
            if key in done:
                continue
            Xtr = _add_L_feature(Rn_tr, sc) if lft else Rn_tr
            Xva = _add_L_feature(Rn_va, sc) if lft else Rn_va
            Xte = _add_L_feature(Rn_te, sc) if lft else Rn_te
            print(f"[ablate] {name}, seed {seed} ...")
            net = _build_inverse_v(torch, seed, n_in=Xtr.shape[1],
                                   bounded=bnd)
            _train_variant(torch, net, Xtr, Gn_tr, Gr_tr.astype(np.float32),
                           Xva, Gn_va, Gr_va.astype(np.float32), sc, met)
            net.eval()
            with torch.no_grad():
                P = _geo_physical(net(torch.from_numpy(Xte)).numpy(), sc)
            agg, per, p12 = _rms_rge(P, Gr_te)
            oob = 0
            for j, k in enumerate(INPUTS):
                lo, hi = RANGES[{"R_um": "R", "w_nm": "w", "g_nm": "g",
                                 "Lc_um": "Lc"}[k]]
                scl = {"R_um": 1e6, "w_nm": 1e9, "g_nm": 1e9,
                       "Lc_um": 1e6}[k]
                oob += int(((P[:, j] < lo*scl) | (P[:, j] > hi*scl)).sum())
            done[key] = dict(agg=agg, per=per, p12=p12, oob=oob)
            json.dump(done, open(ck, "w"), indent=1)
            print(f"[ablate]   RMS-RGE {agg:.3f} %  R term {per[0]:.3f} %  "
                  f"Lc term {per[3]:.3f} %  out-of-range {oob}")

    L = ["ABLATION STUDY -- INVERSE DIRECTION", "=" * 86,
         "Three training-side changes, none of which touch a fixed default "
         "in v1.3.",
         f"Each variant is run over {len(ABLATE_SEEDS)} seeds "
         f"{ABLATE_SEEDS} and reported as mean +/- sd,",
         "so an improvement can be told apart from seed noise.", "",
         "  Phase 5 established that Lc_term ~ 9.46 x R_term, so O3 needs the",
         "  R term to fall from 2.852 % to 2.731 % -- a 4.2 % relative gain.",
         "",
         f"  {'variant':24s} {'RMS-RGE %':>18s} {'R term %':>16s} "
         f"{'Lc term %':>16s} {'out-of-range':>13s}"]
    base = None
    for name, *_ in variants:
        vals = [done[f"{name}|{s}"] for s in ABLATE_SEEDS
                if f"{name}|{s}" in done]
        if not vals:
            continue
        a = np.array([v["agg"] for v in vals])
        r = np.array([v["per"][0] for v in vals])
        lc = np.array([v["per"][3] for v in vals])
        ob = np.mean([v["oob"] for v in vals])
        if base is None:
            base = (a.mean(), r.mean())
        L.append(f"  {name:24s} {a.mean():8.3f} +/- {a.std():5.3f} "
                 f"{r.mean():8.3f} +/- {r.std():5.3f} "
                 f"{lc.mean():8.3f} +/- {lc.std():5.3f} {ob:13.1f}")
    L.append("")
    L.append(f"  {'variant':24s} {'dRMS-RGE':>11s} {'dR term':>11s} "
             f"{'O3 <= 15%':>11s}")
    for name, *_ in variants:
        vals = [done[f"{name}|{s}"] for s in ABLATE_SEEDS
                if f"{name}|{s}" in done]
        if not vals:
            continue
        a = float(np.mean([v["agg"] for v in vals]))
        r = float(np.mean([v["per"][0] for v in vals]))
        L.append(f"  {name:24s} {a-base[0]:+10.3f} {r-base[1]:+10.3f} "
                 f"{('MET' if a <= 15.0 else 'not met'):>11s}")
    best = min((n for n, *_ in variants
                if any(f"{n}|{s}" in done for s in ABLATE_SEEDS)),
               key=lambda n: np.mean([done[f"{n}|{s}"]["agg"]
                                      for s in ABLATE_SEEDS
                                      if f"{n}|{s}" in done]))
    bm = float(np.mean([done[f"{best}|{s}"]["agg"] for s in ABLATE_SEEDS
                        if f"{best}|{s}" in done]))
    L += ["", f"  best variant: {best} at RMS-RGE {bm:.3f} %",
          f"  O3 threshold (RMS-RGE <= 15%): "
          + ("MET" if bm <= 15.0 else f"still short by {bm-15.0:.3f} pp"),
          "",
          "  Note on scope: these are training-side choices, not changes to "
          "any",
          "  fixed default. The architecture, ranges, split, seed and metric "
          "are",
          "  all unchanged. Report the baseline as the headline Phase 4 "
          "result and",
          "  this table as an ablation."]
    txt = "\n".join(L)
    open(os.path.join(resdir, "ablation_report.txt"), "w").write(txt + "\n")
    print("\n" + txt)


# ---------------------------------------------------------------------------
# PHYSICS-DECODED INVERSE MODEL
# ---------------------------------------------------------------------------
# The information-floor study showed Lc uses only 46% of the information the
# response carries about it, while R and w are at 99-100% and g at 89%. The
# response constrains the round-trip length L = 2*pi*R + 2*Lc tightly through
# FSR = lambda^2/(n_g*L). So instead of asking the network to emit Lc as a
# free fourth output, it emits R, w and g, and Lc is DERIVED from the FSR
# constraint using the network's own predicted w for n_g. Lc then cannot
# disagree with the FSR that was asked for.

def _ng_torch(torch, lam, w, nc):
    """n_g(lambda, w) from the Phase 1 mode-sweep polynomial, differentiable."""
    x = (lam - 1550.0) / 50.0
    y = (w - 450.0) / 50.0
    one = torch.ones_like(x)
    B = [one, x, y, x*x, x*y, y*y, x*x*y, x*y*y, y**3]
    dB = [torch.zeros_like(x), one, torch.zeros_like(x), 2*x, y,
          torch.zeros_like(x), 2*x*y, y*y, torch.zeros_like(x)]
    neff = sum(c * b for c, b in zip(nc, B))
    dneff = sum(c * d for c, d in zip(nc, dB))
    return neff - lam * dneff / 50.0


def _calibrate_fsr_L(nc, Gtr, Rtr):
    """
    FSR = lambda^2/(n_g L) is a first-order relation, while the dataset's FSR
    comes from the actual adjacent-resonance spacing. Fitting one scalar on
    the TRAINING split removes the resulting systematic offset: without it the
    derived Lc carries a +0.14 um bias, which is 4.7% of the Lc range.
    """
    def ng(lam, w):
        x = (lam - 1550) / 50; y = (w - 450) / 50; one = np.ones_like(x)
        B = [one, x, y, x*x, x*y, y*y, x*x*y, x*y*y, y**3]
        dB = [0*x, one, 0*x, 2*x, y, 0*x, 2*x*y, y*y, 0*x]
        return (sum(c*b for c, b in zip(nc, B))
                - lam * sum(c*d for c, d in zip(nc, dB)) / 50)
    Ltrue = 2 * np.pi * Gtr[:, 0] + 2 * Gtr[:, 3]
    Lform = Rtr[:, 0] ** 2 / (ng(Rtr[:, 0], Gtr[:, 1]) * Rtr[:, 1]) / 1e3
    return float(np.sum(Ltrue * Lform) / np.sum(Lform ** 2))


class _PhysicsInverse:
    """Wraps a 3-output network and derives Lc from the FSR constraint."""

    def __init__(self, torch, net, sc, nc, cal=1.0):
        self.torch, self.net, self.sc, self.nc = torch, net, sc, nc
        self.cal = cal

    def __call__(self, Xres_scaled, lam_phys, fsr_phys):
        torch = self.torch
        out = self.net(Xres_scaled)                 # R, w, g in [0,1]
        cols = []
        for j, k in enumerate(("R_um", "w_nm", "g_nm")):
            p = self.sc[k]
            cols.append(out[:, j] * (p["max"] - p["min"]) + p["min"])
        R, w, g = cols
        ng = _ng_torch(torch, lam_phys, w, self.nc)
        L = self.cal * lam_phys ** 2 / (ng * fsr_phys) / 1e3     # um
        Lc = (L - 2 * np.pi * R) / 2.0                           # Eq. 1 inverted
        return torch.stack([R, w, g, Lc], dim=1)


def run_physinv():
    """Train and evaluate the physics-decoded inverse model."""
    import json
    torch = _torch()
    outdir = "data" if os.path.isdir("data") else "."
    resdir = "results" if os.path.isdir("results") else "."
    sc = json.load(open(os.path.join(outdir, "scaler.json")))["scaler"]
    _, nc, _ = _fit_models()
    nc = [float(v) for v in nc]

    Gn_tr, Rn_tr, _ = _load_split("train")
    Gn_va, Rn_va, _ = _load_split("val")
    Gn_te, Rn_te, _ = _load_split("test")
    Gr_tr, Rr_tr = _load_split_raw("train")
    Gr_va, Rr_va = _load_split_raw("val")
    Gr_te, Rr_te = _load_split_raw("test")

    cal = _calibrate_fsr_L(nc, Gr_tr, Rr_tr)
    print(f"[physinv] FSR->L calibration fitted on train: c = {cal:.6f}")

    def phys_inputs(Rr):
        return (torch.from_numpy(Rr[:, 0].astype(np.float32)),
                torch.from_numpy(Rr[:, 1].astype(np.float32)))

    res = {}
    for seed in ABLATE_SEEDS:
        torch.manual_seed(seed)
        net = _build_inverse_v(torch, seed, n_in=4, bounded=True)
        # replace the 4-output head with a 3-output one
        import torch.nn as nn
        layers = list(net)[:-2]                     # drop Linear(…,4)+Sigmoid
        layers += [nn.Linear(INV_HIDDEN[-1], 3), nn.Sigmoid()]
        net = nn.Sequential(*layers)
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        model = _PhysicsInverse(torch, net, sc, nc, cal)

        from torch.utils.data import TensorDataset, DataLoader
        lam_tr, fsr_tr = phys_inputs(Rr_tr)
        dl = DataLoader(TensorDataset(torch.from_numpy(Rn_tr),
                                      torch.from_numpy(
                                          Gr_tr.astype(np.float32)),
                                      lam_tr, fsr_tr),
                        batch_size=NN_BATCH, shuffle=True)
        opt = torch.optim.Adam(net.parameters(), lr=INV_LR,
                               weight_decay=INV_WD)
        sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=NN_LR_DECAY)
        lam_va, fsr_va = phys_inputs(Rr_va)
        xva = torch.from_numpy(Rn_va)
        gva = torch.from_numpy(Gr_va.astype(np.float32))
        best, state, bad = float("inf"), None, 0

        def loss_fn(pred_phys, true_phys):
            # the reported metric itself (Eq. 15), in physical units
            t = []
            for j, k in enumerate(INPUTS):
                den = (torch.full_like(true_phys[:, j], LC_RANGE_UM)
                       if k == "Lc_um"
                       else torch.clamp(true_phys[:, j], min=1e-6))
                t.append(((pred_phys[:, j] - true_phys[:, j]) / den) ** 2)
            return torch.stack(t, dim=1).mean()

        print(f"[physinv] seed {seed} ...")
        for ep in range(NN_MAX_EPOCHS):
            net.train()
            for xb, gb, lb, fb in dl:
                opt.zero_grad()
                l = loss_fn(model(xb, lb, fb), gb)
                l.backward(); opt.step()
            sch.step()
            net.eval()
            with torch.no_grad():
                v = loss_fn(model(xva, lam_va, fsr_va), gva).item()
            if v < best - 1e-12:
                best, bad = v, 0
                state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= NN_PATIENCE:
                    break
        if state is not None:
            net.load_state_dict(state)
        net.eval()
        lam_te, fsr_te = phys_inputs(Rr_te)
        with torch.no_grad():
            P = model(torch.from_numpy(Rn_te), lam_te, fsr_te).numpy()
        agg, per, p12 = _rms_rge(P.astype(np.float64), Gr_te)
        oob = int(((P[:, 3] < 0.0) | (P[:, 3] > 3.0)).sum())
        res[seed] = dict(agg=agg, per=per, p12=p12, lc_oob=oob)
        print(f"[physinv]   RMS-RGE {agg:.3f} %  R {per[0]:.3f}  w {per[1]:.3f}"
              f"  g {per[2]:.3f}  Lc {per[3]:.3f}  Lc out-of-range {oob}")

    A = np.array([res[s]["agg"] for s in res])
    P = np.array([res[s]["per"] for s in res])
    L = ["PHYSICS-DECODED INVERSE MODEL", "=" * 78,
         "Network emits R, w, g. Lc is derived from the FSR constraint",
         "L = lambda^2/(n_g(lambda, w_pred) * FSR), Lc = (L - 2*pi*R)/2,",
         "so the predicted Lc cannot disagree with the FSR that was requested.",
         f"Loss is Eq. 15 itself. {len(res)} seeds {list(res)}.", "",
         f"  {'':22s} {'RMS-RGE':>16s} {'R':>14s} {'w':>14s} {'g':>14s} "
         f"{'Lc':>14s}",
         f"  {'physics-decoded':22s} {A.mean():8.3f} +/-{A.std():5.3f} "
         + "".join(f"{P[:, j].mean():8.3f} +/-{P[:, j].std():5.3f}"
                   for j in range(4)),
         f"  {'baseline (ablation)':22s} {15.448:8.3f} +/-{0.038:5.3f} "
         f"{3.011:8.3f} +/-{0.155:5.3f}{5.300:8.3f} +/-{0.000:5.3f}"
         f"{14.000:8.3f} +/-{0.000:5.3f}{26.931:8.3f} +/-{0.093:5.3f}",
         f"  {'information floor':22s} {13.30:8.3f} {'':5s} {2.38:14.3f} "
         f"{5.30:14.3f} {12.45:14.3f} {22.78:14.3f}", "",
         f"  change vs baseline: {A.mean()-15.448:+.3f} pp",
         f"  O3 (RMS-RGE <= 15%): "
         + ("MET" if A.mean() <= 15.0 else
            f"short by {A.mean()-15.0:.3f} pp"),
         f"  Lc predictions outside [0, 3] um: "
         f"{np.mean([res[s]['lc_oob'] for s in res]):.1f} of {len(Gr_te)}",
         "",
         f"  oracle check: fed the TRUE R and w, the decoder recovers Lc to",
         f"  1.47 % of range, so the formula and calibration are sound and all",
         f"  residual Lc error comes from R and w error.",
         f"  pi-propagation predicts Lc ~ pi x R_error: at R = "
         f"{P[:, 0].mean():.3f} % that is "
         f"{np.pi*P[:, 0].mean()*9.25/3:.2f} %, against the "
         f"{P[:, 3].mean():.2f} % measured.",
         "",
         "  Lc is now a derived quantity, so its error is whatever the FSR",
         "  constraint propagates from the predicted R and w. If the Lc term",
         "  moved toward 22.78 % the decoder worked; if it did not, Lc's",
         "  shortfall is not caused by it being a free output."]
    txt = "\n".join(L)
    open(os.path.join(resdir, "physinv_report.txt"), "w").write(txt + "\n")
    print("\n" + txt)


def run_density():
    """
    Diagnostic: does the measured learning curve predict what happens when the
    training pool is sampled more densely from the same forward model?
    Validation and test splits are untouched; the scaler is the Phase 2 one.
    """
    import json
    from scipy.stats import qmc
    torch = _torch()
    outdir = "data" if os.path.isdir("data") else "."
    resdir = "results" if os.path.isdir("results") else "."
    sc = json.load(open(os.path.join(outdir, "scaler.json")))["scaler"]
    respond = _make_responder()

    Gn_tr, Rn_tr, _ = _load_split("train")
    Gn_va, Rn_va, _ = _load_split("val")
    Gn_te, Rn_te, _ = _load_split("test")
    Gr_tr, _ = _load_split_raw("train")
    Gr_va, _ = _load_split_raw("val")
    Gr_te, _ = _load_split_raw("test")

    need = max(DENSITY_SIZES) - len(Gn_tr)
    print(f"[density] generating {need} extra analytic rows "
          f"(same kappa^2 model, no new physics) ...")
    lo = np.array([5.0, 400.0, 150.0, 0.0])
    hi = np.array([15.0, 500.0, 350.0, 3.0])
    u = qmc.LatinHypercube(d=4, seed=DENSITY_SEED).random(int(need * 1.1))
    extraG, extraR = [], []
    for R, w, g, Lc in lo + u * (hi - lo):
        r = respond(R, w, g, Lc)
        if not all(np.isfinite(r[k]) for k in TARGETS):
            continue
        extraG.append([R, w, g, Lc])
        extraR.append([r[k] for k in TARGETS])
        if len(extraG) >= need:
            break
    extraG = np.array(extraG); extraR = np.array(extraR)

    def scale(arr, cols):
        out = np.empty_like(arr, dtype=np.float32)
        for j, k in enumerate(cols):
            p = sc[k]
            v = np.log10(arr[:, j]) if p["transform"].startswith("log10") \
                else arr[:, j]
            out[:, j] = (v - p["min"]) / (p["max"] - p["min"])
        return out

    Gx = scale(extraG, INPUTS); Rx = scale(extraR, TARGETS)
    allGn = np.vstack([Gn_tr, Gx]); allRn = np.vstack([Rn_tr, Rx])
    allGr = np.vstack([Gr_tr, extraG])
    print(f"[density] pool ready: {len(allGn)} rows "
          f"({len(Gn_tr)} original + {len(Gx)} generated)")

    ck = os.path.join(resdir, "density.json")
    done = json.load(open(ck)) if os.path.isfile(ck) else {}
    for n in DENSITY_SIZES:
        for seed in ABLATE_SEEDS[:2]:
            key = f"{n}|{seed}"
            if key in done:
                continue
            print(f"[density] n={n}, seed {seed} ...")
            net = _build_inverse_v(torch, seed, n_in=4, bounded=True)
            _train_variant(torch, net, allRn[:n], allGn[:n],
                           allGr[:n].astype(np.float32),
                           Rn_va, Gn_va, Gr_va.astype(np.float32), sc, True)
            net.eval()
            with torch.no_grad():
                P = _geo_physical(net(torch.from_numpy(Rn_te)).numpy(), sc)
            agg, per, _ = _rms_rge(P, Gr_te)
            done[key] = dict(agg=agg, per=per)
            json.dump(done, open(ck, "w"), indent=1)
            print(f"[density]   RMS-RGE {agg:.3f} %  R {per[0]:.3f}  "
                  f"Lc {per[3]:.3f}")

    FLOOR, B, C = 13.30, 6.706, 0.1571
    L = ["TRAINING-DENSITY STUDY -- DIAGNOSTIC ONLY", "=" * 80,
         "",
         "  READ THIS BEFORE QUOTING ANY NUMBER BELOW.",
         "  The extra rows are drawn from the SAME fitted kappa^2 model that",
         "  produced the original labels. They add sampling density, not",
         f"  physics. The dataset still carries the information of the "
         f"{_n_couplerset()}-point",
         "  coupler set (plus the Phase 1 sub-studies) however many rows it",
         "  has. A lower RMS-RGE here means the network has learned the",
         f"  analytic surrogate better -- the closed-loop error against FDTD "
         f"({_cl_ref('naive')}) is unaffected.",
         "  **This is not a route to claiming O3.** It is a measurement of how",
         "  fast this inverse problem converges toward its information floor.",
         "",
         "  Validation and test splits are untouched. The Phase 2 scaler is",
         "  unchanged. Only the training pool grows.", "",
         f"  {'n':>7s} {'RMS-RGE %':>18s} {'R %':>16s} {'Lc %':>16s} "
         f"{'curve predicts':>15s}"]
    for n in DENSITY_SIZES:
        v = [done[f"{n}|{s}"] for s in ABLATE_SEEDS[:2]
             if f"{n}|{s}" in done]
        if not v:
            continue
        a = np.array([x["agg"] for x in v])
        r = np.array([x["per"][0] for x in v])
        lc = np.array([x["per"][3] for x in v])
        pred = FLOOR + B * n ** (-C)
        L.append(f"  {n:7d} {a.mean():8.3f} +/-{a.std():5.3f} "
                 f"{r.mean():8.3f} +/-{r.std():5.3f} "
                 f"{lc.mean():8.3f} +/-{lc.std():5.3f} {pred:15.3f}")
    L += ["", "  'curve predicts' is the constrained fit from the Phase 6",
          f"  sample-efficiency data: err(n) = {FLOOR} + {B:.3f}*n^(-{C:.4f}),",
          "  with the asymptote pinned at the measured information floor.",
          "  If the measured column tracks it, the floor study and the",
          "  extrapolation are both validated and the flat exponent is real.",
          "",
          "  WHAT TO REPORT IN THE THESIS:",
          "  - headline O3 result stays the 805-sample number (RMS-RGE 15.45 %,",
          "    not met; closed-loop 3.368 %, met)",
          "  - this table as a convergence-rate measurement, labelled as",
          "    surrogate-density and not as added physics",
          "  - the honest conclusion: the geometry-space threshold is",
          "    reachable in principle (floor 13.30 %) but needs roughly 7.7x",
          "    the training pool, and adding analytic rows would not add the",
          "    physics a reviewer would expect behind that claim"]
    txt = "\n".join(L)
    open(os.path.join(resdir, "density_report.txt"), "w").write(txt + "\n")
    print("\n" + txt)


FLOOR_N = 200000          # response-bank size; the floor depends on it, so it
FLOOR_SEED = 7            # is reported alongside every floor number
FLOOR_CHUNK = 20000       # bank is built in chunks so a long run is resumable


# --- Full 3-D ring validation (the Q1 gap: the ring model is unchecked) ----
# Three geometries spanning the radius range. Gaps are deliberately on the
# strong-coupling side: loaded Q falls as coupling rises, and a lower Q needs a
# shorter simulation window to resolve, which is what makes this affordable.
# All at small radius. A 10 um ring needs ~20 GB of solver memory and a 14 um
# ring ~36 GB at this mesh, so spanning the radius range is not possible on a
# 16 GB machine. Varying g and Lc at fixed R is the better test regardless:
# Q_L and IL depend on geometry only through kappa^2, so this exercises the
# coupled-mode assembly directly -- and small R is where bend loss is highest,
# which stresses the loss model hardest. State the small-radius limitation.
RINGVAL_GEOMS = [
    dict(R=5.20, w=450.0, g=170.0, Lc=1.00),
    dict(R=5.20, w=450.0, g=150.0, Lc=2.00),
    dict(R=5.20, w=450.0, g=180.0, Lc=0.60),
]
# Hours per (million cells x picosecond of window), calibrated from this
# project's own first run: 9.58e6 cells, 59 ps window, 14.5 h wall on 6 cores.
RINGVAL_HOURS_PER_MCELL_PS = 14.5 / (9.58 * 59.0)
RINGVAL_T_MAX = 1500e-12     # refuse a geometry needing a longer window
RINGVAL_MARGIN = 3.0         # FWHM must exceed the transform limit by this
RINGVAL_PTS_PER_FWHM = 5.0   # DFT samples across the linewidth, as sub-study 2
RINGVAL_NF_MAX = 60001       # refuse a geometry needing more monitor points
# Floor, not a target. DFT monitor memory scales LINEARLY with frequency
# points, so asking for more than the linewidth needs buys nothing and can
# exhaust system memory before the solver starts. 2001 points over 1500-1600 nm
# is a 0.05 nm grid -- ample for peak finding and for the adjacent-resonance
# FSR -- and the per-geometry requirement is computed on top of it.
RINGVAL_NF_MIN = 2001
# Domain margins for the ring validation, tighter than the production
# DomainConfig defaults (2.0 / 1.5 / 1.0 um). This is a NUMERICS setting, not
# a physics one: it moves the PML closer, and the only thing it can corrupt is
# the result itself if the PML starts interacting with the evanescent field.
# It does not affect comparability with the analytic model, which knows
# nothing about the simulation domain. Override with --margins=x,y,z (um) and
# verify by re-running one geometry at the production margins if memory allows.
RINGVAL_MARGINS = (1.0, 0.9, 0.7)
# Bytes of solver memory per Yee cell, calibrated from a Lumerical memory
# report on this project's own geometry (11.328 GB for ~16.1e6 cells).
# Used only to WARN before launching; the solver's own estimate governs.
RINGVAL_BYTES_PER_CELL = 703.0
# Auto-shutoff is effectively DISABLED for these runs. If the solver stops
# early the real window is shorter than the one the transform limit was
# computed from, the line is under-resolved, and the diagnostic would report
# it as resolved -- the exact error that invalidated sub-study 2. Forcing the
# full window costs runtime and buys a transform limit that is exactly known.
RINGVAL_SHUTOFF = 1e-12


def _ringval_window(q_pred, lam=LAMBDA_TARGET):
    """
    Simulation window needed to resolve a line of loaded Q = q_pred.

    A rectangular window of length T is transform-limited to
    0.886*lambda^2/(c*T). Sub-study 2 was invalidated by measuring that limit
    instead of the resonator, so the window is sized from the PREDICTED Q
    before the run rather than discovered afterwards.
    """
    fwhm = lam / q_pred
    return 0.886 * lam ** 2 / (C_LIGHT * fwhm / RINGVAL_MARGIN)


def _ringval_freqpoints(q_pred, lam=LAMBDA_TARGET):
    """
    Monitor frequency points needed to put RINGVAL_PTS_PER_FWHM samples across
    the linewidth, over the full 1500-1600 nm span. A long enough window with
    too coarse a DFT grid is just as unresolved as a short window.
    """
    fwhm_nm = (lam / q_pred) * 1e9
    span_nm = (LAMBDA_MAX - LAMBDA_MIN) * 1e9
    return int(np.ceil(span_nm / (fwhm_nm / RINGVAL_PTS_PER_FWHM))) + 1


def _ringval_cells(gm, dom, mesh_nm=BEND_MESH_NM):
    """
    Approximate Yee-cell count and solver memory for a ring geometry, so an
    impossible run is refused before Lumerical is launched rather than after
    it puts up a dialog. Deliberately crude: the solver's own estimate is
    authoritative, this only catches the obviously-too-big case.
    """
    half_x = gm["R"] + gm["Lc"] / 2.0 + gm["w"] * 1e-3 / 2.0
    bus_outer = gm["R"] + gm["w"] * 1e-3 + gm["g"] * 1e-3 + gm["w"] * 1e-3 / 2.0
    sx = 2.0 * (half_x + dom.x_margin * 1e6)
    sy = 2.0 * (bus_outer + dom.y_margin * 1e6)
    sz = 0.22 + 2.0 * dom.z_margin * 1e6
    d = mesh_nm * 1e-3
    cells = (sx / d) * (sy / d) * (sz / d)
    return (sx, sy, sz), cells, cells * RINGVAL_BYTES_PER_CELL / 1e9


def run_ringdry(margins=RINGVAL_MARGINS, extra=()):
    """
    Price every candidate geometry WITHOUT launching Lumerical: predicted Q_L,
    the window it forces, monitor points, domain size, solver memory and an
    estimated runtime. Costs seconds and prevents committing days to a
    campaign that cannot fit in memory -- which is how the first plan for this
    study was wrong.
    """
    respond = _make_responder()
    cands = list(RINGVAL_GEOMS) + [dict(R=c[0], w=c[1], g=c[2], Lc=c[3])
                                   for c in extra]
    print(f"\nRING-VALIDATION CANDIDATES   margins "
          f"{margins[0]:g}/{margins[1]:g}/{margins[2]:g} um, mesh "
          f"{BEND_MESH_NM:g} nm\n")
    print(f"  {'R':>5s} {'w':>5s} {'g':>5s} {'Lc':>5s} {'Q_L':>7s} "
          f"{'IL dB':>6s} {'win ps':>7s} {'pts':>6s} {'Mcell':>6s} "
          f"{'GB':>5s} {'~hours':>7s}  fits?")
    print("  " + "-" * 78)
    for gm in cands:
        try:
            pr = respond(gm["R"], gm["w"], gm["g"], gm["Lc"])
            q = float(pr["Q_L"])
            if not np.isfinite(q) or q <= 0:
                raise ValueError("Q_L undefined")
        except Exception as exc:                        # noqa: BLE001
            print(f"  {gm['R']:5.2f} {gm['w']:5.0f} {gm['g']:5.0f} "
                  f"{gm['Lc']:5.2f}   analytic model failed: {exc}")
            continue
        t = _ringval_window(q)
        nfn = _ringval_freqpoints(q)
        nf = int(min(RINGVAL_NF_MAX, max(RINGVAL_NF_MIN, nfn)))
        dom = DomainConfig(x_margin=margins[0]*1e-6, y_margin=margins[1]*1e-6,
                           z_margin=margins[2]*1e-6)
        _, cells, gb = _ringval_cells(gm, dom)
        hrs = RINGVAL_HOURS_PER_MCELL_PS * (cells / 1e6) * (t * 1e12)
        ok = ("yes" if (gb <= 7.5 and t <= RINGVAL_T_MAX
                        and nfn <= RINGVAL_NF_MAX) else "NO")
        print(f"  {gm['R']:5.2f} {gm['w']:5.0f} {gm['g']:5.0f} {gm['Lc']:5.2f} "
              f"{q:7.0f} {float(pr['IL_dB']):6.3f} {t*1e12:7.0f} {nf:6d} "
              f"{cells/1e6:6.1f} {gb:5.1f} {hrs:7.1f}  {ok}")
    print("\n  'fits?' assumes about 8 GB of free RAM. Memory and hours are")
    print("  estimates calibrated on this project's own first run, not")
    print("  guarantees -- the solver's own estimate governs.")
    print("  Add candidates with --try=R,w,g,Lc (repeatable).\n")


def run_ringvalidate(n=None, movie=True, margins=RINGVAL_MARGINS, extra=(),
                     mesh_nm=BEND_MESH_NM, points=None):
    """
    Validate the analytic ring model against full 3-D FDTD.

    Everything else in this work validates the COUPLER (68 out-of-sample
    kappa^2 runs) or the MODE SOLVER (the published-FSR check). The
    coupled-mode assembly that turns kappa^2, n_g and loss into Q_L and IL is
    checked against nothing: closed-loop re-verification re-simulates the
    coupler and then recomputes the ring with the same analytic model that
    produced the labels. Since Q_L and IL carry the whole degeneracy argument,
    that is the one gap a reviewer will find first.

    This simulates the COMPLETE ring in 3-D FDTD and compares the measured
    lambda_res, FSR, Q_L and IL against the analytic prediction for the same
    geometry. Resumable: rerun the same command to continue.
    """
    import json
    resdir = "results" if os.path.isdir("results") else "."
    respond = _make_responder()
    allg = list(RINGVAL_GEOMS) + [dict(R=c[0], w=c[1], g=c[2], Lc=c[3])
                                  for c in extra]
    geoms = allg[:n] if n else allg

    ck = os.path.join(resdir, "ringvalidate.json")
    done = json.load(open(ck)) if os.path.isfile(ck) else {}
    print(f"[ringval] {len(geoms)} geometries, {len(done)} already done")
    print(f"[ringval] mesh {mesh_nm:g} nm"
          + (" (production)" if abs(mesh_nm - BEND_MESH_NM) < 1e-9
             else f" (NOT production {BEND_MESH_NM:g} nm -- convergence test)")
          + f", movie "
          f"{'ON' if movie else 'OFF'}, auto-shutoff {RINGVAL_SHUTOFF:g}")

    for i, gm in enumerate(geoms, 1):
        key = f"{gm['R']:.3f}_{gm['w']:.1f}_{gm['g']:.1f}_{gm['Lc']:.3f}"
        if abs(mesh_nm - BEND_MESH_NM) > 1e-9:
            key += f"_m{mesh_nm:g}"
        if points:
            key += f"_p{int(points)}"
        if key in done:
            print(f"[ringval] {i}/{len(geoms)}: cached")
            continue

        pred = respond(gm["R"], gm["w"], gm["g"], gm["Lc"])
        qp = float(pred["Q_L"])
        if not np.isfinite(qp) or qp <= 0:
            print(f"[ringval] {i}/{len(geoms)}: analytic Q_L is {qp}; skipped")
            continue
        tneed = _ringval_window(qp)
        print(f"\n[ringval] {i}/{len(geoms)}: R={gm['R']:.2f}um "
              f"w={gm['w']:.0f}nm g={gm['g']:.0f}nm Lc={gm['Lc']:.2f}um")
        print(f"[ringval]   analytic: lam {pred['lambda_res_nm']:.3f} nm, "
              f"FSR {pred['FSR_nm']:.3f} nm, Q_L {qp:.0f}, "
              f"IL {pred['IL_dB']:.3f} dB")
        nf_need = _ringval_freqpoints(qp)
        nf = int(min(RINGVAL_NF_MAX, max(RINGVAL_NF_MIN, nf_need)))
        if points:
            nf = int(points)
        print(f"[ringval]   window needed for Q={qp:.0f} at "
              f"{RINGVAL_MARGIN:g}x transform limit: {tneed*1e12:.0f} ps "
              f"({tneed/50e-12:.1f}x the 50 ps default)")
        print(f"[ringval]   monitor points: {nf}  (physics needs {nf_need} "
              f"for {RINGVAL_PTS_PER_FWHM:g} samples across the FWHM;")
        print(f"[ringval]   floor {RINGVAL_NF_MIN}, cap {RINGVAL_NF_MAX}. "
              "DFT memory scales linearly with this.)")
        if nf_need > RINGVAL_NF_MAX:
            print(f"[ringval]   SKIPPED: needs {nf_need} monitor points, over "
                  f"the {RINGVAL_NF_MAX} cap. The line")
            print("[ringval]   is too narrow to sample over a 100 nm span. "
                  "Pick a stronger-coupling")
            print("[ringval]   geometry (smaller g or longer Lc).")
            continue
        if tneed > RINGVAL_T_MAX:
            print(f"[ringval]   SKIPPED: exceeds the {RINGVAL_T_MAX*1e12:.0f} ps "
                  "cap. Pick a stronger-coupling")
            print("[ringval]   geometry (smaller g or longer Lc) -- a line this "
                  "narrow cannot be")
            print("[ringval]   resolved affordably, and an unresolved line "
                  "measures the window.")
            continue

        p_ = MRRParams(R=gm["R"]*1e-6, w=gm["w"]*1e-9,
                       g=gm["g"]*1e-9, Lc=gm["Lc"]*1e-6)
        mesh = MeshConfig.uniform(mesh_nm * 1e-9)
        dom = DomainConfig(movie=movie, dimension="3D", sim_time=tneed,
                           auto_shutoff_min=RINGVAL_SHUTOFF, freq_points=nf,
                           x_margin=margins[0]*1e-6, y_margin=margins[1]*1e-6,
                           z_margin=margins[2]*1e-6)
        (sx, sy, sz), cells, gb = _ringval_cells(gm, dom, mesh_nm)
        print(f"[ringval]   domain {sx:.2f} x {sy:.2f} x {sz:.2f} um at "
              f"{mesh_nm:g} nm -> {cells/1e6:.1f}M cells")
        print(f"[ringval]   estimated solver memory: {gb:.1f} GB  "
              f"(margins {margins[0]:g}/{margins[1]:g}/{margins[2]:g} um)")
        lumapi = load_lumapi()
        t0 = time.time()
        try:
            with lumapi.FDTD(hide=True) as fdtd:
                build_mrr(fdtd, p_, mesh, dom)
                fdtd.save(os.path.abspath(f"ringval_{key}.fsp"))
                fdtd.run()
                lam, T = get_spectrum(fdtd, "monitor_drop")
                # Save AFTER the run so the monitor data is on disk. Without
                # this the .fsp holds only the layout, and a spectrum cannot
                # be re-examined without re-simulating -- which cost this
                # project a 10-hour rerun to test an extraction question.
                try:
                    fdtd.save(os.path.abspath(f"ringval_{key}_done.fsp"))
                except Exception:
                    pass
                np.savez(os.path.join(resdir, f"ringval_{key}_spectrum.npz"),
                         lam=lam, T=T)
            res = analyse_drop(lam, T)
        except Exception as exc:                        # noqa: BLE001
            print(f"[ringval]   FAILED: {exc}  -- re-run to retry")
            continue
        dt = (time.time() - t0) / 60.0

        grid = float(np.min(np.diff(lam))) * 1e9 if lam.size > 1 else np.nan
        fw = res["FWHM"] * 1e9 if res["FWHM"] == res["FWHM"] else np.nan
        tl = 0.886 * (res["lambda_res"] ** 2) / (C_LIGHT * tneed) * 1e9
        done[key] = dict(
            geom=gm, runtime_min=dt,
            sim=dict(lambda_res_nm=res["lambda_res"]*1e9,
                     FSR_nm=res["FSR"]*1e9, Q_L=res["Q_L"],
                     IL_dB=res["IL_dB"], FWHM_nm=fw, n_peaks=res["n_peaks"]),
            ana={k: float(pred[k]) for k in TARGETS},
            window_ps=tneed*1e12, grid_nm=grid, transform_limit_nm=tl,
            freq_points=nf, shutoff=RINGVAL_SHUTOFF, mesh_nm=mesh_nm,
            pts_across_fwhm=(fw/grid if grid == grid and fw == fw else None),
            fwhm_over_tl=(fw/tl if tl and fw == fw else None))
        json.dump(done, open(ck, "w"), indent=1)
        print(f"[ringval]   simulated: lam {res['lambda_res']*1e9:.3f} nm, "
              f"FSR {res['FSR']*1e9:.3f} nm, Q_L {res['Q_L']:.0f}, "
              f"IL {res['IL_dB']:.3f} dB  ({dt:.1f} min)")

    # ---------------- report ------------------------------------------
    L = ["FULL 3-D RING VALIDATION OF THE ANALYTIC MODEL", "=" * 78,
         "The complete ring simulated in 3-D FDTD, compared against the",
         "analytic coupled-mode prediction for the same geometry. This is the",
         "only check in this work that tests Q_L and IL rather than kappa^2 or",
         "the mode solver.", "",
         "RESOLUTION FIRST -- an unresolved line measures the DFT window, not",
         "the resonator. A run with FWHM/transform-limit below "
         f"{RINGVAL_MARGIN:g} or fewer than",
         "5 points across the FWHM is NOT evidence about Q_L and is marked "
         "so.", ""]
    rows = [v for v in done.values()]
    if not rows:
        L += ["  no completed runs yet."]
    else:
        L += [f"  {'R um':>6s} {'window ps':>10s} {'pts/FWHM':>9s} "
              f"{'FWHM/TL':>8s} {'resolved':>9s} {'min':>7s}", "  " + "-"*56]
        for v in rows:
            pf = v.get("pts_across_fwhm"); ft = v.get("fwhm_over_tl")
            ok = (pf is not None and ft is not None
                  and pf >= 5.0 and ft >= RINGVAL_MARGIN)
            L.append(f"  {v['geom']['R']:6.2f} {v['window_ps']:10.0f} "
                     f"{(pf if pf else float('nan')):9.1f} "
                     f"{(ft if ft else float('nan')):8.2f} "
                     f"{'yes' if ok else 'NO':>9s} {v['runtime_min']:7.1f}")
        L += ["", "  per-target comparison (simulated vs analytic)", ""]
        L.append(f"  {'R um':>6s} " + " ".join(
            f"{t:>22s}" for t in TARGETS))
        L.append("  " + "-" * 96)
        errs = {t: [] for t in TARGETS}
        for v in rows:
            pf = v.get("pts_across_fwhm"); ft = v.get("fwhm_over_tl")
            ok = (pf is not None and ft is not None
                  and pf >= 5.0 and ft >= RINGVAL_MARGIN)
            cells = []
            for t in TARGETS:
                sv = v["sim"].get(t); av = v["ana"].get(t)
                if sv is None or av in (None, 0) or sv != sv:
                    cells.append(f"{'--':>22s}"); continue
                e = 100.0 * (sv - av) / av
                # Q_L and IL from an unresolved line are not evidence
                if ok or t in ("lambda_res_nm", "FSR_nm"):
                    errs[t].append(abs(e))
                cells.append(f"{sv:9.3f}/{av:<8.3f}{e:+5.1f}%")
            L.append(f"  {v['geom']['R']:6.2f} " + " ".join(cells))
        L += ["", "  mean |relative error| over runs counted as evidence:"]
        for t in TARGETS:
            if errs[t]:
                L.append(f"    {t:16s} {np.mean(errs[t]):7.3f} %   "
                         f"(n = {len(errs[t])})")
            else:
                L.append(f"    {t:16s} {'no admissible run':>17s}")
        L += ["", "  HOW TO READ THIS",
              "  lambda_res and FSR test the mode solver and the length "
              "convention,",
              "  which the published-device check already covers. Q_L and IL "
              "are the",
              "  NEW information: they are the first independent test of the "
              "coupled-",
              "  mode assembly, and they are what the degeneracy argument "
              "rests on.",
              "  A systematic offset in Q_L or IL does not by itself "
              "invalidate the",
              "  comparison between models -- every model shares it -- but it "
              "must be",
              "  reported, and a large one would weaken the claim that Q_L "
              "and IL act",
              "  on geometry only through theta."]
    txt = "\n".join(L)
    open(os.path.join(resdir, "ringvalidate.txt"), "w").write(txt + "\n")
    print("\n" + txt)


def run_validate():
    """
    Objective O1, second half -- external validation against a published,
    experimentally characterised device (Phase 1.6 sub-study 3).

    The computation already runs inside --modesweep, but only prints to the
    console, so no auditable record of it survives. This recomputes it from
    the cached mode sweep (no Lumerical) and writes it to disk, which is what
    a published repo needs.
    """
    f = _find(MODESWEEP_CSV)
    rows = list(csv.DictReader(open(f, newline="")))
    sel = [r for r in rows if float(r["w_nm"]) == P11_W_NM
           and float(r["lam_nm"]) == 1550.0]
    if not sel:
        raise ValueError(f"{f} has no w={P11_W_NM:g} nm, 1550 nm row; "
                         "rerun --modesweep")
    ng = float(sel[0]["n_g"]); ne = float(sel[0]["n_eff"])
    Lm = 2 * np.pi * P11_R_UM * 1e-6
    fsr = (1550e-9) ** 2 / (ng * Lm) * 1e9
    err = abs(fsr - P11_FSR_NM) / P11_FSR_NM * 100.0
    ng_implied = (1550e-9) ** 2 / (P11_FSR_NM * 1e-9 * Lm)

    L = ["OBJECTIVE O1 (second half) -- EXTERNAL LITERATURE VALIDATION",
         "=" * 76,
         "Phase 1.6 sub-study 3. Recomputed from the cached mode sweep, so it",
         f"is reproducible without re-running Lumerical. Source: {f}", "",
         f"  reference device   : Tang et al. [11], {P11_W_NM:g} x 220 nm SOI "
         "TE ring,",
         f"                       R = {P11_R_UM:g} um, circular (Lc = 0)",
         f"  measured FSR       : {P11_FSR_NM:.3f} nm   (an experimental "
         "figure)",
         f"  implied n_g        : {ng_implied:.4f}   (from the measured FSR)",
         "",
         f"  this work's n_g    : {ng:.5f}   (mode sweep at w = "
         f"{P11_W_NM:g} nm, 1550 nm)",
         f"  this work's n_eff  : {ne:.5f}",
         f"  predicted FSR      : {fsr:.4f} nm   (Eq. 4, L = 2*pi*R)",
         "",
         f"  relative error     : {err:.2f} %",
         f"  O1 requires <= 10 %: " + ("PASS" if err <= 10.0 else "FAIL"),
         "",
         "  SCOPE AND CAVEATS -- state these when citing this validation:",
         "  1. Only FSR is validated. FSR depends on n_g and round-trip length",
         "     alone, so this tests the mode solver and the length convention,",
         "     NOT the coupling model, the bend-loss model, or Q_L and IL.",
         f"  2. R = {P11_R_UM:g} um is outside this work's 5-15 um design range. "
         "n_eff is",
         "     fitted from a straight-waveguide sweep and is a function of",
         "     (lambda, w) only, so R enters solely through L -- but the device",
         "     is correspondingly less bent than the training range.",
         "  3. Tang et al.'s Q ~ 3e5 is a stated ASSUMPTION in their "
         "scalability",
         "     projection, not a measurement, and must not be used as a "
         "validation",
         "     target. Their measured figures are FSR, propagation loss and",
         "     extinction ratio.",
         "  4. NOT independent of the roughness input: ROUGHNESS_DB_CM = "
         f"{ROUGHNESS_DB_CM:g}",
         "     dB/cm is itself taken from this same paper. That value does not",
         "     enter the FSR prediction, so this particular comparison is",
         "     unaffected -- but the device is not an independent source for",
         "     the loss model.",
         "  5. The nominal n_g = 4.2 used for Phase 1.2 design-space reasoning",
         "     was also justified by citing this device. That nominal value is",
         "     not used in any dataset label; the labels use the fitted sweep",
         "     above. The overlap should still be disclosed."]
    resdir = "results" if os.path.isdir("results") else "."
    txt = "\n".join(L)
    open(os.path.join(resdir, "o1_validation.txt"), "w").write(txt + "\n")
    import json
    json.dump(dict(device="Tang et al. [11]", w_nm=P11_W_NM, R_um=P11_R_UM,
                   measured_fsr_nm=P11_FSR_NM, implied_ng=ng_implied,
                   sim_ng=ng, sim_neff=ne, predicted_fsr_nm=fsr,
                   rel_err_pct=err, passes=bool(err <= 10.0)),
              open(os.path.join(resdir, "o1_validation.json"), "w"), indent=1)
    print("\n" + txt)


def run_floor(n_bank=FLOOR_N):
    """
    INFORMATION FLOOR -- the best RMS-RGE any inverse model could achieve.

    Build a large bank of geometries, compute each one's response with the
    SAME analytic responder that labelled the dataset, then for every test
    target find the bank geometry whose RESPONSE is closest to it. If that
    geometry differs from the true one, no model -- however good -- can tell
    them apart from the response alone. The geometric distance between them
    is therefore a lower bound on achievable error.

    This replaces hard-coded floor constants with a computation a reader can
    rerun. The floor depends on the bank size, which is reported with it.
    """
    import json
    resdir = "results" if os.path.isdir("results") else "."
    Gr_te, Rr_te = _load_split_raw("test")
    respond = _make_responder()
    rng = np.random.default_rng(FLOOR_SEED)
    lo = np.array([5.0, 400.0, 150.0, 0.0])
    hi = np.array([15.0, 500.0, 350.0, 3.0])

    ckpt = os.path.join(resdir, "floor_bank.npz")
    if os.path.isfile(ckpt):
        z = np.load(ckpt)
        G, Rsp = z["G"], z["R"]
        print(f"[floor] resuming: {len(G)} of {n_bank} already built")
    else:
        G = np.zeros((0, 4)); Rsp = np.zeros((0, 4))

    if len(G) > n_bank:
        G = G[:n_bank]; Rsp = Rsp[:n_bank]
        print(f"[floor] using the first {n_bank} of the cached bank")

    while len(G) < n_bank:
        k = min(FLOOR_CHUNK, n_bank - len(G))
        cand = lo + (hi - lo) * rng.random((k, 4))
        gs, rs = [], []
        for row in cand:
            try:
                v = respond(*row)
            except Exception:
                continue
            # apply the SAME quality gate the dataset used: a flagged row
            # (no half-max, T_drop > 1) was dropped in Phase 2, so the bank
            # must not contain geometries the dataset would have excluded
            if isinstance(v, dict) and v.get("flag"):
                continue
            v = np.asarray([v[t] for t in TARGETS], dtype=float) \
                if isinstance(v, dict) else np.asarray(v, dtype=float)
            if v.shape != (4,) or not np.all(np.isfinite(v)):
                continue
            gs.append(row); rs.append(v)
        if gs:
            G = np.vstack([G, np.array(gs)])
            Rsp = np.vstack([Rsp, np.array(rs)])
        np.savez(ckpt, G=G, R=Rsp)
        print(f"[floor] bank {len(G)}/{n_bank}")

    # response distance, scaled per target so no one target dominates
    sd = Rsp.std(axis=0); sd[sd == 0] = 1.0
    Bn = Rsp / sd

    # THE ESTIMATOR MATTERS. Taking the single nearest bank geometry (k = 1)
    # returns one arbitrary member of the degenerate set, which sits sqrt(2)
    # further from the truth than the set's centre. A regression model does
    # not do that -- it learns E[geometry | response]. So the floor is the
    # error of the CONDITIONAL MEAN, estimated by averaging the k nearest
    # matches. Reported against k so the plateau is visible rather than
    # assumed: too small is the single-draw estimator, too large blends in
    # geometries that are not actually degenerate with the target.
    KS = [1, 5, 10, 25, 50, 100, 200, 400]
    order = np.empty((len(Gr_te), max(KS)), dtype=np.int64)
    for i in range(len(Gr_te)):
        d = np.sum((Bn - Rr_te[i] / sd) ** 2, axis=1)
        order[i] = np.argpartition(d, max(KS))[:max(KS)][
            np.argsort(d[np.argpartition(d, max(KS))[:max(KS)]])]
    curve = []
    for k in KS:
        est = G[order[:, :k]].mean(axis=1)
        curve.append((k,) + _rms_rge(est, Gr_te))
    kbest, agg, per, p12 = min(curve, key=lambda t: t[1])

    # refit the learning curve with the asymptote pinned at this floor
    bc = None
    try:
        rep = _find("phase6_report.txt")
        ns = ys = None
        for ln in open(rep):
            t = ln.split()
            if t[:1] == ["method"] and len(t) >= 6 and t[1].isdigit():
                ns = np.array([float(x) for x in t[1:]])
            if ln.strip().startswith("naive DNN inverse") and ns is not None:
                ys = np.array([float(x) for x in ln.split()[3:]])
                break
        if ns is not None and ys is not None and len(ns) == len(ys):
            m = ys > agg
            if m.sum() >= 2:
                cc = np.polyfit(np.log(ns[m]), np.log(ys[m] - agg), 1)
                bc = (float(np.exp(cc[1])), float(-cc[0]))
    except Exception:
        bc = None

    L = ["INFORMATION FLOOR -- LOWER BOUND ON ACHIEVABLE RMS-RGE", "=" * 76,
         f"bank: {len(G)} geometries, uniform over the Phase 1.2 design box, "
         f"seed {FLOOR_SEED}.",
         "Each test target is matched to the bank geometry with the nearest",
         "response (per-target standard-deviation scaling). The geometric",
         "distance to the TRUE geometry bounds what any inverse model can do",
         "from the response alone.", "",
         f"  {'aggregate RMS-RGE bound':28s} {agg:8.3f} %   <-- the bounded "
         "quantity",
         "",
         "  per-parameter errors OF THIS ESTIMATOR (not per-parameter bounds:",
         "  the estimator minimises the AGGREGATE, so a model that trades one",
         "  parameter against another may legitimately beat any single row):",
         f"  {'  R':28s} {per[0]:8.3f} %",
         f"  {'  w':28s} {per[1]:8.3f} %",
         f"  {'  g':28s} {per[2]:8.3f} %",
         f"  {'  Lc (range-normalised)':28s} {per[3]:8.3f} %",
         f"  {'  R,w,g only (Paper 12 conv.)':28s} {p12:8.3f} %", "",
         f"  estimator: conditional mean over the k = {kbest} nearest "
         f"responses.", "",
         f"    {'k':>5s} {'RMS-RGE %':>11s}   (k=1 is one arbitrary member of "
         "the degenerate set;",
         "                        a model learns the centre, not a member)"]
    for k, a_, _p, _q in curve:
        L.append(f"    {k:5d} {a_:11.3f}" + ("   <-- bound" if k == kbest else ""))
    if kbest == curve[-1][0] or kbest == curve[0][0]:
        L += ["", "    WARNING: the minimum sits at the EDGE of the k grid, so "
              "the true",
              "    minimum may lie outside it. Widen the grid before quoting "
              "this."]
    L += ["",
         "  This is the error of a SPECIFIC estimator (conditional-mean "
         "lookup over",
         "  a finite bank), so it is an UPPER BOUND on the true information "
         "floor.",
         "  The asymmetry matters: measuring it BELOW a threshold proves the "
         "floor",
         "  is below that threshold, but measuring it ABOVE proves nothing. It",
         "  falls as the bank grows, so the bank size is part of the result and",
         "  no value from a single bank size should be quoted as 'the floor'."]
    if bc:
        L += ["", f"  learning curve refit with asymptote pinned here:",
              f"    err(n) = {agg:.3f} + {bc[0]:.3f} * n^(-{bc[1]:.4f})"]
    else:
        L += ["", "  learning-curve refit skipped: could not read the naive-"
              "DNN inverse row",
              "  of the sample-efficiency table from phase6_report.txt. Run",
              "  --phase6 first."]
    json.dump(dict(bank=int(len(G)), seed=FLOOR_SEED, agg=agg, per=per,
                   p12=p12, fit=bc, k=kbest,
                   curve=[[c[0], c[1]] for c in curve]),
              open(os.path.join(resdir, "floor.json"), "w"), indent=1)
    txt = "\n".join(L)
    open(os.path.join(resdir, "floor.txt"), "w").write(txt + "\n")
    print("\n" + txt)


def _floor_at(G, Rsp, Gr_te, Rr_te,
              ks=(1, 2, 3, 4, 5, 7, 10, 25, 50, 100, 200, 400)):
    """Conditional-mean floor for a given bank; returns (best_k, agg, per, p12,
    curve). Shared by --floor and --floorconv so both use identical logic."""
    sd = Rsp.std(axis=0); sd[sd == 0] = 1.0
    Bn = Rsp / sd
    kmax = min(max(ks), len(G) - 1)
    ks = [k for k in ks if k <= kmax] or [1]
    order = np.empty((len(Gr_te), kmax), dtype=np.int64)
    for i in range(len(Gr_te)):
        d = np.sum((Bn - Rr_te[i] / sd) ** 2, axis=1)
        pick = np.argpartition(d, kmax)[:kmax]
        order[i] = pick[np.argsort(d[pick])]
    curve = [(k,) + _rms_rge(G[order[:, :k]].mean(axis=1), Gr_te) for k in ks]
    kb, agg, per, p12 = min(curve, key=lambda t: t[1])
    return kb, agg, per, p12, curve


def run_floorconv():
    """
    Is the information floor CONVERGED, or still falling with bank size?

    The floor is estimated from a finite bank, so it is an UPPER estimate of
    the true floor: more geometries can only find closer response matches.
    Claiming a threshold is unreachable in principle therefore requires
    showing the estimate has stopped moving. This subsamples the cached bank
    at several sizes and reports the trend against O3's 15 % bar. It does NOT
    report an extrapolated asymptote unless that asymptote is identifiable;
    with a handful of nearly-flat points it is not, and a grid search for it
    simply runs to the edge of whatever range it is given.
    """
    import json
    resdir = "results" if os.path.isdir("results") else "."
    ck = os.path.join(resdir, "floor_bank.npz")
    if not os.path.isfile(ck):
        raise FileNotFoundError("run --floor first to build the response bank")
    z = np.load(ck); G0, R0 = z["G"], z["R"]
    Gr_te, Rr_te = _load_split_raw("test")
    sizes, n = [], 12500
    while n <= len(G0):
        sizes.append(n); n *= 2
    if sizes and sizes[-1] != len(G0):
        sizes.append(len(G0))
    print(f"[floorconv] cached bank {len(G0)}; sizes {sizes}")

    rows = []
    for n in sizes:
        kb, agg, per, p12, _ = _floor_at(G0[:n], R0[:n], Gr_te, Rr_te)
        rows.append((n, agg, kb))
        print(f"[floorconv] N={n:7d}  floor {agg:7.3f} %  (k={kb})")

    L = ["INFORMATION FLOOR -- BANK-SIZE CONVERGENCE", "=" * 74,
         "A finite bank can only OVERESTIMATE the floor: more geometries find",
         "closer response matches. So the question is whether the estimate has",
         "stopped falling. If it has not, no claim about a threshold being",
         "unreachable in principle is supportable.", "",
         f"  {'bank N':>9s} {'floor %':>10s} {'best k':>8s}", "  " + "-" * 29]
    for n, a_, kb in rows:
        L.append(f"  {n:9d} {a_:10.3f} {kb:8d}")

    # Report the MEASURED trend. Do not fit a 3-parameter asymptote to a
    # handful of nearly-flat points: the asymptote is not identifiable from
    # such data, and a grid search for it will simply run to the edge of
    # whatever range it is given and report that edge as a result.
    fi = None
    N = np.array([r[0] for r in rows], float)
    F = np.array([r[1] for r in rows], float)
    slope = None
    if len(rows) >= 3:
        # per-e-fold descent over the upper half, where the coarse-bank
        # transient has died away
        h = len(rows) // 2
        slope = float((F[h] - F[-1]) / np.log(N[-1] / N[h]))
        L += ["", f"  measured descent, N={int(N[h])} -> N={int(N[-1])}: "
              f"{F[h]-F[-1]:.3f} pp over {np.log(N[-1]/N[h])/np.log(10):.2f} "
              "decades",
              f"  that is {slope:.4f} pp per e-fold of bank size.", "",
              f"  floor estimate at the largest bank tested: {F[-1]:.3f} % "
              "(an UPPER bound)"]

        # identifiability check, stated rather than hidden
        lo_edge = max(0.0, F.min() - 8.0)
        bestj = None
        for cand in np.linspace(lo_edge, F.min() - 1e-4, 4000):
            d = F - cand
            if np.any(d <= 0):
                continue
            cc = np.polyfit(np.log(N), np.log(d), 1)
            r = np.log(d) - np.polyval(cc, np.log(N))
            ss = float(np.sum(r ** 2))
            if bestj is None or ss < bestj[0]:
                bestj = (ss, cand)
        edge = bestj is not None and abs(bestj[1] - lo_edge) < 0.05
        L += ["", "  ASYMPTOTE IDENTIFIABILITY"]
        if edge:
            L += ["    NOT IDENTIFIABLE. A 3-parameter fit floor_inf + a*N^(-b)",
                  f"    drives floor_inf to {bestj[1]:.3f} %, which is the edge of "
                  "the search",
                  "    range -- the fit wants to go lower still. These points do "
                  "not",
                  "    constrain the asymptote, so NO extrapolated floor is "
                  "reported.",
                  "    Enlarge the bank (--floor --nbank=N) to constrain it."]
        else:
            fi = float(bestj[1])
            L += [f"    identifiable: floor_inf = {fi:.3f} % (fit interior to "
                  "the search range)"]

        # optimal k drifting with N means the estimator's own bias is still
        # unwinding, so no rate from this table describes the true floor
        kk = [r[2] for r in rows]
        if kk[-1] != kk[0]:
            L += ["", "  ESTIMATOR CONVERGENCE WARNING",
                  f"    the best k drifts from {kk[0]} to {kk[-1]} across the "
                  "sweep. A shifting",
                  "    optimum means the finite-sample bias of the estimator is "
                  "still",
                  "    unwinding, so the descent rate above is a property of the "
                  "ESTIMATOR,",
                  "    not of the floor. Treat it as not yet converged and do "
                  "not quote a",
                  "    rate or extrapolate from it."]

        if slope > 0 and F[-1] > 15.0:
            need = float(np.exp((F[-1] - 15.0) / slope)) * N[-1]
            L += ["", "  WHAT IT WOULD TAKE TO REACH O3's 15 % BAR",
                  f"    continuing the measured descent log-linearly, the "
                  f"estimate would",
                  f"    cross 15 % at a bank of about {need:.2e} geometries "
                  f"({need/N[-1]:.3g}x the",
                  "    largest tested). That is an UNJUSTIFIED extrapolation and "
                  "is quoted",
                  "    only to show the scale, not as a prediction."]

        # the achieved best, read from disk rather than hard-coded
        ach = None
        try:
            tv = json.load(open(_find("theta_vs_closedloop.json")))
            ach = min(float(v["rge"]) for v in tv.values())
        except Exception:
            ach = None

        L += ["", "  AGAINST O3's 15 % RMS-RGE BAR -- WHAT THE DATA SUPPORTS",
              f"    The estimate is {F[-1]:.3f} % at N={int(N[-1])}. It is the "
              "error of a",
              "    specific predictor, hence an UPPER BOUND on the true floor."]
        if F[-1] <= 15.0:
            L += [f"    It is BELOW 15 %, and an upper bound below a threshold "
                  "PROVES the",
                  "    true floor is below that threshold. So O3's geometric bar "
                  "is above",
                  "    the information floor: O3 is REACHABLE IN PRINCIPLE, and "
                  "the",
                  "    shortfall is a model-and-data result, not a fundamental "
                  "limit."]
            if ach is not None:
                L += ["", f"    best achieved RMS-RGE: {ach:.3f} % -> at least "
                      f"{ach-F[-1]:.3f} pp of headroom",
                      "    remains between the best model and the bound."]
        else:
            L += [f"    It is ABOVE 15 %, which proves NOTHING either way: an "
                  "upper bound",
                  "    above a threshold cannot show the floor is above it. On "
                  "this",
                  "    evidence the question is OPEN."]
            if ach is not None:
                L += ["", f"    best achieved RMS-RGE: {ach:.3f} % vs bound "
                      f"{F[-1]:.3f} % -- the model is",
                      f"    within {abs(ach-F[-1]):.3f} pp of the bound AT THIS "
                      "BANK SIZE, which is a",
                      "    bank-dependent statement and must be quoted with N."]
    else:
        L += ["", "  too few bank sizes to say anything about convergence."]
    json.dump(dict(rows=[[r[0], r[1], r[2]] for r in rows], floor_inf=fi,
                   slope_per_efold=slope, upper_bound=float(F[-1])),
              open(os.path.join(resdir, "floor_convergence.json"), "w"), indent=1)
    txt = "\n".join(L)
    open(os.path.join(resdir, "floor_convergence.txt"), "w").write(txt + "\n")
    print("\n" + txt)


def run_thetaall():
    """
    Geometry-space error and theta-space error for every inverse model, side
    by side, against whatever closed-loop results exist on disk.

    The naive DNN and xgb_norm sit within 0.33 pp of each other in RMS-RGE
    yet differ by ~3.6x in closed-loop response error. If the degeneracy
    explanation is right, the models that recover theta best should be the
    ones whose predictions reproduce the requested response -- and RMS-RGE
    should NOT order them. This measures that rather than assuming it.
    """
    import json
    torch = _torch()
    outdir = "data" if os.path.isdir("data") else "."
    resdir = "results" if os.path.isdir("results") else "."
    sc = json.load(open(os.path.join(outdir, "scaler.json")))["scaler"]
    _, Rn_te, _ = _load_split("test")
    Gr_te, _ = _load_split_raw("test")
    p5p = os.path.join(resdir, "phase5_results.json")
    p5 = json.load(open(p5p)) if os.path.isfile(p5p) else {}

    preds = {}
    for m in ("naive", "tandem"):
        f = os.path.join(resdir, f"phase4_{m}_inverse.pt")
        if os.path.isfile(f):
            P, _ = _dnn_predict(torch, f, Rn_te)
            preds[m] = _geo_physical(P, sc)
    for m in ("rf_raw", "rf_norm", "xgb_raw", "xgb_norm"):
        if f"inverse_{m}" in p5:
            preds[m] = np.array(p5[f"inverse_{m}"]["pred"])

    # closed-loop numbers already measured, if any
    cl = {}
    for m in preds:
        f = os.path.join(resdir, f"closedloop_{m}.txt")
        if not os.path.isfile(f):
            continue
        for ln in open(f):
            if "RMS closed-loop response error" in ln:
                try:
                    cl[m] = float(ln.split("=")[1].split("%")[0])
                except Exception:
                    pass

    L = ["GEOMETRY ERROR vs THETA ERROR vs CLOSED-LOOP ERROR", "=" * 78,
         "theta = kappa0(w) exp(-gamma(w) g) (Lc + B sqrt(2 pi R / gamma)) "
         "-- the single",
         "coupling phase through which g and Lc act. Two geometries with the "
         "same theta",
         "have the same response, so an error that moves ALONG a theta level "
         "set is a",
         "different valid design; one that moves ACROSS it is a wrong "
         "answer.", "",
         f"  {'model':10s} {'RMS-RGE %':>10s} {'theta MAPE %':>13s} "
         f"{'theta<=5%':>10s} {'closed-loop %':>14s}",
         "  " + "-" * 62]
    L[-2] = (f"  {'model':10s} {'RMS-RGE %':>10s} {'theta MAPE %':>13s} "
             f"{'buildable':>10s} {'theta|built %':>14s} "
             f"{'closed-loop %':>14s}")
    L[-1] = "  " + "-" * 78
    rows = {}
    for m, P in preds.items():
        d = _degeneracy_check(P, Gr_te, m)
        rge = _rms_rge(P, Gr_te)[0]   # aggregate, not the per-parameter list
        # A prediction outside the design box cannot be built, and theta is
        # arithmetically defined there but physically meaningless (Lc < 0).
        # Closed-loop error is measured ONLY on buildable predictions, so the
        # fair comparison restricts theta the same way.
        ins = np.ones(len(P), dtype=bool)
        for j, (lo, hi) in enumerate(((5.0, 15.0), (400., 500.),
                                      (150., 350.), (0., 3.))):
            ins &= (P[:, j] >= lo) & (P[:, j] <= hi)
        frac = 100.0 * float(np.mean(ins))
        tb = (_degeneracy_check(P[ins], Gr_te[ins], m)["theta_mape"]
              if ins.any() else float("nan"))
        rows[m] = dict(rge=rge, theta=d["theta_mape"], w5=d["within5"],
                       built=frac, theta_built=tb, cl=cl.get(m))
        c = f"{cl[m]:14.3f}" if m in cl else f"{'not measured':>14s}"
        L.append(f"  {m:10s} {rge:10.3f} {d['theta_mape']:13.3f} "
                 f"{frac:9.1f}% {tb:14.3f} {c}")

    have = {m: r for m, r in rows.items() if r["cl"] is not None}
    L += ["", f"  closed-loop measured for {len(have)} of {len(rows)} models."]

    # ---- ordering is NOT a discriminating test ---------------------------
    # With few models any metric that happens to rank them correctly scores a
    # match, including one that cannot predict the magnitude at all. What a
    # usable metric must do is PREDICT closed-loop error, so the test is
    # whether closed-loop/metric is constant.
    if len(have) >= 3:
        L += ["", "  PREDICTIVE POWER (the discriminating test)",
              "  A metric is usable if closed-loop error is proportional to "
              "it, i.e.",
              "  if closed-loop/metric is constant. Ordering alone proves "
              "nothing:",
              "  a metric can rank models correctly and still be unable to "
              "say how",
              "  much worse one is.", "",
              f"    {'metric':10s} {'ratio closed-loop/metric':>34s} "
              f"{'spread':>9s}"]
        best = {}
        for key, nm in (("theta_built", "theta|built"), ("rge", "RMS-RGE")):
            rt = [have[m]["cl"] / have[m][key] for m in have]
            spread = max(rt) / min(rt)
            best[key] = spread
            if key == "theta_built":
                best["theta"] = spread
            L.append(f"    {nm:10s} "
                     + "  ".join(f"{v:.3f}" for v in rt).rjust(34)
                     + f" {spread:8.2f}x")
        win = "theta|built" if best["theta"] < best["rge"] else "RMS-RGE"
        L += ["", f"    closed-loop error is proportional to {win} "
              f"({min(best.values()):.2f}x spread);",
              f"    the other metric's constant varies by "
              f"{max(best.values()):.2f}x and so cannot",
              "    predict response-space error from geometry-space error."]

    # ---- which unmeasured model would discriminate hardest ---------------
    # A pair the two metrics RANK DIFFERENTLY is a direct test: the metrics
    # make opposite predictions, so one run decides between them.
    if len(have) >= 2:
        todo = [m for m in rows if rows[m]["cl"] is None]
        clean = {m: r for m, r in have.items() if r["built"] >= 99.9}
        if not clean:
            clean = have
        inv = []
        for m in todo:
            for b in clean:
                # scale from the MEASURED model b under each metric -- a
                # pairwise prediction, not a global constant, since RMS-RGE
                # has no constant to speak of
                pt = (clean[b]["cl"] * rows[m]["theta_built"]
                      / rows[b]["theta_built"])
                pg = clean[b]["cl"] * rows[m]["rge"] / rows[b]["rge"]
                if ((rows[m]["theta_built"] - rows[b]["theta_built"]) *
                        (rows[m]["rge"] - rows[b]["rge"]) < 0):
                    inv.append((m, b, abs(pt - pg), pt, pg))
        if inv:
            inv.sort(key=lambda t: -t[2])
            L += ["", "  NEXT RUN THAT WOULD DECIDE IT",
                  "  These unmeasured models are ranked DIFFERENTLY by the two "
                  "metrics,",
                  "  so the metrics predict different closed-loop errors and "
                  "one run",
                  "  separates them. Competing predictions:", ""]
            seen = set()
            for m, b, gap, pt, pg in inv:
                if m in seen:
                    continue
                seen.add(m)
                flag = ""
                if (pt <= 20.0) != (pg <= 20.0):
                    flag = "  <-- the two metrics disagree on O3 itself"
                L.append(f"    {m:10s} theta predicts {pt:6.2f} %   "
                         f"RMS-RGE predicts {pg:6.2f} %   "
                         f"(scaled from {b}, gap {gap:5.2f} pp){flag}")
            L += ["", f"    run the largest-gap model first: "
                  f"py -3 mrr_template.py --closedloop --{inv[0][0]} --n=20"]
    txt = "\n".join(L)
    open(os.path.join(resdir, "theta_vs_closedloop.txt"), "w").write(txt + "\n")
    json.dump(rows, open(os.path.join(resdir, "theta_vs_closedloop.json"), "w"),
              indent=1)
    print("\n" + txt)


def run_seedcheck(seeds=(42, 43, 44, 45, 46)):
    """
    Retrain the SELECTED forward configuration over several seeds and report
    per-target accuracy as mean +/- sd. A single run cannot tell a genuine
    threshold crossing from initialisation noise, and the Phase 3 criterion
    (R2 >= 0.90 AND MAPE <= 10%) is a hard threshold -- so a target sitting
    near 10% has to be measured, not read off one run.
    """
    import json
    torch = _torch()
    outdir = "data" if os.path.isdir("data") else "."
    resdir = "results" if os.path.isdir("results") else "."
    sc = json.load(open(os.path.join(outdir, "scaler.json")))["scaler"]
    ck = torch.load(os.path.join(resdir, "phase3_forward_model.pt"),
                    weights_only=False)
    Xtr, Ytr, _ = _load_split("train")
    Xva, Yva, _ = _load_split("val")
    Xte, Yte, _ = _load_split("test")
    print(f"[seedcheck] config: depth {ck['depth']} width {ck['width']} "
          f"lr {ck['lr']:g} wd {ck['wd']:g}")
    print(f"[seedcheck] {len(seeds)} seeds {list(seeds)}\n")

    rows = {k: {"r2": [], "mape": []} for k in TARGETS}
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        net, _ = _build_mlp(torch, ck["depth"], ck["width"])
        _train_mlp(torch, net, Xtr, Ytr, Xva, Yva, ck["lr"], ck["wd"])
        net.eval()
        with torch.no_grad():
            P = net(torch.from_numpy(Xte)).numpy()
        line = []
        for j, k in enumerate(TARGETS):
            yt = _unscale(Yte[:, j].astype(np.float64), sc[k])
            yp = _unscale(P[:, j].astype(np.float64), sc[k])
            r2, mp, _ = _metrics(yt, yp)
            rows[k]["r2"].append(r2); rows[k]["mape"].append(mp)
            line.append(f"{k.split('_')[0]} {mp:.3f}")
        print(f"[seedcheck] seed {sd}: " + "  ".join(line))

    L = ["FORWARD MODEL -- SEED STABILITY OF THE PHASE 3 CRITERION", "=" * 74,
         f"config: depth {ck['depth']}, width {ck['width']}, "
         f"lr {ck['lr']:g}, weight decay {ck['wd']:g}",
         f"seeds: {list(seeds)}   (architecture and data identical "
         "throughout)", "",
         f"  {'target':16s} {'R2':>18s} {'MAPE %':>18s} {'passes':>8s}",
         "  " + "-" * 64]
    npass = 0
    for k in TARGETS:
        r2 = np.array(rows[k]["r2"]); mp = np.array(rows[k]["mape"])
        ok = int(np.sum((r2 >= 0.90) & (mp <= 10.0)))
        npass += 1 if ok > len(seeds) / 2 else 0
        L.append(f"  {k:16s} {r2.mean():9.4f} +/-{r2.std():6.4f} "
                 f"{mp.mean():9.3f} +/-{mp.std():6.3f} {ok:4d}/{len(seeds)}")
    L += ["", "  'passes' counts seeds meeting BOTH R2 >= 0.90 and "
          "MAPE <= 10%.", ""]
    # the decisive question: is any target straddling the 10% line?
    for k in TARGETS:
        mp = np.array(rows[k]["mape"])
        if mp.min() <= 10.0 <= mp.max():
            L += [f"  *** {k} STRADDLES the 10 % threshold: "
                  f"{mp.min():.3f} to {mp.max():.3f} across seeds.",
                  "      The Phase 3 verdict for this target is decided by "
                  "initialisation,",
                  "      not by the model. Report the mean and spread, not a "
                  "single run."]
    L += ["", f"  targets passing on a majority of seeds: {npass} of 4",
          f"  Phase 3 criterion (>= 3 of 4): "
          + ("MET" if npass >= 3 else "NOT MET")]
    txt = "\n".join(L)
    open(os.path.join(resdir, "phase3_seedcheck.txt"), "w").write(txt + "\n")
    json.dump({k: rows[k] for k in TARGETS},
              open(os.path.join(resdir, "phase3_seedcheck.json"), "w"), indent=1)
    print("\n" + txt)


def run_kappafit(folds=8):
    """
    Cross-validated assessment of the kappa^2 coupling model (Eq. 1.6).
    Refits the 5-parameter model on each training fold and predicts the
    held-out fold, so the reported error is out-of-sample. Writes
    kappa2_model.json (coefficients + CV statistics) and a parity plot.
    """
    import json
    from scipy.optimize import least_squares
    resdir = "results" if os.path.isdir("results") else "."
    path = _find("couplerset_3d.csv")
    C = np.array([[float(r[k]) for k in
                   ("R_um", "w_nm", "g_nm", "Lc_um", "kappa2")]
                  for r in csv.DictReader(open(path))])
    R, w, g, Lc, k2 = C.T
    n = len(k2)
    th = np.arcsin(np.sqrt(np.clip(k2, 0, 1)))

    def km(q, R, w, g, Lc):
        lk0, a1, g0, g1, B = q
        dw = w - 450.0
        gam = g0 + g1 * dw
        return np.exp(lk0 + a1 * dw) * np.exp(-gam * g) * (
            Lc + B * np.sqrt(2 * np.pi * R * 1e3 / gam) * 1e-3)

    def fit(idx):
        return least_squares(
            lambda q: np.log(km(q, R[idx], w[idx], g[idx], Lc[idx]))
            - np.log(th[idx]),
            [0, 0, .008, 3e-5, 1], max_nfev=40000).x

    full = fit(np.arange(n))
    print(f"[kappafit] {n} coupler simulations, "
          f"kappa^2 {k2.min():.6f} to {k2.max():.6f}")

    rng = np.random.default_rng(COUPLERSET_SEED)
    order = rng.permutation(n)
    pred = np.empty(n)
    for f in range(folds):
        test = order[f::folds]
        train = np.setdiff1d(order, test)
        q = fit(train)
        pred[test] = np.sin(np.minimum(km(q, R[test], w[test], g[test],
                                          Lc[test]), np.pi / 2)) ** 2
    err = 100.0 * (pred - k2) / k2
    mape = float(np.mean(np.abs(err)))
    r2 = float(1 - np.sum((k2 - pred) ** 2) / np.sum((k2 - k2.mean()) ** 2))
    w10 = float(100 * np.mean(np.abs(err) <= 10))
    w5 = float(100 * np.mean(np.abs(err) <= 5))

    L = [f"KAPPA^2 MODEL -- {folds}-FOLD CROSS-VALIDATION", "=" * 66,
         f"  coupler simulations : {n}",
         f"  kappa^2 range       : {k2.min():.6f} to {k2.max():.6f}",
         "", f"  out-of-sample MAPE  : {mape:.2f} %",
         f"  out-of-sample R^2   : {r2:.4f}",
         f"  within 10 %         : {w10:.1f} %  ({int(round(w10*n/100))}/{n})",
         f"  within  5 %         : {w5:.1f} %",
         f"  bias                : {np.mean(err):+.2f} %",
         f"  worst               : {np.max(np.abs(err)):.2f} %", "",
         "  fitted on all points (these are the shipped coefficients):",
         f"    ln kappa0   {full[0]:+.8f}",
         f"    a1          {full[1]:+.8e}  per nm",
         f"    gamma0      {full[2]:+.8e}  per nm",
         f"    gamma1      {full[3]:+.8e}  per nm^2",
         f"    B           {full[4]:+.6f}   (first-principles value 1)"]
    txt = "\n".join(L)
    json.dump(dict(n_points=int(n), folds=int(folds),
                   kappa2_min=float(k2.min()), kappa2_max=float(k2.max()),
                   cv_mape=mape, cv_r2=r2, within_10pct=w10,
                   within_5pct=w5, bias=float(np.mean(err)),
                   coefficients=[float(v) for v in full]),
              open(os.path.join(resdir, "kappa2_model.json"), "w"), indent=1)
    open(os.path.join(resdir, "kappa2_cv.txt"), "w").write(txt + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "font.size": 8,
                             "axes.spines.top": False,
                             "axes.spines.right": False})
        fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
        lo, hi = k2.min() * 0.7, k2.max() * 1.4
        ax[0].plot([lo, hi], [lo, hi], color="#888888", lw=1.0, zorder=1)
        ax[0].scatter(k2, pred, s=14, color="#0072B2", alpha=0.75,
                      edgecolor="white", linewidth=0.4, zorder=2)
        ax[0].set_xscale("log"); ax[0].set_yscale("log")
        ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
        ax[0].set_xlabel("FDTD $\\kappa^2$")
        ax[0].set_ylabel("model $\\kappa^2$ (held out)")
        ax[0].set_title(f"(a) parity, {folds}-fold CV", loc="left", fontsize=8.5)
        ax[0].grid(alpha=0.25, lw=0.5)
        ax[1].scatter(k2, err, s=14, color="#0072B2", alpha=0.75,
                      edgecolor="white", linewidth=0.4)
        ax[1].axhline(0, color="#888888", lw=1.0)
        for v in (10, -10):
            ax[1].axhline(v, color="#D55E00", lw=0.9, ls=(0, (4, 2)))
        ax[1].set_xscale("log"); ax[1].set_xlabel("FDTD $\\kappa^2$")
        ax[1].set_ylabel("relative error (%)")
        ax[1].set_title(f"(b) residuals, MAPE {mape:.2f}%", loc="left",
                        fontsize=8.5)
        ax[1].grid(alpha=0.25, lw=0.5)
        fig.tight_layout()
        f = os.path.join(resdir, "kappa2_parity.png")
        fig.savefig(f, dpi=300); fig.savefig(f.replace(".png", ".pdf"))
        plt.close(fig)
        L.append(f"\n  parity plot: {f}")
    except ImportError:
        pass
    print("\n" + "\n".join(L))


# ---------------------------------------------------------------------------
# SPECTRUM EXTRACTION  (also used by Phase 1.5 step 6)
# ---------------------------------------------------------------------------

def get_spectrum(fdtd, monitor):
    """Return (wavelength[m] ascending, |T| normalised) for a power monitor."""
    r = fdtd.getresult(monitor, "T")
    lam = np.array(r["lambda"]).flatten().astype(float)
    T = np.abs(np.array(r["T"]).flatten().astype(float))
    order = np.argsort(lam)
    return lam[order], T[order]


def _cross(x0, y0, x1, y1, y):
    """Linear interpolation for the x where the segment crosses y."""
    if y1 == y0:
        return x0
    return x0 + (x1 - x0) * (y - y0) / (y1 - y0)


def analyse_drop(lam, T, target=LAMBDA_TARGET):
    """
    Extract lambda_res, peak T, IL (Eq. 7), FWHM, Q_L (Eq. 5) and FSR (from
    the adjacent peak) from a drop-port spectrum. Returns a dict; values that
    cannot be resolved come back as nan rather than a guess.
    """
    from scipy.signal import find_peaks
    out = dict(lambda_res=np.nan, T_peak=np.nan, IL_dB=np.nan,
               FWHM=np.nan, Q_L=np.nan, FSR=np.nan, n_peaks=0)
    if lam.size < 3:
        return out

    peaks, _ = find_peaks(T, prominence=0.05 * (T.max() - T.min() + 1e-12))
    out["n_peaks"] = int(peaks.size)
    if peaks.size == 0:
        return out

    # `target` is normally 1550 nm (the Section 4.1.5 step 6 definition). For a
    # mesh sweep pass the FIRST point's lambda_res instead, so every point
    # tracks the SAME physical mode -- otherwise a sub-nm mesh shift flips the
    # selection to the adjacent resonance and the gradient is meaningless.
    ip = int(peaks[np.argmin(np.abs(lam[peaks] - target))])

    # Parabolic sub-grid interpolation: without it lambda_res snaps to the
    # DFT grid and cannot resolve mesh-induced shifts smaller than one bin.
    lam_pk, T_pk = float(lam[ip]), float(T[ip])
    if 0 < ip < T.size - 1:
        y0, y1, y2 = float(T[ip - 1]), float(T[ip]), float(T[ip + 1])
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) <= 1.0:
                lam_pk = float(lam[ip] + delta * (lam[ip + 1] - lam[ip]))
                T_pk = float(y1 - 0.25 * (y0 - y2) * delta)
    out["lambda_res"] = lam_pk
    out["T_peak"] = T_pk
    if T_pk > 0:
        out["IL_dB"] = float(-10.0 * np.log10(min(T_pk, 1.0)))

    # FWHM by interpolated half-maximum crossings either side of the peak
    half = T_pk / 2.0
    l = ip
    while l > 0 and T[l] > half:
        l -= 1
    r = ip
    while r < T.size - 1 and T[r] > half:
        r += 1
    if l > 0 and r < T.size - 1:
        lam_l = _cross(lam[l], T[l], lam[l + 1], T[l + 1], half)
        lam_r = _cross(lam[r - 1], T[r - 1], lam[r], T[r], half)
        fw = lam_r - lam_l
        if fw > 0:
            out["FWHM"] = float(fw)
            out["Q_L"] = float(lam_pk / fw)          # Eq. 5

    # FSR from the nearest neighbouring peak
    if peaks.size >= 2:
        others = lam[peaks][lam[peaks] != lam[ip]]
        if others.size:
            out["FSR"] = float(np.min(np.abs(others - lam[ip])))
    return out


# ---------------------------------------------------------------------------
# PHASE 1.6 SUB-STUDY 2 -- MESH CONVERGENCE
# ---------------------------------------------------------------------------

_S2_FIELDS = ["d_nm", "cells_across_gap", "lambda_res_nm", "T_peak", "IL_dB",
              "FWHM_nm", "Q_L", "FSR_nm", "n_peaks", "runtime_s"]


def _s2_load(path):
    if not os.path.isfile(path):
        return {}
    with open(path, newline="") as fh:
        return {float(row["d_nm"]): row for row in csv.DictReader(fh)}


def _s2_append(path, row):
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=_S2_FIELDS)
        if new:
            wr.writeheader()
        wr.writerow(row)


def _s2_report(done):
    """Print the convergence table and the verdict against the 1e-4 criterion."""
    ds = sorted(done.keys(), reverse=True)           # coarse -> fine
    print("\n" + "=" * 86)
    print("PHASE 1.6 SUB-STUDY 2 -- MESH CONVERGENCE")
    print("=" * 86)
    print(f"{'d(nm)':>7} {'cells':>6} {'lam_res(nm)':>12} {'T_peak':>9} "
          f"{'Q_L':>9} {'FSR(nm)':>8} {'|dT/dN|':>10} {'runtime':>9}")
    print("-" * 86)
    prev = None
    verdict_d = None
    for d in ds:
        r = done[d]
        T = float(r["T_peak"]); N = float(r["cells_across_gap"])
        grad = ""
        if prev is not None:
            dN = abs(N - prev[1])
            g = abs(T - prev[0]) / dN if dN else float("nan")
            grad = f"{g:.3e}"
            if g < CONVERGENCE_GRADIENT and verdict_d is None:
                verdict_d = d
        print(f"{d:7.1f} {N:6.1f} {float(r['lambda_res_nm']):12.4f} "
              f"{T:9.5f} {float(r['Q_L']):9.1f} {float(r['FSR_nm']):8.4f} "
              f"{grad:>10} {float(r['runtime_s']):8.1f}s")
        prev = (T, N)
    print("-" * 86)
    print(f"Criterion: |dT_peak/d(cells across gap)| < {CONVERGENCE_GRADIENT:g}")
    if verdict_d is None:
        print("VERDICT: not yet converged across the swept range. Extend "
              "MESH_SWEEP_NM to finer d before fixing the mesh.")
    else:
        print(f"VERDICT: converged at d = {verdict_d:g} nm. Use "
              f"MeshConfig.uniform({verdict_d:g}e-9) for sub-study 1, "
              f"sub-studies 3-4, and production.")
    print("=" * 86)


def run_calibration(dimension="2D", d_nm=32.0):
    """
    Phase 1.6 sub-study 2 prerequisite: measure the TRUE linewidth.

    Sub-study 2 was run with a 50 ps window, which is transform-limited to
    ~0.141 nm -- wider than the line itself, so FWHM and Q_L measured the DFT
    window rather than the resonator. This runs ONE simulation at the coarsest
    (fastest) mesh with a long window and reports what resolution the device
    actually needs, which is the Section 4.1.4 item 3 verification.
    """
    p = MRRParams(R=10.0e-6, w=450e-9, g=200e-9, Lc=1.5e-6)
    mesh = MeshConfig.uniform(d_nm * 1e-9)
    dom = DomainConfig(movie=False, dimension=dimension,
                       sim_time=CALIBRATE_SIM_TIME)

    print(f"[calibrate] {dimension}, d = {d_nm:g} nm, window "
          f"{CALIBRATE_SIM_TIME*1e12:.0f} ps, {N_FREQ_POINTS} frequency points")
    print("[calibrate] auto-shutoff will usually end this early")

    lumapi = load_lumapi()
    t0 = time.time()
    with lumapi.FDTD(hide=True) as fdtd:
        build_mrr(fdtd, p, mesh, dom)
        fdtd.save(os.path.abspath(f"calibrate_{dimension.lower()}_d{d_nm:g}nm.fsp"))
        fdtd.run()
        lam, T = get_spectrum(fdtd, "monitor_drop")
        res = analyse_drop(lam, T)
    dt = time.time() - t0

    grid = float(np.min(np.diff(lam))) if lam.size > 1 else float("nan")
    fwhm = res["FWHM"]
    # rectangular-window transform limit, 0.886 * lambda^2 / (c * T)
    tl = 0.886 * res["lambda_res"] ** 2 / (2.99792458e8 * CALIBRATE_SIM_TIME)

    print("\n" + "=" * 70)
    print("CALIBRATION RESULT")
    print("=" * 70)
    print(f"  lambda_res            : {res['lambda_res']*1e9:.5f} nm")
    print(f"  T_peak                : {res['T_peak']:.5f}")
    print(f"  IL (Eq. 7)            : {res['IL_dB']:.4f} dB")
    print(f"  FWHM                  : {fwhm*1e9:.5f} nm")
    print(f"  Q_L (Eq. 5)           : {res['Q_L']:.1f}")
    print(f"  FSR                   : {res['FSR']*1e9:.4f} nm   "
          f"({res['n_peaks']} peaks)")
    print("-" * 70)
    print(f"  DFT grid spacing      : {grid*1e9:.5f} nm")
    print(f"  transform limit @ {CALIBRATE_SIM_TIME*1e12:.0f} ps : "
          f"{tl*1e9:.5f} nm")
    if not np.isnan(fwhm):
        print(f"  points across FWHM    : {fwhm/grid:.1f}   "
              + ("OK (>=5)" if fwhm / grid >= 5 else "*** TOO FEW ***"))
        print(f"  FWHM / transform limit: {fwhm/tl:.2f}   "
              + ("OK (>=3, line is resolved)" if fwhm / tl >= 3
                 else "*** STILL WINDOW-LIMITED, lengthen the window ***"))
        need_T = 0.886 * res["lambda_res"] ** 2 / (2.99792458e8 * fwhm / 3.0)
        need_N = int((LAMBDA_MAX - LAMBDA_MIN) / (fwhm / 5.0)) + 1
        print(f"  window needed (3x)    : {need_T*1e12:.0f} ps")
        print(f"  freq points needed    : {need_N}")
    print(f"  runtime               : {dt/60:.1f} min")
    print("=" * 70)


def run_substudy2(dimension="3D"):
    """Sweep the scalar mesh override and test the transmittance gradient."""
    p = MRRParams(R=10.0e-6, w=450e-9, g=200e-9, Lc=1.5e-6)
    # movie OFF: it would distort runtime
    dom = DomainConfig(movie=False, dimension=dimension)

    csv_path = os.path.abspath(SUBSTUDY2_CSV.format(dim=dimension.lower()))
    done = _s2_load(csv_path)
    todo = [d for d in MESH_SWEEP_NM if d not in done]
    print(f"[substudy2] {len(done)} point(s) already done, {len(todo)} to run")
    if done:
        print(f"[substudy2] resuming; delete {os.path.basename(csv_path)} "
              f"to start over")

    track = None
    if done:                                  # resuming: reuse the locked mode
        first = done[max(done.keys())]
        track = float(first["lambda_res_nm"]) * 1e-9

    lumapi = load_lumapi()
    for d_nm in todo:
        mesh = MeshConfig.uniform(d_nm * 1e-9)
        print(f"\n[substudy2] d = {d_nm:g} nm "
              f"({p.g/(d_nm*1e-9):.1f} cells across the gap) ...")
        t0 = time.time()
        with lumapi.FDTD(hide=True) as fdtd:
            build_mrr(fdtd, p, mesh, dom)
            fdtd.save(os.path.abspath(
                f"substudy2_{dimension.lower()}_d{d_nm:g}nm.fsp"))
            fdtd.run()
            lam, T = get_spectrum(fdtd, "monitor_drop")
            res = analyse_drop(lam, T, target=track if track else LAMBDA_TARGET)
        if track is None and not np.isnan(res["lambda_res"]):
            track = res["lambda_res"]        # lock every later point to this mode
            print(f"[substudy2] tracking the resonance at "
                  f"{track*1e9:.4f} nm for all remaining points")
        dt = time.time() - t0
        row = dict(d_nm=d_nm,
                   cells_across_gap=round(p.g / (d_nm * 1e-9), 3),
                   lambda_res_nm=res["lambda_res"] * 1e9,
                   T_peak=res["T_peak"],
                   IL_dB=res["IL_dB"],
                   FWHM_nm=res["FWHM"] * 1e9,
                   Q_L=res["Q_L"],
                   FSR_nm=res["FSR"] * 1e9,
                   n_peaks=res["n_peaks"],
                   runtime_s=round(dt, 1))
        _s2_append(csv_path, row)          # written per point, so a crash
        done[d_nm] = {k: str(v) for k, v in row.items()}   # loses nothing
        print(f"[substudy2] lam_res {row['lambda_res_nm']:.4f} nm, "
              f"T_peak {row['T_peak']:.5f}, Q_L {row['Q_L']:.1f}, "
              f"{dt:.1f}s")

    _s2_report(done)
    print(f"\nResults: {csv_path}")


# ---------------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------------

def main():
    if "--density" in sys.argv:
        run_density()
        return

    if "--physinv" in sys.argv:
        run_physinv()
        return

    if "--ablate" in sys.argv:
        run_ablate()
        return

    if "--phase7sim" in sys.argv:
        run_phase7sim()
        return

    if "--phase7" in sys.argv:
        run_phase7()
        return

    if "--closedloop" in sys.argv:
        mth = "naive"
        for a in ("naive", "tandem", "rf_norm", "rf_raw", "xgb_norm",
                  "xgb_raw"):
            if f"--{a}" in sys.argv:
                mth = a
        nn = CLOSEDLOOP_N
        for a in sys.argv:
            if a.startswith("--n="):
                nn = int(a[4:])
        run_closedloop(mth, nn)
        return

    if "--phase6" in sys.argv:
        run_phase6()
        return

    if "--phase5" in sys.argv:
        run_phase5()
        return

    if "--phase4" in sys.argv:
        run_phase4()
        return

    if "--phase3" in sys.argv:
        run_phase3()
        return

    if "--phase2" in sys.argv:
        run_phase2()
        return

    if "--generate" in sys.argv:
        run_generate()
        return

    if "--modesweep" in sys.argv:
        run_modesweep()
        return

    if "--floorconv" in sys.argv:
        run_floorconv()
        return

    if "--seedcurve" in sys.argv:
        run_seedcurve()
        return

    if "--labelsens" in sys.argv:
        run_labelsens()
        return

    if "--trimbudget" in sys.argv:
        run_trimbudget()
        return

    if "--thermal" in sys.argv:
        run_thermal()
        return

    if "--matcheck" in sys.argv:
        run_matcheck()
        return

    if "--fsrcheck" in sys.argv:
        run_fsrcheck()
        return

    if "--modediag" in sys.argv:
        run_modediag()
        return

    if "--bentmode" in sys.argv:
        run_bentmode()
        return

    if "--ringvalidate" in sys.argv:
        nn = None
        for a_ in sys.argv:
            if a_.startswith("--n="):
                nn = int(a_[4:])
        mg = RINGVAL_MARGINS
        extra = []
        for a_ in sys.argv:
            if a_.startswith("--margins="):
                mg = tuple(float(x) for x in a_[10:].split(","))
            if a_.startswith("--try="):
                extra.append(tuple(float(x) for x in a_[6:].split(",")))
        if "--dry" in sys.argv:
            run_ringdry(margins=mg, extra=extra)
            return
        mnm = BEND_MESH_NM
        pts = None
        for a_ in sys.argv:
            if a_.startswith("--mesh="):
                mnm = float(a_[7:])
            if a_.startswith("--points="):
                pts = int(a_[9:])
        run_ringvalidate(nn, movie=("--no-movie" not in sys.argv), margins=mg,
                         extra=extra, mesh_nm=mnm, points=pts)
        return

    if "--validate" in sys.argv:
        run_validate()
        return

    if "--floor" in sys.argv:
        nb = FLOOR_N
        for a_ in sys.argv:
            if a_.startswith("--nbank="):
                nb = int(a_[8:])
        run_floor(nb)
        return

    if "--thetaall" in sys.argv:
        run_thetaall()
        return

    if "--seedcheck" in sys.argv:
        run_seedcheck()
        return

    if "--kappafit" in sys.argv:
        run_kappafit()
        return

    if "--couplerext" in sys.argv:
        nn = COUPLEREXT_N
        for a in sys.argv:
            if a.startswith("--n="):
                nn = int(a[4:])
        run_couplerext(nn)
        return

    if "--couplerset" in sys.argv:
        run_couplerset("2D" if "--2d" in sys.argv else "3D")
        return

    if "--bendloss" in sys.argv:
        run_bendloss("2D" if "--2d" in sys.argv else "3D")
        return

    if "--substudy2c" in sys.argv:
        run_substudy2_coupler("3D" if "--3d" in sys.argv else "2D")
        return

    if "--calibrate" in sys.argv:
        run_calibration("2D" if "--2d" in sys.argv else "3D")
        return

    if "--substudy2" in sys.argv:
        run_substudy2("2D" if "--2d" in sys.argv else "3D")
        return

    movie = "--no-movie" not in sys.argv
    quick = "--quick" in sys.argv

    # Mid-range representative geometry. This same geometry is reused for
    # Phase 1.6 sub-studies 2 and 1.
    p = MRRParams(R=10.0e-6, w=450e-9, g=200e-9, Lc=1.5e-6)
    mesh = MeshConfig()
    dom = DomainConfig(movie=movie,
                       sim_time=MOVIE_SIM_TIME if quick else DomainConfig.sim_time)

    lumapi = load_lumapi()
    with lumapi.FDTD(hide=False) as fdtd:
        lay = build_mrr(fdtd, p, mesh, dom)
        print(inspect_build(p, lay, mesh, dom))
        prefix = "QUICK_" if quick else "template_"
        out = os.path.abspath(f"{prefix}{p.tag()}.fsp")
        fdtd.save(out)
        print(f"\nSaved: {out}")
        input("\nInspect the layout in the GUI, then press Enter to close...")


if __name__ == "__main__":
    main()
