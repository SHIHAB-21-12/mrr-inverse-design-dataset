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

# --- mode sweep: n_eff(lambda, w) and n_g(lambda, w) -----------------------
# FDTD's source eigensolver reports n_eff but NOT group index (the GUI shows
# N/A), so n_g is derived from dispersion: n_g = n_eff - lambda*dn_eff/dlambda.
# 440 nm is included because Paper 11's fabricated ring is 440 x 220 nm --
# that point drives the sub-study 3 literature validation.
MODE_W_NM = [400.0, 425.0, 440.0, 450.0, 475.0, 500.0]
MODE_LAM_NM = [1500.0, 1525.0, 1550.0, 1575.0, 1600.0]
MODE_MESH_NM = 10.0                 # cross-section only, so a fine mesh is cheap
MODESWEEP_CSV = "modesweep.csv"
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
    if ok:
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
        L += ["", f"  RMS closed-loop response error = {overall:.3f} %  "
              f"(over the {len(ok)} realisable predictions)",
              f"  realisable fraction = {100*frac:.1f} %",
              f"  O3 requires RMS closed-loop error <= 20%: "
              + ("MET" if overall <= 20.0 and not un else
                 ("MET on the realisable subset, but "
                  f"{len(un)} predictions could not be built" if
                  overall <= 20.0 else "NOT MET"))]
    txt = "\n".join(L)
    open(os.path.join(resdir, f"closedloop_{method}.txt"), "w").write(
        txt + "\n")
    print("\n" + txt)


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
