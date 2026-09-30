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


def run_generate(n=DATASET_N):
    from scipy.optimize import brentq
    from scipy.stats import qmc
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
