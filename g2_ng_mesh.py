"""
G2 -- is the ~3 % group-index deficit of the full-ring FDTD a mesh artefact?

A STRAIGHT 450 nm x 220 nm waveguide is simulated in 3-D FDTD at several mesh
sizes. n_eff(lambda) comes from the phase advance of the field between two
point monitors a distance D apart; n_g = n - lambda dn/dlambda. The result is
compared with the mode solver (FDE) at the same geometry. If the FDTD n_g
moves toward the FDE value as the mesh is refined, the ring-FDTD deficit is
numerical dispersion; if it stays ~3 % below, it is not.

Run from the project folder (the one with mrr_template.py):
    py -3 g2_ng_mesh.py --quick        (one 40 nm run: tests that everything works)
    py -3 g2_ng_mesh.py                (40, 32, 25, 20 nm: the real study)
Writes results/g2_ng_mesh.csv and results/g2_ng_mesh_report.txt
Nothing is sent to the paper until the numbers have been looked at.
"""
import csv, os, sys, time
import numpy as np

C = 299792458.0
W_NM = 450.0
LAM0 = 1550.0
FIT_LO, FIT_HI = 1520.0, 1580.0       # fit window inside the 1500-1600 source band
LW = 14e-6                            # waveguide length
DX = 6e-6                             # monitor separation
SIM_TIME = 1.0e-12


def ng_from_phase(freq, e1, e2, dist, n_prior, lam_fit=(FIT_LO, FIT_HI), order=3):
    """n_eff and n_g at LAM0 from the complex fields at two points `dist` apart.
    Lumerical fields go as exp(i(beta x - omega t)), so E2/E1 = exp(i beta dist).
    The phase is only known up to a constant 2*pi*k; k is the integer that puts
    n_eff nearest n_prior (the mode-solver value). Neighbouring k differ by
    lambda/dist = 0.26 in n_eff, far more than any FDTD-vs-FDE difference."""
    freq = np.asarray(freq, float)
    lam = C / freq * 1e9                               # nm
    ph = np.unwrap(np.angle(np.asarray(e2) / np.asarray(e1)))
    k_all = np.arange(-200, 201)
    mid = int(np.argmin(np.abs(lam - LAM0)))
    n_mid = (ph[mid] + 2 * np.pi * k_all) * C / (2 * np.pi * freq[mid] * dist)
    k = k_all[int(np.argmin(np.abs(n_mid - n_prior)))]
    neff = (ph + 2 * np.pi * k) * C / (2 * np.pi * freq * dist)
    sel = (lam >= lam_fit[0]) & (lam <= lam_fit[1])
    p = np.polyfit(lam[sel] - LAM0, neff[sel], order)
    n0 = np.polyval(p, 0.0)
    dn = np.polyval(np.polyder(p), 0.0)
    resid = float(np.max(np.abs(np.polyval(p, lam[sel] - LAM0) - neff[sel])))
    return float(n0), float(n0 - LAM0 * dn), resid


def selftest():
    """Check the extraction on a synthetic field with a known n_eff(lambda)."""
    lam = np.linspace(1500, 1600, 101)
    n = 2.353 - 0.00128 * (lam - 1550) - 4e-6 * (lam - 1550) ** 2
    ngt = n[50] - 1550 * (-0.00128)
    f = C / (lam * 1e-9)
    e1 = np.exp(1j * 0.3) * np.ones_like(f)
    e2 = e1 * np.exp(1j * 2 * np.pi * f / C * n * DX)
    n0, ng, res = ng_from_phase(f, e1, e2, DX, n_prior=2.34)
    assert abs(n0 - n[50]) < 1e-4 and abs(ng - ngt) < 2e-3, (n0, n[50], ng, ngt)
    return n0, ng


def fde_reference(m, fdtd):
    """FDE n_eff, n_g at 1550 nm, w = 450 nm, 11 wavelengths, same code path as the paper's mode sweep."""
    lams = np.arange(1500.0, 1600.1, 10.0)
    ne = np.array([m._neff_at(fdtd, W_NM * 1e-9, l * 1e-9) for l in lams])
    p = np.polyfit(lams - LAM0, ne, 3)
    n0 = np.polyval(p, 0.0)
    dn = np.polyval(np.polyder(p), 0.0)
    return float(n0), float(n0 - LAM0 * dn)


def build_straight(fdtd, m, d):
    h = m.H_CORE
    fdtd.switchtolayout(); fdtd.deleteall()
    fdtd.addrect(); fdtd.set("name", "cladding")
    fdtd.set("x", 0.0); fdtd.set("x span", LW + 2e-6)
    fdtd.set("y", 0.0); fdtd.set("y span", 6e-6)
    fdtd.set("z", 0.0); fdtd.set("z span", 6e-6)
    fdtd.set("material", m.MAT_CLAD)
    fdtd.set("override mesh order from material database", True); fdtd.set("mesh order", 3)
    fdtd.addrect(); fdtd.set("name", "wg")
    fdtd.set("x", 0.0); fdtd.set("x span", LW + 2e-6)
    fdtd.set("y", 0.0); fdtd.set("y span", W_NM * 1e-9)
    fdtd.set("z", 0.0); fdtd.set("z span", h)
    fdtd.set("material", m.MAT_CORE)
    fdtd.set("override mesh order from material database", True); fdtd.set("mesh order", 2)
    fdtd.addfdtd(); fdtd.set("dimension", "3D")
    fdtd.set("x", 0.0); fdtd.set("x span", LW)
    fdtd.set("y", 0.0); fdtd.set("y span", 3.0e-6)
    fdtd.set("z", 0.0); fdtd.set("z span", 2.4e-6)
    fdtd.set("mesh accuracy", 3)
    fdtd.set("simulation time", SIM_TIME)
    for bc in ("x min bc", "x max bc", "y min bc", "y max bc", "z min bc", "z max bc"):
        fdtd.set(bc, "PML")
    fdtd.addmesh(); fdtd.set("name", "mesh_wg")
    fdtd.set("x", 0.0); fdtd.set("x span", LW)
    fdtd.set("y", 0.0); fdtd.set("y span", 1.8e-6)
    fdtd.set("z", 0.0); fdtd.set("z span", 1.2e-6)
    for ax in ("x", "y", "z"):
        fdtd.set(f"override {ax} mesh", True); fdtd.set(f"d{ax}", d)
    fdtd.setglobalsource("set wavelength", True)
    fdtd.setglobalsource("wavelength start", m.LAMBDA_MIN)
    fdtd.setglobalsource("wavelength stop", m.LAMBDA_MAX)
    fdtd.setglobalmonitor("use source limits", True)
    fdtd.setglobalmonitor("use wavelength spacing", True)
    fdtd.setglobalmonitor("frequency points", 201)
    xs = -LW / 2 + 2.0e-6
    fdtd.addmode(); fdtd.set("name", "src")
    fdtd.set("injection axis", "x-axis"); fdtd.set("direction", "Forward")
    fdtd.set("x", xs); fdtd.set("y", 0.0); fdtd.set("y span", 2.5e-6)
    fdtd.set("z", 0.0); fdtd.set("z span", 2.2e-6)
    fdtd.set("mode selection", "fundamental TE mode")
    fdtd.set("number of trial modes", m.N_TRIAL_MODES)
    fdtd.set("override global source settings", False)
    for i, x in enumerate((xs + 2.0e-6, xs + 2.0e-6 + DX), 1):
        fdtd.addpower(); fdtd.set("name", f"pt{i}")
        fdtd.set("monitor type", "point")
        fdtd.set("x", x); fdtd.set("y", 0.0); fdtd.set("z", 0.0)


def field(fdtd, name):
    v = np.squeeze(np.array(fdtd.getdata(name, "Ey"), dtype=complex))
    f = np.squeeze(np.array(fdtd.getdata(name, "f"), dtype=float))
    return f, v


def main():
    if "--selftest" in sys.argv:
        print("selftest OK:", selftest()); return
    selftest()
    sys.path.insert(0, os.getcwd())
    import mrr_template as m
    meshes = [40.0] if "--quick" in sys.argv else [40.0, 32.0, 25.0, 20.0]
    lumapi = m.load_lumapi()
    out = []
    with lumapi.FDTD(hide=True) as fdtd:
        ne_fde, ng_fde = fde_reference(m, fdtd)
        print(f"[g2] FDE reference: n_eff {ne_fde:.5f}  n_g {ng_fde:.5f}")
        for d_nm in meshes:
            t0 = time.time()
            build_straight(fdtd, m, d_nm * 1e-9)
            fdtd.save(os.path.abspath("g2_scratch.fsp"))
            fdtd.run()
            f1, e1 = field(fdtd, "pt1"); f2, e2 = field(fdtd, "pt2")
            order = np.argsort(f1)
            n0, ng, res = ng_from_phase(f1[order], e1[order], e2[order], DX, n_prior=ne_fde)
            print(f"[g2] mesh {d_nm:4.0f} nm: n_eff {n0:.5f}  n_g {ng:.5f}  "
                  f"(n_g vs FDE {100*(ng/ng_fde-1):+.2f} %, fit residual {res:.1e}, {(time.time()-t0)/60:.1f} min)")
            out.append((d_nm, n0, ng, res))
    os.makedirs("results", exist_ok=True)
    with open("results/g2_ng_mesh.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["mesh_nm", "neff_fdtd", "ng_fdtd", "fit_residual", "neff_fde", "ng_fde"])
        for r in out: w.writerow([*r, ne_fde, ng_fde])
    L = ["G2: straight 450 x 220 nm waveguide, FDTD vs mode solver", "=" * 66,
         f"FDE reference (same solver path as the paper): n_eff {ne_fde:.5f}, n_g {ng_fde:.5f}",
         f"{'mesh nm':>8s} {'n_eff':>9s} {'n_g':>9s} {'n_g vs FDE %':>13s}"]
    for d, n0, ng, res in out:
        L.append(f"{d:8.0f} {n0:9.5f} {ng:9.5f} {100*(ng/ng_fde-1):+13.2f}")
    if len(out) >= 3:
        d = np.array([o[0] for o in out]); g = np.array([o[2] for o in out])
        A = np.vstack([np.ones_like(d), d ** 2]).T
        a, b = np.linalg.lstsq(A, g, rcond=None)[0]
        L.append(f"extrapolation to zero mesh (n_g = a + b*d^2, least squares): {a:.5f}  ({100*(a/ng_fde-1):+.2f} % vs FDE)")
        L.append("assumes second-order convergence, which four points can only suggest")
    txt = "\n".join(L); print("\n" + txt)
    open("results/g2_ng_mesh_report.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
