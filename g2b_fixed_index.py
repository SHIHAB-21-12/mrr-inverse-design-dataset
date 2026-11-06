"""
G2b -- WHY does straight-waveguide FDTD give n_g about 3 % below the mode solver?

Two checks, both cheap:
 (1) MATERIAL: how does the index that FDTD actually uses (its fit of the
     Palik data over 1500-1600 nm) differ from the raw Palik index? A fit that
     is right at 1550 nm but has the wrong slope changes n_g without changing
     n_eff much.
 (2) NUMERICS: replace both materials by fixed (non-dispersive) indices in BOTH
     the FDTD and the mode solver. If they now agree, the earlier gap is the
     material model; if they still differ, it is FDTD numerics or the way n_g
     is extracted.

Run from the project folder:   py -3 g2b_fixed_index.py
Needs g2_ng_mesh.py in the same folder. Writes results/g2b_report.txt
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.getcwd())
import g2_ng_mesh as g
import mrr_template as m

N_SI, N_OX = 3.47656, 1.44402        # indices at 1550 nm, as both solvers report them
LAMS = np.arange(1500.0, 1600.1, 10.0)


def material_report(fdtd, name):
    """Raw Palik index vs the index FDTD fits to it, over 1500-1600 nm."""
    f = g.C / (LAMS * 1e-9)
    raw = np.real(np.array(fdtd.getindex(name, f)).flatten())
    fmin, fmax = g.C / 1600e-9, g.C / 1500e-9
    fit = np.real(np.array(fdtd.getfdtdindex(name, f, fmin, fmax)).flatten())
    pr = np.polyfit(LAMS - 1550, raw, 3); pf = np.polyfit(LAMS - 1550, fit, 3)
    ng_raw = np.polyval(pr, 0) - 1550 * np.polyval(np.polyder(pr), 0)
    ng_fit = np.polyval(pf, 0) - 1550 * np.polyval(np.polyder(pf), 0)
    return (np.polyval(pr, 0), np.polyval(pf, 0), ng_raw, ng_fit,
            float(np.max(np.abs(fit - raw))))


def main():
    lumapi = m.load_lumapi()
    L = ["G2b: material fit and fixed-index comparison", "=" * 66]

    with lumapi.FDTD(hide=True) as fdtd:
        L.append("(1) index of each material: raw Palik vs the FDTD fit (1500-1600 nm)")
        L.append(f"    {'material':24s} {'n raw':>8s} {'n fit':>8s} {'ng raw':>8s} {'ng fit':>8s} {'max|fit-raw|':>13s}")
        for nm in (m.MAT_CORE, m.MAT_CLAD):
            try:
                a, b, c, d, e = material_report(fdtd, nm)
                L.append(f"    {nm:24s} {a:8.5f} {b:8.5f} {c:8.5f} {d:8.5f} {e:13.2e}")
            except Exception as exc:                    # noqa: BLE001
                L.append(f"    {nm:24s} could not be evaluated: {exc}")
        print("\n".join(L))

    # (2a) mode solver with fixed indices
    print("[g2b] FDE with fixed indices ...")
    with lumapi.MODE(hide=True) as md:
        ne = np.array([m._fde_fixed_index(md, g.W_NM, l, N_SI, N_OX) for l in LAMS])
    p = np.polyfit(LAMS - 1550, ne, 3)
    ne_fde = float(np.polyval(p, 0)); ng_fde = float(ne_fde - 1550 * np.polyval(np.polyder(p), 0))
    L += ["", "(2) both materials replaced by fixed indices (n_Si 3.47656, n_SiO2 1.44402)",
          f"    mode solver:  n_eff {ne_fde:.5f}   n_g {ng_fde:.5f}"]
    print(L[-1])

    # (2b) FDTD with fixed indices at three meshes
    rows = []
    with lumapi.FDTD(hide=True) as fdtd:
        for d_nm in (40.0, 25.0, 20.0):
            t0 = time.time()
            g.build_straight(fdtd, m, d_nm * 1e-9)
            for obj, idx in (("cladding", N_OX), ("wg", N_SI)):
                fdtd.select(obj)
                fdtd.set("material", "<Object defined dielectric>")
                fdtd.set("index", idx)
            fdtd.save(os.path.abspath("g2b_scratch.fsp"))
            fdtd.run()
            f1, e1 = g.field(fdtd, "pt1"); f2, e2 = g.field(fdtd, "pt2")
            o = np.argsort(f1)
            n0, ng, res = g.ng_from_phase(f1[o], e1[o], e2[o], g.DX, n_prior=ne_fde)
            rows.append((d_nm, n0, ng))
            line = (f"    FDTD {d_nm:4.0f} nm: n_eff {n0:.5f}  n_g {ng:.5f}  "
                    f"(n_g vs FDE {100*(ng/ng_fde-1):+.2f} %, {(time.time()-t0)/60:.1f} min)")
            L.append(line); print(line)
    L += ["", "Reading: if the fixed-index FDTD and FDE agree to <0.5 %, the earlier gap is the",
          "material model; if the gap persists, it is FDTD numerics or the n_g extraction."]
    txt = "\n".join(L); print("\n" + txt)
    os.makedirs("results", exist_ok=True)
    open("results/g2b_report.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
