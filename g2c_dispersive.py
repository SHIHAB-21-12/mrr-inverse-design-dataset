"""
G2c -- CONFIRM that restoring silicon's dispersion in FDTD brings the straight-guide
n_g back to the mode-solver value (G2b found FDTD's material fit has no dispersion).

Step 1: try to get FDTD to use a dispersive fit of the Palik data (three methods, tried
        in order; each is CHECKED by asking FDTD for its fitted index and comparing the
        fitted group index with the raw Palik one).
Step 2: re-run the straight 450 x 220 nm guide with the working material at 40, 25, 20 nm
        and compare n_g with the (dispersive) mode solver.

Run from the project folder:   py -3 g2c_dispersive.py
Needs g2_ng_mesh.py and g2b_fixed_index.py beside it.  Writes results/g2c_report.txt
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.getcwd())
import g2_ng_mesh as g
import g2b_fixed_index as b
import mrr_template as m

TOL_NG = 0.01          # a fit is accepted if its group index is within 1 % of the raw data's


def fit_ng(fdtd, name):
    a, bfit, ng_raw, ng_fit, mx = b.material_report(fdtd, name)
    return ng_raw, ng_fit


def try_make_dispersive(fdtd, src_name, tag, log):
    """Return the name of a material whose FDTD fit reproduces the raw group index, or None."""
    f = g.C / (b.LAMS * 1e-9)
    raw = np.real(np.array(fdtd.getindex(src_name, f)).flatten())
    ng_raw, _ = fit_ng(fdtd, src_name)

    def ok(name):
        try:
            _, ng_fit = fit_ng(fdtd, name)
            log.append(f"      {name}: fitted n_g {ng_fit:.5f} (raw {ng_raw:.5f})")
            return abs(ng_fit / ng_raw - 1) < TOL_NG
        except Exception as exc:                        # noqa: BLE001
            log.append(f"      {name}: check failed: {exc}")
            return False

    # method A: tighten the fit tolerance of a copy of the database material
    try:
        new = f"{tag}_A"
        fdtd.copymaterial(src_name, new)
        for prop, val in (("tolerance", 1e-5), ("max coefficients", 8)):
            try:
                fdtd.setmaterial(new, prop, val)
            except Exception as exc:                    # noqa: BLE001
                log.append(f"      A: could not set '{prop}': {exc}")
        if ok(new):
            return new
    except Exception as exc:                            # noqa: BLE001
        log.append(f"      method A failed: {exc}")

    # method B / C: sampled data built from the raw index, two possible conventions
    for variant, data in (("B", np.column_stack([f, raw])),
                          ("C", np.column_stack([f, raw ** 2]))):
        try:
            new = f"{tag}_{variant}"
            made = fdtd.addmaterial("Sampled 3D data")
            made = made if isinstance(made, str) else "Sampled 3D data"
            fdtd.setmaterial(made, "name", new)
            fdtd.setmaterial(new, "sampled data", data)
            for prop, val in (("tolerance", 1e-5), ("max coefficients", 8)):
                try:
                    fdtd.setmaterial(new, prop, val)
                except Exception:                       # noqa: BLE001
                    pass
            if ok(new):
                return new
        except Exception as exc:                        # noqa: BLE001
            log.append(f"      method {variant} failed: {exc}")
    return None


def main():
    lumapi = m.load_lumapi()
    L = ["G2c: straight guide with a dispersive silicon fit in FDTD", "=" * 66]
    with lumapi.FDTD(hide=True) as fdtd:
        ne_fde, ng_fde = g.fde_reference(m, fdtd)
        L.append(f"dispersive mode solver: n_eff {ne_fde:.5f}  n_g {ng_fde:.5f}")
        print(L[-1])
        L.append("making dispersive material fits:")
        si = try_make_dispersive(fdtd, m.MAT_CORE, "Si_disp", L)
        ox = try_make_dispersive(fdtd, m.MAT_CLAD, "SiO2_disp", L)
        print("\n".join(L))
        if si is None:
            L.append("NO working dispersive silicon material could be made; see lines above.")
            txt = "\n".join(L); print(txt)
            os.makedirs("results", exist_ok=True)
            open("results/g2c_report.txt", "w").write(txt + "\n")
            return
        L.append(f"using silicon = {si}, silica = {ox if ox else 'original (flat)'}")
        for d_nm in (40.0, 25.0, 20.0):
            t0 = time.time()
            g.build_straight(fdtd, m, d_nm * 1e-9)
            fdtd.select("wg"); fdtd.set("material", si)
            if ox:
                fdtd.select("cladding"); fdtd.set("material", ox)
            fdtd.save(os.path.abspath("g2c_scratch.fsp"))
            fdtd.run()
            f1, e1 = g.field(fdtd, "pt1"); f2, e2 = g.field(fdtd, "pt2")
            o = np.argsort(f1)
            n0, ng, res = g.ng_from_phase(f1[o], e1[o], e2[o], g.DX, n_prior=ne_fde)
            line = (f"FDTD {d_nm:4.0f} nm: n_eff {n0:.5f}  n_g {ng:.5f}  "
                    f"(n_g vs dispersive FDE {100*(ng/ng_fde-1):+.2f} %, {(time.time()-t0)/60:.1f} min)")
            L.append(line); print(line)
    L += ["", "Earlier, with the flat default fit: -2.5, -3.4, -3.6 % at 40, 25, 20 nm.",
          "Expected if the explanation is right: the gap shrinks to well under 1 %."]
    txt = "\n".join(L); print("\n" + txt)
    os.makedirs("results", exist_ok=True)
    open("results/g2c_report.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
