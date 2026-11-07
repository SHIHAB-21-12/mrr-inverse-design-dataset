"""
H3 -- does the flat silicon index used in the coupler FDTD runs bias kappa^2?

Takes N_PICK of the 96 coupler runs already in results/closedloop_naive.json (chosen by
kappa^2 quantiles, so weak, medium and strong coupling are all covered), re-runs each with
the dispersive silicon / silica materials that G2c validated, and compares:
    kappa^2 flat (existing run)  |  kappa^2 dispersive (new run)  |  kappa^2 from the model
Resumable: each finished run is written to results/h3_dispersive_coupler.json.

Run from the project folder:   py -3 h3_dispersive_coupler.py
Needs g2_ng_mesh.py, g2b_fixed_index.py, g2c_dispersive.py beside it.
Writes results/h3_dispersive_coupler.json and results/h3_report.txt
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.getcwd())
import mrr_template as m
import g2c_dispersive as c

N_PICK = 6
CK = os.path.join("results", "h3_dispersive_coupler.json")
SRC = os.path.join("results", "closedloop_naive.json")
CORE_OBJS = ["ring_straight_bottom", "ring_bend_left", "ring_bend_right",
             "ring_exit_left", "ring_exit_right", "bus_input_through"]


def pick():
    d = json.load(open(SRC))
    ok = sorted([(v["kappa2"], k) for k, v in d.items()
                 if v.get("status") == "ok" and "kappa2" in v])
    idx = sorted(set(int(round(x)) for x in np.linspace(0, len(ok) - 1, N_PICK)))
    return [(ok[i][1], d[ok[i][1]]) for i in idx]


def run_one(lumapi, R, w, g, Lc):
    mesh = m.MeshConfig.uniform(m.BEND_MESH_NM * 1e-9)
    dom = m.DomainConfig(movie=False, dimension="3D", sim_time=m.COUPLER_SIM_TIME,
                         freq_points=m.COUPLER_FREQ_POINTS)
    p = m.MRRParams(R * 1e-6, w * 1e-9, g * 1e-9, Lc * 1e-6)
    log = []
    with lumapi.FDTD(hide=True) as fdtd:
        m.build_coupler(fdtd, p, mesh, dom)
        si = c.try_make_dispersive(fdtd, m.MAT_CORE, "Si_disp", log)
        ox = c.try_make_dispersive(fdtd, m.MAT_CLAD, "SiO2_disp", log)
        if si is None or ox is None:
            raise RuntimeError("dispersive material not made: " + " | ".join(log))
        for nm in CORE_OBJS:
            if nm == "ring_straight_bottom" and Lc <= 0:
                continue
            fdtd.select(nm); fdtd.set("material", si)
        fdtd.select("cladding"); fdtd.set("material", ox)
        fdtd.save(os.path.abspath("h3_scratch.fsp"))
        fdtd.run()
        lam_c, Tc = m.get_spectrum(fdtd, "monitor_cross")
        lam_t, Tt = m.get_spectrum(fdtd, "monitor_through")
    return m._at_target(lam_c, Tc), m._at_target(lam_t, Tt)


def main():
    done = json.load(open(CK)) if os.path.isfile(CK) else {}
    lumapi = m.load_lumapi()
    respond = m._make_responder()
    sel = pick()
    print(f"[h3] {len(sel)} couplers selected; {len(done)} already done")
    for key, v in sel:
        if key in done:
            continue
        R, w, g, Lc = v["geom"]
        print(f"[h3] target {key}: R={R:.3f} w={w:.1f} g={g:.1f} Lc={Lc:.3f}  "
              f"flat kappa^2 {v['kappa2']:.6f}")
        t0 = time.time()
        try:
            k2d, t2d = run_one(lumapi, R, w, g, Lc)
        except Exception as exc:                        # noqa: BLE001
            print(f"[h3] FAILED {key}: {exc}\n[h3] re-run the same command to retry")
            time.sleep(20)
            continue
        try:
            kmod = float(respond(R, w, g, Lc)["kappa2"])
        except Exception:                               # noqa: BLE001
            kmod = float("nan")
        done[key] = dict(geom=[R, w, g, Lc], k2_flat=v["kappa2"], k2_disp=float(k2d),
                         t2_disp=float(t2d), k2_model=kmod,
                         minutes=round((time.time() - t0) / 60, 1))
        json.dump(done, open(CK, "w"), indent=1)
        print(f"[h3]   dispersive kappa^2 {k2d:.6f}  t^2 {t2d:.6f}  "
              f"({done[key]['minutes']} min)")

    rows = [done[k] for k, _ in sel if k in done]
    if not rows:
        return
    L = ["H3: coupler kappa^2 with flat vs dispersive silicon/silica (3-D FDTD, production mesh)",
         "=" * 78,
         f"{'target':>6} {'R um':>6} {'w nm':>6} {'g nm':>6} {'Lc um':>6} "
         f"{'flat':>9} {'disp':>9} {'model':>9} {'disp/flat-1':>12} {'model/disp-1':>13}"]
    dk, dm, df = [], [], []
    for k, _ in sel:
        if k not in done:
            continue
        r = done[k]; R, w, g, Lc = r["geom"]
        a = 100 * (r["k2_disp"] / r["k2_flat"] - 1)
        b = 100 * (r["k2_model"] / r["k2_disp"] - 1)
        c0 = 100 * (r["k2_model"] / r["k2_flat"] - 1)
        dk.append(a); dm.append(b); df.append(c0)
        L.append(f"{k:>6} {R:6.3f} {w:6.1f} {g:6.1f} {Lc:6.3f} {r['k2_flat']:9.5f} "
                 f"{r['k2_disp']:9.5f} {r['k2_model']:9.5f} {a:+11.2f}% {b:+12.2f}%")
    L += ["", f"n = {len(dk)}",
          f"kappa^2 dispersive vs flat: mean {np.mean(dk):+.2f} %, max |.| {np.max(np.abs(dk)):.2f} %",
          f"model vs flat   : mean {np.mean(df):+.2f} %   (the paper's coupler bias is about +3.3 % on 96 runs)",
          f"model vs disp.  : mean {np.mean(dm):+.2f} %, max |.| {np.max(np.abs(dm)):.2f} %"]
    txt = "\n".join(L); print("\n" + txt)
    open("results/h3_report.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
