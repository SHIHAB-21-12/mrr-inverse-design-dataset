"""
H3 stability test -- why did the dispersive coupler runs blow up (t^2 > 1, Auto Shutoff growing)?

Target 140 (smallest coupler), 3 ps instead of 5 ps. Three runs, cheapest first:
  V0  flat silicon/silica (the production setting)        -> shows what a HEALTHY log looks like
  V2  dispersive fit over a WIDER band (1400-1700 nm), loose fit (tolerance 1e-3, 4 coefficients)
  V1  dispersive fit as in the failed runs, but dt stability factor 0.5
For each run the script prints Auto Shutoff at 25/50/75/100 % and kappa^2, t^2.
Healthy = Auto Shutoff falls or stays small, and t^2 <= 1.   Unstable = it keeps growing.
Resumable per variant: results/h3_stability.json.   Run:  py -3 h3_stability_test.py
"""
import os, re, sys, json, time
import numpy as np

sys.path.insert(0, os.getcwd())
import mrr_template as m
import g2c_dispersive as c

CK = os.path.join("results", "h3_stability.json")
SRC = os.path.join("results", "closedloop_naive.json")
KEY = "140"
SIM_TIME = 3e-12
LOG = "h3_stab_scratch_p0.log"
CORE_OBJS = ["ring_straight_bottom", "ring_bend_left", "ring_bend_right",
             "ring_exit_left", "ring_exit_right", "bus_input_through"]


def make_wide(fdtd, src, tag, log):
    """Sampled-data material over 1400-1700 nm, loose fit (few coefficients)."""
    lam = np.linspace(1400.0, 1700.0, 31)
    f = c.g.C / (lam * 1e-9)
    raw = np.real(np.array(fdtd.getindex(src, f)).flatten())
    made = fdtd.addmaterial("Sampled 3D data")
    made = made if isinstance(made, str) else "Sampled 3D data"
    fdtd.setmaterial(made, "name", tag)
    fdtd.setmaterial(tag, "sampled data", np.column_stack([f, raw ** 2]))
    for prop, val in (("tolerance", 1e-3), ("max coefficients", 4)):
        try:
            fdtd.setmaterial(tag, prop, val)
        except Exception as exc:                        # noqa: BLE001
            log.append(f"could not set {prop}: {exc}")
    try:
        _, ng_fit = c.fit_ng(fdtd, tag)
        ng_raw, _ = c.fit_ng(fdtd, src)
        log.append(f"{tag}: fitted n_g {ng_fit:.4f} (raw {ng_raw:.4f}, {100*(ng_fit/ng_raw-1):+.2f} %)")
    except Exception as exc:                            # noqa: BLE001
        log.append(f"{tag}: n_g check failed: {exc}")
    return tag


def shutoff_samples():
    txt = open(LOG, errors="ignore").read().replace("\n", " ")
    pts = [(float(a), float(b)) for a, b in
           re.findall(r"([\d.]+)% complete.*?Auto Shutoff: ([\deE+.\-]+)", txt)]
    out = {}
    for target in (25, 50, 75, 100):
        near = [v for p, v in pts if p <= target]
        out[str(target)] = near[-1] if near else None
    return out


def run_variant(lumapi, tag, geom):
    R, w, g, Lc = geom
    mesh = m.MeshConfig.uniform(m.BEND_MESH_NM * 1e-9)
    dom = m.DomainConfig(movie=False, dimension="3D", sim_time=SIM_TIME,
                         freq_points=m.COUPLER_FREQ_POINTS)
    p = m.MRRParams(R * 1e-6, w * 1e-9, g * 1e-9, Lc * 1e-6)
    log = []
    with lumapi.FDTD(hide=True) as fdtd:
        m.build_coupler(fdtd, p, mesh, dom)
        if tag == "V0_flat":
            pass
        else:
            if tag == "V2_wide_loose":
                si = make_wide(fdtd, m.MAT_CORE, "Si_wide", log)
                ox = make_wide(fdtd, m.MAT_CLAD, "SiO2_wide", log)
            else:                                        # V1_dt0.5, fit as in the failed runs
                si = c.try_make_dispersive(fdtd, m.MAT_CORE, "Si_disp", log)
                ox = c.try_make_dispersive(fdtd, m.MAT_CLAD, "SiO2_disp", log)
                fdtd.select("FDTD"); fdtd.set("dt stability factor", 0.5)
            if si is None or ox is None:
                raise RuntimeError("material not made: " + " | ".join(log))
            for nm in CORE_OBJS:
                if nm == "ring_straight_bottom" and Lc <= 0:
                    continue
                fdtd.select(nm); fdtd.set("material", si)
            fdtd.select("cladding"); fdtd.set("material", ox)
        fdtd.save(os.path.abspath("h3_stab_scratch.fsp"))
        fdtd.run()
        lam_c, Tc = m.get_spectrum(fdtd, "monitor_cross")
        lam_t, Tt = m.get_spectrum(fdtd, "monitor_through")
    return m._at_target(lam_c, Tc), m._at_target(lam_t, Tt), log


def main():
    d = json.load(open(CK)) if os.path.isfile(CK) else {}
    geom = json.load(open(SRC))[KEY]["geom"]
    lumapi = m.load_lumapi()
    print(f"[stab] target {KEY} geom {geom}, sim time {SIM_TIME*1e12:.1f} ps")
    for tag in ("V0_flat", "V2_wide_loose", "V1_dt0.5"):
        if tag in d:
            continue
        t0 = time.time()
        try:
            k2, t2, log = run_variant(lumapi, tag, geom)
        except Exception as exc:                         # noqa: BLE001
            print(f"[stab] FAILED {tag}: {exc}\n[stab] re-run the same command to retry")
            time.sleep(20)
            continue
        try:
            sh = shutoff_samples()
        except Exception as exc:                         # noqa: BLE001
            sh = {"error": str(exc)}
        d[tag] = dict(k2=float(k2), t2=float(t2), shutoff=sh, notes=log,
                      minutes=round((time.time() - t0) / 60, 1))
        json.dump(d, open(CK, "w"), indent=1)
        print(f"[stab] {tag}: kappa^2 {k2:.6f}  t^2 {t2:.6f}  Auto Shutoff at 25/50/75/100 %: "
              f"{sh}  ({d[tag]['minutes']} min)")
        for ln in log:
            print("        " + ln)
    print("\n[stab] SUMMARY")
    for tag, v in d.items():
        print(f"  {tag:14s} kappa^2 {v['k2']:.6f}  t^2 {v['t2']:.6f}  shutoff {v['shutoff']}")


if __name__ == "__main__":
    main()
