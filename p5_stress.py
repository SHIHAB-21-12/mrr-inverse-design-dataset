"""
P5 (weakness 3, short check, seconds, NO Lumerical): do the CPO specs survive the KNOWN bias of the analytic ring?
Full-3-D-ring runs (ringvalidate.txt) found, at R=5.2 um:
   Q_L: simulated/analytic = 0.94-0.92 near analytic Q 4264, 0.73-0.83 at analytic Q 6351
   IL : simulated/analytic = up to 2.60 (0.85-1.42 dB simulated vs 0.36-0.56 dB analytic); 0.38-0.62 at the lowest Q
This script (a) re-scores the four P4 designs with those factors applied, and
(b) searches for designs that meet every band/bound AFTER applying them (feasibility under stress).
Levels:  moderate = Q x0.85, IL x1.6 ;  worst = Q x0.725, IL x2.6 .
Usage: py -3 p5_stress.py          writes results/p5_stress.json and results/p5_stress_report.txt
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
import mrr_template as m

NAMES = ["lambda_res_nm", "FSR_nm", "Q_L", "IL_dB"]
BOUNDS = [(5.0, 15.0), (400.0, 500.0), (150.0, 350.0), (0.0, 3.0)]
LEVELS = {"none": (1.0, 1.0), "moderate": (0.85, 1.6), "worst": (0.725, 2.6)}
resdir = "results" if os.path.isdir("results") else "."
respond = m._make_responder()


def stressed(r, lvl):
    qf, ilf = LEVELS[lvl]
    return dict(lambda_res_nm=r["lambda_res_nm"], FSR_nm=r["FSR_nm"], Q_L=r["Q_L"] * qf, IL_dB=r["IL_dB"] * ilf)


def verdict(S, r):
    miss = []
    if abs(r["lambda_res_nm"] - S["lam"]) > S["dlam_ch"] / 2: miss.append(f"lam {r['lambda_res_nm']-S['lam']:+.2f}")
    if r["FSR_nm"] < S["fsr_min"]: miss.append(f"FSR -{S['fsr_min']-r['FSR_nm']:.2f}")
    if r["Q_L"] < S["q_lo"]: miss.append(f"Q -{S['q_lo']-r['Q_L']:.0f}")
    if S["q_hi"] and r["Q_L"] > S["q_hi"]: miss.append(f"Q +{r['Q_L']-S['q_hi']:.0f}")
    if r["IL_dB"] > S["il_max"]: miss.append(f"IL +{r['IL_dB']-S['il_max']:.2f}")
    return "PASS" if not miss else "FAIL: " + "; ".join(miss)


def penalty(x, sk, lvl):
    S = m.SCENARIOS[sk]
    try:
        r = stressed(respond(*map(float, x)), lvl)
        if not np.all(np.isfinite([r[n] for n in NAMES])):
            return 1e3
    except Exception:                                          # noqa: BLE001
        return 1e3
    half = S["dlam_ch"] / 2.0
    v = [max(0.0, abs(r["lambda_res_nm"] - S["lam"]) - 0.25 * half) / half,   # keep a quarter-window margin on lambda
         max(0.0, S["fsr_min"] - r["FSR_nm"]) / S["fsr_min"],
         max(0.0, S["q_lo"] - r["Q_L"]) / S["q_lo"],
         max(0.0, r["IL_dB"] - S["il_max"]) / S["il_max"]]
    if S["q_hi"]:
        v.append(max(0.0, r["Q_L"] - S["q_hi"]) / S["q_hi"])
    return float(np.sqrt(np.sum(np.square(v))))


def solve(sk, lvl, restarts=6):
    from scipy.optimize import differential_evolution
    best, allres = None, []
    for k in range(restarts):
        res = differential_evolution(lambda xx: penalty(xx, sk, lvl), BOUNDS, seed=1000 + k, popsize=30, maxiter=600,
                                     tol=0, atol=0, mutation=(0.5, 1.0), recombination=0.8, polish=False,
                                     init="latinhypercube", callback=lambda xk, convergence=None: penalty(xk, sk, lvl) < 1e-9)
        allres.append(float(res.fun))
        if best is None or res.fun < best.fun:
            best = res
    return [float(v) for v in best.x], float(best.fun), allres


out, L = {}, []
FA = os.path.join(resdir, "p4_scen_analytic.json")
p4 = json.load(open(FA)) if os.path.isfile(FA) else {}
L += ["P5 -- do the CPO specs survive the known bias of the analytic ring?", "=" * 78,
      "Factors from the full 3-D ring runs at R=5.2 um (not measured at R>5.2 um):",
      "  moderate: Q x0.85, IL x1.6      worst: Q x0.725, IL x2.6", ""]
L += ["(a) the four P4 designs re-scored with the factors", "-" * 78,
      f"{'design':10s} {'level':9s} {'lam':>9s} {'FSR':>7s} {'Q_L':>8s} {'IL':>6s}  verdict"]
for sk in m.SCENARIOS:
    for var in ("spec0", "spec"):
        a = p4.get(f"{sk}|{var}")
        if not a:
            continue
        r = respond(*a["x"])
        for lvl in LEVELS:
            s = stressed(r, lvl)
            L.append(f"{sk}|{var:5s} {lvl:9s} {s['lambda_res_nm']:9.3f} {s['FSR_nm']:7.3f} {s['Q_L']:8.0f} {s['IL_dB']:6.3f}  {verdict(m.SCENARIOS[sk], s)}")
L += ["", "(b) feasibility under stress: best penalty found (0 = every spec met after the factors; lambda keeps 1/4-window margin)", "-" * 78]
for sk in m.SCENARIOS:
    for lvl in ("moderate", "worst"):
        t0 = time.time()
        x, loss, allres = solve(sk, lvl)
        r = respond(*x)
        s = stressed(r, lvl)
        out[f"{sk}|{lvl}"] = dict(x=x, loss=loss, restarts=allres, nominal={n: float(r[n]) for n in NAMES},
                                  stressed={n: float(s[n]) for n in NAMES}, verdict=verdict(m.SCENARIOS[sk], s))
        L.append(f"{sk} {lvl:9s} penalty {loss:.6f}  restarts [{', '.join(f'{v:.4f}' for v in allres)}]  ({time.time()-t0:.0f} s)")
        L.append(f"   geometry R={x[0]:.3f} w={x[1]:.1f} g={x[2]:.1f} Lc={x[3]:.3f}   nominal Q {r['Q_L']:.0f} IL {r['IL_dB']:.3f}  "
                 f"-> stressed Q {s['Q_L']:.0f} IL {s['IL_dB']:.3f}   {out[f'{sk}|{lvl}']['verdict']}")
json.dump(out, open(os.path.join(resdir, "p5_stress.json"), "w"), indent=1)
txt = "\n".join(L)
open(os.path.join(resdir, "p5_stress_report.txt"), "w").write(txt + "\n")
print(txt)

# ---------------------------------------------------------------- stage 2 (--fdtd): coupler in 3-D FDTD, ring analytic
if "--fdtd" in sys.argv:
    FF = os.path.join(resdir, "p5_stress_fdtd.json")
    cl = json.load(open(FF)) if os.path.isfile(FF) else {}
    for key, a in out.items():
        if key in cl:
            continue
        R, w, g, Lc = a["x"]
        print(f"\n[p5] FDTD {key}: R={R:.3f} w={w:.1f} g={g:.1f} Lc={Lc:.3f} ...")
        t0 = time.time()
        try:
            k2, t2 = m._coupler_kappa2_fdtd(R, w, g, Lc)
        except Exception as exc:                               # noqa: BLE001
            print(f"[p5]   FAILED: {exc}  -- re-run to retry")
            time.sleep(20)
            continue
        r = m._ring_given_kappa2(R, w, g, Lc, k2)
        cl[key] = dict(kappa2=float(k2), t2=float(t2), resim={n: float(r[n]) for n in NAMES}, minutes=(time.time() - t0) / 60)
        json.dump(cl, open(FF, "w"), indent=1)
        flag = "" if abs(k2 + t2 - 1) < 0.05 else "   <-- CHECK: energy not conserved"
        print(f"[p5]   kappa^2 {k2:.6f}  t^2 {t2:.6f}  sum {k2+t2:.4f}  ({cl[key]['minutes']:.1f} min){flag}")
    L2 = ["", "(c) robust designs re-simulated: coupler in 3-D FDTD, ring analytic, then the bias factors applied", "-" * 78,
          f"{'design':14s} {'level':9s} {'lam':>9s} {'FSR':>7s} {'Q_L':>8s} {'IL':>6s}  verdict"]
    for key, a in cl.items():
        sk, lvl_design = key.split("|")
        for lvl in ("none", "moderate", "worst"):
            s = stressed(a["resim"], lvl)
            L2.append(f"{key:14s} {lvl:9s} {s['lambda_res_nm']:9.3f} {s['FSR_nm']:7.3f} {s['Q_L']:8.0f} {s['IL_dB']:6.3f}  {verdict(m.SCENARIOS[sk], s)}")
    t2 = "\n".join(L2)
    open(os.path.join(resdir, "p5_stress_report.txt"), "a").write(t2 + "\n")
    print(t2)
