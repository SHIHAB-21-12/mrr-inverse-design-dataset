"""
P4: can a direct optimiser meet the CPO scenarios?  (weakness 1 of the review)
Stage 1 (seconds, NO Lumerical):  py -3 p4_scenarios_de.py
Stage 2 (hours, 4 FDTD couplers): py -3 p4_scenarios_de.py --fdtd
Uses the SAME scenario definitions (m.SCENARIOS) and the SAME pass/fail rules as run_phase7sim.
Variants per scenario:
  paper   : target tuple exactly as in the paper (bounds hit exactly -> zero margin)
  margin  : FSR x1.04 and, where Q has only a lower bound, Q x1.10, so a pass is possible
Writes results/p4_scen_analytic.json, results/p4_scen_fdtd.json, results/p4_scen_report.txt.
mrr_template.py is not modified.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
import mrr_template as m

ap = argparse.ArgumentParser()
ap.add_argument("--fdtd", action="store_true")
ap.add_argument("--all", action="store_true", help="also re-simulate the infeasible point-target variants")
ap.add_argument("--restarts", type=int, default=6)
args = ap.parse_args()

NAMES = ["lambda_res_nm", "FSR_nm", "Q_L", "IL_dB"]
BOUNDS = [(5.0, 15.0), (400.0, 500.0), (150.0, 350.0), (0.0, 3.0)]
MODEL_LAM_UNC_NM = 1.55          # 0.10 % of 1550 nm: analytic ring vs full-ring FDTD (paper, ring validation)
resdir = "results" if os.path.isdir("results") else "."
FA, FF = os.path.join(resdir, "p4_scen_analytic.json"), os.path.join(resdir, "p4_scen_fdtd.json")
respond = m._make_responder()


def target_for(sk, variant):
    S = m.SCENARIOS[sk]
    t = list(S["target"])
    if variant == "margin":
        t[1] = S["fsr_min"] * 1.04
        if S["q_hi"] is None:
            t[2] = S["q_lo"] * 1.10
    return np.array(t, dtype=float)


def verdict(S, r):
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
    return ("PASS" if not miss else "FAIL: " + "; ".join(miss))


def penalty(x, sk, margin):
    """Violation of the scenario bands/bounds, normalised; 0 means every spec is met (with margin if margin=1)."""
    S = m.SCENARIOS[sk]
    try:
        r = respond(*map(float, x))
        lam, fsr, q, il = (r[n] for n in NAMES)
        if not np.all(np.isfinite([lam, fsr, q, il])):
            return 1e3
    except Exception:                                          # noqa: BLE001
        return 1e3
    half = S["dlam_ch"] / 2.0
    lam_tol = half * (0.25 if margin else 1.0)
    f = 1.04 if margin else 1.0
    qlo = S["q_lo"] * (1.10 if margin else 1.0)
    il_lim = S["il_max"] * (0.90 if margin else 1.0)
    v = [max(0.0, abs(lam - S["lam"]) - lam_tol) / half,
         max(0.0, S["fsr_min"] * f - fsr) / S["fsr_min"],
         max(0.0, qlo - q) / S["q_lo"],
         max(0.0, il - il_lim) / S["il_max"]]
    if S["q_hi"]:
        v.append(max(0.0, q - S["q_hi"] * (0.90 if margin else 1.0)) / S["q_hi"])
    return float(np.sqrt(np.sum(np.square(v))))


def objective(x, tgt):
    try:
        r = respond(*map(float, x))
        v = np.array([r[n] for n in NAMES], dtype=float)
        return float(np.sqrt(np.mean(((v - tgt) / tgt) ** 2))) if np.all(np.isfinite(v)) else 1e3
    except Exception:                                          # noqa: BLE001
        return 1e3


def solve(tgt, fun=None):
    from scipy.optimize import differential_evolution
    fun = fun or (lambda x: objective(x, tgt))
    best, allres = None, []
    for k in range(args.restarts):
        res = differential_evolution(fun, BOUNDS, seed=1000 + k, popsize=30, maxiter=600,
                                     tol=0, atol=0, mutation=(0.5, 1.0), recombination=0.8, polish=False,
                                     init="latinhypercube", callback=lambda xk, convergence=None: fun(xk) < 1e-9)
        allres.append(float(res.fun))
        if best is None or res.fun < best.fun:
            best = res
    return [float(v) for v in best.x], float(best.fun), allres


an = json.load(open(FA)) if os.path.isfile(FA) else {}
for sk in m.SCENARIOS:
    for var in ("paper", "margin", "spec0", "spec"):
        key = f"{sk}|{var}"
        if key in an:
            continue
        tgt = target_for(sk, "margin" if var == "spec" else var if var in ("paper", "margin") else "paper")
        t0 = time.time()
        if var in ("spec0", "spec"):
            x, loss, allres = solve(tgt, lambda xx, sk=sk, mg=int(var == "spec"): penalty(xx, sk, mg))
        else:
            x, loss, allres = solve(tgt)
        r = respond(*x)
        an[key] = dict(target=tgt.tolist(), x=x, loss=loss, restarts=allres, analytic={n: float(r[n]) for n in NAMES},
                       analytic_verdict=verdict(m.SCENARIOS[sk], r), seconds=time.time() - t0)
        json.dump(an, open(FA, "w"), indent=1)
        print(f"[p4] {key}: target {np.round(tgt,2).tolist()}  best objective {loss:.6f}  "
              f"(restarts: {', '.join(f'{v:.4f}' for v in allres)})  {an[key]['seconds']:.0f} s")
        print(f"       geometry R={x[0]:.3f} w={x[1]:.1f} g={x[2]:.1f} Lc={x[3]:.3f}   analytic verdict: {an[key]['analytic_verdict']}")

if not args.fdtd:
    print("\n[p4] stage 1 only. If the analytic results look right, run again with --fdtd (4 couplers, hours).")
    sys.exit(0)

cl = json.load(open(FF)) if os.path.isfile(FF) else {}
for key, a in an.items():
    if key in cl or (not args.all and key.split("|")[1] in ("paper", "margin")):
        continue
    R, w, g, Lc = a["x"]
    print(f"[p4] FDTD {key}: R={R:.3f} w={w:.1f} g={g:.1f} Lc={Lc:.3f} ...")
    t0 = time.time()
    try:
        k2, t2 = m._coupler_kappa2_fdtd(R, w, g, Lc)
    except Exception as exc:                                   # noqa: BLE001
        print(f"[p4]   FAILED: {exc}  -- re-run to retry")
        time.sleep(20)
        continue
    r = m._ring_given_kappa2(R, w, g, Lc, k2)
    cl[key] = dict(kappa2=float(k2), t2=float(t2), resim={n: float(r[n]) for n in NAMES}, minutes=(time.time() - t0) / 60)
    json.dump(cl, open(FF, "w"), indent=1)
    flag = "" if abs(k2 + t2 - 1) < 0.05 else "   <-- CHECK: energy not conserved"
    print(f"[p4]   kappa^2 {k2:.6f}  t^2 {t2:.6f}  sum {k2+t2:.4f}  ({cl[key]['minutes']:.1f} min){flag}")

L = ["P4 -- direct optimiser vs the CPO scenarios (coupler in 3-D FDTD, ring analytic)", "=" * 86,
     f"Resonance uncertainty of the analytic ring vs full-ring FDTD: +/-{MODEL_LAM_UNC_NM} nm (0.10 %).", ""]
for sk, S in m.SCENARIOS.items():
    L += [f"Scenario {sk}: window on lambda_res +/-{S['dlam_ch']/2:g} nm   (model uncertainty / window = {MODEL_LAM_UNC_NM/(S['dlam_ch']/2):.2f})",
          f"   {'variant':8s} {'lam':>9s} {'FSR':>7s} {'Q_L':>8s} {'IL':>6s}  verdict (3-D FDTD coupler)"]
    for var in ("spec0", "spec", "paper", "margin"):
        key = f"{sk}|{var}"
        if key in cl:
            r = cl[key]["resim"]
            L.append(f"   {var:8s} {r['lambda_res_nm']:9.3f} {r['FSR_nm']:7.3f} {r['Q_L']:8.0f} {r['IL_dB']:6.3f}  {verdict(S, r)}")
        else:
            L.append(f"   {var:8s} not simulated")
    L.append("")
L += ["Variants: spec0 = meet every band/bound exactly; spec = meet them with margin (FSR x1.04, Q_lo x1.10, IL x0.9, lambda within a quarter of the window).",
      "paper / margin = point targets as in the paper (infeasible in the model, not simulated unless --all).",
      "lambda_res is exact inside the model by construction; its real uncertainty is the +/- value above."]
txt = "\n".join(L)
open(os.path.join(resdir, "p4_scen_report.txt"), "w").write(txt + "\n")
print("\n" + txt)
