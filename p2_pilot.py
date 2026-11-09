"""
P2 pilot -- does the choice among equivalent designs change fabrication sensitivity?
No Lumerical, a few minutes.  Run:  py -3 p2_pilot.py
For each of the 20 test targets:
  1. find many designs (R, w, g, Lc) that all meet the four targets (random-start least squares on the analytic model);
  2. apply the SAME random width/gap errors to every design (common random numbers) and measure the spread of
     resonance wavelength, Q_L and insertion loss;
  3. compare the best design with the median design of that family.
Writes results/p2_pilot_report.txt and results/p2_pilot.json.  mrr_template.py is not modified.
The sigma values are ASSUMPTIONS (adjust with --sigma_w / --sigma_g in nm); they must be replaced by cited values.
The model has no thickness variable, so thickness variation is NOT covered.
"""
import argparse, json, os, sys, time
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.getcwd())
import mrr_template as m

ap = argparse.ArgumentParser()
ap.add_argument("--sigma_w", type=float, default=5.0)    # nm, assumption
ap.add_argument("--sigma_g", type=float, default=5.0)    # nm, assumption
ap.add_argument("--starts", type=int, default=300)
ap.add_argument("--mc", type=int, default=200)
ap.add_argument("--max_members", type=int, default=60)
args = ap.parse_args()

resdir = "results" if os.path.isdir("results") else "."
LO = np.array([5.0, 400.0, 150.0, 0.0]); HI = np.array([15.0, 500.0, 350.0, 3.0]); SPAN = HI - LO
_, Rr_te = m._load_split_raw("test")
respond = m._make_responder()
keys = list(json.load(open(os.path.join(resdir, "p0_de_closedloop.json"))).keys())


def resp_vec(x):
    r = respond(*map(float, x))
    return r, np.array([r[t] for t in m.TARGETS], dtype=float)


def find_family(tgt, rng):
    def res(z):
        try:
            _, v = resp_vec(LO + SPAN * z)
            r = (v - tgt) / tgt
            return r if np.all(np.isfinite(r)) else np.full(4, 10.0)
        except Exception:                                   # noqa: BLE001
            return np.full(4, 10.0)
    out = []
    for _ in range(args.starts):
        z0 = rng.random(4)
        try:
            s = least_squares(res, z0, bounds=(0, 1), diff_step=1e-4, xtol=1e-12, ftol=1e-12, max_nfev=200)
        except Exception:                                   # noqa: BLE001
            continue
        rms = float(np.sqrt(np.mean(s.fun ** 2)))
        out.append((rms, LO + SPAN * s.x))
    return out


def wrap(d, fsr):
    return (d + fsr / 2.0) % fsr - fsr / 2.0


def mc_spread(x, tgt_lam, draws):
    r0, _ = resp_vec(x)
    q0, il0, fsr = r0["Q_L"], r0["IL_dB"], r0["FSR_nm"]
    dl, dq, di = [], [], []
    for dw, dg in draws:
        try:
            r = respond(float(x[0]), float(x[1] + dw), float(x[2] + dg), float(x[3]))
            dl.append(wrap(r["lambda_res_nm"] - r0["lambda_res_nm"], fsr))
            dq.append(r["Q_L"] / q0 - 1.0)
            di.append(r["IL_dB"] - il0)
        except Exception:                                   # noqa: BLE001
            pass
    return (float(np.sqrt(np.mean(np.square(dl)))), float(np.std(dq)), float(np.std(di)), len(dl))


rng_draw = np.random.default_rng(20261109)
draws = np.c_[rng_draw.normal(0, args.sigma_w, args.mc), rng_draw.normal(0, args.sigma_g, args.mc)]
rng = np.random.default_rng(7)
res_json = {}
t0 = time.time()
for c, k in enumerate(keys, 1):
    tgt = np.array(Rr_te[int(k)], dtype=float)
    fam = find_family(tgt, rng)
    entry = {}
    for tol in (1e-3, 1e-2):
        mem = [x for rms, x in fam if rms < tol]
        if len(mem) > args.max_members:
            mem = [mem[i] for i in rng.choice(len(mem), args.max_members, replace=False)]
        if len(mem) < 5:
            entry[str(tol)] = dict(n=len(mem))
            continue
        sp = np.array([mc_spread(x, tgt[0], draws)[:3] for x in mem])
        med, best = np.median(sp, axis=0), sp.min(axis=0)
        entry[str(tol)] = dict(n=len(mem), median=med.tolist(), best=best.tolist(),
                               saving=((med - best) / med).tolist(),
                               ratio_worst_best=(sp.max(axis=0) / sp.min(axis=0)).tolist())
    res_json[k] = entry
    e = entry["0.01"]
    msg = f"{e['n']} designs, saving (best vs median) lambda/Q/IL = " + \
        " / ".join(f"{100*v:.0f}%" for v in e["saving"]) if e.get("saving") else f"only {e['n']} designs found"
    print(f"[p2] {c}/{len(keys)} target {k}: {msg}  ({time.time()-t0:.0f} s)")
json.dump(res_json, open(os.path.join(resdir, "p2_pilot.json"), "w"), indent=1)

L = ["P2 pilot -- fabrication sensitivity across equivalent designs (analytic model)", "=" * 78,
     f"Assumed sigma: width {args.sigma_w} nm, gap {args.sigma_g} nm (assumptions). Thickness not modelled.",
     f"{args.mc} common random draws per design. 'saving' = (median - best) / median over the family members.", ""]
for tol, name in (("0.001", "targets matched within 0.1 %"), ("0.01", "targets matched within 1 % (below the 4.5 % model-fidelity floor)")):
    ok = [v[tol] for v in res_json.values() if v.get(tol, {}).get("saving")]
    L.append(f"{name}: {len(ok)} of {len(res_json)} targets had >= 5 designs")
    if not ok:
        continue
    S = np.array([o["saving"] for o in ok]); W = np.array([o["ratio_worst_best"] for o in ok])
    n = np.array([o["n"] for o in ok])
    L.append(f"   designs per target: median {int(np.median(n))}")
    for j, nm in enumerate(["resonance shift (trimming proxy)", "Q_L spread", "IL spread"]):
        L.append(f"   {nm:34s} saving median {100*np.median(S[:, j]):5.1f} %   targets with >= 30 %: "
                 f"{int((S[:, j] >= 0.3).sum())} of {len(ok)}   worst/best ratio median {np.median(W[:, j]):.2f}")
    L.append("")
L += ["Gate: resonance-shift saving >= 30 % (median over targets) and in most targets -> proceed with idea A.",
      "Below ~10 % -> A is a null result; submit the corrected current paper."]
txt = "\n".join(L)
open(os.path.join(resdir, "p2_pilot_report.txt"), "w").write(txt + "\n")
print("\n" + txt)
