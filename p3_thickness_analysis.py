"""
P3 thickness test, step 2 (NO Lumerical): does choosing among equivalent designs reduce the combined
width + thickness resonance spread?   py -3 p3_thickness_analysis.py
Uses results/p3_thickness_neff.json and results/p0_de_closedloop.json.
First-order resonance shift: d(lambda) = lambda * d(n_eff) / n_g, with n_g = n_eff - lambda dn/dlambda.
Writes results/p3_thickness_report.txt. mrr_template.py is not modified.
"""
import json, os, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.getcwd())
import mrr_template as m

resdir = "results" if os.path.isdir("results") else "."
D = json.load(open(os.path.join(resdir, "p3_thickness_neff.json")))
W = [400, 425, 450, 475, 500]
n = lambda w, t, l: D[f"{w}|{t}|{l}"]
ng = np.array([n(w, 220, 1550) - 1550.0 * (n(w, 220, 1560) - n(w, 220, 1540)) / 20.0 for w in W])
dn_dw = np.gradient([n(w, 220, 1550) for w in W], W)                       # per nm
dn_dt = np.array([(n(w, 225, 1550) - n(w, 215, 1550)) / 10.0 for w in W])   # per nm
Sw = 1550.0 * dn_dw / ng                                                   # nm resonance per nm width
St = 1550.0 * dn_dt / ng                                                   # nm resonance per nm thickness
cSw, cSt = np.polyfit(W, Sw, 2), np.polyfit(W, St, 2)
L = ["P3 thickness test -- first-order resonance sensitivity from the mode solver", "=" * 78,
     "   w (nm)   n_eff    n_g     dlam/dw (nm/nm)   dlam/dt (nm/nm)"]
for i, w in enumerate(W):
    L.append(f"   {w:5d}  {n(w,220,1550):.4f}  {ng[i]:.3f}     {Sw[i]:8.4f}        {St[i]:8.4f}")
L.append(f"   thickness/width sensitivity ratio at w=400: {St[0]/Sw[0]:.2f},  at w=500: {St[-1]/Sw[-1]:.2f}")
L.append("")

_, Rr_te = m._load_split_raw("test")
respond = m._make_responder()
keys = list(json.load(open(os.path.join(resdir, "p0_de_closedloop.json"))).keys())
LO = np.array([5.0, 400.0, 150.0, 0.0]); HI = np.array([15.0, 500.0, 350.0, 3.0]); SPAN = HI - LO
rng = np.random.default_rng(7)


def family(tgt, starts=150):
    def res(z):
        try:
            r = respond(*map(float, LO + SPAN * z))
            v = np.array([r[t] for t in m.TARGETS], dtype=float)
            e = (v - tgt) / tgt
            return e if np.all(np.isfinite(e)) else np.full(4, 10.0)
        except Exception:                                  # noqa: BLE001
            return np.full(4, 10.0)
    out = []
    for _ in range(starts):
        try:
            s = least_squares(res, rng.random(4), bounds=(0, 1), diff_step=1e-4, xtol=1e-12, ftol=1e-12, max_nfev=200)
        except Exception:                                  # noqa: BLE001
            continue
        out.append((float(np.sqrt(np.mean(s.fun ** 2))), LO + SPAN * s.x))
    return out


fams = {}
for k in keys:
    fams[k] = [x for rms, x in family(np.array(Rr_te[int(k)], dtype=float)) if rms < 1e-2][:60]
L.append(f"equivalent designs per target (within 1 %): median {int(np.median([len(v) for v in fams.values()]))}")
L.append("")
L.append("Best-versus-median saving in resonance spread, median over targets (and targets with >= 30 %):")
L.append("   sigma_w (nm)  sigma_t (nm)   saving     targets >= 30 %   worst/best ratio (median)")
for sw in (0.7, 1.5):
    for st in (0.0, 1.0, 2.0, 3.0):
        sav, ratio = [], []
        for k, mem in fams.items():
            if len(mem) < 5:
                continue
            s = np.array([np.hypot(np.polyval(cSw, x[1]) * sw, np.polyval(cSt, x[1]) * st) for x in mem])
            sav.append((np.median(s) - s.min()) / np.median(s)); ratio.append(s.max() / s.min())
        sav = np.array(sav)
        L.append(f"   {sw:8.1f}      {st:8.1f}     {100*np.median(sav):5.1f} %     {int((sav >= 0.3).sum()):3d} of {len(sav):2d}"
                 f"          {np.median(ratio):.2f}")
L += ["", "Decision rule fixed in advance: saving >= 30 % (median over targets, and in most targets) at a",
      "realistic thickness spread reopens idea A; otherwise A stays a bounded model-level result.",
      "Thickness spread is a parameter here, not a measured value: a cited figure is needed before any claim."]
txt = "\n".join(L)
open(os.path.join(resdir, "p3_thickness_report.txt"), "w").write(txt + "\n")
print(txt)
