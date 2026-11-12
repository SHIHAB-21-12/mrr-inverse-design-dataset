r"""
P9 step 1 -- two further response maps built on the SAME fitted coupler model,
mode-solver polynomial and bend-loss fit as the add-drop dataset (results\model_fits.json).

  addrop      : original map (identical couplers)          params R, w, g, Lc          (4-D)
  allpass     : one coupler, notch response                 params R, w, g, Lc          (4-D)
  addrop_asym : add-drop with two DIFFERENT couplers        params R, w, g1, Lc1, g2, Lc2 (6-D)

Targets (all maps): lambda_res, FSR, Q_L, and  IL_dB (add-drop maps)  /  depth = 1 - T_min (all-pass).

Usage (from the Phase1 folder):
  py -3 p9_responders.py --selftest            # verification against the existing dataset + brute force
  py -3 p9_responders.py --table               # target ranges and gate counts for each map
  py -3 p9_responders.py --jac                 # constraint-counting prediction (no training involved)
Nothing here trains a model or runs FDTD.
"""
import argparse, csv, glob, json, os, sys
import numpy as np
from scipy.optimize import brentq
from scipy.stats import qmc

RANGES = dict(R=(5e-6, 15e-6), w=(400e-9, 500e-9), g=(150e-9, 350e-9), Lc=(0.0, 3e-6))
RANGES_UM = dict(R=(5.0, 15.0), w=(400.0, 500.0), g=(150.0, 350.0), Lc=(0.0, 3.0))


def load_fits():
    if os.environ.get("P9_MOCK"):                      # sandbox logic test only, never used on real data
        return dict(kappa2=[-0.7, 0.0, 0.008, 3e-5, 1.02],
                    neff=[2.45, -0.0623, 0.12, -0.01, 0.003, -0.01, 0, 0, 0],
                    bend=[-6.0, -0.9], roughness_dB_cm=1.3)
    for cand in ("model_fits.json", os.path.join("results", "model_fits.json"),
                 os.path.join("data", "model_fits.json")):
        if os.path.isfile(cand):
            return json.load(open(cand))
    sys.exit("model_fits.json not found in ., results\\ or data\\")


class Model:
    def __init__(self, fits):
        self.kq = fits["kappa2"]; self.nc = np.array(fits["neff"])
        self.bp = fits["bend"]; self.rough = fits["roughness_dB_cm"]

    @staticmethod
    def _b(x, y):
        return np.array([1, x, y, x*x, x*y, y*y, x*x*y, x*y*y, y**3])

    @staticmethod
    def _db(x, y):
        return np.array([0, 1, 0, 2*x, y, 0, 2*x*y, y*y, 0])

    def neff(self, l, w):
        return float(self._b((l-1550)/50, (w-450)/50) @ self.nc)

    def ng(self, l, w):
        return self.neff(l, w) - l*float(self._db((l-1550)/50, (w-450)/50) @ self.nc)/50

    def kappa2(self, R, w, g, Lc):
        kq = self.kq; dw = w - 450.0; gam = kq[2] + kq[3]*dw
        th = np.exp(kq[0] + kq[1]*dw)*np.exp(-gam*g)*(Lc + kq[4]*np.sqrt(2*np.pi*R*1e3/gam)*1e-3)
        return float(np.sin(min(th, np.pi/2))**2)

    def loss_amp(self, R, L_um):
        bend = 4*np.exp(self.bp[0] + self.bp[1]*R)
        rough = self.rough*L_um*1e-4
        return 10**(-(bend + rough)/20), bend, rough

    def resonances(self, w, L_nm):
        f = lambda l, m: self.neff(l, w)*L_nm/l - m
        lo = int(np.ceil(self.neff(1600, w)*L_nm/1600)); hi = int(np.floor(self.neff(1500, w)*L_nm/1500))
        res = np.sort([brentq(f, 1495, 1605, args=(m,)) for m in range(lo, hi+1)])
        i = int(np.argmin(abs(res - 1550))); lr = res[i]
        fsr = float(np.min(np.abs(np.delete(res, i) - lr))) if res.size > 1 else np.nan
        return lr, fsr, int(round(self.neff(lr, w)*L_nm/lr))

    # ---------------- add-drop (identical or different couplers) -----------------
    def _addrop(self, R, w, c1, c2):
        (g1, L1), (g2, L2) = c1, c2
        L_um = 2*np.pi*R + L1 + L2; L = L_um*1e3
        lr, fsr, m = self.resonances(w, L)
        k1, k2 = self.kappa2(R, w, g1, L1), self.kappa2(R, w, g2, L2)
        t1, t2 = np.sqrt(1-k1), np.sqrt(1-k2)
        a, bend, rough = self.loss_amp(R, L_um)
        x = t1*t2*a
        Tpk = (1-t1**2)*(1-t2**2)*a/(1-x)**2
        ch = (1 + x*x - 2*(1-x)**2)/(2*x)
        flag = ""
        if -1 < ch < 1:
            fwhm = 2*np.arccos(ch)*lr**2/(2*np.pi*self.ng(lr, w)*L)
        else:
            fwhm = np.nan; flag = "no_half_max"
        if Tpk > 1:
            flag = (flag + ";T_drop_gt_1").strip(";")
        return dict(lambda_res_nm=lr, FSR_nm=fsr, FWHM_nm=fwhm, Q_L=lr/fwhm if fwhm == fwhm else np.nan,
                    IL_dB=-10*np.log10(Tpk), k1=k1, k2=k2, a=a, m=m, flag=flag)

    def addrop(self, R, w, g, Lc):
        return self._addrop(R, w, (g, Lc), (g, Lc))

    def addrop_asym(self, R, w, g1, Lc1, g2, Lc2):
        return self._addrop(R, w, (g1, Lc1), (g2, Lc2))

    # ---------------- all-pass ---------------------------------------------------
    def allpass(self, R, w, g, Lc):
        return self.allpass_given_k2(R, w, g, Lc, self.kappa2(R, w, g, Lc))

    def allpass_given_k2(self, R, w, g, Lc, k):
        """All-pass response with kappa^2 supplied (e.g. from a fresh 3-D FDTD coupler run)."""
        L_um = 2*np.pi*R + 2*Lc; L = L_um*1e3
        lr, fsr, m = self.resonances(w, L)
        t = np.sqrt(1-k)
        a, bend, rough = self.loss_amp(R, L_um)
        Tmin = ((t-a)/(1-t*a))**2
        h = (1+Tmin)/2
        A, B, C = a*a + t*t, 2*a*t, 1 + (a*t)**2
        flag = ""
        if abs(B*(h-1)) < 1e-15:
            c = np.inf
        else:
            c = (h*C - A)/(B*(h-1))
        if -1 < c < 1:
            fwhm = 2*np.arccos(c)*lr**2/(2*np.pi*self.ng(lr, w)*L)
        else:
            fwhm = np.nan; flag = "no_half_max"
        return dict(lambda_res_nm=lr, FSR_nm=fsr, FWHM_nm=fwhm, Q_L=lr/fwhm if fwhm == fwhm else np.nan,
                    depth=1-Tmin, Tmin=Tmin, k1=k, a=a, m=m, flag=flag)


MAPS = {
    "addrop":      dict(fn="addrop",      params=("R", "w", "g", "Lc"),               targets=("lambda_res_nm", "FSR_nm", "Q_L", "IL_dB")),
    "allpass":     dict(fn="allpass",     params=("R", "w", "g", "Lc"),               targets=("lambda_res_nm", "FSR_nm", "Q_L", "depth")),
    "addrop_asym": dict(fn="addrop_asym", params=("R", "w", "g1", "Lc1", "g2", "Lc2"), targets=("lambda_res_nm", "FSR_nm", "Q_L", "IL_dB")),
}


def box(params):
    lo, hi = [], []
    for p in params:
        key = "g" if p.startswith("g") else ("Lc" if p.startswith("Lc") else p)
        lo.append(RANGES_UM[key][0]); hi.append(RANGES_UM[key][1])
    return np.array(lo), np.array(hi)


def lhs(map_name, n, seed):
    lo, hi = box(MAPS[map_name]["params"])
    return lo + qmc.LatinHypercube(d=len(lo), seed=seed).random(n)*(hi - lo)


def evaluate(model, map_name, P):
    fn = getattr(model, MAPS[map_name]["fn"])
    return [fn(*p) for p in P]


# =============================================================================== selftest
def _find_dataset_csv():
    best = None
    for f in glob.glob(os.path.join("data", "*.csv")) + glob.glob("*.csv"):
        try:
            with open(f, newline="") as fh:
                rd = csv.DictReader(fh)
                if rd.fieldnames and {"R_um", "FWHM_nm", "kappa2", "Q_L"} <= set(rd.fieldnames):
                    rows = list(rd)
                    if best is None or len(rows) > best[1]:
                        best = (f, len(rows), rows)
        except Exception:
            pass
    return best


def selftest(model):
    ok = True
    print("P9 self-test")
    print("=" * 78)
    # 1. add-drop against the existing dataset ------------------------------------
    ds = _find_dataset_csv()
    if ds is None:
        print("1. add-drop vs existing dataset : dataset CSV not found (skipped) -- FIX before continuing")
        ok = False
    else:
        f, n, rows = ds
        worst = dict(lam=0, FSR=0, Q=0, IL=0); cnt = 0
        for r in rows[:300]:
            if r.get("flag"):
                continue
            o = model.addrop(float(r["R_um"]), float(r["w_nm"]), float(r["g_nm"]), float(r["Lc_um"]))
            worst["lam"] = max(worst["lam"], abs(o["lambda_res_nm"]/float(r["lambda_res_nm"]) - 1))
            worst["FSR"] = max(worst["FSR"], abs(o["FSR_nm"]/float(r["FSR_nm"]) - 1))
            worst["Q"] = max(worst["Q"], abs(o["Q_L"]/float(r["Q_L"]) - 1))
            worst["IL"] = max(worst["IL"], abs(o["IL_dB"]/float(r["IL_dB"]) - 1))
            cnt += 1
        good = max(worst.values()) < 1e-6
        ok &= good
        print(f"1. add-drop vs {f} ({cnt} rows): max rel. diff  lambda {worst['lam']:.1e}  FSR {worst['FSR']:.1e}  "
              f"Q {worst['Q']:.1e}  IL {worst['IL']:.1e}   -> {'PASS' if good else 'FAIL'}")
    # 2. asymmetric with equal couplers == add-drop --------------------------------
    P = lhs("addrop", 200, 1)
    d = 0
    for p in P:
        a = model.addrop(*p); b = model.addrop_asym(p[0], p[1], p[2], p[3], p[2], p[3])
        for k in ("lambda_res_nm", "FSR_nm", "Q_L", "IL_dB"):
            if a[k] == a[k] and b[k] == b[k]:
                d = max(d, abs(a[k] - b[k])/abs(a[k]))
    good = d < 1e-12; ok &= good
    print(f"2. asymmetric map with equal couplers == add-drop (200 points): max rel. diff {d:.1e}   -> {'PASS' if good else 'FAIL'}")
    # 3. swap symmetry of the asymmetric map ----------------------------------------
    P = lhs("addrop_asym", 200, 2); d = 0
    for p in P:
        a = model.addrop_asym(*p); b = model.addrop_asym(p[0], p[1], p[4], p[5], p[2], p[3])
        for k in ("FSR_nm", "Q_L", "IL_dB"):
            if a[k] == a[k] and b[k] == b[k]:
                d = max(d, abs(a[k] - b[k])/abs(a[k]))
    good = d < 1e-12; ok &= good
    print(f"3. swapping the two couplers leaves the add-drop response unchanged: max rel. diff {d:.1e}   -> {'PASS' if good else 'FAIL'}")
    # 4. all-pass closed form against a brute-force spectrum ------------------------
    P = lhs("allpass", 60, 3); worstQ = 0; worstD = 0; n_ok = 0
    for p in P:
        o = model.allpass(*p)
        if o["flag"]:
            continue
        R, w, g, Lc = p; L = (2*np.pi*R + 2*Lc)*1e3; lr = o["lambda_res_nm"]
        k = model.kappa2(R, w, g, Lc); t = np.sqrt(1-k); a = o["a"]
        dl = np.linspace(-1.5*o["FWHM_nm"]*3, 1.5*o["FWHM_nm"]*3, 40001)
        lam = lr + dl
        phi = 2*np.pi*np.array([model.neff(x, w) for x in lam])*L/lam
        z = np.exp(-1j*phi)
        T = np.abs((t - a*z)/(1 - t*a*z))**2
        Tmin_n = T.min(); half = (1 + Tmin_n)/2
        below = lam[T < half]
        if below.size < 2 or T[0] < half or T[-1] < half:
            continue
        fw = below.max() - below.min()
        worstQ = max(worstQ, abs(fw/o["FWHM_nm"] - 1)); worstD = max(worstD, abs((1-Tmin_n)/o["depth"] - 1)); n_ok += 1
    good = (n_ok >= 20) and worstQ < 5e-3 and worstD < 5e-3; ok &= good
    print(f"4. all-pass closed form vs brute-force spectrum ({n_ok} points): max rel. diff FWHM {worstQ:.1e}, depth {worstD:.1e}   -> {'PASS' if good else 'FAIL'}")
    # 5. all-pass physical limits ----------------------------------------------------
    # a = t -> critical coupling -> Tmin = 0 ; a = 1 -> Tmin = 1
    t = 0.9; a = 0.9
    c1 = ((t - a)/(1 - t*a))**2; c2 = ((t - 1.0)/(1 - t*1.0))**2
    good = abs(c1) < 1e-15 and abs(c2 - 1) < 1e-12; ok &= good
    print(f"5. all-pass limits: critical coupling Tmin = {c1:.1e} (expect 0), lossless Tmin = {c2:.3f} (expect 1)   -> {'PASS' if good else 'FAIL'}")
    P = lhs("allpass", 100, 5); d = 0
    for p in P:
        a = model.allpass(*p); b = model.allpass_given_k2(*p, model.kappa2(*p))
        for k in ("lambda_res_nm", "FSR_nm", "Q_L", "depth"):
            if a[k] == a[k]:
                d = max(d, abs(a[k]-b[k])/abs(a[k]))
    good = d < 1e-12; ok &= good
    print(f"6. all-pass with kappa^2 supplied == all-pass with the fitted kappa^2 (100 points): max rel. diff {d:.1e}   -> {'PASS' if good else 'FAIL'}")
    print("=" * 78)
    print("OVERALL:", "ALL CHECKS PASSED" if ok else "AT LEAST ONE CHECK FAILED -- do not continue, paste this output")
    return ok


# =============================================================================== table
def table(model, n=1150):
    print("P9 response maps -- %d Latin-hypercube points each (seed 20261112)" % n)
    for name, spec in MAPS.items():
        P = lhs(name, n, 20261112)
        rows = evaluate(model, name, P)
        flagged = [r for r in rows if r["flag"]]
        print("-" * 78)
        print(f"{name:12s}  params {spec['params']}   flagged {len(flagged)} of {n}")
        for tg in spec["targets"]:
            v = np.array([r[tg] for r in rows if not r["flag"]], float)
            v = v[np.isfinite(v)]
            print(f"   {tg:14s} min {v.min():10.4g}  p5 {np.percentile(v,5):10.4g}  median {np.median(v):10.4g}  "
                  f"p95 {np.percentile(v,95):10.4g}  max {v.max():10.4g}   (<1% of median: {np.mean(v < 0.01*np.median(v))*100:.1f}%)")


# =============================================================================== jacobian
def jac(model, n=200, seed=7):
    print("Constraint-counting prediction: singular values of d log(targets) / d (parameters scaled to their range)")
    print("(computed before any model is trained; medians over %d random designs)" % n)
    for name, spec in MAPS.items():
        lo, hi = box(spec["params"]); span = hi - lo
        P = lhs(name, n, seed); fn = getattr(model, spec["fn"]); S = []
        for p in P:
            base = fn(*p)
            if base["flag"]:
                continue
            J = []
            good = True
            for j in range(len(p)):
                h = 1e-3*span[j]; up = p.copy(); dn = p.copy(); up[j] += h; dn[j] -= h
                if up[j] > hi[j] or dn[j] < lo[j]:
                    good = False; break
                a, b = fn(*up), fn(*dn)
                if a["flag"] or b["flag"] or a["m"] != base["m"] or b["m"] != base["m"]:
                    good = False; break
                J.append([(np.log(a[t]) - np.log(b[t]))/2.0/1e-3 if np.isfinite(a[t]) and a[t] > 0 and b[t] > 0 else np.nan
                          for t in spec["targets"]])
            if not good:
                continue
            J = np.array(J).T                      # targets x params
            if np.all(np.isfinite(J)):
                S.append(np.linalg.svd(J, compute_uv=False))
        S = np.array(S)
        print("-" * 78)
        print(f"{name:12s} {len(spec['params'])} parameters, 4 targets, {len(S)} usable designs")
        print("   median singular values:", "  ".join(f"{x:.3g}" for x in np.median(S, axis=0)))
        print("   directions visible to the targets at s > 0.5 / 0.05 (median count): "
              f"{np.median((S > 0.5).sum(1)):.0f} / {np.median((S > 0.05).sum(1)):.0f}")
        print(f"   => parameters minus visible directions (s > 0.05): {len(spec['params']) - np.median((S > 0.05).sum(1)):.0f} "
              "(approximate dimension of the family of equivalent designs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--table", action="store_true")
    ap.add_argument("--jac", action="store_true"); ap.add_argument("--n", type=int, default=1150)
    a = ap.parse_args()
    M = Model(load_fits())
    if a.selftest: selftest(M)
    if a.table: table(M, a.n)
    if a.jac: jac(M)
    if not (a.selftest or a.table or a.jac):
        ap.print_help()
