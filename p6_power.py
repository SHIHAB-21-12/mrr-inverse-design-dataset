"""
P6 (weakness 5, no Lumerical, seconds): what can 20 closed-loop targets detect?   Run: py -3 p6_power.py
Reads per-target closed-loop records (same format as p0_paired.py) for: optimiser (DE), naive network,
bounded tandem, bounded+weighted tandem, RF raw, XGB norm, original tandem. Only buildable designs count.
For each pair on their COMMON targets it reports: n, difference in overall RMS (percentage points), a paired
bootstrap 95 % interval, the bootstrap standard error, the minimum detectable difference at 80 % power
(2.8 x SE, the usual two-sided 5 % / 80 % rule), and how many targets would be needed to detect the OBSERVED
difference at 80 % power (n x (2.8 SE / |diff|)^2). It also shows how much the naive network's interval narrows from
20 to 100 targets. Writes results/p6_power_report.txt.  mrr_template.py is not modified.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import mrr_template as m

resdir = "results" if os.path.isdir("results") else "."
_, Rr_te = m._load_split_raw("test")
idx20 = [str(int(i)) for i in np.linspace(0, len(Rr_te) - 1, 20).astype(int)]
FILES = {"DE": "p0_de_closedloop.json", "naive": "closedloop_naive.json", "bnd-tandem": "h2_bounded.json",
         "bnd+wt-tandem": "h2_bounded_weighted.json", "RF-raw": "closedloop_rf_raw.json",
         "XGB-norm": "closedloop_xgb_norm.json", "tandem": "closedloop_tandem.json"}


def load(path, keys=None):
    d = json.load(open(os.path.join(resdir, path)))
    out = {}
    for k, v in d.items():
        if keys is not None and k not in keys:
            continue
        if isinstance(v, dict) and v.get("status") == "ok" and "resim" in v:
            e = [100.0 * (v["resim"][t] - Rr_te[int(k), j]) / Rr_te[int(k), j] for j, t in enumerate(m.TARGETS)]
            out[k] = float(np.mean(np.square(e)))
    return out


data = {}
for name, f in FILES.items():
    try:
        data[name] = load(f, idx20)
    except Exception as exc:                                   # noqa: BLE001
        print(f"could not read {name}: {exc}")
idx100 = [str(int(i)) for i in np.linspace(0, len(Rr_te) - 1, 100).astype(int)]
naive100 = load(FILES["naive"], idx100)
rng = np.random.default_rng(20261103)
B = 10000


def boot_rms(a):
    n = len(a)
    r = np.array([np.sqrt(a[rng.integers(0, n, n)].mean()) for _ in range(B)])
    return float(np.sqrt(a.mean())), np.percentile(r, [2.5, 97.5])


L = ["P6 -- what can twenty closed-loop targets detect?", "=" * 100,
     "RMS closed-loop error (%), buildable designs only, bootstrap 95 % interval (B=10000):", ""]
for name, d in data.items():
    a = np.array(list(d.values()))
    pt, (lo, hi) = boot_rms(a)
    L.append(f"  {name:14s} n={len(a):3d}   RMS {pt:6.3f}   [{lo:6.3f}, {hi:6.3f}]   interval width {hi-lo:5.2f}")
a = np.array(list(naive100.values()))
pt, (lo, hi) = boot_rms(a)
L.append(f"  {'naive (100)':14s} n={len(a):3d}   RMS {pt:6.3f}   [{lo:6.3f}, {hi:6.3f}]   interval width {hi-lo:5.2f}")
L += ["", "Paired comparisons on common targets (difference = A minus B, percentage points; negative = A better):",
      f"  {'A':14s} {'B':14s} {'n':>3s} {'diff':>8s} {'95% interval':>20s} {'SE':>6s} {'MDD(80%)':>9s} {'n for observed diff':>20s}"]
pairs = [("DE", o) for o in data if o != "DE"] + [("naive", o) for o in data if o not in ("DE", "naive")]
for A, Bn in pairs:
    common = [k for k in idx20 if k in data[A] and k in data[Bn]]
    n = len(common)
    if n < 4:
        L.append(f"  {A:14s} {Bn:14s} {n:3d}   too few common buildable targets")
        continue
    a = np.array([data[A][k] for k in common]); b = np.array([data[Bn][k] for k in common])
    dif = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        dif.append(np.sqrt(a[i].mean()) - np.sqrt(b[i].mean()))
    dif = np.array(dif)
    pt = np.sqrt(a.mean()) - np.sqrt(b.mean())
    lo, hi = np.percentile(dif, [2.5, 97.5])
    se = float(dif.std())
    mdd = 2.8 * se
    nreq = int(np.ceil(n * (mdd / abs(pt)) ** 2)) if abs(pt) > 1e-9 else -1
    L.append(f"  {A:14s} {Bn:14s} {n:3d} {pt:+8.3f} {f'[{lo:+.3f}, {hi:+.3f}]':>20s} {se:6.3f} {mdd:9.3f} {nreq:20d}")
L += ["", "MDD = smallest true difference in overall RMS that these targets would detect 80 % of the time (two-sided 5 %).",
      "'n for observed diff' = targets needed to detect a difference the size of the observed one at 80 % power",
      "(if the observed difference is itself noise, this number is meaningless: read it only where the interval excludes 0",
      "or where it is of a size the paper would care about)."]
txt = "\n".join(L)
open(os.path.join(resdir, "p6_power_report.txt"), "w").write(txt + "\n")
print(txt)
