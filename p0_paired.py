"""
Paired bootstrap over the SAME twenty targets: is the optimiser's closed-loop error
different from each network's?  No Lumerical, seconds.   Run:  py -3 p0_paired.py
Reads results/p0_de_closedloop.json plus closedloop_naive.json, h2_bounded.json,
h2_bounded_weighted.json (same formats as p0_baselines.py). Writes results/p0_paired_report.txt.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import mrr_template as m

resdir = "results" if os.path.isdir("results") else "."
_, Rr_te = m._load_split_raw("test")
idx = np.linspace(0, len(Rr_te) - 1, 20).astype(int)
keys = [str(int(i)) for i in idx]


def per_target_sq(path):
    d = json.load(open(os.path.join(resdir, path)))
    out = {}
    for k in keys:
        if k in d and d[k].get("status") == "ok":
            e = [100.0 * (d[k]["resim"][t] - Rr_te[int(k), j]) / Rr_te[int(k), j] for j, t in enumerate(m.TARGETS)]
            out[k] = float(np.mean(np.square(e)))          # mean squared error of the four responses, one target
    return out


de = per_target_sq("p0_de_closedloop.json")
others = {"naive network": "closedloop_naive.json", "bounded tandem": "h2_bounded.json",
          "bounded+weighted tandem": "h2_bounded_weighted.json"}
rng = np.random.default_rng(20261103)
L = ["Paired bootstrap: overall RMS closed-loop error, optimiser minus network (percentage points)", "=" * 78,
     "Negative = optimiser better. B = 10000 resamples of the common targets; 95 % percentile interval.", ""]
for name, f in others.items():
    try:
        o = per_target_sq(f)
    except Exception as exc:                                  # noqa: BLE001
        L.append(f"{name}: could not be read ({exc})"); continue
    common = [k for k in keys if k in de and k in o]
    a = np.array([de[k] for k in common]); b = np.array([o[k] for k in common]); n = len(common)
    diffs = []
    for _ in range(10000):
        i = rng.integers(0, n, n)
        diffs.append(np.sqrt(a[i].mean()) - np.sqrt(b[i].mean()))
    diffs = np.array(diffs)
    pt = np.sqrt(a.mean()) - np.sqrt(b.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    L.append(f"{name:26s} n={n}  difference {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]   optimiser lower in {100*np.mean(diffs < 0):.1f} % of resamples"
             f"   optimiser better on {int(np.sum(a < b))} of {n} targets")
L += ["", "If an interval contains 0, the two are not separable on these targets."]
txt = "\n".join(L)
open(os.path.join(resdir, "p0_paired_report.txt"), "w").write(txt + "\n")
print(txt)
