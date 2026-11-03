"""
E2 analysis -- bootstrap CIs over the test rows and paired differences.
Inputs : results/inverse_test_predictions.csv (from export_preds.py),
         results/phase5_results.json (tree models, per-row 'pred'),
         data/test.csv (true geometry, physical units)
Usage  : py -3 analyze_pairs.py [predictions.csv] [phase5.json] [test.csv]
"""
import csv, json, sys
import numpy as np

pred_csv = sys.argv[1] if len(sys.argv) > 1 else "results/inverse_test_predictions.csv"
p5_json  = sys.argv[2] if len(sys.argv) > 2 else "results/phase5_results.json"
test_csv = sys.argv[3] if len(sys.argv) > 3 else "data/test.csv"
INPUTS = ["R_um", "w_nm", "g_nm", "Lc_um"]
LC_RANGE = 3.0
B, SEED = 10000, 20261103

G = np.array([[float(r[k]) for k in INPUTS] for r in csv.DictReader(open(test_csv))])
n = len(G)

def sq_terms(P):                         # per-row, per-parameter squared normalised error
    t = np.empty_like(G)
    for j in range(4):
        sp = LC_RANGE if j == 3 else G[:, j]
        t[:, j] = ((P[:, j] - G[:, j]) / sp) ** 2
    return t.mean(axis=1)                # per-row mean over the 4 parameters

def rge(rowterm, idx=None):
    v = rowterm if idx is None else rowterm[idx]
    return 100.0 * np.sqrt(v.mean())

rows = list(csv.DictReader(open(pred_csv)))
assert len(rows) == n, f"{len(rows)} prediction rows vs {n} test rows"
def col(prefix): return np.array([[float(r[f"{prefix}_{k}"]) for k in INPUTS] for r in rows])
assert np.allclose(col("true"), G, rtol=1e-6, atol=1e-6), "true geometry does not match test.csv (row order?)"

d5 = json.load(open(p5_json))
models = {
    "naive DNN":    sq_terms(col("naive")),
    "tandem":       sq_terms(col("tandem")),
    "RF (raw)":     sq_terms(np.array(d5["inverse_rf_raw"]["pred"])),
    "RF (norm)":    sq_terms(np.array(d5["inverse_rf_norm"]["pred"])),
    "XGBoost (raw)":  sq_terms(np.array(d5["inverse_xgb_raw"]["pred"])),
    "XGBoost (norm)": sq_terms(np.array(d5["inverse_xgb_norm"]["pred"])),
}
rng = np.random.default_rng(SEED)
idx = rng.integers(0, n, size=(B, n))
boot = {k: 100.0 * np.sqrt(v[idx].mean(axis=1)) for k, v in models.items()}

L = [f"E2 -- bootstrap over {n} test rows, B = {B}, seed {SEED}, percentile intervals", "=" * 74,
     f"{'model':16s} {'RMS-RGE %':>10s} {'95% CI low':>11s} {'high':>8s}"]
for k, v in models.items():
    lo, hi = np.percentile(boot[k], [2.5, 97.5])
    L.append(f"{k:16s} {rge(v):10.3f} {lo:11.3f} {hi:8.3f}")
L += ["", "Paired differences (same resampled rows); negative = first model better",
      f"{'comparison':32s} {'diff pp':>8s} {'95% CI low':>11s} {'high':>8s} {'P(diff<0)':>10s}"]
pairs = [("naive DNN", "RF (raw)"), ("naive DNN", "RF (norm)"),
         ("naive DNN", "XGBoost (raw)"), ("naive DNN", "XGBoost (norm)"),
         ("RF (raw)", "XGBoost (norm)"), ("naive DNN", "tandem")]
for a, b in pairs:
    d = boot[a] - boot[b]
    lo, hi = np.percentile(d, [2.5, 97.5])
    L.append(f"{a+' - '+b:32s} {rge(models[a])-rge(models[b]):8.3f} {lo:11.3f} {hi:8.3f} {(d<0).mean():10.3f}")
thr = 15.0
d = boot["naive DNN"] - thr
lo, hi = np.percentile(boot["naive DNN"], [2.5, 97.5])
L += ["", f"naive DNN against the 15 % O3 threshold: point {rge(models['naive DNN']):.3f} %, "
          f"CI [{lo:.3f}, {hi:.3f}], bootstrap P(RMS-RGE <= 15 %) = {(boot['naive DNN']<=thr).mean():.3f}"]
txt = "\n".join(L)
print(txt)
open("results/e2_pairs.txt", "w").write(txt + "\n")
