"""
G3 -- clean report for the 100-target closed-loop run. NO SIMULATION.
Why this exists: the report printed by mrr_template.py counts EVERY record in
results/closedloop_naive.json, including 8 records left from the earlier
20-target run whose test rows are not among the 100 evenly spaced ones
(it printed 104 + 4 = 108 'targets' for a 100-target run). This script keeps
only the 100 rows the run was asked for.

Run from the project folder:  py -3 closedloop_report100.py
Reads  results/closedloop_naive.json, data/test.csv
Writes results/g3_closedloop100_report.txt
"""
import csv, json, os, sys
import numpy as np

N_TARGETS, B, SEED = 100, 10000, 20261103
TARGETS = ["lambda_res_nm", "FSR_nm", "Q_L", "IL_dB"]

def find(name):
    for d in ("data", "results", "."):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    sys.exit(f"cannot find {name}")

method = sys.argv[1] if len(sys.argv) > 1 else "naive"
done = json.load(open(find(f"closedloop_{method}.json")))
rows = list(csv.DictReader(open(find("test.csv"))))
Y = np.array([[float(r[k]) for k in TARGETS] for r in rows])
idx = [str(int(i)) for i in np.linspace(0, len(rows) - 1, N_TARGETS).astype(int)]
uniq = list(dict.fromkeys(idx))                       # linspace can repeat a row
missing = [k for k in uniq if k not in done]
stale = sorted(k for k in done if k not in set(uniq))

ok = [k for k in uniq if k in done and done[k]["status"] == "ok"]
un = [k for k in uniq if k in done and done[k]["status"] == "unrealisable"]

L = []
P = L.append
P(f"CLOSED-LOOP, {method}: clean report over the requested rows only")
P("=" * 70)
P(f"test rows in file: {len(rows)};  requested evenly spaced rows: {N_TARGETS} ({len(uniq)} distinct)")
P(f"records in json: {len(done)};  of these NOT in the requested set (ignored): {len(stale)} {stale}")
P(f"requested rows still missing (never simulated): {len(missing)} {missing}")
P(f"simulated successfully: {len(ok)};  unrealisable: {len(un)}  -> total accounted for {len(ok)+len(un)} of {len(uniq)}")
P("")
if missing:
    P("WARNING: some requested rows have no result. Re-run the closed-loop command first.")

# per-row relative errors (%), re-simulated vs requested
E = np.array([[100.0 * (done[k]["resim"][t] - Y[int(k), j]) / Y[int(k), j]
               for j, t in enumerate(TARGETS)] for k in ok])
def overall(e):
    return float(np.sqrt(np.mean(np.sqrt(np.mean(e ** 2, axis=0)) ** 2)))
P("per-target error (re-simulated vs requested), realisable rows")
P(f"  {'target':14s} {'MAPE %':>8s} {'RMS %':>8s} {'max %':>8s}")
for j, t in enumerate(TARGETS):
    P(f"  {t:14s} {np.mean(np.abs(E[:, j])):8.3f} {np.sqrt(np.mean(E[:, j] ** 2)):8.3f} {np.max(np.abs(E[:, j])):8.3f}")
ov = overall(E)
rng = np.random.default_rng(SEED)
bs = np.array([overall(E[rng.integers(0, len(E), len(E))]) for _ in range(B)])
lo, hi = np.percentile(bs, [2.5, 97.5])
P("")
P(f"RMS closed-loop response error = {ov:.3f} %   95% bootstrap CI [{lo:.2f}, {hi:.2f}]  (B={B}, seed {SEED}, over {len(ok)} realisable rows)")

# buildable fraction with Wilson interval
n, x = len(ok) + len(un), len(ok)
p = x / n; z = 1.96
c = (p + z * z / (2 * n)) / (1 + z * z / n)
h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
P(f"buildable fraction = {x}/{n} = {100*p:.1f} %   95% Wilson CI [{100*(c-h):.1f}, {100*(c+h):.1f}]")
P(f"share of realisable rows with every target within 20 %: {100*np.mean(np.all(np.abs(E) <= 20, axis=1)):.1f} %")
P(f"worst single-target error over all rows: {np.max(np.abs(E)):.2f} %")
txt = "\n".join(L)
print(txt)
os.makedirs("results", exist_ok=True)
open("results/g3_closedloop100_report.txt", "w").write(txt + "\n")
