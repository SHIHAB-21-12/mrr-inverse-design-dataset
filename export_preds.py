"""
E2 -- export per-row TEST predictions of the saved naive and tandem inverse
networks. NO TRAINING. Loads results/phase4_naive_inverse.pt and
results/phase4_tandem_inverse.pt, exactly the files behind the paper's
15.506 % (naive) and 29.461 % (tandem) figures.

Run from the project folder (the one containing mrr_template.py, data/ and
results/):      py -3 export_preds.py
Writes:         results/inverse_test_predictions.csv
"""
import csv, os, sys
import numpy as np

sys.path.insert(0, os.getcwd())
import mrr_template as m           # safe: mrr_template only runs main() under __main__

torch = m._torch()
resdir = "results" if os.path.isdir("results") else "."
outdir = "data" if os.path.isdir("data") else "."
import json
sc = json.load(open(os.path.join(outdir, "scaler.json")))["scaler"]

Gte_s, Rte_s, rows = m._load_split("test")          # scaled arrays, test rows in file order
Gt = m._geo_physical(Gte_s, sc)                      # true geometry, physical units

preds = {}
for tag, fn in (("naive", "phase4_naive_inverse.pt"),
                ("tandem", "phase4_tandem_inverse.pt")):
    P_s, _ = m._dnn_predict(torch, os.path.join(resdir, fn), Rte_s)
    preds[tag] = m._geo_physical(P_s, sc)
    agg, per, p12 = m._rms_rge(preds[tag], Gt)
    print(f"[export] {tag:7s} RMS-RGE = {agg:.3f} %   per-parameter: "
          + ", ".join(f"{k} {v:.3f}" for k, v in zip(m.INPUTS, per)))

# integrity check against the numbers already in the paper
agg_n = m._rms_rge(preds["naive"], Gt)[0]
agg_t = m._rms_rge(preds["tandem"], Gt)[0]
ok_n = abs(agg_n - 15.506) < 0.01
ok_t = abs(agg_t - 29.461) < 0.01
print(f"[export] check naive  15.506 : {'OK' if ok_n else 'MISMATCH'}")
print(f"[export] check tandem 29.461 : {'OK' if ok_t else 'MISMATCH'}")

cols = (["row"]
        + [f"true_{k}" for k in m.INPUTS]
        + [f"naive_{k}" for k in m.INPUTS]
        + [f"tandem_{k}" for k in m.INPUTS])
path = os.path.join(resdir, "inverse_test_predictions.csv")
with open(path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    for i in range(len(Gt)):
        w.writerow([i] + [f"{v:.10g}" for v in Gt[i]]
                   + [f"{v:.10g}" for v in preds["naive"][i]]
                   + [f"{v:.10g}" for v in preds["tandem"][i]])
print(f"[export] wrote {path}  ({len(Gt)} rows)")
if not (ok_n and ok_t):
    print("[export] WARNING: a headline number did not reproduce -- do NOT send "
          "the CSV until this is explained; send me this printout instead.")
