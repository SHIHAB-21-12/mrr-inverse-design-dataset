"""
G1 -- direct evidence of degeneracy. NO TRAINING, NO SIMULATION.
Finds pairs of dataset rows whose four TARGETS are practically the same
but whose GEOMETRIES are very different.

Run from the project folder (the one containing data/):  py -3 degeneracy_pairs.py
Reads:  data/mrr_dataset.csv   (falls back to ./mrr_dataset.csv)
Writes: results/g1_degeneracy_pairs.csv, results/g1_degeneracy_report.txt
"""
import csv, os, sys
import numpy as np

def find(name):
    for d in ("data", ".", "results"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    sys.exit(f"cannot find {name}")

rows = list(csv.DictReader(open(find("mrr_dataset.csv"))))
rows = [r for r in rows if (r.get("flag") or "").strip() == ""]
G = np.array([[float(r[k]) for k in ("R_um", "w_nm", "g_nm", "Lc_um")] for r in rows])
T = np.array([[float(r[k]) for k in ("lambda_res_nm", "FSR_nm", "Q_L", "IL_dB")] for r in rows])
n = len(G)

# tolerance for "same target" (stated in the paper): FSR 1 %, Q_L 5 %, IL 5 %, lambda_res 0.5 nm
TOL = dict(fsr=0.01, q=0.05, il=0.05, lam=0.5)

def rge_pair(a, b):
    """geometry difference, same convention as RMS-RGE: R,w,g relative; Lc over 3 um range"""
    rel = [(a[0]-b[0])/b[0], (a[1]-b[1])/b[1], (a[2]-b[2])/b[2], (a[3]-b[3])/3.0]
    return 100*np.sqrt(np.mean(np.square(rel)))

pairs = []
for i in range(n):
    for j in range(i+1, n):
        if abs(T[i,1]-T[j,1])/T[j,1] > TOL["fsr"]: continue
        if abs(T[i,2]-T[j,2])/T[j,2] > TOL["q"]:   continue
        if abs(T[i,3]-T[j,3])/T[j,3] > TOL["il"]:  continue
        if abs(T[i,0]-T[j,0]) > TOL["lam"]:        continue
        pairs.append((i, j, rge_pair(G[i], G[j])))
pairs.sort(key=lambda p: -p[2])

lines = []
P = lines.append
P(f"rows used: {n}")
P(f"tolerance: FSR {TOL['fsr']*100:.0f} %, Q_L {TOL['q']*100:.0f} %, IL {TOL['il']*100:.0f} %, lambda_res {TOL['lam']} nm")
P(f"pairs with indistinguishable targets: {len(pairs)}")
if pairs:
    d = np.array([p[2] for p in pairs])
    P(f"geometry difference (RMS-RGE-style, %): median {np.median(d):.2f}, 90th pct {np.percentile(d,90):.2f}, max {d.max():.2f}")
    P(f"pairs with geometry difference > 15 % (the naive network's own error): {(d>15).sum()}")
    P(f"pairs with geometry difference > 30 %: {(d>30).sum()}")
    P("")
    P("ten most different geometry pairs (all with matching targets):")
    P("  i    j   diff%   R_i/R_j (um)     w_i/w_j (nm)     g_i/g_j (nm)     Lc_i/Lc_j (um)   FSR_i/FSR_j (nm)  QL_i/QL_j        IL_i/IL_j (dB)")
    for i, j, dd in pairs[:10]:
        P(f"{i:4d} {j:4d} {dd:6.1f}  {G[i,0]:5.2f}/{G[j,0]:5.2f}   {G[i,1]:5.0f}/{G[j,1]:5.0f}     {G[i,2]:5.0f}/{G[j,2]:5.0f}     {G[i,3]:5.2f}/{G[j,3]:5.2f}     {T[i,1]:6.3f}/{T[j,1]:6.3f}   {T[i,2]:7.0f}/{T[j,2]:7.0f}   {T[i,3]:5.3f}/{T[j,3]:5.3f}")
txt = "\n".join(lines)
print(txt)
os.makedirs("results", exist_ok=True)
open("results/g1_degeneracy_report.txt", "w").write(txt + "\n")
with open("results/g1_degeneracy_pairs.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["i","j","geom_diff_pct"] + [f"{k}_i" for k in ("R","w","g","Lc","lam","FSR","QL","IL")] + [f"{k}_j" for k in ("R","w","g","Lc","lam","FSR","QL","IL")])
    for i, j, dd in pairs:
        w.writerow([i, j, f"{dd:.3f}"] + [f"{v:.6g}" for v in list(G[i])+list(T[i])] + [f"{v:.6g}" for v in list(G[j])+list(T[j])])
