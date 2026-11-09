"""
P1 step 2: how many of (R, w, g, Lc) do the four targets pin down?
No Lumerical. Run:  py -3 p1_jacobian.py
At each of the 20 DE designs, finite-difference Jacobian of log(lambda_res, FSR, Q_L, IL)
w.r.t. the parameters scaled to their range (0..1). Singular values -> effective rank;
the right singular vector of the smallest value = the direction the targets cannot see.
Writes results/p1_jacobian_report.txt and results/p1_jacobian.json. mrr_template.py is not modified.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import mrr_template as m

resdir = "results" if os.path.isdir("results") else "."
cl = json.load(open(os.path.join(resdir, "p0_de_closedloop.json")))
respond = m._make_responder()
LO = np.array([5.0, 400.0, 150.0, 0.0]); HI = np.array([15.0, 500.0, 350.0, 3.0])
SPAN = HI - LO
NAMES = ["R", "w", "g", "Lc"]
h = 1e-3                                           # step in scaled units (0.1 % of range)


def out(x):
    r = respond(*map(float, LO + SPAN * x))
    return np.log(np.array([r[t] for t in m.TARGETS], dtype=float))


rows, out_json = [], {}
for k in cl:
    if cl[k].get("status") != "ok":
        continue
    x0 = (np.array(cl[k]["geom"]) - LO) / SPAN
    x0 = np.clip(x0, h, 1 - h)
    J = np.zeros((4, 4))
    for j in range(4):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h; xm[j] -= h
        J[:, j] = (out(xp) - out(xm)) / (2 * h)
    U, s, Vt = np.linalg.svd(J)
    rows.append((k, s, Vt))
    out_json[k] = dict(sing=s.tolist(), free_dir=Vt[-1].tolist(), second_dir=Vt[-2].tolist())

S = np.array([r[1] for r in rows])
L = ["P1 step 2 -- Jacobian of log(lambda_res, FSR, Q_L, IL) wrt scaled (R, w, g, Lc)", "=" * 78,
     f"{len(rows)} designs. Singular values s1>=s2>=s3>=s4 (median over designs, and min..max):"]
for i in range(4):
    L.append(f"   s{i+1}: median {np.median(S[:, i]):10.4g}   range {S[:, i].min():.4g} .. {S[:, i].max():.4g}")
L.append(f"   ratio s3/s1 median {np.median(S[:, 2]/S[:, 0]):.3g};  s4/s1 median {np.median(S[:, 3]/S[:, 0]):.3g}")
L.append("")
for tol in (1e-2, 1e-3, 1e-4):
    rk = (S / S[:, [0]] > tol).sum(1)
    L.append(f"   effective rank (s_i/s_1 > {tol:g}): " + ", ".join(f"{r}: {int((rk == r).sum())} designs" for r in sorted(set(rk))))
L.append("")
L.append("Direction the targets see LEAST (right singular vector of s4), averaged with sign alignment,")
L.append("components in order R, w, g, Lc (scaled units):")
V = np.array([r[2][-1] for r in rows])
V = V * np.sign(V[:, [0]] + 1e-12)
L.append("   mean " + np.array2string(V.mean(0), precision=3) + "   std " + np.array2string(V.std(0), precision=3))
L.append("Second least-seen direction (s3):")
V3 = np.array([r[2][-2] for r in rows]); V3 = V3 * np.sign(V3[:, [0]] + 1e-12)
L.append("   mean " + np.array2string(V3.mean(0), precision=3) + "   std " + np.array2string(V3.std(0), precision=3))
L.append("")
L.append("Reading: if s3 and s4 are far below s1, the four targets constrain only about two parameter")
L.append("combinations and the solution set is a 2-D family. If s3 is comparable to s2 the family is 1-D.")
txt = "\n".join(L)
open(os.path.join(resdir, "p1_jacobian_report.txt"), "w").write(txt + "\n")
json.dump(out_json, open(os.path.join(resdir, "p1_jacobian.json"), "w"), indent=1)
print(txt)
