"""
P1 step 1: how big and how systematic is the kappa^2 discrepancy (3-D FDTD vs fitted model)?
No Lumerical. Uses results/p0_de_closedloop.json. Run:  py -3 p1_kappa_discrepancy.py
Writes results/p1_kappa_report.txt. mrr_template.py is not modified.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import mrr_template as m

resdir = "results" if os.path.isdir("results") else "."
cl = json.load(open(os.path.join(resdir, "p0_de_closedloop.json")))
_, Rr_te = m._load_split_raw("test")
respond = m._make_responder()

keys = [k for k in cl if cl[k].get("status") == "ok"]
G = np.array([cl[k]["geom"] for k in keys])                 # R, w, g, Lc
k2f = np.array([cl[k]["kappa2"] for k in keys])
k2m = np.array([respond(*map(float, g))["kappa2"] for g in G])
lr = np.log(k2f / k2m)                                      # log ratio FDTD / model
n = len(keys)

L = []
L.append("P1 step 1 -- kappa^2: 3-D FDTD vs fitted model, at the 20 DE designs")
L.append("=" * 78)
L.append(f"n = {n}.  ratio = kappa2_FDTD / kappa2_model")
L.append(f"median ratio {np.exp(np.median(lr)):.3f}   mean |log ratio| {np.mean(np.abs(lr)):.3f}"
         f"   std of log ratio {np.std(lr, ddof=1):.3f}   range {np.exp(lr.min()):.3f} .. {np.exp(lr.max()):.3f}")
L.append("")
L.append("Correlation of log ratio with each geometry parameter (|r| > 0.45 is significant at n=20, ~5 %):")
for j, nm in enumerate(["R", "w", "g", "Lc"]):
    L.append(f"   {nm:3s} r = {np.corrcoef(G[:, j], lr)[0, 1]:+.2f}")
L.append(f"   log(kappa2_model) r = {np.corrcoef(np.log(k2m), lr)[0, 1]:+.2f}")

# leave-one-out linear correction of log ratio on standardised [R, w, g, Lc]
X = (G - G.mean(0)) / G.std(0)
X1 = np.c_[np.ones(n), X]
pred = np.zeros(n)
for i in range(n):
    tr = np.arange(n) != i
    beta = np.linalg.lstsq(X1[tr], lr[tr], rcond=None)[0]
    pred[i] = X1[i] @ beta
loo_const = np.array([np.median(np.delete(lr, i)) for i in range(n)])


def closed_loop_rms(k2_used):
    E = []
    for i, k in enumerate(keys):
        resp = m._ring_given_kappa2(*map(float, G[i]), float(k2_used[i]))
        E.append([100.0 * (resp[t] - Rr_te[int(k), j]) / Rr_te[int(k), j] for j, t in enumerate(m.TARGETS)])
    E = np.array(E)
    return np.sqrt(np.mean(E ** 2)), np.sqrt(np.mean(E ** 2, axis=0))


rms_f, per_f = closed_loop_rms(k2f)                                 # reproduces the 2.89 % (FDTD kappa2)
rms_m, per_m = closed_loop_rms(k2m)                                 # model kappa2 (what DE assumed)
rms_c, per_c = closed_loop_rms(k2m * np.exp(loo_const))             # model x constant LOO factor
rms_l, per_l = closed_loop_rms(k2m * np.exp(pred))                  # model x LOO linear correction
L.append("")
L.append("Overall RMS error vs the targets, using different kappa^2 in the ring model (%):")
L.append(f"   FDTD kappa2 (reproduces closed-loop table)     {rms_f:6.3f}   per target {np.round(per_f, 3).tolist()}")
L.append(f"   model kappa2 (check: should be ~0)             {rms_m:6.3f}")
L.append(f"   model kappa2 x constant factor (leave-one-out) {rms_c:6.3f}")
L.append(f"   model kappa2 x linear fit in R,w,g,Lc (LOO)    {rms_l:6.3f}")
L.append("")
L.append("Reading: if the LOO-corrected value is clearly below the FDTD value, the discrepancy is systematic")
L.append("and removable. If not, it is the real fidelity limit and is reported as such. n = 20: wide uncertainty.")
txt = "\n".join(L)
open(os.path.join(resdir, "p1_kappa_report.txt"), "w").write(txt + "\n")
print(txt)
