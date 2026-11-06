"""
H1 -- confidence interval for the information-bound estimate (14.626 %).

No simulation and no training. It reuses the cached 10^6-geometry response bank
(results/floor_bank.npz, built by  mrr_template.py --floor) and the test split,
recomputes the estimate, and bootstraps over the 173 test targets
(B = 10000, seed 20261103, the same convention as the rest of the paper).

Three intervals are reported, because the estimate has a tuning step:
  (a) k fixed at the value that minimised the estimate       (optimistic)
  (b) k fixed at 5, chosen in advance of looking at the data  (the one to quote)
  (c) k re-chosen inside every resample                        (accounts for tuning)

Run from the project folder:  py -3 h1_floor_ci.py
Writes results/h1_floor_ci.txt
"""
import os, sys
import numpy as np

sys.path.insert(0, os.getcwd())
import mrr_template as m

B, SEED = 10000, 20261103
KS = (1, 2, 3, 4, 5, 7, 10, 25, 50, 100)
THRESH = 15.0


def row_terms(est, true):
    """per-row mean over the four squared relative errors (RMS-RGE convention)."""
    t = []
    for j, k in enumerate(m.INPUTS):
        d = est[:, j] - true[:, j]
        sp = m.LC_RANGE_UM if k == "Lc_um" else true[:, j]
        t.append((d / sp) ** 2)
    return np.mean(np.stack(t), axis=0)          # shape (n_test,)


def per_k_rows(G, R, Gr_te, Rr_te, ks):
    sd = R.std(axis=0); sd[sd == 0] = 1.0
    Bn = R / sd
    kmax = min(max(ks), len(G) - 1)
    ks = [k for k in ks if k <= kmax]
    order = np.empty((len(Gr_te), kmax), dtype=np.int64)
    for i in range(len(Gr_te)):
        d = np.sum((Bn - Rr_te[i] / sd) ** 2, axis=1)
        pick = np.argpartition(d, kmax)[:kmax]
        order[i] = pick[np.argsort(d[pick])]
    return ks, {k: row_terms(G[order[:, :k]].mean(axis=1), Gr_te) for k in ks}


def agg(rows, idx=None):
    r = rows if idx is None else rows[idx]
    return 100.0 * float(np.sqrt(np.mean(r)))


def main():
    resdir = "results" if os.path.isdir("results") else "."
    z = np.load(os.path.join(resdir, "floor_bank.npz"))
    G, R = z["G"], z["R"]
    Gr_te, Rr_te = m._load_split_raw("test")
    n = len(Gr_te)
    print(f"[h1] bank {len(G)} geometries, {n} test targets")
    ks, rows = per_k_rows(G, R, Gr_te, Rr_te, KS)
    pt = {k: agg(rows[k]) for k in ks}
    kbest = min(pt, key=pt.get)

    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, n, size=(B, n))
    res = {k: np.array([agg(rows[k], b) for b in boot]) for k in ks}
    reselect = np.array([min(res[k][i] for k in ks) for i in range(B)])

    def ci(x): return np.percentile(x, [2.5, 97.5])
    L = ["H1: information-bound estimate with bootstrap intervals", "=" * 66,
         f"bank {len(G)} geometries; {n} test targets; B = {B}, seed {SEED}",
         "", f"{'k':>5s} {'estimate %':>11s} {'95% interval':>18s} {'P(<15%)':>8s}"]
    for k in ks:
        lo, hi = ci(res[k])
        L.append(f"{k:5d} {pt[k]:11.3f} {'[%.2f, %.2f]' % (lo, hi):>18s} {np.mean(res[k] < THRESH):8.3f}"
                 + ("   <-- minimum" if k == kbest else "")
                 + ("   <-- k fixed in advance" if k == 5 else ""))
    lo, hi = ci(reselect)
    L += ["",
          f"(a) k = {kbest} (minimum): {pt[kbest]:.3f} %  [{ci(res[kbest])[0]:.2f}, {ci(res[kbest])[1]:.2f}]   P(<15 %) = {np.mean(res[kbest] < THRESH):.3f}",
          f"(b) k = 5 (fixed in advance): {pt[5]:.3f} %  [{ci(res[5])[0]:.2f}, {ci(res[5])[1]:.2f}]   P(<15 %) = {np.mean(res[5] < THRESH):.3f}",
          f"(c) k re-chosen in every resample: median {np.median(reselect):.3f} %  [{lo:.2f}, {hi:.2f}]   P(<15 %) = {np.mean(reselect < THRESH):.3f}",
          "",
          "What this covers: sampling variation over the 173 test targets only. It does not",
          "cover the variation from drawing a different 10^6 bank (seed), which is not measured here.",
          "The paper's quoted estimate is the minimum over k, so (a) and (c) show how much of the",
          "margin below 15 % could be a product of choosing k on the test targets."]
    txt = "\n".join(L); print("\n" + txt)
    open(os.path.join(resdir, "h1_floor_ci.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
