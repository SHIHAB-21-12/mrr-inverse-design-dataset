"""
P7 (weakness 6): is the information-bound estimate stable across INDEPENDENT response banks?
The paper's interval covers variation over the test targets only, not a different draw of the bank. This builds
independent banks (different seeds, same box, same analytic responder, same quality gate), computes the same
conditional-mean estimate with mrr_template._floor_at, and compares with the original bank (floor_bank.npz).
No Lumerical. Resumable (one .npz checkpoint per seed, saved every chunk).
  py -3 p7_banks.py --n 20000 --seeds 11            short check: timing + a first estimate (do this first)
  py -3 p7_banks.py --n 1000000 --seeds 11 22 33    the real run
Writes results/p7_banks_report.txt and results/p7_banks.json.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
import mrr_template as m

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seeds", type=int, nargs="+", default=[11])
args = ap.parse_args()
resdir = "results" if os.path.isdir("results") else "."
Gr_te, Rr_te = m._load_split_raw("test")
respond = m._make_responder()
lo = np.array([5.0, 400.0, 150.0, 0.0])
hi = np.array([15.0, 500.0, 350.0, 3.0])
CH = 10000


def build(seed, n):
    ck = os.path.join(resdir, f"p7_bank_s{seed}.npz")
    rng = np.random.default_rng(seed)
    if os.path.isfile(ck):
        z = np.load(ck)
        G, R, drawn = z["G"], z["R"], int(z["drawn"])
        print(f"[p7] seed {seed}: resuming, {len(G)} kept of {drawn} drawn")
        for _ in range(drawn // CH):                # replay the generator so the stream continues identically
            rng.random((CH, 4))
    else:
        G, R, drawn = np.zeros((0, 4)), np.zeros((0, 4)), 0
    t0, t_start = time.time(), len(G)
    while len(G) < n:
        cand = lo + (hi - lo) * rng.random((CH, 4))
        drawn += CH
        gs, rs = [], []
        for row in cand:
            try:
                v = respond(*row)
            except Exception:                        # noqa: BLE001
                continue
            if isinstance(v, dict) and v.get("flag"):
                continue
            v = np.asarray([v[t] for t in m.TARGETS], dtype=float) if isinstance(v, dict) else np.asarray(v, dtype=float)
            if v.shape != (4,) or not np.all(np.isfinite(v)):
                continue
            gs.append(row); rs.append(v)
        if gs:
            G = np.vstack([G, np.array(gs)]); R = np.vstack([R, np.array(rs)])
        np.savez(ck, G=G, R=R, drawn=drawn)
        rate = (len(G) - t_start) / max(time.time() - t0, 1e-9)
        print(f"[p7] seed {seed}: {len(G)}/{n} kept  ({rate:.0f} per s; about {(n-len(G))/max(rate,1e-9)/60:.1f} min left)")
    return G[:n], R[:n]


rows = []
for sd in args.seeds:
    G, R = build(sd, args.n)
    kb, agg, per, p12, curve = m._floor_at(G, R, Gr_te, Rr_te)
    k5 = [c for c in curve if c[0] == 5]
    k5v = k5[0][1] if k5 else float("nan")
    rows.append(dict(seed=sd, n=len(G), best_k=int(kb), floor_best_k=float(agg), floor_k5=float(k5v)))
    print(f"[p7] seed {sd}: N={len(G)}  best k={kb}  floor {agg:.3f} %  (k=5: {k5v:.3f} %)")

ref = None
zf = os.path.join(resdir, "floor_bank.npz")
if os.path.isfile(zf):
    z = np.load(zf)
    n = min(args.n, len(z["G"]))
    kb, agg, per, p12, curve = m._floor_at(z["G"][:n], z["R"][:n], Gr_te, Rr_te)
    k5 = [c for c in curve if c[0] == 5]
    ref = dict(seed="original", n=int(n), best_k=int(kb), floor_best_k=float(agg), floor_k5=float(k5[0][1]) if k5 else float("nan"))
    rows.insert(0, ref)

L = ["P7 -- information-bound estimate across independent response banks", "=" * 74,
     "Same box, same analytic responder, same quality gate; only the random draw of the bank differs.", "",
     f"  {'bank':>9s} {'N':>9s} {'best k':>7s} {'floor (best k) %':>17s} {'floor (k=5) %':>14s}", "  " + "-" * 62]
for r in rows:
    L.append(f"  {str(r['seed']):>9s} {r['n']:9d} {r['best_k']:7d} {r['floor_best_k']:17.3f} {r['floor_k5']:14.3f}")
new = [r for r in rows if r["seed"] != "original"]
if len(new) >= 2:
    v5 = np.array([r["floor_k5"] for r in new]); vb = np.array([r["floor_best_k"] for r in new])
    L += ["", f"  independent banks only: k=5 floor mean {v5.mean():.3f} %, sd {v5.std(ddof=1):.3f}, range {v5.min():.3f}-{v5.max():.3f}",
          f"                         best-k floor mean {vb.mean():.3f} %, sd {vb.std(ddof=1):.3f}, range {vb.min():.3f}-{vb.max():.3f}",
          f"  banks with k=5 floor below 15 %: {int(np.sum(v5 < 15))} of {len(v5)}"]
txt = "\n".join(L)
open(os.path.join(resdir, "p7_banks_report.txt"), "w").write(txt + "\n")
json.dump(rows, open(os.path.join(resdir, "p7_banks.json"), "w"), indent=1)
print("\n" + txt)
