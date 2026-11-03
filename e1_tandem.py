"""
E1 -- is the paper's tandem network a fair baseline?
Trains four tandem variants over five seeds and reports test accuracy,
reconstructed-response accuracy, theta error and buildable fraction.

  baseline          : unbounded outputs, plain MSE            (the paper's tandem)
  bounded           : sigmoid output, plain MSE
  weighted          : unbounded outputs, each target's squared error divided by
                      the frozen forward model's own validation MSE on that target
                      (no free parameter; down-weights lambda_res automatically)
  bounded+weighted  : both

Run from the project folder:
    py -3 e1_tandem.py --quick     (about 1 minute: 1 seed, 5 epochs, smoke test)
    py -3 e1_tandem.py             (full run: 4 variants x 5 seeds)
Writes results/e1_tandem.csv and results/e1_tandem.txt
Needs results/phase3_forward_model.pt, data/*.csv, data/scaler.json (already present).
NO file of mrr_template.py is modified.
"""
import csv, json, os, sys, time
import numpy as np

sys.path.insert(0, os.getcwd())
import mrr_template as m

torch = m._torch()
from torch.utils.data import TensorDataset, DataLoader

QUICK = "--quick" in sys.argv
SEEDS = [42] if QUICK else [42, 43, 44, 45, 46]
if QUICK:
    m.NN_MAX_EPOCHS = 5
VARIANTS = [("baseline", False, False), ("bounded", True, False),
            ("weighted", False, True), ("bounded+weighted", True, True)]

outdir = "data" if os.path.isdir("data") else "."
resdir = "results" if os.path.isdir("results") else "."
sc = json.load(open(os.path.join(outdir, "scaler.json")))["scaler"]

Gtr, Rtr, _ = m._load_split("train")
Gva, Rva, _ = m._load_split("val")
Gte, Rte, _ = m._load_split("test")
f, ck = m._load_forward(torch)                      # frozen forward model
Gt = m._geo_physical(Gte, sc)                       # true test geometry (physical)
Rt = m._res_physical(Rte, sc)                       # true test response (physical)

with torch.no_grad():                               # per-target weights, from validation only
    ev = ((f(torch.from_numpy(Gva)) - torch.from_numpy(Rva)) ** 2).mean(dim=0)
inv = 1.0 / ev
W = (inv / inv.mean()).detach()
print("[e1] forward-model validation MSE per target (scaled):",
      ", ".join(f"{k} {v:.5f}" for k, v in zip(m.TARGETS, ev.tolist())))
print("[e1] loss weights (mean 1):",
      ", ".join(f"{k} {v:.3f}" for k, v in zip(m.TARGETS, W.tolist())))


def train_tandem(seed, bounded, weighted):
    g = m._build_inverse_v(torch, seed, bounded=bounded)

    def lossf(pred_geo, rb):
        e = (f(pred_geo) - rb) ** 2
        return (e * W).mean() if weighted else e.mean()

    dl = DataLoader(TensorDataset(torch.from_numpy(Rtr), torch.from_numpy(Gtr)),
                    batch_size=m.NN_BATCH, shuffle=True)
    opt = torch.optim.Adam(g.parameters(), lr=m.INV_LR, weight_decay=m.INV_WD)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=m.NN_LR_DECAY)
    rva = torch.from_numpy(Rva)
    best, state, bad, ep = float("inf"), None, 0, 0
    for ep in range(m.NN_MAX_EPOCHS):
        g.train()
        for rb, gb in dl:
            opt.zero_grad()
            l = lossf(g(rb), rb)
            l.backward()
            opt.step()
        sch.step()
        g.eval()
        with torch.no_grad():
            v = lossf(g(rva), rva).item()
        if v < best - 1e-9:
            best, bad = v, 0
            state = {k: t.clone() for k, t in g.state_dict().items()}
        else:
            bad += 1
            if bad >= m.NN_PATIENCE:
                break
    if state is not None:
        g.load_state_dict(state)
    return g, ep + 1


def buildable(P):
    ok = np.ones(len(P), dtype=bool)
    for j, k in enumerate(m.INPUTS):
        lo, hi = m.RANGES[{"R_um": "R", "w_nm": "w", "g_nm": "g", "Lc_um": "Lc"}[k]]
        scl = {"R_um": 1e6, "w_nm": 1e9, "g_nm": 1e9, "Lc_um": 1e6}[k]
        ok &= (P[:, j] >= lo * scl) & (P[:, j] <= hi * scl)
    return 100.0 * float(ok.mean())


def evaluate(g):
    g.eval()
    with torch.no_grad():
        p = g(torch.from_numpy(Rte))
        r = f(p).numpy()
    P = m._geo_physical(p.numpy(), sc)
    agg, per, _ = m._rms_rge(P, Gt)
    Rrec = m._res_physical(r, sc)
    mape = [m._metrics(Rt[:, j], Rrec[:, j])[1] for j in range(4)]
    dd = m._degeneracy_check(P, Gt, "x")
    return dict(rge=agg, theta_mape=dd["theta_mape"], buildable=buildable(P),
                mape_lam=mape[0], mape_fsr=mape[1], mape_q=mape[2], mape_il=mape[3],
                mape_resp_mean=float(np.mean(mape)))


rows = []
for name, bounded, weighted in VARIANTS:
    for seed in SEEDS:
        t0 = time.time()
        g, ne = train_tandem(seed, bounded, weighted)
        r = evaluate(g)
        r.update(variant=name, seed=seed, epochs=ne)
        rows.append(r)
        print(f"[e1] {name:17s} seed {seed}  epochs {ne:4d}  RMS-RGE {r['rge']:7.3f}  "
              f"theta MAPE {r['theta_mape']:7.3f}  buildable {r['buildable']:5.1f}%  "
              f"resp MAPE {r['mape_resp_mean']:7.3f}  ({time.time()-t0:.0f}s)")

cols = ["variant", "seed", "epochs", "rge", "theta_mape", "buildable",
        "mape_lam", "mape_fsr", "mape_q", "mape_il", "mape_resp_mean"]
with open(os.path.join(resdir, "e1_tandem.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r[c] for c in cols})

L = [f"E1 -- tandem variants, seeds {SEEDS}{'  [QUICK SMOKE TEST, NOT RESULTS]' if QUICK else ''}", "=" * 96,
     f"{'variant':18s} {'RMS-RGE %':>16s} {'theta MAPE %':>16s} {'buildable %':>14s} "
     f"{'resp MAPE % (mean of 4)':>26s} {'lam_res MAPE':>13s}"]
def ms(v):
    v = np.array(v); s = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():8.3f} +/- {s:5.3f}"
for name, _, _ in VARIANTS:
    rr = [r for r in rows if r["variant"] == name]
    L.append(f"{name:18s} {ms([r['rge'] for r in rr]):>16s} {ms([r['theta_mape'] for r in rr]):>16s} "
             f"{np.mean([r['buildable'] for r in rr]):14.1f} {ms([r['mape_resp_mean'] for r in rr]):>26s} "
             f"{np.mean([r['mape_lam'] for r in rr]):13.2f}")
L += ["", "Reference points from the paper: naive inverse RMS-RGE 15.500 +/- 0.069 (5 seeds);",
      "tandem (seed 43) RMS-RGE 29.461, buildable 33.5 %, closed-loop error 5.936 % on buildable designs.",
      "Check: 'baseline' at seed 43 should print RMS-RGE 29.461 (it is the same network and seed as Phase 4).",
      "(--quick mode trains 5 epochs only, so its numbers mean nothing; it only tests that the script runs.)"]
txt = "\n".join(L)
print("\n" + txt)
open(os.path.join(resdir, "e1_tandem.txt"), "w").write(txt + "\n")
