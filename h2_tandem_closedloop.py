"""
H2 -- closed-loop re-simulation of the bounded and bounded+weighted tandem networks.

The paper evaluates these variants for geometry error, theta error and buildable fraction
but NOT in closed loop. This script trains them at seed 43 (the seed of the paper's tandem
network), takes the same 20 test targets as the closed-loop table (evenly spaced through the
test set), re-simulates every realisable predicted geometry with the production 3-D FDTD
coupler run (flat material fit, same as the 96 + 20 existing runs), and computes the ring
response analytically -- exactly the pipeline of run_closedloop in mrr_template.py.

Run from the project folder:
    py -3 h2_tandem_closedloop.py --check     (about 5 min, NO Lumerical: trains + checks reproducibility)
    py -3 h2_tandem_closedloop.py             (full run, about 40 FDTD couplers; resumable)
Needs e1_tandem.py's inputs (results/phase3_forward_model.pt, data/*.csv, data/scaler.json).
Writes results/h2_<variant>.json, results/h2_<variant>_s43.pt, results/h2_report.txt
mrr_template.py is not modified.
"""
import csv, json, os, sys, time
import numpy as np

sys.path.insert(0, os.getcwd())
import mrr_template as m

torch = m._torch()
from torch.utils.data import TensorDataset, DataLoader

CHECK = "--check" in sys.argv
SEED = 43
N_TARGETS = 20
BOOT, BOOT_SEED = 10000, 20261103
VARIANTS = [("bounded", True, False), ("bounded+weighted", True, True)]

outdir = "data" if os.path.isdir("data") else "."
resdir = "results" if os.path.isdir("results") else "."
sc = json.load(open(os.path.join(outdir, "scaler.json")))["scaler"]
Gtr, Rtr, _ = m._load_split("train")
Gva, Rva, _ = m._load_split("val")
Gte, Rte, _ = m._load_split("test")
Gr_te, Rr_te = m._load_split_raw("test")
f, _ck = m._load_forward(torch)
Gt = m._geo_physical(Gte, sc)

with torch.no_grad():
    ev = ((f(torch.from_numpy(Gva)) - torch.from_numpy(Rva)) ** 2).mean(dim=0)
inv = 1.0 / ev
W = (inv / inv.mean()).detach()


def train_tandem(bounded, weighted):               # identical to e1_tandem.py
    g = m._build_inverse_v(torch, SEED, bounded=bounded)

    def lossf(pred_geo, rb):
        e = (f(pred_geo) - rb) ** 2
        return (e * W).mean() if weighted else e.mean()

    dl = DataLoader(TensorDataset(torch.from_numpy(Rtr), torch.from_numpy(Gtr)),
                    batch_size=m.NN_BATCH, shuffle=True)
    opt = torch.optim.Adam(g.parameters(), lr=m.INV_LR, weight_decay=m.INV_WD)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=m.NN_LR_DECAY)
    rva = torch.from_numpy(Rva)
    best, state, bad = float("inf"), None, 0
    for ep in range(m.NN_MAX_EPOCHS):
        g.train()
        for rb, gb in dl:
            opt.zero_grad()
            lossf(g(rb), rb).backward()
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
    return g


def get_net(tag, bounded, weighted):
    path = os.path.join(resdir, f"h2_{tag}_s{SEED}.pt")
    g = m._build_inverse_v(torch, SEED, bounded=bounded)
    if os.path.isfile(path):
        g.load_state_dict(torch.load(path))
        print(f"[h2] {tag}: loaded saved network {path}")
    else:
        t0 = time.time()
        g = train_tandem(bounded, weighted)
        torch.save(g.state_dict(), path)
        print(f"[h2] {tag}: trained in {time.time()-t0:.0f} s, saved {path}")
    g.eval()
    return g


def e1_reference(tag):
    p = os.path.join(resdir, "e1_tandem.csv")
    if not os.path.isfile(p):
        return None
    for r in csv.DictReader(open(p)):
        if r["variant"] == tag and int(r["seed"]) == SEED:
            return float(r["rge"]), float(r["buildable"])
    return None


def inrange(row):
    bad = []
    for nm, v, (lo, hi) in (("R", row[0], (5.0, 15.0)), ("w", row[1], (400., 500.)),
                            ("g", row[2], (150., 350.)), ("Lc", row[3], (0., 3.))):
        if not (lo <= v <= hi):
            bad.append(f"{nm}={v:.3f} outside [{lo:g},{hi:g}]")
    return bad


def boot_rms(E):
    """E: (n, 4) relative errors in %. Statistic = sqrt(mean of squared errors), as in the report."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(E)
    s = np.array([np.sqrt(np.mean(E[rng.integers(0, n, n)] ** 2)) for _ in range(BOOT)])
    return float(np.sqrt(np.mean(E ** 2))), np.percentile(s, [2.5, 97.5])


idx = np.linspace(0, len(Gte) - 1, N_TARGETS).astype(int)
nets, preds = {}, {}
for tag, bd, wt in VARIANTS:
    nets[tag] = get_net(tag, bd, wt)
    with torch.no_grad():
        P_s = nets[tag](torch.from_numpy(Rte)).numpy()
    preds[tag] = m._geo_physical(P_s, sc)
    agg = m._rms_rge(preds[tag], Gt)[0]
    nb = sum(1 for i in idx if not inrange(preds[tag][i]))
    ref = e1_reference(tag)
    chk = ""
    if ref:
        chk = f"  | e1 CSV: RMS-RGE {ref[0]:.3f} -> {'MATCH' if abs(ref[0]-agg) < 0.01 else 'MISMATCH'}"
    print(f"[h2] {tag:17s} seed {SEED}: RMS-RGE {agg:.3f} %, {nb}/{N_TARGETS} of the chosen targets realisable{chk}")

if CHECK:
    print("[h2] --check finished: no FDTD was run. If both lines say MATCH (or no e1 CSV was found), run without --check.")
    sys.exit(0)

for tag, _, _ in VARIANTS:
    ckf = os.path.join(resdir, f"h2_{tag.replace('+','_')}.json")
    done = json.load(open(ckf)) if os.path.isfile(ckf) else {}
    print(f"[h2] {tag}: {len(done)} of {N_TARGETS} already done")
    for c, i in enumerate(idx, 1):
        key = str(int(i))
        if key in done:
            continue
        row = [float(x) for x in preds[tag][i]]
        bad = inrange(row)
        if bad:
            done[key] = dict(status="unrealisable", why="; ".join(bad), geom=row)
            json.dump(done, open(ckf, "w"), indent=1)
            print(f"[h2] {tag} {c}/{N_TARGETS}: UNREALISABLE -- {'; '.join(bad)}")
            continue
        R, w, g, Lc = row
        print(f"[h2] {tag} {c}/{N_TARGETS}: target {key} R={R:.3f} w={w:.1f} g={g:.1f} Lc={Lc:.3f} ...")
        t0 = time.time()
        try:
            k2, t2 = m._coupler_kappa2_fdtd(R, w, g, Lc)
        except Exception as exc:                        # noqa: BLE001
            print(f"[h2]   FAILED: {exc}  -- re-run to retry")
            time.sleep(20)
            continue
        resp = m._ring_given_kappa2(R, w, g, Lc, k2)
        done[key] = dict(status="ok", kappa2=float(k2), t2=float(t2), geom=row,
                         resim={k: float(resp[k]) for k in m.TARGETS},
                         minutes=(time.time() - t0) / 60.0)
        json.dump(done, open(ckf, "w"), indent=1)
        flag = "" if 0.95 <= k2 + t2 <= 1.05 and t2 <= 1.02 else "   <-- CHECK: energy not conserved"
        print(f"[h2]   kappa^2 {k2:.6f}  t^2 {t2:.6f}  sum {k2+t2:.4f}  ({done[key]['minutes']:.1f} min){flag}")

L = ["H2 -- closed-loop re-simulation of tandem variants (seed 43, 20 evenly spaced test targets)", "=" * 84,
     "3-D FDTD coupler (flat material fit, production mesh) + analytic ring; same pipeline as the", 
     "existing closed-loop table. RMS is over the realisable predictions, relative error in %.", ""]
for tag, _, _ in VARIANTS:
    ckf = os.path.join(resdir, f"h2_{tag.replace('+','_')}.json")
    done = json.load(open(ckf))
    ok = [k for k, v in done.items() if v["status"] == "ok"]
    un = [k for k, v in done.items() if v["status"] == "unrealisable"]
    L += [f"{tag}:  simulated {len(ok)}, unrealisable {len(un)} of {len(done)}"]
    if not ok:
        L.append("   no realisable prediction"); continue
    E = np.array([[100.0 * (done[k]["resim"][t] - Rr_te[int(k), j]) / Rr_te[int(k), j]
                   for j, t in enumerate(m.TARGETS)] for k in ok])
    for j, t in enumerate(m.TARGETS):
        L.append(f"   {t:16s} MAPE {np.mean(np.abs(E[:, j])):8.3f}   RMS {np.sqrt(np.mean(E[:, j]**2)):8.3f}   max {np.max(np.abs(E[:, j])):8.3f}")
    o, ci = boot_rms(E)
    L.append(f"   overall RMS closed-loop error {o:.3f} %  (bootstrap 95% [{ci[0]:.2f}, {ci[1]:.2f}], B={BOOT}, seed {BOOT_SEED})")
    L.append(f"   energy check: kappa^2 + t^2 mean {np.mean([done[k]['kappa2'] + done[k]['t2'] for k in ok]):.4f}")
    L.append("")
L += ["Reference (existing table): naive 3.989 % on 96 realisable of 100; tandem baseline 8 of 20 buildable.",
      "Single initialisation (seed 43); not a seed sweep."]
txt = "\n".join(L)
open(os.path.join(resdir, "h2_report.txt"), "w").write(txt + "\n")
print("\n" + txt)
