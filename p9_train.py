r"""
P9 step 2 -- train the paper's models on the three response maps and test the pre-registered predictions
(p9_predictions.md).  No FDTD.  Settings are those of Appendix B of the paper; nothing is tuned here.

  py -3 p9_train.py --smoke                 # 1 seed, 20 epochs, small trees: only checks that everything runs
  py -3 p9_train.py                         # full run: 3 maps x 5 seeds x {naive, tandem, RF, XGB}
  py -3 p9_train.py --maps addrop --methods naive,rf   # subsets

Writes results\p9_results.json and results\p9_report.txt.
"""
import argparse, json, os, sys, time
import numpy as np
from sklearn.metrics import r2_score
import p9_responders as R9

SEEDS = [42, 43, 44, 45, 46]
ORDERED = False        # post-hoc diagnostic only (--ordered): canonical coupler order g1 <= g2
N = 1150
LHS_SEED = 20261112
HID_INV = (64, 128, 64)
FWD_DEPTH, FWD_WIDTH = 4, 256          # "depth 4, width 256" read as four hidden layers (assumption, affects the tandem forward model only)


def pkey(p):
    return "g" if p.startswith("g") else ("Lc" if p.startswith("Lc") else p)


def make_data(model, name):
    spec = R9.MAPS[name]
    P = R9.lhs(name, N, LHS_SEED)
    if ORDERED and name == "addrop_asym":            # swap the two couplers so that g1 <= g2 (response is unchanged: p9_responders self-test 3)
        sw = P[:, 2] > P[:, 4]
        P[sw, 2], P[sw, 4] = P[sw, 4].copy(), P[sw, 2].copy()
        P[sw, 3], P[sw, 5] = P[sw, 5].copy(), P[sw, 3].copy()
    rows = R9.evaluate(model, name, P)
    keep = [i for i, r in enumerate(rows) if not r["flag"] and all(np.isfinite(r[t]) and r[t] > 0 for t in spec["targets"])]
    G = P[keep]; T = np.array([[rows[i][t] for t in spec["targets"]] for i in keep], float)
    rng = np.random.RandomState(42); perm = rng.permutation(len(G))
    ntr, nva = 805, 172
    idx = dict(train=perm[:ntr], val=perm[ntr:ntr+nva], test=perm[ntr+nva:])
    return spec, G, T, idx, N - len(keep)


def tt(T):                                  # transformed targets: log10 of Q and of IL/depth
    X = T.copy(); X[:, 2] = np.log10(X[:, 2]); X[:, 3] = np.log10(X[:, 3]); return X


def rge_terms(Gp, Gt, params):
    out = []
    for j, p in enumerate(params):
        if pkey(p) == "Lc":
            out.append((Gp[:, j] - Gt[:, j])/3.0)
        else:
            out.append((Gp[:, j] - Gt[:, j])/Gt[:, j])
    return np.array(out).T                  # n x d


def evaluate_pred(model, name, spec, Gp, Gt, Tt_true):
    params = spec["params"]; lo, hi = R9.box(params)
    d = len(params)
    terms = rge_terms(Gp, Gt, params)
    agg = 100*np.sqrt(np.mean(terms**2))
    nonlc = [j for j, p in enumerate(params) if pkey(p) != "Lc"]
    agg_nonlc = 100*np.sqrt(np.mean(terms[:, nonlc]**2))
    r2 = [r2_score(Gt[:, j], Gp[:, j]) for j in range(d)]
    inside = np.all((Gp >= lo - 1e-9) & (Gp <= hi + 1e-9), axis=1)
    fn = getattr(model, spec["fn"]); errs = []; nflag = 0
    for i in np.where(inside)[0]:
        try:
            o = fn(*Gp[i])
        except Exception:
            nflag += 1; continue
        if o["flag"] or not all(np.isfinite(o[t]) and o[t] > 0 for t in spec["targets"]):
            nflag += 1; continue
        v = np.array([o[t] for t in spec["targets"]])
        errs.append((v - Tt_true[i])/Tt_true[i])
    errs = np.array(errs)
    resp = 100*np.sqrt(np.mean(errs**2)) if len(errs) else float("nan")
    return dict(rge=agg, rge_nonlc=agg_nonlc, r2=r2, n_rec=int(sum(x >= 0.8 for x in r2)),
                buildable=float(inside.mean()), resp_rms=resp, resp_flag=int(nflag), n_resp=int(len(errs)))


# ------------------------------------------------------------------ neural networks
def torch_mods():
    import torch, torch.nn as nn
    return torch, nn


def mlp(nn, din, hidden, dout, leaky=0.2):
    L = []; d = din
    for h in hidden:
        L += [nn.Linear(d, h), nn.LeakyReLU(leaky)]; d = h
    L.append(nn.Linear(d, dout)); return nn.Sequential(*L)


def fit_net(torch, nn, net, loss_fn, Xtr, Ytr, Xva, Yva, lr, wd, seed, max_epochs, patience=40, decay=0.97, batch=16):
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay)
    Xtr = torch.tensor(Xtr, dtype=torch.float32); Ytr = torch.tensor(Ytr, dtype=torch.float32)
    Xva = torch.tensor(Xva, dtype=torch.float32); Yva = torch.tensor(Yva, dtype=torch.float32)
    rng = np.random.RandomState(seed); best = (float("inf"), None); bad = 0
    for ep in range(max_epochs):
        net.train(); perm = rng.permutation(len(Xtr))
        for k in range(0, len(perm), batch):
            b = perm[k:k+batch]; opt.zero_grad()
            loss_fn(net(Xtr[b]), Xtr[b], Ytr[b]).backward(); opt.step()
        sched.step(); net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(Xva), Xva, Yva))
        if v < best[0] - 1e-9:
            best = (v, {k: x.clone() for k, x in net.state_dict().items()}); bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best[1]); net.eval(); return net, best[0], ep + 1


def run_map(model, name, methods, seeds, smoke, p5):
    spec, G, T, idx, ndrop = make_data(model, name)
    params = spec["params"]; d = len(params)
    Ttr_all = tt(T)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    tm, ts = Ttr_all[tr].mean(0), Ttr_all[tr].std(0); gm, gs = G[tr].mean(0), G[tr].std(0)
    Zt = (Ttr_all - tm)/ts; Zg = (G - gm)/gs
    results = {m: [] for m in methods}
    epochs = 20 if smoke else 600
    for sd in seeds:
        t0 = time.time()
        if "naive" in methods or "tandem" in methods:
            torch, nn = torch_mods()
        if "naive" in methods:
            torch.manual_seed(sd)
            net = mlp(nn, 4, HID_INV, d)
            mse = nn.MSELoss()
            net, v, ne = fit_net(torch, nn, net, lambda p, x, y: mse(p, y), Zt[tr], Zg[tr], Zt[va], Zg[va], 1e-3, 1e-5, sd, epochs)
            with torch.no_grad():
                Gp = net(torch.tensor(Zt[te], dtype=torch.float32)).numpy()*gs + gm
            r = evaluate_pred(model, name, spec, Gp, G[te], T[te]); r.update(seed=sd, epochs=ne); results["naive"].append(r)
        if "tandem" in methods:
            torch.manual_seed(sd)
            fwd = mlp(nn, d, (FWD_WIDTH,)*FWD_DEPTH, 4); mse = nn.MSELoss()
            fwd, vf, nf = fit_net(torch, nn, fwd, lambda p, x, y: mse(p, y), Zg[tr], Zt[tr], Zg[va], Zt[va], 5e-3, 1e-4, sd, epochs)
            for q in fwd.parameters():
                q.requires_grad_(False)
            torch.manual_seed(sd + 1)
            net = mlp(nn, 4, HID_INV, d)
            net, v, ne = fit_net(torch, nn, net, lambda p, x, y: mse(fwd(p), y), Zt[tr], Zt[tr], Zt[va], Zt[va], 1e-3, 1e-5, sd + 1, epochs)
            with torch.no_grad():
                Gp = net(torch.tensor(Zt[te], dtype=torch.float32)).numpy()*gs + gm
            r = evaluate_pred(model, name, spec, Gp, G[te], T[te]); r.update(seed=sd, epochs=ne, fwd_val=vf); results["tandem"].append(r)
        if "rf" in methods:
            from sklearn.ensemble import RandomForestRegressor
            bp = dict(p5.get("inverse_rf_norm", {})) if p5 else {}
            if not bp:
                bp = dict(n_estimators=300, min_samples_leaf=1, max_features=None)
            if smoke:
                bp["n_estimators"] = 30
            rf = RandomForestRegressor(random_state=sd, n_jobs=-1, **bp).fit(Zt[tr], Zg[tr])
            Gp = rf.predict(Zt[te])*gs + gm
            r = evaluate_pred(model, name, spec, Gp, G[te], T[te]); r.update(seed=sd, params=bp); results["rf"].append(r)
        if "xgb" in methods:
            try:
                from xgboost import XGBRegressor
                from sklearn.multioutput import MultiOutputRegressor
            except Exception as e:
                print("   xgboost unavailable:", e); methods = [m for m in methods if m != "xgb"]; continue
            bp = dict(p5.get("inverse_xgb_norm", {})) if p5 else {}
            if not bp:
                bp = dict(n_estimators=300, max_depth=6, learning_rate=0.05)
            if smoke:
                bp["n_estimators"] = 30
            xg = MultiOutputRegressor(XGBRegressor(random_state=sd, n_jobs=-1, **bp)).fit(Zt[tr], Zg[tr])
            Gp = xg.predict(Zt[te])*gs + gm
            r = evaluate_pred(model, name, spec, Gp, G[te], T[te]); r.update(seed=sd, params=bp); results["xgb"].append(r)
        print(f"   {name} seed {sd} done ({time.time()-t0:.0f} s)")
    return dict(params=list(params), n_dropped=ndrop, n_train=len(tr), n_test=len(te), results=results)


def load_p5():
    for cand in (os.path.join("results", "phase5_results.json"), "phase5_results.json"):
        if os.path.isfile(cand):
            d = json.load(open(cand)); out = {}
            for k in ("inverse_rf_norm", "inverse_xgb_norm"):
                if k in d and "best" in d[k]:
                    out[k] = {a.replace("estimator__", ""): b for a, b in d[k]["best"].items()}
            return out
    return {}


def report(allres, smoke):
    L = []
    L.append("P9 -- generality across response maps (model-level response error, NOT FDTD)")
    L.append("Settings: Appendix B of the paper; nothing tuned. Test split 805/172/173; targets: lambda, FSR, log10 Q, log10 IL (or depth).")
    L.append("=" * 112)
    summ = {}
    for name, blk in allres.items():
        params = blk["params"]
        L.append(f"\n{name}  params {params}   rows dropped by the quality gate: {blk['n_dropped']}   test rows {blk['n_test']}")
        L.append(f"{'method':8s} {'RMS-RGE %':>14s} {'non-Lc RGE %':>14s} {'n params R2>=0.8':>17s} {'buildable %':>12s} {'response RMS %':>15s}   mean R^2 per parameter")
        for m, rs in blk["results"].items():
            if not rs:
                continue
            f = lambda k: np.array([r[k] for r in rs], float)
            r2 = np.mean([r["r2"] for r in rs], axis=0)
            L.append(f"{m:8s} {f('rge').mean():8.2f}+-{f('rge').std():4.2f} {f('rge_nonlc').mean():8.2f}+-{f('rge_nonlc').std():4.2f} "
                     f"{f('n_rec').mean():17.2f} {100*f('buildable').mean():12.1f} {np.nanmean(f('resp_rms')):15.2f}   "
                     + " ".join(f"{p}:{x:+.2f}" for p, x in zip(params, r2)))
            summ[(name, m)] = dict(rge=f("rge").mean(), n_rec=f("n_rec").mean(), r2=r2, build=f("buildable").mean(),
                                   resp=np.nanmean(f("resp_rms")), params=params)
    L.append("\nPRE-REGISTERED PREDICTIONS (p9_predictions.md)")
    def get(n, m, k):
        return summ.get((n, m), {}).get(k)
    def verdict(c, ok):
        return "PASS" if ok else "FAIL"
    ctl = get("addrop", "naive", "rge")
    if ctl is not None:
        r2R = get("addrop", "naive", "r2")[0]
        L.append(f"C0 control: add-drop naive RMS-RGE {ctl:.2f} % (14-17 %), R^2(R) {r2R:.3f} (>=0.95)  -> {verdict('C0', 14 <= ctl <= 17 and r2R >= 0.95)}")
    for n in allres:
        k = get(n, "naive", "n_rec")
        if k is not None:
            r2 = get(n, "naive", "r2"); ps = get(n, "naive", "params")
            rec = [p for p, x in zip(ps, r2) if x >= 0.8]
            L.append(f"C1 {n}: mean #parameters with R^2>=0.8 = {k:.2f}; recovered: {rec or 'none'} (<=1 and it is R) -> {verdict('C1', k <= 1.0 and all(p == 'R' for p in rec))}")
            L.append(f"C2 {n}: naive response RMS {get(n,'naive','resp'):.2f} % (<20 %) -> {verdict('C2', get(n,'naive','resp') < 20)}")
        b = get(n, "tandem", "build")
        if b is not None:
            L.append(f"C5 {n}: tandem buildable {100*b:.1f} % (<90 %) -> {verdict('C5', b < 0.9)}")
    if get("addrop_asym", "naive", "r2") is not None:
        r2 = get("addrop_asym", "naive", "r2"); ps = get("addrop_asym", "naive", "params")
        sel = [x for p, x in zip(ps, r2) if p in ("g1", "Lc1", "g2", "Lc2")]
        L.append(f"C3 addrop_asym: R^2 of g1,Lc1,g2,Lc2 = {[round(float(x),2) for x in sel]} (all <0.5) -> {verdict('C3', all(x < 0.5 for x in sel))}")
        if ctl is not None:
            a = get("addrop_asym", "naive", "rge")
            L.append(f"C4 addrop_asym RMS-RGE {a:.2f} % vs control {ctl:.2f} % (higher) -> {verdict('C4', a > ctl)}")
    if get("allpass", "naive", "rge") is not None and ctl is not None:
        a = get("allpass", "naive", "rge")
        L.append(f"C6 all-pass RMS-RGE {a:.2f} % vs control {ctl:.2f} % (within 3 points) -> {verdict('C6', abs(a - ctl) <= 3)}")
    if smoke:
        L.append("\n(SMOKE RUN: 1 seed, 20 epochs, small trees -- the verdicts above are meaningless; only 'it ran' matters.)")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--maps", default="addrop,allpass,addrop_asym")
    ap.add_argument("--methods", default="naive,tandem,rf,xgb")
    ap.add_argument("--ordered", action="store_true", help="post-hoc diagnostic: canonical coupler order (asymmetric map only)")
    a = ap.parse_args()
    ORDERED = a.ordered
    model = R9.Model(R9.load_fits()); p5 = load_p5()
    seeds = SEEDS[:1] if a.smoke else SEEDS
    methods = a.methods.split(","); allres = {}
    for name in a.maps.split(","):
        print(f"[p9] {name}"); allres[name] = run_map(model, name, list(methods), seeds, a.smoke, p5)
    os.makedirs("results", exist_ok=True)
    tag = ("p9_smoke" if a.smoke else "p9") + ("_ordered" if a.ordered else "")
    json.dump(allres, open(os.path.join("results", tag + "_results.json"), "w"), indent=1, default=float)
    txt = report(allres, a.smoke); print(txt)
    open(os.path.join("results", tag + "_report.txt"), "w").write(txt)
