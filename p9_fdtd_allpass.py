r"""
P9 step 3 -- FDTD closed loop for the ALL-PASS map (independent check of the model-level result).

  py -3 p9_fdtd_allpass.py --prepare --method naive      # trains the network (seed 42), writes results\p9_allpass_preds_naive.json
  py -3 p9_fdtd_allpass.py --dry     --method naive      # lists the 20 designs and which are inside the box; runs NO FDTD
  py -3 p9_fdtd_allpass.py --run     --method naive      # 3-D FDTD coupler for each design (resumable; one design at a time)
  py -3 p9_fdtd_allpass.py --report  --method naive      # errors, bootstrap interval, verdicts on C7-C10 (p9_predictions_fdtd.md)

Uses mrr_template._coupler_kappa2_fdtd (the same call as run_closedloop). Method 'rf' is also supported.
Set P9_FAKE_FDTD=1 to test the plumbing without Lumerical (uses the fitted kappa^2 with 4.5% log-normal noise; never for results).
"""
import argparse, json, os, sys, time
import numpy as np
import p9_responders as R9
import p9_train as T9

NT = 20
RES = "results"


def paths(method):
    os.makedirs(RES, exist_ok=True)
    return (os.path.join(RES, f"p9_allpass_preds_{method}.json"),
            os.path.join(RES, f"p9_allpass_closedloop_{method}.json"))


def prepare(method):
    model = R9.Model(R9.load_fits())
    spec, G, T, idx, _ = T9.make_data(model, "allpass")
    tr, va, te = idx["train"], idx["val"], idx["test"]
    Tt = T9.tt(T); tm, ts = Tt[tr].mean(0), Tt[tr].std(0); gm, gs = G[tr].mean(0), G[tr].std(0)
    Zt = (Tt - tm)/ts; Zg = (G - gm)/gs; d = G.shape[1]
    sd = 42
    if method == "naive":
        torch, nn = T9.torch_mods(); torch.manual_seed(sd)
        net = T9.mlp(nn, 4, T9.HID_INV, d); mse = nn.MSELoss()
        net, v, ne = T9.fit_net(torch, nn, net, lambda p, x, y: mse(p, y), Zt[tr], Zg[tr], Zt[va], Zg[va], 1e-3, 1e-5, sd, 600)
        with torch.no_grad():
            Gp = net(torch.tensor(Zt[te], dtype=torch.float32)).numpy()*gs + gm
    elif method == "rf":
        from sklearn.ensemble import RandomForestRegressor
        bp = T9.load_p5().get("inverse_rf_norm") or dict(n_estimators=300, min_samples_leaf=1, max_features=None)
        Gp = RandomForestRegressor(random_state=sd, n_jobs=-1, **bp).fit(Zt[tr], Zg[tr]).predict(Zt[te])*gs + gm
    else:
        sys.exit("method must be naive or rf")
    sel = np.linspace(0, len(te) - 1, NT).astype(int)
    lo, hi = R9.box(spec["params"])
    out = []
    for j in sel:
        gp = Gp[j]; req = T[te][j]
        inside = bool(np.all((gp >= lo) & (gp <= hi)))
        mod = None
        if inside:
            o = model.allpass(*gp)
            if not o["flag"] and all(np.isfinite(o[t]) and o[t] > 0 for t in spec["targets"]):
                mod = [float(o[t]) for t in spec["targets"]]
        out.append(dict(test_pos=int(j), requested=[float(x) for x in req], true_geom=[float(x) for x in G[te][j]],
                        pred=[float(x) for x in gp], inside=inside, model_resp=mod))
    json.dump(dict(method=method, seed=sd, targets=list(spec["targets"]), items=out), open(paths(method)[0], "w"), indent=1)
    print(f"[prepare] wrote {paths(method)[0]}: {len(out)} designs, {sum(o['inside'] for o in out)} inside the box")


def dry(method):
    d = json.load(open(paths(method)[0]))
    print(f"{len(d['items'])} designs ({method}); inside the design box: {sum(o['inside'] for o in d['items'])}")
    for k, o in enumerate(d["items"], 1):
        R, w, g, Lc = o["pred"]
        print(f"  {k:2d} test row {o['test_pos']:3d}  R={R:.3f} w={w:.1f} g={g:.1f} Lc={Lc:.3f}  {'OK' if o['inside'] else 'OUTSIDE BOX (unrealisable)'}")


def run(method):
    d = json.load(open(paths(method)[0])); ck = paths(method)[1]
    done = json.load(open(ck)) if os.path.isfile(ck) else {}
    model = R9.Model(R9.load_fits())
    fake = bool(os.environ.get("P9_FAKE_FDTD"))
    if not fake:
        import mrr_template as m
    print(f"[p9-fdtd] {method}: {len(done)} of {len(d['items'])} already done")
    rng = np.random.RandomState(0)
    for c, o in enumerate(d["items"], 1):
        key = str(o["test_pos"])
        if key in done:
            continue
        if not o["inside"]:
            done[key] = dict(status="unrealisable", geom=o["pred"]); json.dump(done, open(ck, "w"), indent=1)
            print(f"[p9-fdtd] {c}/{len(d['items'])}: UNREALISABLE (outside the design box)"); continue
        R, w, g, Lc = o["pred"]
        print(f"[p9-fdtd] {c}/{len(d['items'])}: R={R:.3f}um w={w:.1f}nm g={g:.1f}nm Lc={Lc:.3f}um ...", flush=True)
        t0 = time.time()
        try:
            if fake:
                k2 = model.kappa2(R, w, g, Lc)*float(np.exp(0.045*rng.randn())); t2 = 1 - k2
            else:
                k2, t2 = m._coupler_kappa2_fdtd(R, w, g, Lc)
        except Exception as exc:
            print(f"[p9-fdtd]   FAILED: {exc} -- run again to retry"); time.sleep(20); continue
        r = model.allpass_given_k2(R, w, g, Lc, float(k2))
        done[key] = dict(status="ok", kappa2=float(k2), t2=float(t2), kappa2_model=model.kappa2(R, w, g, Lc),
                         geom=o["pred"], resim={t: float(r[t]) for t in d["targets"]}, flag=r["flag"],
                         minutes=(time.time() - t0)/60)
        json.dump(done, open(ck, "w"), indent=1)
        print(f"[p9-fdtd]   kappa^2 {k2:.6f} (model {done[key]['kappa2_model']:.6f})  t^2 {t2:.6f}  sum {k2+t2:.4f}  ({(time.time()-t0)/60:.1f} min)", flush=True)


def agg(E):
    E = np.asarray(E); return 100*np.sqrt(np.mean(np.mean(E**2, axis=0)))   # sqrt of the mean of the four per-response mean-square errors


def report(method):
    d = json.load(open(paths(method)[0])); done = json.load(open(paths(method)[1]))
    tg = d["targets"]; E = []; M = []; n_un = 0; kr = []; n_flag = 0
    for o in d["items"]:
        r = done.get(str(o["test_pos"]))
        if r is None:
            continue
        if r["status"] == "unrealisable":
            n_un += 1; continue
        if r.get("flag") or not all(np.isfinite(r["resim"][t]) and r["resim"][t] > 0 for t in tg):
            n_flag += 1; continue
        req = np.array(o["requested"]); sim = np.array([r["resim"][t] for t in tg])
        E.append((sim - req)/req)
        if o["model_resp"] is not None:
            M.append((np.array(o["model_resp"]) - req)/req)
        kr.append(r["kappa2"]/r["kappa2_model"])
    E = np.array(E); n = len(E)
    L = [f"P9 step 3 -- ALL-PASS closed loop with a fresh 3-D FDTD coupler run ({method})", "=" * 90,
         f"designs requested {len(d['items'])}; simulated and evaluable {n}; unrealisable (outside the box) {n_un}; half-maximum/other flags {n_flag}"]
    if n == 0:
        print("\n".join(L)); return
    L.append(f"{'response':14s} {'RMS %':>8s} {'mean |err| %':>13s} {'max |err| %':>12s}")
    for j, t in enumerate(tg):
        L.append(f"{t:14s} {100*np.sqrt(np.mean(E[:, j]**2)):8.3f} {100*np.mean(np.abs(E[:, j])):13.3f} {100*np.max(np.abs(E[:, j])):12.3f}")
    a = agg(E)
    rng = np.random.RandomState(20261103); bs = [agg(E[rng.randint(0, n, n)]) for _ in range(10000)]
    L.append(f"AGGREGATE closed-loop error (FDTD coupler): {a:.3f} %   bootstrap 95% interval [{np.percentile(bs,2.5):.2f}, {np.percentile(bs,97.5):.2f}] (over designs, B=10000)")
    am = agg(M) if len(M) else float("nan")
    L.append(f"model-level error on the same designs (fitted kappa^2, replay): {am:.3f} %")
    L.append(f"FDTD kappa^2 / fitted-model kappa^2: median {np.median(kr):.3f}, range {min(kr):.3f}-{max(kr):.3f}, s.d. of log {np.std(np.log(kr)):.3f}")
    worst = tg[int(np.argmax([np.sqrt(np.mean(E[:, j]**2)) for j in range(4)]))]
    L.append("")
    L.append("PRE-REGISTERED PREDICTIONS (p9_predictions_fdtd.md)")
    L.append(f"C7 aggregate {a:.2f} % (<20 %) -> {'PASS' if a < 20 else 'FAIL'}")
    L.append(f"C8 realisable {len(d['items']) - n_un} of {len(d['items'])} (>=18) -> {'PASS' if len(d['items']) - n_un >= 18 else 'FAIL'}")
    L.append(f"C9 largest per-response error: {worst} (Q_L or depth) -> {'PASS' if worst in ('Q_L', 'depth') else 'FAIL'}")
    L.append(f"C10 FDTD {a:.2f} % vs model-level {am:.2f} % (ratio <= 2) -> {'PASS' if a <= 2*am else 'FAIL'}")
    txt = "\n".join(L); print(txt); open(os.path.join(RES, f"p9_allpass_closedloop_{method}_report.txt"), "w").write(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("prepare", "dry", "run", "report"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--method", default="naive"); a = ap.parse_args()
    if a.prepare: prepare(a.method)
    if a.dry: dry(a.method)
    if a.run: run(a.method)
    if a.report: report(a.method)
    if not (a.prepare or a.dry or a.run or a.report):
        ap.print_help()
