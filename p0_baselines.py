"""
P0 baselines -- can plain optimisation do what the inverse networks do?

Same 20 test targets as the closed-loop table (evenly spaced through the test set).
For each target the 4-parameter geometry (R, w, g, Lc) is found by a global optimiser that
minimises the RMS relative error of the four responses (lambda_res, FSR, Q_L, IL) of the
ANALYTIC ring model (m._make_responder). No training data, no network.

Stage 1 (default, minutes, NO Lumerical):   py -3 p0_baselines.py --method de
Stage 2 (hours, needs Lumerical):           py -3 p0_baselines.py --method de --fdtd
    Re-simulates each optimiser design with the production 3-D FDTD coupler and the analytic
    ring, exactly like run_closedloop and h2_tandem_closedloop.py, and compares with the
    networks on the same targets.
--method cma uses CMA-ES with restarts (needs:  py -3 -m pip install cma ).
Resumable. Writes results/p0_<method>.json, results/p0_<method>_closedloop.json,
results/p0_<method>_report.txt. mrr_template.py is not modified.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.getcwd())
import mrr_template as m

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["de", "cma"], default="de")
ap.add_argument("--fdtd", action="store_true")
ap.add_argument("--pop", type=int, default=30)          # DE population = 4 * pop
ap.add_argument("--maxiter", type=int, default=400)
ap.add_argument("--restarts", type=int, default=8)      # CMA-ES restarts
args = ap.parse_args()

N_TARGETS = 20
BOUNDS = [(5.0, 15.0), (400.0, 500.0), (150.0, 350.0), (0.0, 3.0)]   # R um, w nm, g nm, Lc um
STOP = 1e-6                    # stop when RMS relative error < 1e-4 %
BOOT, BOOT_SEED = 10000, 20261103
resdir = "results" if os.path.isdir("results") else "."
F_AN = os.path.join(resdir, f"p0_{args.method}.json")
F_CL = os.path.join(resdir, f"p0_{args.method}_closedloop.json")

_, Rr_te = m._load_split_raw("test")
respond = m._make_responder()
idx = np.linspace(0, len(Rr_te) - 1, N_TARGETS).astype(int)


def objective(x, tgt):
    try:
        r = respond(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
        v = np.array([r[t] for t in m.TARGETS], dtype=float)
        if not np.all(np.isfinite(v)):
            return 1e3
        return float(np.sqrt(np.mean(((v - tgt) / tgt) ** 2)))
    except Exception:                                      # noqa: BLE001
        return 1e3


def run_de(tgt, seed):
    from scipy.optimize import differential_evolution
    cnt = [0]

    def f(x):
        cnt[0] += 1
        return objective(x, tgt)

    def cb(xk, convergence=None):
        return f(xk) < STOP

    res = differential_evolution(f, BOUNDS, seed=seed, popsize=args.pop, maxiter=args.maxiter,
                                 tol=0, atol=0, mutation=(0.5, 1.0), recombination=0.8,
                                 polish=False, init="latinhypercube", callback=cb)
    return [float(v) for v in res.x], float(res.fun), cnt[0]


def run_cma(tgt, seed):
    import cma
    lo = np.array([b[0] for b in BOUNDS])
    span = np.array([b[1] - b[0] for b in BOUNDS])
    best, bx, evals = 1e9, None, 0
    for k in range(args.restarts):
        rs = seed * 100 + k
        z0 = np.random.default_rng(rs).random(4)
        es = cma.CMAEvolutionStrategy(z0, 0.3, {"bounds": [0, 1], "seed": rs + 1, "verbose": -9,
                                                "maxiter": 300, "tolfun": 1e-12})
        while not es.stop():
            Z = es.ask()
            F = [objective(lo + span * np.clip(z, 0, 1), tgt) for z in Z]
            es.tell(Z, F)
            evals += len(Z)
            if min(F) < best:
                best, bx = min(F), lo + span * np.clip(Z[int(np.argmin(F))], 0, 1)
        if best < STOP:
            break
    return [float(v) for v in bx], float(best), evals


# ---------------- stage 1: optimise on the analytic model ----------------
an = json.load(open(F_AN)) if os.path.isfile(F_AN) else {}
print(f"[p0] method {args.method}: {len(an)} of {N_TARGETS} targets already optimised")
for c, i in enumerate(idx, 1):
    key = str(int(i))
    if key in an:
        continue
    tgt = np.array(Rr_te[int(i)], dtype=float)
    t0 = time.time()
    x, loss, ev = (run_de if args.method == "de" else run_cma)(tgt, 20261109 + int(i))
    an[key] = dict(x=x, loss=loss, evals=ev, seconds=time.time() - t0)
    json.dump(an, open(F_AN, "w"), indent=1)
    print(f"[p0] {c}/{N_TARGETS} target {key}: RMS rel. error {100*loss:.5f} %  "
          f"({ev} evals, {an[key]['seconds']:.0f} s)  R={x[0]:.3f} w={x[1]:.1f} g={x[2]:.1f} Lc={x[3]:.3f}")
L_an = np.array([an[str(int(i))]["loss"] for i in idx])
print(f"[p0] stage 1 done: {int(np.sum(L_an < 1e-4))} of {N_TARGETS} reached < 0.01 % in the analytic model; "
      f"median {100*np.median(L_an):.5f} %, worst {100*L_an.max():.4f} %")
if not args.fdtd:
    print("[p0] stage 1 only. Run again with --fdtd for the 3-D FDTD closed loop (hours).")
    sys.exit(0)

# ---------------- stage 2: 3-D FDTD closed loop ----------------
cl = json.load(open(F_CL)) if os.path.isfile(F_CL) else {}
print(f"[p0] closed loop: {len(cl)} of {N_TARGETS} already done")
for c, i in enumerate(idx, 1):
    key = str(int(i))
    if key in cl:
        continue
    R, w, g, Lc = an[key]["x"]
    print(f"[p0] {c}/{N_TARGETS}: target {key} R={R:.3f} w={w:.1f} g={g:.1f} Lc={Lc:.3f} ...")
    t0 = time.time()
    try:
        k2, t2 = m._coupler_kappa2_fdtd(R, w, g, Lc)
    except Exception as exc:                               # noqa: BLE001
        print(f"[p0]   FAILED: {exc}  -- re-run to retry")
        time.sleep(20)
        continue
    resp = m._ring_given_kappa2(R, w, g, Lc, k2)
    cl[key] = dict(status="ok", kappa2=float(k2), t2=float(t2), geom=[R, w, g, Lc],
                   analytic_loss=an[key]["loss"],
                   resim={t: float(resp[t]) for t in m.TARGETS}, minutes=(time.time() - t0) / 60.0)
    json.dump(cl, open(F_CL, "w"), indent=1)
    flag = "" if 0.95 <= k2 + t2 <= 1.05 and t2 <= 1.02 else "   <-- CHECK: energy not conserved"
    print(f"[p0]   kappa^2 {k2:.6f}  t^2 {t2:.6f}  sum {k2+t2:.4f}  ({cl[key]['minutes']:.1f} min){flag}")

# ---------------- report ----------------
def errors(done, keys):
    ks = [k for k in keys if k in done and done[k].get("status") == "ok"]
    E = np.array([[100.0 * (done[k]["resim"][t] - Rr_te[int(k), j]) / Rr_te[int(k), j]
                   for j, t in enumerate(m.TARGETS)] for k in ks]) if ks else np.zeros((0, 4))
    return ks, E


def boot(E):
    rng = np.random.default_rng(BOOT_SEED)
    n = len(E)
    s = [np.sqrt(np.mean(E[rng.integers(0, n, n)] ** 2)) for _ in range(BOOT)]
    return float(np.sqrt(np.mean(E ** 2))), np.percentile(s, [2.5, 97.5])


keys20 = [str(int(i)) for i in idx]
ks, E = errors(cl, keys20)
lines = [f"P0 baseline -- {args.method.upper()} on the analytic model, closed loop in 3-D FDTD", "=" * 84,
         f"{len(ks)} of {N_TARGETS} targets simulated. Relative error in %, over those targets.", ""]
for j, t in enumerate(m.TARGETS):
    lines.append(f"   {t:16s} MAPE {np.mean(np.abs(E[:, j])):8.3f}   RMS {np.sqrt(np.mean(E[:, j]**2)):8.3f}"
                 f"   max {np.max(np.abs(E[:, j])):8.3f}")
o, ci = boot(E)
lines += [f"   overall RMS closed-loop error {o:.3f} %  (bootstrap 95% [{ci[0]:.2f}, {ci[1]:.2f}])", "",
          "Same targets, other methods (overall RMS closed-loop error, %):"]
for name, f in (("bounded tandem (H2)", "h2_bounded.json"),
                ("bounded+weighted tandem (H2)", "h2_bounded_weighted.json"),
                ("naive network (existing table)", "closedloop_naive.json")):
    p = os.path.join(resdir, f)
    if not os.path.isfile(p):
        continue
    try:
        k2s, E2 = errors(json.load(open(p)), keys20)
        if len(E2):
            o2, c2 = boot(E2)
            lines.append(f"   {name:34s} {o2:6.3f}  [{c2[0]:.2f}, {c2[1]:.2f}]   on {len(k2s)} common targets")
    except Exception as exc:                               # noqa: BLE001
        lines.append(f"   {name}: could not be read ({exc})")
lines += ["", "Reading: if the optimiser matches the networks, the closed-loop floor is set by the coupler",
          "model and the physics, not by the inverse method. Intervals are wide with 20 targets."]
txt = "\n".join(lines)
open(os.path.join(resdir, f"p0_{args.method}_report.txt"), "w").write(txt + "\n")
print("\n" + txt)
