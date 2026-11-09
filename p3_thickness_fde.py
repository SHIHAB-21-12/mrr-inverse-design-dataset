"""
P3 thickness test, step 1 (needs Lumerical): effective index of the straight guide versus
width, silicon thickness and wavelength, using the project's own mode solve (mrr_template._neff_at).
  py -3 p3_thickness_fde.py --check     # 9 solves, ~1-3 min: validates the setup against the ring model
  py -3 p3_thickness_fde.py             # 45 solves (5 widths x 3 thicknesses x 3 wavelengths), resumable
Writes results/p3_thickness_neff.json ("w|t|lam" -> n_eff). mrr_template.py is not modified.
"""
import argparse, json, os, sys, time, inspect
import numpy as np
sys.path.insert(0, os.getcwd())
import mrr_template as m

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
args = ap.parse_args()

resdir = "results" if os.path.isdir("results") else "."
F = os.path.join(resdir, "p3_check_neff.json" if args.check else "p3_thickness_neff.json")
W = [425, 450, 475] if args.check else [400, 425, 450, 475, 500]      # nm
T = [220] if args.check else [215, 220, 225]                           # nm
LAM = [1540, 1550, 1560]                                               # nm

print("[p3] _neff_at signature:", inspect.signature(m._neff_at), "| H_CORE =", m.H_CORE)
if list(inspect.signature(m._neff_at).parameters)[:3] != ["fdtd", "w", "lam"]:
    sys.exit("[p3] STOP: _neff_at has a different signature than expected; paste this line to Claude.")

data = json.load(open(F)) if os.path.isfile(F) else {}
lumapi = m.load_lumapi()
H0 = m.H_CORE
todo = [(w, t, l) for w in W for t in T for l in LAM if f"{w}|{t}|{l}" not in data]
print(f"[p3] {len(data)} done, {len(todo)} to solve")
t0 = time.time()
try:
    with lumapi.FDTD(hide=True) as fdtd:
        for i, (w, t, l) in enumerate(todo, 1):
            m.H_CORE = t * 1e-9
            try:
                n = m._neff_at(fdtd, w * 1e-9, l * 1e-9)
            except Exception as exc:                          # noqa: BLE001
                print(f"[p3] FAILED w={w} t={t} lam={l}: {exc}  -- re-run to retry")
                continue
            data[f"{w}|{t}|{l}"] = n
            json.dump(data, open(F, "w"), indent=1)
            print(f"[p3] {i}/{len(todo)}  w={w} t={t} lam={l}:  n_eff = {n:.6f}   ({time.time()-t0:.0f} s)")
finally:
    m.H_CORE = H0

if args.check:
    need = [f"{w}|220|{l}" for w in W for l in LAM]
    if not all(k in data for k in need):
        sys.exit("[p3] check incomplete; re-run the same command")
    n = lambda w, l: data[f"{w}|220|{l}"]
    n0 = n(450, 1550)
    dn_dl = (n(450, 1560) - n(450, 1540)) / 20.0
    ng = n0 - 1550.0 * dn_dl
    Sw_fde = 1550.0 * (n(475, 1550) - n(425, 1550)) / 50.0 / ng          # nm resonance per nm width
    r = m._make_responder()
    a = r(10, 450 - 0.35, 250, 1)
    b = r(10, 450 + 0.35, 250, 1)
    fsr = a["FSR_nm"]
    d = (b["lambda_res_nm"] - a["lambda_res_nm"] + fsr / 2) % fsr - fsr / 2
    Sw_model = d / 0.7
    print("\n[p3] CHECK  (w=450, t=220, lam=1550)")
    print(f"   n_eff {n0:.5f}   group index {ng:.4f}")
    print(f"   resonance shift per nm of width:  FDE {Sw_fde:.4f}   ring model {Sw_model:.4f}   ratio {Sw_fde/Sw_model:.3f}")
    print("   OK if n_eff is about 2.3-2.5, group index about 4.0-4.7 and the ratio is within 0.85-1.15.")
