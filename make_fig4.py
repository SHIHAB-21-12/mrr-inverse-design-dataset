"""
Figure 4 (information bound), regenerated from the saved results so it can be reproduced.
Left : estimate against response-bank size, best k annotated, 15 % threshold and the best
       model's 15.506 % marked.
Right: per-parameter error of the best model against the bound estimator's per-parameter error.
Run from the project folder:  py -3 make_fig4.py
Reads  results/floor_convergence.json, results/floor.json
Writes fig/fig4_information_floor.pdf (and .png for a quick look)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# best model (naive network): per-parameter terms of its RMS-RGE, from results/phase4_report.txt
BEST_MODEL = {"R": 2.841, "w": 5.321, "g": 14.144, "Lc": 26.930}
BEST_AGG = 15.506
THRESH = 15.0

conv = json.load(open("results/floor_convergence.json"))["rows"]
fl = json.load(open("results/floor.json"))
N = np.array([r[0] for r in conv], float); F = np.array([r[1] for r in conv]); K = [r[2] for r in conv]
per = dict(zip(("R", "w", "g", "Lc"), fl["per"]))

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})

a = ax[0]
a.plot(N, F, "o-", color="#1f4e79", lw=1.4, ms=4, label="bound estimate (upper bound)")
for n, f, k in zip(N, F, K):
    a.annotate(f"k={k}", (n, f), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5)
a.axhline(THRESH, color="#b03a2e", ls="--", lw=1.0)
a.text(N[0], THRESH + 0.06, "O3 threshold, 15 %", color="#b03a2e", fontsize=7, va="bottom")
a.axhline(BEST_AGG, color="#555555", ls=":", lw=1.0)
a.text(N[0], BEST_AGG - 0.06, "best model, 15.506 %", color="#555555", fontsize=7, va="top", ha="left")
a.set_xscale("log"); a.set_xlabel("response-bank size $N$"); a.set_ylabel("RMS-RGE (%)")
a.set_ylim(min(F.min(), 14.0) - 0.3, max(F.max(), BEST_AGG) + 0.7)
a.set_title("(a) estimate against bank size", fontsize=8)

b = ax[1]
keys = ["R", "w", "g", "Lc"]; x = np.arange(len(keys)); wdt = 0.36
b.bar(x - wdt / 2, [BEST_MODEL[k] for k in keys], wdt, color="#7f8c8d", label="best model")
b.bar(x + wdt / 2, [per[k] for k in keys], wdt, color="#1f4e79", label=f"bound estimator ($k={fl['k']}$)")
for xi, k in zip(x, keys):
    b.text(xi - wdt / 2, BEST_MODEL[k] + 0.4, f"{BEST_MODEL[k]:.1f}", ha="center", fontsize=6.5)
    b.text(xi + wdt / 2, per[k] + 0.4, f"{per[k]:.1f}", ha="center", fontsize=6.5)
b.set_xticks(x); b.set_xticklabels(["$R$", "$w$", "$g$", "$L_c$"])
b.set_ylabel("per-parameter error (%)"); b.legend(frameon=False, fontsize=7, loc="upper left")
b.set_ylim(0, max(BEST_MODEL.values()) * 1.15)
b.set_title("(b) per-parameter, not individual bounds", fontsize=8)

fig.tight_layout()
os.makedirs("fig", exist_ok=True)
fig.savefig("fig/fig4_information_floor.pdf")
fig.savefig("fig/fig4_information_floor.png", dpi=200)
print("wrote fig/fig4_information_floor.pdf and .png")
