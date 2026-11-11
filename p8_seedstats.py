"""
P8 (weakness 5, no retraining, seconds): turn the five-seed "gap > pooled sd" rule into proper tests.
Reads results/seedcurve.json (per seed x size x method, written by --seedcurve) and, for every pair of methods at every
training size, reports the difference of means, a Welch t-test and an exact permutation test over the seeds (n = 5 per
group gives 252 distinct labellings, so the smallest attainable two-sided p is 2/252 = 0.0079), with the Holm correction
over the pairs within one (direction, size) family. It also states the paper's old rule (gap > pooled sd, divisor n) next
to the test, so the two can be compared. Seed variation only: it says nothing about variation over test rows.
Run:  py -3 p8_seedstats.py     Writes results/p8_seedstats_report.txt
"""
import itertools, json, os
import numpy as np
from scipy import stats

resdir = "results" if os.path.isdir("results") else "."
d = json.load(open(os.path.join(resdir, "seedcurve.json")))
seeds = sorted({int(k.split("|")[0]) for k in d})
sizes = sorted({int(k.split("|")[1]) for k in d})
FWD = ["DNN forward", "RF forward", "XGB forward"]
INV = ["naive DNN inverse", "Tandem inverse", "RF inverse", "XGB inverse"]


def vals(method, n):
    return np.array([d[f"{s}|{n}"][method] for s in seeds if f"{s}|{n}" in d and method in d[f"{s}|{n}"]], float)


def perm_p(a, b):
    x = np.concatenate([a, b]); na = len(a); obs = abs(a.mean() - b.mean()); cnt = tot = 0
    for idx in itertools.combinations(range(len(x)), na):
        m = np.zeros(len(x), bool); m[list(idx)] = True
        tot += 1
        cnt += abs(x[m].mean() - x[~m].mean()) >= obs - 1e-12
    return cnt / tot


L = ["P8 -- seed-level tests behind the sample-efficiency comparisons", "=" * 104,
     f"seeds {seeds}; training sizes {sizes}. Welch t and exact permutation over seeds; Holm over the pairs in each block.",
     "'old rule' = the paper's earlier criterion, gap between two means > sqrt(sd_a^2 + sd_b^2) with divisor n.", ""]
for title, keys in (("FORWARD (aggregate test MAPE, %)", FWD), ("INVERSE (RMS-RGE, %)", INV)):
    L += [title, "-" * 104]
    for n in sizes:
        rows = []
        for a, b in itertools.combinations(keys, 2):
            va, vb = vals(a, n), vals(b, n)
            if len(va) < 3 or len(vb) < 3:
                continue
            gap = va.mean() - vb.mean()
            t = stats.ttest_ind(va, vb, equal_var=False)
            pp = perm_p(va, vb)
            pooled = float(np.sqrt(va.std() ** 2 + vb.std() ** 2))
            rows.append([a, b, gap, float(t.pvalue), pp, abs(gap) > pooled])
        # Holm on the permutation p-values (the exact ones), within this block
        order = sorted(range(len(rows)), key=lambda i: rows[i][4])
        m = len(rows); run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (m - rank) * rows[i][4])); rows[i].append(run)
        means = "  ".join(f"{k.split()[0]} {vals(k, n).mean():.3f}+-{vals(k, n).std():.3f}" for k in keys)
        L.append(f"  n = {n}:  {means}")
        for a, b, gap, pw, pp, old, ph in rows:
            L.append(f"      {a:18s} - {b:18s} diff {gap:+8.3f}   Welch p {pw:8.4f}   perm p {pp:6.4f}   Holm {ph:6.4f}"
                     f"   old rule {'sep' if old else 'no '}   test {'SEPARATED' if ph < 0.05 else 'not separated'}")
    L.append("")
L += ["Reading: with five seeds per group the permutation test cannot give p below 0.0079, and after Holm over six pairs",
      "the smallest attainable p is 0.047, so a pair can be 'separated' only if all five seeds of one method beat all five",
      "of the other. That is a property of n = 5, not of the methods; it is the sample-size limitation the paper states."]
txt = "\n".join(L)
open(os.path.join(resdir, "p8_seedstats_report.txt"), "w").write(txt + "\n")
print(txt)
