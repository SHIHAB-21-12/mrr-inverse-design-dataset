r"""
housekeeping.py -- makes the written record agree with the paper.
Run from D:\Abroad\full guide\Phase1 :   py -3 housekeeping.py
Every file is backed up first (name.bak_housekeeping). Each fix prints PATCHED,
ALREADY DONE, or NOT FOUND (a NOT FOUND changes nothing: send me that line).
"""
import glob, os, py_compile, re, shutil, sys

def read(path):
    raw = open(path, "r", encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), nl

def write(path, text, nl):
    open(path, "w", encoding="utf-8", newline="").write(text.replace("\n", nl))

def backup(path):
    b = path + ".bak_housekeeping"
    if not os.path.exists(b):
        shutil.copy2(path, b)

def patch(path, label, old, new, regex=False, done=None):
    if not os.path.isfile(path):
        print(f"  NOT FOUND (file missing)  {label}: {path}"); return False
    text, nl = read(path)
    if (new in text and not regex) or (done and done in text):
        print(f"  ALREADY DONE              {label}"); return True
    if regex:
        m = re.search(old, text, flags=re.S)
        if not m:
            print(f"  NOT FOUND (text)          {label}"); return False
        backup(path)
        text = text[:m.start()] + new + text[m.end():]
    else:
        if text.count(old) != 1:
            print(f"  NOT FOUND (text, count={text.count(old)})  {label}"); return False
        backup(path)
        text = text.replace(old, new)
    write(path, text, nl)
    print(f"  PATCHED                   {label}")
    return True

R = "results" if os.path.isdir("results") else "."

print("1. results/floor.txt")
patch(os.path.join(R, "floor.txt"), "floor.txt title",
      "INFORMATION FLOOR -- LOWER BOUND ON ACHIEVABLE RMS-RGE",
      "INFORMATION FLOOR -- UPPER-BOUND ESTIMATE (k=5 run, seed 7; the headline 14.626 % at k=7 is in floor_convergence.txt)")
patch(os.path.join(R, "floor.txt"), "floor.txt asymptote line",
      "  learning curve refit with asymptote pinned here:",
      "  learning curve refit with asymptote pinned here (DIAGNOSTIC ONLY; the asymptote is not identifiable, see floor_convergence.txt):")

print("2. results/floor_convergence.txt")
patch(os.path.join(R, "floor_convergence.txt"), "floor_convergence.txt O3 verdict",
"""    It is BELOW 15 %, and an upper bound below a threshold PROVES the
    true floor is below that threshold. So O3's geometric bar is above
    the information floor: O3 is REACHABLE IN PRINCIPLE, and the
    shortfall is a model-and-data result, not a fundamental limit.

    best achieved RMS-RGE: 15.506 % -> at least 0.879 pp of headroom
    remains between the best model and the bound.""",
"""    It is BELOW 15 %. We therefore have NO EVIDENCE that the bar is
    unreachable. We do NOT claim it is reachable: k was chosen on the same
    173 test targets, a second run (k=5) gives 14.669 %, the k-curve is not
    smooth, and no sampling interval has been computed.

    best achieved RMS-RGE: 15.506 % vs estimate 14.626 % (difference
    0.879 pp; an estimate, not guaranteed headroom).""")

print("3. results/fsr_attribution.txt")
patch(os.path.join(R, "fsr_attribution.txt"), "fsr_attribution.txt retired line",
      "  RESIDUAL: FDE n_g model vs FDTD      -2.06%   -2.36%   -1.73%",
      "  RESIDUAL: FDE n_g model vs FDTD      -2.06%   -2.36%   -1.73%   [RETIRED: percentage-FSR comparison was contaminated; use --fsrcheck: -2.83% / -3.03% in group index]")

print("4. results/trim_budget.txt")
OLD_TRIM = """  Note what this rewards: the figure of merit is how tightly the
  lambda_res errors CLUSTER once wrapped into one FSR, not the
  RMS error. A large but consistent offset is absorbed by the
  design bias and costs nothing; scatter is what costs power."""
NEW_TRIM = """  Note: circular spread alone did not rank the three models the same way
  as their maximum heater demand (see the CLUSTERING vs HEATER DEMAND
  table). A consistent offset is absorbed by a design bias; the worst
  residual error sets the heater range."""
patch(os.path.join(R, "trim_budget.txt"), "trim_budget.txt note", OLD_TRIM, NEW_TRIM)

print("5. mrr_template.py (so a re-run writes the corrected text)")
src = "mrr_template.py"
if os.path.isfile(src):
    text0, nl0 = read(src)
    backup(src)
    ok = []
    ok.append(patch(src, "source: floor title",
        '"INFORMATION FLOOR -- LOWER BOUND ON ACHIEVABLE RMS-RGE"',
        '"INFORMATION FLOOR -- UPPER-BOUND ESTIMATE (k-nearest-response predictor)"'))
    ok.append(patch(src, "source: asymptote line",
        'f"  learning curve refit with asymptote pinned here:"',
        'f"  learning curve refit with asymptote pinned here (DIAGNOSTIC ONLY; not identifiable, see floor_convergence.txt):"'))
    ok.append(patch(src, "source: O3 verdict block",
        r'L \+= \[f"    It is BELOW 15 %, and an upper bound below a threshold "\s*"PROVES the",.*?"limit\."\]',
        'L += ["    It is BELOW 15 %. We therefore have NO EVIDENCE that the bar is",\n'
        '                  "    unreachable. We do NOT claim it is reachable: k was chosen on the",\n'
        '                  "    same test targets, a second run (k=5) differs slightly, the",\n'
        '                  "    k-curve is not smooth, and no sampling interval has been computed."]',
        regex=True, done="NO EVIDENCE that the bar is"))
    ok.append(patch(src, "source: headroom lines",
        r'L \+= \["", f"    best achieved RMS-RGE: \{ach:\.3f\} % -> at least "\s*f"\{ach-F\[-1\]:\.3f\} pp of headroom",\s*"    remains between the best model and the bound\."\]',
        'L += ["", f"    best achieved RMS-RGE: {ach:.3f} % vs estimate {F[-1]:.3f} % "\n'
        '                      f"(difference {ach-F[-1]:.3f} pp; an estimate, not guaranteed headroom)."]',
        regex=True, done="an estimate, not guaranteed headroom"))
    ok.append(patch(src, "source: trimming note",
        r'"  Note what this rewards: the figure of merit is how tightly the",\s*"  lambda_res errors CLUSTER once wrapped into one FSR, not the",\s*"  RMS error\. A large but consistent offset is absorbed by the",\s*"  design bias and costs nothing; scatter is what costs power\.",',
        '"  Note: circular spread alone did not rank the three models the same way",\n'
        '              "  as their maximum heater demand (see the CLUSTERING vs HEATER DEMAND",\n'
        '              "  table). A consistent offset is absorbed by a design bias; the worst",\n'
        '              "  residual error sets the heater range.",',
        regex=True, done="circular spread alone did not rank"))
    try:
        py_compile.compile(src, doraise=True)
        print("  mrr_template.py still compiles: OK")
    except py_compile.PyCompileError as e:
        print("  mrr_template.py FAILED to compile -- restoring the backup.\n", e)
        shutil.copy2(src + ".bak_housekeeping", src)
else:
    print("  NOT FOUND (file missing)  mrr_template.py -- run this script in the Phase1 folder")

print("6. MASTER_FINDINGS.md")
mf = None
for cand in ["MASTER_FINDINGS.md", os.path.join(R, "MASTER_FINDINGS.md")] + glob.glob("*/MASTER_FINDINGS.md") + glob.glob("*/*/MASTER_FINDINGS.md"):
    if os.path.isfile(cand):
        mf = cand; break
if mf is None:
    print("  NOT FOUND (file missing)  MASTER_FINDINGS.md -- tell me where it is")
else:
    print("  using", mf)
    patch(mf, "MASTER 7.3 O3 verdict",
"""Since 14.626% < 15%: **O3's threshold lies above the information content of
four scalar targets. O3 is reachable in principle; the shortfall is a
model-and-data result, not a fundamental limit.** At least **0.879 pp** of
headroom remains.""",
"""Since 14.626% < 15%: **we have no evidence that O3's threshold is unreachable.**
We do not claim it is reachable: k was chosen on the same 173 test targets, a
second run (k=5) gives 14.669%, the k-curve is not smooth, and no sampling
interval was computed.""")
    patch(mf, "MASTER 10c affordable",
"""**Trimming is affordable.** The paper's negative result on resonance
placement prices the approach rather than invalidating it.""",
"""**Trimming is feasible under the stated assumptions** (one heater per ring,
3 K/mW). Affordability against a CPO power budget was NOT assessed, and the
paper makes no such claim.""")
    patch(mf, "MASTER 10c consistency beats accuracy",
"""Under a biased design the figure of merit is how tightly the lambda_res
errors CLUSTER once wrapped into one FSR -- NOT their RMS. xgb_norm has the
lowest RMS of the three fully-evaluated models and the HIGHEST trimming
demand; rf_raw is worse on RMS and cheaper to trim. A systematic offset is
absorbed by the design bias for free; scatter is paid for in heater range
and static power. For any system that will be trimmed -- i.e. every deployed
microring link -- consistency beats accuracy, and inverse models should be
selected on spread.""",
"""RETRACTED (2026-11-03): the earlier claim that clustering, not RMS, sets the
heater demand, and that inverse models should be selected on spread, is NOT
supported once circular spread is computed. Over the twenty targets the naive
network has the smallest circular spread (6.898 nm) but a maximum biased dT of
125.0 K; RF (raw) has a larger spread (7.873 nm) but the smallest maximum dT
(121.4 K); XGBoost (norm) lies between (7.114 nm, 131.9 K). Spread does not
rank the three models in the same order as heater demand, and the three are
within 10.5 K of each other on maximum dT. What the data support: the
requirement is capped at one FSR by construction (about 227 K) and the three
models need 121-132 K at worst under a biased design.""")

print("7. results/material_check.txt (superseded by --fsrcheck)")
mc = os.path.join(R, "material_check.txt")
if os.path.isfile(mc):
    os.makedirs(os.path.join(R, "_to_delete"), exist_ok=True)
    shutil.move(mc, os.path.join(R, "_to_delete", "material_check.txt"))
    print("  MOVED to results\\_to_delete\\ (delete that folder yourself when you are sure)")
else:
    print("  ALREADY DONE (file not present)")
print("\nDone. Backups end in .bak_housekeeping. Copy the whole printout above and send it to me.")
