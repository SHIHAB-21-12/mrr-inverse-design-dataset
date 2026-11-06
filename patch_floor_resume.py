"""
Fixes a bug in mrr_template.py --floor: when a saved bank is resumed (or extended
with a larger --nbank), the random generator restarted from its seed, so the new
chunks repeated geometries already in the bank. A bank "extended" from 200,000 to
1,000,000 that way holds far fewer distinct geometries than its size says.

After the fix a RESUMED build draws from a stream keyed by the bank length already
built. A build from scratch is unchanged (same seed, same bank as before).

Run from the project folder:   py -3 patch_floor_resume.py
It edits mrr_template.py in place and writes mrr_template.py.bak2 first.
"""
import shutil, sys
p = "mrr_template.py"
s = open(p, encoding="utf-8").read()
old = "    while len(G) < n_bank:\n        k = min(FLOOR_CHUNK, n_bank - len(G))"
new = ("    if len(G) > 0:      # resumed build: do NOT replay the seed's first draws\n"
       "        rng = np.random.default_rng([FLOOR_SEED, len(G)])\n"
       "    while len(G) < n_bank:\n        k = min(FLOOR_CHUNK, n_bank - len(G))")
if s.count(old) != 1:
    sys.exit(f"expected exactly 1 match, found {s.count(old)}; nothing changed")
shutil.copyfile(p, p + ".bak2")
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("patched OK; backup written to mrr_template.py.bak2")
