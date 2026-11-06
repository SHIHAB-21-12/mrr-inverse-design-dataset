"""
Fixes the closed-loop report in mrr_template.py so it counts only the requested
evenly spaced rows (it used to count every record left in the json, giving
104 + 4 = 108 'targets' for a 100-target run). Run from the project folder:
    py -3 patch_closedloop_count.py
It edits mrr_template.py in place (a backup mrr_template.py.bak is written first).
"""
import shutil, sys
p = "mrr_template.py"
s = open(p, encoding="utf-8").read()
old = '''    ok = [k for k, v in done.items() if v["status"] == "ok"]
    un = [k for k, v in done.items() if v["status"] == "unrealisable"]
    L += [f"  simulated successfully : {len(ok)}",'''
new = '''    _req = set(str(int(i)) for i in idx)       # only the rows this run asked for
    ok = [k for k, v in done.items() if v["status"] == "ok" and k in _req]
    un = [k for k, v in done.items() if v["status"] == "unrealisable" and k in _req]
    L += [f"  simulated successfully : {len(ok)}",'''
if s.count(old) != 1:
    sys.exit(f"expected exactly 1 match, found {s.count(old)}; nothing changed")
shutil.copyfile(p, p + ".bak")
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("patched OK; backup written to mrr_template.py.bak")
