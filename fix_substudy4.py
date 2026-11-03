r"""
fix_substudy4.py -- marks the Q_L figures in records\substudy4_coupling_boundary.txt as
superseded and appends the correct figures for the current dataset.
Run from D:\Abroad\full guide\Phase1 :   py -3 fix_substudy4.py
Backs up first (name.bak_housekeeping). Safe to run twice.
"""
import os, shutil

path = os.path.join("records", "substudy4_coupling_boundary.txt")
if not os.path.isfile(path):
    raise SystemExit("NOT FOUND: " + path + "  (run this in the Phase1 folder)")
raw = open(path, "r", encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in raw else "\n"
text = raw.replace("\r\n", "\n")
MARK = "UPDATE 2026-11-03"
if MARK in text:
    print("ALREADY DONE"); raise SystemExit
b = path + ".bak_housekeeping"
if not os.path.exists(b):
    shutil.copy2(path, b)

banner = (
"NOTE (2026-11-03): the Q_L figures in this file (min 765, max 187,611, and the\n"
"band counts) were computed on an EARLIER 1,000-sample label set and are\n"
"SUPERSEDED. The current figures, for the 1,000 primary rows and for the full\n"
"1,150-row dataset, are in the UPDATE block at the end of this file. The\n"
"coupler-set figures (48 points) are the first stage; the final coupler set has\n"
"200 points.\n\n")
head, sep, rest = text.partition("====================================================================\n")
text = head + sep + banner + rest

update = (
"\n====================================================================\n"
"UPDATE 2026-11-03 -- CURRENT DATASET (computed from data/train+val+test.csv)\n"
"====================================================================\n"
"                                  primary 1,000      full 1,150\n"
"  Q_L min                                774              774\n"
"  Q_L max                            190,358          190,358\n"
"  log10 range                      2.89 to 5.28     2.89 to 5.28\n"
"     below 10^3                            5                5\n"
"     10^3-10^4                           380              435\n"
"     10^4-10^5                           588              683\n"
"     above 10^5                           27               27\n"
"  inside the 10^3-10^5 band            96.8%            97.2%\n"
"  kappa^2 / (1 - a^2)    min 0.23, max 106.4 over the 1,150 rows\n"
"  The conclusion (target band populated across its full width) is unchanged.\n"
"  Quote Q_L as 774 to 190,358 in the paper and in MASTER_FINDINGS.\n")
text = text.rstrip("\n") + "\n" + update
open(path, "w", encoding="utf-8", newline="").write(text.replace("\n", nl))
print("PATCHED  " + path)
