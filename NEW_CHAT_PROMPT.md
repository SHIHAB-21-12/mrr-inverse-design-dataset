# THE PROMPT FOR THE NEW CHAT
Paste everything between the lines, after attaching the five files.

---

I am Md Shidul Islam Shihab, 4th-year BSc student in Electronics &
Communication Engineering at KUET. I am writing a research paper titled
"ML-Driven Inverse Design of Microring Resonators for Co-Packaged Optics /
High-Bandwidth Optical Interconnects", targeting a Q1 or Q2 journal and MSc/PhD
applications in the Netherlands and the US. This continues work from a previous
conversation that ran out of context.

I have attached five files. **Read HANDOVER.md first — it indexes everything
else and contains the constraints below in full.** Then read MASTER_FINDINGS.md
before you answer anything substantive. Do not ask me to re-explain what is in
them.

## Non-negotiable constraints

1. `Thesis_Master_Reference_v1_3.pdf` is settled ground truth. Do not audit it,
   do not suggest changes to its parameter ranges, equations, thresholds or
   objectives, do not propose restructuring anything in it. Treat every fixed
   default as settled for the entire project. The only exception: if a specific
   simulation result I show you contradicts a stated assumption, apply the
   decision rule or fallback the document itself already specifies (Section 7
   and Phase 1.6 have these) rather than proposing a new one.
2. I need working output, not planning documents. Actual Python (lumapi or
   MEEP, matching whichever solver applies), help setting it up, help debugging
   errors I hit, and help interpreting results — checked against the specific
   numeric thresholds already fixed in the document, not against your judgement
   of what looks reasonable.
3. **ONE script only.** Everything goes into `mrr_template.py` as a new
   `--mode`. Never give me a second script.
4. Give me one step at a time and confirm each works before the next.
5. Always give instructions as numbered steps.
6. I am on Windows PowerShell. `&&` is invalid. `del a b` and `rmdir /s /q` are
   CMD syntax and fail. Use `Remove-Item`.
7. My commits carry no assistant attribution. They are my commits only.
8. Don't restate methodology I already have in the PDF — execute it.

## What I know and don't

I know Python and basic ML/DL. I am still learning photonics simulation tools.
Explain photonics reasoning when it matters to a decision; don't explain Python.

## Where the work stands

The framework is complete and the simulation campaign is finished. Draft v9 of
the paper exists (20 pages). Three of seven objectives are not met and are
reported as measured. Section 7 of HANDOVER.md lists what remains.

The single most important open item is an independent expert review of the
physics, which I will arrange myself.

## How I need you to work

The previous assistant was wrong more than a dozen times on this project, and
every correction came from my data rather than its reasoning. Section 5 of
HANDOVER.md lists the recurring failure modes. Specifically:

- Before you build a test, check that the quantity you are perturbing can
  actually change the result. Two tests were built that were mathematically
  incapable of measuring anything, and the flat output was presented to me as
  evidence of robustness.
- Do not guess at solver API behaviour. Measure it and show me the measurement.
- When you state a number, say where it was computed and against what. Two
  different values for the same residual were quoted before one was computed
  correctly.
- If a result looks impossible or internally inconsistent, say so instead of
  explaining it away.
- Tell me plainly when you are uncertain, and tell me when you have been wrong.

## My first question

[Replace this line with what you actually want. For example:]
- "Audit every caller of `_fit_sized` for the ignored `seed` argument and tell
  me whether any published result is affected."
- "Help me write the email to a photonics professor asking for a methods
  review, with the two specific questions from HANDOVER.md Section 7."
- "Walk me through what a referee would attack first in draft v9."

---

## Attach these five files, in this order

1. `HANDOVER.md`
2. `MASTER_FINDINGS.md`
3. `Thesis_Master_Reference_v1_3.pdf`
4. `paper/paper.tex`
5. `mrr_template.py`

If the new chat supports Projects, put these five in the project knowledge
rather than attaching them to a single message — then they persist across
conversations and you will not have to re-upload.
