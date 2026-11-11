# P9 pre-registered predictions (written 2026-11-12, BEFORE any model is trained on the new maps)

Basis: the Jacobian output of `p9_responders.py --jac` (add-drop reproduces the dataset to 1e-12).

| map | median singular values of d log(targets)/d(parameters scaled to range) |
|---|---|
| add-drop (4 parameters) | 5.59, 0.917, 0.125, 0.0272 |
| all-pass (4 parameters) | 4.85, 0.816, 0.124, 0.0249 |
| asymmetric add-drop (6 parameters) | 4.95, 3.02, 0.494, 0.102 |

Reading: a singular value s means a move of one parameter-range along that direction changes the log targets by about s.
Counting argument, applied at a 10-15 % response tolerance: add-drop and all-pass have 2 visible directions, i.e. about
a 2-D family of equivalent designs; the asymmetric map has 3-4 visible directions on 6 parameters, i.e. about a 2-3-D family.
The all-pass loss term adds essentially no information beyond the add-drop case (same singular values).

## Predictions (naive DNN inverse, five seeds 42-46, test split 805/172/173, model-level response error)

C0 (pipeline validity, add-drop control): mean RMS-RGE between 14 % and 17 %, and R^2(R) >= 0.95. If not, the
   new pipeline differs from the paper's and nothing below is interpreted until that is explained.
C1 (all three maps): at most one parameter has test R^2 >= 0.80, and it is R.
C2 (all three maps): response error of buildable predictions, re-evaluated with the fitted model, is below 20 % (RMS over the four targets).
C3 (asymmetric map): each of g1, Lc1, g2, Lc2 has mean test R^2 < 0.5.
C4 (asymmetric vs add-drop control): the asymmetric naive RMS-RGE is higher than the control's.
C5 (all three maps): the tandem network's buildable fraction is below 90 %.
C6 (all-pass vs add-drop control): naive RMS-RGE differs by no more than 3 percentage points.

## What counts as a failure of the counting argument
- C1 fails if two or more parameters are recovered on any map, or if the recovered one is not R.
- C2 fails if the naive network's response error exceeds 20 % on any map: the "fails in geometry, passes in response" finding would then not generalise.
- Failures are reported, not tuned away: no hyperparameter is changed after seeing results; the settings are those of Appendix B of the paper.

## Limits stated in advance
- Response error here replays the fitted coupler model; only the FDTD coupler re-simulation (step 3, about 20 targets) is an independent check.
- All three maps share one platform (220 nm SOI) and one coupler model: this is generality across response maps and dimensionality, not across materials.
