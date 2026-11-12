# P9 step 3 pre-registered predictions: all-pass closed loop with a fresh 3-D FDTD coupler run (written 2026-11-12, before any FDTD run)

Method: the naive inverse network (seed 42) trained on the all-pass map predicts geometries for 20 evenly spaced test targets
(np.linspace over the 173 test rows, as in the paper's closed loop). Each buildable prediction is re-simulated with the same
3-D FDTD coupler call as `run_closedloop` (kappa^2 measured), and the all-pass ring is assembled analytically with that kappa^2.
Error = (re-simulated - requested)/requested for lambda_res, FSR, Q_L and notch depth (1 - T_min).
Aggregate = sqrt(mean of the four per-response mean-square errors), as in the paper.

C7: aggregate closed-loop error of the buildable designs is below 20 % (the paper's O3 closed-loop threshold).
C8: at least 18 of the 20 predictions are inside the design box (realisable).
C9: the largest per-response error is Q_L or notch depth, not lambda_res or FSR.
C10: aggregate closed-loop error is no more than twice the model-level error on the same designs (here the model-level number replays the fitted kappa^2).

A failure of any of these is reported as a failure. Nothing is tuned after seeing results.
Limits: same platform, same coupler FDTD set and bend/mode fits as the add-drop dataset; the ring is analytic.
