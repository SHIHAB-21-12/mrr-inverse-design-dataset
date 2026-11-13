# P10 -- candidate new methods: pre-registration (write BEFORE any test is run; commit first)
Date written: 2026-11-13. Author: Shihab (with Claude). Baselines are from the paper (v23) and the P9 full run.

## Purpose
Find a method that goes beyond the present ceiling of the paper. The ceilings found so far:
 (a) geometry-space recovery: only R is recoverable (R^2 >= 0.8) for all methods on all maps;
 (b) closed-loop response error floor ~3-4 % = coupler-model (kappa^2) scatter vs 3-D FDTD (s.d. of log ratio 0.032-0.045);
 (c) six-parameter asymmetric ring: naive response error 37.9 % (9.4 % after post-hoc g1<=g2 relabelling vs 3-4 % on four-parameter maps);
 (d) lambda_res is not placeable (wrapped target; needs R to ~7 nm);
 (e) tandem implementation unbuildable on the new maps (not used).
Baselines to beat (model-level response RMS, naive, 5 seeds): add-drop 3.78 %, all-pass 3.04 %, asym 37.93 %.
Closed-loop FDTD baseline (naive): add-drop 3.99 % (96 of 100 buildable); all-pass 3.24 % (19 of 20).

## Candidates and the single small test for each (no FDTD unless stated)
C1 Invariant-coordinate inverse model. The network predicts identifiable coordinates z = (L, theta, [w]) for the four-parameter maps and (L, theta1, theta2 unordered) for the asymmetric ring; geometry is recovered by Eq. (1) with a fixed rule for the free coordinate Lc. 
   T2 PASS if ALL: (i) buildable = 100 % by construction; (ii) add-drop and all-pass model-level response RMS <= naive mean + 1 s.d. (<= 4.9 % and <= 3.7 %); (iii) asym response RMS <= 12 %. STRONG PASS if (iii) <= 5 %.
C2 One-step FDTD theta correction. After the network design, run ONE 3-D FDTD coupler simulation, convert kappa^2_FDTD to theta, move g along the theta level set (L and Lc fixed) so that theta matches the target, re-simulate once.
   T4 (offline precondition): residual r = ln(kappa^2_FDTD / kappa^2_model) over all available FDTD points must be spatially smooth: PASS if the correlation of r between pairs closer than 0.1 (range-scaled distance in w,g,Lc,R) is >= 0.6 AND a repeat-FDTD noise estimate (same geometry, mesh offset changed) is <= 1/3 of the s.d. of r. FAIL (white-noise scatter): C2 is dropped and the paper reports that the floor is not model error.
   T4b (needs FDTD, only if T4 passes): 19-20 designs, corrected closed-loop RMS <= 50 % of the uncorrected value (<= 1.6 % on the all-pass designs).
C3 Hybrid coupler surrogate (closed-form theta model + Gaussian-process residual). 
   T3 PASS if on the 68 out-of-sample FDTD runs (fit on the 200 only) the MAPE falls by >= 25 % relative to 1.23-5.16 %-band baseline MAPE (recompute the baseline on the same 68) and the 95 % bootstrap interval of the difference excludes zero.
C5 Jacobian-pullback response-equivalent error (device-agnostic generalisation of theta-error): D_J = sqrt(mean over test rows of ||W^(1/2) J (p_pred - p_true)||^2), J = Jacobian of [log FSR, log Q, log IL (or depth)] w.r.t. range-scaled parameters at the true geometry (lambda_res excluded, stated).
   T1 PASS if on EACH of the three maps: pooled per-row Spearman correlation of D_J with the per-row model-level response error is >= 0.85 and exceeds that of per-row RMS-RGE by >= 0.20.
C6 Data-driven identifiable-subspace discovery (forward-surrogate Jacobian SVD, no analytic knowledge). 
   T5 PASS if on the add-drop map the median principal angle between the discovered stiff 2-D subspace and the analytic (grad L, grad theta) subspace is < 15 degrees over the 173 test rows, and on the asymmetric map the number of singular values above 10 % of the largest is 3 (L, theta1, theta2).
C8 Conformal certificate for model-level response error (split conformal, calibration on the 172 validation rows).
   T6 PASS if empirical coverage on the 173 test rows at nominal 90 % lies in [85 %, 95 %] on every map.

## Decision rule after Stage 1 (written now, applied later)
Rank by: number of PASS criteria met, then by size of the improvement over the baselines above, then by novelty risk (re-run the literature search on the exact claim). Carry forward at most two. A candidate that fails its test is reported as a negative result and is not tuned and re-run.
## Not carried (and why)
C4 robust level-set selection (benefit already measured small and vanishing with thickness variation, Sec. 3.4); lambda_res placement methods (limited by 1.5 nm ring-model uncertainty, not by the inverse method).
