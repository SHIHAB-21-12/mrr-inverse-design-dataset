DRAFT v3 -- ML-Driven Inverse Design of Microring Resonators for CPO
Md Shidul Islam Shihab, ECE, KUET

CHANGED FROM v2
  New Section 2.4: full 3-D validation of the ring model (9 FDTD runs,
  ~120 h). New Table 3. Three new limitations. Abstract updated.
  All v2 numbers unchanged -- nothing in the model comparison moved.

WHAT THE RING VALIDATION FOUND
  GOOD   lambda_res agrees to 0.10% across all nine runs.
  GOOD   At Q_L <= 4300 the analytic model is correct to within 2%, by
         Richardson extrapolation to zero mesh. The 4-6% first observed at
         the production mesh was NUMERICAL error in the validation, not
         model error -- 32 nm was converged for kappa^2 (single pass) but
         not for Q_L (hundreds of round trips).
  BAD    Above Q_L ~ 4300 the model over-predicts by 17-27%, mesh
         refinement makes it worse, and the cause is NOT established.
         A wavelength-blind bend-loss model is a hypothesis only.
  BAD    IL does not validate. Transmission errors -12.5%, +1.5%, -17.9%,
         no consistent sign.
  SYSTEMATIC  FSR biased +2.2 to +2.8% in all nine runs -> the real ring's
         group index is ~2.5% below the model's, because n_g comes from a
         STRAIGHT-waveguide sweep. Fixable, but it would change every
         dataset label. Future work.

UNAFFECTED BY ALL OF THIS
  Every model was trained and scored on identical labels, so a shared label
  bias cancels from every model-to-model comparison: O6, the theta metric
  result, buildability, O4 and O5 all stand.

MUST BE DONE BEFORE SUBMISSION
  1. INDEPENDENT EXPERT REVIEW OF THE PHYSICAL METHODS. Still not done.
     The ring validation is a demonstration of why: three days of
     simulation surfaced a defect neither the author nor an AI assistant
     anticipated, and that is one category of problem out of many.
  2. LICENSE file for the repository.
  3. Reference [12] (Torabi et al.) is an unrefereed arXiv preprint.
     Check for a peer-reviewed version.
  4. Reference [13] (Bogaerts et al.): the typeset Eq. 6 was never read
     directly (paywalled). Confirm with library access.
  5. Replace the placeholder in "Data and code availability" with the real
     repository URL/DOI.

NUMBERS DELIBERATELY ABSENT
  - No extrapolated information-floor asymptote: the bank-size sweep does
    not identify one, and a grid search for it runs to the edge of its own
    search range.
  - No per-parameter information "floors": the estimator minimises the
    aggregate, so per-parameter values are not individual bounds.
  - O1's FSR agreement is "within the published figure's precision", not
    0.02%. Tang et al. quote 4.4 nm to two significant figures.
  - No mechanism claimed for the high-Q discrepancy. Hypothesis only.

OPEN ITEM FOR THE AUTHOR
  Section 2.6 records that O3 was reachable by enlarging the training set
  with rows drawn from the FITTED analytic model (measured 14.981%), and
  that this was declined because those rows add sampling density, not
  physics. Confirm you are happy to state that publicly.
