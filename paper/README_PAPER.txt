DRAFT v2 -- ML-Driven Inverse Design of Microring Resonators for CPO
Md Shidul Islam Shihab, ECE, KUET

WHAT CHANGED FROM v1
  Every number is re-derived from the 200-coupler dataset. v1 was built on
  the 48-coupler fit and all of its tables and figures are superseded.
  Figures 1-5 are new; v1's figures are not reused.
  The paper is now organised around O6 as the central contribution.

MUST BE DONE BEFORE SUBMISSION
  1. Independent expert review of the physical methods. The dual-fidelity
     construction and the kappa^2 model are the places where a photonics
     specialist would catch what the author and an AI assistant cannot.
     This has NOT been done.
  2. Add a LICENSE file to the code repository.
  3. Reference [12] (Torabi et al.) is an unrefereed arXiv preprint. Check
     for a peer-reviewed version and update the citation.
  4. Reference [13] (Bogaerts et al.): the typeset Eq. 6 in the original
     was never read directly (paywalled). Confirm with library access.
  5. Decide the data/code release location and replace the placeholder
     sentence in "Data and code availability" with the real URL/DOI.

NUMBERS THAT ARE DELIBERATELY ABSENT
  - No extrapolated information-floor asymptote. The bank-size sweep does
    not identify one, and this is stated in the paper rather than papered
    over with an unidentifiable fit.
  - No per-parameter information "floors". The estimator minimises the
    aggregate, so per-parameter values are not individual bounds; the best
    model legitimately beats two of them.
  - O1's FSR agreement is reported as "within the published figure's
    precision", not as 0.02%. Tang et al. quote 4.4 nm to two significant
    figures, i.e. about +/-1.1%; quoting 0.02% would claim precision the
    reference does not have.

OPEN ITEM FOR THE AUTHOR
  O3 was reachable by enlarging the training set with rows drawn from the
  fitted analytic model (measured at 14.981%). That route was declined
  because those rows add sampling density, not physics. The paper records
  the decision in Section 4.6. Confirm you are happy to state it publicly.
