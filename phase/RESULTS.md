# Uniform PnP SDP outputs

All seven scripts pnp-stage-1.py, pnp-stage-2.py, pnp-stage-3.py,
pnp-stage-4.py, pnp-stage-5a.py, pnp-stage-5a-PM.py and pnp-stage-5b.py
now print the same concise table and export CSV/JSON with identical columns.

Run e.g.:

    python phase/pnp-stage-5a.py
    python phase/pnp-stage-5a-PM.py

Files are written to phase/results/<stage>/results.csv and results.json.
Standard plots eph-vs-distance.png and rate-vs-distance.png appear when
there are >=2 usable distances. **All future stage-specific plots also go
into the same phase/results/<stage>/ directory**, rather than phase/ or
separate stage*-plots directories. Historical image files already present in
older folders are not automatically relocated.

Stage 5A additionally writes stage5a-rate-vs-plob.png. This compares
*illustrative per-original-round* Stage 5A rates (R_original, not conditional
R_acc) with the repeaterless pure-loss PLOB reference
K(eta)=-log2(1-eta), eta=10**(-0.2*L_km/10), where L_km is the total
Alice-Bob distance. The PLOB curve starts at L>0 because it diverges at
zero distance. Its units are bits per end-to-end channel use, so this
overlay is a visual reference only: protocol resource accounting, physical
monitor acceptance and observed statistics must be justified before
claiming any PLOB violation.

Stage 5A--PM provides a separate multiprocessing test-transmissivity
optimizer in the SAME stage script (no standalone optimizer file):

    python phase/pnp-stage-5a-PM.py --optimize --workers 4 --solver mosek --plot

This searches lambda_1,...,lambda_(M-1) while lambda_0=1, using a seeded
parallel heuristic. Its outputs are placed at:
    phase/results/stage5a-PM/optimization-results.json
    phase/results/stage5a-PM/pm-optimized-eph-comparison.png
    phase/results/stage5a-PM/pm-optimized-rate-comparison.png

Without --optimize the original fixed-vs-hand-picked comparison runs and
writes different filenames, so the two runs do not overwrite one another.
Auxiliary non-stage analysis scripts (fig6.py, comparison.py, nbar_scan.py)
write future plots/scans under phase/results/analysis/.

Uniform columns:

    stage, profile, L_km, distance_kind, M, n_bar, synthetic_model,
    e_ph_upper, e_bit, P_pass_cond, P_monitor, R_acc, R_original,
    epsilon_tail, beta, status, phase_bound_scope, tail_kind,
    P_monitor_scope, parameters_json, notes

Model: All synthetic announcement data now use the shared, two-arm,
coherent-reference interference model in pnp_benchmark.py. L_km is the
TOTAL Alice-Bob fiber distance (two equal arms, each L_km/2).
Per-arm transmission: eta_det * 10**(-XI*L_km/20). A balanced central
splitter and two on/off detectors determine the single-click outcomes.
The nominal n_bar is the mean before optional PM setting-dependent loss
and after any base signal conditioning/attenuation. It is used only to
synthesize honest reference detection data, not to restrict Eve to a
coherent or Poissonian source. Stage-specific security constraints are
unchanged.

The PM fixed (Stage5A) baseline uses lambda_x=1 in every basis, and its
synthetic statistics and overlap kernel agree with Stage5A at the same
N_plus, cutoff, monitor settings and distance. The PM default N_plus=3
is included in the original Stage5A cutoff scan.

Denominators: P_pass_cond is the observed central key-basis probability,
conditional on monitor acceptance where one exists. R_acc is per accepted
round; for Stages 1--3 it is per original round. P_monitor is 1 in
Stages 1--3 (no monitor), unknown in Stage4, and currently an
ILLUSTRATIVE 0.99 lower bound in the default Stage5A/PM/5B scans.
R_original is NULL where P_monitor is unknown. Nominal Stage5A/PM/5B
R_original values are not an experimental certified security rate until
monitor acceptance and observed channel data are independently verified.

Tail semantics: epsilon_tail is an upper bound or relaxation budget, not
the optimized bad mass beta. See tail_kind for the stage-specific meaning.
Plots distinguish different parameters_json values to avoid connecting
incomparable cutoff/basis settings.

IMPORTANT: Stages 1--4 currently include the HISTORICAL e_ph<=1/2
constraint in the SDP; its without-loss-of-generality justification has
not been proved. Treat their rates as exploratory, not security-certified.
Stage5A/PM/5B instead allow e_ph>1/2 in the SDP and apply a conservative
zero-privacy rule when only an upper bound >=1/2 is available.

Different stages still enforce different physical interfaces. Stage4
assumes an abstract monitor tail, Stage5A/PM use ideal PNA-BS occupancy
and tail bounds, Stage5B combines ideal PNA with extra attenuation and
continuum relaxation. Shared output does not make those assumptions
identical.

The older scripts used a simplified click-rate model and a different
distance convention, so rerun all scripts before comparing figures.
No PLOB-bound violation is established by these synthetic scans.
