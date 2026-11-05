# Frozen timedep regression outputs

These expectations were captured on 2026-11-05 from the pre-Numba growth
implementation at commit `840994d`, after the user confirmed numerical parity
and passing production-integration tests. The former test oracle was used once
to export the data and then removed. Production Numba output was not used to
generate expectations. Source hashes, seeds and library versions are recorded
under `provenance` in `solvers.json`.

- `growth.npz` stores every field of 32 growth cases: 12 model/detection/initial-
  length combinations with 400 samples, six scalar cases, nine nonuniform-series
  cases across the three stijghoogte methods, and five event-boundary cases.
  Keys have the form `<case>__<field>`. Non-eroding legacy histories were expanded
  to constant columns to match the public `(time points, samples)` matrix shape.
  `final_intervention_hours` freezes the input required for an intervention
  scheduled exactly at the final time point.
- `solvers.json` stores every `ProbResult` field for both FORM variants, Monte
  Carlo and directional sampling across the supported growth-model cases, plus
  Monte Carlo with stochastic intervention: nine results in total.

Tests load these files directly with `allow_pickle=False`. They do not execute
old model code or regenerate expectations. Growth fields retain exact NaN-aware comparisons,
except `dldt_avg`, which retains its original relative tolerance of `1e-12`.
Solver floating-point values, including the design-point and importance-factor
dictionaries, use relative tolerance `1e-9` and absolute tolerance `1e-12` for
rounding differences amplified by the iterative solvers between numerical-library
builds. The environment migration was checked against the historical calculation:
all nine solver results were exactly equal to it in the new environment. Field
names, convergence flags, evaluation counts, messages and absent values remain
exact. Larger numerical
changes still require deliberate review when changing OpenTURNS or other
numerical-library versions.

Do not regenerate these files from the implementation being tested to make a
failing test pass. Changes to the numerical model require an independently
reviewed reference and an explanation of the changed expectations. Historical
source remains available in git; no alternate implementation is kept in tests.
