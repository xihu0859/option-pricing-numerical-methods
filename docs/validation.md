# Validation of the public edition

Checked on 6 November 2026 using Python 3.13.7, NumPy 2.3.3, SciPy 1.16.2 and Matplotlib 3.10.6.

## Executed checks

- `python verify.py`: PASS. Analytical digital benchmark, two independent barrier quadratures,
  second-order FD refinement, nonnegative damped FD solutions, pathwise bridge ordering,
  Asian monitoring dates/payoffs, the geometric one-date vanilla limit and all three app engines.
- `python audit_asian.py`: PASS. Recomputed all six saved Asian production rows and 160
  replication rows, general lognormal moments, pilot coefficient and control mean.
- Independent two-million-path joint-normal Asian validation: PASS. Difference from the
  production result is approximately 0.981 combined standard errors.
- `python scripts/plot_summary.py`: generated plots from the recorded CSV results.

Detailed audit output is in [asian_validation.json](asian_validation.json).

All Python sources were parsed. The public report was checked for student-identifier removal
and its first page rendered for visual inspection. Source code contains no credentials.

The full original timing experiment was not rerun. CSV timing fields retain the original
three-run medians; this audit checks numerical outputs and recomputes the efficiency ratios
from those recorded timings. The desktop calculation engine was tested through `calculate`;
the Tkinter window was not manually exercised during this publication pass.
