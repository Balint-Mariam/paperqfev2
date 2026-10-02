# Numerical reproduction requires review

Status: **Stage 1 reproduced within the specified tolerances; full analysis FAILED.**
The package has not been accepted as reproducing the frozen final paper results.
The original design, source files, and Stage 1–6 outputs remain unchanged.

The two entry points reconstructed the raw data and ran the complete corrected
analysis. Stage 1 passed 642 checks. Stage 2 passed its 21 accounting/chronology
checks with three inherited warnings, but 2,358 final numerical comparisons
failed; 1,684 passed. The Excel workbook contains the actual recomputed results,
with `Stage2_status=FAILED` and all comparison failures in `21_Checks`.

The reconstructed grid has the same rows, dates, nodes, missingness and coverage.
However, 3,434 finite grid values differ from the certified CSV when parsed for
forecasting. Maximum difference: `9.506284648352903e-15`. The first mismatch is
2013-10-04, `iv_x16_t08`: new `0.1317435036201068`, reference
`0.1317435036201067`.

The first detected upstream divergence is rate annualization, before IV
inversion. For 2014-05-28 the stored daily dividend yield is
`0.0001527619999999`. The frozen clean CSV contains annual q
`0.039243540856820136`; the current frozen formula and runtime calculate
`0.03924354085682036`. Full comparisons found annual-rate differences up to
`3.001072e-16` and IV differences up to `1.870726e-14`. This does not establish
which historical numerical backend produced the stored value. Isolated NumPy
1.26.4, 2.2.6, and 2.4.2 checks also produced the current value; they were
diagnostics, not alternative package specifications.

These differences are below the requested data tolerance, but fresh ARIMA order
selection/final convergence is discontinuous at these inputs. A read-only
reference-grid refit reproduced the certified outcomes for three affected nodes:

| Node | Exact original grid | Reconstructed grid |
|---|---|---|
| iv_x03_t02 | OK, predictions identical to certified fit | FINAL_NONCONVERGENCE |
| iv_x01_t04 | FINAL_NONCONVERGENCE | OK |
| iv_x08_t05 | OK, predictions identical to certified fit | FINAL_NONCONVERGENCE |

The strict common forecast sample changed from **159,514 to 159,060**; the economic
candidate count changed from **163,089 to 162,186**. Trading dates remain 408, but
active contract counts changed from 7,796/7,337/7,615 to 7,805/7,362/7,637. All
12 BID_ASK means remain negative, which does not resolve the reproduction failure.

No reference values were substituted into computed results. Certified fitted
states were rejected because reconstructed model inputs were not bit-identical.
No hyperparameters, convergence rules, cleaning filters or signals were changed
to force a match. The original numerical environment/execution provenance for
the clean/IV files is needed to resolve this divergence; methodological review
is required before accepting this package's results.

Machine-readable evidence is under `outputs/diagnostics/`, including
`stage1_reproduction_checks.csv`, `replication_final_comparison.csv`,
`reference_surface_refit_convergence_test.json`, and the execution manifests.
The original first-run log also records a shared-log closing error after results
were exported. That I/O defect has been repaired and separately unit-tested; it
does not explain the numerical comparison failures.
