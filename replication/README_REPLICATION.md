# QFE replication

**Current certification: STOP — ANALYSIS REPRODUCTION FAILURE.** Stage 1 passes
the numerical data checks, but tiny upstream value differences materially change
fresh ARIMA fits. See `REPRODUCTION_REVIEW_REQUIRED.md` before using these results.

Run from `replication/` with Python **3.12.6** in the pinned environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python 01_build_clean_iv_data.py --workers 8
.venv\Scripts\python 02_run_full_paper_analysis.py --workers 4
```

On other systems use the environment's `python` executable. These are the only
two required analysis entry points. Installing dependencies is environment setup.
The original workspace and `paper_qfe/` are never written by this package.

## Inputs and configuration

`config/replication_config.json` contains the locked specification and input paths.
Relative configuration paths resolve against this package directory. CLI path
overrides resolve against the caller's working directory:

```text
--raw-options-file PATH
--rates-file PATH
--output-root PATH
--reference-root PATH
--config PATH
--workers N
--force
```

Raw inputs are CSV files: `options_eod_all.csv` and `div yield and rfr.csv`.
The default paths locate them in `inputs/` inside this package. Keep their column names and
quote fields intact. The source options contain the labelled 15:45 option and
underlying books, expiration, strike, option type, volume, and open interest.
The rate file supplies exact calendar dates, daily dividend yields, and six
maturity rates. No extra rate dates are fabricated. No input conversion to Excel
is required. The installed dependency versions and actual platform are recorded
for every execution. Set no `OMP_NUM_THREADS` override: the certified XGBoost
context has that variable unset; OpenBLAS/MKL use one thread during analysis.

The four original source workbooks are also copied into `inputs/` for inspection.
The scripts consume the two CSV files directly; the workbooks are supporting
source material. `inputs/INPUTS_MANIFEST.json` records their roles, sizes, and
SHA256 hashes. Original workspace files are preserved. The raw options archive
is approximately 9.15 GB; all six input files must accompany a portable handoff.
Optional historical comparisons under `reference_root` are not required inputs.

## Stage 1

Stage 1 scans all raw observations, records the sequential cleaning funnel,
inverts prices with the frozen Brent bracket/tolerance, and interpolates the
25-by-20 grid. Working option data use Parquet. Historical clean/IV CSV numeric
parsing boundaries are retained in memory so serialization does not change model
inputs. `scipy.special.ndtr` calls the same normal-CDF implementation used by
`scipy.stats.norm.cdf`; the certified pricing arithmetic and root solver are
unchanged. Its equality check is included in the tests.

Outputs include `data/options_clean.parquet`, `data/options_with_iv.parquet`,
`data/iv_grid_{wide,long,map,day_stats}.{csv,parquet}`, row counters, IV statuses,
cleaning funnel, full-value comparisons, and the upstream manifest. Failed
inversions remain NaN. Invalid inputs rejected by the historical IV function are
counted separately. A failed numerical comparison closes the analysis gate.
Optional original upstream CSV files under `reference_root` provide independent
full-row comparisons. Reference-only compact final-result fixtures are bundled
under `manifests/reference_results/`; they are used exclusively for comparisons.

## Stage 2

Stage 2 verifies Stage 1 hashes and its reproduction gate, then directly runs the
corrected XNYS h1/h2 design. It fits the certified model candidates, creates the
target-independent signal intersection, maps contracts using entry observations,
nets collisions, and extracts exact exit books. Processed exits take priority;
the raw archive supplies exact-date/key recovery without volume/OI/IV/spread
eligibility filters. A single targeted raw scan is cached by input content and
required pairs. Full original and four-state premium-normalized portfolios,
Greek attribution, inference, corrections, and robustness tables are recomputed.

`outputs/excel/QFE_Full_Results.xlsx` has 23 ordered worksheets with plain cells,
numeric values, and no charts, colors, merged cells, or decorative borders.
Each sheet is mirrored to `outputs/tables/<sheet-name>.csv`. Raw option records
remain outside Excel. `outputs/diagnostics/replication_final_comparison.csv`
contains reference/new/difference/tolerance/status fields. Numerical failures
return a nonzero exit code and are retained in diagnostics.

## Caches and manifests

Cache partitions include source SHA256, the effective locked configuration,
code identity, dependencies, and the OpenMP context where applicable. Completion
manifests verify checkpoint contents before reuse. `--force` recomputes the
generated checkpoints. No reference result table is a computational input.
Without cached fits, a complete model run can take substantial time; interrupted
completed node checkpoints can be reused by the same code/configuration.
Optional historical fitted states are reused only after their content hashes,
versions, fitting functions, and bit-identical model inputs are verified; all
predictions are regenerated. A raw-data/code-only handoff fits models directly.

`manifests/method_source_provenance.json` identifies the original functions and
source hashes. `REPLICATION_DELIVERY_MANIFEST.json` hashes the entry points,
helpers, requirements, configuration, tests, workbook, outputs, and logs.
The embedded Excel file inventory excludes its own workbook and inventory files
to avoid circular hashes; the external delivery manifest hashes these artifacts.

## Tests

From this directory:

```text
python -B -m unittest discover -s tests -v
```

Tests require no raw options archive. They include independently calculated
pricing/accounting examples, missing/future-data mutations, calendar exclusions,
entry-only mapping and controls, finite-difference Greeks, normalization,
multiple-testing calculations, and corrupted-cache rejection.

## Public distribution

Repository: https://github.com/Balint-Mariam/paperqfev2

The complete package, including raw data and caches, is supplied as a release
archive. The Git checkout includes code, supporting source workbooks, rates,
result tables, and audit evidence. Download and extract the release for the raw
options input. The default `reference_root` is the package itself; historical
workspace comparisons can be enabled with `--reference-root PATH`.
