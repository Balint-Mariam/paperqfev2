# Replication instructions

This package runs the complete sequence from raw option observations to forecasting,
contract execution, Greek attribution, risk controls, and statistical evaluation.
The empirical definitions and the relationship between inputs and outputs are in
[METHODOLOGY.md](METHODOLOGY.md).

## Files

- `01_build_clean_iv_data.py`: cleaning, implied volatility, and the IV grid.
- `02_run_full_paper_analysis.py`: forecasting and the complete economic analysis.
- `config/replication_config.json`: paths and the fixed empirical specification.
- `inputs/`: the raw options CSV, rate/dividend CSV, and four source workbooks.
- `src/`: functions called by the two entry points.
- `tests/`: independent numerical and timing checks.
- `outputs/data/`: generated datasets, including intermediate contract records.
- `outputs/tables/`: numerical result tables and workbook CSV counterparts.
- `outputs/excel/QFE_Full_Results.xlsx`: 23 plain numerical worksheets.
- `outputs/figures/`: figures generated from the numerical results.
- `outputs/diagnostics/`: input identities, runtime environments, and automated checks.
- `manifests/`: the empirical specification and historical comparison targets.

## Environment and execution

Reference environment: **Windows 11 x64, Python 3.12.6**, with the exact package
versions in `requirements.txt`. The recorded numerical environment is supplied
in `outputs/diagnostics/stage2_environment.json`.

Installation and execution:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python 01_build_clean_iv_data.py --workers 8
.venv\Scripts\python 02_run_full_paper_analysis.py --workers 4
```

On other systems use the environment's `python` executable. Run these commands
from this package directory. The input paths already point inside `inputs/`.
A new execution creates logs and content-identified caches as needed; neither
is required in the download. Avoid setting `OMP_NUM_THREADS`; OpenBLAS and MKL
use one thread during the analysis. Platform and dependency versions are recorded.

## Configuration

Paths in the JSON configuration resolve against this package directory. CLI
path overrides resolve against the caller's working directory:

```text
--raw-options-file PATH
--rates-file PATH
--output-root PATH
--reference-root PATH
--config PATH
--workers N
--force
```

The locked empirical fields are checked against `manifests/locked_specification.json`.
Changing file paths or worker counts does not change the empirical design. `--force`
recomputes generated checkpoints. The package-local expected grid files are selected by `--reference-root` when
checking a repeated upstream build. No external workspace is required.

## Verification

```powershell
.venv\Scripts\python -B -m unittest discover -s tests -v
```

Unit tests require no raw options archive. They check pricing, chronology,
future-data independence, contract mapping, PnL accounting, Greek controls,
normalization, inference, and rejection of corrupted caches.

`inputs/INPUTS_MANIFEST.json` provides the unchanged input checksums.
`REPLICATION_DELIVERY_MANIFEST.json` inventories the package. Runtime checks and
numerical comparisons are written to `outputs/diagnostics/`. A failed required
check produces a nonzero process exit code. The workbook records actual run statuses. Expected tables are produced by a
complete package execution and checked by an independent rerun.

## Download

The complete package is provided at:
https://github.com/Balint-Mariam/paperqfev2/releases/tag/replication-package-v2

The repository checkout includes the code, source spreadsheets, rate input, and
result tables. Obtain the release archive to include the large raw options CSV
and generated datasets. Follow the release's `DOWNLOAD_REPLICATION.md` to verify
and extract both archive parts into an empty directory.

The reference release passed 43 unit tests, 20 implementation checks, and 4,042
numerical comparisons. All 64 comparable result CSVs matched an independent
model refit within 1e-10. `outputs/diagnostics/reproducibility.json` records this
verification. The recorded calculation times were approximately 28 minutes for
the complete fresh run and 16 minutes for the independent refit with verified
quote-input caches. Runtime depends on the machine.
