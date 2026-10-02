# QFE: data, methodology, and replication

The complete pipeline runs from raw option quotes to IV surfaces, forecasting,
real-contract portfolios, Greek attribution, risk controls, and statistical evaluation.

- [Methodology, end to end](replication/METHODOLOGY.md)
- [Installation and execution](replication/README_REPLICATION.md)
- [Inputs and checksums](replication/inputs/INPUTS_MANIFEST.json)
- [Results workbook: 23 worksheets](replication/outputs/excel/QFE_Full_Results.xlsx)
- [Complete package download](https://github.com/Balint-Mariam/paperqfev2/releases/tag/replication-package-v2)

## Download

The Git repository includes the source code, configuration, tests, supporting
Excel inputs, rates CSV, result tables, and figures. The complete release also
includes the 9.15 GB raw options CSV and all generated datasets.

Download the two archive parts and follow [the verification and extraction
instructions](DOWNLOAD_REPLICATION.md). Everything extracts into `replication/`.
At least 20 GB of free space is recommended for downloading, joining, and extracting.
No original author workspace or historical execution cache is required.

## Run

From `replication/`, using Python **3.12.6**:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python 01_build_clean_iv_data.py --workers 8
.venv\Scripts\python 02_run_full_paper_analysis.py --workers 4
```

These are the two analysis entry points. Pinned dependencies, fixed empirical
settings, and package-relative input paths are supplied. The functions are in
`replication/src/`; intermediate data and final tables are in `replication/outputs/`.

## Verification

```powershell
.venv\Scripts\python -B -m unittest discover -s tests -v
```

The 43 tests pass. Independent model refits reproduced all 64 comparable result
tables; all 4,042 numerical comparisons and 20 implementation checks passed. Input and delivery manifests provide SHA256 identities;
the scripts calculate independent timing/accounting checks and numerical comparisons.
The published results are computed by the supplied scripts. An independent
rerun verifies them against the package-local expected tables; actual run statuses
and check values are available in the workbook.
