# QFE replication package

This public repository contains the two-script replication pipeline, original
supporting spreadsheets, rates input, result workbook, tests, and audit evidence.

**Certification status: Stage 1 reproduced; Stage 2 FAILED numerical reproduction.**
Small upstream floating-point differences alter fresh ARIMA fits. These outputs
are available for review and replication, but are not a certified reproduction.
See [the review evidence](replication/REPRODUCTION_REVIEW_REQUIRED.md).

## Download the complete package

The raw options CSV is approximately 9.15 GB. Large raw data, generated datasets,
and execution caches are distributed in the
[replication release](https://github.com/Balint-Mariam/paperqfev2/releases/tag/replication-package-v1).
The release contains **every file in the local replication package**, including
all six original input files, generated outputs, caches, and checksum manifests.
The Git checkout alone does not include the large raw options archive.

Download `DOWNLOAD_REPLICATION.md` and follow its extraction/checksum instructions.
The release archive extracts a `replication/` directory. At least 20 GB of free
space is recommended for the archive plus the extracted package.

## Run the pipeline

Use Python **3.12.6** and the pinned requirements. From the extracted `replication/`:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python 01_build_clean_iv_data.py --workers 8
.venv\Scripts\python 02_run_full_paper_analysis.py --workers 4
```

The input paths point to `replication/inputs/`; the original author's workspace
is not required. Historical workspace comparisons are optional. The second
script returns a nonzero status when numerical reproduction checks fail.

See [the full instructions](replication/README_REPLICATION.md),
[the input inventory](replication/inputs/INPUTS_MANIFEST.json), and
[the 23-sheet results workbook](replication/outputs/excel/QFE_Full_Results.xlsx).

## Tests

From `replication/`, in the same environment:

```text
python -B -m unittest discover -s tests -v
```

The recorded run passed all 43 unit tests. This does not resolve the numerical
reproduction failure. Original CSV and Excel inputs were copied unchanged and
verified by SHA256. 
