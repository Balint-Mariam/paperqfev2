# Inputs

The scripts read `options_eod_all.csv` (raw option records) and
`div yield and rfr.csv` (daily dividend yields and maturity-specific rates).
The four XLSX files are original supporting source workbooks. They are supplied
for inspection; the scripts use the CSV inputs directly.

All six input files are unchanged copies of the original inputs. Their roles,
byte sizes, and SHA256 values are in `INPUTS_MANIFEST.json`. The raw options CSV
is approximately 9.15 GB and is included in the complete release download.
