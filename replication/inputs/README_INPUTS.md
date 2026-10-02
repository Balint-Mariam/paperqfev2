# Bundled inputs

The two CSV files are the direct inputs to the replication scripts:
- `options_eod_all.csv`: raw option observations (approximately 9.15 GB).
- `div yield and rfr.csv`: dividend yields and risk-free rates.

The four XLSX files are original supporting source workbooks, included unchanged
for inspection. The pipeline uses the CSV inputs rather than reading these
workbooks. No new conversion or recalculation was applied to the inputs.

`INPUTS_MANIFEST.json` lists file sizes and SHA256 checksums. Both CSV checksums
match the inputs used in the recorded Stage 1 run. Copying inputs into the package
does not resolve the documented Stage 2 numerical reproduction failure.
