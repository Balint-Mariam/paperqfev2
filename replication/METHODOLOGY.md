# Methodology and file map

This document describes the implemented sequence. The JSON configuration fixes
all filters, model candidates, dates, portfolio definitions, and inference settings.
The two scripts execute the functions in `src/` directly.

## 1. Input data and units

`inputs/options_eod_all.csv` supplies labelled 15:45 option bid/ask prices,
underlying bid/ask prices, quote and expiration dates, strikes, contract types,
volume, and open interest. The archive contains 69,596,970 observations.
`inputs/div yield and rfr.csv` supplies exact calendar dates, daily dividend
yields, and daily rates for 1M, 2M, 3M, 4M, 6M, and 12M maturities.
The four original workbooks are supporting source material. The merged rate CSV
is the direct computational input; the spreadsheets are not reprocessed on each run.

In the source preparation, annual Treasury percentages were converted to daily
rates as `(1 + yield / 100) ** (1 / 252) - 1`. Missing 2M and 4M Treasury values
were interpolated across available neighbouring maturities before merging on
calendar dates with the dividend series. In the cleaning pipeline, missing rate
values are interpolated over calendar time and remaining boundaries are filled
forward/backward. Dates absent from the rate table are not fabricated.
The rate and dividend CSV is supplied unchanged, with its checksum, so subsequent
computations begin from the same serialized numeric input.

All prices and Greeks use a normalized contract multiplier of 1. IV is a decimal
annual volatility. Time to maturity and the Theta clock use actual calendar days
and a 365-day year. Vega is the price derivative per one decimal unit of IV.

## 2. Cleaning and rates

Implemented by `data_pipeline.py` and `cleaning.py`:

1. Calculate option and underlying mid prices from their 15:45 books.
2. Require positive bids, asks, strikes, mid prices, underlying prices, volume,
   and open interest; require option ask strictly greater than bid.
3. Parse quote/expiration dates and retain maturities from 14 calendar days to
   one year, inclusive.
4. Join the rate table on the exact quote date. Linearly interpolate the six
   maturity rates to the contract's remaining maturity, using the boundary
   maturities outside the supplied tenor range.
5. Calculate annual rate parameters as `(1 + daily_rate) ** 252 - 1` for both
   interest and dividend inputs.
6. Retain strike/spot between 0.8 and 1.2 and relative option spread
   `(ask - bid) / mid <= 0.20`. Require call/put contract labels.
7. Apply the European lower price bound, with a tolerance of 0.0001.

The sequential funnel is `outputs/diagnostics/data_cleaning_funnel.csv`.
The recorded run retained 6,713,294 rows. The annualized parameters above enter
the exponential discount/carry factors directly, without another rate conversion;
this is the fixed convention implemented by the pricing and Greek functions.

## 3. Implied volatility and the grid

`implied_volatility.py` uses European Black?Scholes pricing at the observed mid,
with discount factors `exp(-r*T)` and `exp(-q*T)`. The Brent root search is bounded
by 0.0001 and 5, with absolute tolerance 0.000001 and at most 100 iterations.
Failed inversions remain missing; they are not replaced by nearby IV values.
The recorded run obtained 6,713,293 valid IVs and one solver failure.

`iv_surface.py` sets `x = log(strike / spot)`. For each quote date, duplicate
`(x,T)` locations are aggregated by their median IV. A day needs at least ten
usable observations. Linear interpolation maps observations onto 25 equally
spaced x coordinates from -0.22 to 0.18 and 20 T coordinates from 0.08 to 1.
Outside the supported interpolation region values remain missing. There is no
nearest-neighbour filling or smoothing.

The grid has 500 nodes on 2,808 dates, from 2013-10-01 to 2024-12-31, with 992,765
observed grid cells out of 1,404,000 potential cells. Intermediate data use Parquet;
wide/long grids and the node map are also supplied as CSV. The CSV
numeric parsing boundaries are preserved when constructing model inputs.

## 4. Chronology and forecasting

`forecasting.py`, `economic_signals.py`, and `analysis_pipeline.py` use the XNYS
session calendar. Reindexing retains missing sessions without filling IV.
For a decision session t, forecasts target t+1 and t+2 using information through t.
Economic entry is t+1 and exit is t+2. Calendar session labels and the labelled
15:45 snapshots are kept distinct from the actual elapsed calendar time used by Theta.

Candidate selection uses training labels through **2021-08-12** and validation
labels through **2023-04-21**. Final fitting uses labels through 2023-04-21.
Horizon-specific masks use target dates, with overlapping validation origins
excluded where required. Test outcomes do not enter fitting or preprocessing.

Four forecasters are compared:

| Model | Specification |
|---|---|
| Persistence | IV observed at the decision session |
| ARIMA | Orders (1,0,0), (2,0,0), (1,1,0), (1,0,1), (2,0,1); no trend; stationarity/invertibility constraints disabled; 200 fitting iterations |
| Ridge | Alphas 0.01, 0.1, 1, 10, 100; training-only median imputation with missingness indicators and standardization |
| XGBoost | 200 trees/depth 3/learning rate 0.05 or 300 trees/depth 2/learning rate 0.03; subsample and column sample 0.9; histogram trees; square-error objective; seed 42; two threads |

Features are IV lags 1, 2, 3, 5, and 10, five-observation rolling mean/standard
deviation with at least two observations, and the target weekday. Lag 1 denotes
the IV available at the forecast origin. ARIMA needs at least 60 finite training
observations; feature models need 80. Validation needs at least 20 labels.
Candidates are selected by validation RMSE, with the implemented MAE tie-break.
Nonconverged ARIMA candidates/final fits are rejected and produce missing forecasts.
Parameters remain fixed in the test period; ARIMA filtering uses observations
through the decision date. The two-step prediction advances the filtered state
without observing the intermediate t+1 outcome.

Forecast scores use the common finite sample of all four predictions and actual
IV. Reported metrics include RMSE, MAE, and QLIKE:
`actual_IV**2 / forecast_IV**2 - log(actual_IV**2 / forecast_IV**2) - 1`.
QLIKE requires positive actual/predicted IV. Daily rank correlations require at
least 20 scored nodes and nonconstant values. Forecast inference uses daily loss
differences, HAC lag 5 with a Student-t reference, 2,000 circular-block bootstrap
replications, blocks 5/10, and model confidence sets at alpha 0.05.

## 5. Economic signals and contracts

The economic universe requires current observed IV and finite h1/h2 predictions
from ARIMA, Ridge, and XGBoost. Future realized IV or exit quote availability is
not an eligibility condition. For each model, the signal is predicted h2 IV minus
predicted h1 IV. It is standardized cross-sectionally on each decision day.
Nodes with z >= 0.5 are eligible long; z <= -0.5 are eligible short. At most ten
per side are selected, ordered by signal strength and then node identity.

`contract_mapping.py` maps each selected node at entry to an available processed
real contract, minimizing
`sqrt(((contract_x - node_x)/0.02)**2 + ((contract_T - node_T)/0.05)**2)`.
Ties within 1e-12 use the stable contract key, not volume or open interest rankings.
Contract identity includes underlying, root, expiration, strike at six decimal
places, and call/put type. Entry weights allocate +0.5 across mapped long nodes
and -0.5 across mapped short nodes. Multiple nodes mapping to the same contract
are netted by signed weight. Netting and mapping losses are explicitly tabulated.

`execution_quotes.py` and `quote_data.py` recover the exact t+2 contract book:
first the processed 15:45 book, then the raw archive for the exact date and key.
A valid recovery requires finite quotes and `0 < bid <= ask`; volume, open
interest, IV availability, and entry spread filters are not imposed on exits.
Calendar exclusions apply to whole model-days, rather than selectively removing
individual contracts and renormalizing weights after seeing exit availability.
The implementation tables retain the mapping/recovery chain and coverage.

## 6. PnL and implementation

`portfolio.py` computes two price conventions for each signed contract weight w:

- MID: `w * (exit_mid - entry_mid)`.
- BID_ASK: a long enters at ask and exits at bid; a short enters at bid and exits
  at ask. PnL is `w * (exit_execution_price - entry_execution_price)`.

MID minus BID_ASK PnL equals the sum of weighted entry/exit half-spreads.
Daily returns divide PnL by the sum of absolute weights times entry premiums.
These are premium-normalized option returns. They are not returns on measured
margin capital; cumulative plots sum the daily normalized returns arithmetically.

## 7. Greeks and risk controls

`greeks.py` and `risk_controls.py` calculate European entry Delta, Gamma, Vega,
and annual Theta from entry IV, rate, dividend, spot, strike, and maturity.
The full available sample decomposes MID PnL into Delta, Gamma, Theta, and a residual:
`w*Delta*dS + 0.5*w*Gamma*dS**2 + w*Theta*dt + residual`.
This residual includes effects not represented by those three first-order terms;
it is not interpreted as pure Vega PnL.

A separate matched diagnostic subset adds `w*Vega*dIV`. It requires exact
entry/exit IV observations, compatible maturity timing and quote identity,
single-observation consistency, and repricing within 0.001. No additional IV
inversion is used to expand this subset.

All four portfolio states are evaluated on the same definitions:

| State | Entry rule |
|---|---|
| ORIGINAL | Netted option weights |
| DELTA_NEUTRAL | Add static underlying quantity `h = -sum(w*Delta)` |
| VEGA_BALANCED | With positive leg Vega VL and short magnitude VS, set B=min(VL,VS), aL=B/VL, aS=B/VS and scale the two option sides uniformly |
| VEGA_BALANCED_DELTA_NEUTRAL | Balance Vega, then hedge the remaining entry Delta |

Vega scaling factors are between zero and one; they do not increase option
leverage. All controls use entry observations only. The underlying hedge is a
static frictionless price proxy in the base results, with transaction-cost
sensitivity reported separately. The labelled 15:45 books do not establish
actual execution fills or every historical timezone alignment.

## 8. Statistical evaluation and robustness

`robustness.py` evaluates daily economic returns with Bartlett HAC lag 5,
finite-sample correction, and a normal reference; lags 1 and 10 are sensitivity
checks. Economic bootstraps use 4,000 circular-block draws, block lengths 5/10,
and seed 42.

The primary multiple-testing families contain the 12 model/state means separately
for MID and BID_ASK. Holm and Benjamini?Hochberg adjusted p-values are reported.
Within-model comparisons, paired control effects, and Greek-component tests keep
their specified separate families. Reported nominal and adjusted significance
must be read with the stated family, rather than selecting individual tests.

Other analyses include pooled-premium normalization, a common original-MID
premium denominator, hedge-cost break-even and stress at 0.5/1/2/5 basis points
per side, 2023/2024 subperiods, and IV regimes. The regime threshold is the
pre-test median daily cross-sectional observed IV, fixed at
0.17683139218966898 through 2023-04-21; LOW includes equality. No test-period
outcome is used to estimate this threshold.

## 9. Outputs and verification

The workbook's 23 ordered sheets cover run information, cleaning, the grid,
forecast scores, DM tests, MCS, signals, execution, original PnL, Greek exposures,
DGT/DGVT attribution, risk controls, multiple testing, paired effects,
normalization, hedge costs, subperiods, regimes, full-chain accounting, model
ranks, checks, and file identity. CSV counterparts are in `outputs/tables/`.
Generated contract-level data are in `outputs/data/`; figures use those results.

The tests include independent accounting/pricing examples, future-data mutations,
calendar handling, entry-only control rules, and cache corruption. Automated
checks remain in both scripts and stop on required failures. Expected tables
under `manifests/expected_results/` are generated from this package execution and
used only to check subsequent runs; they are never substituted into calculations.

### Reproducibility

The release reports the results computed by the supplied code, inputs, empirical
specification, and pinned environment. All forecasting and portfolio outputs are
calculated directly. Expected-result tables and their checksums are recorded after
a complete calculation. A second independent fit and analysis checks the resulting
numbers against those tables with the declared absolute tolerances. Subsequent
users can repeat the same two-script sequence and inspect every intermediate output.
