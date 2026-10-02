"""Frozen functions extracted from paper_qfe/02_repair_timing_and_signals.py."""
from __future__ import annotations
import sys, os, json, time, inspect, hashlib, warnings, logging, itertools
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, spearmanr
import exchange_calendars as xc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .io_utils import digest, token, write_json, save
import xgboost
from xgboost import XGBRegressor
LOG=logging.getLogger('replication')
GRID=[dict(n_estimators=200,max_depth=3,learning_rate=.05,subsample=.9,colsample_bytree=.9),dict(n_estimators=300,max_depth=2,learning_rate=.03,subsample=.9,colsample_bytree=.9)]
XGB_BASE=dict(objective='reg:squarederror',tree_method='hist',random_state=42,n_jobs=2,missing=np.nan)


def features(series, calendar, horizon):
    """Origin-indexed legacy lag architecture. No target series is referenced.

Legacy target-row lag 1 is origin IV; 2,3,5,10 are origin lags 1,2,4,9.
Missing features remain native XGBoost missing values; there is no imputation.
The horizon's future weekday is deterministic calendar information.
"""
    frame = pd.DataFrame(index=series.index)
    for lag in (1, 2, 3, 5, 10):
        frame[f"lag_{lag}"] = series.shift(lag - 1)
    frame["roll_mean_5"] = series.rolling(5, min_periods=2).mean()
    frame["roll_std_5"] = series.rolling(5, min_periods=2).std()
    future_dates = pd.Series(calendar, index=calendar).shift(-horizon).reindex(series.index)
    frame["dow"] = future_dates.dt.weekday.astype(float)
    return frame

def fit_forecasts(wide, calendar, origins, train_end, fit_end, cache):
    """Direct h=1 and h=2 levels, target-date split boundaries and frozen fits.

Only HISTORICAL finite labels filter training/validation. Prediction row creation
uses all origins regardless of future labels, which are never passed to predict.
"""
    signature = token(dict(wide=token(wide.to_json()), calendar=list(map(str, calendar)), origins=list(map(str, origins)),
                           train_end=train_end, fit_end=fit_end, grid=GRID, xgb_version=xgboost.__version__,
                           code=inspect.getsource(features) + inspect.getsource(fit_forecasts)))
    cache.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], []
    tail_dates = pd.Series(calendar, index=calendar)
    for j, node in enumerate(wide.columns, 1):
        series = wide[node]
        for h in (1, 2):
            prefix = cache / f"{node}_h{h}"
            meta_path = prefix.with_suffix(".json")
            forecast_path = prefix.with_suffix(".parquet")
            model_path = prefix.with_suffix(".ubj")
            if meta_path.exists() and forecast_path.exists():
                info = json.loads(meta_path.read_text(encoding="utf-8"))
                if info["signature"] != signature:
                    raise ValueError(f"Forecast checkpoint provenance differs: {prefix}; archive it before an intentional rebuild")
                rows.append(pd.read_parquet(forecast_path))
                summaries.append(info)
                continue
            X = features(series, calendar, h)
            labels = series.shift(-h)
            target_dates = tail_dates.shift(-h).reindex(series.index)
            train = target_dates.le(train_end) & labels.notna()
            # Purge h=2 validation origins preceding the model-training cutoff.
            # Target-date splitting alone would allow a validation origin t to
            # use a model fitted to a label from t+1 at the boundary.
            validation = target_dates.gt(train_end) & target_dates.le(fit_end) & labels.notna() & (series.index >= train_end)
            fit = target_dates.le(fit_end) & labels.notna()
            info = dict(signature=signature, node=node, horizon=h, n_train=int(train.sum()), n_validation=int(validation.sum()),
                        fit_label_max_date=str(target_dates[fit].max()), train_label_max_date=str(target_dates[train].max()),
                        validation_origin_min_date=str(series.index[validation].min()),
                        status="INSUFFICIENT_HISTORICAL_LABELS", params=None)
            predictions = np.full(len(origins), np.nan)
            if train.sum() >= 80 and validation.sum() >= 20:
                best, best_rmse, best_mae = None, np.inf, np.inf
                for cfg in GRID:
                    model = XGBRegressor(**XGB_BASE, **cfg)
                    model.fit(X.loc[train], labels.loc[train])
                    err = model.predict(X.loc[validation]) - labels.loc[validation].to_numpy()
                    rmse, mae = float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))
                    if rmse < best_rmse or (np.isclose(rmse, best_rmse, atol=1e-12) and mae < best_mae):
                        best, best_rmse, best_mae = cfg, rmse, mae
                final_model = XGBRegressor(**XGB_BASE, **best)
                final_model.fit(X.loc[fit], labels.loc[fit])
                predictions = final_model.predict(X.reindex(origins)).astype(float)
                # Booster serialization avoids incompatible estimator tags in
                # the installed sklearn/XGBoost wrapper, preserving the model.
                final_model.get_booster().save_model(model_path)
                info.update(status="OK", params=best)
            result = pd.DataFrame(dict(decision_date=origins, node=node, horizon=h, forecast=predictions))
            save(result, forecast_path)
            write_json(info, meta_path)
            rows.append(result); summaries.append(info)
        if j % 20 == 0 or j == len(wide.columns):
            LOG.info("Economic models: %s/%s nodes (paired direct horizons)", j, len(wide.columns))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)

def normalize(chunk):
    """Frozen source IV retained, ISO dates and six-decimal contract identity."""
    chunk = chunk.copy()
    chunk["quote_date"] = pd.to_datetime(chunk.quote_date, format="%Y-%m-%d", errors="raise")
    chunk["expiration"] = pd.to_datetime(chunk.expiration, format="%Y-%m-%d", errors="raise")
    for col in ("underlying_symbol", "root", "option_type"):
        chunk[col] = chunk[col].astype(str).str.strip().str.upper()
    chunk["option_type"] = chunk.option_type.str[0]
    numeric = [c for c in chunk if c not in ("quote_date", "expiration", "underlying_symbol", "root", "option_type")]
    for col in numeric:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    chunk = chunk.rename(columns={"bid_1545": "bid", "ask_1545": "ask", "r_annual": "r", "q_annual": "q", "implied_vol": "implied_vol"})
    chunk["strike"] = chunk.strike.round(6)
    chunk["contract_key"] = (chunk.underlying_symbol + "|" + chunk.root + "|" + chunk.expiration.dt.strftime("%Y-%m-%d") + "|" +
                              chunk.strike.map(lambda x: f"{x:.6f}") + "|" + chunk.option_type)
    chunk["log_moneyness"] = np.log(chunk.moneyness.where(chunk.moneyness > 0))
    chunk["snapshot_label"] = "1545"
    return chunk

def load_day(cache, date):
    folder = cache / ("date=" + pd.Timestamp(date).strftime("%Y-%m-%d"))
    parts = sorted(folder.glob("part_*.parquet"))
    if not parts:
        return pd.DataFrame()
    raw = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    group = ["quote_date", "contract_key", "underlying_symbol", "root", "expiration", "strike", "option_type"]
    means = ["bid", "ask", "bid_eod", "ask_eod", "mid", "spot", "moneyness", "log_moneyness", "T", "implied_vol", "r", "q"]
    aggregates = {c: "mean" for c in means}
    aggregates.update(trade_volume="sum", open_interest="sum", snapshot_label="first")
    day = raw.groupby(group, as_index=False, dropna=False).agg(aggregates)
    day = day.merge(raw.groupby(["quote_date", "contract_key"]).size().rename("source_observation_count").reset_index(), on=["quote_date", "contract_key"], validate="one_to_one")
    return day.sort_values("contract_key", kind="stable").reset_index(drop=True)

def valid_market(day):
    if day.empty:
        return day
    mask = (np.isfinite(day.bid) & np.isfinite(day.ask) & day.bid.gt(0) & day.ask.ge(day.bid) &
            np.isfinite(day.spot) & day.spot.gt(0) & np.isfinite(day["T"]) & day["T"].gt(0) &
            np.isfinite(day.log_moneyness) & day.option_type.isin(["C", "P"]) & (day.expiration > day.quote_date))
    return day.loc[mask].copy()

def bs_greeks(frame):
    """Unchanged definitions from compute_portfolio_greeks.py; no IV inversion."""
    out = pd.DataFrame(index=frame.index, columns=["Delta", "Gamma", "Vega", "Theta"], dtype=float)
    ok = np.isfinite(frame[["spot", "strike", "T", "implied_vol", "r", "q"]]).all(axis=1) & frame[["spot", "strike", "T", "implied_vol"]].gt(0).all(axis=1)
    if not ok.any(): return out
    f = frame.loc[ok]; S,K,T,v,r,q = [f[c].to_numpy(float) for c in ("spot","strike","T","implied_vol","r","q")]
    sqrt_t = np.sqrt(T); dq = np.exp(-q*T); dr = np.exp(-r*T)
    d1 = (np.log(S/K)+(r-q+0.5*v*v)*T)/(v*sqrt_t); d2 = d1-v*sqrt_t; call=f.option_type.eq("C").to_numpy()
    out.loc[ok,"Delta"] = np.where(call,dq*norm.cdf(d1),dq*(norm.cdf(d1)-1))
    out.loc[ok,"Gamma"] = dq*norm.pdf(d1)/(S*v*sqrt_t)
    out.loc[ok,"Vega"] = S*dq*norm.pdf(d1)*sqrt_t
    out.loc[ok,"Theta"] = np.where(call,-S*dq*norm.pdf(d1)*v/(2*sqrt_t)-r*K*dr*norm.cdf(d2)+q*S*dq*norm.cdf(d1),
                                  -S*dq*norm.pdf(d1)*v/(2*sqrt_t)+r*K*dr*norm.cdf(-d2)-q*S*dq*norm.cdf(-d1))
    return out

def map_entry(selected, day, snapshot_open):
    """Only selected decision inputs and this entry-date market are accepted."""
    if len(day) and (selected.entry_date.nunique()!=1 or not day.quote_date.eq(selected.entry_date.iloc[0]).all()):
        raise ValueError("Entry mapping received non-entry-date market rows")
    rows = []
    candidates = valid_market(day) if snapshot_open else pd.DataFrame()
    for row in selected.itertuples(index=False):
        base = dict(decision_date=row.decision_date, entry_date=row.entry_date, node=row.node,
                    entry_mapping_status="NO_ENTRY_MARKET" if snapshot_open else "MARKET_CLOSED_AT_1545",
                    contract_key=None, mapping_distance=np.nan, abs_log_moneyness_diff=np.nan, abs_T_diff=np.nan)
        if len(candidates):
            distances = np.sqrt(((candidates.log_moneyness-row.log_moneyness)/0.02)**2 + ((candidates["T"]-row.T)/0.05)**2)
            minimum = distances.min()
            # Explicit stable key tie-break avoids un-timestamped daily volume/OI.
            c = candidates.loc[(distances-minimum).abs()<=1e-12].sort_values("contract_key").iloc[0]
            base.update(entry_mapping_status="MAPPED", contract_key=c.contract_key, mapping_distance=float(minimum),
                        abs_log_moneyness_diff=abs(c.log_moneyness-row.log_moneyness), abs_T_diff=abs(c["T"]-row.T))
            for col in ("underlying_symbol","root","expiration","strike","option_type"):
                base[col] = c[col]
            for col in ("bid","ask","mid","spot","implied_vol","T","r","q","trade_volume","open_interest","log_moneyness","source_observation_count"):
                base["entry_"+col] = c[col]
            base["entry_execution_price"] = c.ask if row.signal==1 else c.bid
        rows.append(base)
    return pd.DataFrame(rows)

def lock_weights(frame):
    """Half long, half short, counted exclusively from entry-mapped positions."""
    out = frame.copy(); out["weight_at_entry"] = 0.0; out["entry_weight_status"] = "NO_ENTRY"
    mapped = out.entry_mapping_status.eq("MAPPED")
    for _, day in out.loc[mapped].groupby("entry_date", sort=True):
        long = day.index[day.signal.eq(1)]; short = day.index[day.signal.eq(-1)]
        if len(long) and len(short):
            out.loc[long,"weight_at_entry"] = 0.5/len(long)
            out.loc[short,"weight_at_entry"] = -0.5/len(short)
            out.loc[day.index,"entry_weight_status"] = "LOCKED_SIDE_NEUTRAL"
        else:
            out.loc[day.index,"entry_weight_status"] = "NO_BOTH_SIDES_AT_ENTRY"
    return out
