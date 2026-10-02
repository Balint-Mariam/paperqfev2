"""Option quote eligibility, rate interpolation, and lower price bounds."""
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
RATE_COLS=['1M','2M','3M','4M','6M','12M']
RATE_TERMS_YEARS=np.array([1,2,3,4,6,12],dtype=float)/12.
TRADING_DAYS=252.


def interpolate_rates(rates: np.ndarray, terms: np.ndarray, T: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(terms, T, side="right")
    low_idx = np.clip(idx - 1, 0, len(terms) - 1)
    high_idx = np.clip(idx, 0, len(terms) - 1)
    low_rates = np.take_along_axis(rates, low_idx[:, None], axis=1).reshape(-1)
    high_rates = np.take_along_axis(rates, high_idx[:, None], axis=1).reshape(-1)
    low_terms = terms[low_idx]
    high_terms = terms[high_idx]
    span = high_terms - low_terms
    weight = np.zeros_like(T, dtype=float)
    np.divide(T - low_terms, span, out=weight, where=span != 0)
    return low_rates + (high_rates - low_rates) * weight

def prepare_rates_df(rates_path: pathlib.Path) -> pd.DataFrame:
    rates_df = pd.read_csv(rates_path)
    rates_df.columns = rates_df.columns.str.strip()
    required_rates = {"Calendar Date", "Dividend Yield (Value-Weighted)"} | set(RATE_COLS)
    rate_missing = required_rates - set(rates_df.columns)
    if rate_missing:
        raise ValueError(f"Lipsesc coloanele in rates: {', '.join(sorted(rate_missing))}")

    rates_df["Calendar Date"] = pd.to_datetime(
        rates_df["Calendar Date"],
        dayfirst=True,
        errors="coerce",
        format="mixed",
    )
    rates_df = rates_df.dropna(subset=["Calendar Date"])
    rates_df = rates_df.sort_values("Calendar Date")
    rates_df = rates_df.set_index("Calendar Date")
    rates_df[RATE_COLS] = rates_df[RATE_COLS].interpolate(method="time")
    rates_df = rates_df.reset_index()
    rates_df[RATE_COLS] = rates_df[RATE_COLS].ffill().bfill()
    rates_df["Dividend Yield (Value-Weighted)"] = (
        rates_df["Dividend Yield (Value-Weighted)"].ffill().bfill()
    )
    return rates_df

def lower_bound_price(spot, strike, T, r, q, is_call):
    """European no-arbitrage lower bound with carry and discounting."""
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    call_lb = np.maximum(0.0, spot * disc_q - strike * disc_r)
    put_lb = np.maximum(0.0, strike * disc_r - spot * disc_q)
    return np.where(is_call, call_lb, put_lb)

def apply_basic_filters(
    chunk: pd.DataFrame,
    rates_df: pd.DataFrame,
    t_min_days: int,
    t_max_years: float,
    spread_rel_max: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean option data using standard filters and lower-bound checks."""
    stats = {
        "input": len(chunk),
        "dropped_basic": 0,
        "dropped_bad_dates": 0,
        "dropped_t_window": 0,
        "dropped_no_rates": 0,
        "dropped_moneyness": 0,
        "dropped_spread": 0,
        "dropped_option_type": 0,
        "dropped_lower_bound": 0,
        "kept_basic": 0,
    }
    chunk["spot"] = (chunk["underlying_bid_1545"] + chunk["underlying_ask_1545"]) / 2
    chunk["mid"] = (chunk["bid_1545"] + chunk["ask_1545"]) / 2

    valid = (
        (chunk["bid_1545"] > 0)
        & (chunk["ask_1545"] > 0)
        & (chunk["ask_1545"] > chunk["bid_1545"])
        & (chunk["trade_volume"] > 0)
        & (chunk["open_interest"] > 0)
        & (chunk["spot"] > 0)
        & (chunk["strike"] > 0)
        & (chunk["mid"] > 0)
    )

    stats["dropped_basic"] = int((~valid).sum())
    chunk = chunk.loc[valid].copy()
    if chunk.empty:
        return chunk, stats
    chunk["quote_date"] = pd.to_datetime(
        chunk["quote_date"],
        dayfirst=True,
        errors="coerce",
        format="mixed",
    )
    chunk["expiration"] = pd.to_datetime(
        chunk["expiration"],
        dayfirst=True,
        errors="coerce",
        format="mixed",
    )
    before_dates = len(chunk)
    chunk = chunk.dropna(subset=["quote_date", "expiration"])
    stats["dropped_bad_dates"] = before_dates - len(chunk)
    if chunk.empty:
        return chunk, stats
    chunk["T"] = (chunk["expiration"] - chunk["quote_date"]).dt.days / 365.0
    before_t = len(chunk)
    chunk = chunk[(chunk["T"] >= (t_min_days / 365.0)) & (chunk["T"] <= t_max_years)]
    stats["dropped_t_window"] = before_t - len(chunk)
    if chunk.empty:
        return chunk, stats

    before_rates = len(chunk)
    chunk = chunk.merge(
        rates_df,
        left_on="quote_date",
        right_on="Calendar Date",
        how="inner",
    )
    stats["dropped_no_rates"] = before_rates - len(chunk)
    if chunk.empty:
        return chunk, stats
    rate_matrix = chunk[RATE_COLS].to_numpy(dtype=float)
    T = chunk["T"].to_numpy(dtype=float)
    r_daily = interpolate_rates(rate_matrix, RATE_TERMS_YEARS, T)
    chunk["r_annual"] = (1.0 + r_daily) ** TRADING_DAYS - 1.0
    q_daily = chunk["Dividend Yield (Value-Weighted)"].to_numpy(dtype=float)
    chunk["q_annual"] = (1.0 + q_daily) ** TRADING_DAYS - 1.0

    chunk["moneyness"] = chunk["strike"] / chunk["spot"]
    before_mny = len(chunk)
    chunk = chunk[chunk["moneyness"].between(0.8, 1.2)]
    stats["dropped_moneyness"] = before_mny - len(chunk)
    if chunk.empty:
        return chunk, stats

    spread_rel = (chunk["ask_1545"] - chunk["bid_1545"]) / chunk["mid"]
    before_spread = len(chunk)
    chunk = chunk[spread_rel <= spread_rel_max]
    stats["dropped_spread"] = before_spread - len(chunk)
    if chunk.empty:
        return chunk, stats

    chunk["option_type"] = chunk["option_type"].astype(str).str.upper().str.strip()
    before_opt_type = len(chunk)
    chunk = chunk[chunk["option_type"].isin(["C", "P"])]
    stats["dropped_option_type"] = before_opt_type - len(chunk)
    if chunk.empty:
        return chunk, stats

    is_call = chunk["option_type"] == "C"
    lb = lower_bound_price(
        spot=chunk["spot"].to_numpy(dtype=float),
        strike=chunk["strike"].to_numpy(dtype=float),
        T=chunk["T"].to_numpy(dtype=float),
        r=chunk["r_annual"].to_numpy(dtype=float),
        q=chunk["q_annual"].to_numpy(dtype=float),
        is_call=is_call.to_numpy(),
    )
    tol = 1e-4
    before_lb = len(chunk)
    chunk = chunk[chunk["mid"] + tol >= lb]
    stats["dropped_lower_bound"] = before_lb - len(chunk)
    stats["kept_basic"] = len(chunk)
    if chunk.empty:
        return chunk, stats

    return chunk, stats
