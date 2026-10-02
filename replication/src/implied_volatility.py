"""European option pricing and bounded implied-volatility inversion."""
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
from scipy.optimize import brentq
from scipy.special import ndtr
from .cleaning import RATE_COLS, RATE_TERMS_YEARS, TRADING_DAYS, interpolate_rates


def bs_price(S, K, T, r, q, sigma, call=True):
    """Black-Scholes price for European options."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if call else (K - S))
    vol_sqrt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    if call:
        return S * np.exp(-q * T) * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
    return K * np.exp(-r * T) * ndtr(-d2) - S * np.exp(-q * T) * ndtr(-d1)

def implied_vol(price, S, K, T, r=0.0, q=0.0, call=True):
    """Return implied volatility via Brent root finder; NaN when not solvable."""
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan

    def f(sig):
        return bs_price(S, K, T, r, q, sig, call) - price

    try:
        # Wide bounds that work for most equity options; adjust if needed.
        return brentq(f, 1e-4, 5.0, maxiter=100, xtol=1e-6)
    except ValueError:
        return np.nan

def compute_chunk_iv(chunk: pd.DataFrame, rates_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Compute IV for a single chunk."""
    if "spot" in chunk.columns:
        chunk["spot"] = pd.to_numeric(chunk["spot"], errors="coerce")
    else:
        chunk["spot"] = (chunk["underlying_bid_1545"] + chunk["underlying_ask_1545"]) / 2

    if "mid" in chunk.columns:
        chunk["mid"] = pd.to_numeric(chunk["mid"], errors="coerce")
    else:
        chunk["mid"] = (chunk["bid_1545"] + chunk["ask_1545"]) / 2

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
    chunk = chunk.dropna(subset=["quote_date", "expiration"])
    chunk["T"] = (chunk["expiration"] - chunk["quote_date"]).dt.days / 365.0

    has_rq = {"r_annual", "q_annual"}.issubset(chunk.columns)
    if has_rq:
        chunk["r_annual"] = pd.to_numeric(chunk["r_annual"], errors="coerce")
        chunk["q_annual"] = pd.to_numeric(chunk["q_annual"], errors="coerce")
    else:
        if rates_df is None:
            raise ValueError("Lipsesc r_annual/q_annual in input si nu exista fisierul de rates.")
        chunk = chunk.merge(
            rates_df,
            left_on="quote_date",
            right_on="Calendar Date",
            how="inner",
        )
        if chunk.empty:
            return chunk
        rate_matrix = chunk[RATE_COLS].to_numpy(dtype=float)
        T = chunk["T"].to_numpy(dtype=float)
        r_daily = interpolate_rates(rate_matrix, RATE_TERMS_YEARS, T)
        chunk["r_annual"] = (1.0 + r_daily) ** TRADING_DAYS - 1.0
        q_daily = chunk["Dividend Yield (Value-Weighted)"].to_numpy(dtype=float)
        chunk["q_annual"] = (1.0 + q_daily) ** TRADING_DAYS - 1.0

    chunk["strike"] = pd.to_numeric(chunk["strike"], errors="coerce")
    chunk["option_type"] = chunk["option_type"].astype(str).str.upper().str.strip()
    chunk = chunk.dropna(subset=["mid", "spot", "strike", "T", "r_annual", "q_annual"])
    if chunk.empty:
        return chunk

    calls = chunk["option_type"].eq("C").to_numpy()
    chunk["implied_vol"] = [
        implied_vol(p, s, k, t, r, q, call)
        for p, s, k, t, r, q, call in zip(
            chunk["mid"].to_numpy(dtype=float),
            chunk["spot"].to_numpy(dtype=float),
            chunk["strike"].to_numpy(dtype=float),
            chunk["T"].to_numpy(dtype=float),
            chunk["r_annual"].to_numpy(dtype=float),
            chunk["q_annual"].to_numpy(dtype=float),
            calls,
        )
    ]
    return chunk
