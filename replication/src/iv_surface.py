"""Frozen functions extracted from build_iv_grid_dataset.py."""
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
import pathlib
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


def prepare_points(
    csv_path: pathlib.Path,
    chunksize: int,
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    cols = set(header.columns)
    required = {"quote_date", "strike", "implied_vol"}
    missing = required - cols
    if missing:
        raise ValueError(f"Lipsesc coloane obligatorii: {', '.join(sorted(missing))}")

    usecols = ["quote_date", "strike", "implied_vol"]
    for c in ["T", "expiration", "spot", "underlying_bid_1545", "underlying_ask_1545", "moneyness", "k_over_s"]:
        if c in cols:
            usecols.append(c)

    dt_from = pd.to_datetime(date_from, errors="coerce") if date_from else pd.NaT
    dt_to = pd.to_datetime(date_to, errors="coerce") if date_to else pd.NaT
    if date_from and pd.isna(dt_from):
        raise ValueError("`--date-from` invalid. Use YYYY-MM-DD.")
    if date_to and pd.isna(dt_to):
        raise ValueError("`--date-to` invalid. Use YYYY-MM-DD.")

    parts = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        chunk["quote_date"] = parse_dates(chunk["quote_date"])
        chunk["implied_vol"] = pd.to_numeric(chunk["implied_vol"], errors="coerce")
        chunk["strike"] = pd.to_numeric(chunk["strike"], errors="coerce")

        if "T" in chunk.columns:
            chunk["T"] = pd.to_numeric(chunk["T"], errors="coerce")
        elif "expiration" in chunk.columns:
            chunk["expiration"] = parse_dates(chunk["expiration"])
            chunk["T"] = (chunk["expiration"] - chunk["quote_date"]).dt.days / 365.0
        else:
            raise ValueError("Lipseste T si nu exista expiration pentru calcul maturitate.")

        if "k_over_s" in chunk.columns:
            chunk["k_over_s"] = pd.to_numeric(chunk["k_over_s"], errors="coerce")
        elif "moneyness" in chunk.columns:
            chunk["k_over_s"] = pd.to_numeric(chunk["moneyness"], errors="coerce")
        elif "spot" in chunk.columns:
            chunk["spot"] = pd.to_numeric(chunk["spot"], errors="coerce")
            chunk["k_over_s"] = chunk["strike"] / chunk["spot"]
        elif {"underlying_bid_1545", "underlying_ask_1545"}.issubset(chunk.columns):
            bid = pd.to_numeric(chunk["underlying_bid_1545"], errors="coerce")
            ask = pd.to_numeric(chunk["underlying_ask_1545"], errors="coerce")
            spot = (bid + ask) / 2.0
            chunk["k_over_s"] = chunk["strike"] / spot
        else:
            raise ValueError("Nu pot calcula K/S (lipsesc moneyness/k_over_s/spot).")

        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.dropna(subset=["quote_date", "implied_vol", "k_over_s", "T"])
        chunk = chunk[(chunk["implied_vol"] > 0) & (chunk["k_over_s"] > 0) & (chunk["T"] >= t_min) & (chunk["T"] <= t_max)]
        if chunk.empty:
            continue

        if not pd.isna(dt_from):
            chunk = chunk[chunk["quote_date"] >= dt_from]
        if not pd.isna(dt_to):
            chunk = chunk[chunk["quote_date"] <= dt_to]
        if chunk.empty:
            continue

        chunk["log_moneyness"] = np.log(chunk["k_over_s"])
        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.dropna(subset=["log_moneyness"])
        chunk = chunk[(chunk["log_moneyness"] >= x_min) & (chunk["log_moneyness"] <= x_max)]
        if chunk.empty:
            continue

        parts.append(
            chunk[["quote_date", "log_moneyness", "T", "implied_vol"]].copy()
        )

    if not parts:
        return pd.DataFrame(columns=["quote_date", "log_moneyness", "T", "implied_vol"])

    points = pd.concat(parts, ignore_index=True)
    points["quote_date"] = points["quote_date"].dt.strftime("%Y-%m-%d")
    return points

def build_grid_and_features(
    x_min: float,
    x_max: float,
    x_points: int,
    t_min: float,
    t_max: float,
    t_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    x_grid = np.linspace(x_min, x_max, x_points)
    t_grid = np.linspace(t_min, t_max, t_points)
    grid_x, grid_t = np.meshgrid(x_grid, t_grid)

    features = []
    grid_map_rows = []
    for t_idx, t_val in enumerate(t_grid):
        for x_idx, x_val in enumerate(x_grid):
            feat = f"iv_x{x_idx:02d}_t{t_idx:02d}"
            features.append(feat)
            grid_map_rows.append(
                {
                    "feature": feat,
                    "x_idx": x_idx,
                    "t_idx": t_idx,
                    "log_moneyness": float(x_val),
                    "T": float(t_val),
                }
            )
    grid_map = pd.DataFrame(grid_map_rows)
    return x_grid, t_grid, grid_x, grid_t, features, grid_map

def interpolate_day_surface(
    day_df: pd.DataFrame,
    grid_x: np.ndarray,
    grid_t: np.ndarray,
    method: str,
    fill_nearest: bool,
    smooth_sigma: float,
) -> tuple[np.ndarray, str]:
    day_unique = (
        day_df.groupby(["log_moneyness", "T"], as_index=False)
        .agg(implied_vol=("implied_vol", "median"))
    )
    points = day_unique[["log_moneyness", "T"]].to_numpy(dtype=float)
    values = day_unique["implied_vol"].to_numpy(dtype=float)

    z_grid = griddata(points, values, (grid_x, grid_t), method=method)
    used_method = method

    if np.isnan(z_grid).all() and method == "cubic":
        z_grid = griddata(points, values, (grid_x, grid_t), method="linear")
        used_method = "linear"

    if fill_nearest and np.isnan(z_grid).any():
        z_nn = griddata(points, values, (grid_x, grid_t), method="nearest")
        z_grid = np.where(np.isnan(z_grid), z_nn, z_grid)
        used_method = f"{used_method}+nearest_fill"

    z_grid = smooth_surface(z_grid, sigma=smooth_sigma)
    return z_grid, used_method

def smooth_surface(z_grid: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return z_grid
    mask = np.isfinite(z_grid).astype(float)
    values = np.where(np.isfinite(z_grid), z_grid, 0.0)
    smooth_vals = gaussian_filter(values, sigma=sigma)
    smooth_mask = gaussian_filter(mask, sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = smooth_vals / smooth_mask
    out[smooth_mask < 1e-6] = np.nan
    return out

def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")
